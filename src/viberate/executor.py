import pickle
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, List
import functools

from viberate.program import Program, Test, InputOutput, TestFunction, Pass, Fail
from viberate.utils import panic, ContentAddressable


# from LiveCodeBench/lcb_runner/evaluation/utils_execute.py
LIVECODEBENCH_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search  # Assuming 're' refers to a regex search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod  # 'log' and 'prod' are functions in the math module
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
# from sortedcontainers import SortedList, SortedDict, SortedSet
# import sortedcontainers
from operator import iand
import sys
"""


@dataclass
class Success:
    output: Any


@dataclass
class Error:
    '''Error raised by the executed function'''
    type: str
    message: str


@dataclass
class Panic:
    '''Failed to execute the function (e.g. malformed source code)'''
    message: str


@dataclass
class Timeout:
    pass


type ExecutionOutcome = Success | Error | Panic | Timeout


type TestOutcome = Pass | Fail | Error | Panic | Timeout


EXECUTION_TIMEOUT_SECONDS = 2


def test_harness(p: Program, input_file: Path, output_file: Path):
    return f"""
import pickle
if __name__ == '__main__':
    with open('{str(input_file)}', 'rb') as __input_file:
        __input_list = pickle.load(__input_file)
    report = dict()
    try:
        output = {p.signature.name}(*__input_list)
        report['status'] = 'success'
        report['value'] = output
    except Exception as e:
        report['status'] = 'error'
        report['error_type'] = type(e).__name__
        report['error_message'] = str(e)
    with open('{str(output_file)}', 'wb') as f:
        pickle.dump(report,f)
    """


def assertion_test_harness(p: Program, assertion: TestFunction, output_file: Path):
    return f"""
import pickle

{p.code}

{assertion.test_function_code}

if __name__ == '__main__':
    report = dict()
    try:
        {assertion.test_function_name}()
        report['status'] = 'success'
        report['test_passed'] = True
    except AssertionError as e:
        report['status'] = 'success'
        report['test_passed'] = False
        report['assertion_message'] = str(e)
    except Exception as e:
        report['status'] = 'error'
        report['error_type'] = type(e).__name__
        report['error_message'] = str(e)
    
    with open('{str(output_file)}', 'wb') as f:
        pickle.dump(report, f)
    """

def cache_content_addressable(func):
    """
    Decorator that caches method results using ContentAddressable.hash_id()
    for ContentAddressable arguments, and repr() for others.
    The cache is stored on the instance as _cache_<funcname>.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Build the cache storage on first use
        cache_attr = f"_cache_{func.__name__}"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)

        # Convert arguments to a key
        def arg_key(arg):
            if isinstance(arg, ContentAddressable):
                return f"CA:{arg.hash_id()}"
            else:
                return f"VAL:{repr(arg)}"

        key_parts = tuple(arg_key(a) for a in args) + \
                    tuple(f"{k}={arg_key(v)}" for k, v in sorted(kwargs.items()))

        # Check cache
        if key_parts in cache:
            print("c", end="", file=sys.stderr, flush=True)            
            return cache[key_parts]

        # Call original function and store result
        result = func(self, *args, **kwargs)
        cache[key_parts] = result
        return result

    return wrapper


class Executor:

    def __init__(self, test_venv: Path):
        self.test_venv = test_venv

    @cache_content_addressable
    def run(self, p: Program, inputs: list[Any], add_lcb_imports=True) -> ExecutionOutcome:
        assert isinstance(inputs, list)
        if add_lcb_imports:
            p = p.add_imports(LIVECODEBENCH_IMPORTS)
        with TemporaryDirectory() as tmp:
            exec_dir = Path(tmp)
            input_file = exec_dir / 'input.pkl'
            with input_file.open('wb') as f:
                pickle.dump(inputs, f)
            output_file = exec_dir / 'output.pkl'
            source_code = p.code + "\n" + test_harness(p, input_file, output_file)
            (exec_dir / 'code.py').write_text(source_code)
            try:
                interpreter = str(self.test_venv.resolve() / 'bin' / 'python')
                result = subprocess.run(
                    [interpreter, 'code.py'],
                    cwd=exec_dir,
                    capture_output=True,   # Captures stdout and stderr
                    text=True,             # Returns output as string, not bytes
                    timeout=EXECUTION_TIMEOUT_SECONDS,
                    check=False            # Don't raise exception for nonzero status
                )
                if result.returncode != 0:
                    print("!", end="", file=sys.stderr, flush=True)
                    return Panic(result.stderr)
                if not output_file.exists():
                    print("!", end="", file=sys.stderr, flush=True)
                    return Panic("no output")
                with output_file.open('rb') as f:
                    report = pickle.load(f)
                    if report['status'] == 'success':
                        print(".", end="", file=sys.stderr, flush=True)
                        return Success(report['value'])
                    else:
                        print("!", end="", file=sys.stderr, flush=True)
                        assert report['status'] == 'error'
                        return Error(report['error_type'], report['error_message'])
                
            except subprocess.TimeoutExpired:
                print("!", end="", file=sys.stderr, flush=True)
                return Timeout()          
    
    def _run_test_function(self, p: Program, assertion: TestFunction) -> TestOutcome:
        with TemporaryDirectory() as tmp:
            exec_dir = Path(tmp)
            output_file = exec_dir / 'output.pkl'
            source_code = assertion_test_harness(p, assertion, output_file)
            (exec_dir / 'code.py').write_text(source_code)
            
            try:
                interpreter = str(self.test_venv.resolve() / 'bin' / 'python')
                result = subprocess.run(
                    [interpreter, 'code.py'],
                    cwd=exec_dir,
                    capture_output=True,
                    text=True,
                    timeout=EXECUTION_TIMEOUT_SECONDS,
                    check=False
                )
                
                if result.returncode != 0:
                    return Panic(result.stderr)
                    
                if not output_file.exists():
                    return Panic("no output")
                    
                with output_file.open('rb') as f:
                    report = pickle.load(f)
                    if report['status'] == 'success':
                        if report['test_passed']:
                            print(".", end="", file=sys.stderr, flush=True)
                            return Pass()
                        else:
                            print("!", end="", file=sys.stderr, flush=True)
                            return Fail()
                    else:
                        print("!", end="", file=sys.stderr, flush=True)
                        return Error(report['error_type'], report['error_message'])
                        
            except subprocess.TimeoutExpired:
                return Timeout()

    @cache_content_addressable
    def run_test(self, p: Program, t: Test, add_lcb_imports=True) -> TestOutcome:
        if add_lcb_imports:
            p = p.add_imports(LIVECODEBENCH_IMPORTS)
        match t:
            case InputOutput(inputs, expected):
                execution_outcome = self.run(p, inputs)
                match execution_outcome:
                    case Success(actual):
                        if isinstance(actual, float) and isinstance(expected, float) and math.isclose(actual, expected):
                            return Pass()
                        elif actual == expected:
                            return Pass()
                        else:
                            return Fail()
                    case _:
                        return execution_outcome
            
            case TestFunction():
                return self._run_test_function(p, t)
