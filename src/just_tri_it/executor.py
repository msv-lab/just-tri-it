import pickle
import math
import sys
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
import functools
from copy import deepcopy
from abc import ABC, abstractmethod
import multiprocessing
import time


from just_tri_it.program import Program, Signature, Test, InputOutput, TestFunction, Pass, Fail
from just_tri_it.utils import ContentAddressable


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


class Executor(ABC):

    @abstractmethod
    def run(self, p: Program, inputs: list[Any], add_lcb_imports=True) -> ExecutionOutcome:
        pass

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
            
            case TestFunction(test_function_name, test_function_code, target_signature):
                test_sig = Signature(test_function_name, [], "None")
                test_program = Program(test_sig, p.code + "\n" + test_function_code)
                execution_outcome = self.run(test_program, [])
                match execution_outcome:
                    case Success(_):
                        return Pass()
                    case Error(t, _):
                        if t == "AssertionError":
                            return Fail()
                        else:
                            return execution_outcome
                    case _:
                        return execution_outcome

                return self._run_test_function(p, t)
        pass


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


class SubprocessExecutor(Executor):

    def __init__(self, test_venv: Path):
        self.test_venv = test_venv

    def _test_harness(self, p: Program, input_file: Path, output_file: Path, load_inputs: bool):
        if load_inputs:
            inputs_loader = f"""\
    with open('{str(input_file)}', 'rb') as __input_file:
        __input_list = pickle.load(__input_file)"""
        else:
            inputs_loader = """\
    __input_list = []"""
        return f"""
import pickle
if __name__ == '__main__':
    report = dict()
{inputs_loader}
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

    @cache_content_addressable
    def run(self, p: Program, inputs, add_lcb_imports=True) -> ExecutionOutcome:
        assert isinstance(inputs, list)
        if add_lcb_imports:
            p = p.add_imports(LIVECODEBENCH_IMPORTS)
        with TemporaryDirectory() as tmp:
            exec_dir = Path(tmp)
            input_file = exec_dir / 'input.pkl'
            if len(inputs) > 0:
                with input_file.open('wb') as f:
                    pickle.dump(inputs, f)
            output_file = exec_dir / 'output.pkl'
            source_code = p.code + "\n" + self._test_harness(p, input_file, output_file, len(inputs) > 0)
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
                print("T", end="", file=sys.stderr, flush=True)
                return Timeout()          
    
    def shutdown(self):
        pass


class PersistentWorkerExecutor(Executor):

    def _runner(self, task_queue: multiprocessing.Queue, return_queue: multiprocessing.Queue):
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        callable_cache = {}
        
        while True:
            item = task_queue.get()
            if item is None:
                break  # Shutdown signal
            program_code, func_name, args = item

            try:
                func = None
                if program_code in callable_cache:
                    func = callable_cache[program_code]
                else:
                    namespace = {}

                    exec(program_code, namespace)

                    if func_name in namespace and callable(namespace[func_name]):
                        func = namespace[func_name]
                        callable_cache[program_code] = func
                    else:
                        return_queue.put({ "status": "error",
                                           "error_type": "panic",
                                           "error_message": "no function found" })

                if func is not None:
                    result = func(*args)

                    return_queue.put({ "status": "success",
                                       "value": result })

            except Exception as e:
                return_queue.put({ "status": "error",
                                   "error_type": type(e).__name__,
                                   "error_message": str(e) })

    def __init__(self):
        self.task_queue = None
        self.result_queue = None
        self.worker = None
        self._start_worker()

    def _start_worker(self):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker = multiprocessing.Process(
            target=self._runner, 
            args=(self.task_queue, self.result_queue)
        )
        self.worker.start()

    def _restart_worker(self):
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join()
        self._start_worker()

    def shutdown(self):
        self.task_queue.put(None)
        self.worker.join()

    def _execute_code(self, program_code: str, func_name: str, args: list, timeout: int):
        self.task_queue.put((program_code, func_name, args))
        start = time.time()
        
        while time.time() - start < timeout:
            if not self.result_queue.empty():
                report = self.result_queue.get()
                if report['status'] == 'success':
                    print(".", end="", file=sys.stderr, flush=True)
                    return Success(report['value'])
                else:
                    print("!", end="", file=sys.stderr, flush=True)
                    assert report['status'] == 'error'
                    if report['error_type'] == 'panic':
                        return Panic(report['error_message'])
                    return Error(report['error_type'], report['error_message'])
            time.sleep(0.01)  # Small delay to avoid busy wait

        # Timeout
        self._restart_worker()
        print("T", end="", file=sys.stderr, flush=True)
        return Timeout()

    @cache_content_addressable
    def run(self, p: Program, inputs, add_lcb_imports=True) -> ExecutionOutcome:
        assert isinstance(inputs, list)
        args = deepcopy(inputs)
        if add_lcb_imports:
            p = p.add_imports(LIVECODEBENCH_IMPORTS)
        return self._execute_code(p.code, p.signature.name, args, EXECUTION_TIMEOUT_SECONDS)
