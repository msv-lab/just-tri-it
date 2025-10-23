import os
from pathlib import Path

from just_tri_it.logic import Side
from just_tri_it.property_checker import Interpreter
from just_tri_it.executor import PersistentWorkerExecutor, SubprocessExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import (
    make_partial_fwd_sinv,
)

import pytest


@pytest.fixture()
def checker():
    executor = PersistentWorkerExecutor()
    yield Interpreter(executor)
    executor.shutdown()


FWD_SINV = make_partial_fwd_sinv(1, 0).hyperproperty


def check(checker, p, q, inputs, prop):
    return checker.check({ Side.LEFT: inputs, Side.RIGHT: [] },
                         { Side.LEFT: p, Side.RIGHT: q },
                         prop)


FWD_INPUTS = [
    [4],
    [0],
    [-1]
]

FWD = Program.from_function_code("""
def closer_to_0_or_3(a: int) -> int:
    if a <= 1:
        return 0
    return 3
""")

SINV = Program.from_function_code("""
def sinv_closer_to_0_or_3(a: int) -> tuple[bool,int]:
    if a == 0:
        return (False, list(range(-5, 2)))
    if a == 3:
        return (False, list(range(2, 5)))
    raise ValueError('Invalid input')
""")

def test_smaller_fwd_sinv_correct_agree(checker):
    assert check(checker, FWD, SINV, FWD_INPUTS, FWD_SINV)

BUGGY_SINV = Program.from_function_code("""
def sinv_closer_to_0_or_3(a: int) -> tuple[bool,int]:
    if a == 0:
        return (False, list(range(-5, 1)))
    if a == 3:
        return (False, list(range(1, 5)))
    raise ValueError('Invalid input')
""")

def test_smaller_fwd_sinv_incorrect_disagree(checker):
    assert not check(checker, FWD, BUGGY_SINV, FWD_INPUTS, FWD_SINV)
