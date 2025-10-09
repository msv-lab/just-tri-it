import os
from pathlib import Path

from just_tri_it.logic import Side
from just_tri_it.property_checker import EvaluatorChecker
from just_tri_it.executor import PersistentWorkerExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import (
    make_partial_for_inv,
    make_syntactic,
    make_trivial_semantic
)

import pytest


FWD_INPUTS = [
    [1, 1],
    [0, 1],
    [-1, 0],
    [4, 3],
    [4, 4]
]


FWD_INV_0 = make_partial_for_inv(2, 0).hyperproperty

EQUIV = make_syntactic(2).hyperproperty

OFF_BY_ONE = make_trivial_semantic(2).hyperproperty


def check(checker, p, q, prop):
    return checker.check({ Side.LEFT: FWD_INPUTS, Side.RIGHT: [] },
                         { Side.LEFT: p, Side.RIGHT: q },
                         prop)


@pytest.fixture()
def checker():
    executor = PersistentWorkerExecutor()
    yield EvaluatorChecker(executor)
    executor.shutdown()


FWD = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    return a + b
""")

BUGGY_FWD = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 1:
        return a + b + 1
    return a + b
""")

def test_equiv_reflective(checker):
    assert check(checker, FWD, FWD, EQUIV)

def test_equiv_detect_buggy(checker):
    assert not check(checker, FWD, BUGGY_FWD, EQUIV)


INV = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    return sum_of_two_numbers - b
""")

BUGGY_INV = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if b == 1:
        return sum_of_two_numbers - b + 1
    return sum_of_two_numbers - b
""")

def test_fwd_inv_correct_agree(checker):
    assert check(checker, FWD, INV, FWD_INV_0)

def test_fwd_inv_detect_buggy_fwd(checker):
    assert not check(checker, BUGGY_FWD, INV, FWD_INV_0)

def test_fwd_inv_detect_buggy_inv(checker):
    assert not check(checker, FWD, BUGGY_INV, FWD_INV_0)

FWD_INVALID_INPUT = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 4 and b == 3:
        raise ValueError('Invalid input')
    return a + b
""")

FWD_CRASH = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 4 and b == 3:
        raise Exception('Crash')
    return a + b
""")

def test_equiv_agree_invalid(checker):
    assert check(checker, FWD_INVALID_INPUT, FWD_INVALID_INPUT, EQUIV)

def test_equiv_disagree_crashes(checker):
    assert not check(checker, FWD_CRASH, FWD_CRASH, EQUIV)
    
def test_equiv_detect_invalid(checker):
    assert not check(checker, FWD, FWD_INVALID_INPUT, EQUIV)

def test_equiv_detect_crash(checker):
    assert not check(checker, FWD, FWD_CRASH, EQUIV)

def test_equiv_crash_is_not_invalid(checker):
    assert not check(checker, FWD_INVALID_INPUT, FWD_CRASH, EQUIV)


INV_MATCHING_INVALID_INPUT = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if sum_of_two_numbers == 7 and b == 3:
        raise ValueError('Invalid input')
    return sum_of_two_numbers - b
""")

INV_NONMATCHING_INVALID_INPUT = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if sum_of_two_numbers == 8 and b == 4:
        raise ValueError('Invalid input')
    return sum_of_two_numbers - b
""")

INV_MATCHING_CRASH = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if sum_of_two_numbers == 7 and b == 3:
        raise Exception('Crash')
    return sum_of_two_numbers - b
""")

def test_fwd_inv_detect_inv_invalid(checker):
    assert not check(checker, FWD, INV_MATCHING_INVALID_INPUT, FWD_INV_0)

def test_fwd_inv_detect_inv_crash(checker):
    assert not check(checker, FWD, INV_MATCHING_CRASH, FWD_INV_0)

def test_fwd_inv_detect_fwd_crash(checker):
    assert not check(checker, FWD_CRASH, INV, FWD_INV_0)

def test_fwd_inv_agree_invalid(checker):
    assert check(checker, FWD_INVALID_INPUT, INV_MATCHING_INVALID_INPUT, FWD_INV_0)

def test_fwd_inv_disagree_invalid(checker):
    assert not check(checker, FWD_INVALID_INPUT, INV_NONMATCHING_INVALID_INPUT, FWD_INV_0)

def test_fwd_inv_disagree_fwd_crash_inv_invalid(checker):
    assert not check(checker, FWD_CRASH, INV_NONMATCHING_INVALID_INPUT, FWD_INV_0)

def test_fwd_inv_disagree_crashes(checker):
    assert not check(checker, FWD_CRASH, INV_MATCHING_CRASH, FWD_INV_0)
    



