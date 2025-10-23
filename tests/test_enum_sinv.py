import os
from pathlib import Path

from just_tri_it.logic import Side
from just_tri_it.property_checker import Interpreter
from just_tri_it.executor import PersistentWorkerExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import (
    make_partial_enum_sinv
)
from just_tri_it.inversion import ParameterInversion

import pytest


@pytest.fixture()
def checker():
    executor = PersistentWorkerExecutor()
    yield Interpreter(executor)
    executor.shutdown()


ENUM_SINV_0 = make_partial_enum_sinv(1, ParameterInversion(0)).hyperproperty


INPUTS = [
    [-1],
    [0],
    [1],
    [2]
]


def check(checker, p, q):
    return checker.check({ Side.LEFT: INPUTS, Side.RIGHT: INPUTS },
                         { Side.LEFT: p, Side.RIGHT: q },
                         ENUM_SINV_0)


ENUM = Program.from_function_code("""
def enum(i: int) -> list[int]:
    '''returns all integers strictly greater than i by at most 2'''
    return [i+1, i+2]
""")

SINV = Program.from_function_code("""
def enum(i: int) -> list[int]:
    '''returns all integers strictly smaller than i by at most 2'''
    return [i-1, i-2]
""")

def test_correct_enum_sinv(checker):
    assert check(checker, ENUM, SINV)





ENUM_MISS = Program.from_function_code("""
def enum(i: int) -> list[int]:
    return [i + 1]  
""")

def test_enum_sinv_detect_enum_missing_values(checker):
    assert not check(checker, ENUM_MISS, SINV)


ENUM_OVER = Program.from_function_code("""
def enum(i: int) -> list[int]:
    return [i + 1, i + 2, i + 3]  
""")

def test_enum_sinv_detect_enum_overapprox(checker):
    assert not check(checker, ENUM_OVER, SINV)


ENUM_SHIFTED = Program.from_function_code("""
def enum(i: int) -> list[int]:
    return [i + 2, i + 3]  
""")

def test_enum_sinv_detect_enum_wrong_relation(checker):
    assert not check(checker, ENUM_SHIFTED, SINV)



SINV_MISS = Program.from_function_code("""
def enum(i: int) -> list[int]:
    return [i - 1]  
""")

def test_enum_sinv_detect_inverse_missing(checker):
    assert not check(checker, ENUM, SINV_MISS)


SINV_OVER = Program.from_function_code("""
def enum(i: int) -> list[int]:
    return [i - 1, i - 2, i - 3]  
""")

def test_enum_sinv_detect_inverse_overapprox(checker):
    assert not check(checker, ENUM, SINV_OVER)


SINV_SHIFTED = Program.from_function_code("""
def enum(i: int) -> list[int]:
    return [i - 2, i - 3]  
""")

def test_enum_sinv_detect_inverse_wrong_relation(checker):
    assert not check(checker, ENUM, SINV_SHIFTED)





ENUM_INVALID_ON_ZERO = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i <= 0 :
        raise ValueError("Invalid input")
    return [i + 1, i + 2]
""")

SINV_INVALID_ON_ZERO = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i <= 0 :
        raise ValueError("Invalid input")
    return [i - 1, i - 2]
""")

def test_enum_sinv_disagree_on_mismatched_invalid(checker):
    assert not check(checker, ENUM_INVALID_ON_ZERO, SINV)   
    assert not check(checker, ENUM, SINV_INVALID_ON_ZERO)   


SINV_INVALID_ON_ONE = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i == 1:
        raise ValueError("Invalid input")
    return [i - 1, i - 2]
""")

def test_enum_sinv_disagree_on_invalid_points_mismatch(checker):
    assert not check(checker, ENUM_INVALID_ON_ZERO, SINV_INVALID_ON_ONE)




ENUM_CRASH_ON_ONE = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i == 1:
        raise Exception("Crash")
    return [i + 1, i + 2]
""")

def test_enum_sinv_disagree_on_crash(checker):
    assert not check(checker, ENUM_CRASH_ON_ONE, SINV)


SINV_CRASH_ON_ONE = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i == 1:
        raise Exception("Crash")
    return [i - 1, i - 2]
""")

def test_enum_sinv_both_crash_still_fail(checker):
    assert not check(checker, ENUM_CRASH_ON_ONE, SINV_CRASH_ON_ONE)





ENUM_PSEUDO_INFINITE = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i >= 0:
        return list(range(i + 1, i + 1 + 100000))
    return [i + 1, i + 2]
""")

def test_enum_sinv_detect_enumerator_pseudo_infinite(checker):
    assert not check(checker, ENUM_PSEUDO_INFINITE, SINV)


SINV_PSEUDO_INFINITE = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i >= 0:
        return list(range(i - 100000, i))
    return [i - 1, i - 2]
""")

def test_enum_sinv_detect_inverse_pseudo_infinite(checker):
    assert not check(checker, ENUM, SINV_PSEUDO_INFINITE)

def test_enum_sinv_both_pseudo_infinite_still_fail(checker):
    assert not check(checker, ENUM_PSEUDO_INFINITE, SINV_PSEUDO_INFINITE)




ENUM_INFINITE_SIGNALED = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i == 1:
        raise Exception("Infinite/Non-enumerable output set")
    return [i + 1, i + 2]
""")

def test_enum_sinv_enumerator_signaled_infinite_is_failure(checker):
    assert not check(checker, ENUM_INFINITE_SIGNALED, SINV)

SINV_INFINITE_SIGNALED = Program.from_function_code("""
def enum(i: int) -> list[int]:
    if i == 1:
        raise Exception("Infinite/Non-enumerable fiber")
    return [i - 1, i - 2]
""")

def test_enum_sinv_inverse_signaled_infinite_is_failure(checker):
    assert not check(checker, ENUM, SINV_INFINITE_SIGNALED)

def test_enum_sinv_both_infinite_like_still_fail(checker):
    assert not check(checker, ENUM_PSEUDO_INFINITE, SINV_INFINITE_SIGNALED)

    assert not check(checker, ENUM_INFINITE_SIGNALED, SINV_INFINITE_SIGNALED)
