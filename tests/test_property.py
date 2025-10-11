import os
from pathlib import Path

from just_tri_it.logic import Side
from just_tri_it.property_checker import EvaluatorChecker
from just_tri_it.executor import PersistentWorkerExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import (
    make_partial_fwd_inv,
    make_partial_fwd_sinv,
    make_syntactic,
    make_trivial_semantic
)

import pytest


@pytest.fixture()
def checker():
    executor = PersistentWorkerExecutor()
    yield EvaluatorChecker(executor)
    executor.shutdown()


FWD_INV_0 = make_partial_fwd_inv(2, 0).hyperproperty

FWD_SINV_0 = make_partial_fwd_sinv(2, 0).hyperproperty

EQUIV = make_syntactic(2).hyperproperty

OFF_BY_ONE = make_trivial_semantic(2).hyperproperty


def check(checker, p, q, inputs, prop):
    return checker.check({ Side.LEFT: inputs, Side.RIGHT: [] },
                         { Side.LEFT: p, Side.RIGHT: q },
                         prop)


FWD_PLUS_INPUTS = [
    [1, 1],
    [0, 1],
    [-1, 0],
    [4, 3],
    [4, 4]
]

FWD_PLUS = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    return a + b
""")

BUGGY_FWD_PLUS = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 1:
        return a + b + 1
    return a + b
""")

def test_plus_equiv_reflective(checker):
    assert check(checker, FWD_PLUS, FWD_PLUS, FWD_PLUS_INPUTS, EQUIV)

def test_plus_equiv_detect_buggy(checker):
    assert not check(checker, FWD_PLUS, BUGGY_FWD_PLUS, FWD_PLUS_INPUTS, EQUIV)


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

def test_plus_fwd_inv_correct_agree(checker):
    assert check(checker, FWD_PLUS, INV, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_detect_buggy_fwd(checker):
    assert not check(checker, BUGGY_FWD_PLUS, INV, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_detect_buggy_inv(checker):
    assert not check(checker, FWD_PLUS, BUGGY_INV, FWD_PLUS_INPUTS, FWD_INV_0)

FWD_PLUS_INVALID_INPUT = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 4 and b == 3:
        raise ValueError('Invalid input')
    return a + b
""")

FWD_PLUS_CRASH = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 4 and b == 3:
        raise Exception('Crash')
    return a + b
""")

def test_plus_equiv_agree_invalid(checker):
    assert check(checker, FWD_PLUS_INVALID_INPUT, FWD_PLUS_INVALID_INPUT, FWD_PLUS_INPUTS, EQUIV)

def test_plus_equiv_disagree_crashes(checker):
    assert not check(checker, FWD_PLUS_CRASH, FWD_PLUS_CRASH, FWD_PLUS_INPUTS, EQUIV)
    
def test_plus_equiv_detect_invalid(checker):
    assert not check(checker, FWD_PLUS, FWD_PLUS_INVALID_INPUT, FWD_PLUS_INPUTS, EQUIV)

def test_plus_equiv_detect_crash(checker):
    assert not check(checker, FWD_PLUS, FWD_PLUS_CRASH, FWD_PLUS_INPUTS, EQUIV)

def test_plus_equiv_crash_is_not_invalid(checker):
    assert not check(checker, FWD_PLUS_INVALID_INPUT, FWD_PLUS_CRASH, FWD_PLUS_INPUTS, EQUIV)


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

def test_plus_fwd_inv_detect_inv_invalid(checker):
    assert not check(checker, FWD_PLUS, INV_MATCHING_INVALID_INPUT, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_detect_inv_crash(checker):
    assert not check(checker, FWD_PLUS, INV_MATCHING_CRASH, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_detect_fwd_crash(checker):
    assert not check(checker, FWD_PLUS_CRASH, INV, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_agree_invalid(checker):
    assert check(checker, FWD_PLUS_INVALID_INPUT, INV_MATCHING_INVALID_INPUT, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_disagree_invalid(checker):
    assert not check(checker, FWD_PLUS_INVALID_INPUT, INV_NONMATCHING_INVALID_INPUT, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_disagree_fwd_crash_inv_invalid(checker):
    assert not check(checker, FWD_PLUS_CRASH, INV_NONMATCHING_INVALID_INPUT, FWD_PLUS_INPUTS, FWD_INV_0)

def test_plus_fwd_inv_disagree_crashes(checker):
    assert not check(checker, FWD_PLUS_CRASH, INV_MATCHING_CRASH, FWD_PLUS_INPUTS, FWD_INV_0)
    

FWD_CLOSEST_POWER_INPUTS = [
  [1, 2],  # x equals y**0 → k=0
  [2, 2],  # exact power: 2^1=2 → k=1
  [4, 2],  # exact power: 2^2=4 → k=2
  [3, 2],  # tie between 2 and 4; distances 1 and 1 → choose smaller k → k=1
  [8, 2],  # exact power: 2^3=8 → k=3
  [9, 3],  # exact power: 3^2=9 → k=2
  [10, 3], # between 9 and 27; distances 1 vs 17 → k=2
  [12, 3], # between 9 and 27; distances 3 vs 15 → k=2 (not a tie)
  [18, 3], # tie between 9 and 27; distances 9 and 9 → choose smaller k → k=2
  [5, 2],  # between 4 and 8; distances 1 vs 3 → k=2
  [512, 2],  # exact at loop upper bound: 2^9=512 → k=9
  [300, 3],  # between 243 (3^5) and 729 (3^6); distances 57 vs 429 → k=5
  [2, 3],    # between 1 and 3; distances 1 vs 1? by tie rule → k=0
  [3, 3],    # exact power: 3^1=3 → k=1
  [0, 2],    # x < 1; prev_power=1 >= x → k=0
  [10, 1],   # invalid y <= 1 → raises ValueError
  [10, 0],   # invalid y <= 1 → raises ValueError
  [10, -2],  # invalid y <= 1 → raises ValueError
  [100, 2],  # between 64 (2^6) and 128 (2^7); distances 36 vs 28 → k=7
  [70, 2],   # between 64 and 128; distances 6 vs 58 → k=6
  [1000, 3], # between 729 (3^6) and 2187 (3^7); distances 271 vs 1187 → k=6
  [1500, 3], # between 729 and 2187; distances 771 vs 687 → k=7
]

FWD_CLOSEST_POWER = Program.from_function_code("""
def closest_power(x: int, y: int) -> int:
    '''
    Given x > 0 and y > 1, return k >= 0 such that y**k is closest to x. In ties, return the smaller power.
    '''
    if y <= 1:
        raise ValueError("Invalid input")

    if x < 1:
        raise ValueError("Invalid input")

    prev_power = 1  # y**0
    if prev_power >= x:
        return 0

    for i in range(1, 20):
        curr_power = prev_power * y
        if curr_power >= x:
            if x - prev_power <= curr_power - x:
                return i - 1
            else:
                return i
        prev_power = curr_power
    raise Exception("Too large arguments")
""")

def test_closest_power_equiv_reflective(checker):
    assert check(checker, FWD_CLOSEST_POWER, FWD_CLOSEST_POWER, FWD_CLOSEST_POWER_INPUTS, EQUIV)


SINV_CLOSEST_POWER = Program.from_function_code("""
def closest_power_sinv_x(k: int, y: int) -> list[int]:
    '''
    Given k >= 0 and y > 1, return an exhaustive list of x such that y**k is
    closest to x among different k (in ties, the smaller k is chosen).
    '''
    if y <= 1:
        raise ValueError("Invalid input")
    if k < 0:
        raise ValueError("Invalid input")

    def mid_floor(a: int, b: int) -> int:
        return (a + b) // 2

    yk = pow(y, k)

    if k == 0:
        y1 = y  # y^1
        # x in [1, floor((1 + y)/2)]
        lo = 1
        hi = mid_floor(1, y1)
    else:
        ykm1 = yk // y            # y^(k-1)
        ykp1 = yk * y             # y^(k+1)

        # Lower bound: strictly greater than midpoint with previous power
        low_mid = mid_floor(yk, ykm1)
        lo = low_mid + 1

        # Upper bound: at most midpoint with next power
        hi = mid_floor(yk, ykp1)

    if lo > hi:
        return []  # No integers satisfy (can happen for small y/k due to tight midpoints)

    return list(range(lo, hi + 1))
""")

def test_closest_power_correct_fwd_sinv_agree(checker):
    assert check(checker, FWD_CLOSEST_POWER, SINV_CLOSEST_POWER, FWD_CLOSEST_POWER_INPUTS, FWD_SINV_0)
