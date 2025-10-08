import os
from pathlib import Path

from just_tri_it.logic import Side
from just_tri_it.property_checker import EvaluatorChecker
from just_tri_it.executor import PersistentWorkerExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import make_partial_for_inv

import pytest



FORWARD_PROGRAM = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    return a + b
""")

FORWARD_PROGRAM_WITH_INVALID_INPUT = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 4 and b == 3:
        raise ValueError('Invalid input')
    return a + b
""")

FORWARD_PROGRAM_WITH_CRASH = Program.from_function_code("""
def plus(a: int, b: int) -> int:
    if a == 4 and b == 3:
        raise Exception('Crash')
    return a + b
""")


INVERSE_PROGRAM = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    return sum_of_two_numbers - b
""")

INVERSE_PROGRAM_WITH_MATCHING_INVALID_INPUT = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if sum_of_two_numbers == 7 and b == 3:
        raise ValueError('Invalid input')
    return sum_of_two_numbers - b
""")

INVERSE_PROGRAM_WITH_NONMATCHING_INVALID_INPUT = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if sum_of_two_numbers == 8 and b == 4:
        raise ValueError('Invalid input')
    return sum_of_two_numbers - b
""")

INVERSE_PROGRAM_WITH_MATCHING_CRASH = Program.from_function_code("""
def inverse_plus_wrt_a(sum_of_two_numbers: int, b: int) -> int:
    if sum_of_two_numbers == 7 and b == 3:
        raise Exception('Crash')
    return sum_of_two_numbers - b
""")


FORWARD_INPUTS = [
    [1, 1],
    [0, 1],
    [-1, 0],
    [4, 3],
    [4, 4]
]

TRIANGULATION = make_partial_for_inv(2, 0)

def check_programs(checker, p, q):
    return checker.check({ Side.LEFT: FORWARD_INPUTS, Side.RIGHT: [] },
                         { Side.LEFT: p, Side.RIGHT: q },
                         TRIANGULATION.hyperproperty)


@pytest.fixture()
def checker():
    executor = PersistentWorkerExecutor()
    yield EvaluatorChecker(executor)
    executor.shutdown()


def test_normal_agreement(checker):
    assert check_programs(checker, FORWARD_PROGRAM, INVERSE_PROGRAM)


def test_detect_right_invalid_inputs(checker):
    assert not check_programs(checker, FORWARD_PROGRAM, INVERSE_PROGRAM_WITH_MATCHING_INVALID_INPUT)


def test_detect_right_crash(checker):
    assert not check_programs(checker, FORWARD_PROGRAM, INVERSE_PROGRAM_WITH_MATCHING_CRASH)


def test_detect_left_crash(checker):
    assert not check_programs(checker, FORWARD_PROGRAM_WITH_CRASH, INVERSE_PROGRAM)


def test_agreement_on_invalid_inputs(checker):
    assert check_programs(checker, FORWARD_PROGRAM_WITH_INVALID_INPUT, INVERSE_PROGRAM_WITH_MATCHING_INVALID_INPUT)


def test_disagreement_on_invalid_inputs(checker):
    assert not check_programs(checker, FORWARD_PROGRAM_WITH_INVALID_INPUT, INVERSE_PROGRAM_WITH_NONMATCHING_INVALID_INPUT)


def test_left_crash_right_invalid(checker):
    assert not check_programs(checker, FORWARD_PROGRAM_WITH_CRASH, INVERSE_PROGRAM_WITH_NONMATCHING_INVALID_INPUT)


def test_should_not_agree_on_matching_crashes(checker):
    assert not check_programs(checker, FORWARD_PROGRAM_WITH_CRASH, INVERSE_PROGRAM_WITH_MATCHING_CRASH)
    



