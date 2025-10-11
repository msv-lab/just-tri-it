import os
from pathlib import Path

from just_tri_it.logic import Side
from just_tri_it.property_checker import Interpreter
from just_tri_it.executor import PersistentWorkerExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import (
    make_partial_enum_sinv
)

import pytest


@pytest.fixture()
def checker():
    executor = PersistentWorkerExecutor()
    yield Interpreter(executor)
    executor.shutdown()


ENUM_SINV_0 = make_partial_enum_sinv(1, 0).hyperproperty


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
def enum(i: int) -> int:
    '''returns all integers strictly greater than i by at most 2'''
    return [i+1, i+2]
""")

SINV = Program.from_function_code("""
def enum(i: int) -> int:
    '''returns all integers strictly smaller than i by at most 2'''
    return [i-1, i-2]
""")

def test_correct_enum_sinv(checker):
    assert check(checker, ENUM, SINV)
