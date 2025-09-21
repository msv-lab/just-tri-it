import sys
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Iterator, List, Tuple

from just_tri_it.cached_llm import Model
from just_tri_it.executor import Executor, Pass
from just_tri_it.program import Test, Requirements
from just_tri_it.test_generator import TestGenerator
from just_tri_it.code_generator import (
    Selector,
    SelectionOutcome,
    Selected,
    Abstained,
    Generator
)
from just_tri_it.utils import ExperimentFailure, RawData


class CodeT(Selector):
    def __init__(self,
                 executor: Executor,
                 code_generator: Generator,
                 test_generator: TestGenerator,
                 num_programs: int,
                 num_tests: int):
        self.executor = executor
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.num_programs = num_programs
        self.num_tests = num_tests

    def generate_and_select(self, model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        """Schema:
        {
          "programs": ...,
          "tests": ...
        }
        """
        
        programs = list(islice(self.code_generator.generate(model, req, self.num_programs), self.num_programs))
        tests = list(islice(self.test_generator.generate(model, req), self.num_tests))

        if not programs or not tests:
            raise ExperimentFailure()
        
        results: List[List[bool]] = [[False for _ in range(len(tests))] for _ in range(len(programs))]

        for i, program in enumerate(programs):
            for j, test in enumerate(tests):
                try:
                    match self.executor.run_test(program, test):
                        case Pass():
                            results[i][j] = True
                        case _:
                            pass
                except Exception as e:
                    pass
                
        best_score = -1
        selected_programs = []
        
        for i in range(len(programs)):
            for j in range(len(tests)):
                if results[i][j]:
                    #FIXME: this formula is incorrect. It must find all programs that pass S_y, not y
                    s_x_count = sum(1 for k in range(len(programs)) if results[k][j])
                    s_y_count = sum(1 for l in range(len(tests)) if results[i][l])
                    score = s_x_count * s_y_count
                    #print(f"Program {i} passes test {j}: s_x={s_x_count}, s_y={s_y_count}, score={score}")

                    if score > best_score:
                        best_score = score
                        selected_programs = [programs[i]]
                    elif score == best_score:
                        selected_programs.append(programs[i])

        raw_data = {
            "programs": programs,
            "tests": tests
        }

        if len(selected_programs) > 0:
            return (Selected(selected_programs), raw_data)
        else:
            return (Abstained(), raw_data)
