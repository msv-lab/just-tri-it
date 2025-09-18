from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Iterator, List, Tuple
from viberate.cached_llm import Model
from viberate.executor import Executor, Pass
from viberate.program import Test, ExpectedOutput, Assertion
from viberate.requirements import Requirements
from viberate.test_generator import generate_test_cases
from viberate.code_generator import (
    Selector,
    SelectionOutcome,
    Selected,
    Abstained,
    Generator
)
from viberate.dataset import Task


class MODE(Enum):
    IO_COMPARE = 1
    ASSERTION = 2


class CodeT(Selector):
    def __init__(self, executor: Executor, generator: Generator, mode: MODE, n: int, m: int):
        self.executor = executor
        self.generator = generator
        self.n = n
        self.m = m
        self.mode = mode

    def generate_and_select(self, model: Model, task: Task, p_dir, p_dict):
        req = task.requirements
        bench_tests = task.tests
        exp_results = {
            "generated_programs": [],
            "generated_tests": [],
            "chosen_programs": [],
            "decision": None
        }
        tests, tests_new_form = self._generate_tests(model, req, self.mode)
        exp_results["generated_tests"] = tests_new_form

        programs = list(islice(self.generator.generate(model, req, p_dir, self.n), self.n))
        exp_results["generated_programs"] = [p.hash() for p in programs]

        p_dict = Selector.update_program_correctness(task.id, self.executor, programs, bench_tests, p_dict)

        if not programs or not tests:
            exp_results["decision"] = "Abstained"
            return exp_results, p_dict
        
        results: List[List[bool]] = [[False for _ in range(len(tests))] for _ in range(self.n)]

        
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
        selected_p_hash = []
        
        for i in range(self.n):
            for j in range(len(tests)):
                if results[i][j]:
                    s_x_count = sum(1 for k in range(self.n) if results[k][j])
                    s_y_count = sum(1 for l in range(len(tests)) if results[i][l])
                    score = s_x_count * s_y_count
                    print(f"Program {i} passes test {j}: s_x={s_x_count}, s_y={s_y_count}, score={score}")
                    if score > best_score:
                        best_score = score
                        selected_p_hash = [exp_results["generated_programs"][i]]
                    elif score == best_score:
                        if exp_results["generated_programs"][i] not in selected_p_hash:
                            selected_p_hash.append(exp_results["generated_programs"][i])

        if selected_p_hash is not None:
            exp_results["chosen_programs"] = selected_p_hash
            exp_results["decision"] = "Selected"
        else:
            exp_results["decision"] = "Abstained"
        return exp_results, p_dict
    
    def _generate_tests(self, model: Model, req: Requirements, mode: MODE):
        tests = []
        tests_new_form = []
        if mode == MODE.IO_COMPARE:
            try:
                traditional_tests = list(islice(generate_test_cases(model, req, self.executor), self.m))
                for test_input, expected_output in traditional_tests:
                    if len(test_input) != len(req.signature.params) and len(req.signature.params) == 1:
                        adjusted_input = [test_input]
                    else:
                        adjusted_input = test_input

                    test = Test(adjusted_input, ExpectedOutput(expected_output))
                    tests.append(test)
                    tests_new_form.append(test.special_reformat())
            except Exception as e:
                print(f"Failed to generate traditional tests: {e}")
        elif mode == MODE.ASSERTION:
            try:
                print(f"Attempting to generate assertion tests for signature: {req.signature.pretty_print()}")
                assertions = Assertion.generate_from_problem(model, req.description, req.signature, num_tests=self.m)
                print(f"Successfully generated {len(assertions)} assertion tests")
                for assertion in assertions:
                    test = Test.from_assertion(assertion)
                    tests.append(test)
                    tests_new_form.append(test.special_reformat())
            except Exception as e:
                print(f"Failed to generate assertion tests: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unsupported mode: {mode}")

        return tests, tests_new_form
