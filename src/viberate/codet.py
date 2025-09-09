from dataclasses import dataclass
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


class CodeT(Selector):
    def __init__(self, executor: Executor, generator: Generator, n: int = 10, m: int = 10):
        self.executor = executor
        self.generator = generator
        self.n = n
        self.m = m
    
    def generate_and_select(self, model: Model, req: Requirements):
        programs = list(islice(self.generator.generate(model, req), self.n))
        
        tests = self._generate_tests(model, req)
        
        if not programs or not tests:
            return Abstained()
        
        results: List[List[bool]] = [[False for _ in range(len(tests))] for _ in range(self.n)]
        
        for i, program in enumerate(programs):
            for j, test in enumerate(tests):
                try:
                    test_passed = self._execute_test(program, test, req)
                    results[i][j] = test_passed
                except Exception as e:
                    pass
        
        best_score = -1
        selected_program_id = None
        
        for i in range(self.n):
            for j in range(len(tests)):
                if results[i][j]:
                    s_x_count = sum(1 for k in range(self.n) if results[k][j])
                    s_y_count = sum(1 for l in range(len(tests)) if results[i][l])
                    score = s_x_count * s_y_count
                    print(f"Program {i} passes test {j}: s_x={s_x_count}, s_y={s_y_count}, score={score}")
                    
                    if score > best_score:
                        best_score = score
                        selected_program_id = i
        
        if selected_program_id is not None:
            return selected_program_id, programs, Selected(programs[selected_program_id])
        else:
            return None, programs, Abstained()
    
    def _generate_tests(self, model: Model, req: Requirements) -> List[Test]:
        tests = []
        
        try:
            traditional_tests = list(islice(generate_test_cases(model, req, self.executor), self.m))
            for test_input, expected_output in traditional_tests:
                if len(test_input) != len(req.signature.params) and len(req.signature.params) == 1:
                    adjusted_input = [test_input]
                else:
                    adjusted_input = test_input
                
                test = Test(adjusted_input, ExpectedOutput(expected_output))
                tests.append(test)
        except Exception as e:
            print(f"Failed to generate traditional tests: {e}")
        
        try:
            print(f"Attempting to generate assertion tests for signature: {req.signature.pretty_print()}")
            assertions = Assertion.generate_from_signature(
                model, 
                req.signature, 
                num_tests=min(3, self.m - len(tests))
            )
            print(f"Successfully generated {len(assertions)} assertion tests")
            for assertion in assertions:
                test = Test.from_assertion(assertion)
                tests.append(test)
        except Exception as e:
            print(f"Failed to generate assertion tests: {e}")
            import traceback
            traceback.print_exc()
        
        if len(tests) < self.m:
            try:
                additional_tests = list(islice(
                    generate_test_cases(model, req, self.executor), 
                    self.m - len(tests)
                ))
                for test_input, expected_output in additional_tests:
                    if len(test_input) != len(req.signature.params) and len(req.signature.params) == 1:
                        adjusted_input = [test_input]
                    else:
                        adjusted_input = test_input
                    
                    test = Test(adjusted_input, ExpectedOutput(expected_output))
                    tests.append(test)
            except Exception:
                pass
        
        return tests[:self.m]
    
    def _execute_test(self, program, test: Test, req: Requirements) -> bool:
        match test.oracle:
            case ExpectedOutput(_):
                match self.executor.run_test(program, test):
                    case Pass():
                        return True
                    case _:
                        return False
            
            case Assertion(_):
                match self.executor.run_test(program, test):
                    case Pass():
                        return True
                    case _:
                        return False
            
            case None:
                return False
            
            case _:
                return False