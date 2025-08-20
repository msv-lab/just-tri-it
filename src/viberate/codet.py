from dataclasses import dataclass
from itertools import islice
from typing import Iterator, List, Tuple
from viberate.cached_llm import Model
from viberate.executor import Executor, Success
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
    
    def generate_and_select(self, model: Model, req: Requirements) -> SelectionOutcome:
        programs = list(islice(self.generator.generate(model, req), self.n))
        
        test_cases = list(islice(generate_test_cases(model, req, self.executor), self.m))
        
        if not programs or not test_cases:
            return Abstained()
        
        results: List[List[bool]] = [[False for _ in range(self.m)] for _ in range(self.n)]
        
        for i, program in enumerate(programs):
            for j, (test_input, expected_output) in enumerate(test_cases):
                try:
                    if len(test_input) != len(req.signature.params) and len(req.signature.params) == 1:
                        adjusted_input = [test_input]
                    else:
                        adjusted_input = test_input
                    
                    match self.executor.run(program, adjusted_input):
                        case Success(actual_output):
                            if actual_output == expected_output:
                                results[i][j] = True
                        case _:
                            pass
                except Exception as e:
                    pass
        
        best_score = -1
        selected_program = None
        
        for i in range(self.n):
            for j in range(self.m):
                if results[i][j]:
                    s_x_count = sum(1 for k in range(self.n) if results[k][j])
                    s_y_count = sum(1 for l in range(self.m) if results[i][l])
                    score = s_x_count * s_y_count
                    print(f"Program {i} passes test {j}: s_x={s_x_count}, s_y={s_y_count}, score={score}")
                    
                    if score > best_score:
                        best_score = score
                        selected_program = programs[i]
        
        if selected_program is not None:
            return Selected(selected_program)
        else:
            return Abstained()