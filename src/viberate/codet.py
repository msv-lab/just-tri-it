from dataclasses import dataclass
from itertools import islice
from typing import Iterator, List, Tuple

from viberate.cached_llm import Model
from viberate.executor import Executor, Success
from viberate.requirements import Requirements
from viberate.input_generator import generate_inputs
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
        inputs = list(islice(generate_inputs(model, req), self.m))
        
        if not programs or not inputs:
            return Abstained()

        results: List[List[bool]] = [[False for _ in range(self.m)] for _ in range(self.n)]
        for i, program in enumerate(programs):
            for j, input_data in enumerate(inputs):
                match self.executor.run(program, input_data):
                    case Success(v):
                        results[i][j] = True
                    case _:
                        pass

        best_score = -1
        selected_program = None
        
        for i in range(self.n):
            for j in range(self.m):
                if results[i][j]:
                    s_x_count = sum(1 for k in range(self.n) if results[k][j])
                    s_y_count = sum(1 for l in range(self.m) if results[i][l])
                    score = s_x_count * s_y_count
                    
                    if score > best_score:
                        best_score = score
                        selected_program = programs[i]
        
        if selected_program is not None:
            return Selected(selected_program)
        else:
            return Abstained()

