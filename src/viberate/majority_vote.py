from dataclasses import dataclass
from itertools import islice

from viberate.llm import LLM
from viberate.executor import Executor, Success
from viberate.requirements import Requirements
from viberate.input_generator import generate_inputs
from viberate.code_generator import (
    Selector,
    SelectionOutcome,
    Selected,
    Generator
)


@dataclass
class UncertainOutput:
    pass


class MajorityVote(Selector):

    def __init__(self, executor: Executor, generator: Generator, n: int):
        self.executor = executor
        self.generator = generator
        self.n = n
        
    def generate_and_select(self, model, req: Requirements) -> SelectionOutcome:
        inputs = generate_inputs(model, req)
        programs = islice(self.generator.generate(model, req), self.n)
        classes = []
        outputs = []
        generated = []
        for p in programs:
            results = []
            generated.append(p)
            for i in inputs:
                match self.executor.run(p, i):
                    case Success(v):
                        results.append(v)
                    case _:
                        results.append(UncertainOutput())
            if len(classes) == 0:
                classes.append(0)
            else:
                for i, outs in enumerate(outputs):
                    if outs == results:
                        classes.append(classes[i])
                        break
                if len(classes) < len(generated):
                    # need to create a new equivalence class
                    classes.append(max(classes) + 1)
            outputs.append(results)

        largest_class = max(set(classes), key=classes.count)
        return Selected(generated[classes.index(largest_class)])
