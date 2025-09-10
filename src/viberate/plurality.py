import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import List, Any
from viberate.executor import Executor, Success, Pass, Fail
from viberate.requirements import Requirements
from viberate.input_generator import generate_inputs
from viberate.program import Test, Assertion, ExpectedOutput
from viberate.code_generator import (
    Selector,
    SelectionOutcome,
    Selected,
    Generator, Abstained
)

@dataclass
class UncertainOutput:
    pass

@dataclass
class TestFailed:
    pass

@dataclass
class TestPassed:
    pass


class Plurality(Selector):
    def __init__(self, executor: Executor, generator: Generator, n: int):
        self.executor = executor
        self.generator = generator
        self.n = n
        
    def generate_and_select(self, model, req: Requirements):
        inputs = generate_inputs(model, req, self.executor)
        programs = list(islice(self.generator.generate(model, req), self.n))
        classes = []
        outputs = []
        generated = []
        for p_id, p in enumerate(programs):
            results = []
            generated.append(p_id)
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
                    classes.append(max(classes) + 1)
            outputs.append(results)
        
        class_to_pid = {}
        for class_id, p_id in zip(classes, generated):
            if class_id not in class_to_pid:
                class_to_pid[class_id] = []
            class_to_pid[class_id].append(p_id)

        class_to_outputs = {}
        for class_id, output in zip(classes, outputs):
            if class_id not in class_to_outputs:
                class_to_outputs[class_id] = []
            class_to_outputs[class_id].append(output)

        valid_classes = {}
        for class_id, outputs_list in class_to_outputs.items():
            all_uncertain = True
            for output in outputs_list:
                if not all(isinstance(item, UncertainOutput) for item in output):
                    all_uncertain = False
                    break
            if not all_uncertain:
                valid_classes[class_id] = class_to_pid[class_id]
        
        if not valid_classes:
            return [], programs, Abstained(), {}
        
        largest_class_id = max(valid_classes.items(), key=lambda x: len(x[1]))[0]
        return (class_to_pid[largest_class_id], programs, Selected(programs[valid_classes[largest_class_id][0]]),
                class_to_pid)
