import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import List, Any, Tuple
from viberate.executor import Executor, Success, Pass, Fail
from viberate.input_generator import generate_inputs
from viberate.program import Test, Requirements
from viberate.code_generator import (
    Selector,
    SelectionOutcome,
    Selected,
    Generator,
    Abstained
)
from viberate.utils import RawData, ExperimentFailure


@dataclass
class UncertainOutput:
    pass


class Plurality(Selector):
    def __init__(self, executor: Executor, generator: Generator, num_programs: int):
        self.executor = executor
        self.generator = generator
        self.num_programs = num_programs

    def generate_and_select(self, model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        """Schema:
           {
              "programs": ...,
              "inputs": ...,
              "classes": <mapping from ids of valid classes to programs>,
              "outputs": <pairs of programs and corresponding output vectors>,
           }
        """
        inputs = generate_inputs(model, req, self.executor)
        programs = list(islice(self.generator.generate(model, req, self.num_programs), self.num_programs))

        classes = []
        outputs = []
        for p in programs:
            results = []
            for i in inputs:
                match self.executor.run(p, i):
                    case Success(v):
                        results.append(v)
                    case _:
                        results.append(UncertainOutput())
            if len(classes) == 0:
                classes.append(0)
            else:
                found = False
                for i, outs in enumerate(outputs):
                    if outs == results:
                        classes.append(classes[i])
                        found = True
                        break
                if not found:
                    classes.append(max(classes) + 1)
            outputs.append(results)
     
        valid_class_to_programs = {}
        for i in range(len(programs)):
            class_id = classes[i]
            program = programs[i]
            output = outputs[i]
            if not all(isinstance(o, UncertainOutput) for o in output):
                if class_id not in valid_class_to_programs:
                    valid_class_to_programs[class_id] = []
                valid_class_to_programs[class_id].append(program)

        raw_data = {
            "programs": programs,
            "inputs": inputs,
            "classes": valid_class_to_programs,
            "outputs": zip(programs, outputs),
        }

        if not valid_class_to_programs:
            raise ExperimentFailure()
        else:
            largest_class_id = max(valid_class_to_programs.items(), key=lambda x: len(x[1]))[0]
            selected_programs = []
            for i, c in enumerate(classes):
                if c == largest_class_id:
                    selected_programs.append(programs[i])
            return (Selected(selected_programs), raw_data)
