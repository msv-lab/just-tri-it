from dataclasses import dataclass
from itertools import islice
from typing import Tuple

from just_tri_it.cached_llm import Model
from just_tri_it.executor import Executor, Success
from just_tri_it.input_generator import generate_inputs
from just_tri_it.program import Requirements
from just_tri_it.code_generator import Generator
from just_tri_it.selection import Agreement, AgreementOutcome
from just_tri_it.utils import RawData, ExperimentFailure


@dataclass
class UncertainOutput:
    pass


class Plurality(Agreement):

    def __init__(self, executor: Executor, generator: Generator, num_programs: int, prob_threshold=0.0):
        self.executor = executor
        self.generator = generator
        self.num_programs = num_programs
        self.prob_threshold = prob_threshold

    def compute_witnesses(self, model: Model, req: Requirements) -> Tuple[AgreementOutcome, RawData]:
        """Schema:
           {
              "method": "plurality_<prob_threshold>",
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

        def _to_serializable(outputs):
            result = []
            for o in outputs:
                if isinstance(o, UncertainOutput):
                    result.append("__UncertainOutput")
                else:
                    result.append(o)
            return result

        raw_data = {
            "method": "plurality_" + str(self.prob_threshold),            
            "programs": programs,
            "inputs": inputs,
            "classes": valid_class_to_programs,
            "outputs": [(programs[i], _to_serializable(outputs[i])) for i in range(len(programs))]
        }

        if not valid_class_to_programs:
            raise ExperimentFailure()

        total_valid_samples = sum(len(v) for v in valid_class_to_programs.values())

        programs_and_witnesses = []

        for _, programs in valid_class_to_programs.items():
            if len(programs) < 2:
                continue
            with_witnesses = [(p, [q for q in programs if p.hash_id() != q.hash_id()]) for p in programs]
            if len(programs) / total_valid_samples > self.prob_threshold:
                programs_and_witnesses.extend(with_witnesses)

        return (programs_and_witnesses, raw_data)
