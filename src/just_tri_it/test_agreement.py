from itertools import islice
from typing import Tuple

from just_tri_it.cached_llm import Model
from just_tri_it.executor import Executor, Pass
from just_tri_it.program import Requirements
from just_tri_it.test_generator import TestGenerator
from just_tri_it.selection import Agreement, AgreementOutcome
from just_tri_it.code_generator import Generator
from just_tri_it.utils import RawData, ONLY_CACHE


class TestAgreement(Agreement):
    def __init__(self,
                 executor: Executor,
                 code_generator: Generator,
                 test_generator: TestGenerator,
                 num_programs: int):
        self.executor = executor
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.num_programs = num_programs

    def compute_witnesses(self, model: Model, req: Requirements) -> Tuple[AgreementOutcome, RawData]:
        """Schema:
        {
          "method": "test_<generator>"  # where generator is "assert" or "IO"
          "programs": ...,
          "tests": ...
        }
        """

        programs = self.code_generator.generate(model, req, self.num_programs)
        programs = list(islice(programs, self.num_programs))
        tests = list(self.test_generator.generate(model, req))

        raw_data = {
            "method": "test_" + self.test_generator.display_name(),
            "programs": programs,
            "tests": tests
        }

        agreement = []

        if not ONLY_CACHE:
            for program in programs:
                agreed_tests = []
                for test in tests:
                    match self.executor.run_test(program, test):
                        case Pass():
                            agreed_tests.append(test)
                        case _:
                            pass
                if len(agreed_tests) > 0:
                    agreement.append((program, agreed_tests))

        return agreement, raw_data
