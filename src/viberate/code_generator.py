import hashlib
import sys
import threading
from dataclasses import dataclass
from itertools import islice
from typing import Iterable
from abc import ABC, abstractmethod

from viberate.executor import Error
from viberate.utils import extract_code
from viberate.program import Signature, Program
from viberate.requirements import Requirements, specific_requirements


class Generator(ABC):
    @abstractmethod
    def generate(self, model, req: Requirements) -> Iterable['Program']:
        pass


@dataclass
class Selected:
    program: Program


@dataclass
class Abstained:
    pass


type SelectionOutcome = Selected | Abstained


class Selector(ABC):

    @abstractmethod
    def generate_and_select(self, model, req: Requirements) -> SelectionOutcome:
        pass


def generate_no_input(model, req: Requirements) -> Iterable[Program]:
    PROMPT = f"""Write a Python function
    '{req.signature.pretty_print()}' without any input to solve the following problem.
    
    Include all necessary imports. Put the complete code inside a
    Markdown code block. Please generate the program by implementing
    only the function, without using if __name__ == "__main__": or any
    code outside the function. Please ensure the generated code
    returns pure Python types. If there are requirements for the
    output format in the problem description, please be sure to format
    it correctly.

    Problem:
    {req.description}
    """
    for s in model.sample(PROMPT):
        yield Program(req.signature, extract_code(s))


def partitioning_generate(model, req: Requirements) -> Iterable[Program]:
    PROMPT = f""" Write a Python function
    {req.signature.pretty_print()} to solve the following problem.
    If the problem cannot be solved correctly with a single approach,
    consider using different algorithms for different input ranges.
    Clearly define the conditions for switching between algorithms,
    so that the solution is guaranteed correct within each applicable
    range, even if not optimal for all inputs.
    
    Include all necessary imports. Put the complete code inside a
    Markdown code block. Please generate the program by implementing
    only the function, without using if __name__ == "__main__": or any
    code outside the function. Please ensure the generated code
    returns pure Python types. If there are requirements for the
    output format in the problem description, please be sure to format
    it correctly.

    Problem:
    {req.description}
    """
    for s in model.sample(PROMPT):
        yield Program(req.signature, extract_code(s))


class Vanilla(Generator):

    def generate(self, model, req: Requirements) -> Iterable[Program]:
        PROMPT = f""" Write a Python function
        {req.signature.pretty_print()} to solve the following problem.
        Include all necessary imports. Put the complete code inside a
        Markdown code block. Please generate the program by implementing
        only the function, without using if __name__ == "__main__": or any
        code outside the function. Please ensure the generated code
        returns pure Python types. If there are requirements for the
        output format in the problem description, please be sure to format
        it correctly. When handling invalid inputs, please include logic to
        raise a ValueError with the message 'Invalid input'.
         
        Problem:
        {req.description}
        """
        for s in model.sample(PROMPT):
            yield Program(req.signature, extract_code(s))


@dataclass
class SpecificGenerator:
    specific_dict = {}
    series_dict = {}
    lock = threading.Lock()

    @staticmethod
    def _generate_key(*args):
        hash_obj = hashlib.sha256()
        for arg in args:
            hash_obj.update(repr(arg).encode('utf-8'))
        return hash_obj.hexdigest()

    def generate_specific_code_and_run(self, executor, model, t, num, n2, inputs):
        base_key = self._generate_key(t.get_name(), inputs)
        specific_key = self._generate_key(t.get_name(), inputs, num)

        with self.lock:
            if specific_key in self.specific_dict:
                # print("hit")
                return self.specific_dict[specific_key]

            if base_key not in self.series_dict:
                # print("miss and generate")
                specific_req = specific_requirements(model, t.req, inputs, t.get_name())
                self.series_dict[base_key] = islice(generate_no_input(model, specific_req), n2)

            code_iterator = self.series_dict[base_key]

            specific_code = next(code_iterator)
            outcome = executor.run(specific_code, [])
            self.specific_dict[specific_key] = outcome
            return outcome
