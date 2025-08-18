import sys
from dataclasses import dataclass
from itertools import islice
from typing import Iterable
from abc import ABC, abstractmethod

from viberate.llm import extract_code
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


specific_dict = {}
series_dict = {}


def generate_specific_code(executor, model, req, choice, num, n2, inputs):
    if specific_dict.get(choice + str(inputs) + str(num)) is None:
        specific_req = specific_requirements(model, req, inputs, choice)
        if series_dict.get(choice + str(inputs)) is None:
            series_dict[choice + str(inputs)] = islice(generate_no_input(model, specific_req), n2)
        specific_code = next(series_dict[choice + str(inputs)])
        print(specific_code, file=sys.stderr)
        outcome = executor.run(specific_code, [])
        specific_dict[choice + str(inputs) + str(num)] = outcome
    else:
        outcome = specific_dict[choice + str(inputs) + str(num)]
    return outcome


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
    print(PROMPT)
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
        it correctly.

        Problem:
        {req.description}
        """
        for s in model.sample(PROMPT):
            yield Program(req.signature, extract_code(s))
