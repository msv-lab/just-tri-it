from dataclasses import dataclass
from typing import Iterable
from abc import ABC, abstractmethod

from viberate.llm import extract_code
from viberate.program import Signature, Program
from viberate.requirements import Requirements



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
