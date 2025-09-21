import os
import hashlib
import sys
import threading
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Tuple, List
from abc import ABC, abstractmethod

from just_tri_it.executor import Error
from just_tri_it.utils import extract_code, print_annotated_hr, RawData
from just_tri_it.program import Signature, Program, Requirements
from just_tri_it.executor import Executor


class Generator(ABC):
    @abstractmethod
    def generate(self, model, req: Requirements, batch=1) -> Iterable['Program']:
        pass


@dataclass
class Selected:
    programs: List[Program]


@dataclass
class Abstained:
    pass


type SelectionOutcome = Selected | Abstained


class Selector(ABC):

    @abstractmethod
    def generate_and_select(self, model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        pass


@dataclass
class Vanilla(Generator):

    def generate(self, model, req: Requirements, batch=1) -> Iterable[Program]:
        PROMPT = f""" Write a Python function '{req.signature.pretty_print()}'
to solve the following problem. Include all necessary imports. Put the complete
code inside a Markdown code block:
```python
```

Please generate the program by implementing only the function, without
using if __name__ == "__main__": or any code outside the function. Do
not print anything and just return a value of the type specified in
the function signature. When handling invalid inputs that are not
explicitly stated how to deal with in the problem description, please
raise a ValueError with the message 'Invalid input'.
         
Problem:
{req.description}
        """
        for s in model.sample(PROMPT, batch):
            yield Program(req.signature, extract_code(s))
