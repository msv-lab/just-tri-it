from viberate.llm import LLM
from viberate.code_generator import (
    Selector,
    SelectionOutcome,
    Selected,
    Abstained
)


class CodeT(Selector):

    def __init__(self, executor):
        self.executor = executor
        
    def generate_and_select(self, model, req: Requirements) -> SelectionOutcome:
        # TODO: implement CodeT algorithm
        pass
