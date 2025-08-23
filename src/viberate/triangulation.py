import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable
from viberate.executor import Success, Executor

from viberate.cached_llm import Model
from viberate.code_generator import generate_specific_code
from viberate.logic import Formula, checker
from viberate.requirements import Requirements, NamedReturnSignature, _inverse_signature, \
    _inverse_description_single_arg, _inverse_description, _fiber_signature, _fiber_description_single_arg, \
    _fiber_description
from viberate.utils import print_annotated_hr


class Transformation(ABC):
    @abstractmethod
    def transform(self, model, req: Requirements) -> Requirements:
        pass

    @abstractmethod
    def get_name(self):
        pass


class Forward(Transformation):
    req: Requirements

    def transform(self, model, req: Requirements) -> Requirements:
        return req

    def get_name(self):
        return 'forward'


class PartialInverse(Transformation):
    inverse_index: int
    req: Requirements

    def __init__(self, model, req, inverse_index):
        self.inverse_index = inverse_index
        self.req = self.transform(model, req)

    def transform(self, model, req: Requirements) -> Requirements:
        print_annotated_hr("Signature")
        print(req.signature.pretty_print(), file=sys.stderr)
        # step 1: infer the name of return value
        named_sig = NamedReturnSignature.infer_name(model, req)
        # step 2: inverse the signature
        inverse_sig = _inverse_signature(model, named_sig, self.inverse_index)
        print_annotated_hr(f"Inverse signature wrt {self.inverse_index}")
        print(inverse_sig.pretty_print(), file=sys.stderr)
        # step 3: inverse the description
        if len(req.signature.params) == 1:
            inverse_desc = _inverse_description_single_arg(model, req, inverse_sig)
        else:
            inverse_desc = _inverse_description(model, req, inverse_sig, self.inverse_index)
        return Requirements(inverse_sig, inverse_desc)

    def get_name(self):
        return "partial inverse wrt " + str(self.inverse_index)


class PartialFiber(Transformation):
    inverse_index: int
    req: Requirements

    def __init__(self, model, req, inverse_index):
        self.inverse_index = inverse_index
        self.req = self.transform(model, req)

    def transform(self, model, req: Requirements) -> Requirements:
        print_annotated_hr("Signature")
        print(req.signature.pretty_print(), file=sys.stderr)
        # step 1: infer the name of return value
        named_sig = NamedReturnSignature.infer_name(model, req)
        # step 2: fiber the signature
        fiber_sig = _fiber_signature(model, named_sig, self.inverse_index)
        print_annotated_hr(f"Fiber signature wrt {self.inverse_index}")
        print(fiber_sig.pretty_print(), file=sys.stderr)
        # step 3: fiber the description
        if len(req.signature.params) == 1:
            fiber_desc = _fiber_description_single_arg(model, req, fiber_sig)
        else:
            fiber_desc = _fiber_description(model, req, fiber_sig, self.inverse_index)
        return Requirements(fiber_sig, fiber_desc)

    def get_name(self):
        return 'partial fiber wrt ' + str(self.inverse_index)


@dataclass
class Wrapper:
    executor: Executor

    @abstractmethod
    def function_wrapper(self, program_1, num_1, program_2, num_2) -> list[Callable]:
        return [partial(self.executor.run, program_1), partial(self.executor.run, program_2)]


@dataclass
class SpecificWrapper(Wrapper):
    executor: Executor
    model: Model
    transformation: Transformation
    n2: int

    def function_wrapper(self, program_1, num_1, program_2, num_2) -> list[Callable]:
        # a bug here to solve: the params for generate_specific_code
        return [partial(self.executor.run, program_1), partial(generate_specific_code, self.executor,
                                                               self.model, self.transformation, num_2, self.n2)]


@dataclass
class Property:
    formula: Formula
    wrapper: Wrapper


@dataclass
class Triangulation:
    trans_1: Transformation
    trans_2: Transformation
    property: Property

    def print_name(self):
        return self.trans_1.get_name() + "_" + self.trans_2.get_name()


def enumerate_pair(trans_to_programs: dict, t: Triangulation, arity):
    resonating_pairs = []
    selected_num = []
    for i, program_1 in enumerate(trans_to_programs[t.trans_1]):
        for j, program_2 in enumerate(trans_to_programs[t.trans_2]):
            print_annotated_hr(f"testing forward {i} and transformed {j}")
            if checker(t.property.formula, t.property.wrapper.function_wrapper(program_1, i, program_2, j), arity):
                resonating_pairs.append((i, j))
                selected_num.append(i)
    return selected_num, resonating_pairs
