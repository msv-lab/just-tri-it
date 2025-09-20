import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Callable, Tuple

from viberate.executor import Success, Executor
from viberate.cached_llm import Model
from viberate.code_generator import Selector, Generator, SelectionOutcome, Selected, Abstained
from viberate.input_generator import generate_inputs
from viberate.logic import Formula, check, Side, Var, Func, ForAll, Equals, SetEquals, OffByOne, And, Map, MapUnpack, Member
from viberate.program import Requirements, NamedReturnSignature, Signature, Parameter
from viberate.utils import (
    print_annotated_hr,
    gen_and_extract_answer_with_retry,
    ExperimentFailure,
    RawData
)


class TriSelector(Selector):
    def __init__(self,
                 executor: Executor,
                 code_generator: Generator,
                 triangulation: 'Triangulation',
                 num_left_programs: int,
                 num_right_programs: int):
        self.executor = executor
        self.code_generator = code_generator
        self.triangulation = triangulation
        self.num_left_programs = num_left_programs
        self.num_right_programs = num_right_programs

    def generate_and_select(self, model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        transformed_left = self.triangulation.left_trans.transform(model, req)
        left_inputs = generate_inputs(model, transformed_left, self.executor)
        left_programs = self.code_generator.generate(model, transformed_left, self.num_left_programs)
        left_programs = islice(left_programs, self.num_left_programs)

        transformed_right = self.triangulation.right_trans.transform(model, req)
        right_inputs = generate_inputs(model, transformed_right, self.executor)        
        right_programs = self.code_generator.generate(model, transformed_right, self.num_right_programs)
        right_programs = islice(right_programs, self.num_right_programs)

        selected_pairs = []

        for p in left_programs:
            for q in right_programs:
                if check(self.executor,
                         { Side.LEFT: left_inputs, Side.RIGHT: right_inputs },
                         { Side.LEFT: p, Side.RIGHT: q },
                         self.triangulation.hyperproperty):
                    selected_pairs.append((p, q))

        raw_data = {}

        if len(selected_pairs) > 0:
            #NOTE: assume that the left transformation is what we want to select:
            return (Selected(list(map(lambda x: x[0], selected_pairs))), raw_data)
        else:
            return (Abstained(), raw_data)


class Transformation(ABC):

    @abstractmethod
    def transform(self, model, req: Requirements) -> Requirements:
        pass


class Identity(Transformation):

    def transform(self, model, req: Requirements) -> Requirements:
        return req


class Syntactic(Transformation):

    def transform(self, model, req: Requirements) -> Requirements:
        PROMPT = f"""Translate a given problem, which requires
implementing the function '{req.signature.pretty_print()}', into
Chinese. When mentioning each input parameter, include its original
English name from the function signature in parentheses immediately
after the Chinese description of that parameter. Keep technical terms
(e.g., programming keywords, variable names) in English where
necessary for clarity. Enclose your translated problem in <answer>
tags.

Problem description:
{req.description}
        """
        translated_description = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        return Requirements(req.signature, translated_description)


class TrivialSemantic(Transformation):

    def transform(self, model, req: Requirements) -> Requirements:
        return_type = req.signature.return_type
        EXTRA_INSTR = "When you get the final answer, please {sth} and then return."
        match return_type:
            case "int":
                add_sentence = EXTRA_INSTR.format(sth="add 1 to it")
            case "float":
                add_sentence = EXTRA_INSTR.format(sth="add 1.0 to it")
            case "bool":
                add_sentence = EXTRA_INSTR.format(sth="negate it")
            case "str":
                add_sentence = EXTRA_INSTR.format(sth="add a suffix '_1' to it")
            case "list[int]" | "List[int]":
                add_sentence = EXTRA_INSTR.format(sth="append 1 to it")
            case _:
                # troublesome to support more complex types
                raise ExperimentFailure()
        return Requirements(req.signature, req.description + "\n" + add_sentence)


def choose_parameter_to_invert(model: Model, req: Requirements) -> int:
    if len(req.signature.params) == 1:
        return 0
    else:
        PROMPT = f"""The problem below is solved using a function with
the signature {req.signature.pretty_print()}. Choose function one
parameter to swap with the return value to form an inverse
problem. Inversing which parameter do you think would make the inverse problem
natural? Do not choose parameters that can be easily derived from other
parameters. Answer only the full name of this parameter (not its type)
in <answer> tags.

Problem:
{req.description}
        """
        valid_name = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        return_param = [p.name for p in req.signature.params].index(valid_name)
        return return_param


@dataclass
class PartialInverse(Transformation):
    inverse_index: int

    def __init__(self, inverse_index):
        self.inverse_index = inverse_index

    def _inverse_signature(self,
                           model: Model,
                           sig: NamedReturnSignature,
                           inverse_index: int) -> Signature:
        new_return_type = sig.params[inverse_index].type
        new_params = [Parameter(sig.return_name, sig.return_type)]
        new_params.extend(p for i, p in enumerate(sig.params) if i != inverse_index)
        new_func_name = "inverse_" + sig.name + "_wrt_" + sig.params[inverse_index].name
        new_sig = Signature(new_func_name, new_params, new_return_type)
        return new_sig

    def _inverse_description_single_arg(self,
                                        model: Model,
                                        req: Requirements,
                                        inverted_sig: Signature) -> str:
        PROMPT = f"""Rewrite the given problem, which requires
implementing the function '{req.signature.pretty_print()}', so that it
instead requires implementing the function
'{inverted_sig.pretty_print()}'. The new function should, for each
possible output of the original function, return the possible input
that produce that output. Try to maintain accuracy during the
conversion process. Enclose your rewritten problem in <answer> tags.

Problem:
{req.description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)

    def _inverse_description(self,
                             model: Model,
                             req: Requirements,
                             inverted_sig: Signature,
                             inverse_index: int) -> str:
        PROMPT = f"""Rewrite the given problem, which requires
implementing the function '{req.signature.pretty_print()}', so that it
requires implementing the function '{inverted_sig.pretty_print()}'
instead. The new function should return the value of the parameter
'{req.signature.params[inverse_index].name}' that produce that given
output. Try to maintain accuracy during the conversion
process. Enclose your rewritten problem in <answer> tags.

Problem:
{req.description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)

    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        inverse_sig = self._inverse_signature(model, named_sig, self.inverse_index)
        if len(req.signature.params) == 1:
            inverse_desc = self._inverse_description_single_arg(model, req, inverse_sig)
        else:
            inverse_desc = self._inverse_description(model, req, inverse_sig, self.inverse_index)
        return Requirements(inverse_sig, inverse_desc)


@dataclass
class PartialFiber(Transformation):
    inverse_index: int

    def __init__(self, inverse_index):
        self.inverse_index = inverse_index

    def _fiber_signature(self,
                         model: Model,
                         sig: NamedReturnSignature,
                         inverse_index: int) -> Signature:
        new_return_type = "list[" + sig.params[inverse_index].type + "]"
        new_params = [Parameter(sig.return_name, sig.return_type)]
        new_params.extend(p for i, p in enumerate(sig.params) if i != inverse_index)
        new_func_name = "fiber_" + sig.name + "_wrt_" + sig.params[inverse_index].name
        new_sig = Signature(new_func_name, new_params, new_return_type)
        return new_sig
        
    def _fiber_description_single_arg(self,
                                      model: Model,
                                      req: Requirements,
                                      fiber_sig: Signature):
        PROMPT = f"""Rewrite the given problem, which requires
implementing the function '{req.signature.pretty_print()}', so that it
instead requires implementing the function
'{fiber_sig.pretty_print()}'. The new function should, for each
possible output of the original function, return the exhaustive list
of all inputs that produce that output. Try to maintain accuracy
during the conversion process. Enclose your rewritten problem in
<answer> tags.

Problem:
{req.description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)

    def _fiber_description(self,
                           model: Model,
                           req: Requirements,
                           fiber_sig: Signature,
                           inverse_index: int):
        PROMPT = f"""Rewrite the given problem, which requires
implementing the function '{req.signature.pretty_print()}', so that it
requires implementing the function '{fiber_sig.pretty_print()}'
instead. The new function should return the exhaustive list of all
possible values of the parameter
'{req.signature.params[inverse_index].name}' that produce the given
output. Try to maintain accuracy during the conversion process.
Enclose your rewritten problem in <answer> tags.

Problem:
{req.description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)
        
    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        fiber_sig = self._fiber_signature(model, named_sig, self.inverse_index)
        if len(req.signature.params) == 1:
            fiber_desc = self._fiber_description_single_arg(model, req, fiber_sig)
        else:
            fiber_desc = self._fiber_description(model, req, fiber_sig, self.inverse_index)
        return Requirements(fiber_sig, fiber_desc)


@dataclass
class Triangulation:
    left_trans: Transformation
    right_trans: Transformation
    hyperproperty: Formula


def make_syntactic(arity):
    args = [Var(f"x_{i}") for i in range(arity)]
    f = Func(Side.LEFT)
    g = Func(Side.RIGHT)
    
    return Triangulation(
        Identity(),
        Identity(),
        ForAll(args, Side.LEFT, Equals([f(args), g(args)]))
    )


def make_trivial_semantic(arity):
    args = [Var(f"x_{i}") for i in range(arity)]
    f = Func(Side.LEFT)
    g = Func(Side.RIGHT)
    
    return Triangulation(
        Identity(),
        TrivialSemantic(),
        ForAll(args, Side.LEFT, Equals([OffByOne([f(args)]), g(args)]))
    )


def make_partial_for_inv(arity, inverse_index):
    args = [Var(f"x_{i}") for i in range(arity)]
    inv_arg = Var(f"x_{inverse_index}")
    remaining_args = args[:inverse_index] + args[inverse_index + 1:]
    f = Func(Side.LEFT)
    g = Func(Side.RIGHT)
    
    return Triangulation(
        Identity(),
        PartialInverse(inverse_index),
        ForAll(args, Side.LEFT, Equals([inv_arg, g([f(args)] + remaining_args)]))
    )


def make_partial_for_fib(arity, inverse_index):
    args = [Var(f"x_{i}") for i in range(arity)]
    inv_arg = Var(f"x_{inverse_index}")
    remaining_args = args[:inverse_index] + args[inverse_index + 1:]
    f = Func(Side.LEFT)
    g = Func(Side.RIGHT)

    ReplaceAt = Func(lambda v: args[:inverse_index] + [v] + args[inverse_index+1:], "replace_at")
    Wrap = Func(lambda v: [v], "wrap")
    def replace_at(l, i, v):
        return 
    
    return Triangulation(
        Identity(),
        PartialFiber(inverse_index),
        ForAll(args, Side.LEFT,
               And(Member([inv_arg, g([f(args)] + remaining_args)]),
                   SetEquals([MapUnpack(f,
                                        Map(ReplaceAt, g([f(args)] + remaining_args))),
                              [f(args)]])))
    )
