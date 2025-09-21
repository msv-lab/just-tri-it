import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Callable, Tuple

from just_tri_it.executor import Success, Executor
from just_tri_it.cached_llm import Model
from just_tri_it.code_generator import Selector, Generator, SelectionOutcome, Selected, Abstained
from just_tri_it.input_generator import generate_inputs
from just_tri_it.logic import Formula, check, Side, Var, Func, ForAll, Equals, SetEquals, OffByOne, And, Map, MapUnpack, Member
from just_tri_it.program import Requirements, NamedReturnSignature, Signature, Parameter
from just_tri_it.utils import (
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
        PROMPT = f"""
You are given a programming problem written in English that requires implementing the function:

{req.signature.pretty_print()}

Your task is to translate the entire problem description into Chinese, while preserving all the technical accuracy and meaning. Whenever the description first mentions an input parameter, include its original English name from the function signature in parentheses immediately after the Chinese text describing that parameter. 

Translation guidelines:
1. Keep technical terms such as data types, built-in function names, variable names, and programming language keywords in English.
2. Preserve all constraints, examples, and formatting from the original text.

Output the translated problem enclosed in `<answer>` and `</answer>` tags.

Original Problem:
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

    def _inverse_description(self,
                             model: Model,
                             req: Requirements,
                             inverted_sig: Signature,
                             inverse_index: int) -> str:
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

Rewrite this problem so that it instead requires implementing the inverted function:

{inverted_sig.pretty_print()}

Given the desired output value `{inverted_sig.params[0].name}` (corresponding to the original function's return value), the new function should return a value for the parameter `{req.signature.params[inverse_index].name}` such that if the original function were called with this value (and the other parameters unchanged), it would produce `{inverted_sig.params[0].name}` as the result.

Important points to follow:
1. Preserve all original constraints, rules, and assumptions from the problem statement.
2. If multiple values of `{req.signature.params[inverse_index].name}` could produce the desired result, clearly state that any valid value is acceptable.
4. Update any examples so they demonstrate calling the inverted function instead of the original one.

Output the rewritten problem statement enclosed within `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)

    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        inverse_sig = self._inverse_signature(model, named_sig, self.inverse_index)
        inverse_desc = self._inverse_description(model, req, inverse_sig, self.inverse_index)
        return Requirements(inverse_sig, inverse_desc)


@dataclass
class Postcondition(Transformation):

    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        last_param = Parameter(named_sig.return_name, named_sig.return_type)
        post_sig = Signature('postcondition', named_sig.params + [last_param], 'bool')
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

Your task is to rewrite this problem so that it instead requires implementing the function:

{post_sig.pretty_print()}

The new function should verify whether the given output value (`{last_param.name}`) is correct for the specified inputs, according to the original problem description.

Important points to follow:
1. Preserve all the original problem's rules, edge cases, and constraints.
2. If the original problem allows multiple correct solutions, clarify that the new function must return True for any valid output that meets the problem criteria.
3. Keep any provided examples, but modify them so they demonstrate calling the postcondition function instead of the original implementation.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        post_desc = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        return Requirements(post_sig, post_desc)


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
        
    def _fiber_description(self,
                           model: Model,
                           req: Requirements,
                           fiber_sig: Signature,
                           inverse_index: int):
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

Rewrite this problem so that it instead requires implementing the set-valued inverted function:

{fiber_sig.pretty_print()}

Given the desired output value `{inverted_sig.params[0].name}` (corresponding to the original function's return value), the new function should return a comprehensive list of values for the parameter `{req.signature.params[inverse_index].name}` such that if the original function were called with any of these values (and the other parameters unchanged), it would produce `{inverted_sig.params[0].name}` as the result.

Important points to follow:
1. Preserve all constraints, domain assumptions, and rules from the original problem.
2. Clearly explain that the output must include all possible values.
3. Specify explicitly that if no such values exist, the function should return an empty list.
4. Update any example test cases so they show returning a comprehensive lists of solutions.

Enclose the rewritten problem description inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)
        
    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        fiber_sig = self._fiber_signature(model, named_sig, self.inverse_index)
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


def make_postcondition(arity):
    args = [Var(f"x_{i}") for i in range(arity)]
    f = Func(Side.LEFT)
    g = Func(Side.RIGHT)
    
    return Triangulation(
        Identity(),
        Postcondition(),
        ForAll(args, Side.LEFT, g(args + [f(args)]))
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
