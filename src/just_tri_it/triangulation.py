import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import Tuple
from functools import partial

from just_tri_it.executor import Executor
from just_tri_it.inversion import (
    ParameterInversion,
    ListSuffixInversion,
    list_split_signature,
    fwd_program_adapter,
    fwd_input_adapter
)
from just_tri_it.cached_llm import Model, Independent
from just_tri_it.code_generator import Generator
from just_tri_it.selection import Agreement, AgreementOutcome
from just_tri_it.input_generator import generate_inputs
from just_tri_it.logic import (
    Formula, Side, Var, Func, ForAll, Equals, SetEquals, OffByOne, And, Map, Member, TolerateInvalid, TimeoutGuard, FullOrPartial
)
from just_tri_it.program import Requirements, NamedReturnSignature, Signature, Parameter
from just_tri_it.utils import (
    gen_and_extract_answer_with_retry,
    ExperimentFailure,
    RawData, extract_answer
)
from just_tri_it.property_checker import Interpreter


class Triangulator(Agreement):
    def __init__(self,
                 executor: Executor,
                 code_generator: Generator,
                 triangulation: 'Triangulation',
                 num_left_programs: int,
                 num_right_programs: int,
                 gen_left_time_predicates: bool = False,
                 gen_right_time_predicates: bool = False):
        self.checker = Interpreter(executor)
        self.code_generator = code_generator
        self.triangulation = triangulation
        self.num_left_programs = num_left_programs
        self.num_right_programs = num_right_programs
        self.gen_left_time_predicates = gen_left_time_predicates
        self.gen_right_time_predicates = gen_right_time_predicates

    def compute_witnesses(self, model: Model, req: Requirements) -> Tuple[AgreementOutcome, RawData]:
        """Schema:
        {
            "method": "tri_<triangulation>",
            "left_programs": ...,
            "left_time_predicates": ...,
            "right_programs": ...,
            "right_time_predicates": ...
        }
        """
        def gen_time_predicate(model, req, program):
            return program.gen_time_predicate(model, req)
        
        transformed_left = self.triangulation.left_trans.transform(model, req)
        left_inputs = generate_inputs(model, transformed_left)
        left_programs = self.code_generator.generate(model, transformed_left, self.num_left_programs)
        left_programs = list(islice(left_programs, self.num_left_programs))
        if self.gen_left_time_predicates:
            left_programs = list(map(partial(gen_time_predicate, model, transformed_left), left_programs))

        transformed_right = self.triangulation.right_trans.transform(model, req)
        right_inputs = generate_inputs(model, transformed_right)        
        right_programs = self.code_generator.generate(model, transformed_right, self.num_right_programs)
        right_programs = list(islice(right_programs, self.num_right_programs))
        if self.gen_right_time_predicates:
            right_programs = list(map(partial(gen_time_predicate, model, transformed_right), right_programs))

        programs_and_witnesses = []

        for p in left_programs:
            p_witnesses = []
            for q in right_programs:
                if self.checker.check({Side.LEFT: left_inputs, Side.RIGHT: right_inputs},
                                      {Side.LEFT: p, Side.RIGHT: q },
                                      self.triangulation.hyperproperty):
                    p_witnesses.append(q)
            if len(p_witnesses) > 0:
                programs_and_witnesses.append((p, p_witnesses))

        raw_data = {
            "method": "tri_" + self.triangulation.name,
            "left_programs": left_programs,
            "left_time_predicates": list(map(lambda p: p.time_predicate, left_programs)),
            "right_programs": right_programs,
            "right_time_predicates": list(map(lambda p: p.time_predicate, right_programs))
        }

        return (programs_and_witnesses, raw_data)


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
3. The problem must be self-contained, and must not refer to the original function.

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

        def is_all_int_tuple(type_str: str) -> bool:
            type_str = type_str.strip()
            pattern = re.compile(
                r"^(?:tuple|Tuple)\[\s*int\s*(?:,\s*int\s*)*"
                r"(?:,\s*\.\.\.)?\]$"
            )
            return bool(pattern.match(type_str))

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
            case t if is_all_int_tuple(t):
                add_sentence = EXTRA_INSTR.format(sth="create a new tuple by adding 1 to it")           
            case _:
                # troublesome to support more complex types
                raise ExperimentFailure()
        return Requirements(req.signature, req.description + "\n" + add_sentence)


@dataclass
class PartialInverse(Transformation):
    inverse_index: int

    def __init__(self, inversion_scheme):
        self.inversion_scheme = inversion_scheme

    def _inverse_signature(self,
                           sig: NamedReturnSignature,
                           inversion_scheme) -> Signature:
        match inversion_scheme:
            case ParameterInversion(index):
                new_return_type = sig.params[index].type
                new_params = [Parameter(sig.return_name, sig.return_type)]
                new_params.extend(p for i, p in enumerate(sig.params) if i != index)
                new_func_name = "inverse_" + sig.name + "_wrt_" + sig.params[index].name
                new_sig = Signature(new_func_name, new_params, new_return_type)
                return new_sig
            case ListSuffixInversion(index, suffix_length):
                lss = list_split_signature(index, sig)
                return self._inverse_signature(lss, ParameterInversion(index+1))

    def _inverse_description(self,
                             model: Model,
                             description: str,
                             fwd_sig: Signature,
                             fwd_sig_note: str,
                             inv_sig: Signature,
                             inverse_index: int) -> str:
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{fwd_sig.pretty_print()}

{fwd_sig_note}

Rewrite this problem so that it instead requires implementing the inverted function:

{inv_sig.pretty_print()}

Given the desired output value `{inv_sig.params[0].name}` (corresponding to the original function's return value), the new function should return a value for the parameter `{fwd_sig.params[inverse_index].name}` such that if the original function were called with this value (and the other parameters unchanged), it would produce `{inv_sig.params[0].name}` as the result.

Important points to follow:
1. Preserve all original constraints, rules, and assumptions from the problem statement.
2. If multiple values of `{fwd_sig.params[inverse_index].name}` could produce the desired result, clearly state that any valid value is acceptable.
3. The problem must be self-contained, and must not refer to the original function.        
4. Update any examples so they demonstrate calling the inverted function instead of the original one.

Output the rewritten problem statement enclosed within `<answer>` and `</answer>` tags.

Original Problem:
{description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)

    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        inv_sig = self._inverse_signature(named_sig, self.inversion_scheme)
        match self.inversion_scheme:
            case ParameterInversion(index):
                fwd_sig = req.signature
                fwd_sig_note = ""
                inversion_index = index
            case ListSuffixInversion(index, suffix_length):
                fwd_sig = list_split_signature(index, req.signature)
                match suffix_length:
                    case 1:
                        fwd_sig_note = f"""\
where the input {req.signature.params[index].name} is split into {fwd_sig.params[index].name} + {fwd_sig.params[index+1].name}, so that {fwd_sig.params[index+1].name} has exactly one element if {req.signature.params[index].name} is not empty, and {fwd_sig.params[index+1].name} is empty if {req.signature.params[index].name} is empty.
                        """
                    case _:
                        fwd_sig_note = f"""\
where the input {req.signature.params[index].name} is split into {fwd_sig.params[index].name} + {fwd_sig.params[index+1].name}, so that len({fwd_sig.params[index+1].name}) is {suffix_length} when {req.signature.params[index].name} has at least {suffix_length} elements, otherwise {fwd_sig.params[index+1].name} contains the entire {req.signature.params[index].name}.
                        """
                inversion_index = index + 1
        inv_desc = self._inverse_description(model,
                                             req.description,
                                             fwd_sig,
                                             fwd_sig_note,
                                             inv_sig,
                                             inversion_index)
        return Requirements(inv_sig, inv_desc)


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
3. The problem must be self-contained, and must not refer to the original function.        
4. Keep any provided examples, but modify them so they demonstrate calling the postcondition function instead of the original implementation.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        post_desc = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        return Requirements(post_sig, post_desc)


@dataclass
class AnswerEnumeration(Transformation):

    def transform(self, model, req: Requirements) -> Requirements:
        sig = req.signature
        enum_sig = Signature(f"{sig.name}_answer_enum", sig.params, f"list[{sig.return_type}]")
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{sig.pretty_print()}

Your task is to rewrite this problem so that it instead requires implementing the function:

{enum_sig.pretty_print()}

The new function must exhaustively enumerate all valid outputs for a given input according to the original problem description. If only one output is valid, then the function must return a list consisting of this one element.

Important points to follow:
1. Preserve all the original problem's rules, edge cases, and constraints.
2. Update any example test cases so they show returning a exhaustive lists of answers.
3. The problem must be self-contained, and must not refer to the original function.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        enum_desc = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        return Requirements(enum_sig, enum_desc)


@dataclass
class PartialSetValuedInverse(Transformation):
    inverse_index: int

    def __init__(self, inversion_scheme):
        self.inversion_scheme = inversion_scheme

    def _sinv_signature(self,
                        sig: NamedReturnSignature,
                        inversion_scheme) -> Signature:
        match inversion_scheme:
            case ParameterInversion(index):
                new_return_type = "tuple[bool,list[" + sig.params[index].type + "]]"
                new_params = [Parameter(sig.return_name, sig.return_type)]
                new_params.extend(p for i, p in enumerate(sig.params) if i != index)
                new_func_name = "sinv_" + sig.name + "_wrt_" + sig.params[index].name
                new_sig = Signature(new_func_name, new_params, new_return_type)
                return new_sig
            case ListSuffixInversion(index, suffix_length):
                lss = list_split_signature(index, sig)
                return self._sinv_signature(lss, ParameterInversion(index+1))
        
        
    def _sinv_description(self,
                          model: Model,
                          description: str,
                          fwd_sig: Signature,
                          fwd_sig_note: str,
                          sinv_sig: Signature,
                          inverse_index: int):
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{fwd_sig.pretty_print()}

{fwd_sig_note}

Rewrite this problem so that it instead requires implementing the set-valued inverted function:

{sinv_sig.pretty_print()}

Given the desired output value `{sinv_sig.params[0].name}` (corresponding to the original function's return value), the new function should return a list of values for the parameter `{fwd_sig.params[inverse_index].name}` such that if the original function were called with any of these values (and the other parameters unchanged), it would produce `{sinv_sig.params[0].name}` as the result.

The function should return a tuple: (is_exhaustive_list, list_of_values). When it is feasible to enumerate all such values, return the complete list and set is_exhaustive_list to True. If a complete enumeration is impossible (e.g., the set is infinite or prohibitively large), return a representative subset and set is_exhaustive_list to False.

Important points to follow:
1. Preserve all constraints, domain assumptions, and rules from the original problem.
2. Clearly explain that the output must include all possible values.
3. Specify explicitly that if no such values exist, the function should return an empty list, and mark it as exhaustive.
3. The problem must be self-contained, and must not refer to the original function.
4. Update any example test cases so they show returning a full or a partial answer.

Enclose the rewritten problem description inside `<answer>` and `</answer>` tags.

Original Problem:
{description}
        """
        return gen_and_extract_answer_with_retry(model, PROMPT, 3)
        
    def transform(self, model, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(model, req)
        sinv_sig = self._sinv_signature(named_sig, self.inversion_scheme)
        match self.inversion_scheme:
            case ParameterInversion(index):
                fwd_sig = req.signature
                fwd_sig_note = ""
                inversion_index = index
            case ListSuffixInversion(index, suffix_length):
                fwd_sig = list_split_signature(index, req.signature)
                match suffix_length:
                    case 1:
                        fwd_sig_note = f"""\
where the input {req.signature.params[index].name} is split into {fwd_sig.params[index].name} + {fwd_sig.params[index+1].name}, so that {fwd_sig.params[index+1].name} has exactly one element if {req.signature.params[index].name} is not empty, and {fwd_sig.params[index+1].name} is empty if {req.signature.params[index].name} is empty.
                        """
                    case _:
                        fwd_sig_note = f"""\
where the input {req.signature.params[index].name} is split into {fwd_sig.params[index].name} + {fwd_sig.params[index+1].name}, so that len({fwd_sig.params[index+1].name}) is {suffix_length} when {req.signature.params[index].name} has at least {suffix_length} elements, otherwise {fwd_sig.params[index+1].name} contains the entire {req.signature.params[index].name}.
                        """
                inversion_index = index + 1
        sinv_desc = self._sinv_description(model,
                                             req.description,
                                             fwd_sig,
                                             fwd_sig_note,
                                             sinv_sig,
                                             inversion_index)
        return Requirements(sinv_sig, sinv_desc)


@dataclass
class Triangulation:
    name: str
    left_trans: Transformation
    right_trans: Transformation
    hyperproperty: Formula


def make_syntactic(arity):
    args = [Var(f"i_{i}") for i in range(arity)]
    p = Func(Side.LEFT)
    q = Func(Side.RIGHT)
    
    return Triangulation(
        "syntactic",
        Identity(),
        Identity(),
        ForAll(args, Side.LEFT, Equals([p(args), q(args)]))
    )


def make_trivial_semantic(arity):
    args = [Var(f"i_{i}") for i in range(arity)]
    p = Func(Side.LEFT)
    q = Func(Side.RIGHT)
    
    return Triangulation(
        "off-by-one",
        Identity(),
        TrivialSemantic(),
        ForAll(args, Side.LEFT, Equals([OffByOne([p(args)]), q(args)]))
    )


def make_postcondition(arity):
    args = [Var(f"i_{i}") for i in range(arity)]
    p = Func(Side.LEFT)
    q = Func(Side.RIGHT)
    
    return Triangulation(
        "post",
        Identity(),
        Postcondition(),
        ForAll(args, Side.LEFT, TolerateInvalid([q(args + [p(args)])]))
    )


def make_partial_fwd_inv(arity, inversion_scheme):
    match inversion_scheme:
        case ParameterInversion(index):
            inversion_index = index
        case ListSuffixInversion(index, suffix_length):
            inversion_index = index + 1
            arity += 1
    input_adapter = fwd_input_adapter(inversion_scheme)
    program_adapter = fwd_program_adapter(inversion_scheme)
            
    args = [Var(f"i_{i}") for i in range(arity)]
    inv_arg = Var(f"i_{inversion_index}")
    remaining_args = args[:inversion_index] + args[inversion_index + 1:]
    p = Func((Side.LEFT, program_adapter))
    q = Func(Side.RIGHT)

    return Triangulation(
        "fwd-inv",
        Identity(),
        PartialInverse(inversion_scheme),
        ForAll(args, (Side.LEFT, input_adapter),
               Equals([inv_arg, q([TolerateInvalid([p(args)])] + remaining_args)]))
    )


def make_partial_fwd_sinv(arity, inversion_scheme):
    match inversion_scheme:
        case ParameterInversion(index):
            inversion_index = index
        case ListSuffixInversion(index, suffix_length):
            inversion_index = index + 1
            arity += 1
    input_adapter = fwd_input_adapter(inversion_scheme)
    program_adapter = fwd_program_adapter(inversion_scheme)

    args = [Var(f"i_{i}") for i in range(arity)]
    inv_arg = Var(f"i_{inversion_index}")
    arg_prime = Var(f"i_prime")
    remaining_args = args[:inversion_index] + args[inversion_index + 1:]
    args_with_prime = args[:inversion_index] + [arg_prime] + args[inversion_index+1:]
    p = Func((Side.LEFT, program_adapter))
    q = Func(Side.RIGHT)

    return Triangulation(
        "fwd-sinv",
        Identity(),
        PartialSetValuedInverse(inversion_scheme),
        ForAll(args, (Side.LEFT, input_adapter),
               And(Member([inv_arg, FullOrPartial([q([TolerateInvalid([p(args)])] + remaining_args)])]),
                   ForAll(arg_prime, FullOrPartial([q([TolerateInvalid([p(args)])] + remaining_args)]),
                          Equals([p(args), TimeoutGuard(p)(args_with_prime)])))))


def make_partial_enum_sinv(arity, inversion_scheme):
    assert isinstance(inversion_scheme, ParameterInversion)   
    left_args = [Var(f"i_{i}") for i in range(arity)]
    inv_arg = Var(f"i_{inversion_scheme.index}")
    out = Var(f"o")
    right_args = [out] + left_args[:inversion_scheme.index] + left_args[inversion_scheme.index + 1:]
    
    p = Func(Side.LEFT)
    q = Func(Side.RIGHT)

    return Triangulation(
        "enum-sinv",
        AnswerEnumeration(),
        PartialSetValuedInverse(inversion_scheme.index),
        And(ForAll(left_args, Side.LEFT,
                   ForAll(out, TolerateInvalid([p(left_args)]),
                          Member([inv_arg, TimeoutGuard(q)(right_args)]))),
            ForAll(right_args, Side.RIGHT,
                   ForAll(inv_arg, TolerateInvalid([q(right_args)]),
                          Member([out, TimeoutGuard(p)(left_args)])))))
