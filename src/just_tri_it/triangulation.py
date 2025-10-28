import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import Tuple
from functools import partial
from enum import Enum
import copy
import sys
import ast

from just_tri_it.executor import Executor
from just_tri_it.cached_llm import Model, Independent
from just_tri_it.code_generator import Generator
from just_tri_it.selection import Agreement, AgreementOutcome
from just_tri_it.input_generator import generate_inputs, remove_duplicates
from just_tri_it.logic import (
    Formula, Side, Var, Func, ForAll, Equals, SetEquals, OffByOne, And, Map, Member, TolerateInvalid, TimeoutGuard, FullOrPartial, FlattenMap
)
from just_tri_it.program import Program, Requirements, NamedReturnSignature, Signature, Parameter
from just_tri_it.utils import (
    gen_and_extract_answer_with_retry,
    gen_and_extract_code_with_retry,
    ExperimentFailure,
    RawData, extract_answer
)
from just_tri_it.property_checker import Interpreter
import just_tri_it.utils


@dataclass
class ParameterInversion:
    index: int


@dataclass
class ListSuffixInversion:
    index: str
    suffix_length: int


type InversionScheme = ParameterInversion | ListSuffixInversion


class TriangulationMode(Enum):
    FWD_INV = "fwd-inv"
    FWD_SINV = "fwd-sinv"
    ENUM_SINV = "enum-sinv"
    Postcondition = "postcondition"
    OffByOne = "off-by-one"
    Syntactic = "syntactic"


class Triangulator:

    def __init__(self, executor, code_generator, triangulation_mode, num_left_samples, num_right_samples):
        self.checker = Interpreter(executor)
        self.executor = executor
        self.code_generator = code_generator
        self.triangulation_mode = triangulation_mode
        self.num_left_samples = num_left_samples
        self.num_right_samples = num_right_samples

    def compute_witnesses(self, model: Model, fwd_problem: Requirements) -> Tuple[AgreementOutcome, RawData]:
        """Schema:
        {
            "method": "<triangulation mode>"
        }
        """
        self.model = model

        stream_processing = False
        if self.triangulation_mode in [TriangulationMode.FWD_INV,
                                       TriangulationMode.FWD_SINV,
                                       TriangulationMode.ENUM_SINV] and \
           self.is_stream_processing_problem(fwd_problem):
            stream_processing = True

        fwd_inputs = self.generate_inputs(fwd_problem)
        #FIXME: decide when we need time predicates
        fwd_solutions = self.sample_solutions(fwd_problem, self.num_left_samples)

        match self.triangulation_mode:
            case TriangulationMode.FWD_INV:
                if stream_processing:
                    _, _, result = self.stream_fwd_inv(fwd_problem, fwd_inputs, fwd_solutions)
                else: 
                    _, _, result = self.fwd_inv(fwd_problem, fwd_inputs, fwd_solutions)
            case TriangulationMode.FWD_SINV:
                if stream_processing:
                    _, _, result = self.stream_fwd_sinv(fwd_problem, fwd_inputs, fwd_solutions)
                else: 
                    _, _, result = self.fwd_sinv(fwd_problem, fwd_inputs, fwd_solutions)
            case TriangulationMode.ENUM_SINV:
                if stream_processing:
                    _, _, result = self.stream_enum_sinv(fwd_problem, fwd_inputs, fwd_solutions)
                else: 
                    _, _, result = self.cascade_enum_sinv(fwd_problem, fwd_inputs, fwd_solutions)
            case TriangulationMode.Postcondition:
                _, _, result = self.postcondition(fwd_problem, fwd_inputs, fwd_solutions)
            case TriangulationMode.OffByOne:
                _, _, result = self.off_by_one(fwd_problem, fwd_inputs, fwd_solutions)
            case TriangulationMode.Syntactic:
                _, _, result = self.syntactic(fwd_problem, fwd_inputs, fwd_solutions)
        raw_data = {
            "method": self.triangulation_mode.value
        }

        return (result, raw_data)
        

    def triangulate(self, prop, left_inputs, left_solutions, right_inputs, right_solutions, bijective=False):
        programs_and_witnesses = []

        for (p, pws) in left_solutions:
            p_witnesses = []
            for (q, qws) in right_solutions:
                if self.checker.check({Side.LEFT: left_inputs, Side.RIGHT: right_inputs},
                                      {Side.LEFT: p, Side.RIGHT: q },
                                      prop):
                    p_witnesses.extend(qws) # inherit witnesses
            if len(p_witnesses) > 0:
                programs_and_witnesses.append((p, p_witnesses))

        return programs_and_witnesses

    def unwrap(self, solutions):

        def unwrap_program(p):
            assert len(p.nested) > 0
            remaining, last = p.nested[:-1], p.nested[-1]
            return Program(last, p.code, nested=remaining)

        return list(map(lambda s: (unwrap_program(s[0]), s[1]), solutions))

    def sample_solutions(self, req: Requirements, n: int, time_predicates=False):
        def gen_time_predicate(req, program):
            return program.gen_time_predicate(self.model, req)
        
        programs = self.code_generator.generate(self.model, req, n)
        programs = list(islice(programs, n))
        if time_predicates:
            programs = list(map(partial(gen_time_predicate, req), programs))

        #NOTE: initially, each solution is its own witness            
        return list(map(lambda p: (p, [p]), programs))

    def generate_inputs(self, req: Requirements):
        return generate_inputs(self.model, req)
 
    def is_stream_processing_problem(self, req: Requirements) -> bool:
        if not(len(req.signature.params) == 1 and
               req.signature.params[0].type.lower().startswith('list') and
               req.signature.return_type.lower().startswith('list')):
            return False

        PROMPT = f"""\
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}

Does the problem consist of applying the same operation independently to each element of the input list and returning the per-element results in order? Respond Yes or No. Wrap your answer with the tags `<answer>` and `</answer>`.

Problem:
{req.description}
        """
        judgement = gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        return judgement.lower() == "yes"

    def choose_inversion_scheme(self, req: Requirements) -> InversionScheme:
        if len(req.signature.params) == 1:
            if req.signature.params[0].type.lower().startswith('list'):
                return ListSuffixInversion(0, 1)
            return ParameterInversion(0)
        else:
            PROMPT = f"""\
The problem below is solved using a function with the signature:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}

Choose a single input parameter of the function to be replaced by
its output, thereby formulating an inverse problem. Determine which
parameter, when inverted, would yield the most natural or well-posed
inverse formulation. Exclude parameters whose values can be readily
deduced from other inputs.

Output only the full name of this parameter (not its type) enclosed
within `<answer>` and `</answer>` tags.

Problem:
{req.description}"""
            ind_model = Independent(self.model)
            return_param = None
            tried_samples = []
            for attempt in range(3):
                try:
                    sample = next(ind_model.sample(PROMPT, 3))
                    tried_samples.append(sample)
                    valid_name = extract_answer(sample)
                    valid_name = valid_name.strip() if valid_name else None
                    if valid_name:
                        return_param = [p.name for p in req.signature.params].index(valid_name)
                        break
                    else:
                        continue
                except Exception as e:
                    if attempt == 2:
                        raise ExperimentFailure(f"retry failed with {type(e).__name__}: {e}")
            return ParameterInversion(return_param)

    def split_list_signature(self, index: int, s: Signature):
        new_sig = copy.deepcopy(s)
        new_sig.name = new_sig.name + f"_split_{index}"
        new_sig.params = new_sig.params[0:index] + \
            [ Parameter(new_sig.params[index].name + "_prefix", new_sig.params[index].type),
              Parameter(new_sig.params[index].name + "_suffix", new_sig.params[index].type) ] + \
            new_sig.params[index+1:]
        return new_sig

    def split_list_adapter(self, problem, inputs, solutions, index, suffix_length):
        sig = problem.signature
        new_sig = self.split_list_signature(index, problem.signature)
        
        assert problem.signature_note is None
        match suffix_length:
            case 1:
                sig_note = f"""\
where the input {sig.params[index].name} is split into {new_sig.params[index].name} + {new_sig.params[index+1].name}, so that {new_sig.params[index+1].name} has exactly one element if {sig.params[index].name} is not empty, and {new_sig.params[index+1].name} is empty if {sig.params[index].name} is empty."""
            case _:
                sig_note = f"""\
where the input {sig.params[index].name} is split into {new_sig.params[index].name} + {new_sig.params[index+1].name}, so that len({new_sig.params[index+1].name}) is exactly {suffix_length} when {sig.params[index].name} has at least {suffix_length} elements, otherwise {new_sig.params[index+1].name} contains the entire {sig.params[index].name}."""

        adapted_problem = Requirements(new_sig, problem.description, signature_note=sig_note)
        
        def adapt_program(p):
            ADAPTER_CODE=f"""
def {new_sig.name}(*args):
    args = list(args)
    new_args = args[:{index}] + [ args[{index}] + args[{index+1}] ] + args[{index+2}:]
    return {p.signature.name}(*new_args)
            """
            return Program(new_sig, p.code + "\n" + ADAPTER_CODE, nested = p.nested + [p.signature])
        adapted_solutions = list(map(lambda s: (adapt_program(s[0]), s[1]), solutions))

        def adapt_input(args):
            args = copy.deepcopy(args)
            lst = args[index]
            assert isinstance(lst, list) # we only adapt inputs that go through type check
            split_at = max(0, len(lst) - suffix_length)
            prefix = lst[:split_at]
            suffix = lst[split_at:]
            args[index:index+1] = [prefix, suffix]
            return args                

        adapted_inputs = list(map(adapt_input, inputs))

        return adapted_problem, adapted_inputs, adapted_solutions


    def unpack_argument_signature(self, req):
        PROMPT_CODE = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}

Unpack the input tuple so that each element of the tuple becomes a separate function parameters. 

- The parameters must have descriptive names
- The parameters must be annotated with types according to elements of the original tuple
- The return type should remain the same.
        
Please return only the function definition with 'pass' as the body inside a Markdown code block.

Problem:
{req.description}
        """
        code = gen_and_extract_code_with_retry(self.model, PROMPT_CODE, 3)
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = Signature.from_function_ast(node)
                sig.name = sig.name + "_unpacked"
                return sig
        raise ValueError("No function definition found in code")

    def transform_unpack_argument(self, req):
        new_sig = self.unpack_argument_signature(req)
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}        

Your task is to rewrite this problem so that it instead requires implementing the function that unpacks the input tuple {req.signature.params[0].name}:

{new_sig.pretty_print()}

Important points to follow:
1. Preserve all the original problem's rules, edge cases, and constraints.
2. The problem must be self-contained, and must not refer to the original function.        
3. Exclude all examples.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        new_desc = gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        return Requirements(new_sig, new_desc)
    
    def unpack_argument_adapter(self, problem, inputs, solutions):
        sig = problem.signature
        assert len(sig.params) == 1 and sig.params[0].type.lower().startswith("tuple")
        
        adapted_problem = self.transform_unpack_argument(problem)
        new_sig = adapted_problem.signature
        
        def adapt_program(p):
            ADAPTER_CODE=f"""
def {new_sig.name}(*args):
    return {p.signature.name}(args)
            """
            return Program(new_sig, p.code + "\n" + ADAPTER_CODE, nested = p.nested + [p.signature])
        adapted_solutions = list(map(lambda s: (adapt_program(s[0]), s[1]), solutions))

        def adapt_input(args):
            return list(args[0])

        adapted_inputs = list(map(adapt_input, inputs))

        return adapted_problem, adapted_inputs, adapted_solutions    

    def transform_syntactic(self, req: Requirements) -> Requirements:
        PROMPT = f"""
You are given a programming problem written in English that requires implementing the function:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}

Your task is to translate the entire problem description into Chinese, while preserving all the technical accuracy and meaning. Whenever the description first mentions an input parameter, include its original English name from the function signature in parentheses immediately after the Chinese text describing that parameter. 

Translation guidelines:
1. Keep technical terms such as data types, built-in function names, variable names, and programming language keywords in English.
2. Preserve all constraints, and formatting from the original text.
3. The problem must be self-contained, and must not refer to the original function.

Output the translated problem enclosed in `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        translated_description = gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        return Requirements(req.signature, translated_description)

    def transform_off_by_one(self, req: Requirements) -> Requirements:
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
                add_sentence = EXTRA_INSTR.format(sth="add 1 to each element of the resulting tuple")           
            case _:
                # troublesome to support more complex types
                raise ExperimentFailure()
        return Requirements(req.signature, req.description + "\n" + add_sentence)


    def inv_signature(self, sig: NamedReturnSignature, inversion_index) -> Signature:
        new_return_type = sig.params[inversion_index].type
        new_params = [Parameter(sig.return_name, sig.return_type)]
        new_params.extend(p for i, p in enumerate(sig.params) if i != inversion_index)
        new_func_name = sig.name + "_inv_" + sig.params[inversion_index].name
        new_sig = Signature(new_func_name, new_params, new_return_type)
        return new_sig

    def inv_description(self, fwd_req: Requirements, inv_sig: Signature, inverse_index: int) -> str:
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{fwd_req.signature.pretty_print()}

{fwd_req.signature_note if fwd_req.signature_note is not None else ""}

Rewrite this problem so that it instead requires implementing the inverted function:

{inv_sig.pretty_print()}

Given the desired output value `{inv_sig.params[0].name}` (corresponding to the original function's return value), the new function should return a value for the parameter `{fwd_req.signature.params[inverse_index].name}` such that if the original function were called with this value (and the other parameters unchanged), it would produce `{inv_sig.params[0].name}` as the result.

Important points to follow:
1. Preserve all original constraints, rules, and assumptions from the problem statement.
2. If multiple values of `{fwd_req.signature.params[inverse_index].name}` could produce the desired result, clearly state that any valid value is acceptable.
3. The problem must be self-contained, and must not refer to the original function.        
4. Exclude all examples.

Output the rewritten problem statement enclosed within `<answer>` and `</answer>` tags.

Original Problem:
{fwd_req.description}
        """
        return gen_and_extract_answer_with_retry(self.model, PROMPT, 3)

    def transform_inv(self, fwd_req: Requirements, inversion_index: int) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(self.model, fwd_req)
        inv_sig = self.inv_signature(named_sig, inversion_index)
        inv_desc = self.inv_description(fwd_req, inv_sig, inversion_index)
        return Requirements(inv_sig, inv_desc)

    def transform_postcondition(self, req: Requirements) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(self.model, req)
        last_param = Parameter(named_sig.return_name, named_sig.return_type)
        post_sig = Signature('postcondition', named_sig.params + [last_param], 'bool')
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}        

Your task is to rewrite this problem so that it instead requires implementing the function:

{post_sig.pretty_print()}

The new function should verify whether the given output value (`{last_param.name}`) is correct for the specified inputs, according to the original problem description.

Important points to follow:
1. Preserve all the original problem's rules, edge cases, and constraints.
2. If the original problem allows multiple correct solutions, clarify that the new function must return True for any valid output that meets the problem criteria.
3. The problem must be self-contained, and must not refer to the original function.        
4. Exclude all examples.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        post_desc = gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        return Requirements(post_sig, post_desc)

    def transform_enum(self, req: Requirements) -> Requirements:
        #TODO: support infinite sets
        sig = req.signature
        enum_sig = Signature(f"{sig.name}_enum", sig.params, f"list[{sig.return_type}]")
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{sig.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}        

Your task is to rewrite this problem so that it instead requires implementing the function:

{enum_sig.pretty_print()}

The new function must exhaustively enumerate all correct outputs for a given input according to the original problem description.

Important points to follow:
1. Preserve all the original problem's rules, edge cases, and constraints, but avoid your own interpretation that is not mentioned in the original problem.
2. The problem must be self-contained, and must not refer to the original function.
3. Exclude all examples.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        enum_desc = gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        return Requirements(enum_sig, enum_desc)

    def sinv_signature(self, sig: NamedReturnSignature, inversion_index) -> Signature:
        #TODO: should we also have a simple version without bool?
        new_return_type = "tuple[bool,list[" + sig.params[inversion_index].type + "]]"
        new_params = [Parameter(sig.return_name, sig.return_type)]
        new_params.extend(p for i, p in enumerate(sig.params) if i != inversion_index)
        new_func_name = sig.name + "_sinv_" + sig.params[inversion_index].name
        new_sig = Signature(new_func_name, new_params, new_return_type)
        return new_sig
        
    def sinv_description(self, fwd_req: Requirements, sinv_sig: Signature, inverse_index: int):
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{fwd_req.signature.pretty_print()}

{fwd_req.signature_note}

Rewrite this problem so that it instead requires implementing the set-valued inverted function:

{sinv_sig.pretty_print()}

Given the desired output value `{sinv_sig.params[0].name}` (corresponding to the original function's return value), the new function should return a list of values for the parameter `{fwd_req.signature.params[inverse_index].name}` such that if the original function were called with any of these values (and the other parameters unchanged), it would produce `{sinv_sig.params[0].name}` as the result.

The function should return a tuple: (is_exhaustive_list, list_of_values). When it is feasible to enumerate all such values, return the complete list and set is_exhaustive_list to True. If a complete enumeration is impossible (e.g., the set is infinite or prohibitively large), return a representative subset and set is_exhaustive_list to False.

Important points to follow:
1. Preserve all constraints, domain assumptions, and rules from the original problem.
2. Clearly explain that the output must include all possible values.
3. Specify explicitly that if no such values exist, the function should return an empty list, and mark it as exhaustive.
4. The problem must be self-contained, and must not refer to the original function.
5. Exclude all examples.

Enclose the rewritten problem description inside `<answer>` and `</answer>` tags.

Original Problem:
{fwd_req.description}
        """
        return gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        
    def transform_sinv(self, fwd_req: Requirements, inversion_index: int) -> Requirements:
        named_sig = NamedReturnSignature.infer_name(self.model, fwd_req)
        sinv_sig = self.sinv_signature(named_sig, inversion_index)
        sinv_desc = self.sinv_description(fwd_req, sinv_sig, inversion_index)
        return Requirements(sinv_sig, sinv_desc)

    def pointwise_signature(self, sig: Signature) -> Signature:
        assert len(sig.params) == 1
        assert sig.params[0].type.lower().startswith('list')
        assert sig.return_type.lower().startswith('list')
        input_element_type = sig.params[0].type[5:-1]
        return_element_type = sig.return_type[5:-1]
        return Signature(sig.name + "_single", [Parameter("single_query", input_element_type)], return_element_type)

    def transform_pointwise(self, req: Requirements) -> Requirements:
        pointwise_sig = self.pointwise_signature(req.signature)
        PROMPT = f"""
You are given a programming problem that requires implementing the function:

{req.signature.pretty_print()}

{req.signature_note if req.signature_note is not None else ""}        

Your task is to rewrite this problem so that it instead requires implementing the function:

{pointwise_sig.pretty_print()}

The new function must process individual elements, mirroring the original list-processing logic applied element-wise.

Important points to follow:
1. Preserve all the original problem's rules, edge cases, and constraints.
2. The problem must be self-contained, and must not refer to the original function.
3. Exclude all examples.

Enclose the rewritten problem statement inside `<answer>` and `</answer>` tags.

Original Problem:
{req.description}
        """
        pointwise_desc = gen_and_extract_answer_with_retry(self.model, PROMPT, 3)
        return Requirements(pointwise_sig, pointwise_desc)

    def adapt_pointwise(self, problem, inputs, solutions):
        adapted_problem = self.transform_pointwise(problem)
        new_sig = adapted_problem.signature
        
        def adapt_program(p):
            ADAPTER_CODE=f"""
def {new_sig.name}(el):
    return {problem.signature.name}([el])[0]
            """
            return Program(new_sig, p.code + "\n" + ADAPTER_CODE, nested = p.nested + [p.signature])
        adapted_solutions = list(map(lambda s: (adapt_program(s[0]), s[1]), solutions))

        adapted_inputs = [[y] for sub in inputs for x in sub for y in x]
        adapted_inputs = remove_duplicates(adapted_inputs)

        return adapted_problem, adapted_inputs, adapted_solutions

    def make_equiv(self, req):
        arity = len(req.signature.params)
        args = [Var(f"i_{i}") for i in range(arity)]
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)

        return ForAll(args, Side.LEFT, Equals([p(args), q(args)]))

    def make_off_by_one(self, req):
        arity = len(req.signature.params)
        args = [Var(f"i_{i}") for i in range(arity)]
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)
        
        return ForAll(args, Side.LEFT, Equals([OffByOne([p(args)]), q(args)]))
                      
    def make_postcondition(self, req):
        arity = len(req.signature.params)
        args = [Var(f"i_{i}") for i in range(arity)]
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)

        return ForAll(args, Side.LEFT, TolerateInvalid([q(args + [p(args)])]))

    def make_stateless_map(self, req):
        assert len(req.signature.params) == 1
        
        arg = Var(f"i")
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)
        
        return ForAll(arg, Side.LEFT,
                      Equals([TolerateInvalid([p(arg)]), FlattenMap(q, arg)]))

    def make_fwd_inv(self, req, inversion_index):
        arity = len(req.signature.params)
        args = [Var(f"i_{i}") for i in range(arity)]
        inv_arg = Var(f"i_{inversion_index}")
        remaining_args = args[:inversion_index] + args[inversion_index + 1:]
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)
        
        return ForAll(args, Side.LEFT,
                      Equals([inv_arg, q([TolerateInvalid([p(args)])] + remaining_args)]))

    def make_fwd_sinv(self, req, inversion_index):
        arity = len(req.signature.params)
        args = [Var(f"i_{i}") for i in range(arity)]
        inv_arg = Var(f"i_{inversion_index}")
        arg_prime = Var(f"i_prime")
        remaining_args = args[:inversion_index] + args[inversion_index + 1:]
        args_with_prime = args[:inversion_index] + [arg_prime] + args[inversion_index+1:]
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)

        return ForAll(args, Side.LEFT,
                      And(Member([inv_arg, FullOrPartial([q([TolerateInvalid([p(args)])] + remaining_args)])]),
                          ForAll(arg_prime, FullOrPartial([q([TolerateInvalid([p(args)])] + remaining_args)]),
                                 Equals([p(args), TimeoutGuard(p)(args_with_prime)]))))

    def make_enum_sinv(self, req, inversion_index):
        arity = len(req.signature.params)
        left_args = [Var(f"i_{i}") for i in range(arity)]
        inv_arg = Var(f"i_{inversion_index}")
        out = Var(f"o")
        right_args = [out] + left_args[:inversion_index] + left_args[inversion_index + 1:]

        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)

        return And(ForAll(left_args, Side.LEFT,
                          ForAll(out, FullOrPartial([TolerateInvalid([p(left_args)])]),
                                 Member([inv_arg, FullOrPartial([TimeoutGuard(q)(right_args)])]))),
                   ForAll(right_args, Side.RIGHT,
                          ForAll(inv_arg, FullOrPartial([TolerateInvalid([q(right_args)])]),
                                 Member([out, FullOrPartial([TimeoutGuard(p)(left_args)])]))))

    def make_fwd_enum(self, req):
        arity = len(req.signature.params)
        args = [Var(f"i_{i}") for i in range(arity)]
        p = Func(Side.LEFT)
        q = Func(Side.RIGHT)

        return ForAll(args, Side.LEFT, Member([p(args), FullOrPartial([q(args)])]))        

    def postcondition(self, fwd_problem, fwd_inputs, fwd_solutions):
        post_problem = self.transform_postcondition(fwd_problem)
        post_solutions = self.sample_solutions(post_problem, self.num_right_samples)
        post_prop = self.make_postcondition(fwd_problem)
        triangulated_fwd_solutions = self.triangulate(post_prop,
                                                      fwd_inputs,
                                                      fwd_solutions,
                                                      [],
                                                      post_solutions,
                                                      bijective=False)
        return fwd_problem, fwd_inputs, triangulated_fwd_solutions

    def off_by_one(self, fwd_problem, fwd_inputs, fwd_solutions):
        off_by_one_problem = self.transform_off_by_one(fwd_problem)
        off_by_one_solutions = self.sample_solutions(off_by_one_problem, self.num_right_samples)
        off_by_one_prop = self.make_off_by_one(fwd_problem)
        triangulated_fwd_solutions = self.triangulate(off_by_one_prop,
                                                      fwd_inputs,
                                                      fwd_solutions,
                                                      [],
                                                      off_by_one_solutions,
                                                      bijective=False)
        return fwd_problem, fwd_inputs, triangulated_fwd_solutions

    def syntactic(self, fwd_problem, fwd_inputs, fwd_solutions):
        syntactic_problem = self.transform_syntactic(fwd_problem)
        syntactic_solutions = self.sample_solutions(syntactic_problem, self.num_right_samples)
        equiv_prop = self.make_equiv(fwd_problem)
        triangulated_fwd_solutions = self.triangulate(equiv_prop,
                                                      fwd_inputs,
                                                      fwd_solutions,
                                                      [],
                                                      syntactic_solutions,
                                                      bijective=False)
        return fwd_problem, fwd_inputs, triangulated_fwd_solutions

    def fwd_inv(self, fwd_problem, fwd_inputs, fwd_solutions):
        print(f"\n[fwd_inv]", file=sys.stderr, flush=True)
        
        match self.choose_inversion_scheme(fwd_problem):
            case ParameterInversion(i):
                inv_problem = self.transform_inv(fwd_problem, i)
                inv_solutions = self.sample_solutions(inv_problem, self.num_right_samples)
                fwd_inv_prop = self.make_fwd_inv(fwd_problem, i)
                triangulated_fwd_solutions = \
                    self.triangulate(fwd_inv_prop,
                                     fwd_inputs,
                                     fwd_solutions,
                                     [],
                                     inv_solutions,
                                     bijective=True)
            case ListSuffixInversion(i, l):
                split_list_problem, split_list_inputs, split_list_solutions = \
                    self.split_list_adapter(fwd_problem, fwd_inputs, fwd_solutions, i, l)
                inv_problem = self.transform_inv(split_list_problem, i+1)
                inv_solutions = self.sample_solutions(inv_problem, self.num_right_samples)
                fwd_inv_prop = self.make_fwd_inv(split_list_problem, i+1)
                triangulated_split_list_solutions = \
                    self.triangulate(fwd_inv_prop,
                                     split_list_inputs,
                                     split_list_solutions,
                                     [],
                                     inv_solutions,
                                     bijective=True)
                triangulated_fwd_solutions = self.unwrap(triangulated_split_list_solutions)

        return fwd_problem, fwd_inputs, triangulated_fwd_solutions

    def fwd_sinv(self, fwd_problem, fwd_inputs, fwd_solutions):

        match self.choose_inversion_scheme(fwd_problem):
            case ParameterInversion(i):
                sinv_problem = self.transform_sinv(fwd_problem, i)
                sinv_solutions = self.sample_solutions(sinv_problem, self.num_right_samples)
                
                fwd_sinv_prop = self.make_fwd_sinv(fwd_problem, i)
                triangulated_fwd_solutions = \
                    self.triangulate(fwd_sinv_prop,
                                     fwd_inputs,
                                     fwd_solutions,
                                     [],
                                     sinv_solutions,
                                     bijective=True)
            case ListSuffixInversion(i, l):
                split_list_problem, split_list_inputs, split_list_solutions = \
                    self.split_list_adapter(fwd_problem, fwd_inputs, fwd_solutions, i, l)
                fwd_sinv_prop = self.make_fwd_sinv(split_list_problem, i+1)
                sinv_problem = self.transform_sinv(split_list_problem, i+1)
                sinv_solutions = self.sample_solutions(sinv_problem, self.num_right_samples)
                triangulated_split_list_solutions = \
                    self.triangulate(fwd_sinv_prop,
                                     split_list_inputs,
                                     split_list_solutions,
                                     [],
                                     sinv_solutions,
                                     bijective=True)
                triangulated_fwd_solutions = self.unwrap(triangulated_split_list_solutions)

        return fwd_problem, fwd_inputs, triangulated_fwd_solutions


    def enum_sinv(self, fwd_problem, fwd_inputs, fwd_solutions):

        match self.choose_inversion_scheme(fwd_problem):
            case ParameterInversion(i):
                enum_problem = self.transform_enum(fwd_problem)
                if (just_tri_it.utils.DEBUG):
                    print(enum_problem.get_content(), file=sys.stderr, flush=True)
                
                enum_inputs = self.generate_inputs(enum_problem)
                enum_solutions = self.sample_solutions(enum_problem, self.num_left_samples)
                
                sinv_problem = self.transform_sinv(fwd_problem, i)
                sinv_inputs = self.generate_inputs(sinv_problem)
                sinv_solutions = self.sample_solutions(sinv_problem, self.num_right_samples)
                
                enum_sinv_prop = self.make_enum_sinv(fwd_problem, i)
                triangulated_enum_solutions = \
                    self.triangulate(enum_sinv_prop,
                                     enum_inputs,
                                     enum_solutions,
                                     sinv_inputs,
                                     sinv_solutions,
                                     bijective=True)
            case ListSuffixInversion(i, l):
                split_list_problem, _, _ = \
                    self.split_list_adapter(fwd_problem, fwd_inputs, fwd_solutions, i, l)

                enum_problem = self.transform_enum(fwd_problem)
                enum_inputs = self.generate_inputs(enum_problem)
                enum_solutions = self.sample_solutions(enum_problem, self.num_left_samples)
                
                _, split_list_enum_inputs, split_list_enum_solutions = \
                    self.split_list_adapter(enum_problem, enum_inputs, enum_solutions, i, l)
                
                sinv_problem = self.transform_sinv(split_list_problem, i+1)
                sinv_inputs = self.generate_inputs(sinv_problem)
                sinv_solutions = self.sample_solutions(sinv_problem, self.num_right_samples)

                enum_sinv_prop = self.make_enum_sinv(split_list_problem, i+1)
                
                triangulated_split_list_enum_solutions = \
                    self.triangulate(enum_sinv_prop,
                                     split_list_enum_inputs,
                                     split_list_enum_solutions,
                                     sinv_inputs,
                                     sinv_solutions,
                                     bijective=True)
                
                triangulated_enum_solutions = self.unwrap(triangulated_split_list_enum_solutions)

        return enum_problem, enum_inputs, triangulated_enum_solutions


    def cascade_enum_sinv(self, fwd_problem, fwd_inputs, fwd_solutions):

        print(f"\n[cascade_enum_sinv]", file=sys.stderr, flush=True)

        _, _, triangulated_enum_solutions = self.enum_sinv(fwd_problem, fwd_inputs, fwd_solutions)

        print(f"\n[enum solutions: {len(triangulated_enum_solutions)}]", file=sys.stderr, flush=True)

        fwd_enum_prop = self.make_fwd_enum(fwd_problem)
        triangulated_fwd_solutions = \
            self.triangulate(fwd_enum_prop,
                             fwd_inputs,
                             fwd_solutions,
                             [],
                             triangulated_enum_solutions,
                             bijective=False)

        print(f"\n[fwd solutions: {len(triangulated_fwd_solutions)}]", file=sys.stderr, flush=True)

        return fwd_problem, fwd_inputs, triangulated_fwd_solutions

    def stream_fwd_inv(self, multiple_queries_problem, multiple_queries_inputs, multiple_queries_solutions):
        print(f"\n[stream_fwd_inv]", file=sys.stderr, flush=True)
        
        single_query_adapted_problem, single_query_adapted_inputs, single_query_adapted_solutions = \
            self.adapt_pointwise(multiple_queries_problem,
                                 multiple_queries_inputs,
                                 multiple_queries_solutions)

        tuple_unpacked = False
        if len(single_query_adapted_problem.signature.params) == 1 and \
           single_query_adapted_problem.signature.params[0].type.lower().startswith("tuple"):
            single_query_adapted_problem, single_query_adapted_inputs, single_query_adapted_solutions = \
                self.unpack_argument_adapter(single_query_adapted_problem,
                                             single_query_adapted_inputs,
                                             single_query_adapted_solutions)
            tuple_unpacked = True

        _, _, triangulated_single_query_adapted_solutions = \
            self.fwd_inv(single_query_adapted_problem,
                         single_query_adapted_inputs,
                         single_query_adapted_solutions)

        print(f"\n[single query solutions: {len(triangulated_single_query_adapted_solutions)}]", file=sys.stderr, flush=True)

        if tuple_unpacked:
            triangulated_single_query_adapted_solutions = self.unwrap(triangulated_single_query_adapted_solutions)
        
        triangulated_single_query_unwraped_solutions = self.unwrap(triangulated_single_query_adapted_solutions)

        
        stateless_map = self.make_stateless_map(multiple_queries_problem)
        triangulated_multiple_queries_solutions = \
            self.triangulate(stateless_map,
                             multiple_queries_inputs,
                             triangulated_single_query_unwraped_solutions,
                             [],
                             triangulated_single_query_adapted_solutions,
                             bijective=True)

        print(f"\n[multi-query solutions: {len(triangulated_multiple_queries_solutions)}]", file=sys.stderr, flush=True)

        return multiple_queries_problem, multiple_queries_inputs, triangulated_multiple_queries_solutions

    def stream_fwd_sinv(self, multiple_queries_problem, multiple_queries_inputs, multiple_queries_solutions):
        print(f"\n[stream_fwd_sinv]", file=sys.stderr, flush=True)
        
        single_query_adapted_problem, single_query_adapted_inputs, single_query_adapted_solutions = \
            self.adapt_pointwise(multiple_queries_problem,
                                 multiple_queries_inputs,
                                 multiple_queries_solutions)

        _, _, triangulated_single_query_adapted_solutions = \
            self.fwd_sinv(single_query_adapted_problem,
                          single_query_adapted_inputs,
                          single_query_adapted_solutions)

        print(f"\n[single query solutions: {len(triangulated_single_query_adapted_solutions)}]", file=sys.stderr, flush=True)

        triangulated_single_query_unwraped_solutions = self.unwrap(triangulated_single_query_adapted_solutions)

        stateless_map = self.make_stateless_map(multiple_queries_problem)
        triangulated_multiple_queries_solutions = \
            self.triangulate(stateless_map,
                             multiple_queries_inputs,
                             triangulated_single_query_unwraped_solutions,
                             [],
                             triangulated_single_query_adapted_solutions,
                             bijective=True)

        print(f"\n[multi-query solutions: {len(triangulated_multiple_queries_solutions)}]", file=sys.stderr, flush=True)        
        return multiple_queries_problem, multiple_queries_inputs, triangulated_multiple_queries_solutions


    def stream_enum_sinv(self, multiple_queries_problem, multiple_queries_inputs, multiple_queries_solutions):
        print(f"\n[stream_enum_sinv]", file=sys.stderr, flush=True)
        
        single_query_problem = self.transform_pointwise(multiple_queries_problem)
        single_query_inputs = self.generate_inputs(single_query_problem)
        single_query_solutions = self.sample_solutions(single_query_problem, self.num_left_samples)
        
        single_query_adapted_problem, single_query_adapted_inputs, single_query_adapted_solutions = \
            self.adapt_pointwise(multiple_queries_problem,
                                 multiple_queries_inputs,
                                 multiple_queries_solutions)
        
        single_query_enum_problem, single_query_enum_inputs, triangulated_single_query_enum_solutions = \
            self.enum_sinv(single_query_problem,
                           single_query_inputs,
                           single_query_solutions)

        print(f"\n[single query enum solutions: {len(triangulated_single_query_enum_solutions)}]", file=sys.stderr, flush=True)

        fwd_enum_prop = self.make_fwd_enum(single_query_adapted_problem)
        triangulated_single_query_adapted_solutions = \
            self.triangulate(fwd_enum_prop,
                             single_query_adapted_inputs,
                             single_query_adapted_solutions,
                             single_query_enum_inputs,
                             triangulated_single_query_enum_solutions,
                             bijective=False)

        print(f"\n[single query adapted solutions: {len(triangulated_single_query_adapted_solutions)}]", file=sys.stderr, flush=True)

        triangulated_single_query_unwraped_solutions = self.unwrap(triangulated_single_query_adapted_solutions)

        stateless_map = self.make_stateless_map(multiple_queries_problem)
        triangulated_multiple_queries_solutions = \
            self.triangulate(stateless_map,
                             multiple_queries_inputs,
                             triangulated_single_query_unwraped_solutions,
                             [],
                             triangulated_single_query_adapted_solutions,
                             bijective=True)

        print(f"\n[multi-query solutions: {len(triangulated_multiple_queries_solutions)}]", file=sys.stderr, flush=True)

        return multiple_queries_problem, multiple_queries_inputs, triangulated_multiple_queries_solutions    
