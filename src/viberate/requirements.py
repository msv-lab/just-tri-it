from dataclasses import dataclass
import sys
from itertools import islice

from viberate.program import Signature, Parameter
from viberate.llm import extract_answer
from viberate.utils import print_annotated_hr
from viberate.llm import LLM


@dataclass
class Requirements:
    signature: Signature
    description: str

    def __str__(self):
        return self.signature.pretty_print() + "\n" + self.description

    @staticmethod
    def from_description(model: LLM, desc: str) -> 'Requirements':
        return Requirements(Signature.from_description(model, desc), desc)


@dataclass
class NamedReturnSignature(Signature):
    return_name: str

    @staticmethod
    def infer_name(model: LLM, req: Requirements) -> 'NamedReturnSignature':
        PROMPT = f""" For the problem below, name its return value
        descriptively using Python's snake_case naming convention.
        Enclose the name in <answer> tags.
        
        Problem:
        {req.description}
        """
        return_name = extract_answer(next(model.sample(PROMPT)))
        return NamedReturnSignature(req.signature.name,
                                    req.signature.params,
                                    req.signature.return_type,
                                    return_name)


def choose_parameter_to_invert(model: LLM, req: Requirements) -> int:
    # print(req.signature.pretty_print())
    if len(req.signature.params) == 1:
        return 0
    else:
        PROMPT = f""" For the problem below and its corresponding
        function signature, I need to invert this problem, so I need
        to select a parameter to swap with the return value.
        Which parameter do you think would make the inverse
        problem relatively easier to solve? Please pay attention that
        you **must not** choose parameters that can be easily derived from other parameters.
        Only answer the full name of this parameter (not its type) in <answer> tags.
        In function signature, when type hints are used, the part
        before the colon is exactly the name of that parameter. 
        
        Problem:
        {req.description}
        Function signature:
        {req.signature.pretty_print()}
        """
        name_lst = list(islice(model.sample(PROMPT), 3))
        valid_name = None
        for n in name_lst:
            try:
                valid_name = extract_answer(n)
                if valid_name is not None:
                    break
            except:
                continue
        return [p.name for p in req.signature.params].index(valid_name)


def inverse_requirements(model: LLM, req: Requirements, inverse_index: int) -> Requirements:
    print_annotated_hr("Signature")
    print(req.signature.pretty_print(), file=sys.stderr)

    named_sig = NamedReturnSignature.infer_name(model, req)
    inverse_sig = _inverse_signature(model, named_sig, inverse_index)
    print_annotated_hr(f"Inverse signature wrt {inverse_index}")
    print(inverse_sig.pretty_print(), file=sys.stderr)

    if len(req.signature.params) == 1:
        inverse_desc = _inverse_description_single_arg(model, req, inverse_sig)
    else:
        inverse_desc = _inverse_description(model, req, inverse_sig, inverse_index)

    return Requirements(inverse_sig, inverse_desc)


def _inverse_signature(model: LLM,
                       sig: NamedReturnSignature,
                       inverse_index: int) -> Signature:
    new_return_type = sig.params[inverse_index].type
    new_params = [Parameter(sig.return_name, sig.return_type)]
    new_params.extend(p for i, p in enumerate(sig.params) if i != inverse_index)
    new_func_name = "inverse_" + sig.name + "_wrt_" + sig.params[inverse_index].name
    new_sig = Signature(new_func_name, new_params, new_return_type)
    return new_sig


def _inverse_description_single_arg(model: LLM,
                                    req: Requirements,
                                    inverted_sig: Signature) -> str:
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {req.signature.pretty_print()}, so that
    it instead requires implementing the function
    {inverted_sig.pretty_print()}. The new function should, for each
    possible output of the original function, return the possible
    input that produce that output. Try to maintain accuracy during
    the conversion process. Enclose your rewritten problem in
    <answer> tags.

    Problem:
    {req.description}
    """
    return extract_answer(next(model.sample(PROMPT)))


def _inverse_description(model: LLM,
                         req: Requirements,
                         inverted_sig: Signature,
                         inverse_index: int) -> str:
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {req.signature.pretty_print()}, so that
    it requiresimplementing the function {inverted_sig.pretty_print()}
    instead. The new function should return the value of the parameter
    {req.signature.params[inverse_index].name} that produce that given
    output. Try to maintain accuracy during the conversion process. 
    Enclose your rewritten problem in <answer> tags.

    Problem:
    {req.description}
    """
    return extract_answer(next(model.sample(PROMPT)))


def fiber_requirements(model: LLM, req: Requirements, inverse_index: int) -> Requirements:
    print_annotated_hr("Signature")
    print(req.signature.pretty_print(), file=sys.stderr)

    named_sig = NamedReturnSignature.infer_name(model, req)
    fiber_sig = _fiber_signature(model, named_sig, inverse_index)
    print_annotated_hr(f"Fiber signature wrt {inverse_index}")
    print(fiber_sig.pretty_print(), file=sys.stderr)

    if len(req.signature.params) == 1:
        fiber_desc = _fiber_description_single_arg(model, req, fiber_sig)
    else:
        fiber_desc = _fiber_description(model, req, fiber_sig, inverse_index)

    return Requirements(fiber_sig, fiber_desc)


def specific_requirements(model, fiber_req, fiber_input, choice):
    specific_question = "What's the answer when "
    # unfinished
    for index, param in enumerate(fiber_req.signature.params):
        if index == 0:
            specific_question += "'" + param.name + "'" + " is " + str(fiber_input[index])
        elif index == len(fiber_req.signature.params) - 1:
            specific_question += " and '" + param.name + "' is " + str(fiber_input[index]) + "?"
        else:
            specific_question += ", '" + param.name + "' is " + str(fiber_input[index])
    if len(fiber_req.signature.params) == 1:
        specific_question += "?"
    print(specific_question)
    REMOVE_PROMPT = f"""
    Remove the sections such as Input, Output, Constraints, and Example from the
    following problem description, leaving only the complete problem statement. Words like
    "implement a function ..." should also be removed.
    Enclose your rewritten problem in <answer> tags.
    
    Problem Description:
    {fiber_req.description}
    """
    revised_question = extract_answer(next(model.sample(REMOVE_PROMPT)))
    match choice:
        case 'fiber':
            complete_question = (revised_question + "\n" + specific_question +
                                 " Please return all possible answers in a list.")
        case 'inverse':
            complete_question = revised_question + "\n" + specific_question
        case _:
            complete_question = None
    print(complete_question)
    new_sig = Signature(fiber_req.signature.name, [], fiber_req.signature.return_type)
    return Requirements(new_sig, complete_question)


def _fiber_signature(model: LLM,
                     sig: NamedReturnSignature,
                     inverse_index: int) -> Signature:
    new_return_type = "list[" + sig.params[inverse_index].type + "]"
    new_params = [Parameter(sig.return_name, sig.return_type)]
    new_params.extend(p for i, p in enumerate(sig.params) if i != inverse_index)
    new_func_name = "fiber_" + sig.name + "_wrt_" + sig.params[inverse_index].name
    new_sig = Signature(new_func_name, new_params, new_return_type)
    return new_sig


def _fiber_description_single_arg(model: LLM,
                                  req: Requirements,
                                  fiber_sig: Signature):
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {req.signature.pretty_print()}, so that
    it instead requires implementing the function
    {fiber_sig.pretty_print()}. The new function should, for each
    possible output of the original function, return the exhaustive
    list of all inputs that produce that output. Try to maintain 
    accuracy during the conversion process. Enclose your rewritten 
    problem in <answer> tags.

    Problem:
    {req.description}
    """
    return extract_answer(next(model.sample(PROMPT)))


def _fiber_description(model: LLM,
                       req: Requirements,
                       fiber_sig: Signature,
                       inverse_index: int):
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {req.signature.pretty_print()}, so that
    it requires implementing the function {fiber_sig.pretty_print()}
    instead. The new function should return the exhaustive list of all
    possible values of the parameter
    {req.signature.params[inverse_index].name} that produce the given
    output. Try to maintain accuracy during the conversion process.
    Enclose your rewritten problem in <answer> tags.

    Problem:
    {req.description}
    """
    return extract_answer(next(model.sample(PROMPT)))


def _fiber_description_wo_example(model: LLM,
                                  req: Requirements,
                                  fiber_sig: Signature,
                                  inverse_index: int):
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {req.signature.pretty_print()}, so that
    it requires implementing the function {fiber_sig.pretty_print()}
    instead. The new function should return the exhaustive list of all
    possible values of the parameter
    {req.signature.params[inverse_index].name} that produce the given
    output. The revised problem must not include input-output
    examples. Enclose your rewritten problem in <answer> tags.

    Problem:
    {req.description}
    """
    return extract_answer(next(model.sample(PROMPT)))
