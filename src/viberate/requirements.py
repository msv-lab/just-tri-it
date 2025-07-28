from dataclasses import dataclass
from typing import Self

from viberate.program import Signature, Parameter
from viberate.llm import extract_answer


@dataclass
class NamedReturnSignature(Signature):
    return_name: str

    @staticmethod
    def from_requirements(model, sig: Signature, req: str) -> Self:
        PROMPT = f""" For the problem below, name its return value
        descriptively using Python's snake_case naming convention.
        Enclose the name in <answer> tags.
        
        Problem:
        {req}
        """
        return_name = extract_answer(next(model.sample(PROMPT)))
        return NamedReturnSignature(sig.name,
                                    sig.params,
                                    sig.return_type,
                                    return_name)
        

def choose_parameter_to_invert(model, sig: Signature, req: str) -> int:
    if len(sig.params) == 1:
        return 0
    else:
        PROMPT = f""" For the problem below and its corresponding
        function signature, I need to invert this problem, so I need
        to select a parameter to swap with the return value
        first. Which parameter do you think would make the inverse
        problem easier to solve? Enclose the full name of this
        parameter in <answer> tags.
        
        Problem:
        {req}
        Function signature:
        {sig.pretty_print()}
        """
        name = extract_answer(next(model.sample(PROMPT)))
        return [p.name for p in sig.params].index(name)


def inverse_signature(model,
                      sig: NamedReturnSignature,
                      inverse_index: int,
                      req: str) -> Signature:
    new_return_type = sig.params[inverse_index].type
    new_params = [Parameter(sig.return_name, sig.return_type)]
    new_params.extend(p for i, p in enumerate(sig.params) if i != inverse_index)
    new_func_name = "inverse_" + sig.name + "_wrt_" + sig.params[inverse_index].name
    new_sig = Signature(new_func_name, new_params, new_return_type)
    # PROMPT_NAME = f""" I have inverted the problem below, so its
    # corresponding function signature has changed from
    # {sig.pretty_print()} to {new_sig.pretty_print()}. Please help me
    # create a descriptive name for "inverse_function" following
    # Python's snake_case naming convention. Enclose your answer in
    # <answer> tags.
    #
    # Problem:
    # {req}
    # """
    # new_name = extract_answer(next(model.sample(PROMPT_NAME)))  # sig.name + "_inv_param_" + str(inverse_index)
    # new_sig.name = new_name
    return new_sig


def inverse_requirements_old(model,
                             signature: Signature,
                             inverted_sig: Signature,
                             requirements: str) -> str:
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {signature.pretty_print()}, so that it
    instead requires implementing the function
    {inverted_sig.pretty_print()}. The new function should, for each
    possible output of the original function, return the possible
    input that produce that output. Enclose your rewritten problem in
    <answer> tags.

    Problem:
    {requirements}
    """
    return extract_answer(next(model.sample(PROMPT)))


def inverse_requirements(model,
                         sig: Signature,
                         inverted_sig: Signature,
                         inverse_index: int,
                         requirements: str) -> str:
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {sig.pretty_print()}, so that it
    requiresimplementing the function {inverted_sig.pretty_print()}
    instead. The new function should return the value of the parameter
    {sig.params[inverse_index].name} that produce that given output.
    Enclose your rewritten problem in <answer> tags.

     Problem:
     {requirements}
     """
    return extract_answer(next(model.sample(PROMPT)))


def fiber_signature(model,
                    sig: NamedReturnSignature,
                    inverse_index: int,
                    req: str):
    return_type = "list[" + sig.params[inverse_index].type + "]"
    new_params = [Parameter(sig.return_name, sig.return_type)]
    new_params.extend(p for i, p in enumerate(sig.params) if i != inverse_index)
    new_func_name = "inverse_" + sig.name + "_wrt_" + sig.params[inverse_index].name
    new_sig = Signature(new_func_name, new_params, return_type)
    # PROMPT_NAME = f""" I have inverted the problem below, so its
    #     corresponding function signature has changed from
    #     {sig.pretty_print()} to {new_sig.pretty_print()}. Please
    #     help me create a descriptive name for "inverse_function()"
    #     following Python's snake_case naming convention. Enclose
    #     your answer in <answer> tags.
    #
    #     Problem:
    #     {req}
    #     """
    # new_name = extract_answer(next(model.sample(PROMPT_NAME)))
    # new_sig.name = new_name
    return new_sig


def fiber_requirements_old(model,
                           signature: Signature,
                           fiber_sig: Signature,
                           requirements: str):
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {signature.pretty_print()}, so that it
    instead requires implementing the function
    {fiber_sig.pretty_print()}. The new function should, for each
    possible output of the original function, return the list of all
    inputs that produce that output. Enclose your rewritten problem in
    <answer> tags.

    Problem:
    {requirements}
    """
    return extract_answer(next(model.sample(PROMPT)))


def fiber_requirements(model,
                       sig: Signature,
                       fiber_sig: Signature,
                       inverse_index: int,
                       requirements: str):
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {sig.pretty_print()}, so that it
    requires implementing the function {fiber_sig.pretty_print()}
    instead. The new function should return the list of **all**
    possible values of the parameter {sig.params[inverse_index].name}
    that produce the given output. Enclose your rewritten problem in
    <answer> tags.

    Problem:
    {requirements}
    """
    return extract_answer(next(model.sample(PROMPT)))


def fiber_requirements_wo_example(model,
                                  sig: Signature,
                                  fiber_sig: Signature,
                                  inverse_index: int,
                                  requirements: str):
    PROMPT = f""" Rewrite the given problem, which requires
    implementing the function {sig.pretty_print()}, so that it
    requires implementing the function {fiber_sig.pretty_print()}
    instead. The new function should return the list of **all**
    possible values of the parameter {sig.params[inverse_index].name}
    that produce the given output. The revised problem must not
    include specific examples. Enclose your rewritten problem in
    <answer> tags.

    Problem:
    {requirements}
    """
    return extract_answer(next(model.sample(PROMPT)))

