from viberate.program import Signature, Parameter
from viberate.llm import extract_answer


def inverse_signature_old(sig):
    assert len(sig.params) == 1
    return Signature(sig.name + "_inv", [Parameter("x", sig.return_type)], sig.params[0].type)


def inverse_signature(model, sig, req):
    assert len(sig.params) - 1 >= sig.inverse_index
    return_type = sig.params[sig.inverse_index]
    new_params = [Parameter(sig.return_value.name, sig.return_value.type)]
    new_params.extend(Parameter(param.name, param.type) for i, param in enumerate(sig.params) if i != sig.inverse_index)
    # new_sig = Signature("inverse_function", new_params, return_type)
    new_sig = Signature("inverse_" + sig.name + "_wrt_" + sig.params[sig.inverse_index].name, new_params, return_type)
    # PROMPT_NAME = f"""
    # I have inverted the problem below, so its corresponding function signature has changed from {sig.pretty_print()} to {new_sig.pretty_print()}. Please help me create a descriptive name for "inverse_function" following Python's snake_case naming convention. Enclose your answer in <answer> tags.
    #
    # Problem:
    # {req}
    # """
    # new_name = extract_answer(model.sample(PROMPT_NAME)[0])  # sig.name + "_inv_param_" + str(sig.inverse_index)
    # new_sig.name = new_name
    return new_sig


def inverse_requirements_old(model, signature, inverted_sig, requirements):
#     PROMPT = f"""
# Rewrite the problem below to require implementing the inverted
# function {inverted_sig} instead of {signature}.  Enclose the new
# problem in <answer> tags.
#
# Problem:
# {requirements}
#     """
    PROMPT = f"""
Rewrite the given problem, which requires implementing the function
{signature}, so that it instead requires implementing the function
{inverted_sig}. The new function should, for each possible output of the
original function, return the possible input that produce that
output. Enclose your rewritten problem in <answer> tags.

Problem:
{requirements}
    """
    return extract_answer(model.sample(PROMPT)[0])


def inverse_requirements(model, signature, inverted_sig, requirements):
    PROMPT = f"""
Rewrite the given problem, which requires implementing the function {signature}, so that it requires implementing the function {inverted_sig} instead. The new function should return the input of the {signature.inverse_index}th argument that produce that output.
Enclose your rewritten problem in <answer> tags.

Problem:
{requirements}
    """
    return extract_answer(model.sample(PROMPT)[0])


def fiber_signature_old(sig):
    assert len(sig.params) == 1
    return Signature(sig.name + "_fib", [Parameter("x", sig.return_type)], "list[" + sig.params[0].type + "]")


def fiber_signature(model, sig, req):
    assert len(sig.params) - 1 >= sig.inverse_index
    return_type = "list[" + sig.params[sig.inverse_index].type + "]"
    new_return_value = Parameter("all_possible_values_of_" + sig.return_value.name, return_type)
    new_params = [Parameter(sig.return_value.name, sig.return_value.type)]
    new_params.extend(Parameter(param.name, param.type) for i, param in enumerate(sig.params) if i != sig.inverse_index)
    # new_sig = Signature("fiber_function", new_params, new_return_value)
    new_sig = Signature("inverse_" + sig.name + "_wrt_" + sig.params[sig.inverse_index].name, new_params, new_return_value)
    print(new_sig)
    # PROMPT_NAME = f"""
    #     I have inverted the problem below, so its corresponding function signature has changed from {sig.pretty_print()} to {new_sig.pretty_print()}. Please help me create a descriptive name for "inverse_function()" following Python's snake_case naming convention. Enclose your answer in <answer> tags.
    #
    #     Problem:
    #     {req}
    #     """
    # new_name = extract_answer(model.sample(PROMPT_NAME)[0])
    # new_sig.name = new_name
    return new_sig


def fiber_requirements_old(model, signature, fiber_sig, requirements):
    PROMPT = f"""
Rewrite the given problem, which requires implementing the function
{signature}, so that it instead requires implementing the function
{fiber_sig}. The new function should, for each possible output of the
original function, return the list of all inputs that produce that
output. Enclose your rewritten problem in <answer> tags.

Problem:
{requirements}
    """
    return extract_answer(model.sample(PROMPT)[0])


def fiber_requirements(model, signature, fiber_sig, requirements):
    PROMPT = f"""
Rewrite the given problem, which requires implementing the function {signature}, so that it requires implementing the function {fiber_sig} instead. The new function should return the list of **all** possible inputs of the {signature.inverse_index}th argument that produce that output. 
Enclose your rewritten problem in <answer> tags.

Problem:
{requirements}
    """
    return extract_answer(model.sample(PROMPT)[0])

