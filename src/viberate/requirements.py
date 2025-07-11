from viberate.program import Signature, Parameter
from viberate.llm import extract_answer


def inverse_signature(sig):
    assert len(sig.params) == 1
    return Signature(sig.name + "_inv", [Parameter("x", sig.return_type)], sig.params[0].type)


def inverse_requirements(model, signature, requirements):
    inverted_sig = inverse_signature(signature)
    PROMPT = f"""
Rewrite the problem below to require implementing the inverted
function {inverted_sig} instead of {signature}.  Enclose the new
problem in <answer> tags.

Problem:
{requirements}
    """
    return extract_answer(model.sample(PROMPT)[0])


def fiber_signature(sig):
    assert len(sig.params) == 1
    return Signature(sig.name + "_fib", [Parameter("x", sig.return_type)], "list[" + sig.params[0].type + "]")


def fiber_requirements(model, signature, requirements):
    fiber_sig = fiber_signature(signature)
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
