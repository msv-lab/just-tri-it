from viberate.llm import extract_code
from viberate.program import Signature, Program


def generate_programs(model, sig: Signature, requirements: str, k=1) -> list[Program]:
    PROMPT = f""" 
Write a Python function {sig.pretty_print()} to solve the following
problem.  Include all necessary imports. Put the complete code inside
a Markdown code block.

Problem:
{requirements}
"""
    for s in model.sample(PROMPT, k):
        yield Program(sig, extract_code(s))



