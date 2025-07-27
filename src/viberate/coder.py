from viberate.llm import extract_code
from viberate.program import Signature, Program


def generate_programs(model, sig: Signature, requirements: str, k=1) -> list[Program]:
    PROMPT = f""" 
Write a Python function {sig.pretty_print()} to solve the following
problem.  Include all necessary imports. Put the complete code inside
a Markdown code block. Please generate the program by implementing only the function, 
without using if __name__ == "__main__": or any code outside the function.
Please ensure the generated code returns pure Python types. If there are requirements 
for the output format in the problem description, please be sure to format it correctly.

Problem:
{requirements}
"""
    for s in model.sample(PROMPT, k):
        yield Program(sig, extract_code(s))



