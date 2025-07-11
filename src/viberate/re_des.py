import re
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, retry_if_result

PROMPT_INV = """
Any programming competition problem can be viewed as implementing a mapping algorithm. A problem described in natural language (let's call it R) defines a mapping f: A → B (where f is assumed to be an injection). Now I need your help with a task: Rewrite R into its inverse requirement, i.e., produce the natural language description of the inverse mapping g: B → A.

Here is an example:

Original requirement:
Given a positive number n, please output its square.

Inverse requirement:
Given a number x, please identify the positive number n such that its square is x.

Now, here is the next task:

Original requirement:
{description}

Please answer in the following format:
Inverse requirement:
"""

PROMPT_FIB = """
Any programming competition problem can be viewed as implementing a mapping algorithm. A problem described in natural language (let's call it R) defines a mapping f: A → B (where f is assumed to be an injection). Now I need your help with a task: According to R, write a new requirement where the task is to, given an element from set B, determine all possible preimages from set A.

Here is an example:

Original requirement:
Given n, please output the n-th Fibonacci number.

New requirement:
Given a number x, please identify its index in the Fibonacci sequence. If x corresponds to multiple indices, please output them as a list.（the answer should be formatted as ['','',...], with each answer enclosed in single quotes ('') and all answers wrapped in square brackets ([]), separated by commas）

Original requirement:
{description}

Please answer in the following format:
New requirement:
"""


def is_result_none(result):
    return result is None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=(retry_if_exception_type() | retry_if_result(is_result_none)),
    reraise=False
)
def gen_inverse_des(model, description):
    response = model.sample(PROMPT_INV.format(description=description))
    print(response)
    pattern = r'^Inverse requirement:\s*(.*)$'
    match = re.match(pattern, response[0], re.DOTALL)
    return match.group(1) if match else None


def gen_fiber_des(model, description):
    response = model.sample(PROMPT_FIB.format(description=description))
    print(response)
    pattern = r'^New requirement:\s*(.*)$'
    match = re.match(pattern, response[0], re.DOTALL)
    return match.group(1) if match else None
