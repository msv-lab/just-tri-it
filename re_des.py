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

PROMPT_INV_WZ_SIG = """
You are given the following information:

1. An original competitive programming problem description.
2. The Python function signature and comments for the inverse function of the original problem.

Your task is to write a **new problem description**, describing the inverse problem. This problem description should follow the same style and clarity as competitive programming platforms like Codeforces or LeetCode.

**Key requirements:**

* The description should clearly describe the **inverse task**, meaning what the inverse function computes.
* You may reuse context from the original problem description where appropriate, but you should adjust the story or technical description to fit the **inverse perspective**.
* Do **not refer to the original problem or the word "inverse" in the description itself**. Write the description as if this is an independent problem.
* Make sure the description matches the input and output described in the inverse function's comments.
* Keep the tone formal, clear, and concise, suitable for a programming contest.

Now I'll give you the task.

Original requirement:
{description}

Inverse function signature:
{inv_func_sig}

Please answer in the following format:
New requirement:
"""

PROMPT_FIB_WZ_SIG = """
You are given the following information:

1. An original competitive programming problem description.
2. The Python function signature and comments for the **fiber function** corresponding to the original problem.

Your task is to write a **new problem description**, describing the fiber task. This problem description should follow the same style and clarity as competitive programming platforms like Codeforces or LeetCode.

**Key requirements:**

* The description should clearly describe the **fiber task**, meaning what the fiber function computes.
* You may reuse context from the original problem description where appropriate, but you should adjust the story or technical description to fit the **fiber perspective**. Specifically, this task involves finding all possible original input values that would result in the given output.
* Do **not refer to the original problem or the term "fiber function"** in the description itself. Write the description as if this is an independent problem.
* Make sure the description matches the input and output described in the fiber function's comments.
* Keep the tone formal, clear, and concise, suitable for a programming contest.

Now I'll give you the task.

Original requirement:
{description}

Fiber function signature:
{fib_func_sig}

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


def gen_inverse_des_new(model, description, inv_func_sig):
    response = model.sample(PROMPT_INV_WZ_SIG.format(description=description, inv_func_sig=inv_func_sig))
    print(response)
    pattern = r'^New requirement:\s*(.*)$'
    match = re.match(pattern, response[0], re.DOTALL)
    return match.group(1) if match else None


def gen_fiber_des_new(model, description, fib_func_sig):
    response = model.sample(PROMPT_FIB_WZ_SIG.format(description=description, fib_func_sig=fib_func_sig))
    print(response)
    pattern = r'^New requirement:\s*(.*)$'
    match = re.match(pattern, response[0], re.DOTALL)
    return match.group(1) if match else None
