import llm
import re_des
from llm import AI302

PROMPT_SIG = """
You are given a competitive programming problem description.
Your task is to infer and generate an appropriate Python function signature to solve the problem.
Follow these instructions carefully:

1. Add **Python comments above the function**, explaining:
   * **Each input parameter:** its meaning, data type, and value constraints (if any).
   * **The return value:** its meaning and data type.
2. Use **type hints** in the function signature (e.g., `int`, `List[int]`, `str`, etc.).
3. Do **not write the full implementation**, only the **function signature and comments**.

Example output format:

```python
# Parameters:
#   a (int): The first integer number (1 <= a <= 10^9).
#   b (int): The second integer number (1 <= b <= 10^9).
# Returns:
#   int: The sum of a and b.
def solve(a: int, b: int) -> int:
    pass
```

Now, read the following problem description and generate the function signature with comments.

{description}
"""

PROMPT_INV_SIG = """
You are given the Python function signature and its documentation comments for a function that solves a computational problem.
Your task is to write the **function signature of the inverse function**, which reverses the behavior of the original function.

Please follow these instructions carefully:

1. **Assume the original function is mathematically or logically invertible**.
2. Use **Python type hints** (`int`, `List[int]`, etc.) in the function signature.
3. Write **Python comments above the function**, explaining:
   * Each input parameter (these are the original function's return values): its meaning, data type, and constraints (if any).
   * The return value: its meaning and data type (these are the original function's input parameters).
4. **Only output the inverse function's signature and comments. Do not implement the function.**
Example input:
```python
# Parameters:
#   n (int): The starting number (1 <= n <= 10^6).
# Returns:
#   int: The smallest prime number that is greater than or equal to n.
def solve(n: int) -> int:
    pass
```

Example output:
```python
# Parameters:
#   p (int): A prime number (2 <= p <= 10^6).
# Returns:
#   int: The smallest integer n such that p is the smallest prime >= n.
def solve(p: int) -> int:
    pass
```

Now, generate the inverse function signature for the following original function:
{origin_sig}
"""

PROMPT_FIB_SIG = """
You are given the Python function signature and its documentation comments for a function that solves a computational problem.
Your task is to write the **function signature of the fiber function**, which reverses the behavior of the original function.

Please follow these instructions carefully:

1. **Assume the original function is mathematically or logically invertible**, but note that it may not be one-to-one. That is, a single output from the original function may correspond to multiple possible inputs.
2. Use **Python type hints** (`int`, `List[int]`, etc.) in the function signature.
3. Write **Python comments above the function**, explaining:
   * Each input parameter (these are the original function's return values): its meaning, data type, and constraints (if any).
   * The return value: a **list** containing all possible original inputs, including their meaning and data type.
4. **Only output the fiber function's signature and comments. Do not implement the function.**

Example input:
```python
# Parameters:
#   n (int): An integer number (-10^6 <= n <= 10^6).
# Returns:
#   int: The square of n.
def solve(n: int) -> int:
    pass
````

Example output:

```python
# Parameters:
#   s (int): The square of some integer (-10^6 <= n <= 10^6), so s >= 0.
# Returns:
#   List[int]: A list of all integers n such that n * n == s.
def reverse_solve(s: int) -> List[int]:
    pass
```

Now, generate the fiber function signature for the following original function:
{origin_sig}
"""


def gen_signature(model, description):
    prompt = PROMPT_SIG.format(description=description)
    response = model.sample(prompt)[0]
    func_sign = llm.extract_code(response)
    return func_sign


def gen_inv_signature(model, origin_sig):
    prompt = PROMPT_INV_SIG.format(origin_sig=origin_sig)
    response = model.sample(prompt)[0]
    inv_func_sign = llm.extract_code(response)
    return inv_func_sign


def gen_fib_signature(model, origin_sig):
    prompt = PROMPT_FIB_SIG.format(origin_sig=origin_sig)
    response = model.sample(prompt)[0]
    fib_func_sign = llm.extract_code(response)
    return fib_func_sign


if __name__ == "__main__":
    model_name = "gpt-4o"
    chosen_model = AI302(model_name, 1.0)
    # with open("test_prob2.md", 'r', encoding='utf-8') as f:
    #     des = f.read()
    # sig = gen_signature(chosen_model, des)
    # print(sig)
    # inv_sig = gen_inv_signature(chosen_model, sig)
    # print(inv_sig)
    # inv_des = re_des.gen_inverse_des_new(chosen_model, des, inv_sig)
    # print(inv_des)

    with open("test_prob1.md", 'r', encoding='utf-8') as f:
        des = f.read()
    sig = gen_signature(chosen_model, des)
    print(sig)
    fib_sig = gen_fib_signature(chosen_model, sig)
    print(fib_sig)
    fib_des = re_des.gen_fiber_des_new(chosen_model, des, fib_sig)
    print(fib_des)
