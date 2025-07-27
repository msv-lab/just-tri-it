import ast
import re


def generate_tests_old(model, req):
    PROMPT = f"""
You are a professional program tester. I will provide you with the problem description of a certain competition problem, which will explain the problem content as well as the input constraints. Your task is to generate diverse and well-considered test inputs. These test cases should focus on:

Covering the most important edge cases based on the constraints and problem details.
Ensuring that all key functionalities and aspects of the problem description are thoroughly tested.
The inputs do not necessarily need to be large, but they should be carefully chosen to maximize coverage of edge behaviors, special cases, and typical scenarios that validate the correctness and robustness of the implementation.

You only need to provide the inputs, not the expected outputs.

Please follow the following format.
# **Input 1**
```

```
# **Input 2**
```

```
# **Input 3**
```

```
...
Now I will give you the problem description:
{req}
"""
    response = model.sample(PROMPT)
    pattern = r"# \*\*Input \d+\*\*\n```(.*?)```"
    matches = re.findall(pattern, response[0], re.DOTALL)
    input_list = [block.strip() for block in matches]
    return input_list


def generate_tests(model, req, sig):
    PROMPT = f"""
You are a professional program tester. I will give you a problem description and the function signature from a coding competition. Your job is to create a set of diverse and insightful test inputs. These test cases should:

- Cover important edge cases based on the constraints and problem details.
- Thoroughly test all key functionalities and aspects of the problem.

Focus on maximizing coverage of edge behaviors, special cases, and typical scenarios to validate the correctness and robustness of the implementation. The inputs should be presented as lists to match the function signature format.

You don't need to provide the expected outputs. Only provide test cases, no explanations needed.

Please format the inputs strictly as follows:

# **Input 1**
```
[input1_param1, input1_param2, ...]
```
# **Input 2**
```
[input2_param1, input2_param2, ...]
```
...

Now, here is the problem description and function signature:
# **Problem Description**:
{req}
# **Function Signature**:
{sig}
"""
    response = model.sample(PROMPT)
    pattern = r"# \*\*Input \d+\*\*\n```(.*?)```"
    matches = re.findall(pattern, response[0], re.DOTALL)
    # print(response)
    input_list = [eval(block.strip()) for block in matches]
    return input_list


