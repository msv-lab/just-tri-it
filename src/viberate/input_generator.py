import re


def value_is_too_large(data, int_bound, seq_bound):
    if isinstance(data, int):
        if data < -int_bound or data > int_bound:
            return True
    elif isinstance(data, str):
        if len(data) > seq_bound:
            return True
    elif isinstance(data, list):
        if len(data) > seq_bound:
            return True
        for item in data:
            if value_is_too_large(item, int_bound, seq_bound):
                return True
    elif isinstance(data, dict):
        for key, val in data.items():
            if value_is_too_large(val, int_bound, seq_bound):
                return True
    elif isinstance(data, tuple):
        for item in data:
            if value_is_too_large(item, int_bound, seq_bound):
                return True

    return False


def generate_inputs(model, req):
    PROMPT = f""" Given a problem description and the function
    signature, create a set of test inputs to thoroughly cover all key
    functionalities and aspects of the problem. The inputs should be
    presented as lists to match the function signature format. You
    don't need to provide the expected outputs. Only provide test
    cases, no explanations needed. Format the inputs strictly as
    follows:

    ```
    [input1_param1, input1_param2, ...]
    ```

    ```
    [input2_param1, input2_param2, ...]
    ```
    ...

    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {req.signature}
    """
    response = next(model.sample(PROMPT))
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    input_list = [eval(block.strip()) for block in matches]
    input_list = [i for i in input_list if not value_is_too_large(i, 10000, 10)]
    return input_list
