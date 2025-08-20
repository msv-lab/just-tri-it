import copy
import re
import sys

from viberate.executor import Success
from viberate.utils import extract_code
from viberate.program import Program
from viberate.utils import print_annotated_hr


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


def range_checker(executor, model, req, input_list):
    checker_sig = copy.deepcopy(req.signature)
    checker_sig.name = "is_valid_input"
    PROMPT = f"""
    Given a problem description, please write a function based on the
    provided function signature to check whether a input meets the range
    and format requirements described in the problem. The function 
    should return True/False, indicating whether the input is valid. 
    You only need to check the explicitly mentioned data range 
    constraints and formats, and do not need to verify the specific
    logical relationships between various data points. 
    Name your function 'is_valid_input'.
    
    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {checker_sig}
    
    Please answer in the following format:
    ```python
    ```
    """
    response = next(model.sample(PROMPT))
    print(response)
    checker_sig.return_type = 'int'
    valid_checker = Program(checker_sig, extract_code(response))
    # print(valid_checker, file=sys.stderr)
    filtered_input = []
    for unchecked_input in input_list:
        if len(unchecked_input) != len(req.signature.params) and len(req.signature.params) == 1:
            unchecked_input = [unchecked_input]
        check_outcome = executor.run(valid_checker, unchecked_input)
        print(check_outcome)
        match check_outcome:
            case Success(outcome):
                if outcome:
                    filtered_input.append(unchecked_input)
            case _:
                pass
    return filtered_input


def generate_inputs(model, req, executor=None):
    PROMPT = f""" Given a problem description and the function
    signature, create a set of test inputs to thoroughly cover all key
    functionalities and aspects of the problem. The inputs should be
    presented as lists to match the function signature format. You
    don't need to provide the expected outputs. Only provide test
    cases, no explanations needed. Wrap each of your answer with ```
    like this:

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
    # print(response)
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    input_list = [eval(block.strip()) for block in matches]
    input_list = [i for i in input_list if not value_is_too_large(i, 10000, 10)]
    print(input_list, file=sys.stderr)
    if executor:
        input_list = range_checker(executor, model, req, input_list)
    return input_list
