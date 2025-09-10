import copy
import re
import sys

from viberate.cached_llm import Independent, Repeatable
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
    {checker_sig.pretty_print()}
    
    Please answer in the following format:
    ```python
    ```
    """
    response = next(model.sample(PROMPT))
    # print(response)
    checker_sig.return_type = 'int'
    valid_checker = Program(checker_sig, extract_code(response))
    # print(valid_checker, file=sys.stderr)
    filtered_input = []
    for unchecked_input in input_list:
        if len(unchecked_input) != len(req.signature.params) and len(req.signature.params) == 1:
            unchecked_input = [unchecked_input]
        check_outcome = executor.run(valid_checker, unchecked_input)
        # print(check_outcome)
        match check_outcome:
            case Success(outcome):
                if outcome:
                    filtered_input.append(unchecked_input)
            case _:
                pass
    return filtered_input


def generate_inputs(model, req, executor=None):
    # Define three different types of prompts
    PROMPT_SMALL = f"""Given a problem description and the function signature, 
    create 10 small-scale test inputs to test basic functionality. 
    These inputs should have small integer values and short strings/lists.
    Present each input as a list of parameters. Wrap each input with ``` as below:

    ```
    [input1_param1, input1_param2, ...]
    ```

    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {req.signature.pretty_print()}
    """

    PROMPT_MEDIUM = f"""Given a problem description and the function signature, 
    create 10 medium-scale test inputs to test moderate performance. 
    These inputs should have medium integer values and medium-length strings/lists.
    Present each input as a list of parameters. Wrap each input with ``` as below:

    ```
    [input1_param1, input1_param2, ...]
    ```

    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {req.signature.pretty_print()}
    """

    PROMPT_BOUNDARY = f"""Given a problem description and the function signature, 
    create 10 boundary test inputs to test edge cases and special conditions. 
    These should include minimum/maximum values, empty inputs, and other edge cases.
    Present each input as a list of parameters. Wrap each input with ``` as below:

    ```
    [input1_param1, input1_param2, ...]
    ```

    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {req.signature.pretty_print()}
    """

    # Helper function to create hashable representation of input data
    def make_hashable(data):
        if isinstance(data, (int, float, str, bool, type(None))):
            return data
        elif isinstance(data, list):
            return tuple(make_hashable(item) for item in data)
        elif isinstance(data, dict):
            return tuple(sorted((key, make_hashable(value)) for key, value in data.items()))
        elif isinstance(data, tuple):
            return tuple(make_hashable(item) for item in data)
        else:
            # For other types, use string representation
            return str(data)

    def sample_and_extract_with_retry(prompt, used_model, num_retry=3):
        ind_model = Independent(used_model)
        pattern = r"```(.*?)```"
        inputs = []
        for attempt in range(num_retry):
            try:
                response = next(ind_model.sample(prompt))
                matches = re.findall(pattern, response, re.DOTALL)
                inputs = [eval(block.strip()) for block in matches]
                inputs = [i for i in inputs if not value_is_too_large(i, 10000, 10)]
                break
            except Exception as e:
                if attempt == num_retry - 1:
                    raise Exception(f"did not get good response: {e}")
            pass
        return inputs

    small_inputs = sample_and_extract_with_retry(PROMPT_SMALL, model)
    medium_inputs = sample_and_extract_with_retry(PROMPT_MEDIUM, model)
    boundary_inputs = sample_and_extract_with_retry(PROMPT_BOUNDARY, model)

    # Use range_checker to filter inputs if executor is provided
    if executor:
        small_inputs = range_checker(executor, model, req, small_inputs)
        medium_inputs = range_checker(executor, model, req, medium_inputs)
        boundary_inputs = range_checker(executor, model, req, boundary_inputs)

    # Select 5 inputs from each type
    selected_inputs = []

    # Select small-scale inputs
    selected_small = small_inputs[:5] if len(small_inputs) >= 5 else small_inputs
    selected_inputs.extend(selected_small)

    # Select medium-scale inputs
    selected_medium = medium_inputs[:5] if len(medium_inputs) >= 5 else medium_inputs
    selected_inputs.extend(selected_medium)

    # Select boundary inputs
    selected_boundary = boundary_inputs[:5] if len(boundary_inputs) >= 5 else boundary_inputs
    selected_inputs.extend(selected_boundary)

    # Supplement from remaining inputs if total is less than 15
    if len(selected_inputs) < 15:
        remaining_inputs = []
        if len(selected_small) < 5 and len(small_inputs) > len(selected_small):
            remaining_inputs.extend(small_inputs[len(selected_small):])
        if len(selected_medium) < 5 and len(medium_inputs) > len(selected_medium):
            remaining_inputs.extend(medium_inputs[len(selected_medium):])
        if len(selected_boundary) < 5 and len(boundary_inputs) > len(selected_boundary):
            remaining_inputs.extend(boundary_inputs[len(selected_boundary):])

        # Select enough inputs from remaining pool
        needed = 15 - len(selected_inputs)
        selected_inputs.extend(remaining_inputs[:needed])

    # Remove duplicate inputs and ensure we have 15 unique inputs
    unique_inputs = []
    seen_inputs = set()

    for input_data in selected_inputs:
        # Create hashable representation of the input
        hashable_input = make_hashable(input_data)

        if hashable_input not in seen_inputs:
            seen_inputs.add(hashable_input)
            unique_inputs.append(input_data)

    # If we have duplicates and need to replace them
    if len(unique_inputs) < 15:
        # Collect all available inputs from all categories
        all_available_inputs = small_inputs + medium_inputs + boundary_inputs

        # Find inputs that are not in our current selection
        additional_inputs = []
        for input_data in all_available_inputs:
            hashable_input = make_hashable(input_data)
            if hashable_input not in seen_inputs:
                additional_inputs.append(input_data)
                seen_inputs.add(hashable_input)  # Mark as seen

        # Add enough unique inputs to reach 15
        needed_additional = 15 - len(unique_inputs)
        unique_inputs.extend(additional_inputs[:needed_additional])

    ans = unique_inputs[:15]
    print_annotated_hr(f"Generated {len(ans)} inputs")
    for index, input_data in enumerate(ans):
        print(f"Test input {index}: {input_data}")

    return ans  # Ensure no more than 15 unique inputs are returned
