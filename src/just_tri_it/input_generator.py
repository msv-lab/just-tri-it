import copy
import sys
import random
from typing import List, Any
from just_tri_it.type_checker import type_checker
from just_tri_it.cached_llm import Independent
from just_tri_it.executor import Executor, Success
from just_tri_it.utils import extract_code
from just_tri_it.program import Program, Requirements
from just_tri_it.utils import extract_all_code, remove_duplicates, ExperimentFailure


#FIXME: control number of inputs:
MINIMUM_NUM_INPUTS = 15
MAXIMUM_NUM_INPUTS = 25


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
        for _, val in data.items():
            if value_is_too_large(val, int_bound, seq_bound):
                return True
    elif isinstance(data, tuple):
        for item in data:
            if value_is_too_large(item, int_bound, seq_bound):
                return True
    return False


def input_checker(req, input_list):
    filtered_input = []
    param_list = req.signature.params
    # print(param_list, file=sys.stderr, flush=True)
    for unchecked_input in input_list:
        # print(unchecked_input, file=sys.stderr, flush=True)
        if not isinstance(unchecked_input, list):
            if len(param_list) == 1:
                unchecked_input = [unchecked_input]
            else:
                continue
        else:
            if len(unchecked_input) != len(param_list):
                if len(param_list) == 1:
                    unchecked_input = [unchecked_input]
                else:
                    continue
        type_error = True
        for index in range(len(param_list)):
            # print(unchecked_input[index], param_list[index].type, file=sys.stderr, flush=True)
            if not type_checker(unchecked_input[index], param_list[index].type):
                type_error = False
                break
        if type_error:
            filtered_input.append(unchecked_input)
    return filtered_input


def generate_inputs(model, req: Requirements, max_attempts: int = 3) -> List[Any]:
    # Define three different types of prompts
    PROMPT_SMALL = f"""Given a problem description and the function
signature, generate a comprehensive set of small-scale test cases to
verify basic functionality. Each test should contain simple, minimal
values such as small integers, short strings or lists, depending on
the parameter types. Present each input as a list of function
arguments inside a separate Markdown code block:
```
[argument1, argument2, ...]
```

# **Function Signature**:
{req.signature.pretty_print()}

# **Problem Description**:
{req.description}
    """

    PROMPT_MEDIUM = f"""Given a problem description and the function
signature, generate a comprehensive set of medium-scale test cases.
Each test case should use values that are not trivial like 0 or empty
string, but still manageable to read and reason about, e.g., medium
integers, medium-length strings and lists, depending on parameter
types. Present each input as a list of function arguments inside a
separate Markdown code block:
```
[argument1, argument2, ...]
```

# **Function Signature**:
{req.signature.pretty_print()}

# **Problem Description**:
{req.description}
    """

    PROMPT_BOUNDARY = f"""Given a problem description and the function
signature, generate a comprehensive set of boundary test cases to
verify edge cases and special conditions.  Tests may include minimum
and maximum allowed values, empty inputs where applicable, and unusual
or corner-case scenarios that could cause unexpected behavior. Present
each input as a list of function arguments inside a separate
Markdown code block:
```
[argument1, argument2, ...]
```

# **Function Signature**:
{req.signature.pretty_print()}
    
# **Problem Description**:
{req.description}
    """

    def sample_and_extract_with_retry(prompt, used_model, num_retry=3):
        inputs = []
        for attempt in range(num_retry):
            try:
                response = next(ind_model.sample(prompt, num_retry))
                blocks = extract_all_code(response)
                inputs = [eval(block.strip()) for block in blocks]
                inputs = [i for i in inputs if not value_is_too_large(i, 10000, 100)]
                break
            except Exception as e:
                if attempt == num_retry - 1:
                    raise ExperimentFailure(f"reply for input generation failed: {e}")
            pass
        return inputs

    all_inputs = []
    attempts = 0
    ind_model = Independent(model)
    while len(all_inputs) < MINIMUM_NUM_INPUTS and attempts < max_attempts:
        attempts += 1
        small_inputs = sample_and_extract_with_retry(PROMPT_SMALL, ind_model)
        medium_inputs = sample_and_extract_with_retry(PROMPT_MEDIUM, ind_model)
        boundary_inputs = sample_and_extract_with_retry(PROMPT_BOUNDARY, ind_model)

        small_inputs = input_checker(req, small_inputs)
        medium_inputs = input_checker(req, medium_inputs)
        boundary_inputs = input_checker(req, boundary_inputs)

        current_batch = []
        current_batch.extend(small_inputs)
        current_batch.extend(medium_inputs)
        current_batch.extend(boundary_inputs)
        
        current_batch = remove_duplicates(current_batch)
        
        all_inputs.extend(current_batch)
        all_inputs = remove_duplicates(all_inputs)
        
        # print(f"Attempt {attempts}: Generated {len(current_batch)} new inputs, total unique inputs: {len(all_inputs)}")

    if len(all_inputs) < MINIMUM_NUM_INPUTS:
        raise ExperimentFailure(f"Warning: Only generated {len(all_inputs)} unique inputs after {max_attempts} attempts (target: {MINIMUM_NUM_INPUTS})")
     
    if len(all_inputs) > MAXIMUM_NUM_INPUTS:
        return random.sample(all_inputs, MAXIMUM_NUM_INPUTS)

    return all_inputs