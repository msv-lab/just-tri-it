import copy
from typing import List, Any

from just_tri_it.cached_llm import Independent
from just_tri_it.executor import Executor, Success
from just_tri_it.utils import extract_code
from just_tri_it.program import Program, Requirements
from just_tri_it.utils import extract_all_code, remove_duplicates, ExperimentFailure


#FIXME: control number of inputs:
MINIMUM_NUM_INPUTS = 10
MAXIMUM_NUM_INPUTS = 50


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


def range_checker(executor, model, req, input_list):
    checker_sig = copy.deepcopy(req.signature)
    checker_sig.name = "is_valid_input"
    checker_sig.return_type = "bool"
    PROMPT = f"""A program input is valid if the problem description
specifies how the program must behave on this input. Otherwise, it is
invalid. Given a problem description, please write a function named
'is_valid_input' with the signature below that checks if the given
input is valid. Put the complete code inside a Markdown code block:
```python
```

# **Function Signature**:
{checker_sig.pretty_print()}

# **Problem Description**:
{req.description}

    """
    response = next(model.sample(PROMPT))
    valid_checker = Program(checker_sig, extract_code(response))
    filtered_input = []
    for unchecked_input in input_list:
        if len(unchecked_input) != len(req.signature.params) and len(req.signature.params) == 1:
            unchecked_input = [unchecked_input]
        check_outcome = executor.run(valid_checker, unchecked_input)
        match check_outcome:
            case Success(outcome):
                if outcome:
                    filtered_input.append(unchecked_input)
            case _:
                pass
    return filtered_input


def generate_inputs(model, req: Requirements, executor: Executor) -> List[Any]:
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
        ind_model = Independent(used_model)
        inputs = []
        for attempt in range(num_retry):
            try:
                response = next(ind_model.sample(prompt, num_retry))
                blocks = extract_all_code(response)
                inputs = [eval(block.strip()) for block in blocks]
                inputs = [i for i in inputs if not value_is_too_large(i, 10000, 10)]
                break
            except Exception as e:
                if attempt == num_retry - 1:
                    raise ExperimentFailure(f"reply for input generation failed: {e}")
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

    all_inputs = []
        
    all_inputs.extend(small_inputs)
    all_inputs.extend(medium_inputs)
    all_inputs.extend(boundary_inputs)

    return remove_duplicates(all_inputs)
