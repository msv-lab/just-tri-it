import copy
import sys
import random
from typing import List, Any
from just_tri_it.cached_llm import Independent
from just_tri_it.executor import Executor, Success
from just_tri_it.utils import extract_code
from just_tri_it.program import Program, Requirements, Signature
from just_tri_it.utils import extract_all_code, remove_duplicates, ExperimentFailure, args_match_signature
from just_tri_it.inverse_config import config


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

def trans_to_list(executor, call_string, func_name):
    sig = Signature("f", [], "list")
    to_list_code = f"""
def {func_name}(*args):
   return list(args)
def f():
   return {call_string}
    """
    to_list = Program(sig, to_list_code)
    outcome = executor.run(to_list, [])
    match outcome:
        case Success(outcome):
            return outcome
        case _:
            raise ValueError(f"can't transform {call_string} to list due to {outcome}")


def fix_and_filter_bad_inputs(input_list: List[Any], sig: Signature):
    filtered_input = []
    for unchecked_input in input_list:
        if not isinstance(unchecked_input, list):
            if len(sig.params) == 1:
                unchecked_input = [unchecked_input]
            else:
                continue
        else:
            if len(unchecked_input) != len(sig.params):
                if len(sig.params) == 1:
                    unchecked_input = [unchecked_input]
                else:
                    continue
        if args_match_signature(unchecked_input, sig):
            filtered_input.append(unchecked_input)
    return filtered_input


def range_checker(executor, model, req, input_list):
    checker_sig = copy.deepcopy(req.signature)
    checker_sig.name = "is_valid_input"
    checker_sig.return_type = "bool"
    PROMPT = f""" Given a problem description, please write a simple function named
'is_valid_input' with the signature below that checks two key points:
1. Whether the problem's data is self-consistent - for example, if it mentions that three elements will be provided, then exactly three elements should follow.
2. Whether the problem's data falls within the required range specified by the problem.
Put the complete code inside a Markdown code block:
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
        fined_input = unchecked_input
        if not isinstance(fined_input, list):
            fined_input = [fined_input]
        elif len(fined_input) != len(req.signature.params) and len(req.signature.params) == 1:
            fined_input = [fined_input]
        check_outcome = executor.run(valid_checker, fined_input)
        match check_outcome:
            case Success(outcome):
                if outcome:
                    filtered_input.append(fined_input)
            case _:
                pass
    return filtered_input


def generate_inputs(model, req: Requirements, executor,
                    gen_large=False,
                    min_inputs=MINIMUM_NUM_INPUTS,
                    max_inputs=MAXIMUM_NUM_INPUTS) -> List[Any]:
    # Define three different types of prompts
    PROMPT_SMALL = f"""Given a problem description and the function
signature, generate a comprehensive set of small-scale test cases to
verify basic functionality. Each test should contain simple, minimal
values such as small integers, short strings or lists, strictly
confirming to the parameter types. Do not write any comments. Present
each input as a function call inside a separate Markdown code block:

```
target_function(argument1, argument2, ...)
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
integers, medium-length strings and lists, strictly confirming to the
parameter types. Do not write any comments. Present each input as a
function call inside a separate Markdown code block:

```
target_function(argument1, argument2, ...)
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
or corner-case scenarios that could cause unexpected behavior. Do not
write any comments. Present each input as a function call inside a
separate Markdown code block:

```
target_function(argument1, argument2, ...)
```

# **Function Signature**:
{req.signature.pretty_print()}
    
# **Problem Description**:
{req.description}
    """

    def sample_and_extract_with_retry(prompt, used_model, func_name, num_retry=3):
        inputs = []
        for attempt in range(num_retry):
            try:
                response = next(ind_model.sample(prompt, num_retry))
                blocks = extract_all_code(response)
                inputs = [ trans_to_list(executor, block.strip(), func_name) for block in blocks]
                if not gen_large:
                    seq_bound = 20
                    flag, bound_spec = config("bound_spec")
                    if flag:
                        seq_bound = bound_spec
                    inputs = [i for i in inputs if not value_is_too_large(i, 10000, seq_bound)]
                if gen_large or config("enable_filter"):
                    inputs = range_checker(executor, model, req, inputs)
                break
            except Exception as e:
                if attempt == num_retry - 1:
                    raise ExperimentFailure(f"reply for input generation failed: {e}")
            pass
        return inputs

    all_inputs = []
    
    ind_model = Independent(model)

    max_attempts = 10
    attempt = 0
    
    while len(all_inputs) < min_inputs and attempt < max_attempts:
        attempt += 1

        current_batch = []
        current_batch.extend(sample_and_extract_with_retry(PROMPT_SMALL, ind_model, req.signature.name))
        current_batch.extend(sample_and_extract_with_retry(PROMPT_MEDIUM, ind_model, req.signature.name))
        current_batch.extend(sample_and_extract_with_retry(PROMPT_BOUNDARY, ind_model, req.signature.name))

        current_batch = fix_and_filter_bad_inputs(current_batch, req.signature)
        flag, spec_in = config("in_spec")
        if flag and spec_in in current_batch:
            current_batch = [t for t in current_batch if t != spec_in]

        all_inputs.extend(current_batch)
        all_inputs = remove_duplicates(all_inputs)
        
    if len(all_inputs) < min_inputs:
        raise ExperimentFailure(f"only generated {len(all_inputs)} unique inputs after {max_attempts} attempts (target: {min_inputs})")
     
    if len(all_inputs) > max_inputs:
        return random.sample(all_inputs, max_inputs)

    return all_inputs
