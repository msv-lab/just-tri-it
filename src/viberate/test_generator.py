import copy
import re
import sys
from viberate.executor import Success
from viberate.utils import extract_code
from viberate.program import Program
from viberate.utils import print_annotated_hr
from viberate.input_generator import value_is_too_large


def validate_test_case(executor, model, req, test_case):
    validator_sig = copy.deepcopy(req.signature)
    validator_sig.name = "validate_test_case"
    validator_sig.params.append(("expected_output", req.signature.return_type))
    
    PROMPT = f"""
    Given a problem description and a test case (input + expected output), 
    please write a function to validate whether the expected output is 
    correct for the given input according to the problem requirements.
    The function should return True if the test case is valid, False otherwise.
    Name your function 'validate_test_case'.
    
    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {validator_sig}
    
    Please answer in the following format:
    ```python
    ```
    """
    response = next(model.sample(PROMPT))
    print(response, file=sys.stderr)
    validator_sig.return_type = 'bool'
    validator = Program(validator_sig, extract_code(response))
    
    validation_input = list(test_case[0]) + [test_case[1]]
    validation_outcome = executor.run(validator, validation_input)
    
    match validation_outcome:
        case Success(outcome):
            return outcome
        case _:
            return False


def extract_test_case(matches):
    test_cases = []
    for block in matches:
        try:
            lines = block.strip().split('\n')
            input_line = None
            output_line = None

            for line in lines:
                line = line.strip()
                if line.startswith('input:'):
                    input_line = line[6:].strip()
                elif line.startswith('output:'):
                    output_line = line[7:].strip()

            if input_line and output_line:
                test_input = eval(input_line)
                test_output = eval(output_line)

                if not value_is_too_large(test_input, 10000, 10):
                    test_cases.append((test_input, test_output))
        except Exception as e:
            print(f"Error parsing test case: {e}", file=sys.stderr)
            continue
    return test_cases


def generate_test_cases(model, req, executor=None):
    PROMPT = f"""
    Given a problem description and function signature, create a comprehensive 
    set of test cases that include both inputs and their corresponding expected 
    outputs. The test cases should cover various scenarios including:
    - Normal cases
    - Edge cases
    - Boundary conditions
    - Corner cases
    
    Present each test case as: input -> expected_output
    Wrap each test case with ``` like this:
    # Test case 1
    ```
    input: [param1, param2, ...]
    output: expected_result
    ```
    # Test case 2
    ```
    input: [param1, param2, ...]
    output: expected_result
    ```
    ...
    
    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {req.signature}
    
    Please provide at least 5-10 diverse test cases.
    """
    
    response = next(model.sample(PROMPT))
    print(response, file=sys.stderr)
    
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    test_cases = extract_test_case(matches)
    print(f"Generated {len(test_cases)} test cases", file=sys.stderr)
    
    if executor:
        validated_cases = []
        for test_case in test_cases:
            try:
                test_input = test_case[0]
                if len(test_input) != len(req.signature.params) and len(req.signature.params) == 1:
                    test_input = [test_input]
                
                if validate_test_case(executor, model, req, (test_input, test_case[1])):
                    validated_cases.append((test_input, test_case[1]))
            except Exception as e:
                print(f"Error validating test case: {e}", file=sys.stderr)
                continue
        
        print(f"Validated {len(validated_cases)} test cases", file=sys.stderr)
        return validated_cases
    
    return test_cases


def generate_additional_test_cases(model, req, existing_cases, executor=None):
    existing_inputs_str = "\n".join([str(case[0]) for case in existing_cases])
    
    PROMPT = f"""
    Given a problem description and some existing test cases, generate 
    additional diverse test cases that cover different scenarios not 
    already covered. Focus on edge cases, boundary conditions, and 
    corner cases that might be missing.
    
    # **Problem Description**:
    {req.description}
    # **Function Signature**:
    {req.signature}
    
    # **Existing Test Cases (inputs only)**:
    {existing_inputs_str}
    
    Generate 3-5 additional test cases in the same format:
    ```
    input: [param1, param2, ...]
    output: expected_result
    ```
    """
    
    response = next(model.sample(PROMPT))
    
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    additional_cases = extract_test_case(matches)
    return additional_cases


def generate_comprehensive_tests(model, req, executor=None, min_cases=10):
    initial_cases = generate_test_cases(model, req, executor)
    
    all_cases = initial_cases
    while len(all_cases) < min_cases:
        additional = generate_additional_test_cases(model, req, all_cases, executor)
        all_cases.extend(additional)
        
        if len(additional) == 0:
            break
    
    print(f"Final test suite contains {len(all_cases)} test cases", file=sys.stderr)
    return all_cases[:min_cases]