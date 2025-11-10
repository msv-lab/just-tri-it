import sys
import random
from typing import Iterator, List
from abc import ABC, abstractmethod

from just_tri_it.program import Test, TestFunction, InputOutput, Requirements, Signature
from just_tri_it.utils import extract_all_code, ExperimentFailure, check_type, args_match_signature
from just_tri_it.input_generator import value_is_too_large, MAXIMUM_NUM_INPUTS


MINIMUM_NUM_TESTS = 10
MAXIMUM_NUM_TESTS = 50  # the upper bound is less important here


class TestGenerator(ABC):

    @abstractmethod
    def generate(self, model, req: Requirements) -> Iterator[Test]:
        pass

    @abstractmethod
    def display_name(self) -> str:
        pass


class InputOutputGenerator(TestGenerator):

    def display_name(self) -> str:
        return "IO"

    def generate(self, model, req: Requirements) -> Iterator[Test]:
        tests = self._generate_initial_tests(model, req)
        tests = self._fix_and_filter_bad_tests(tests, req.signature)

        max_attempts = 10
        attempt = 0
        
        while len(tests) < MINIMUM_NUM_TESTS and attempt < max_attempts:
            attempt += 1
            additional = self._generate_additional_tests(model, req, tests)
            additional = self._fix_and_filter_bad_tests(additional, req.signature)
            tests.extend(additional)

        if len(tests) < MINIMUM_NUM_TESTS:
            raise ExperimentFailure(f"only generated {len(tests)} tests after {max_attempts} attempts (target: {MINIMUM_NUM_TESTS})")

        if len(tests) > MAXIMUM_NUM_TESTS:
            return random.sample(tests, MAXIMUM_NUM_INPUTS)

        return tests

    def _fix_and_filter_bad_tests(self, tests: List[InputOutput], sig: Signature):
        adjusted_tests = []

        for t in tests:
            unchecked_input = t.inputs
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

            if args_match_signature(unchecked_input, sig) and \
               check_type(t.output, sig.return_type):
                adjusted_tests.append(InputOutput(unchecked_input, t.output))
            
        return adjusted_tests

    def _extract_test_cases(self, blocks):
        test_cases = []
        for block in blocks:
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
                        test_cases.append(InputOutput(test_input, test_output))
            except Exception as e:
                print(f"Error parsing test case: {e}", file=sys.stderr)
                continue
        return test_cases


    def _generate_initial_tests(self, model, req: Requirements) -> List:
        PROMPT = f"""Given a problem description and function signature,
create a comprehensive set of test cases that cover various scenarios
including normal cases and edge cases. Present each tests inside a
separate code block which contains the input as a list of
function arguments and the expected output:
```
input: [argument1, argument2, ...]
output: expected_output
```

```
input: [argument1, argument2, ...]
output: expected_output
```
...

# **Function Signature**:
{req.signature}

# **Problem Description**:
{req.description}
        """

        response = next(model.sample(PROMPT))
        test_cases = self._extract_test_cases(extract_all_code(response))

        return test_cases


    def _generate_additional_tests(self, model, req, existing_cases):
        existing_inputs_str = "\n".join([repr(c.inputs) for c in existing_cases])

        PROMPT = f"""Given a problem description, a function
signature, and a list of existing test inputs, generate additional
tests that cover missing cases. Present each tests inside a separate
code block which contains the input as a list of function arguments
and the expected output:

```
input: [argument1, argument2, ...]
output: expected_output
```

```
input: [argument1, argument2, ...]
output: expected_output
```

# **Function Signature**:
{req.signature}

# **Problem Description**:
{req.description}

# **Existing Test Cases (inputs only)**:
{existing_inputs_str}

Output only new tests.
        """
        response = next(model.sample(PROMPT))

        additional_cases = self._extract_test_cases(extract_all_code(response))
        return additional_cases


class TestFunctionGenerator(TestGenerator):

    def display_name(self) -> str:
        return "assert"

    def _extract_test_cases(self, blocks, target_signature):
        tests = []
        for code in blocks:
            try:
                assertion = TestFunction.from_code(code.strip(), target_signature)
                tests.append(assertion)
            except Exception as e:
                print(f"Error parsing test case: {e}", file=sys.stderr)
                continue
        return tests

    def _generate_additional_tests(self, model, req, existing_cases):
        target_signature = req.signature
        existing_tests_str = "\n".join([c.test_function_code for c in existing_cases])

        PROMPT = f"""Given a problem description, a function
signature, and a list of existing tests, generate additional
tests that cover missing cases.

The signature of the target function under test is
{target_signature.pretty_print()}. Each test should be a function
whose name starts with test_, it calls the target function, and
contains assertions to check output correctness. Account for cases
when the problem description allows multiple correct outputs for the
same input if such cases exists.
        
Problem description:
{req.description}

Output each test function in a separate code block, for example

```python
def test_basic_functionality():
    result = {target_signature.name}(typical_input)
    assert result == expected_value
```

```python  
def test_boundary_condition():
    result = {target_signature.name}(edge_case_input)
    assert some_property_holds(result)
```        

# **Existing Test Cases**:
{existing_tests_str}

Output only new tests.
        """
        response = next(model.sample(PROMPT))
        tests = self._extract_test_cases(extract_all_code(response), target_signature)
        return tests
    

    def _generate_initial_tests(self, model, req: Requirements) -> Iterator[Test]:
        target_signature = req.signature
        PROMPT = f"""For the given problem description,
write a comprehensive set of tests. The signature of the target
function under test is {target_signature.pretty_print()}. Each test
should be a function whose name starts with test_, it calls the
target function, and contains assertions to check output
correctness. Account for cases when the problem description allows
multiple correct outputs for the same input if such cases exists.
        
Problem description:
{req.description}

Output each test function in a separate code block, for example

```python
def test_basic_functionality():
    result = {target_signature.name}(typical_input)
    assert result == expected_value
```

```python  
def test_boundary_condition():
    result = {target_signature.name}(edge_case_input)
    assert some_property_holds(result)
```
        """
        response = next(model.sample(PROMPT))
        tests = self._extract_test_cases(extract_all_code(response), target_signature)
        return tests
    

    def generate(self, model, req: Requirements) -> Iterator[Test]:
        target_signature = req.signature
        tests = self._generate_initial_tests(model, req)

        max_attempts = 10
        attempt = 0
        
        while len(tests) < MINIMUM_NUM_TESTS and attempt < max_attempts:
            attempt += 1
            additional = self._generate_additional_tests(model, req, tests)
            tests.extend(additional)

        if len(tests) < MINIMUM_NUM_TESTS:
            raise ExperimentFailure(f"only generated {len(tests)} tests after {max_attempts} attempts (target: {MINIMUM_NUM_TESTS})")

        if len(tests) > MAXIMUM_NUM_TESTS:
            return random.sample(tests, MAXIMUM_NUM_INPUTS)

        return tests
