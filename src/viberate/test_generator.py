import copy
import sys
from typing import Iterator, List
from abc import ABC, abstractmethod

from viberate.utils import extract_code
from viberate.program import Program, Test, TestFunction, InputOutput, Requirements
from viberate.utils import print_annotated_hr, extract_all_code, ExperimentFailure
from viberate.input_generator import value_is_too_large


MINIMUM_NUM_TESTS = 10
MAXIMUM_NUM_TESTS = 50


class TestGenerator(ABC):

    @abstractmethod
    def generate(self, model, req: Requirements) -> Iterator[Test]:
        pass


class InputOutputGenerator(TestGenerator):

    def generate(self, model, req: Requirements) -> Iterator[Test]:
        tests = self._generate_initial_tests(model, req)

        while len(tests) < MINIMUM_NUM_TESTS:
            additional = self._generate_additional_tests(model, req, tests)
            tests.extend(additional)

            if len(additional) == 0:
                break

        adjusted_tests = []

        for t in tests:
            if len(t.inputs) != len(req.signature.params) and len(req.signature.params) == 1:
                adjusted_inputs = [t.inputs]
            else:
                adjusted_inputs = t.inputs
                
            new_test = InputOutput(adjusted_inputs, t.output)
            adjusted_tests.append(new_test)
            
        print(f"InputOutputGenerator: generated {len(adjusted_tests)} tests", file=sys.stderr)
        return adjusted_tests

    def _extract_test_case(self, blocks):
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
        test_cases = self._extract_test_case(extract_all_code(response))

        return test_cases


    def _generate_additional_tests(self, model, req, existing_cases, executor=None):
        existing_inputs_str = "\n".join([str(case[0]) for case in existing_cases])

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

        additional_cases = self._extract_test_case(extract_all_code(response))
        return additional_cases


class TestFunctionGenerator(TestGenerator):

    def generate(self, model, req: Requirements) -> Iterator[Test]:
        target_signature = req.signature
        PROMPT_ASSERTIONS = f"""For the given problem description,
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

        #FIXME: why in this generator we do not use MINIMUM_NUM_TESTS?
        response = next(model.sample(PROMPT_ASSERTIONS))

        code_blocks = extract_all_code(response)

        tests = []
        for code in code_blocks:
            if code and code.strip():
                try:
                    assertion = TestFunction.from_code(code.strip(), target_signature)
                    tests.append(assertion)
                except ValueError as e:
                    print(f"Failed to create assertion from code: {e}")
                    continue

        if len(tests) == 0:
            raise ExperimentFailure("Failed to generate test functions")

        print(f"TestFunctionGenerator: generated {len(tests)} tests", file=sys.stderr)        

        return tests
