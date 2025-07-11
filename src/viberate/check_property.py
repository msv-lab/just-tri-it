import ast
import os
import re
import subprocess

TEST_PROMPT = """
You are a professional program tester. I will provide you with the problem description of a certain competition problem, which will explain the problem content as well as the input constraints. Your task is to generate {number} diverse and well-considered test inputs. These test cases should focus on:

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
{description}
"""


def run_code(code, input_data):
    temp_file = "temp_solution.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        process = subprocess.Popen(
            ['python3', temp_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=input_data, timeout=5)
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.remove(temp_file)


def generate_tests(model, description):
    response = model.sample(TEST_PROMPT.format(description=description, number=20))
    print("got the original response")
    pattern = r"# \*\*Input \d+\*\*\n```(.*?)```"
    matches = re.findall(pattern, response[0], re.DOTALL)
    input_list = [block.strip() for block in matches]
    print(f"\nExtracted {len(input_list)}  test inputs!\n")
    return input_list


def run_for_inv_test(test_input, for_resonator, inv_resonator):
    # run through stdin
    print(f"input is {test_input}")
    output_result = run_code(for_resonator, test_input)
    print(f"output is {output_result}")
    input_result = run_code(inv_resonator, output_result)
    print(f"inverse_input is {input_result}")
    if input_result == test_input:
        return True
    else:
        return False


def check_for_inv(test_inputs, for_resonator, inv_resonator):
    print("start to generate test inputs")
    for idx, item in enumerate(test_inputs, start=1):
        print(f"Running on test {idx}")
        if run_for_inv_test(item, for_resonator, inv_resonator) is False:
            print(f"Failed on test {idx}")
            return False
    return True


def run_for_fib_test_l(test_output, for_resonator, fib_resonator):
    print(f"output is {test_output}")
    input_results_lst = run_code(fib_resonator, test_output)
    print(input_results_lst)
    input_results_lst = ast.literal_eval(input_results_lst)
    print(input_results_lst)
    for idx, input_result in enumerate(input_results_lst):
        print(f"input {idx} is {input_result}")
        output_result = run_code(for_resonator, input_result)
        print(output_result)
        if output_result != test_output:
            return False
    return True


def check_for_fib_l(test_outputs, for_resonator, fib_resonator):
    for idx, item in enumerate(test_outputs, start=1):
        # for each b
        print(f"Running on test {idx}")
        if run_for_fib_test_l(item, for_resonator, fib_resonator) is False:
            return False
    return True


def run_for_fib_test_r(test_input, for_resonator, fib_resonator):
    print(f"input is {test_input}")
    output_result = run_code(for_resonator, test_input)
    input_results_lst = ast.literal_eval(run_code(fib_resonator, output_result))
    if test_input in input_results_lst:
        return True
    else:
        return False


def check_for_fib_r(test_inputs, for_resonator, fib_resonator):
    for idx, item in enumerate(test_inputs, start=1):
        # for each a
        print(f"Running on test {idx}")
        if run_for_fib_test_r(item, for_resonator, fib_resonator) is False:
            return False
    return True

