import ast
import os
import re
import subprocess


def run_code(test_venv, code, input_data):
    temp_file = "temp_solution.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        interpreter = str(test_venv / 'bin' / 'python')
        process = subprocess.Popen(
            [interpreter, temp_file],
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


def run_for_inv_test(test_venv, test_input, for_resonator, inv_resonator):
    # run through stdin
    print(f"input is {test_input}")
    output_result = run_code(test_venv, for_resonator, test_input)
    print(f"output is {output_result}")
    input_result = run_code(test_venv, inv_resonator, output_result)
    print(f"inverse_input is {input_result}")
    if input_result == test_input:
        return True
    else:
        return False


def check_for_inv(test_venv, test_inputs, for_resonator, inv_resonator):
    print("start to generate test inputs")
    for idx, item in enumerate(test_inputs, start=1):
        print(f"Running on test {idx}")
        if run_for_inv_test(test_venv, item, for_resonator, inv_resonator) is False:
            print(f"Failed on test {idx}")
            return False
    return True


def run_for_fib_test_l(test_venv, test_output, for_resonator, fib_resonator):
    print(f"output is {test_output}")
    input_results_lst = run_code(test_venv, fib_resonator, test_output)
    print(input_results_lst)
    input_results_lst = ast.literal_eval(input_results_lst)
    print(input_results_lst)
    for idx, input_result in enumerate(input_results_lst):
        print(f"input {idx} is {input_result}")
        output_result = run_code(test_venv, for_resonator, input_result)
        print(output_result)
        if output_result != test_output:
            return False
    return True


def check_for_fib_l(test_venv, test_outputs, for_resonator, fib_resonator):
    for idx, item in enumerate(test_outputs, start=1):
        # for each b
        print(f"Running on test {idx}")
        if run_for_fib_test_l(test_venv, item, for_resonator, fib_resonator) is False:
            return False
    return True


def run_for_fib_test_r(test_venv, test_input, for_resonator, fib_resonator):
    print(f"input is {test_input}")
    output_result = run_code(test_venv, for_resonator, test_input)
    input_results_lst = ast.literal_eval(run_code(test_venv, fib_resonator, output_result))
    if test_input in input_results_lst:
        return True
    else:
        return False


def check_for_fib_r(test_venv, test_inputs, for_resonator, fib_resonator):
    for idx, item in enumerate(test_inputs, start=1):
        # for each a
        print(f"Running on test {idx}")
        if run_for_fib_test_r(test_venv, item, for_resonator, fib_resonator) is False:
            return False
    return True

