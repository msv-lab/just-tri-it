import argparse
import sys
from pathlib import Path


from viberate.llm import Cached, AI302, extract_code
import viberate.checker as checker
import viberate.requirements as re_des
from viberate.program import Signature
from viberate.coder import generate_programs
from viberate.executor import Executor, Success
from viberate.tester import generate_tests
from viberate.utils import print_hr


PROMPT_CODE = '''
You are an expert competitive programmer.
Write efficient and clean Python 3 code to solve the following problem. Read input with input(), output with print(). Make sure the code can be submitted directly to an online judge. Output the complete code only, no explanations.
{description}

Please answer in the following format:
```python
```
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=str,
        help="Set LLM cache root directory (default: ~/.viberate_cache/)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input file containing the problem description"
    )
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    return parser.parse_args()


def gen_resonator(model, description, number):
    prompt = PROMPT_CODE.format(description=description)
    response = model.sample(prompt, number)
    code_lst = []
    for ge_code in response:
        code_lst.append(extract_code(ge_code))
    print(code_lst)
    return code_lst


def main_for_inv(executor, model, req, n1, n2):
    sig = Signature.from_requirements(model, req)
    forward_programs = list(generate_programs(model, sig, req, n1))
    inverse_sig = re_des.inverse_signature(sig)
    inverse_req = re_des.inverse_requirements(model, sig, req)
    print(f"the inverse description is: {inverse_req}")
    inverse_programs = list(generate_programs(model, inverse_sig, inverse_req, n2))
    test_inputs = generate_tests(model, req)
    resonating_pairs = []
    for forward in forward_programs:
        for inverse in inverse_programs:
            resonate = True
            for test_input in test_inputs:
                forward_outcome = executor.run(forward, [test_input])
                match forward_outcome:
                    case Success(forward_output):
                        inverse_outcome = executor.run(inverse, [forward_output])
                        match forward_outcome:
                            case Success(inverse_output):
                                if test_input != inverse_output:
                                    resonate = False
            if resonate:
                resonating_pairs.append((forward, inverse))
    return resonating_pairs


def main_for_fib(test_venv, model, description, n1, n3):
    # forward resonator
    print("generating forward resonator:")
    for_code_lst = gen_resonator(model, description, n1)

    print("generating fiber resonator:")
    fiber_des = re_des.gen_fiber_des(model, description)
    print(f"the fiber des is: {fiber_des}")
    print("already generated inverse description")
    fiber_code_lst = gen_resonator(model, fiber_des, n3)
    print(fiber_code_lst)

    cur_test_a = generate_tests(model, description)
    print(cur_test_a)
    cur_test_b = generate_tests(model, fiber_des)
    print(cur_test_b)
    code_decision = []
    for i in range(n1):
        for j in range(n3):
            print(f"checking for forward resonator {i} and inverse resonator {j}")
            if (checker.check_for_fib_l(test_venv, cur_test_b, for_code_lst[i], fiber_code_lst[j])
                    and checker.check_for_fib_r(test_venv, cur_test_a, for_code_lst[i], fiber_code_lst[j])):
                print("find one")
                code_decision.append(for_code_lst[i])
                break
                # unfinished: means this i-th forward resonator is right
    if code_decision:
        print("decision: selection")
        return "selection", code_decision
    else:
        print("decision: abstention")
        return "abstention", None


def main():
    args = parse_args()
    model_name = "gpt-4o"
    chosen_model = AI302(model_name, 1.0)

    if not args.no_cache:
        if args.cache_root:
            chosen_model = Cached(chosen_model, Path(args.cache_root))
        else:
            chosen_model = Cached(chosen_model, Path.home() / ".viberate_cache")

    test_venv = Path(args.test_venv)
    executor = Executor(test_venv)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        des = f.read()

    n1, n2, n3, n4 = 5, 5, 5, 5
    resonating = main_for_inv(executor, chosen_model, des, n1, n2)
    if len(resonating) > 0:
        print_hr()
        print(resonating[0][0].code, file=sys.stderr)
        print_hr()        
        print(resonating[0][1].code, file=sys.stderr)
        print("SELECTED")
    else:
        print("ABSTAIN")

    # print(main_for_fib(test_venv, chosen_model, des, n1, n3))

if __name__ == "__main__":
    main()
