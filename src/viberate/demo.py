import argparse
from pathlib import Path

import viberate.check_property as check_property
import viberate.re_des as re_des
from viberate.llm import Cached, AI302, extract_code

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
    return parser.parse_args()


def gen_resonator(model, description, number):
    prompt = PROMPT_CODE.format(description=description)
    response = model.sample(prompt, number)
    code_lst = []
    for ge_code in response:
        code_lst.append(extract_code(ge_code))
    print(code_lst)
    return code_lst


def main_for_inv(model, description, n1, n2):
    # forward resonator
    print("generating forward resonator:")
    for_code_lst = gen_resonator(model, description, n1)

    # check forward-inverse
    # inverse resonator
    print("generating inverse resonator:")
    inverse_des = re_des.gen_inverse_des(model, description)
    print(f"the inverse des is: {inverse_des}")
    print("already generated inverse description")
    inv_code_lst = gen_resonator(model, inverse_des, n2)

    cur_test = check_property.generate_tests(model, description)
    code_decision = []
    for i in range(n1):
        for j in range(n2):
            print(f"checking for forward resonator {i} and inverse resonator {j}")
            if check_property.check_for_inv(cur_test, for_code_lst[i], inv_code_lst[j]):
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


def main_for_fib(model, description, n1, n3):
    # forward resonator
    print("generating forward resonator:")
    for_code_lst = gen_resonator(model, description, n1)

    print("generating fiber resonator:")
    fiber_des = re_des.gen_fiber_des(model, description)
    print(f"the fiber des is: {fiber_des}")
    print("already generated inverse description")
    fiber_code_lst = gen_resonator(model, fiber_des, n3)
    print(fiber_code_lst)

    cur_test_a = check_property.generate_tests(model, description)
    print(cur_test_a)
    cur_test_b = check_property.generate_tests(model, fiber_des)
    print(cur_test_b)
    code_decision = []
    for i in range(n1):
        for j in range(n3):
            print(f"checking for forward resonator {i} and inverse resonator {j}")
            if (check_property.check_for_fib_l(cur_test_b, for_code_lst[i], fiber_code_lst[j])
                    and check_property.check_for_fib_r(cur_test_a, for_code_lst[i], fiber_code_lst[j])):
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

    with open(args.input_file, 'r', encoding='utf-8') as f:
        des = f.read()

    n1, n2, n3, n4 = 5, 5, 5, 5
    if main_for_inv(chosen_model, des, n1, n2)[1] is None:
        print("check for-inv failed, then check for-fib")
        print(main_for_fib(chosen_model, des, n1, n3))

if __name__ == "__main__":
    main()
