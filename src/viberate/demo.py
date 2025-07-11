import argparse
import sys
from pathlib import Path


from viberate.llm import Cached, AI302
from viberate.checker import select_for_inv, select_for_fib_lib
from viberate.utils import print_annotated_hr
from viberate.executor import Executor


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
        requirements = f.read()

    n1, n2 = 5, 5

    resonating = select_for_inv(executor, chosen_model, requirements, n1, n2)
    if len(resonating) > 0:
        print_annotated_hr("Forward function")
        print(resonating[0][0].code, file=sys.stderr)
        print_annotated_hr("Inverse function")
        print(resonating[0][1].code, file=sys.stderr)
        print("FOR-INV: selected")
    else:
        print("FOR-INV: abstain")

    resonating = select_for_fib_lib(executor, chosen_model, requirements, n1, n2)
    if len(resonating) > 0:
        print_annotated_hr("Forward function")
        print(resonating[0][0].code, file=sys.stderr)
        print_annotated_hr("Fiber function")
        print(resonating[0][1].code, file=sys.stderr)
        print("FOR-FIB: selected")
    else:
        print("FOR-FIB: abstain")


if __name__ == "__main__":
    main()
