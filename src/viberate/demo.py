import argparse
from pathlib import Path


from viberate.cached_llm import Persistent, AI302
from viberate.code_generator import Vanilla
from viberate.new_version import VibeRate
from viberate.executor import Executor
from viberate.requirements import Requirements


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=str,
        help="Set LLM cache root directory (default: ~/.viberate_cache/)."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache."
    )
    parser.add_argument(
        "--replicate",
        action="store_true",
        help="Use cache only."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input file containing the problem description."
    )
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    parser.add_argument(
        "--export-cache",
        type=str,
        help="Explore all responsed generated during the run."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = "gpt-4o"
    model = AI302(model_name, 1.0)

    if not args.no_cache:
        if args.cache_root:
            cache_root = Path(args.cache_root)
        else:
            cache_root = Path.home() / ".viberate_cache"
        if args.replicate:
            model = Persistent(model, cache_root, replication=True)
        else:
            model = Persistent(model, cache_root)

    if not args.no_cache and args.export_cache:
        export_root = Path(args.export_cache)
        export_root.mkdir(parents=True, exist_ok=True)
        model = Persistent(model, export_root)
            
    test_venv = Path(args.test_venv)
    executor = Executor(test_venv)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        description = f.read()

    n1, n2 = 5, 5
    requirements = Requirements.from_description(model, description)
    viberate = VibeRate(executor, Vanilla(), n1, n2)
    sel_num, all_code, result, resonating = viberate.generate_and_select(model, requirements)
    if result:
        print("Result: selected")
        print(sel_num)
    else:
        print("Result: abstain")


if __name__ == "__main__":
    main()
