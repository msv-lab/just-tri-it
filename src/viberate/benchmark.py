import argparse
import sys
from pathlib import Path
from itertools import islice
from typing import List

from viberate.llm import Cached, AI302, LLM
from viberate.program import Program, Test
from viberate.utils import print_annotated_hr
from viberate.executor import Executor, Pass, Fail
from viberate.dataset import Dataset, load_dataset
from viberate.code_generator import Vanilla, Generator, Selector
from viberate.majority_vote import MajorityVote
from viberate.codet import CodeT

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
        "--export-cache",
        type=str,
        help="Explore all responsed generated during the run."
    )
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input file containing the dataset."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Identifier of task to run (all by default)."
    )
    parser.add_argument(
        "--generator",
        type=str,
        help="Code generator configuration to benchmark."
    )
    parser.add_argument(
        "--selector",
        type=str,
        help="Code selector configuration to benchmark."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM to use."
    )
    return parser.parse_args()


def passes_tests(executor: Executor, program: Program, tests: List[Test]) -> bool:
    ok = True
    for test in tests:
        match executor.run_test(program, test):
            case Pass():
                pass
            case Fail():
                ok = False
                break
    return ok


def evaluate_generator(model: LLM, executor: Executor, generator: Generator, dataset: Dataset):
    N = 5
    for task in dataset:
        programs = islice(generator.generate(model, task.requirements), N)
        results = []
        for program in programs:
            if passes_tests(executor, program, task.tests):
                results.append(1)
            else:
                results.append(0)
        print(f"Task {task.id} pass@1: {sum(results)/len(results)}")


def evaluate_selector(model: LLM, executor: Executor, selector: Selector, dataset: Dataset):
    for task in dataset:
        program = selector.generate_and_select(model, task.requirements).program
        if passes_tests(executor, program, task.tests):
            print(f"Task {task.id}: solved")
        else:
            print(f"Task {task.id}: failed")


def main():
    args = parse_args()
    
    model = AI302(args.model, 1.0)

    if not args.no_cache:
        if args.cache_root:
            cache_root = Path(args.cache_root)
        else:
            cache_root = Path.home() / ".viberate_cache"
        if args.replicate:
            model = Cached(model, cache_root, replication=True)
        else:
            model = Cached(model, cache_root)

    if not args.no_cache and args.export_cache:
        export_root = Path(args.export_cache)
        export_root.mkdir(parents=True, exist_ok=True)
        model.start_slicing(export_root)
            
    test_venv = Path(args.test_venv)
    executor = Executor(test_venv)

    dataset = load_dataset(Path(args.dataset))

    if args.task:
        dataset = [t for t in dataset if t.id == args.task]

    GENERATORS = {
        "Vanilla": Vanilla()
    }

    SELECTORS = {
        "MajorityVote": MajorityVote(executor, Vanilla(), 5),
        "CodeT": CodeT(executor, Vanilla(), 5, 5)
    }

    if args.generator:
        evaluate_generator(model, executor, GENERATORS[args.generator], dataset)
    if args.selector:
        evaluate_selector(model, executor, SELECTORS[args.selector], dataset)


if __name__ == "__main__":
    main()
