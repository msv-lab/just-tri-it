import argparse
import sys
from pathlib import Path


from viberate.llm import Cached, AI302, LLM
from viberate.utils import print_annotated_hr
from viberate.executor import Executor, Pass, Fail
from viberate.dataset import Dataset, load_dataset
from viberate.coder import Vanilla, Generator


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
        "--config",
        type=str,
        help="Tool configuration to benchmark."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM to use."
    )
    return parser.parse_args()


def evaluate_generator(model: LLM, executor: Executor, generator: Generator, dataset: Dataset):
    for task in dataset:
        program = next(generator.generate(model, task.requirements))
        solved = True
        for test in task.tests:
            match executor.run_test(program, test):
                case Pass():
                    pass
                case Fail():
                    solved = False
        print(f"Task {task.id}: {solved}")


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

    CONFIGURATIONS = {
        "Vanilla": Vanilla()
    }

    config = CONFIGURATIONS[args.config]

    evaluate_generator(model, executor, config, dataset)


if __name__ == "__main__":
    main()
