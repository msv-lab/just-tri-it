import argparse
from pathlib import Path

from just_tri_it.cached_llm import AI302, XMCP
from just_tri_it.executor import SubprocessExecutor, PersistentWorkerExecutor
from just_tri_it.dataset import load_dataset
from just_tri_it.code_generator import Vanilla
from just_tri_it.selection import Selected, Abstained
from just_tri_it.utils import print_annotated_hr, add_cache_options, setup_cache, print_legend, init_random
import just_tri_it.utils
from just_tri_it.config import init_selectors
from just_tri_it.program import Program


def parse_args():
    parser = argparse.ArgumentParser()
    add_cache_options(parser)
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug messages."
    )
    return parser.parse_args()


def evaluate_selector(model, executor, selector, dataset):
    for task in dataset:
        print()
        print_annotated_hr(f"Task {task.id}")
        just_tri_it.utils.CURRENT_TASK = task.id
        if callable(selector):
            s = selector(task)
        else:
            s = selector
        outcome, _ = s.generate_and_select(model, task.requirements)
        match outcome:
            case Selected(program, _):
                if program.passes(executor, task.tests)[0]:
                    print("\nSELECTED: correct")
                else:
                    print("\nSELECTED: incorrect")
            case Abstained():
                print("\nABSTAINED")

def main():    
    init_random()
    
    args = parse_args()

    if args.debug:
        just_tri_it.utils.DEBUG = True
    
    model = {
        "gpt-4o": AI302("gpt-4o", 1.0, max_batch=50),
        "deepseek-v3": AI302("deepseek-v3.1", 1.0, alias="deepseek-v3"),
        "gemini-2.5-flash": AI302("gemini-2.5-flash", 1.0)
    }[args.model]
    # model = XMCP(args.model, 1.0)

    just_tri_it.utils.CURRENT_MODEL=args.model

    model = setup_cache(model, args)
    if args.test_venv:
        executor = SubprocessExecutor(Path(args.test_venv))
    else:
        executor = PersistentWorkerExecutor()

    dataset = load_dataset(Path(args.dataset))
    if args.task:
        dataset = [t for t in dataset if t.id == args.task]

    print_legend()
    
    selectors = init_selectors(executor, Vanilla(), model)
        
    evaluate_selector(model, executor, selectors[args.selector], dataset)

    executor.shutdown()


if __name__ == "__main__":
    main()
