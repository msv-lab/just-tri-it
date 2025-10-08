import argparse
from pathlib import Path
import json

from just_tri_it.logic import check, Side
from just_tri_it.executor import SubprocessExecutor, PersistentWorkerExecutor
from just_tri_it.program import Program
from just_tri_it.triangulation import make_postcondition
from just_tri_it.dataset import load_dataset
from just_tri_it.utils import print_annotated_hr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Disable cache."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Left program."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.test_venv:
        executor = SubprocessExecutor(Path(args.test_venv))
    else:
        executor = PersistentWorkerExecutor()

    dataset = load_dataset(Path(args.dataset))

    for task in dataset:
        print()
        print_annotated_hr(f"Task {task.id}")
        
        if "correct_solution" in task.metadata:
            correct_program = Program.from_function_code(task.metadata["correct_solution"])
            status, results = correct_program.passes(executor, task.tests)
            if status:
                print("\nCorrect program - OK")
            else:
                print(f"\nCorrect program - MISCLASSIFIED {results}")
        else:
            print("\nCorrect program - N/A")

        if "incorrect_solution" in task.metadata:
            incorrect_program = Program.from_function_code(task.metadata["incorrect_solution"])
            status, results = incorrect_program.passes(executor, task.tests)
            if not status:
                print("\nIncorrect program - OK")
            else:
                print(f"\nIncorrect program - MISCLASSIFIED")
        else:
            print("\nIncorrect program - N/A")

    executor.shutdown()
    print()
    

if __name__ == "__main__":
    main()
