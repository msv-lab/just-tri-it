import argparse
from pathlib import Path
import json

from just_tri_it.logic import check, Side
from just_tri_it.executor import Executor
from just_tri_it.program import Program
from just_tri_it.triangulation import make_postcondition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    parser.add_argument(
        "--program",
        type=str,
        required=True,
        help="Left program."
    )
    parser.add_argument(
        "--judge",
        type=str,
        required=True,
        help="Right program."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Inputs."
    )    
    return parser.parse_args()

def main():
    args = parse_args()

    executor = Executor(Path(args.test_venv))    
    left_program_file = Path(args.program)
    right_program_file = Path(args.judge)
    inputs_file = Path(args.inputs)

    left_program = Program.from_function_code(left_program_file.read_text(encoding="utf-8"))
    right_program = Program.from_function_code(right_program_file.read_text(encoding="utf-8"))
    inputs = json.loads(inputs_file.read_text(encoding="utf-8"))
        

    tri = make_postcondition(len(left_program.signature.params))

    check(executor,
          { Side.LEFT: inputs, Side.RIGHT: None },
          { Side.LEFT: left_program, Side.RIGHT: right_program },
          tri.hyperproperty)

    print()
    

if __name__ == "__main__":
    main()


