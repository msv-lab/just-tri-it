import sys

from itertools import islice

from viberate.program import Signature
from viberate.code_generator import Vanilla
from viberate.executor import Success
from viberate.input_generator import generate_inputs
from viberate.requirements import (
    choose_parameter_to_invert,
    inverse_requirements,
    fiber_requirements,
    Requirements
)
from viberate.utils import print_annotated_hr
from viberate.llm import LLM
from viberate.logic import test_for_inv_property, test_for_fib_property


def check_for_inv(executor, forward, inverse, forward_inputs, arg_index):
    for forward_input in forward_inputs:
        forward_outcome = executor.run(forward, forward_input)
        match forward_outcome:
            case Success(forward_output):
                args = [forward_output] + forward_input[:arg_index] + forward_input[arg_index+1:]
                inverse_outcome = executor.run(inverse, args)
                match inverse_outcome:
                    case Success(inverse_output):
                        if forward_input[arg_index] != inverse_output:
                            print(f"Property failed on {forward_input}: expected {forward_input[arg_index]}, but got {inverse_output}", file=sys.stderr)
                            return False
                    case _:
                        print(f"Error on input {args}: {inverse_outcome}", file=sys.stderr)
                        return False
            case _:
                print(f"Error on input {forward_input}: {forward_outcome}", file=sys.stderr)
                return False
    return True


def check_for_fib_lib(executor, forward, fiber, forward_inputs, arg_index):
    for forward_input in forward_inputs:
        forward_outcome = executor.run(forward, forward_input)  # calculate f(a)
        match forward_outcome:
            case Success(forward_output):
                fiber_input = [forward_output] + forward_input[:arg_index] + forward_input[arg_index+1:]
                fiber_outcome = executor.run(fiber, fiber_input)
                # calculate g(f(a),a1,...,ai-1,ai+1,...,an) --> List
                match fiber_outcome:
                    case Success(fiber_outputs):
                        if forward_input[arg_index] not in fiber_outputs:
                            # check the right of the formula
                            print(f"Failed on {forward_input}: {forward_input[arg_index]} didn't appear in {fiber_outputs}")
                            return False
                        for fiber_output in fiber_outputs:
                            reorg_forward_input = forward_input[:arg_index] + forward_input[arg_index+1:]
                            reorg_forward_input.insert(arg_index, fiber_output)
                            forward_outcome_2 = executor.run(forward, reorg_forward_input)  # calculate f(g(f(a),...),...)
                            match forward_outcome_2:
                                case Success(forward_output_2):
                                    if forward_output_2 != forward_output:
                                        print(f"Property failed on {forward_input}: expected {forward_output}, but got {forward_output_2}", file=sys.stderr)
                                        return False
                                case _:
                                    print(f"Error on input {reorg_forward_input}: {forward_outcome_2}", file=sys.stderr)
                                    return False
                    case _:
                        print(f"Error on input {fiber_input}: {fiber_outcome}", file=sys.stderr)
                        return False
            case _:
                print(f"Error on input {forward_input}: {forward_outcome}", file=sys.stderr)
                return False
    return True


def select(executor, model: LLM, req: Requirements, n1: int, n2: int):
    forward_programs = islice(Vanilla().generate(model, req), n1)

    inverse_index = choose_parameter_to_invert(model, req)

    inverse_req = inverse_requirements(model, req, inverse_index)
    print_annotated_hr("Inverse requirements")
    print(inverse_req, file=sys.stderr)
    inverse_programs = islice(Vanilla().generate(model, inverse_req), n2)

    fiber_req = fiber_requirements(model, req, inverse_index)
    print_annotated_hr("Fiber requirements")
    print(fiber_req, file=sys.stderr)
    fiber_programs = list(islice(Vanilla().generate(model, fiber_req), n2))

    forward_inputs = generate_inputs(model, req)
    print_annotated_hr("Forward tests")
    print(forward_inputs, file=sys.stderr)

    resonating_pairs = []

    for i, forward in enumerate(forward_programs):
        print_annotated_hr(f"forward {i}")
        print(forward, file=sys.stderr)
        for j, inverse in enumerate(inverse_programs):
            print_annotated_hr(f"testing forward {i} and inverse {j}")
            print(inverse, file=sys.stderr)
            if check_for_inv(executor, forward, inverse, forward_inputs, inverse_index):
                resonating_pairs.append((forward, inverse))
            #TODO: switch to a general-purpose checker
            # if test_for_inv_property(executor, forward, inverse, len(req.signature.params), forward_inputs, inverse_index):
            #     resonating_pairs.append((forward, inverse))

        for j, fiber in enumerate(fiber_programs):
            print_annotated_hr(f"testing forward {i} and fiber {j}")
            print(fiber, file=sys.stderr)
            if check_for_fib_lib(executor, forward, fiber, forward_inputs, inverse_index):
                resonating_pairs.append((forward, fiber))
            #TODO: switch to a general-purpose checker                
            # if test_for_fib_property(executor, forward, fiber, len(sig.params), forward_inputs, inverse_index):
            #     resonating_pairs.append((forward, fiber))

    return resonating_pairs
