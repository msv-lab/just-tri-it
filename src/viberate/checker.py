import sys

from itertools import islice

from viberate.program import Signature
from viberate.coder import generate_programs
from viberate.executor import Success
from viberate.tester import generate_tests
from viberate.requirements import (
    NamedReturnSignature,
    choose_parameter_to_invert,
    inverse_requirements_old,
    inverse_signature,
    inverse_requirements,
    fiber_signature,
    fiber_requirements,
    fiber_requirements_old
)
from viberate.utils import print_annotated_hr
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
                            print(f"Expected:{forward_input[arg_index]}, Actual:{inverse_output}")
                            return False
                    case _:
                        print("Facing error or panic or timeout!")
                        return False
            case _:
                print("Facing error or panic or timeout!")
                return False
    return True


def select_for_inv(executor, model, req, n1, n2):
    # generate forward signature and programs
    sig = Signature.from_requirements(model, req)
    sig = NamedReturnSignature.from_requirements(model, sig, req)
    print_annotated_hr("Signature")
    print(sig.pretty_print(), file=sys.stderr)
    forward_programs = list(islice(generate_programs(model, sig, req), n1))

    if len(sig.params) == 1:
        inverse_index = 0
    else:
        inverse_index = choose_parameter_to_invert(model, sig, req)

    inverse_sig = inverse_signature(model, sig, inverse_index, req)
    print_annotated_hr(f"Inverse signature wrt {inverse_index}")
    print(inverse_sig.pretty_print(), file=sys.stderr)
        
    if len(sig.params) == 1:
        inverse_req = inverse_requirements_old(model, sig, inverse_sig, req)
    else:
        inverse_req = inverse_requirements(model, sig, inverse_sig, inverse_index, req)
    print_annotated_hr("Inverse requirements")
    print(inverse_req, file=sys.stderr)

    # generate inverse programs and tests
    inverse_programs = list(islice(generate_programs(model, inverse_sig, inverse_req), n2))

    forward_inputs = generate_tests(model, req, sig)
    print_annotated_hr("Forward tests")
    print(forward_inputs, file=sys.stderr)

    # check for every pair (unfinished!)
    resonating_pairs = []
    for for_index, forward in enumerate(forward_programs):
        for inv_index, inverse in enumerate(inverse_programs):
            print_annotated_hr(f"testing forward {for_index} and inverse {inv_index}")
            print(inverse, file=sys.stderr)
            # if check_for_inv(executor, forward, inverse, forward_inputs, sig.inverse_index):
            #     resonating_pairs.append((forward, inverse))
            if test_for_inv_property(executor, forward, inverse, len(sig.params), forward_inputs, inverse_index):
                resonating_pairs.append((forward, inverse))
    return resonating_pairs


def check_for_fib_lib(executor, forward, fiber, forward_inputs, arg_index):
    for forward_input in forward_inputs:
        forward_outcome = executor.run(forward, forward_input)  # calculate f(a)
        match forward_outcome:
            case Success(forward_output):
                fiber_outcome = executor.run(fiber, [forward_output] + forward_input[:arg_index]
                                             + forward_input[arg_index+1:])
                # calculate g(f(a),a1,...,ai-1,ai+1,...,an) --> List
                match fiber_outcome:
                    case Success(fiber_outputs):
                        if forward_input[arg_index] not in fiber_outputs:
                            # check the right of the formula
                            print(f"{forward_input[arg_index]} didn't appear in {fiber_outputs}")
                            return False
                        for fiber_output in fiber_outputs:
                            reorg_forward_input = forward_input[:arg_index] + forward_input[arg_index+1:]
                            reorg_forward_input.insert(arg_index, fiber_output)
                            forward_outcome_2 = executor.run(forward, reorg_forward_input)  # calculate f(g(f(a),...),...)
                            match forward_outcome_2:
                                case Success(forward_output_2):
                                    if forward_output_2 != forward_output:
                                        print(f"Expected:{forward_output}, Actual:{forward_output_2}")
                                        return False
                                case _:
                                    print(forward_outcome_2)
                                    print("Facing error or panic or timeout!")
                                    return False
                    case _:
                        print(fiber_outcome)
                        print("Facing error or panic or timeout!")
                        return False
            case _:
                print(forward_outcome)
                print("Facing error or panic or timeout!")
                return False
    return True


def select_for_fib_lib(executor, model, req, n1, n2):
    sig = Signature.from_requirements(model, req)
    sig = NamedReturnSignature.from_requirements(model, sig, req)
    print_annotated_hr("Signature")
    print(sig.pretty_print(), file=sys.stderr)

    forward_programs = list(islice(generate_programs(model, sig, req), n1))

    if len(sig.params) == 1:
        inverse_index = 0
    else:
        inverse_index = choose_parameter_to_invert(model, sig, req)

    fiber_sig = fiber_signature(model, sig, inverse_index, req)
    print_annotated_hr(f"Fiber signature wrt {inverse_index}")
    print(fiber_sig.pretty_print(), file=sys.stderr)
        
    if len(sig.params) == 1:
        fiber_req = fiber_requirements_old(model, sig, fiber_sig, req)
    else:
        fiber_req = fiber_requirements(model, sig, fiber_sig, inverse_index, req)
    print_annotated_hr("Fiber requirements")
    print(fiber_req, file=sys.stderr)

    fiber_programs = list(islice(generate_programs(model, fiber_sig, fiber_req), n2))

    # generate input tests
    forward_inputs = generate_tests(model, req, sig)
    print_annotated_hr("Fiber tests (for forward func)")
    print(forward_inputs, file=sys.stderr)

    resonating_pairs = []
    for for_index, forward in enumerate(forward_programs):
        for fib_index, fiber in enumerate(fiber_programs):
            print_annotated_hr(f"testing forward {for_index} and fiber {fib_index}")
            print(fiber, file=sys.stderr)
            # if check_for_fib_lib(executor, forward, fiber, forward_inputs, inverse_index):
            #     resonating_pairs.append((forward, fiber))
            if test_for_fib_property(executor, forward, fiber, len(sig.params), forward_inputs, inverse_index):
                resonating_pairs.append((forward, fiber))
    return resonating_pairs  # unfinished!
