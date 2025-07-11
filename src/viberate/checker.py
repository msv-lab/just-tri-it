import sys

from viberate.program import Signature
from viberate.coder import generate_programs
from viberate.executor import Executor, Success
from viberate.tester import generate_tests
from viberate.requirements import inverse_signature, inverse_requirements, fiber_signature, fiber_requirements
from viberate.utils import print_hr, print_annotated_hr


def check_for_inv(executor, forward, inverse, forward_inputs):
    for forward_input in forward_inputs:
        forward_outcome = executor.run(forward, [forward_input])
        match forward_outcome:
            case Success(forward_output):
                inverse_outcome = executor.run(inverse, [forward_output])
                match forward_outcome:
                    case Success(inverse_output):
                        if forward_input != inverse_output:
                            return False
    return True


def select_for_inv(executor, model, req, n1, n2):
    sig = Signature.from_requirements(model, req)
    forward_programs = list(generate_programs(model, sig, req, n1))
    inverse_sig = inverse_signature(sig)
    inverse_req = inverse_requirements(model, sig, req)
    print_annotated_hr("Inverse requirements")
    print(inverse_req, file=sys.stderr)
    inverse_programs = list(generate_programs(model, inverse_sig, inverse_req, n2))
    forward_inputs = generate_tests(model, req)
    resonating_pairs = []
    for forward in forward_programs:
        for inverse in inverse_programs:
            if check_for_inv(executor, forward, inverse, forward_inputs):
                resonating_pairs.append((forward, inverse))
    return resonating_pairs


def check_for_fib_lib(executor, forward, fiber, fiber_inputs):
    for fiber_input in fiber_inputs:
        fiber_outcome = executor.run(fiber, [fiber_input])
        match fiber_outcome:
            case Success(fiber_outputs):
                for fiber_output in fiber_outputs:
                    forward_outcome = executor.run(forward, [fiber_output])
                    match forward_outcome:
                        case Success(forward_output):
                            if forward_output != fiber_input:
                                return False
    return True


def select_for_fib_lib(executor, model, req, n1, n2):
    sig = Signature.from_requirements(model, req)
    forward_programs = list(generate_programs(model, sig, req, n1))
    fiber_sig = fiber_signature(sig)
    fiber_req = fiber_requirements(model, sig, req)
    print_annotated_hr("Fiber requirements")
    print(fiber_req, file=sys.stderr)
    fiber_programs = list(generate_programs(model, fiber_sig, fiber_req, n2))
    fiber_inputs = generate_tests(model, fiber_req)
    resonating_pairs = []
    for forward in forward_programs:
        for fiber in fiber_programs:
            if check_for_fib_lib(executor, forward, fiber, fiber_inputs):
                resonating_pairs.append((forward, fiber))
    return resonating_pairs
