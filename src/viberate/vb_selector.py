import sys

from itertools import islice
from viberate.code_generator import Selector, Generator, Selected, Abstained, SpecificGenerator
from viberate.executor import Executor
from viberate.input_generator import generate_inputs
from viberate.requirements import (
    choose_parameter_to_invert,
    Requirements
)
from viberate.triangulation import Triangulation, PartialFiber, PartialInverse, enumerate_pair, Wrapper, Property, \
    SpecificWrapper, Forward, program_printer
from viberate.utils import print_annotated_hr
from viberate.cached_llm import Model
from viberate.logic import Var, ForAll, Pred, Func, And, FuncList


class VibeRate(Selector):

    def __init__(self, executor: Executor, generator: Generator, n1: int, n2: int):
        self.executor = executor
        self.generator = generator
        self.n1 = n1
        self.n2 = n2

    def generate_and_select(self, model: Model, req: Requirements, p_dir, p_dict=None, tests=None):
        exp_results = {
            "programs": {},
            "generated_inputs": None,
            "pairs": {},
            "decision": None,
            "chosen_programs": []
        }
        # -------------- step 1: prepare formula --------------
        inverse_index = choose_parameter_to_invert(model, req)
        print_annotated_hr("Inverse index")
        print(inverse_index, file=sys.stderr)

        forward_inputs = generate_inputs(model, req, self.executor)
        exp_results["generated_inputs"] = forward_inputs

        arity = len(req.signature.params)
        all_arg = []
        for i in range(arity):
            all_arg.append(Var(f"x_{i}"))
        new_arg = all_arg[:inverse_index] + all_arg[inverse_index + 1:]

        # for_inv formula
        for_inv_formula = ForAll(all_arg, forward_inputs,
                                 Pred("Equals", [
                                     Var(f"x_{inverse_index}"),  # x_i
                                     Func("g", [Func("f", all_arg)] + new_arg)  # g(...)
                                 ])
                                 )
        # for_fib formula
        for_fib_formula = ForAll(all_arg, forward_inputs,
                                 And(
                                     Pred("Includes", [
                                         Var(f"x_{inverse_index}"),  # x_i
                                         Func("g", [Func("f", all_arg)] + new_arg)  # g(...)
                                     ]),
                                     Pred("Equals_set", [
                                         Func("f", all_arg),
                                         FuncList("f", inverse_index, all_arg, Func("g", [Func("f", all_arg)]
                                                                                    + new_arg))
                                     ])
                                 )
                                 )
        # -------------- step 2: prepare triangulations --------------
        trans_to_programs = {}
        # forward: its transformation and mapping to programs
        forward = Forward()
        trans_to_programs.update(
            {forward: list(islice(self.generator.generate(model, req, p_dir), self.n1))}
        )
        if p_dict and tests:
            p_dict, exp_results["programs"][forward.get_name()] = Selector.store_program_return_correctness(self.executor, trans_to_programs[forward], tests, p_dict)
        else:
            exp_results["programs"][forward.get_name()] = Selector.store_program(trans_to_programs[forward])
        program_printer(trans_to_programs, forward)
        # partial for-inv wrt inverse_index: its transformation and mapping to programs
        partial_for_inv_i = PartialInverse(model, req, inverse_index)
        trans_to_programs.update(
            {partial_for_inv_i: list(islice(self.generator.generate(model, partial_for_inv_i.req, p_dir), self.n2))}
        )
        exp_results["programs"][partial_for_inv_i.get_name()] = Selector.store_program(trans_to_programs[partial_for_inv_i])
        program_printer(trans_to_programs, partial_for_inv_i)
        # for-inv triangulation
        for_inv = Triangulation(forward, partial_for_inv_i, Property(for_inv_formula, Wrapper(self.executor)))

        # partial for-fib wrt inverse_index: its transformation and mapping to programs
        partial_for_fib_i = PartialFiber(model, req, inverse_index)
        trans_to_programs.update(
            {partial_for_fib_i: list(islice(self.generator.generate(model, partial_for_fib_i.req, p_dir), self.n2))}
        )
        exp_results["programs"][partial_for_fib_i.get_name()] = Selector.store_program(trans_to_programs[partial_for_fib_i])
        program_printer(trans_to_programs, partial_for_fib_i)
        # for-fib triangulation
        for_fib = Triangulation(forward, partial_for_fib_i, Property(for_fib_formula, Wrapper(self.executor)))
        # for specific version
        spec_gen = SpecificGenerator()
        # for-inv (specific version) triangulation
        spec_for_inv = Triangulation(forward, partial_for_inv_i,
                                     Property(for_inv_formula,
                                              SpecificWrapper(self.executor, model, partial_for_inv_i, self.n2, spec_gen)))
        # for-fib (specific version) triangulation
        spec_for_fib = Triangulation(forward, partial_for_fib_i,
                                     Property(for_fib_formula,
                                              SpecificWrapper(self.executor, model, partial_for_fib_i, self.n2, spec_gen)))

        # -------------- step 3: check for each triangulation --------------
        # ts: list[Triangulation] = [for_inv, for_fib, spec_for_inv, spec_for_fib]
        ts: list[Triangulation] = [for_inv, for_fib]
        # ts: list[Triangulation] = [spec_for_inv, spec_for_fib]
        selected_forward = []
        resonating_pairs = {}
        for t in ts:
            print_annotated_hr(f"testing triangulation {t.print_name()}")
            new_selected, new_pairs = enumerate_pair(trans_to_programs, t, arity)
            for index in new_selected:
                if index not in selected_forward:
                    selected_forward.append(index)
            if new_pairs:
                resonating_pairs.update({t.print_name(): new_pairs})
        if resonating_pairs is not {}:
            exp_results["pairs"] = resonating_pairs
            exp_results["decision"] = "Selected"
            exp_results["chosen_programs"] = selected_forward
        else:
            exp_results["decision"] = "Abstained"
        return exp_results, p_dict