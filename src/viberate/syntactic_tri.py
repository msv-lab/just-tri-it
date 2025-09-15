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
    SpecificWrapper, Forward, program_printer, SyntacticTrans, TSemanticTrans
from viberate.utils import print_annotated_hr
from viberate.cached_llm import Model
from viberate.logic import Var, ForAll, Pred, Func, And, FuncList


def check_tag(tag):
    tag = tag.lower()
    match tag:
        case "int" | "float" | "bool" | "str":
            return True
        case _ if tag.startswith("list[") and tag.endswith("]"):
            subtype = tag[5:-1].strip()
            return check_tag(subtype)
    return False


class OtherTri(Selector):

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

        # Syntactic formula
        syntactic_formula = ForAll(all_arg, forward_inputs,
                                   Pred("Equals", [
                                       Func("f", all_arg),  # original function
                                       Func("g", all_arg)  # syntactically reformulated version
                                   ])
                                   )

        trans_to_programs = {}
        ts = []
        forward = Forward()
        trans_to_programs.update(
            {forward: list(islice(self.generator.generate(model, req, p_dir), self.n1))}
        )
        if p_dict and tests:
            p_dict, exp_results["programs"][forward.get_name()] = Selector.store_program_return_correctness(
                self.executor, trans_to_programs[forward], tests, p_dict)
        else:
            exp_results["programs"][forward.get_name()] = Selector.store_program(trans_to_programs[forward])
        program_printer(trans_to_programs, forward)

        syntactic_trans = SyntacticTrans(model, req)
        trans_to_programs.update(
            {syntactic_trans: list(islice(self.generator.generate(model, syntactic_trans.req, p_dir), self.n2))}
        )
        exp_results["programs"][syntactic_trans.get_name()] = Selector.store_program(trans_to_programs[syntactic_trans])
        program_printer(trans_to_programs, syntactic_trans)
        syntactic_tri = Triangulation(forward, syntactic_trans, Property(syntactic_formula, Wrapper(self.executor)))
        ts.append(syntactic_tri)

        if check_tag(req.signature.return_type):
            t_semantic_formula = ForAll(all_arg, forward_inputs,
                                        Pred("DeltaEq", [
                                            Func("f", all_arg),  # original function
                                            Func("g", all_arg),  # syntactically reformulated version
                                            req.signature.return_type
                                        ])
                                        )

            t_semantic_trans = TSemanticTrans(model, req)
            trans_to_programs.update(
                {t_semantic_trans: list(islice(self.generator.generate(model, t_semantic_trans.req, p_dir), self.n2))}
            )
            exp_results["programs"][t_semantic_trans.get_name()] = Selector.store_program(trans_to_programs[t_semantic_trans])
            program_printer(trans_to_programs, t_semantic_trans)
            t_semantic_tri = Triangulation(forward, t_semantic_trans, Property(t_semantic_formula, Wrapper(self.executor)))
            ts.append(t_semantic_tri)
        # unfinished
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
