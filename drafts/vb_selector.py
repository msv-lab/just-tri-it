import sys

from itertools import islice
from viberate.code_generator import Selector, Generator
from viberate.executor import Executor
from viberate.input_generator import generate_inputs
from viberate.requirements import (
    choose_parameter_to_invert,
    Requirements
)
from viberate.triangulation import Triangulation, PartialFiber, PartialInverse, enumerate_pair, Wrapper, Property, \
    SpecificWrapper, Forward
from viberate.utils import print_annotated_hr
from viberate.cached_llm import Model
from viberate.logic import Var, ForAll, Pred, Func, And, FuncList
from semantic_deltas import Delta, IntAdd, FloatAdd, StrSuffix,delta_candidates
from syntactictransformation import SyntacticTransformation


class VibeRate(Selector):

    def __init__(self, executor: Executor, generator: Generator, n1: int, n2: int):
        self.executor = executor
        self.generator = generator
        self.n1 = n1
        self.n2 = n2

    def generate_and_select(self, model: Model, req: Requirements):
        # -------------- step 1: prepare formula --------------
        inverse_index = choose_parameter_to_invert(model, req)
        print_annotated_hr("Inverse index")
        print(inverse_index, file=sys.stderr)

        forward_inputs = generate_inputs(model, req, self.executor)
        print_annotated_hr("Forward tests")
        print(forward_inputs, file=sys.stderr)

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
        
        # Syntactic Triangulation Formula
        # This reformulates the problem while keeping semantics intact
        syntactic_formula = ForAll(all_arg, forward_inputs,
                              Pred("Equals", [
                                  Func("f", all_arg),  # original function
                                  Func("g", all_arg)   # syntactically reformulated version
                              ])
                             )
        
        
        # -------------- step 2: prepare triangulations --------------
        trans_to_programs = {}
        # forward: its transformation and mapping to programs
        forward = Forward()
        trans_to_programs.update(
            {forward: list(islice(self.generator.generate(model, req), self.n1))}
        )
        # partial for-inv wrt inverse_index: its transformation and mapping to programs
        partial_for_inv_i = PartialInverse(model, req, inverse_index)
        trans_to_programs.update(
            {partial_for_inv_i: list(islice(self.generator.generate(model, partial_for_inv_i.req), self.n2))}
        )
        # for-inv triangulation
        for_inv = Triangulation(forward, partial_for_inv_i, Property(for_inv_formula, Wrapper(self.executor)))

        # partial for-fib wrt inverse_index: its transformation and mapping to programs
        partial_for_fib_i = PartialFiber(model, req, inverse_index)
        trans_to_programs.update(
            {partial_for_fib_i: list(islice(self.generator.generate(model, partial_for_fib_i.req), self.n2))}
        )
        # for-fib triangulation
        for_fib = Triangulation(forward, partial_for_fib_i, Property(for_fib_formula, Wrapper(self.executor)))
        # for-inv (specific version) triangulation
        spec_for_inv = Triangulation(forward, partial_for_inv_i,
                                     Property(for_inv_formula,
                                              SpecificWrapper(self.executor, model, partial_for_inv_i, self.n2)))
        # for-fib (specific version) triangulation
        spec_for_fib = Triangulation(forward, partial_for_fib_i,
                                     Property(for_fib_formula,
                                              SpecificWrapper(self.executor, model, partial_for_fib_i, self.n2)))
        
    # Create Syntactic Transformation  
        syntactic_transform = SyntacticTransformation(model, req)
        trans_to_programs.update(
        {syntactic_transform: list(islice(self.generator.generate(model, syntactic_transform.req), self.n2))}
    )

    # Create Triangulations
        syntactic_triangulation = Triangulation(forward, syntactic_transform, 
                                          Property(syntactic_formula, Wrapper(self.executor)))
    
    


        return_type = getattr(req.signature, "returns", None) or getattr(req.signature, "return_type", None) or "unknown"
        delta_triangulations = []
        for delta in delta_candidates(return_type):
            delta_transform = DeltaSemanticTransformation(model, req, delta)
            trans_to_programs.update({
            delta_transform: list(islice(self.generator.generate(model, delta_transform.req), self.n2))
        })
        # DeltaEq :g(x) == Δ(f(x))
        delta_formula = ForAll(
            all_arg, forward_inputs,
            Pred("DeltaEq", [Func("f", all_arg), Func("g", all_arg), Func(delta.predicate_tag(), [])])
        )
        delta_tri = Triangulation(
            forward, delta_transform,
            Property(delta_formula, Wrapper(self.executor))  
        )
        delta_triangulations.append(delta_tri)


        # -------------- step 3: check for each triangulation --------------
        ts = [for_inv, for_fib, spec_for_inv, spec_for_fib, syntactic_triangulation] + delta_triangulations

        selected_forward = []
        resonating_pairs = {}
        for t in ts:
            print_annotated_hr(f"testing triangulation {t.print_name()}")
            new_selected, new_pairs = enumerate_pair(trans_to_programs, t, arity)
            for index in new_selected:
                if index not in selected_forward:
                    selected_forward.append(index)
            resonating_pairs.update({t.print_name(): new_pairs})

        if resonating_pairs:
            return selected_forward, trans_to_programs[forward], True, resonating_pairs
        else:
            return [], trans_to_programs[forward], False, []
        


class DeltaSemanticTransformation:
    def __init__(self, model: Model, req: Requirements, delta: Delta):
        self.model = model
        self.original_req = req
        self.delta = delta
        new_desc = (
            f"{req.description.strip()} "
            f"(Semantic variant: the output equals Δ({delta.predicate_tag()}) applied to the original result.)"
        )
        self.req = Requirements(signature=req.signature, description=new_desc)


