from just_tri_it.executor import Executor
from just_tri_it.plurality import Plurality
from just_tri_it.codet import CodeT
from just_tri_it.triangulation import (
    TriSelector,
    choose_parameter_to_invert,
    make_partial_for_fib,
    make_partial_for_inv,
    make_syntactic,
    make_postcondition,
    make_trivial_semantic
)
from just_tri_it.code_generator import Generator
from just_tri_it.cached_llm import Model
from just_tri_it.test_generator import InputOutputGenerator, TestFunctionGenerator


MINIMUM_NUM_INPUTS = 10
MAXIMUM_NUM_INPUTS = 50

NUM_LEFT_SAMPLES = 10
NUM_RIGHT_SAMPLES = 10

def init_selectors(executor: Executor, code_generator: Generator, model: Model):
    return {
        "Plurality": Plurality(executor, code_generator, NUM_LEFT_SAMPLES),

        # "Test_Assertion": TestBasedSelector(executor,
        #                                     code_generator,
        #                                     TestFunctionGenerator(),
        #                                     NUM_LEFT_SAMPLES),

        # "Test_InputOutput": TestBasedSelector(executor,
        #                                       code_generator,
        #                                       InputOutputGenerator(),
        #                                       NUM_LEFT_SAMPLES),

        "CodeT_Assertion": CodeT(executor,
                                 code_generator,
                                 TestFunctionGenerator(),
                                 NUM_LEFT_SAMPLES,
                                 NUM_RIGHT_SAMPLES),

        "CodeT_InputOutput": CodeT(executor,
                                   code_generator,
                                   InputOutputGenerator(),
                                   NUM_LEFT_SAMPLES,
                                   NUM_RIGHT_SAMPLES),

        "Tri_Syntactic": (lambda t:
                          TriSelector(executor,
                                      code_generator,
                                      make_syntactic(len(t.requirements.signature.params)),
                                      NUM_LEFT_SAMPLES,
                                      NUM_RIGHT_SAMPLES)),
                          
        "Tri_OffByOne": (lambda t:
                         TriSelector(executor,
                                     code_generator,
                                     make_trivial_semantic(len(t.requirements.signature.params)),
                                     NUM_LEFT_SAMPLES,
                                     NUM_RIGHT_SAMPLES)),

        "Postcondition": (lambda t:
                          TriSelector(executor,
                                      code_generator,
                                      make_postcondition(len(t.requirements.signature.params)),
                                      NUM_LEFT_SAMPLES,
                                      NUM_RIGHT_SAMPLES)),

        "Tri_FOR_INV": (lambda t:
                        TriSelector(executor,
                                    code_generator,
                                    make_partial_for_inv(len(t.requirements.signature.params),
                                                         choose_parameter_to_invert(model, t.requirements)),
                                    NUM_LEFT_SAMPLES,
                                    NUM_RIGHT_SAMPLES)),

        "Tri_FOR_FIB": (lambda t:
                        TriSelector(executor,
                                    code_generator,
                                    make_partial_for_fib(len(t.requirements.signature.params),
                                                         choose_parameter_to_invert(model, t.requirements)),
                                    NUM_LEFT_SAMPLES,
                                    NUM_RIGHT_SAMPLES))
    }
