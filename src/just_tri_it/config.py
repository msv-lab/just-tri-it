from just_tri_it.executor import Executor
from just_tri_it.plurality import Plurality
from just_tri_it.test_agreement import TestAgreement
from just_tri_it.selection import MaxWitness, Ransac
from just_tri_it.triangulation import (
    Triangulator,
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


NUM_LEFT_SAMPLES = 20
NUM_RIGHT_SAMPLES = 20


def init_selectors(executor: Executor, code_generator: Generator, model: Model):
    return {
        "Plurality": MaxWitness(Plurality(executor, code_generator, NUM_LEFT_SAMPLES)),

        "MajorityVote": MaxWitness(Plurality(executor,
                                             code_generator,
                                             NUM_LEFT_SAMPLES,
                                             prob_threshold=0.5)),

        "MaxTest_Assert": MaxWitness(TestAgreement(executor,
                                                   code_generator,
                                                   TestFunctionGenerator(),
                                                   NUM_LEFT_SAMPLES)),
        
        "MaxTest_IO": MaxWitness(TestAgreement(executor,
                                               code_generator,
                                               InputOutputGenerator(),
                                               NUM_LEFT_SAMPLES)),
        
        "CodeT_Assert": Ransac(TestAgreement(executor,
                                             code_generator,
                                             TestFunctionGenerator(),
                                             NUM_LEFT_SAMPLES)),
        
        "CodeT_IO": Ransac(TestAgreement(executor,
                                         code_generator,
                                         InputOutputGenerator(),
                                         NUM_LEFT_SAMPLES)),
                                    
        "Syntactic": (lambda t:
                      Ransac(Triangulator(executor,
                                          code_generator,
                                          make_syntactic(len(t.requirements.signature.params)),
                                          NUM_LEFT_SAMPLES,
                                          NUM_RIGHT_SAMPLES))),
    
        "OffByOne": (lambda t:
                     Ransac(Triangulator(executor,
                                         code_generator,
                                         make_trivial_semantic(len(t.requirements.signature.params)),
                                         NUM_LEFT_SAMPLES,
                                         NUM_RIGHT_SAMPLES))),

        "Postcondition": (lambda t:
                          Ransac(Triangulator(executor,
                                              code_generator,
                                              make_postcondition(len(t.requirements.signature.params)),
                                              NUM_LEFT_SAMPLES,
                                              NUM_RIGHT_SAMPLES))),

        "FOR_INV": (lambda t:
                    Ransac(Triangulator(executor,
                                        code_generator,
                                        make_partial_for_inv(len(t.requirements.signature.params),
                                                             choose_parameter_to_invert(model, t.requirements)),
                                        NUM_LEFT_SAMPLES,
                                        NUM_RIGHT_SAMPLES))),
        
        "FOR_FIB": (lambda t:
                    Ransac(Triangulator(executor,
                                        code_generator,
                                        make_partial_for_fib(len(t.requirements.signature.params),
                                                             choose_parameter_to_invert(model, t.requirements)),
                                        NUM_LEFT_SAMPLES,
                                        NUM_RIGHT_SAMPLES)))
    }
