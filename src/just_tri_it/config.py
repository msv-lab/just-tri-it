from just_tri_it.executor import Executor
from just_tri_it.plurality import Plurality
from just_tri_it.test_agreement import TestAgreement
from just_tri_it.selection import MaxWitness, Ransac
from just_tri_it.triangulation import (
    Triangulator,
    TriangulationMode
)
from just_tri_it.code_generator import Generator
from just_tri_it.cached_llm import Model
from just_tri_it.test_generator import InputOutputGenerator, TestFunctionGenerator


NUM_LEFT_SAMPLES = 30
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
                                    
        "Syntactic": Ransac(Triangulator(executor,
                                         code_generator,
                                         TriangulationMode.Syntactic,
                                         NUM_LEFT_SAMPLES,
                                         NUM_RIGHT_SAMPLES)),
    
        "OffByOne": Ransac(Triangulator(executor,
                                        code_generator,
                                        TriangulationMode.OffByOne,
                                        NUM_LEFT_SAMPLES,
                                        NUM_RIGHT_SAMPLES)),

        "Postcondition": Ransac(Triangulator(executor,
                                             code_generator,
                                             TriangulationMode.Postcondition,
                                             NUM_LEFT_SAMPLES,
                                             NUM_RIGHT_SAMPLES)),

        "FWD_INV": Ransac(Triangulator(executor,
                                       code_generator,
                                       TriangulationMode.FWD_INV,
                                       NUM_LEFT_SAMPLES,
                                       NUM_RIGHT_SAMPLES)),
        
        "FWD_SINV": Ransac(Triangulator(executor,
                                        code_generator,
                                        TriangulationMode.FWD_SINV,
                                        NUM_LEFT_SAMPLES,
                                        NUM_RIGHT_SAMPLES)),

        "ENUM_SINV": Ransac(Triangulator(executor,
                                         code_generator,
                                         TriangulationMode.ENUM_SINV,
                                         NUM_LEFT_SAMPLES,
                                         NUM_RIGHT_SAMPLES))
    }
