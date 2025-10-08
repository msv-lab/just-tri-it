import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from functools import partial

from just_tri_it.executor import Success, Error
from just_tri_it.executor import Executor
from just_tri_it.utils import ExperimentFailure
from just_tri_it.logic import (
    Side,
    Term, Var, Map, MapUnpack,
    Formula, App, Not, And, Or, Implies, Iff, ForAll,
    Func, Equals, SetEquals, OffByOne, Member, Tolerate,
    SpecialValue, Angelic, Demonic, Undefined
)


class Checker(ABC):

    @abstractmethod
    def check(self,
              inputs: Dict[Side, Any],
              programs: Dict[Side, 'Program'],
              formula: Formula) -> bool:
        pass


EVALUATOR_CHECKER_CALL_BUDGET_PER_INPUT = 500


class CallBudgetExceeded(Exception):
    "Raised when make too many program calls"
    pass


class EvaluatorChecker(Checker):

    def __init__(self, executor: Executor):
        self.executor = executor

    def _eval_list(self, env, programs, terms):
        return list(map(partial(self._eval_term, env, programs), terms))

    def _eval_app(self, env, programs, func, args):
        computed_args = self._eval_term(env, programs, args)
        if not isinstance(computed_args, list):
            computed_args = [Demonic()]
        if isinstance(func.semantics, Side):
            special = [arg for arg in computed_args if isinstance(arg, SpecialValue)]
            if len(special) > 0:
                return SpecialValue.strongest(special)
            if self.available_call_budget <= 0:
                raise CallBudgetExceeded()
            execution_outcome = self.executor.run(programs[func.semantics], computed_args)
            self.available_call_budget -= 1
            match execution_outcome:
                case Success(v):
                    return v
                case Error(error_type, error_msg) \
                     if error_type == "ValueError" and error_msg == "Invalid input":
                    return Undefined()
                case _:
                    return Demonic()
        else:
            result = func.semantics(*computed_args)
            return result

    def _is_formula_true(self,
                        env: Dict[str, Any],
                        inputs: Dict[Side, Any],
                        programs: Dict[Side, 'Program'],
                        formula: Formula):
        match formula:
            case App(func, args):
                return self._eval_app(env, programs, func, args)
            case Not(operand):
                result = self._is_formula_true(env, inputs, programs, operand)
                if isinstance(result, SpecialValue):
                    return result
                if isinstance(result, bool):
                    return not result
                raise ValueError("unexpected operand type")
            case And(left, right):
                result_left = self._is_formula_true(env, inputs, programs, left)
                result_right = self._is_formula_true(env, inputs, programs, right)
                if isinstance(result_left, bool) and isinstance(result_right, bool):
                    return result_left and result_right
                if SpecialValue.in_list([result_left, result_right]):
                    return SpecialValue.strongest([result_left, result_right])
                raise ValueError("unexpected operand type")
            case Or(left, right):
                result_left = self._is_formula_true(env, inputs, programs, left)
                result_right = self._is_formula_true(env, inputs, programs, right)
                if isinstance(result_left, bool) and isinstance(result_right, bool):
                    return result_left or result_right
                if SpecialValue.in_list([result_left, result_right]):
                    return SpecialValue.strongest([result_left, result_right])
                raise ValueError("unexpected operand type")
            case ForAll(ele, domain, body):
                num_angelic = 0
                for inp in inputs[domain]:
                    new_env = env.copy()
                    if isinstance(ele, Var):
                        new_env[ele.name] = inp[0]
                    else:
                        for index, var in enumerate(ele):
                            new_env[var.name] = inp[index]
                    result = self._is_formula_true(new_env, inputs, programs, body)
                    if isinstance(result, SpecialValue):
                        if SpecialValue.as_bool(result) is False:
                            print(f"\n{formula} err on {inp}", file=sys.stderr, flush=True)
                            return False
                        else:
                            num_angelic += 1
                    elif result is False:
                        print(f"\n{formula} failed on {inp}", file=sys.stderr, flush=True)
                        return False
                if (num_angelic / len(inputs[domain])) >= 0.34:
                    print(f"\n{formula} failed due excessive angelic values", file=sys.stderr, flush=True)
                    return False
                return True
            case _:
                raise ValueError(f"Unsupported formula type {formula}")

    def _eval_term(self, env: Dict[str, Any], programs, term: Term) -> Any:
        if isinstance(term, (str, int, bool, float, tuple)) or term is None:
            return term
        if isinstance(term, list):
            return self._eval_list(env, programs, term)
        match term:
            case Var(name):
                return env[name]
            case Func() | SpecialValue():
                return term
            case Map(func, args):
                computed_args = self._eval_term(env, programs, args)
                if isinstance(computed_args, SpecialValue) or computed_args is None:
                    computed_args = [computed_args]
                if not isinstance(computed_args, list):
                    computed_args = [Demonic()]
                return [self._eval_term(env, programs, func([a])) for a in computed_args]
            case MapUnpack(func, args):
                computed_args = self._eval_term(env, programs, args)
                if isinstance(computed_args, SpecialValue):
                    computed_args = [computed_args]
                if not isinstance(computed_args, list):
                    computed_args = [Demonic()]
                return [self._eval_term(env, programs, func(a)) for a in computed_args]
            case App(func, args):
                return self._eval_app(env, programs, func, args)
            case _:
                raise NotImplementedError(f"This term type has not been implemented yet: {term}")
    

    def check(self,
              inputs: Dict[Side, Any],
              programs: Dict[Side, 'Program'],
              formula: Formula):
        max_num_inputs = max(len(inputs[Side.LEFT]), len(inputs[Side.RIGHT]))
        
        self.available_call_budget = EVALUATOR_CHECKER_CALL_BUDGET_PER_INPUT * max_num_inputs

        try:
            result = self._is_formula_true({}, inputs, programs, formula)
            assert isinstance(result, bool)
        except CallBudgetExceeded:
            print(f"[too many calls]", file=sys.stderr, flush=True)
            return False

        return result
