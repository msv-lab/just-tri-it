import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from functools import partial
import copy
import random

from just_tri_it.executor import Success, Error, Timeout
from just_tri_it.executor import Executor, EXECUTION_TIMEOUT_SECONDS
from just_tri_it.logic import (
    Side,
    Term, Var, Map, FlattenMap,
    Formula, App, Not, And, Or, Implies, Iff, ForAll,
    Func, Equals, SetEquals, OffByOne, Member, TolerateInvalid,
    SpecialValue, Angelic, Demonic, Undefined,
    FuncWithTimeoutGuard,
    FullAnswer, PartialAnswer
)
from just_tri_it.utils import hack
import just_tri_it.utils


class Checker(ABC):

    @abstractmethod
    def check(self,
              inputs: Dict[Side, Any],
              programs: Dict[Side, 'Program'],
              formula: Formula) -> bool:
        pass


INTERPRETER_CHECKER_CALL_BUDGET_PER_INPUT = 500

INTERPRETER_CHECKER_MAX_FORALL_DOMAIN = 50


class CallBudgetExceeded(Exception):
    "Raised when make too many program calls"
    pass


class Interpreter(Checker):

    def __init__(self, executor: Executor):
        self.executor = executor

    def _predicted_to_exceed_timeout(self, program, unchecked_input: list):
        i = copy.deepcopy(unchecked_input)
        i.append(EXECUTION_TIMEOUT_SECONDS)
        execution_outcome = self.executor.run(program.time_predicate, i)
        match execution_outcome:
            case Success(v):
                if v is True:
                    return True
            case _:
                pass
        return False

    def _eval_list(self, env, programs, terms):
        return list(map(partial(self._eval_term, env, programs), terms))

    def _eval_app(self, env, programs, func, args, timeout_guard=False):
        if isinstance(func, FuncWithTimeoutGuard):
            return self._eval_app(env, programs, func.inner, args, timeout_guard=True)
        computed_args = self._eval_term(env, programs, args)
        if not isinstance(computed_args, list):
            computed_args = [Demonic()]
        if isinstance(func.semantics, Side):
            special = [arg for arg in computed_args if isinstance(arg, SpecialValue)]
            if len(special) > 0:
                return SpecialValue.strongest(special)
            if self.available_call_budget <= 0:
                raise CallBudgetExceeded()
            if isinstance(func.semantics, Side):
                program = programs[func.semantics]
            if timeout_guard and program.time_predicate is not None:
                if self._predicted_to_exceed_timeout(program, computed_args):
                    return Angelic() # this is because fibers can contain values outside of the problem range
            if len(computed_args) != len(program.signature.params):
                print(f"\nArguments length mismatch: {program.display_id()}({computed_args})", file=sys.stderr, flush=True)
                return Demonic()
            execution_outcome = self.executor.run(program, computed_args)
            self.available_call_budget -= 1
            match execution_outcome:
                case Success(v):
                    if (just_tri_it.utils.DEBUG):
                        print(f"\n{program.display_id()}({computed_args}) => {v}", file=sys.stderr, flush=True)
                    return v
                case Error(error_type, error_msg) \
                    if error_type == "ValueError" and error_msg == "Invalid input":
                    if (just_tri_it.utils.DEBUG):
                        print(f"\n{program.display_id()}({computed_args}) => Invalid Input", file=sys.stderr, flush=True)
                    return Undefined()
                case _:
                    if (just_tri_it.utils.DEBUG):
                        print(f"\n{program.display_id()}({computed_args}) => Demonic()", file=sys.stderr, flush=True)
                    return Demonic()
        else:
            result = func.semantics(*computed_args)
            if (just_tri_it.utils.DEBUG):
                print(f"\n{func}({computed_args}) => {result}", file=sys.stderr, flush=True)
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
                if isinstance(domain, Side):
                    computed_domain = inputs[domain]
                else:
                    computed_domain = self._eval_term(env, programs, domain)
                    if isinstance(computed_domain, FullAnswer) or isinstance(computed_domain, PartialAnswer):
                        computed_domain = computed_domain.values
                    if isinstance(computed_domain, Angelic):
                        return True
                    if isinstance(computed_domain, Demonic) or isinstance(computed_domain, Undefined):
                        return False
                    if not isinstance(computed_domain, list):
                        computed_domain = [Demonic()]
                if len(computed_domain) == 0:
                    return True # not sure about it, but alternatives seem worse
                if len(computed_domain) > INTERPRETER_CHECKER_MAX_FORALL_DOMAIN:
                    random.seed(42) # each sample should be independent
                    computed_domain = list(random.sample(computed_domain, INTERPRETER_CHECKER_MAX_FORALL_DOMAIN))
                for inp in computed_domain:
                    new_env = env.copy()
                    if isinstance(ele, Var):
                        new_env[ele.name] = inp
                    else:
                        if len(inp) != len(ele):
                            print(f"\nInput length mismatch: {ele} {inp}", file=sys.stderr, flush=True)
                            return False
                        for index, var in enumerate(ele):
                            new_env[var.name] = inp[index]
                    result = self._is_formula_true(new_env, inputs, programs, body)
                    if isinstance(result, SpecialValue):
                        if SpecialValue.as_bool(result) is False:
                            if (just_tri_it.utils.DEBUG):
                                print(f"\n{formula} failed with {result} on {new_env}, left program: {programs[Side.LEFT].display_id()}, right program: {programs[Side.RIGHT].display_id()}", file=sys.stderr, flush=True)
                            return False
                        else:
                            num_angelic += 1
                    elif result is False:
                        if (just_tri_it.utils.DEBUG):
                            print(f"\n{formula} failed on {new_env}, left program: {programs[Side.LEFT].display_id()}, right program: {programs[Side.RIGHT].display_id()}", file=sys.stderr, flush=True)
                        return False
                if (num_angelic / len(computed_domain)) >= 0.34:
                    if (just_tri_it.utils.DEBUG):
                        print(f"\n{formula} failed due excessive angelic values", file=sys.stderr, flush=True)
                    return False
                return True
            case _:
                raise ValueError(f"Unsupported formula type {formula}")

    def _eval_term(self, env: Dict[str, Any], programs, term: Term) -> Any:
        if isinstance(term, (str, int, bool, float, tuple)) or term is None:
            return term
        if isinstance(term, list):
            result = self._eval_list(env, programs, term)
            return result
        match term:
            case Var(name):
                return env[name]
            case Func() | SpecialValue():
                return term
            case Map(func, args):
                computed_args = self._eval_term(env, programs, args)
                if isinstance(computed_args, SpecialValue):
                    computed_args = [computed_args]
                if not isinstance(computed_args, list):
                    computed_args = [Demonic()]
                return [self._eval_term(env, programs, func([a])) for a in computed_args]
            case FlattenMap(func, args):
                computed_args = self._eval_term(env, programs, args)
                if isinstance(computed_args, SpecialValue):
                    computed_args = [computed_args]
                if not isinstance(computed_args, list):
                    computed_args = [Demonic()]
                return [self._eval_term(env, programs, func([x])) for a in computed_args for x in a]
            case App(func, args):
                return self._eval_app(env, programs, func, args)
            case _:
                raise NotImplementedError(f"This term type has not been implemented yet: {term}")

    def check(self,
              inputs: Dict[Side, Any],
              programs: Dict[Side, 'Program'],
              formula: Formula):
        max_num_inputs = max(len(inputs[Side.LEFT]), len(inputs[Side.RIGHT]))

        self.available_call_budget = INTERPRETER_CHECKER_CALL_BUDGET_PER_INPUT * max_num_inputs

        try:
            result = self._is_formula_true({}, inputs, programs, formula)
            assert isinstance(result, bool)
        except CallBudgetExceeded:
            print(f"[too many calls]", file=sys.stderr, flush=True)
            return False

        return result
