# This example can be executed with `uvx pytest example.py`
import hashlib
import json
from dataclasses import dataclass
from functools import partial
import random
from typing import Union, Set, Dict, List, Optional, Callable, Any, Iterable

import math

from viberate.code_generator import generate_specific_code
from viberate.executor import Success, Timeout


# This code defines classes representing propositional logic formulas:
# - A `Var` is a propositional variable (like "p", "q")
# - `Not` represents logical negation
# - `And` and `Or` represent the logical connectives

@dataclass
class Var:
    name: str


@dataclass
class VarList:
    name: str
    ele: List[Var]


@dataclass
class Not:
    operand: 'Formula'


@dataclass
class And:
    left: 'Formula'
    right: 'Formula'


@dataclass
class Or:
    left: 'Formula'
    right: 'Formula'


Formula = Union[Var, Not, And, Or]


# Such formulas are processed using recursive functions that match different constructors:

def get_vars(f: Formula) -> Set[str]:
    """Recursively extract variables from the formula."""
    match f:
        case Var(name):
            return {name}
        case Not(operand):
            return get_vars(operand)
        case And(left, right) | Or(left, right):
            return get_vars(left) | get_vars(right)


def eval_formula(f: Formula, assignment: Dict[str, bool]) -> bool:
    """Evaluate formula under an assignment."""
    match f:
        case Var(name):
            return assignment[name]
        case Not(operand):
            return not eval_formula(operand, assignment)
        case And(left, right):
            return eval_formula(left, assignment) and eval_formula(right, assignment)
        case Or(left, right):
            return eval_formula(left, assignment) or eval_formula(right, assignment)
        case _:
            raise ValueError("Unsupported formula type")


# Naive satisfiability checking can be implemented using a recursive function that
# enumerates and checks all possible assignments of propositional variables

def is_satisfiable(f: Formula) -> Optional[Dict[str, bool]]:
    """Find a satisfying assignment, or return None if unsatisfiable."""
    vars_ = sorted(get_vars(f))

    def inner(idx: int, assignment: Dict[str, bool]) -> Optional[Dict[str, bool]]:
        if idx == len(vars_):
            if eval_formula(f, assignment):
                return assignment.copy()
            else:
                return None
        v = vars_[idx]
        for val in [False, True]:
            assignment[v] = val
            result = inner(idx + 1, assignment)
            if result is not None:
                return result
        return None

    return inner(0, {})


# Now, we extend it to first-order logic. First-order logic formulas are
# constructed from predicates applied to Terms (`Term`) that can be
# variables (`Var`), constants (`Const`), or function applications
# (`Func`), where functions have a name and a list of arguments. Complex
# formulas are built by combining formulas using logical connectives
# (`Not`, `And`, `Or`, `Implies`, and `Iff`). Quantified formulas use
# `ForAll` (universal quantification) or `Exists` (existential quantification)

@dataclass
class Const:
    name: str


@dataclass
class Func:
    name: str
    args: List['Term']


@dataclass
class FuncList:
    name: str
    index: int
    args: List['Term']
    enum_arg: 'Term'


class UnknownValue:
    pass


UNKNOWN = UnknownValue()

Term = Union[Var, Const, Func, FuncList]


@dataclass
class Pred:
    name: str
    args: List[Term]


@dataclass
class Implies:
    left: 'Formula'
    right: 'Formula'


@dataclass
class Iff:
    left: 'Formula'
    right: 'Formula'


@dataclass
class ForAll:
    vars: Var | VarList
    body: 'Formula'


@dataclass
class Exists:
    var: Var
    body: 'Formula'


Formula = Union[
    Pred,
    Not,
    And,
    Or,
    Implies,
    Iff,
    ForAll,
    Exists,
]


# An interpretation defines a domain D and maps non-logical symbols to predicates, functions, and constants

@dataclass
class Interpretation:
    domain: Set[tuple[Any, ...]] | List[tuple[Any, ...]]
    consts: Dict[str, Any]
    funcs: Dict[str, tuple[int, Callable[..., Any]]]
    preds: Dict[str, tuple[int, Callable[..., bool]]]


# def eval_term(term: Term, interp: Interpretation, env: Dict[str, Any]) -> Any:
#     match term:
#         case Var(name):
#             return env[name]
#         case Const(name):
#             return interp.consts[name]
#         case Func(name, args):
#             arity, f = interp.funcs[name]
#             argvals = []
#             for arg in args:
#                 new_val = eval_term(arg, interp, env)
#                 if new_val == UNKNOWN:
#                     return UNKNOWN
#                 argvals.append(new_val)
#             print("run " + name + " argvals are")
#             print(argvals)
#             if len(argvals) != arity:
#                 raise ValueError(f"Function {name} expects {arity} arguments")
#             outcome = f(argvals)
#             # return outcome
#             match outcome:
#                 case Success(output):
#                     print("answer")
#                     print(output)
#                     return output
#                 case Timeout():
#                     return UNKNOWN
#                 case _:
#                     print("Facing error or panic or timeout!")
#                     print(outcome)
#                     return False
#         case FuncList(name, index, args, enum_arg):
#             answer = []
#             enum_ele = eval_term(enum_arg, interp, env)
#             if enum_ele == UNKNOWN:
#                 return UNKNOWN
#             if len(enum_ele) > 10:
#                 enum_ele = random.sample(enum_ele, 10)
#             for ele in enum_ele:
#                 new_args = args.copy()
#                 new_args[index] = Var("temp")
#                 new_env = env.copy()
#                 new_env["temp"] = ele
#                 outcome = eval_term(Func(name, new_args), interp, new_env)
#                 if outcome == UNKNOWN:
#                     continue
#                 if outcome not in answer:
#                     answer.append(outcome)
#             return answer


# This function assume that there are no free variables in the formula:

def is_formula_true(formula: Formula, interp: Interpretation, env: Dict[str, Any], cache=None) -> bool:
    match formula:
        case Pred(name, args):
            arity, p = interp.preds[name]
            argvals = []
            for arg in args:
                new_val = eval_term_cache(arg, interp, env, cache)
                if new_val == UNKNOWN:
                    return True
                argvals.append(new_val)
            print(name, argvals)
            if len(argvals) != arity:
                raise ValueError(f"Predicate {name} expects {arity} arguments")
            print(p(*argvals))
            return p(*argvals)
        case Not(operand):
            return not is_formula_true(operand, interp, env)
        case And(left, right):
            return is_formula_true(left, interp, env) and is_formula_true(right, interp, env)
        case Or(left, right):
            return is_formula_true(left, interp, env) or is_formula_true(right, interp, env)
        case Implies(left, right):
            return not is_formula_true(left, interp, env) or is_formula_true(right, interp, env)
        case Iff(left, right):
            return is_formula_true(left, interp, env) == is_formula_true(right, interp, env)
        case ForAll(vars, body):
            for d in interp.domain:
                new_env = env.copy()
                if isinstance(vars, Var):
                    new_env[vars.name] = d[0]
                else:
                    for index, var in enumerate(vars.ele):
                        new_env[var.name] = d[index]
                if not is_formula_true(body, interp, new_env):
                    return False
            return True
        case Exists(var, body):
            # unfinished
            for d in interp.domain:
                new_env = env.copy()
                new_env[var.name] = d
                if is_formula_true(body, interp, new_env):
                    return True
            return False


def term_to_id(term: Term, env: Dict[str, Any], interp: Interpretation) -> str:
    def encode(t: Term):
        match t:
            case Var(name):
                return {"Var": (name, env.get(name))}
            case Const(name):
                return {"Const": (name, interp.consts.get(name))}
            case Func(name, args):
                return {"Func": (name, [encode(arg) for arg in args])}
            case FuncList(name, index, args, enum_arg):
                return {"FuncList": (name, index,
                                     [encode(arg) for arg in args],
                                     encode(enum_arg))}
            case _:
                return {"Unknown": str(t)}

    raw = json.dumps(encode(term), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def eval_term_cache(term: Term,
                    interp: Interpretation,
                    env: Dict[str, Any],
                    cache: Optional[Dict[str, Any]] = None) -> Any:
    if cache is None:
        cache = {}

    key = term_to_id(term, env, interp)

    if key in cache:
        return cache[key]

    match term:
        case Var(name):
            result = env[name]
        case Const(name):
            result = interp.consts[name]
        case Func(name, args):
            arity, f = interp.funcs[name]
            argvals = []
            for arg in args:
                new_val = eval_term_cache(arg, interp, env, cache)
                if new_val == UNKNOWN:
                    return UNKNOWN
                argvals.append(new_val)
            if len(argvals) != arity:
                raise ValueError(f"Function {name} expects {arity} arguments")
            outcome = f(argvals)
            match outcome:
                case Success(output):
                    result = output
                case Timeout():
                    return UNKNOWN
                case _:
                    result = False
        case FuncList(name, index, args, enum_arg):
            answer = []
            enum_ele = eval_term_cache(enum_arg, interp, env, cache)
            if enum_ele == UNKNOWN:
                return UNKNOWN
            if len(enum_ele) > 10:
                enum_ele = random.sample(enum_ele, 10)
            for ele in enum_ele:
                new_args = args.copy()
                new_args[index] = Var("temp")
                new_env = env.copy()
                new_env["temp"] = ele
                outcome = eval_term_cache(Func(name, new_args), interp, new_env, cache)
                if outcome == UNKNOWN:
                    continue
                if outcome not in answer:
                    answer.append(outcome)
            result = answer
        case _:
            result = UNKNOWN

    if result != UNKNOWN:
        cache[key] = result
    return result


def general_checker(formula, executor, program_1, program_2, arity, generated_inputs):
    generated_inputs = list(tuple(sublist) for sublist in generated_inputs)
    interp = Interpretation(
        domain=generated_inputs,
        consts={},
        funcs={
            "f": (arity, partial(executor.run, program_1)),
            "g": (arity, partial(executor.run, program_2))
        },
        preds={
            "Equals": (2, lambda x, y: x == y),
            "Equals_set": (2, lambda x, y: [x] == y),
            "Includes": (2, lambda x, y: x in y if isinstance(y, Iterable) else False)
        }
    )
    return is_formula_true(formula, interp, {}, {})


def new_general_checker(formula, executor, program_1, arity, generated_inputs, model, req, choice, num, n2):
    generated_inputs = list(tuple(sublist) for sublist in generated_inputs)
    interp = Interpretation(
        domain=generated_inputs,
        consts={},
        funcs={
            "f": (arity, partial(executor.run, program_1)),
            "g": (arity, partial(generate_specific_code, executor, model, req, choice, num, n2))
        },
        preds={
            "Equals": (2, lambda x, y: x == y),
            "Equals_set": (2, lambda x, y: [x] == y),
            "Includes": (2, lambda x, y: x in y if isinstance(y, Iterable) else False)
        }
    )
    return is_formula_true(formula, interp, {}, {})
