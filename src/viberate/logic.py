# This example can be executed with `uvx pytest example.py`

from dataclasses import dataclass
from functools import partial
from typing import Union, Set, Dict, List, Optional, Callable, Any, Iterable

import math

from viberate.executor import Success


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


def test_satisfiable():
    # (A ∨ B) ∧ (¬A ∨ ¬B)
    formula = And(
        Or(Var('A'), Var('B')),
        Or(Not(Var('A')), Not(Var('B')))
    )

    assert is_satisfiable(formula)


def test_unsatisfiable():
    # (A ∨ B) ∧ (¬A ∨ ¬B) ∧ (¬A ∨ B) ∧ (A ∨ ¬B)
    formula = And(
        And(
            Or(Var('A'), Var('B')),
            Or(Not(Var('A')), Not(Var('B')))
        ),
        And(
            Or(Not(Var('A')), Var('B')),
            Or(Var('A'), Not(Var('B')))
        )
    )

    assert not is_satisfiable(formula)


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
    domain: Set[tuple[Any, ...]]
    consts: Dict[str, Any]
    funcs: Dict[str, tuple[int, Callable[..., Any]]]
    preds: Dict[str, tuple[int, Callable[..., bool]]]


def eval_term(term: Term, interp: Interpretation, env: Dict[str, Any]) -> Any:
    match term:
        case Var(name):
            return env[name]
        case Const(name):
            return interp.consts[name]
        case Func(name, args):
            arity, f = interp.funcs[name]
            argvals = [eval_term(arg, interp, env) for arg in args]
            if len(argvals) != arity:
                raise ValueError(f"Function {name} expects {arity} arguments")
            outcome = f(argvals)
            # return outcome
            match outcome:
                case Success(output):
                    return output
                case _:
                    print("Facing error or panic or timeout!")
                    return False
        case FuncList(name, index, args):
            answer = set()
            all_ele = eval_term(args[index], interp, env)
            for ele in all_ele:
                new_args = args.copy()
                new_args[index] = Var("temp")
                new_env = env.copy()
                new_env["temp"] = ele
                outcome = eval_term(Func(name, new_args), interp, new_env)
                answer.add(outcome)
            return answer


# This function assume that there are no free variables in the formula:

def is_formula_true(formula: Formula, interp: Interpretation, env: Dict[str, Any]) -> bool:
    match formula:
        case Pred(name, args):
            arity, p = interp.preds[name]
            argvals = [eval_term(arg, interp, env) for arg in args]
            print(argvals)
            if len(argvals) != arity:
                raise ValueError(f"Predicate {name} expects {arity} arguments")
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
            for d in interp.domain:
                new_env = env.copy()
                new_env[var.name] = d
                if is_formula_true(body, interp, new_env):
                    return True
            return False


def test_first_order_true():
    interp = Interpretation(
        domain={(1,), (2,)},
        consts={"a": 1},
        funcs={},
        preds={
            "P": (1, lambda x: x == 2),
            "Q": (1, lambda x: x == 1),
        },
    )
    # ∀x. (x == 2 ∨ a == 1)
    formula = ForAll(Var("x"), Or(Pred("P", [Var("x")]), Pred("Q", [Const("a")])))
    assert is_formula_true(formula, interp, {})


def test_first_order_false():
    interp = Interpretation(
        domain={(1,), (2,)},
        consts={},
        funcs={},
        preds={
            "P": (2, lambda x, y: x == y)
        },
    )
    # ∀x. ∀y. (x == y)
    formula = ForAll(Var("x"),
                     ForAll(Var("y"),
                            Pred("P", [Var("x"), Var("y")])))
    assert not is_formula_true(formula, interp, {})


def add(inputs):
    x = inputs[0]
    y = inputs[1]
    return x + y


def sub(inputs):
    x = inputs[0]
    y = inputs[1]
    return x - y


# def test_for_inv_property(executor, forward, inverse, arity, generated_inputs, inverse_index):
def test_for_inv_property_demo(arity=2, inverse_index=0):
    interp = Interpretation(
        domain={(1, 2), (2, 3)},
        # generated_inputs,
        consts={},
        funcs={
            "f": (arity, add),
            # "f": (arity, partial(executor.run, forward)),
            "g": (arity, sub)
            # "g": (arity, partial(executor.run, inverse))
        },
        preds={
            "Equals": (2, lambda x, y: x == y)
        }
    )
    all_arg = []
    for i in range(arity):
        all_arg.append(Var(f"x_{i}"))
    new_arg = all_arg[:inverse_index] + all_arg[inverse_index + 1:]
    formula = ForAll(VarList("all_x", all_arg),
                     Pred("Equals", [
                         Var(f"x_{inverse_index}"),  # x_i
                         Func("g", [Func("f", all_arg.copy())] + new_arg.copy())  # g(...)
                     ])
                     )
    assert is_formula_true(formula, interp, {})


def test_for_inv_property(executor, forward, inverse, arity, generated_inputs, inverse_index):
    generated_inputs = set(tuple(sublist) for sublist in generated_inputs)
    interp = Interpretation(
        domain=generated_inputs,
        # generated_inputs,
        consts={},
        funcs={
            "f": (arity, partial(executor.run, forward)),
            "g": (arity, partial(executor.run, inverse))
        },
        preds={
            "Equals": (2, lambda x, y: x == y)
        }
    )
    all_arg = []
    for i in range(arity):
        all_arg.append(Var(f"x_{i}"))
    new_arg = all_arg[:inverse_index] + all_arg[inverse_index + 1:]
    formula = ForAll(VarList("all_x", all_arg),
                     Pred("Equals", [
                         Var(f"x_{inverse_index}"),  # x_i
                         Func("g", [Func("f", all_arg.copy())] + new_arg.copy())  # g(...)
                     ])
                     )
    return is_formula_true(formula, interp, {})


def square(inputs):
    x = inputs[0]
    return x*x


def inverse_square(inputs):
    x = inputs[0]
    if x == 0:
        return [0]
    else:
        return [math.sqrt(x), -math.sqrt(x)]


def test_for_fib_property_demo(arity=1, inverse_index=0):
    generated_inputs = {
        (5,), (3,)
    }
    interp = Interpretation(
        domain=generated_inputs,
        consts={},
        funcs={
            "f": (arity, square),
            "g": (arity, inverse_square)
        },
        preds={
            "Equals": (2, lambda x, y: x == y),
            "Includes": (2, lambda x, y: x in y)
        }
    )
    all_arg = []
    for i in range(arity):
        all_arg.append(Var(f"x_{i}"))
    new_arg = all_arg[:inverse_index] + all_arg[inverse_index + 1:]
    formula = ForAll(VarList("all_x", all_arg),
                     And(
                         Pred("Includes", [
                             Var(f"x_{inverse_index}"),  # x_i
                             Func("g", [Func("f", all_arg.copy())] + new_arg.copy())  # g(...)
                         ]),
                         Pred("Equals", [
                            Func("f", all_arg.copy()),
                            FuncList("f", 0, [Func("g", [Func("f", all_arg.copy())] + new_arg.copy())] + new_arg.copy())
                         ])
                     )
                     )
    return is_formula_true(formula, interp, {})


def test_for_fib_property(executor, forward, fiber, arity, generated_inputs, inverse_index):
    generated_inputs = set(tuple(sublist) for sublist in generated_inputs)
    print(generated_inputs)
    interp = Interpretation(
        domain=generated_inputs,
        consts={},
        funcs={
            "f": (arity, partial(executor.run, forward)),
            "g": (arity, partial(executor.run, fiber))
        },
        preds={
            "Equals": (2, lambda x, y: x == y),
            "Equals_set": (2, lambda x, y: {x} == y),
            "Includes": (2, lambda x, y: x in y)
        }
    )
    all_arg = []
    for i in range(arity):
        all_arg.append(Var(f"x_{i}"))
    new_arg = all_arg[:inverse_index] + all_arg[inverse_index + 1:]
    formula = ForAll(VarList("all_x", all_arg),
                     And(
                         Pred("Includes", [
                             Var(f"x_{inverse_index}"),  # x_i
                             Func("g", [Func("f", all_arg.copy())] + new_arg.copy())  # g(...)
                         ]),
                         Pred("Equals_set", [
                            Func("f", all_arg.copy()),
                            FuncList("f", 0, [Func("g", [Func("f", all_arg.copy())] + new_arg.copy())] + new_arg.copy())
                         ])
                     )
                     )
    return is_formula_true(formula, interp, {})
