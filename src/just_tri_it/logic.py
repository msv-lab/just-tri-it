import sys
import math
from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import Union, Set, Dict, List, Callable, Any

from just_tri_it.executor import Success
from just_tri_it.utils import ExperimentFailure


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


def recursive_str(obj):
    """Recursively converts any object into a single concatenated string."""
    if isinstance(obj, dict):
        return "{" + " ".join(recursive_str(k) + ": " + recursive_str(v) for k, v in obj.items()) + "}"
    elif isinstance(obj, set):
        return "{" + " ".join(recursive_str(item) for item in obj) + "}"
    elif isinstance(obj, tuple):
        return "(" + " ".join(recursive_str(item) for item in obj) + ")"
    elif isinstance(obj, list):
        return "[" + " ".join(recursive_str(item) for item in obj) + "]"
    elif isinstance(obj, str):
        return repr(obj)
    else:
        return str(obj)

@dataclass
class Var:
    name: str

    def __str__(self):
        return self.name    


@dataclass
class Not:
    operand: 'Formula'

    def __str__(self):
        return f"¬{self.operand}"


@dataclass
class And:
    left: 'Formula'
    right: 'Formula'

    def __str__(self):
        return f"({self.left} ∧ {self.right})"


@dataclass
class Or:
    left: 'Formula'
    right: 'Formula'

    def __str__(self):
        return f"({self.left} ∨ {self.right})"


Formula = Union[Var, Not, And, Or]


def get_vars(f: Formula) -> Set[str]:
    """Recursively extract variables from the formula."""
    match f:
        case Var(name):
            return {name}
        case Not(operand):
            return get_vars(operand)
        case And(left, right) | Or(left, right):
            return get_vars(left) | get_vars(right)


@dataclass
class Func:
    semantics: Callable | Side
    display: str = "OPAQUE"

    def __call__(self, args: List['Term']) -> 'App':
        return App(func=self, args=args)    

    def __str__(self):
        if self.semantics == Side.LEFT:
            return "left_func"
        elif self.semantics == Side.RIGHT:
            return "right_func"
        else:
            return self.display

@dataclass
class App:
    func: Func
    args: 'Term'

    def __str__(self):
        if isinstance(self.args, list):
            if len(self.func.display) <= 2 and len(self.args) == 2:
                return f"({recursive_str(self.args[0])} {self.func.display} {recursive_str(self.args[1])})"
            else:
                args_str = ", ".join(map(recursive_str, self.args))
                return f"{self.func}({args_str})"
        else:
            return f"{self.func}({recursive_str(self.args)})"


@dataclass
class Map:
    func: Func
    args: 'Term'

    def __str__(self):
        return f"map({str(self.func)}, {recursive_str(self.args)})"


@dataclass
class MapUnpack:
    func: Func
    args: 'Term'

    def __str__(self):
        return f"map*({str(self.func)}, {recursive_str(self.args)})"
    

Term = Union[Var, App, int, bool, float, str, list]


@dataclass
class Implies:
    left: 'Formula'
    right: 'Formula'

    def __str__(self):
        return f"({self.left} → {self.right})"


@dataclass
class Iff:
    left: 'Formula'
    right: 'Formula'


    def __str__(self):
        return f"({self.left} ↔ {self.right})"


@dataclass
class ForAll:
    vars: Var | list[Var]
    domain: Side
    body: 'Formula'

    def __str__(self):
        if isinstance(self.vars, list):
            vars_str = ", ".join(str(v) for v in self.vars)
        else:
            vars_str = str(self.vars)
        return f"∀{vars_str} ∈ {self.domain.value}_inputs: {self.body}"    


Formula = Union[
    App, # here is an application of a boolean function
    Not,
    And,
    Or,
    Implies,
    Iff,
    ForAll
]


# Error Model:
#
# - If a program throws an invalid input exception, its output is
#   considered Undefined.
#
# - If any part of any input is Undefined, then the output of 
#   the program is automatically Undefined.
#
# - Any values containing Undefined as their subcomponents are
#   considered Undefined.
#
# - Programs are only allowed to produce Undefined on non-Undefined
#   arguments if the arguments come from generated inputs, but not
#   intermediate values.
#
# - The only true predicates on Undefined values is Undefined =
#   Undefined and Undefined ∈ Undefined.

@dataclass
class Undefined():
    pass


class PropertyFalseDueToErrors(Exception):
    "Raised when the property is false due to errors"
    pass    


def eval_all(executor, env, programs, terms):
    result = list(map(partial(eval_term, executor, env, programs), terms))
    return result


def eval_app(executor, env, programs, func, args):
    computed_args = eval_term(executor, env, programs, args)
    # print()
    # print("APPLY: " + str(func), flush=True)
    # print("ARGS: " + ", ".join(map(recursive_str, computed_args)), flush=True)
    if isinstance(func.semantics, Side):
        execution_outcome = executor.run(programs[func.semantics], computed_args)
        # print("RESULT: " + str(execution_outcome), flush=True)
        match execution_outcome:
            case Success(v):
                return v
            case _:
                return None
    else:
        result = func.semantics(*computed_args)
        # print("RESULT: " + recursive_str(result), flush=True)
        return result


def is_formula_true(executor,
                    env: Dict[str, Any],
                    inputs: Dict[Side, Any],
                    programs: Dict[Side, 'Program'],
                    formula: Formula) -> bool:
    match formula:
        case App(func, args):
            return eval_app(executor, env, programs, func, args)
        case Not(operand):
            return is_formula_true(executor, env, inputs, programs, operand)
        case And(left, right):
            return is_formula_true(executor, env, inputs, programs, left) and \
                is_formula_true(executor, env, inputs, programs, right)
        case Or(left, right):
            return is_formula_true(executor, env, inputs, programs, left) or \
                is_formula_true(executor, env, inputs, programs, right)
        case ForAll(ele, domain, body):
            for inp in inputs[domain]:
                new_env = env.copy()
                if isinstance(ele, Var):
                    new_env[ele.name] = inp[0]
                else:
                    for index, var in enumerate(ele):
                        new_env[var.name] = inp[index]
                if not is_formula_true(executor, new_env, inputs, programs, body):
                    print(f"\n{formula} failed on {inp}", file=sys.stderr, flush=True)
                    return False
            return True
        case _:
            raise ValueError(f"Unsupported formula type {formula}")


def eval_term(executor, env: Dict[str, Any], programs, term: Term) -> Any:
    if isinstance(term, (str, int, bool, float)):
        return term
    if isinstance(term, list):
        return eval_all(executor, env, programs, term)
    match term:
        case Var(name):
            return env[name]
        case Func():
            return term
        case Map(func, args):
            computed_args = eval_term(executor, env, programs, args)
            return [eval_term(executor, env, programs, func([a])) for a in computed_args]
        case MapUnpack(func, args):
            computed_args = eval_term(executor, env, programs, args)
            return [eval_term(executor, env, programs, func(a)) for a in computed_args]
        case App(func, args):
            return eval_app(executor, env, programs, func, args)
        case _:
            raise NotImplementedError(f"This term type has not been implemented yet: {term}")

        
def _off_by_one(x):
    match x:
        case int():
            return x + 1
        case float():
            return x + 1.0
        case bool():
            return not x
        case str():
            return x + "_1"
        case list() if all(isinstance(i, int) for i in x):
            return x + [1]
        case _:
            raise ExperimentFailure(f"off-by-one does not support this type: {x}")

        
OffByOne = Func(_off_by_one, "off-by-one")


def _equals_func(x, y):
    """Check equality, using math.isclose for floats."""
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y)
    return x == y


Equals = Func(_equals_func, "=")


def _set_equals_func(x, y):
    """
    Check equality of two iterables.
    - If all elements are floats (in both x and y), compare sorted lists with math.isclose.
    - Otherwise, compare as sets.
    """
    x_list = list(x)
    y_list = list(y)

    if all(isinstance(v, float) for v in x_list + y_list):
        x_sorted = sorted(x_list)
        y_sorted = sorted(y_list)

        if len(x_sorted) != len(y_sorted):
            return False

        return all(math.isclose(a, b) for a, b in zip(x_sorted, y_sorted, strict=True))

    return set(x_list) == set(y_list)


SetEquals = Func(_set_equals_func, "=")


def _member_func(x, y):
    """Check membership, using math.isclose for floats."""
    for item in y:
        if isinstance(x, float) and isinstance(item, float):
            if math.isclose(x, item):
                return True
        elif x == item:
            return True
    return False


Member = Func(_member_func, "∈")


def check(executor, inputs: Dict[Side, Any], programs: Dict[Side, 'Program'], formula: Formula):
    # print("\nLEFT:")
    # print(programs[Side.LEFT].get_content())
    # print("RIGHT:")
    # print(programs[Side.RIGHT].get_content())
    result = is_formula_true(executor, {}, inputs, programs, formula)
    return result
