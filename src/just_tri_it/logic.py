import inspect
import sys
import math
import numbers
from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import Union, Set, Dict, List, Callable, Any

from just_tri_it.executor import Success, Error
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

# change 1: define 3 new classes


class SpecialValue:
    pass


class Angelic(SpecialValue):
    pass


class Demonic(SpecialValue):
    pass


class Undefined(SpecialValue):
    pass


def has_special_value_adu(x: list) -> [bool, bool, bool]:
    # change 2: define a new function
    """
    this function checks:
    if in list x, there exists a special value A(Angelic)/D(Demonic)/U(Undefined)
    """
    has_A, has_D, has_U = False, False, False
    for ele in x:
        if isinstance(ele, Demonic):
            has_D = True
        elif isinstance(ele, Undefined):
            has_U = True
        elif isinstance(ele, Angelic):
            has_A = True
    return has_A, has_D, has_U


def eval_all(executor, env, programs, terms):
    result = list(map(partial(eval_term, executor, env, programs), terms))
    return result


def eval_app(executor, env, programs, func, args):
    computed_args = eval_term(executor, env, programs, args)
    # print()
    # print("APPLY: " + str(func), flush=True)
    # print("ARGS: " + ", ".join(map(recursive_str, computed_args)), flush=True)
    if isinstance(func.semantics, Side):
        # change 4: if there exists any special value in 'computed_args', the function can't be executed
        has_A, has_D, has_U = has_special_value_adu(computed_args)
        if has_D:
            return Demonic()
        if has_U:  # no D
            return Undefined()
        if has_A:  # no U no D -> only A
            return Angelic()
        execution_outcome = executor.run(programs[func.semantics], computed_args)
        # print("RESULT: " + str(execution_outcome), flush=True)
        match execution_outcome:
            case Success(v):
                return v
            # change 5: clarify when to return Undefined and Demonic
            case Error(error_type, error_msg) if error_type == "ValueError" and error_msg == "Invalid input":
                return Undefined()
            case _:
                return Demonic()
    else:
        result = func.semantics(*computed_args)
        # print("RESULT: " + recursive_str(result), flush=True)
        return result


def map_to_bool(origin):
    """
    this function map origin to bool if origin is not bool but a special value
    """
    # change 6: define a new function
    if isinstance(origin, bool):
        return origin
    if isinstance(origin, Angelic):
        return True
    else:
        return False


def is_formula_true(executor,
                    env: Dict[str, Any],
                    inputs: Dict[Side, Any],
                    programs: Dict[Side, 'Program'],
                    formula: Formula) -> bool:
    match formula:
        case App(func, args):
            return eval_app(executor, env, programs, func, args)
        # change 7: 'result' now can be a boolean or a special value
        case Not(operand):
            result = is_formula_true(executor, env, inputs, programs, operand)
            if not isinstance(result, bool):
                return map_to_bool(result)
            return not result
        case And(left, right):
            result_left = map_to_bool(is_formula_true(executor, env, inputs, programs, left))
            result_right = map_to_bool(is_formula_true(executor, env, inputs, programs, right))
            return result_left and result_right
        case Or(left, right):
            result_left = map_to_bool(is_formula_true(executor, env, inputs, programs, left))
            result_right = map_to_bool(is_formula_true(executor, env, inputs, programs, right))
            return result_left or result_right
        case ForAll(ele, domain, body):
            for inp in inputs[domain]:
                new_env = env.copy()
                if isinstance(ele, Var):
                    new_env[ele.name] = inp[0]
                else:
                    for index, var in enumerate(ele):
                        new_env[var.name] = inp[index]
                result = map_to_bool(is_formula_true(executor, new_env, inputs, programs, body))
                if result is False:
                    print(f"\n{formula} failed on {inp}", file=sys.stderr, flush=True)
                    return False
            return True
        case _:
            raise ValueError(f"Unsupported formula type {formula}")


def eval_term(executor, env: Dict[str, Any], programs, term: Term) -> Any:
    if isinstance(term, (str, int, bool, float, tuple)):
        return term
    if isinstance(term, list):
        return eval_all(executor, env, programs, term)
    match term:
        case Var(name):
            return env[name]
        # change 14: if term is special value
        case Func() | SpecialValue():
            return term
        case Map(func, args):
            computed_args = eval_term(executor, env, programs, args)
            if isinstance(computed_args, SpecialValue):
                computed_args = [computed_args]
            return [eval_term(executor, env, programs, func([a])) for a in computed_args]
        case MapUnpack(func, args):
            computed_args = eval_term(executor, env, programs, args)
            if isinstance(computed_args, SpecialValue):
                computed_args = [computed_args]
            return [eval_term(executor, env, programs, func(a)) for a in computed_args]
        case App(func, args):
            return eval_app(executor, env, programs, func, args)
        case _:
            raise NotImplementedError(f"This term type has not been implemented yet: {term}")

        
def _off_by_one(x):
    match x:
        # change 15: x can be special value
        case SpecialValue():
            return x
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
        case tuple() if all(isinstance(i, int) for i in x):
            return x + (1,)
        case _:
            raise ExperimentFailure(f"off-by-one does not support this type: {x}")

        
OffByOne = Func(_off_by_one, "off-by-one")


def _equals_func(x, y):
    # print("equal", x, y)
    """Check equality, using math.isclose for floats."""
    # change 8: x, y here can be special values
    if isinstance(x, Demonic) or isinstance(y, Demonic):
        # D != anything
        return Demonic()
    if isinstance(x, Angelic) or isinstance(y, Angelic):  # no Demonic
        # A == anything(apart from D)
        return Angelic()
    if isinstance(x, Undefined) and isinstance(y, Undefined):  # no Demonic and no Angelic
        # U = U
        return True
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y)
    return x == y


Equals = Func(_equals_func, "=")


def _set_equals_func(x, y):
    # Helper: convert lists into tuples (for hashability)
    def make_hashable(obj):
        if isinstance(obj, list):
            return tuple(make_hashable(item) for item in obj)
        return obj

    # Wrap single SpecialValue into a list
    if isinstance(x, SpecialValue):
        x = [x]
    if isinstance(y, SpecialValue):
        y = [y]
    x_list = list(x)
    y_list = list(y)

    # Check for special values A/D/U
    x_has_A, x_has_D, x_has_U = has_special_value_adu(x_list)
    y_has_A, y_has_D, y_has_U = has_special_value_adu(y_list)

    if x_has_D or y_has_D:
        return Demonic()
    if x_has_A or y_has_A:
        return Angelic()
    if x_has_U and not y_has_U:
        return False
    if y_has_U and not x_has_U:
        return False

    # Remove special values, keep only normal elements
    x_list = [ele for ele in x_list if not isinstance(ele, SpecialValue)]
    y_list = [ele for ele in y_list if not isinstance(ele, SpecialValue)]

    # Convert unhashable objects (lists) into hashable ones (tuples)
    x_hashable = [make_hashable(item) for item in x_list]
    y_hashable = [make_hashable(item) for item in y_list]

    # If all elements are numeric, use math.isclose
    if all(isinstance(v, numbers.Real) for v in x_hashable + y_hashable):
        def unique_by_isclose(values):
            values_sorted = sorted(values)
            result = []
            for v in values_sorted:
                if not result or not math.isclose(v, result[-1]):
                    result.append(v)
            return result

        x_unique = unique_by_isclose(x_hashable)
        y_unique = unique_by_isclose(y_hashable)

        if len(x_unique) != len(y_unique):
            return False

        return all(math.isclose(a, b) for a, b in zip(x_unique, y_unique))

    # Fallback: compare as sets
    return set(x_hashable) == set(y_hashable)


SetEquals = Func(_set_equals_func, "=")


def _member_func(x, y):
    # print("member", x, y)
    """Check membership, using math.isclose for floats."""
    # change 11: x, y here can be special values
    if isinstance(x, Demonic):
        # because D!= anything
        return Demonic()
    if isinstance(y, SpecialValue):
        y = [y]
    y_has_A, _, _ = has_special_value_adu(y)
    if y_has_A:
        # if y has A, we can make sure x = A
        return Angelic()
    if isinstance(x, Undefined):
        return False
    if isinstance(x, Angelic):
        return Angelic()

    for item in y:
        if isinstance(x, float) and isinstance(item, float):
            if math.isclose(x, item):
                return True
        elif x == item:
            return True
    return False


Member = Func(_member_func, "∈")


def _tolerate(origin):
    """
    this function only transforms Undefined to Angelic
    """
    # change 12: define a new function
    if isinstance(origin, list):
        for index, item in enumerate(origin):
            if isinstance(item, Undefined):
                origin[index] = Angelic()
    else:
        if isinstance(origin, Undefined):
            origin = Angelic()
    return origin


Tolerate = Func(_tolerate, "tolerate")


def check(executor, inputs: Dict[Side, Any], programs: Dict[Side, 'Program'], formula: Formula):
    # print("\nLEFT:")
    # print(programs[Side.LEFT].get_content())
    # print("RIGHT:")
    # print(programs[Side.RIGHT].get_content())

    # change 13: map special values to bool to get final boolean answer
    result = map_to_bool(is_formula_true(executor, {}, inputs, programs, formula))
    return result
