import inspect
import sys
import math
import numbers
from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import Union, Set, Dict, List, Callable, Any
from copy import deepcopy

from just_tri_it.executor import Success, Error
from just_tri_it.utils import ExperimentFailure


CHECKER_CALL_BUDGET = 8000

available_call_budget = 0


class CallBudgetExceeded(Exception):
    "Raised when make too many program calls"
    pass    


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


class SpecialValue:
    pass


class Angelic(SpecialValue):
    pass


class Demonic(SpecialValue):
    pass


class Undefined(SpecialValue):
    pass


def has_special_value_adu(x: list) -> [bool, bool, bool]:
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
    global available_call_budget
    computed_args = eval_term(executor, env, programs, args)
    # print()
    # print("APPLY: " + str(func), flush=True)
    # print("ARGS: " + ", ".join(map(recursive_str, computed_args)), flush=True)
    if isinstance(func.semantics, Side):
        has_A, has_D, has_U = has_special_value_adu(computed_args)
        if has_D:
            return Demonic()
        if has_A:  # A, no D -> A
            return Angelic()
        if has_U:  # U, no D, no A -> U
            return Undefined()
        if available_call_budget <= 0:
            raise CallBudgetExceeded()
        execution_outcome = executor.run(programs[func.semantics], computed_args)
        available_call_budget -= 1
        # print("RESULT: " + str(execution_outcome), flush=True)
        match execution_outcome:
            case Success(v):
                return v
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
    if isinstance(origin, bool):
        return origin
    if isinstance(origin, Angelic):
        return True
    else:
        return False


def get_priority_from_list(values: List[SpecialValue], all_num=None) -> SpecialValue:
    has_demonic = any(isinstance(v, Demonic) for v in values)
    if has_demonic:
        return Demonic()
    has_undefined = any(isinstance(v, Undefined) for v in values)
    if has_undefined:
        return Undefined()
    angelic_count = sum(1 for v in values if isinstance(v, Angelic))
    if all_num and (angelic_count / all_num) > 0.34:
        return Demonic()
    return Angelic()


def is_formula_true(executor,
                    env: Dict[str, Any],
                    inputs: Dict[Side, Any],
                    programs: Dict[Side, 'Program'],
                    formula: Formula):
    match formula:
        case App(func, args):
            return eval_app(executor, env, programs, func, args)
        case Not(operand):
            result = is_formula_true(executor, env, inputs, programs, operand)
            if isinstance(result, bool):
                return not result
            return result
        case And(left, right):
            result_left = is_formula_true(executor, env, inputs, programs, left)
            result_right = is_formula_true(executor, env, inputs, programs, right)
            if isinstance(result_left, bool) and isinstance(result_right, bool):
                return result_left and result_right
            if isinstance(result_left, bool):
                # right is not bool
                if result_left is False:
                    return False
                return result_right
            if isinstance(result_right, bool):
                # left is not bool
                if result_right is False:
                    return False
                return result_left
            # right and left both not bool
            return get_priority_from_list([result_left, result_right])
        case Or(left, right):
            result_left = is_formula_true(executor, env, inputs, programs, left)
            result_right = is_formula_true(executor, env, inputs, programs, right)
            if isinstance(result_left, bool) and isinstance(result_right, bool):
                return result_left or result_right
            if isinstance(result_left, bool):
                # right is not bool
                if result_left is True:
                    return True
                return result_right
            if isinstance(result_right, bool):
                # left is not bool
                if result_right is True:
                    return True
                return result_left
            # right and left both not bool
            return get_priority_from_list([result_left, result_right])
        case ForAll(ele, domain, body):
            special_list = []
            all_number = len(inputs[domain])
            for inp in inputs[domain]:
                new_env = env.copy()
                if isinstance(ele, Var):
                    new_env[ele.name] = inp[0]
                else:
                    for index, var in enumerate(ele):
                        new_env[var.name] = inp[index]
                result = is_formula_true(executor, new_env, inputs, programs, body)
                if not isinstance(result, bool):
                    special_list.append(result)
                    continue
                if result is False:
                    print(f"\n{formula} failed on {inp}", file=sys.stderr, flush=True)
                    return False
            if len(special_list):
                return get_priority_from_list(special_list, all_number)
            return True
        case _:
            raise ValueError(f"Unsupported formula type {formula}")


def eval_term(executor, env: Dict[str, Any], programs, term: Term) -> Any:
    if isinstance(term, (str, int, bool, float, tuple)) or term is None:
        return term
    if isinstance(term, list):
        return eval_all(executor, env, programs, term)
    match term:
        case Var(name):
            return env[name]
        case Func() | SpecialValue():
            return term
        case Map(func, args):
            computed_args = eval_term(executor, env, programs, args)
            if isinstance(computed_args, SpecialValue) or computed_args is None:
                computed_args = [computed_args]
            return [eval_term(executor, env, programs, func([a])) for a in computed_args]
        case MapUnpack(func, args):
            computed_args = eval_term(executor, env, programs, args)
            if isinstance(computed_args, SpecialValue) or computed_args is None:
                computed_args = [computed_args]
            return [eval_term(executor, env, programs, func(a)) for a in computed_args]
        case App(func, args):
            return eval_app(executor, env, programs, func, args)
        case _:
            raise NotImplementedError(f"This term type has not been implemented yet: {term}")

        
def _off_by_one(x):
    match x:
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
    """Check equality, using math.isclose for floats."""
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
        elif isinstance(obj, tuple):
            return tuple(make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, set):
            return frozenset(make_hashable(item) for item in obj)
        elif isinstance(obj, bytearray):
            return bytes(obj)
        else:
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
    """Check membership, using math.isclose for floats."""
    if isinstance(x, Demonic):
        # because D!= anything
        return Demonic()
    if isinstance(y, SpecialValue):
        y = [y]
    if y is None:
        if x is None:
            return True
        else:
            return False
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
    transformed = deepcopy(origin)
    if isinstance(transformed, list):
        for index, item in enumerate(transformed):
            if isinstance(item, Undefined):
                transformed[index] = Angelic()
    else:
        if isinstance(transformed, Undefined):
            transformed = Angelic()
    return transformed


Tolerate = Func(_tolerate, "tolerate")


def check(executor, inputs: Dict[Side, Any], programs: Dict[Side, 'Program'], formula: Formula):
    global available_call_budget
    # print("\nLEFT:")
    # print(programs[Side.LEFT].get_content())
    # print("RIGHT:")
    # print(programs[Side.RIGHT].get_content())

    available_call_budget = CHECKER_CALL_BUDGET

    try:
        result = map_to_bool(is_formula_true(executor, {}, inputs, programs, formula))
    except CallBudgetExceeded:
        print(f"[too many calls]", file=sys.stderr, flush=True)
        return False

    return result
