import sys
import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Union, Set, Dict, List, Callable, Any, TypeVar, Generic, Tuple
from copy import deepcopy

from just_tri_it.utils import ExperimentFailure


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"

T = TypeVar('T')

@dataclass(frozen=True)
class FullAnswer(Generic[T]):
    values: List[T]

@dataclass(frozen=True)
class PartialAnswer(Generic[T]):
    values: List[T]


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


def get_vars(f: 'Formula') -> Set[str]:
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
    semantics: Callable | Side | Tuple[Side, Callable]
    display: str = "OPAQUE"

    def __call__(self, args: List['Term']) -> 'App':
        return App(func=self, args=args)    

    def __str__(self):
        if isinstance(self.semantics, Side):
            return f"{self.semantics.value}_program"
        if isinstance(self.semantics, tuple):
            return f"{self.semantics[0].value}_program_adapted"
        else:
            return self.display_id()

    def display_id(self):
        return self.display


@dataclass
class FuncWithTimeoutGuard:
    inner: Func

    def __call__(self, args: List['Term']) -> 'App':
        return App(func=self, args=args)    

    def __str__(self):
        return f"{self.inner}_with_timeout_guard"

    def display_id(self):
        return f"{self.inner.display_id()}%"
    

@dataclass
class App:
    func: Func | FuncWithTimeoutGuard
    args: 'Term'

    def __str__(self):
        if isinstance(self.args, list):
            if len(self.func.display_id()) <= 2 and len(self.args) == 2:
                return f"({recursive_str(self.args[0])} {self.func.display_id()} {recursive_str(self.args[1])})"
            else:
                args_str = ", ".join(map(recursive_str, self.args))
                return f"{self.func}({args_str})"
        else:
            return f"{self.func}({recursive_str(self.args)})"


@dataclass
class Map:
    func: Func | FuncWithTimeoutGuard
    args: 'Term'

    def __str__(self):
        return f"map({str(self.func)}, {recursive_str(self.args)})"


Term = Union[Var, App, Map, int, bool, float, str, list]


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
    domain: Side | Term | Tuple[Side, Callable]
    body: 'Formula'

    def __str__(self):
        if isinstance(self.vars, list):
            vars_str = ", ".join(str(v) for v in self.vars)
        else:
            vars_str = str(self.vars)
        if isinstance(self.domain, Side):
            domain_str = f"{self.domain.value}_inputs"
        elif isinstance(self.domain, tuple):
            domain_str = f"{self.domain[0].value}_adapted_inputs"
        else:
            domain_str = str(self.domain)
        return f"∀{vars_str} ∈ {domain_str}: {self.body}"    


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
    @staticmethod
    def as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, SpecialValue):
            return isinstance(value, Angelic)
        else:
            raise ValueError(f"cannot interpret as boolean: {value}")

    @staticmethod        
    def adu_in_list(values):
        has_A, has_D, has_U = False, False, False
        for ele in values:
            if isinstance(ele, Demonic):
                has_D = True
            elif isinstance(ele, Undefined):
                has_U = True
            elif isinstance(ele, Angelic):
                has_A = True
        return has_A, has_D, has_U

    @staticmethod
    def in_list(values):
        for value in values:
            if isinstance(value, SpecialValue):
                return True
        return False

    @staticmethod        
    def strongest(values: List['SpecialValue']) -> 'SpecialValue':
        if any(isinstance(v, Demonic) for v in values):
            return Demonic()
        if any(isinstance(v, Angelic) for v in values):
            return Angelic()
        if any(isinstance(v, Undefined) for v in values):
            return Undefined()
        raise ValueError("no special value found")


@dataclass
class Angelic(SpecialValue):
    pass

@dataclass
class Demonic(SpecialValue):
    pass

@dataclass
class Undefined(SpecialValue):
    pass


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
        return Demonic()
    if isinstance(x, Angelic) or isinstance(y, Angelic):
        return Angelic()
    if isinstance(x, Undefined) and isinstance(y, Undefined):
        return True
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y)
    return x == y


Equals = Func(_equals_func, "=")


def _set_equals_func(x, y):
    
    def make_hashable(obj):
        """convert lists into tuples (for hashability)"""
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

    if isinstance(x, SpecialValue):
        x = [x]
    if isinstance(y, SpecialValue):
        y = [y]
    x_list = list(x)
    y_list = list(y)

    x_has_A, x_has_D, x_has_U = SpecialValue.adu_in_list(x_list)
    y_has_A, y_has_D, y_has_U = SpecialValue.adu_in_list(y_list)

    if x_has_D or y_has_D:
        return Demonic()
    if x_has_A or y_has_A:
        return Angelic()
    if x_has_U and not y_has_U:
        return Undefined()
    if y_has_U and not x_has_U:
        return Undefined()

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
    def in_list(x, y):
        for item in y:
            if isinstance(x, float) and isinstance(item, float):
                if math.isclose(x, item):
                    return True
            elif x == item:
                return True
        return False    

    if isinstance(y, FullAnswer) and not isinstance(x, SpecialValue):
        return in_list(x, y.values)

    if isinstance(y, PartialAnswer) and not isinstance(x, SpecialValue):
        if in_list(x, y.values):
            return True
        else:
            return Angelic()

    if (isinstance(y, PartialAnswer) or isinstance(y, FullAnswer)) and \
       isinstance(x, Angelic):
        return True
   
    if isinstance(x, Undefined) and isinstance(y, Undefined):
        return True
    if isinstance(y, SpecialValue):
        y = [y]
    if not isinstance(y, list):
        return Demonic()
    if isinstance(x, Undefined) and \
       SpecialValue.in_list(y) and \
       isinstance(SpecialValue.strongest(y), Undefined):
        return True
    if SpecialValue.in_list([x] + y):
        return SpecialValue.strongest([x] + y)

    return in_list(x, y)


Member = Func(_member_func, "∈")


def _tolerate_invalid(origin):
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


TolerateInvalid = Func(_tolerate_invalid, "tolerate_invalid")


def TimeoutGuard(f: Func):
    assert isinstance(f, Func)
    return FuncWithTimeoutGuard(f)


def _full_or_partial(origin):
    """
    this function only transforms Undefined to Angelic
    """
    if isinstance(origin, list):
        return FullAnswer(origin)

    if isinstance(origin, SpecialValue):
        return origin
        
    if isinstance(origin, tuple) and \
       len(origin) == 2 and \
       isinstance(origin[0], bool) and \
       isinstance(origin[1], list):
        if origin[0]:
            return FullAnswer(origin[1])
        else:
            return PartialAnswer(origin[1])
    return Demonic()


FullOrPartial = Func(_full_or_partial, "full_or_partial")
