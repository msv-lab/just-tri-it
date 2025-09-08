# This example can be executed with `uvx pytest example.py`
import hashlib
import json
from dataclasses import dataclass
from functools import partial
import random
from typing import Union, Set, Dict, List, Optional, Callable, Any, Iterable
from viberate.code_generator import generate_specific_code
from viberate.executor import Success, Timeout, Error


@dataclass
class Var:
    name: str


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


@dataclass
class UnknownValue:
    reason: str


TIMEOUT = UnknownValue('timeout')
INVALID_IN = UnknownValue('invalid input')
# except timout and invalid_input errors
ERROR = UnknownValue('other errors when evaluating')

# only for is_formula_true: UNKNOWN, True, False
# UNKNOWN will not affect the result of AND, OR...
UNKNOWN = UnknownValue('unknown value for boolean')


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
    vars: Var | list[Var]
    domain: Term | Set[tuple[Any, ...]] | List[tuple[Any, ...]]
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


@dataclass
class Interpretation:
    consts: Dict[str, Any]
    funcs: Dict[str, tuple[int, Callable[..., Any]]]
    preds: Dict[str, tuple[int, Callable[..., bool]]]


def is_formula_true(formula: Formula, interp: Interpretation, env: Dict[str, Any], cache=None) -> bool | UnknownValue:
    match formula:
        case Pred(name, args):
            arity, p = interp.preds[name]
            arg_vals = []
            for arg in args:
                new_val = eval_term_cache(arg, interp, env, cache)
                if new_val == TIMEOUT:
                    return UNKNOWN
                elif new_val == ERROR:  # if we meet ERROR, the pred should be definitely false
                    return False
                # only INVALID_IN can be compared
                arg_vals.append(new_val)
            # print(name, arg_vals)
            if len(arg_vals) != arity:
                raise ValueError(f"Predicate {name} expects {arity} arguments")
            return p(*arg_vals)
        case Not(operand):
            ans = is_formula_true(operand, interp, env)
            if ans == UNKNOWN:
                return UNKNOWN
            else:
                return not ans
        case And(left, right):
            ans_1 = is_formula_true(left, interp, env)
            ans_2 = is_formula_true(right, interp, env)
            if ans_1 == UNKNOWN and ans_2 == UNKNOWN:
                return UNKNOWN
            elif ans_1 == UNKNOWN:
                return ans_2
            elif ans_2 == UNKNOWN:
                return ans_1
            else:
                return ans_1 and ans_2
        case Or(left, right):
            ans_1 = is_formula_true(left, interp, env)
            ans_2 = is_formula_true(right, interp, env)
            if ans_1 == UNKNOWN and ans_2 == UNKNOWN:
                return UNKNOWN
            elif ans_1 == UNKNOWN:
                return ans_2
            elif ans_2 == UNKNOWN:
                return ans_1
            else:
                return ans_1 or ans_2
        case ForAll(ele, domain, body):
            if isinstance(domain, Term):
                domain = eval_term_cache(domain, interp, env, cache)
            if domain == ERROR:
                return False
            elif domain == TIMEOUT or domain == INVALID_IN:
                return UNKNOWN
            if not isinstance(domain, List):
                domain = [domain]
            has_value = False
            for d in domain:
                new_env = env.copy()
                if isinstance(ele, Var):
                    new_env[ele.name] = d[0]
                else:
                    for index, var in enumerate(ele):
                        new_env[var.name] = d[index]
                ans = is_formula_true(body, interp, new_env)
                if ans is False:
                    return False
                elif ans is True:
                    has_value = True
            if has_value:
                return True
            else:
                return UNKNOWN
        case _:
            raise ValueError(f"Unsupported formula type")


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
                if new_val == TIMEOUT or new_val == INVALID_IN or new_val == ERROR:
                    cache[key] = new_val
                    return new_val
                argvals.append(new_val)
            if len(argvals) != arity:
                raise ValueError(f"Function {name} expects {arity} arguments")
            outcome = f(argvals)
            match outcome:
                case Success(output):
                    result = output
                case Timeout():
                    result = TIMEOUT
                case Error("ValueError", "Invalid input"):
                    result = INVALID_IN
                case _:
                    result = ERROR
        case FuncList(name, index, args, enum_arg):
            answer = []
            enum_ele = eval_term_cache(enum_arg, interp, env, cache)
            if enum_ele == TIMEOUT or enum_ele == INVALID_IN or enum_ele == ERROR:
                cache[key] = enum_ele
                return enum_ele
            if not isinstance(enum_ele, List):
                enum_ele = [enum_ele]
            if len(enum_ele) > 10:
                enum_ele = random.sample(enum_ele, 10)
            for ele in enum_ele:
                new_args = args.copy()
                new_args[index] = Var("temp")
                new_env = env.copy()
                new_env["temp"] = ele
                outcome = eval_term_cache(Func(name, new_args), interp, new_env, cache)
                if outcome == TIMEOUT:  # if we meet ERROR or INVALID_IN, it can't be ignored
                    continue
                if outcome not in answer:
                    answer.append(outcome)
            result = answer
        case _:
            raise NotImplementedError("This term type has not been implemented yet")

    cache[key] = result
    return result


def checker(formula: Formula, funcs: list[Callable], arity):
    interp = Interpretation(
        consts={},
        funcs={
            "f": (arity, funcs[0]),
            "g": (arity, funcs[1])
        },
        preds={
            "Equals": (2, lambda x, y: x == y),
            "Equals_set": (2, lambda x, y: [x] == y),
            "Includes": (2, lambda x, y: x in y if isinstance(y, Iterable) else False)
        }
    )
    ans = is_formula_true(formula, interp, {}, {})
    if ans is True or ans is UNKNOWN:
        return True
    else:
        return False
