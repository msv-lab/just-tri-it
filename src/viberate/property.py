from abc import ABC
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Clause(ABC):
    pass


@dataclass
class Var:
    name: str


@dataclass
class Domain:
    name: str


@dataclass
class FuncCall(Var):
    args: List[Any]


@dataclass
class Eq(Clause):
    left: Any
    right: Any


@dataclass
class In(Clause):
    elem: Any
    set_: Any


@dataclass
class And(Clause):
    clauses: List[Any]


@dataclass
class Or(Clause):
    clauses: List[Any]


@dataclass
class ForAll(Clause):
    var: List[Any]
    domain: Domain
    body: Any


def f(vs: List[Any]) -> FuncCall:
    return FuncCall("f", list(vs))


def g(vs: List[Any]) -> FuncCall:
    return FuncCall("g", list(vs))


def eq(a: Any, b: Any) -> Eq:
    return Eq(a, b)


def member(a: Any, s: Any) -> In:
    return In(a, s)


def con(*clauses: Any) -> And:
    return And(list(clauses))


def dis(*clauses: Any) -> Or:
    return Or(list(clauses))


def forall(v: List[Any], domain: Domain, body: Any) -> ForAll:
    return ForAll(v, domain, body)


@dataclass
class Property:
    formula: Clause
    params_num: int
    partial_index: int

    def check(self, generated_inputs):
        # unfinished
        pass


@dataclass
class ForInv(Property):
    def __init__(self):
        var_list = []
        for i in range(self.params_num):
            var_list.append(Var("var_a_" + str(i)))
        A = Domain("A")
        g_param = [f(var_list), var_list[:self.partial_index] + var_list[self.partial_index+1:]]
        clause = eq(g(g_param), var_list)
        self.formula = forall(var_list, A, clause)


@dataclass
class ForFib(Property):
    def __init__(self):
        var_list = []
        for i in range(self.params_num):
            var_list.append(Var("var_a_" + str(i)))
        A = Domain("A")
        output_f = f(var_list)
        g_param = [output_f, var_list[:self.partial_index] + var_list[self.partial_index+1:]]
        f_param = var_list[:self.partial_index] + var_list[self.partial_index+1:]
        output_g = g(g_param)
        f_param.insert(self.partial_index, output_g)
        clause1 = eq(f(f_param), output_f)
        clause2 = member(var_list[self.partial_index], output_g)
        self.formula = forall(var_list, A, con(clause1, clause2))

