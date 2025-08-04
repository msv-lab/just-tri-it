from dataclasses import dataclass
import ast
from typing import Any, List

from viberate.llm import extract_code


@dataclass
class Parameter:
    name: str
    type: str

    def pretty_print(self):
        return self.name + f": {self.type}"


@dataclass
class Signature:
    name: str
    params: list[Parameter]
    return_type: str

    def pretty_print(self):
        params_str = ', '.join(p.pretty_print() for p in self.params)
        return f"def {self.name}({params_str}) -> {self.return_type}"

    @staticmethod
    def from_function_ast(fn_ast) -> 'Signature':
        params = []
        temp_dict = {}
        for index, arg in enumerate(fn_ast.args.args):
            if arg.annotation is None:
                raise ValueError(f"Parameter {arg.arg} lacks type annotation")
            param_type = ast.unparse(arg.annotation)
            params.append(Parameter(arg.arg, param_type))
            temp_dict[arg.arg] = index
        if fn_ast.returns is None:
            raise ValueError("Function lacks return type annotation")
        return_type = ast.unparse(fn_ast.returns)
        return Signature(fn_ast.name, params, return_type)

    @staticmethod
    def from_description(model, desc: str) -> 'Signature':
        PROMPT_CODE = f"""
        For the problem below, write a Python function signature with:
        - Descriptive parameter names
        - Type annotations for all parameters
        - Specified return type
        
        If the problem description instructs to read from stdin or
        write to stdout, please ignore. All inputs should be
        explicitly represented as parameters, and the output as the
        return value. Please return only the function definition (with
        'pass' as the body) inside a Markdown code block.

        Problem:
        {desc}
        """
        code = extract_code(next(model.sample(PROMPT_CODE)))
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return Signature.from_function_ast(node)
        raise ValueError("No function definition found in code")        
 

@dataclass    
class Program:
    signature: Signature
    code: str

    def __str__(self):
        return self.code

    def add_imports(self, imports) -> 'Program':
        return Program(self.signature, imports + "\n" + self.code)


@dataclass
class ExpectedOutput:
    value: Any


@dataclass
class Assertion:
    '''A predicate expression of the arity 1 represented as a string'''
    code: str
    

type Oracle = ExpectedOutput | Assertion


@dataclass
class Test:
    inputs: List[Any]
    oracle: Oracle
