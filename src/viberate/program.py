from dataclasses import dataclass
from typing import Self
import ast

from viberate.llm import extract_code, extract_answer


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
        return f"def {self.name}({params_str}) -> {self.return_type}:"

    @staticmethod
    def from_function_ast(fn_ast) -> Self:
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
    def from_requirements(model, req) -> Self:
        PROMPT_CODE = f"""
        For the problem below, write a Python function signature with:
        - Descriptive parameter names
        - Type annotations for all parameters
        - Specified return type
        
        Notice that the content required to print by the problem is the return value. 
        Please return only the function definition (with 'pass' as the body) inside a Markdown code block.

        Problem:
        {req}
        """
        code = extract_code(next(model.sample(PROMPT_CODE)))
        return Signature.from_function_ast(ast.parse(code).body[0])


@dataclass    
class Program:
    signature: Signature
    code: str
