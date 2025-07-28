from dataclasses import dataclass
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
    return_value: Parameter
    inverse_index: int = -1

    def pretty_print(self):
        params_str = ', '.join(p.pretty_print() for p in self.params)
        return f"def {self.name}({params_str}) -> {self.return_value.type}:"

    @staticmethod
    def from_function_ast(model, req, fn_ast, return_name):
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
        return_value = Parameter(return_name, return_type)
        new_sig = Signature(fn_ast.name, params, return_value)
        if len(params) == 1:
            inverse_index = 0
        else:
            PROMPT_INDEX = f"""
            For the problem below and its corresponding function signature, I need to invert this problem, so I need to select a parameter to swap with the return value first. Which parameter do you think would make the inverse problem easier to solve? Enclose the full name of this parameter in <answer> tags.

            Problem:
            {req}
            Function signature:
            {new_sig.pretty_print()}
            """
            inverse_index = int(temp_dict[extract_answer(next(model.sample(PROMPT_INDEX)))])
        new_sig.inverse_index = inverse_index
        return new_sig

    @staticmethod
    def from_requirements(model, req):
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
        PROMPT_NAME = f"""
        For the problem below, name its return value descriptively using Python's snake_case naming convention.
        Enclose the name in <answer> tags.
        
        Problem:
        {req}
        """
        code = extract_code(next(model.sample(PROMPT_CODE)))
        return_name = extract_answer(next(model.sample(PROMPT_NAME)))
        return Signature.from_function_ast(model, req, ast.parse(code).body[0], return_name)


@dataclass    
class Program:
    signature: Signature
    code: str
