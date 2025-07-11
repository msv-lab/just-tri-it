from dataclasses import dataclass
import ast


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
    def from_function_ast(fn_ast):
        params = []
        for arg in fn_ast.args.args:
            if arg.annotation is None:
                raise ValueError(f"Parameter {arg.arg} lacks type annotation")
            param_type = ast.unparse(arg.annotation)
            params.append(Parameter(arg.arg, param_type))
        if fn_ast.returns is None:
            raise ValueError("Function lacks return type annotation")
        return_type = ast.unparse(fn_ast.returns)
        return Signature(fn_ast.name, params, return_type)


@dataclass    
class Program:
    signature: Signature
    code: str
    
