from dataclasses import dataclass
import ast
import re
import hashlib
import sys
from typing import Any, List, Tuple
from itertools import islice

from viberate.cached_llm import Model
from viberate.utils import (
    extract_code, extract_all_code, RawData,
    print_annotated_hr, extract_answer, gen_and_extract_answer_with_retry,
    ContentAddressable
)


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
class TestFunction(ContentAddressable):
    test_function_name: str
    test_function_code: str
    target_signature: Signature

    def get_content(self) -> str:
        return "# signature: " + self.target_signature.pretty_print() + "\n"\
            + "# test name: " + self.test_function_name + "\n"\
            + self.test_function_code
    
    @staticmethod
    def from_code(code: str, target_signature: Signature) -> 'TestFunction':
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                return TestFunction(
                    test_function_name=node.name,
                    test_function_code=code,
                    target_signature=target_signature,
                )
        raise ValueError("No test function found in code")


@dataclass
class InputOutput(ContentAddressable):
    inputs: List[Any]
    output: Any

    def get_content(self) -> str:
        return "inputs: " + repr(self.inputs) + "\n"\
            + "output: " + repr(self.output)


type Test = TestFunction | InputOutput


@dataclass
class Pass:
    pass
    

@dataclass
class Fail:
    pass


@dataclass
class Requirements(ContentAddressable):
    signature: Signature
    description: str

    def get_content(self) -> str:
        return "# signature: " + self.signature.pretty_print() + "\n" + self.description

    @staticmethod
    def from_description(model: Model, desc: str) -> 'Requirements':
        return Requirements(Signature.from_description(model, desc), desc)


@dataclass
class NamedReturnSignature(Signature):
    return_name: str

    @staticmethod
    def infer_name(model: Model, req: Requirements) -> 'NamedReturnSignature':
        PROMPT = f"""
For the problem below, name its return value descriptively
using Python's snake_case naming convention. Enclose the
name in <answer> tags.
        
Problem:
{req.description}
        """
        valid_name = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        return NamedReturnSignature(req.signature.name,
                                    req.signature.params,
                                    req.signature.return_type,
                                    valid_name)


@dataclass    
class Program(ContentAddressable):
    signature: Signature
    code: str

    def get_content(self) -> str:
        return "# signature: " + self.signature.pretty_print() + "\n" + self.code

    def add_imports(self, imports) -> 'Program':
        return Program(self.signature, imports + "\n" + self.code)

    def hash(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()

    def passes(self, executor, tests: List[Test]) -> Tuple[bool, RawData]:
        """
        Timeout is a failure.
        Raw data schema:
        {
          "outcomes": ["pass", "pass", "fail", "timeout", ...]
        {
        """
        outcomes = []
        never_fails = True
        for index, test in enumerate(tests):
            match executor.run_test(self, test):
                case Pass():
                    outcomes.append('pass')
                case Timeout():
                    outcomes.append('timeout')
                    never_fails = False
                case _:
                    outcomes.append('fail')
                    never_fails = False
        return never_fails, outcomes
