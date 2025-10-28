import copy
from dataclasses import dataclass, field
import ast
import hashlib
from typing import Any, List, Tuple, Optional

from just_tri_it.cached_llm import Model
from just_tri_it.utils import (
    gen_and_extract_answer_with_retry,
    ContentAddressable,
    gen_and_extract_code_with_retry
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
    def from_function_code(code) -> 'Signature':
        return Signature.from_function_ast(ast.parse(code).body[0])

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
        code = gen_and_extract_code_with_retry(model, PROMPT_CODE, 3)
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
        return "# signature: " + self.target_signature.pretty_print() + "\n" \
            + "# test name: " + self.test_function_name + "\n" \
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
        return "inputs: " + repr(self.inputs) + "\n" \
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
    signature_note: Optional[str] = None

    def get_content(self) -> str:
        return "# signature: " + self.signature.pretty_print() + "\n" \
            + ("# signature note: " + self.signature_note + "\n" if self.signature_note is not None else "") \
            + self.description

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
using Python's snake_case naming convention. Output the name
enclosed in `<answer>` and `</answer>` tags.
        
Problem:
{req.description}
        """
        valid_name = gen_and_extract_answer_with_retry(model, PROMPT, 3)
        valid_name = NamedReturnSignature._make_name_unique(valid_name, req.signature.params)
        return NamedReturnSignature(req.signature.name,
                                    req.signature.params,
                                    req.signature.return_type,
                                    valid_name)

    @staticmethod
    def _make_name_unique(name: str, params: List[Parameter]) -> str:
        param_names = list(map(lambda p: p.name, params))

        if name not in param_names:
            return name

        suffixes = ["_result", "_value", "_out"]
        for suffix in suffixes:
            candidate = f"{name}{suffix}"
            if candidate not in param_names:
                return candidate

        counter = 1  # Final fallback: add a numeric suffix
        while True:
            candidate = f"{name}_{counter}"
            if candidate not in param_names:
                return candidate
            counter += 1


@dataclass
class Program(ContentAddressable):
    signature: Signature
    code: str
    nested: list[Signature] = field(default_factory=list)
    time_predicate: Optional['Program'] = None

    def get_content(self) -> str:
        return "# signature: " + self.signature.pretty_print() + "\n" + self.code

    def display_id(self) -> str:
        return self.hash_id()[:7]
    
    def add_imports(self, imports) -> 'Program':
        return Program(self.signature, imports + "\n" + self.code)

    def hash(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()

    def passes(self, executor, tests: List[Test]) -> Tuple[bool, List[str]]:
        """
        Timeout is a failure.
        Raw data schema:
        {
          "outcomes": ["Pass", "Pass", "Fail", "Timeout", ...]
        {
        """
        outcomes = []
        never_fails = True
        for test in tests:
            match executor.run_test(self, test):
                case Pass() as result:
                    outcomes.append(type(result).__name__)
                case _ as result:
                    outcomes.append(type(result).__name__)
                    never_fails = False
        return never_fails, outcomes

    @staticmethod
    def from_function_code(code) -> 'Program':
        """The code must be the function and nothing else. The
        function must have a complete type annotation"""
        return Program(Signature.from_function_code(code), code)

    def gen_time_predicate(self, model, req) -> 'Program':
        new_sig = copy.deepcopy(self.signature)
        new_sig.name = "certainly_exceeds_time_limit"
        new_sig.return_type = "bool"
        new_sig.params.append(Parameter("max_seconds", "int"))
        
        PROMPT = f"""Your task is to generate a Python predicate with the signature
        
{new_sig}

that conservatively estimates whether executing the following solution
to the given problem on the given input will exceed `max_seconds` seconds.

**Requirements:**

1.  Heuristic analysis only: the predicate must not execute the code,
and must be at most of a linear complexity.

2. Conservative analysis: return `True` if the analysis determines that
running this code on this input will certainly take more than `max_seconds`
seconds on consumer-grade hardware. Otherwise, return `False`.

3. Implement only the function, without using `if __name__ ==
"__main__":` or any code outside the function.

Please wrap your final code in a markdown code block.
```python
# Your implementation here
```

**Solution to be checked**:
```
{self.code}
```
        
**Problem description**:
```
{req}
```
    """
        predicate_code = gen_and_extract_code_with_retry(model, PROMPT)
        time_predicate = Program(new_sig, predicate_code)
        return Program(self.signature, self.code, time_predicate)
