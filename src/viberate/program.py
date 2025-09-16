from dataclasses import dataclass
import ast
import re
import hashlib
from typing import Any, List

from viberate.utils import extract_code


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

    def hash(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()


@dataclass
class ExpectedOutput:
    value: Any


@dataclass
class Assertion:
    test_function_code: str
    target_signature: Signature
    test_function_name: str
    
    @staticmethod
    def from_code(code: str, target_signature: Signature) -> 'Assertion':
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                return Assertion(
                    test_function_code=code,
                    target_signature=target_signature,
                    test_function_name=node.name
                )
        raise ValueError("No test function found in code")
    
    @staticmethod
    def generate_from_problem(model, problem_description: str, target_signature: Signature, num_tests: int = 1) -> List['Assertion']:        
        PROMPT_ASSERTIONS = f"""
        For the problem below, write {num_tests} unit test function(s) to verify the correctness of the solution.

        Problem:
        {problem_description}

        Function signature: {target_signature.pretty_print()}

        Each test should be a function whose name starts with test_, calls the target function, 
        and contains assertions to verify correctness. The tests should cover:
        - Normal/typical cases
        - Edge cases and boundary conditions  
        - Error conditions (if applicable)

        For problems where exact output cannot be predetermined, use assertions that check 
        properties, constraints, or relationships rather than exact values.

        Example formats:
        ```python
        def test_basic_functionality():
            result = {target_signature.name}(typical_input)
            assert result == expected_value
        ```

        ```python  
        def test_boundary_condition():
            result = {target_signature.name}(edge_case_input)
            assert some_property_holds(result)
        ```

        Return each test function in a separate Python code block.
        """
        
        try:
            response = next(model.sample(PROMPT_ASSERTIONS))

            code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
                
            assertions = []
            for code in code_blocks:
                if code and code.strip():
                    try:
                        assertion = Assertion.from_code(code.strip(), target_signature)
                        assertions.append(assertion)
                    except ValueError as e:
                        print(f"Failed to create assertion from code: {e}")
                        continue
                        
            return assertions[:num_tests]
            
        except Exception as e:
            print(f"Error in generate_from_problem: {e}")
            return []
    
    def execute(self, program: Program) -> bool:
        try:
            exec_globals = {}
            
            exec(program.code, exec_globals)
            
            exec(self.test_function_code, exec_globals)
            
            test_func = exec_globals[self.test_function_name]
            test_func()
            
            return True
        except Exception as e:
            return False
    
    def extract_inputs(self, program: Program) -> List[List[Any]]:
        inputs_collected = []
        
        def create_dummy_function(signature: Signature):
            def dummy(*args, **kwargs):
                param_values = []
                for i, param in enumerate(signature.params):
                    if i < len(args):
                        param_values.append(args[i])
                    elif param.name in kwargs:
                        param_values.append(kwargs[param.name])
                inputs_collected.append(param_values)
                return None
            return dummy
        
        try:
            exec_globals = {}
            
            dummy_func = create_dummy_function(self.target_signature)
            exec_globals[self.target_signature.name] = dummy_func
            
            exec(self.test_function_code, exec_globals)
            test_func = exec_globals[self.test_function_name]
            
            try:
                test_func()
            except:
                pass
                
        except Exception as e:
            pass
            
        return inputs_collected
    

type Oracle = ExpectedOutput | Assertion


@dataclass
class Test:
    inputs: List[Any]
    oracle: Oracle

    @staticmethod
    def from_assertion(assertion: Assertion) -> 'Test':
        return Test(inputs=[], oracle=assertion)
    
    @staticmethod
    def from_expected_output(inputs: List[Any], expected: Any) -> 'Test':
        return Test(inputs=inputs, oracle=ExpectedOutput(expected))
    
    def special_reformat(self):
        if isinstance(self.oracle, ExpectedOutput):
            return {
                "expexted_input": self.inputs,
                "expexted_output": self.oracle.value
            }
        if isinstance(self.oracle, Assertion):
            return {
                "function_code": self.oracle.test_function_code,
                "function_name": self.oracle.test_function_name
            }