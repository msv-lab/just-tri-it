import pickle
import math
from dataclasses import dataclass
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

from viberate.program import Program, Test, ExpectedOutput, Assertion
from viberate.utils import panic


@dataclass
class Success:
    output: Any


@dataclass
class Error:
    '''Error raised by the executed function'''
    type: str
    message: str


@dataclass
class Panic:
    '''Failed to execute the function (e.g. malformed source code)'''
    message: str


@dataclass
class Timeout:
    pass


type ExecutionOutcome = Success | Error | Panic | Timeout


@dataclass
class Pass:
    pass
    

@dataclass
class Fail:
    pass


type TestOutcome = Pass | Fail | Error | Panic | Timeout


EXECUTION_TIMEOUT_SECONDS = 2


def test_harness(p: Program, input_file: Path, output_file: Path):
    return f"""
import pickle
if __name__ == '__main__':
    with open('{str(input_file)}', 'rb') as __input_file:
        __input_list = pickle.load(__input_file)
    report = dict()
    try:
        output = {p.signature.name}(*__input_list)
        report['status'] = 'success'
        report['value'] = output
    except Exception as e:
        report['status'] = 'error'
        report['error_type'] = type(e).__name__
        report['error_message'] = str(e)
    with open('{str(output_file)}', 'wb') as f:
        pickle.dump(report,f)
    """

def assertion_test_harness(p: Program, assertion: Assertion, output_file: Path):
    return f"""
import pickle

{p.code}

{assertion.test_function_code}

if __name__ == '__main__':
    report = dict()
    try:
        {assertion.test_function_name}()
        report['status'] = 'success'
        report['test_passed'] = True
    except AssertionError as e:
        report['status'] = 'success'
        report['test_passed'] = False
        report['assertion_message'] = str(e)
    except Exception as e:
        report['status'] = 'error'
        report['error_type'] = type(e).__name__
        report['error_message'] = str(e)
    
    with open('{str(output_file)}', 'wb') as f:
        pickle.dump(report, f)
    """

class Executor:

    def __init__(self, test_venv: Path):
        self.test_venv = test_venv

    def run(self, p: Program, inputs: list[Any]) -> ExecutionOutcome:
        assert isinstance(inputs, list)
        with TemporaryDirectory() as tmp:
            exec_dir = Path(tmp)
            input_file = exec_dir / 'input.pkl'
            with input_file.open('wb') as f:
                pickle.dump(inputs, f)
            output_file = exec_dir / 'output.pkl'
            source_code = p.code + "\n" + test_harness(p, input_file, output_file)
            (exec_dir / 'code.py').write_text(source_code)
            try:
                interpreter = str(self.test_venv.resolve() / 'bin' / 'python')
                result = subprocess.run(
                    [interpreter, 'code.py'],
                    cwd=exec_dir,
                    capture_output=True,   # Captures stdout and stderr
                    text=True,             # Returns output as string, not bytes
                    timeout=EXECUTION_TIMEOUT_SECONDS,
                    check=False            # Don't raise exception for nonzero status
                )
                if result.returncode != 0:
                    return Panic(result.stderr)
                if not output_file.exists():
                    return Panic("no output")
                with output_file.open('rb') as f:
                    report = pickle.load(f)
                    if report['status'] == 'success':
                        return Success(report['value'])
                    else:
                        assert report['status'] == 'error'
                        return Error(report['error_type'], report['error_message'])
                
            except subprocess.TimeoutExpired:
                return Timeout()
    
    def run_assertion_test(self, p: Program, assertion: Assertion) -> TestOutcome:
        with TemporaryDirectory() as tmp:
            exec_dir = Path(tmp)
            output_file = exec_dir / 'output.pkl'
            source_code = assertion_test_harness(p, assertion, output_file)
            (exec_dir / 'code.py').write_text(source_code)
            
            try:
                interpreter = str(self.test_venv.resolve() / 'bin' / 'python')
                result = subprocess.run(
                    [interpreter, 'code.py'],
                    cwd=exec_dir,
                    capture_output=True,
                    text=True,
                    timeout=EXECUTION_TIMEOUT_SECONDS,
                    check=False
                )
                
                if result.returncode != 0:
                    return Panic(result.stderr)
                    
                if not output_file.exists():
                    return Panic("no output")
                    
                with output_file.open('rb') as f:
                    report = pickle.load(f)
                    if report['status'] == 'success':
                        if report['test_passed']:
                            return Pass()
                        else:
                            return Fail()
                    else:
                        return Error(report['error_type'], report['error_message'])
                        
            except subprocess.TimeoutExpired:
                return Timeout()

    def run_test(self, p: Program, t: Test) -> TestOutcome:
        match t.oracle:
            case ExpectedOutput(expected):
                execution_outcome = self.run(p, t.inputs)
                match execution_outcome:
                    case Success(actual):
                        if isinstance(actual, float) and isinstance(expected, float) and math.isclose(actual, expected):
                            return Pass()
                        elif actual == expected:
                            return Pass()
                        else:
                            print(f"Expected: {expected}")
                            print(f"Actual: {actual}")
                            return Fail()
                    case _:
                        return execution_outcome
            
            case Assertion() as assertion:
                return self.run_assertion_test(p, assertion)