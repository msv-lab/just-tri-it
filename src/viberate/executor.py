import pickle
from dataclasses import dataclass
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from viberate.program import Program


@dataclass
class Success:
    output: any


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


type Outcome = Success | Error | Panic | Timeout


EXECUTION_TIMEOUT_SECONDS = 2


def test_harness(p: Program, input_file: Path, output_file: Path):
    return f"""
import pickle
if __name__ == '__main__':
    with open('{str(input_file)}', 'rb') as f:
        input = pickle.load(f)
    report = dict()
    try:
        output = {p.signature.name}(*input)
        report['status'] = 'success'
        report['value'] = output
    except Exception as e:
        report['status'] = 'error'
        report['error_type'] = type(e)
        report['error_message'] = str(e)
    with open('{str(output_file)}', 'wb') as f:
        pickle.dump(report,f)
    """


class Executor:

    def __init__(self, test_venv: Path):
        self.test_venv = test_venv

    def run(self, p: Program, inputs: list[any]) -> Outcome:
        with TemporaryDirectory() as tmp:
            exec_dir = Path(tmp)
            input_file = exec_dir / 'input.pkl'
            with input_file.open('wb') as f:
                pickle.dump(inputs, f)
            output_file = exec_dir / 'output.pkl'
            source_code = p.code + "\n" + test_harness(p, input_file, output_file)
            (exec_dir / 'code.py').write_text(source_code)
            try:
                interpreter = str((self.test_venv / 'bin' / 'python').resolve())
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
                print("Timeout")
                return Timeout()
