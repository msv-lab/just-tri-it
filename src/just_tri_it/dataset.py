from dataclasses import dataclass
from typing import List, Any
from pathlib import Path
import ast
import json
import zlib
import pickle
import base64

from just_tri_it.program import Test, Signature, InputOutput, TestFunction, Requirements
from just_tri_it.utils import panic


@dataclass
class Task:
    id: str
    requirements: Requirements
    tests: List[Test]
    metadata: dict[str, Any]


type Dataset = List[Task]


# compression method from LiveCodeBench
def lcb_decompress(s: str) -> Any:
    return pickle.loads(
        zlib.decompress(
            base64.b64decode(s.encode("utf-8"))
        )
    )


def lcb_compress(obj: Any) -> str:
    return base64.b64encode(
        zlib.compress(
            pickle.dumps(obj)
        )
    ).decode("utf-8")


def load_dataset(file: Path) -> Dataset:
    '''Schema:
    [
       {
          "id": "...",
          "requirements": {
              "signature": "...",
              "description": "...",
          },
          "tests": [
              {
                  "type": "InputOutput",
                  "inputs": [ ... ],   # strings containing Python expressions
                  "output": ...        # string with a Python expression
              },
              {
                  "type": "TestFunction",
                  "name": ...,
                  "code": ...
              },
              ...
          ],
          "metadata": { ... }
       },
       ...
    ]
    '''
    with file.open() as f:
        data = json.load(f)

    tasks: Dataset = []
    for task in data:
        sig_str = task["requirements"]["signature"]
        code = f"""
{sig_str}:
   pass
        """
        tree = ast.parse(code)
        fn_node = tree.body[0]
        signature = Signature.from_function_ast(fn_node)
        description = task["requirements"]["description"]
        requirements = Requirements(signature, description)

        tests = []
        if isinstance(task["tests"], str):
            test_data = lcb_decompress(task["tests"])
        else:
            test_data = task["tests"]
        for t in test_data:
            if t["type"] == "InputOutput":
                inputs = list(map(eval, t["inputs"]))
                output = eval(t["output"])
                tests.append(InputOutput(inputs, output))
            elif t["type"] == "TestFunction":
                tests.append(TestFunction.from_code(t["code"], signature))
            else:
                panic("Test assertions are not supported!")

        task_obj = Task(
            id=task["id"],
            requirements=requirements,
            tests=tests,
            metadata=task.get("metadata", {}),
        )
        tasks.append(task_obj)
    return tasks


def save_dataset(dataset: List[Task], file: Path, compress=False):
    data = []

    for task in dataset:
        requirements = {
            "signature": task.requirements.signature.pretty_print(),
            "description": task.requirements.description,
        }

        tests = []
        for test in task.tests:
            match test:
                case InputOutput(inputs, output):
                    tests.append({
                        "type": "InputOutput",
                        "inputs": list(map(repr, inputs)),
                        "output": repr(output)
                    })
                case TestFunction():
                    panic("Test assertions are not supported!")
        task_dict = {
            "id": task.id,
            "requirements": requirements,
            "tests": tests if not compress else lcb_compress(tests),
            "metadata": task.metadata
        }
        data.append(task_dict)

    with file.open("w") as f:
        json.dump(data, f, indent=2)
