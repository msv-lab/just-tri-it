from dataclasses import dataclass
from typing import List, Any
from pathlib import Path
import ast
import json

from viberate.program import Test, Signature, ExpectedOutput, Assertion
from viberate.requirements import Requirements


@dataclass
class Task:
    id: str
    requirements: Requirements
    tests: List[Test]
    metadata: dict[str, Any]


type Dataset = List[Task]


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
                  "inputs": [ ... ],
                  "oracle": {
                       "type": "...",
                       "value": "..."
                  }
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
def {sig_str}:
   pass
        """
        tree = ast.parse(code)
        fn_node = tree.body[0]
        signature = Signature.from_function_ast(fn_node)
        description = task["requirements"]["description"]
        requirements = Requirements(signature, description)

        tests = []
        for t in task["tests"]:
            inputs = t["inputs"]
            oracle_d = t["oracle"]
            if oracle_d["type"] == "expected_output":
                oracle = ExpectedOutput(oracle_d["value"])
            elif oracle_d["type"] == "assertion":
                oracle = Assertion(oracle_d["value"])
            else:
                raise ValueError(f"Unknown oracle type: {oracle_d['type']}")
            tests.append(Test(inputs, oracle))

        task_obj = Task(
            id=task["id"],
            requirements=requirements,
            tests=tests,
            metadata=task.get("metadata", {}),
        )
        tasks.append(task_obj)
    return tasks


def save_dataset(dataset: List[Task], file: Path) -> None:
    data = []

    for task in dataset:
        requirements = {
            "signature": task.requirements.signature.pretty_print(),
            "description": task.requirements.description,
        }

        tests = []
        for test in task.tests:
            oracle = None
            match test.oracle:
                case ExpectedOutput(value):
                    oracle = {
                        "type": "expected_output",
                        "value": value
                    }
                case Assertion(code):
                    oracle = {
                        "type": "assertion",
                        "value": code
                    }
            tests.append({
                "inputs": test.inputs,
                "oracle": oracle
            })

        task_dict = {
            "id": task.id,
            "requirements": requirements,
            "tests": tests,
            "metadata": task.metadata
        }
        data.append(task_dict)

    with file.open("w") as f:
        json.dump(data, f, indent=2)
