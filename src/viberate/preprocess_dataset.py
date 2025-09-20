import textwrap
import re
import ast
import json
import math
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import argparse
import copy

import jsonlines

from viberate.dataset import Dataset
from viberate.utils import panic
from viberate.cached_llm import Model, Persistent, Independent, AI302
from viberate.utils import extract_code, extract_answer
from viberate.program import Signature, Test, Parameter, Program, ExpectedOutput, Requirements
from viberate.executor import Executor, Success
from viberate.dataset import Task, Dataset, save_dataset, load_dataset, lcb_compress, lcb_decompress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=str,
        help="Set LLM cache root directory (default: ~/.viberate_cache/)."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache."
    )
    parser.add_argument(
        "--replicate",
        action="store_true",
        help="Use cache only."
    )
    parser.add_argument(
        "--export-cache",
        type=str,
        help="Explore all responsed generated during the run."
    )
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset file."
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Dataset file format."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file."
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress tests."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task ID to decompress."
    )
    return parser.parse_args()


class FailedToConvertAutomatically(Exception):
    pass


def main():
    args = parse_args()

    if args.decompress:
        decompress_task(Path(args.dataset), args.task, Path(args.output))
        exit(0)
    
    model_name = "gpt-4o"
    model = AI302(model_name, 1.0)
    
    if not args.no_cache:
        if args.cache_root:
            cache_root = Path(args.cache_root)
        else:
            cache_root = Path.home() / ".viberate_cache"
        if args.replicate:
            model = Persistent(model, cache_root, replication=True)
        else:
            model = Persistent(model, cache_root)

    if not args.no_cache and args.export_cache:
        export_root = Path(args.export_cache)
        export_root.mkdir(parents=True, exist_ok=True)
        model = Persistent(model, export_root)
            
    test_venv = Path(args.test_venv)
    executor = Executor(test_venv)

    if args.format == 'LiveCodeBench':
        dataset = lcb_convert(model, executor, Path(args.dataset), Path(args.output))
    else:
        panic("unsupported dataset format")


def decompress_task(dataset_file: Path, task_id: str, output_file: Path):
    d = load_dataset(dataset_file)
    d = [t for t in d if t.id == task_id]
    save_dataset(d, output_file, compress=False)


def lcb_fix_partial_code(src: str) -> str:
    assert src.startswith("class Solution:")
    return src + "\n        pass"


def remove_top_level_function(source: str, func_name: str) -> str:
    """
    Removes the top-level function definition with the name func_name from the source code.
    Returns the updated source as a string. If not found, returns the original source.
    """
    class FuncRemover(ast.NodeTransformer):
        def visit_Module(self, node):
            # Only keep nodes that are not the target function at the top level
            node.body = [
                n for n in node.body
                if not (isinstance(n, ast.FunctionDef) and n.name == func_name)
            ]
            return node

    try:
        tree = ast.parse(source)
        new_tree = FuncRemover().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_source = ast.unparse(new_tree)
        return new_source
    except Exception as e:
        raise ValueError(f"Could not process source: {e}")


def lcb_signature_from_starter_code(starter_code: str) -> Optional[Signature]:
    """
    Parse *source* looking for the very first ``class`` statement, then take the
    very first method defined inside that class and return an ``ast.FunctionDef``
    node that represents the same signature *without* the leading ``self``
    parameter.  The body of the new function is just ``pass``.

    If no suitable method is found the function returns ``None``.
    """
    tree = ast.parse(lcb_fix_partial_code(starter_code))

    class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if class_node is None:
        return None

    func_node = next((n for n in class_node.body if isinstance(n, ast.FunctionDef)), None)
    if func_node is None:
        return None

    new_args = copy.deepcopy(func_node.args)

    if new_args.args and isinstance(new_args.args[0], ast.arg) and new_args.args[0].arg == "self":
        new_args.args = new_args.args[1:]

    new_func = ast.FunctionDef(
        name=func_node.name,
        args=new_args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=func_node.returns,
        type_comment=func_node.type_comment,
    )

    ast.fix_missing_locations(new_func)

    return Signature.from_function_ast(new_func)


def lcb_parse_functional_inputs(s: str) -> list[Any]:
    return [json.loads(line) for line in s.split("\n")]


def lcb_parse_functional_output(s: str) -> Any:
    return json.loads(s)


def strip_multiline(s: str) -> str:
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])


def lcb_generate_input_formatter(model: Model, req: Requirements):
    formatter_sig = Signature("format_input", req.signature.params, "str")
    solution_sig = Signature(req.signature.name, [Parameter("s", "str")], "str")

    PROMPT=f"""The function `{solution_sig.pretty_print()}` takes
    a string simulating stdin and solves the problem described
    below. Please write a Python function as follows:

    - Implement a wrapper function `{formatter_sig.pretty_print()}`
      that receives the problem inputs (as its own parameters),
      formats them according to the specified input format for the
      problem, and converts them into a single string.
    
    - This wrapper should pass the formatted string directly to
      your solution function `{req.signature.name}` (do not do any
      additional processing), and return the result.

    - Include any necessary imports.

    - Place the complete code in a Markdown code block.

    - Only output the function code (no test cases, main guard, or
      stubs for your solution function).

    - Do not import or define solution function
      `{req.signature.name}`; assume it exists.

    Problem description:
    {req.description}
    """

    stdin_extractor = f"{solution_sig.pretty_print()}:\n   return s"
    s = extract_code(next(model.sample(PROMPT)))
    s = remove_top_level_function(s, solution_sig.name)

    return Program(formatter_sig, stdin_extractor + "\n" + s)


def lcb_input_sanity_check(executor: Executor, formatter: Program, stdin: str, parsed: Any):
    result = executor.run(formatter, parsed, add_lcb_imports=True)
    match result:
        case Success(value):
            if value is None:
                print(formatter)
                raise FailedToConvertAutomatically(f"failed to format stdin {stdin}")
            if not (strip_multiline(value) == strip_multiline(stdin)):
                raise FailedToConvertAutomatically(f"failed input sanity check:\n{value}\n !=\n{stdin}")
        case _:
            raise FailedToConvertAutomatically(f"failed to execute input sanity check: {result}")


def lcb_generate_output_formatter(model: Model, req: Requirements) -> Program:
    formatter_sig = Signature("format_output", [Parameter("value", req.signature.return_type)], "str")

    PROMPT=f"""I have developed a program to solve the problem
    described below, and this program returns
    {req.signature.return_type}. Now, I want to format its output as
    specified in the problem description. Please implement the
    following function: `{formatter_sig.pretty_print()}`.

    The function should:

    - Take as input a value representing the output generated by my solution.
    - Format the value according to the precise output format in the
      problem description.
    - Do not perform any correctness checks, error handling, or extra
      computation - your implementation should only format the string
      according to the expected format.
    - Ensure that the entire solution consists solely of this function
      implementation (do not use `if __name__ == "__main__":`).

    **Problem description:**
    {req.description}

    Please provide the function implementation inside a Markdown code block.
    """

    s = extract_code(next(model.sample(PROMPT)))

    return Program(formatter_sig, s)


def lcb_output_sanity_check(executor: Executor, formatter: Program, stdout: str, parsed: Any):
    result = executor.run(formatter, [parsed], add_lcb_imports=True)
    match result:
        case Success(value):
            if value is None:
                print(formatter)
                raise FailedToConvertAutomatically(f"failed to format stdin {stdout}")
            if formatter.signature.params[0].type == "float":
                actual = float(strip_multiline(value))
                expected = float(strip_multiline(stdout))
                if not math.isclose(actual, expected):
                    raise FailedToConvertAutomatically(f"failed sanity check on float output: \n{value} \n!~\n{stdout}")
            elif not (strip_multiline(value) == strip_multiline(stdout)):
                raise FailedToConvertAutomatically(f"failed output sanity check:\n{value} \n!=\n{stdout}")
        case _:
            raise FailedToConvertAutomatically(f"failed to execute output sanity check: {result}")


def lcb_generate_input_parser(model: Model, req: Requirements):
    parser_sig = Signature("parse_input", [Parameter("s", "str")], req.signature.return_type)
    
    PROMPT=f"""Given the problem description below, I have already
    implemented a solution function with the following signature:
    {req.signature.pretty_print()}

    Your tasks are:

    1. Implement a new function {parser_sig.pretty_print()} that:
    - Parses input according to the format described in the problem.
    - Directly passes the parsed arguments to my solution function by calling it.
    - Does not modify the arguments in any way.
    
    2. Include all necessary imports.
    
    3. Place your implementation in a Markdown code block as a single
    code snippet.

    4. Only implement the new parser function. Do not use
    if __name__ == "__main__".

    5. Outside of your {parser_sig.name} implementation, include a
    stub for my solution function so the code is complete.
    
    **Problem:**
    {req.description}
    """
    param_list = ",".join([p.name for p in req.signature.params])
    argument_extractor = f"{req.signature.pretty_print()}:\n   return [{param_list}]"
    
    s = extract_code(next(model.sample(PROMPT)))
    s = remove_top_level_function(s, req.signature.name)

    return Program(parser_sig, argument_extractor + "\n" + s)


def lcb_parse_stdin_inputs(executor: Executor, parser: Program, stdin: str) -> list[Any]:
    result = executor.run(parser, [stdin], add_lcb_imports=True)
    match result:
        case Success(value):
            if not isinstance(value, list):
                raise FailedToConvertAutomatically(f"extracted inputs are not in a list: {value}")
            return value
        case _:
            raise FailedToConvertAutomatically(f"failed to extract parameters: {result}")


def lcb_generate_output_parser(model: Model, req: Requirements):
    parser_sig = Signature("parse_output", [Parameter("output_str", "str")], req.signature.return_type)

    PROMPT_PARSER=f"""I have developed a program to solve the problem
    described below. Now, I want to parse its output to verify that
    the format matches the expected specification. Please implement
    the following function: `{parser_sig.pretty_print()}`.

    The function should:

    - Take as input a string representing the output generated by my solution.
    - Parse the string according to the precise output format in the
      problem description.
    - Do not perform any correctness checks, error handling, or extra
      computation - your implementation should only parse the string
      according to the expected format.
    - Ensure that the entire solution consists solely of this function
      implementation (do not use `if __name__ == "__main__":`).

    **Problem description:**
    {req.description}

    Please provide the function implementation inside a Markdown code block.
    """
    
    s = extract_code(next(model.sample(PROMPT_PARSER)))
    return Program(parser_sig, s)

    
def lcb_parse_stdout_output(executor: Executor, parser: Program, stdout: str) -> Any:
    result = executor.run(parser, [stdout], add_lcb_imports=True)
    match result:
        case Success(value):
            return value
        case _:
            raise FailedToConvertAutomatically(f"failed to parse output: {result}")


def lcb_convert(model: Model,
                executor: Executor,
                input_file: Path,
                output_file: Path) -> Dataset:
    tasks: Dataset = []
    
    skip = set()
    if output_file.exists():
        tasks = load_dataset(output_file)
        skip.update([t.id for t in tasks])
    
    with jsonlines.open(input_file) as reader:
        for entry in reader:
            unique_id = entry["platform"] + "_" + entry["question_id"]
            if unique_id in skip:
                continue
            print(f"converting {unique_id}")
            NUM_ATTEMPTS = 3
            ind_model = Independent(model)
            for attempt in range(NUM_ATTEMPTS):
                try:
                    if entry["starter_code"] != "":
                        signature = lcb_signature_from_starter_code(entry["starter_code"])
                        req = Requirements(signature, entry["question_content"])
                    else:
                        req = Requirements.from_description(ind_model, entry["question_content"])
                    tests = []
                    test_data = json.loads(entry["public_test_cases"])
                    test_data.extend(json.loads(lcb_decompress(entry["private_test_cases"])))
                    input_parser = None
                    output_parser = None
                    input_formatter = None
                    output_formatter = None
                    for t in test_data:
                        if t["testtype"] == "stdin":
                            if input_parser is None:
                                input_parser = lcb_generate_input_parser(ind_model, req)
                                output_parser = lcb_generate_output_parser(ind_model, req)
                                input_formatter = lcb_generate_input_formatter(ind_model, req)
                                output_formatter = lcb_generate_output_formatter(ind_model, req)
                            i = lcb_parse_stdin_inputs(executor, input_parser, t["input"])
                            lcb_input_sanity_check(executor, input_formatter, t["input"], i)
                            o = lcb_parse_stdout_output(executor, output_parser, t["output"])
                            lcb_output_sanity_check(executor, output_formatter, t["output"], o)
                        elif t["testtype"] == "functional":
                            i = lcb_parse_functional_inputs(t["input"])
                            o = lcb_parse_functional_output(t["output"])
                        else:
                            panic("unsupported test type")
                        tests.append(Test(i, ExpectedOutput(o)))
                    task = Task(
                        id=unique_id,
                        requirements=req,
                        tests=tests,
                        metadata={
                            "difficulty": entry["difficulty"],
                            "exact": "unknown",
                            "title": entry["question_title"],
                            "date": entry["contest_date"]
                        }
                    )
                    tasks.append(task)
                    break
                except FailedToConvertAutomatically as e:
                    if attempt < NUM_ATTEMPTS - 1:
                        print("retrying...")
                    else:
                        print(e)
            save_dataset(tasks, output_file, compress=True)


if __name__ == "__main__":
    main()
