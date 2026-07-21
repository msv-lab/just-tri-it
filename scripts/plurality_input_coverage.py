#!/usr/bin/env python3
"""Compute plurality-input coverage and differential-test metrics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
import ast

from just_tri_it.cached_llm import Repeatable, CloseAI, XMCP
from just_tri_it.dataset import Task, load_dataset
from just_tri_it.executor import (
    LIVECODEBENCH_IMPORTS,
    PersistentWorkerExecutor,
    SubprocessExecutor,
)
from just_tri_it.input_generator import generate_inputs, oracle_inputs_from_tests
from just_tri_it.program import (
    EXECUTION_TIMEOUT_SECONDS,
    InputOutput,
    Program,
    Requirements,
    TestFunction,
)
from just_tri_it.utils import add_cache_options, init_random, setup_cache
import just_tri_it.config
import just_tri_it.utils


DEFAULT_SELECTOR_ID = "Plurality"
DEFAULT_INPUT_SOURCE = "generated"
DEFAULT_JOBS = max(1, min(8, os.cpu_count() or 1))
METRIC_COVERAGE = "coverage"
METRIC_COUNT = "count"
METRIC_BOTH = "both"
DIFFERENTIAL_COUNT_SCHEMA_VERSION = 7
ORACLE_EQUIVALENCE_MODE = "all_input_output_tests"
JUDGE_ORACLE_EQUIVALENCE_MODE = "all_test_function_tests"
ORACLE_COVERAGE_INPUT_SOURCE = "oracle_tests"
STATUS_VALUES = ("success", "error", "timeout", "panic")
RAW_OUTPUT_SEMANTICS = "raw_output"
JUDGE_LABEL_SEMANTICS = "judge_label"
JUDGE_PASS = "pass"
JUDGE_FAIL = "fail"

PROGRAM_FIELDS = [
    "task_id",
    "selector_id",
    "program_hash",
    "input_source",
    "num_inputs",
    "covered_lines",
    "total_lines",
    "line_coverage",
    "success_count",
    "error_count",
    "timeout_count",
    "panic_count",
]

TASK_FIELDS = [
    "task_id",
    "selector_id",
    "status",
    "skip_reason",
    "detail",
    "input_source",
    "num_inputs",
    "program_count",
    "mean_line_coverage",
    "weighted_line_coverage",
    "covered_lines",
    "total_lines",
]

DIFFERENTIAL_PROGRAM_FIELDS = [
    "task_id",
    "selector_id",
    "program_hash",
    "oracle_equivalence_class_id",
    "oracle_equivalence_class_size",
    "input_source",
    "num_inputs",
    "valid_input_count",
    "differential_test_count",
    "differential_test_rate",
    "success_count",
    "error_count",
    "timeout_count",
    "panic_count",
]

INPUT_QUALITY_FIELDS = [
    "task_id",
    "selector_id",
    "status",
    "skip_reason",
    "detail",
    "input_source",
    "num_inputs",
    "valid_input_count",
    "program_count",
    "original_program_count",
    "oracle_equivalence_class_count",
    "oracle_equivalence_input_count",
    "valid_program_input_count",
    "differential_test_count",
    "weighted_differential_test_rate",
    "mean_program_differential_test_rate",
    "valid_execution_rate",
    "distinguishing_input_count",
    "distinguishing_input_rate",
    "mean_output_class_count",
    "success_count",
    "error_count",
    "timeout_count",
    "panic_count",
]


class TraceRunResult:
    def __init__(
        self,
        status: str,
        covered_lines: set[int],
        error_type: str | None = None,
        error_message: str | None = None,
        output_key: str | None = None,
    ):
        self.status = status
        self.covered_lines = covered_lines
        self.error_type = error_type
        self.error_message = error_message
        self.output_key = output_key


class TaskExecutionError(ValueError):
    """A task's tests cannot be safely evaluated by the coverage harness."""


class TaskExecutionSpec:
    def __init__(
        self,
        semantics: str,
        oracle_cases: list[Any],
        oracle_case_description: str,
        oracle_equivalence_mode: str,
        judge_context_code: str | None = None,
    ):
        self.semantics = semantics
        self.oracle_cases = oracle_cases
        self.oracle_case_description = oracle_case_description
        self.oracle_equivalence_mode = oracle_equivalence_mode
        self.judge_context_code = judge_context_code


class MetricRows:
    def __init__(
        self,
        coverage_program_rows: list[dict[str, Any]] | None = None,
        coverage_task_rows: list[dict[str, Any]] | None = None,
        differential_program_rows: list[dict[str, Any]] | None = None,
        input_quality_task_rows: list[dict[str, Any]] | None = None,
        oracle_coverage_program_rows: list[dict[str, Any]] | None = None,
        oracle_coverage_task_rows: list[dict[str, Any]] | None = None,
        oracle_differential_program_rows: list[dict[str, Any]] | None = None,
        oracle_input_quality_task_rows: list[dict[str, Any]] | None = None,
    ):
        self.coverage_program_rows = list(coverage_program_rows or [])
        self.coverage_task_rows = list(coverage_task_rows or [])
        self.differential_program_rows = list(differential_program_rows or [])
        self.input_quality_task_rows = list(input_quality_task_rows or [])
        self.oracle_coverage_program_rows = list(oracle_coverage_program_rows or [])
        self.oracle_coverage_task_rows = list(oracle_coverage_task_rows or [])
        self.oracle_differential_program_rows = list(oracle_differential_program_rows or [])
        self.oracle_input_quality_task_rows = list(oracle_input_quality_task_rows or [])


def wants_coverage(metric: str) -> bool:
    return metric in {METRIC_COVERAGE, METRIC_BOTH}


def wants_count(metric: str) -> bool:
    return metric in {METRIC_COUNT, METRIC_BOTH}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce plurality inputs and compute coverage or differential "
            "test-count metrics over sample programs stored in an experiment result."
        )
    )
    add_cache_options(parser)
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input file or task directory containing the dataset.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Experiment result directory containing data.jsonl and content/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where metric CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--task-list",
        type=str,
        help="File with task IDs to run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Single task ID to run.",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Filter loaded tasks to a single task ID.",
    )
    parser.add_argument(
        "--only-list",
        type=str,
        help="Filter loaded tasks to task IDs listed in a file.",
    )
    parser.add_argument(
        "--num-left-samples",
        type=int,
        help="Number of left samples, matching experiment.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt-4o", "deepseek-v3", "gemini-2.5-flash"],
        help="LLM model key used by the original experiment.",
    )
    parser.add_argument(
        "--extra-cache-root",
        type=str,
        help="Accepted for compatibility; ignored because inputs now use --model.",
    )
    parser.add_argument(
        "--selector-id",
        default=DEFAULT_SELECTOR_ID,
        help=f"Selector to read from experiment data. Default: {DEFAULT_SELECTOR_ID}.",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        help="Only compute metrics for the first N programs recorded by the selector.",
    )
    parser.add_argument(
        "--metric",
        choices=[METRIC_COVERAGE, METRIC_COUNT, METRIC_BOTH],
        default=METRIC_COVERAGE,
        help="Metric to compute. Default: coverage.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"Number of programs to cover in parallel per task. Default: {DEFAULT_JOBS}.",
    )
    parser.add_argument(
        "--plurality-input-source",
        choices=["generated", "oracle"],
        default=DEFAULT_INPUT_SOURCE,
        help="Input source to use when reproducing plurality inputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero after writing outputs if any task is skipped.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs; final summary is still printed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing metric CSV files in --output-dir.",
    )
    args = parser.parse_args(argv)
    if args.max_programs is not None and args.max_programs <= 0:
        parser.error("--max-programs must be positive")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    return args


def has_database_layout(path: Path) -> bool:
    return (path / "data.jsonl").is_file() and (path / "content").is_dir()


def resolve_data_dir(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    if has_database_layout(path):
        return path
    raise FileNotFoundError(
        f"Expected data.jsonl and content/ under experiment data directory: {path}"
    )


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {e}") from e
            if isinstance(obj, dict):
                objects.append(obj)
    return objects


def read_content_map(content_dir: Path) -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in content_dir.glob("*.txt")
    }


def strip_content_header(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if lines and lines[0].startswith("# signature:"):
        return "".join(lines[1:])
    return content


def program_from_content(requirements: Requirements, content: str) -> Program:
    return Program(requirements.signature, strip_content_header(content))


def executable_lines(code: str) -> set[int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    excluded = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, excluded):
            lineno = getattr(node, "lineno", None)
            if lineno is not None:
                lines.add(lineno)
    return lines


def get_plurality_raw_data(selector_data: dict[str, Any]) -> dict[str, Any] | None:
    raw_data = selector_data.get("raw_data")
    if not isinstance(raw_data, dict):
        return None

    agreement_raw_data = raw_data.get("agreement_raw_data")
    if (
        isinstance(agreement_raw_data, dict)
        and str(agreement_raw_data.get("method", "")).startswith("plurality_")
    ):
        return agreement_raw_data

    if str(raw_data.get("method", "")).startswith("plurality_"):
        return raw_data

    return None


def extract_program_hashes(obj: dict[str, Any], selector_id: str) -> tuple[list[str], str | None]:
    selectors = obj.get("selectors")
    if not isinstance(selectors, list):
        return [], "missing_selector"

    selector_data = next(
        (
            selector
            for selector in selectors
            if isinstance(selector, dict) and selector.get("id") == selector_id
        ),
        None,
    )
    if selector_data is None:
        return [], "missing_selector"

    plurality_raw_data = get_plurality_raw_data(selector_data)
    if plurality_raw_data is None:
        return [], "missing_plurality_raw_data"

    programs = plurality_raw_data.get("programs")
    if not isinstance(programs, list):
        return [], "missing_programs"

    program_hashes = [program_hash for program_hash in programs if isinstance(program_hash, str)]
    if not program_hashes:
        return [], "missing_programs"

    return program_hashes, None


def judge_context_from_test_functions(task: Task) -> str:
    test_functions = [test for test in task.tests if isinstance(test, TestFunction)]
    if not test_functions:
        raise TaskExecutionError("task.tests contains no TestFunction tests")

    expected_parameter_count = len(task.requirements.signature.params) + 1
    judge_fingerprint = None
    for test in test_functions:
        try:
            tree = ast.parse(test.test_function_code)
        except SyntaxError as error:
            raise TaskExecutionError(
                f"could not parse {test.test_function_name}: {error.msg}"
            ) from error
        judge_nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__judge"
        ]
        if len(judge_nodes) != 1:
            raise TaskExecutionError(
                f"{test.test_function_name} must define exactly one __judge function"
            )
        judge_node = judge_nodes[0]
        positional_parameter_count = len(judge_node.args.posonlyargs) + len(judge_node.args.args)
        if (
            positional_parameter_count != expected_parameter_count
            or judge_node.args.vararg is not None
            or judge_node.args.kwonlyargs
            or judge_node.args.kwarg is not None
        ):
            raise TaskExecutionError(
                "__judge must accept exactly the target parameters followed by the result"
            )
        fingerprint = ast.dump(judge_node, include_attributes=False)
        if judge_fingerprint is None:
            judge_fingerprint = fingerprint
        elif fingerprint != judge_fingerprint:
            raise TaskExecutionError("TestFunction tests use inconsistent __judge definitions")

    # The full source preserves any helper declarations used by the judge. Its test function is
    # only defined here; it is never invoked for generated inputs.
    return test_functions[0].test_function_code


def task_execution_spec(task: Task) -> TaskExecutionSpec:
    input_output_tests = [test for test in task.tests if isinstance(test, InputOutput)]
    test_function_tests = [test for test in task.tests if isinstance(test, TestFunction)]
    if input_output_tests and test_function_tests:
        raise TaskExecutionError("mixed InputOutput and TestFunction tests are unsupported")
    if test_function_tests:
        return TaskExecutionSpec(
            semantics=JUDGE_LABEL_SEMANTICS,
            oracle_cases=test_function_tests,
            oracle_case_description="TestFunction tests",
            oracle_equivalence_mode=JUDGE_ORACLE_EQUIVALENCE_MODE,
            judge_context_code=judge_context_from_test_functions(task),
        )
    return TaskExecutionSpec(
        semantics=RAW_OUTPUT_SEMANTICS,
        oracle_cases=[test.inputs for test in input_output_tests],
        oracle_case_description="InputOutput tests",
        oracle_equivalence_mode=ORACLE_EQUIVALENCE_MODE,
    )


def dataset_execution_metadata(tasks: list[Task]) -> tuple[str, str]:
    semantics = set()
    modes = set()
    for task in tasks:
        try:
            spec = task_execution_spec(task)
        except TaskExecutionError:
            semantics.add("unsupported")
            modes.add("unsupported")
        else:
            semantics.add(spec.semantics)
            modes.add(spec.oracle_equivalence_mode)
    if len(semantics) == 1 and len(modes) == 1:
        return semantics.pop(), modes.pop()
    return "mixed", "mixed"


def _harness_code(
    imports_file: Path,
    program_file: Path,
    input_file: Path,
    output_file: Path,
    function_name: str,
    capture_output_key: bool,
    judge_file: Path | None = None,
    test_function_name: str | None = None,
) -> str:
    judge_file_value = str(judge_file) if judge_file is not None else None
    return f"""
import json
import pickle
import trace
from pathlib import Path

imports_file = Path({str(imports_file)!r})
program_file = Path({str(program_file)!r})
input_file = Path({str(input_file)!r})
output_file = Path({str(output_file)!r})
function_name = {function_name!r}
capture_output_key = {capture_output_key!r}
judge_file_value = {judge_file_value!r}
judge_file = Path(judge_file_value) if judge_file_value is not None else None
test_function_name = {test_function_name!r}

namespace = {{"__file__": str(program_file), "__name__": "__jti_program__"}}
tracer = trace.Trace(count=True, trace=False)
status = None
error_type = None
error_message = None
output_key = None

def _json_sort_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def _normalize_output(value):
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, list):
        return ["list", [_normalize_output(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_normalize_output(item) for item in value]]
    if isinstance(value, dict):
        items = []
        for key, item_value in value.items():
            normalized_key = _normalize_output(key)
            normalized_value = _normalize_output(item_value)
            items.append((_json_sort_key(normalized_key), normalized_key, normalized_value))
        return [
            "dict",
            [
                [normalized_key, normalized_value]
                for _, normalized_key, normalized_value in sorted(items, key=lambda item: item[0])
            ],
        ]
    if isinstance(value, set):
        values = [_normalize_output(item) for item in value]
        return ["set", sorted(values, key=_json_sort_key)]
    return [
        "repr",
        f"{{type(value).__module__}}.{{type(value).__qualname__}}",
        repr(value),
    ]

def _output_key(value):
    try:
        normalized = _normalize_output(value)
    except Exception:
        normalized = [
            "repr",
            f"{{type(value).__module__}}.{{type(value).__qualname__}}",
            repr(value),
        ]
    return _json_sort_key(normalized)

try:
    imports_code = imports_file.read_text(encoding="utf-8")
    program_code = program_file.read_text(encoding="utf-8")
    imports_compiled = compile(imports_code, str(imports_file), "exec")
    program_compiled = compile(program_code, str(program_file), "exec")
    tracer.runctx("exec(imports_compiled, namespace)", globals(), locals())
    tracer.runctx("exec(program_compiled, namespace)", globals(), locals())
    if judge_file is not None:
        judge_code = judge_file.read_text(encoding="utf-8")
        judge_compiled = compile(judge_code, str(judge_file), "exec")
        tracer.runctx("exec(judge_compiled, namespace)", globals(), locals())

    if test_function_name is not None:
        test_func = namespace.get(test_function_name)
        if not callable(test_func):
            status = "panic"
            error_type = "panic"
            error_message = "no test function found"
        else:
            try:
                tracer.runfunc(test_func)
                status = "success"
                if capture_output_key:
                    output_key = {JUDGE_PASS!r}
            except AssertionError:
                status = "success"
                if capture_output_key:
                    output_key = {JUDGE_FAIL!r}
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                error_message = str(e)
    else:
        func = namespace.get(function_name)
        if not callable(func):
            status = "panic"
            error_type = "panic"
            error_message = "no function found"
        else:
            with input_file.open("rb") as f:
                inputs = pickle.load(f)
            try:
                return_value = tracer.runfunc(func, *inputs)
                if judge_file is None:
                    status = "success"
                    if capture_output_key:
                        output_key = _output_key(return_value)
                else:
                    judge = namespace.get("__judge")
                    if not callable(judge):
                        status = "panic"
                        error_type = "panic"
                        error_message = "no __judge function found"
                    else:
                        judge_result = tracer.runfunc(judge, *inputs, return_value)
                        status = "success"
                        if capture_output_key:
                            output_key = {JUDGE_PASS!r} if judge_result else {JUDGE_FAIL!r}
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                error_message = str(e)
except Exception as e:
    status = "panic"
    error_type = type(e).__name__
    error_message = str(e)

counts = tracer.results().counts
covered_lines = sorted(
    lineno
    for (filename, lineno), count in counts.items()
    if filename == str(program_file) and count
)

with output_file.open("wb") as f:
    pickle.dump(
        {{
            "status": status,
            "covered_lines": covered_lines,
            "error_type": error_type,
            "error_message": error_message,
            "output_key": output_key,
        }},
        f,
    )
"""


def trace_program_input(
    program: Program,
    inputs: list[Any],
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    capture_output_key: bool = False,
    judge_context_code: str | None = None,
) -> TraceRunResult:
    interpreter = str(interpreter or sys.executable)
    with TemporaryDirectory() as tmp:
        exec_dir = Path(tmp)
        imports_file = exec_dir / "jti_imports.py"
        program_file = exec_dir / "program_under_test.py"
        input_file = exec_dir / "input.pkl"
        output_file = exec_dir / "output.pkl"
        harness_file = exec_dir / "trace_harness.py"
        judge_file = exec_dir / "judge_context.py" if judge_context_code is not None else None

        imports_file.write_text(LIVECODEBENCH_IMPORTS, encoding="utf-8")
        program_file.write_text(program.code, encoding="utf-8")
        if judge_file is not None:
            judge_file.write_text(judge_context_code, encoding="utf-8")
        with input_file.open("wb") as f:
            pickle.dump(inputs, f)
        harness_file.write_text(
            _harness_code(
                imports_file,
                program_file,
                input_file,
                output_file,
                program.signature.name,
                capture_output_key,
                judge_file=judge_file,
            ),
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                [interpreter, str(harness_file)],
                cwd=exec_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return TraceRunResult("timeout", set())

        if result.returncode != 0 or not output_file.exists():
            message = result.stderr if result.returncode != 0 else "no output"
            return TraceRunResult("panic", set(), "panic", message)

        with output_file.open("rb") as f:
            report = pickle.load(f)

    return TraceRunResult(
        status=report.get("status", "panic"),
        covered_lines=set(report.get("covered_lines", [])),
        error_type=report.get("error_type"),
        error_message=report.get("error_message"),
        output_key=report.get("output_key"),
    )


def trace_program_test(
    program: Program,
    test: TestFunction,
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    capture_output_key: bool = False,
) -> TraceRunResult:
    interpreter = str(interpreter or sys.executable)
    with TemporaryDirectory() as tmp:
        exec_dir = Path(tmp)
        imports_file = exec_dir / "jti_imports.py"
        program_file = exec_dir / "program_under_test.py"
        input_file = exec_dir / "unused_input.pkl"
        output_file = exec_dir / "output.pkl"
        harness_file = exec_dir / "trace_harness.py"
        test_file = exec_dir / "oracle_test.py"

        imports_file.write_text(LIVECODEBENCH_IMPORTS, encoding="utf-8")
        program_file.write_text(program.code, encoding="utf-8")
        test_file.write_text(test.test_function_code, encoding="utf-8")
        with input_file.open("wb") as f:
            pickle.dump([], f)
        harness_file.write_text(
            _harness_code(
                imports_file,
                program_file,
                input_file,
                output_file,
                program.signature.name,
                capture_output_key,
                judge_file=test_file,
                test_function_name=test.test_function_name,
            ),
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                [interpreter, str(harness_file)],
                cwd=exec_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return TraceRunResult("timeout", set())

        if result.returncode != 0 or not output_file.exists():
            message = result.stderr if result.returncode != 0 else "no output"
            return TraceRunResult("panic", set(), "panic", message)

        with output_file.open("rb") as f:
            report = pickle.load(f)

    return TraceRunResult(
        status=report.get("status", "panic"),
        covered_lines=set(report.get("covered_lines", [])),
        error_type=report.get("error_type"),
        error_message=report.get("error_message"),
        output_key=report.get("output_key"),
    )


def normalized_status(status: str) -> str:
    if status in STATUS_VALUES:
        return status
    return "panic"


def result_outcome_key(
    result: TraceRunResult,
    judge_labels: bool = False,
) -> tuple[str, str | None, str | None]:
    if judge_labels:
        if result.status == "success" and result.output_key in {JUDGE_PASS, JUDGE_FAIL}:
            return (result.output_key, None, None)
        return (result.status, None, None)
    if result.status == "success":
        return ("success", result.output_key, None)
    if result.status == "timeout":
        return ("timeout", None, None)
    return (result.status, result.error_type, result.error_message)


def run_program_inputs(
    program: Program,
    inputs: list[Any],
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    capture_output_key: bool = False,
    judge_context_code: str | None = None,
) -> dict[str, Any]:
    total_lines = executable_lines(program.code)
    covered_lines = set()
    status_counts = {status: 0 for status in STATUS_VALUES}
    input_results: list[TraceRunResult] = []

    for input_args in inputs:
        if judge_context_code is None:
            result = trace_program_input(
                program,
                input_args,
                interpreter=interpreter,
                timeout=timeout,
                capture_output_key=capture_output_key,
            )
        else:
            result = trace_program_input(
                program,
                input_args,
                interpreter=interpreter,
                timeout=timeout,
                capture_output_key=capture_output_key,
                judge_context_code=judge_context_code,
            )
        result.status = normalized_status(result.status)
        status_counts[result.status] += 1
        covered_lines.update(result.covered_lines)
        input_results.append(result)

    covered_executable_lines = covered_lines & total_lines
    total_line_count = len(total_lines)
    covered_line_count = len(covered_executable_lines)
    line_coverage = covered_line_count / total_line_count if total_line_count else 0.0

    return {
        "input_results": input_results,
        "covered_lines": covered_line_count,
        "total_lines": total_line_count,
        "line_coverage": line_coverage,
        "success_count": status_counts["success"],
        "error_count": status_counts["error"],
        "timeout_count": status_counts["timeout"],
        "panic_count": status_counts["panic"],
    }


def run_program_tests(
    program: Program,
    tests: list[TestFunction],
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    capture_output_key: bool = False,
) -> dict[str, Any]:
    total_lines = executable_lines(program.code)
    covered_lines = set()
    status_counts = {status: 0 for status in STATUS_VALUES}
    input_results: list[TraceRunResult] = []

    for test in tests:
        result = trace_program_test(
            program,
            test,
            interpreter=interpreter,
            timeout=timeout,
            capture_output_key=capture_output_key,
        )
        result.status = normalized_status(result.status)
        status_counts[result.status] += 1
        covered_lines.update(result.covered_lines)
        input_results.append(result)

    covered_executable_lines = covered_lines & total_lines
    total_line_count = len(total_lines)
    covered_line_count = len(covered_executable_lines)
    line_coverage = covered_line_count / total_line_count if total_line_count else 0.0

    return {
        "input_results": input_results,
        "covered_lines": covered_line_count,
        "total_lines": total_line_count,
        "line_coverage": line_coverage,
        "success_count": status_counts["success"],
        "error_count": status_counts["error"],
        "timeout_count": status_counts["timeout"],
        "panic_count": status_counts["panic"],
    }


def oracle_inputs_from_all_input_output_tests(task: Task) -> list[Any]:
    return [test.inputs for test in task.tests if isinstance(test, InputOutput)]


def oracle_equivalence_classes(
    program_work: list[tuple[int, str, Program]],
    oracle_cases: list[Any],
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    jobs: int = 1,
    log_fn: Callable[[str], None] | None = None,
    judge_context_code: str | None = None,
    oracle_case_description: str = "InputOutput tests",
) -> tuple[
    list[tuple[int, str, Program]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    def classify_one(
        item: tuple[int, str, Program],
    ) -> tuple[int, str, Program, tuple[Any, ...], dict[str, Any]]:
        program_index, program_hash, program = item
        if judge_context_code is None:
            run = run_program_inputs(
                program,
                oracle_cases,
                interpreter=interpreter,
                timeout=timeout,
                capture_output_key=True,
            )
        else:
            run = run_program_tests(
                program,
                oracle_cases,
                interpreter=interpreter,
                timeout=timeout,
                capture_output_key=True,
            )
        outcome_vector = tuple(
            result_outcome_key(result, judge_labels=judge_context_code is not None)
            for result in run["input_results"]
        )
        return program_index, program_hash, program, outcome_vector, run

    def log_classified_program(program_index: int, program_hash: str, run: dict[str, Any]) -> None:
        log(
            "  "
            f"oracle program {program_index}/{len(program_work)} "
            f"{program_hash[:12]} "
            f"success={run['success_count']} error={run['error_count']} "
            f"timeout={run['timeout_count']} panic={run['panic_count']}"
        )

    log(
        "  oracle inputs: "
        f"{len(oracle_cases)} {oracle_case_description} for {len(program_work)} programs"
    )
    if jobs == 1 or len(program_work) <= 1:
        classified = []
        for item in program_work:
            classified_item = classify_one(item)
            classified.append(classified_item)
            program_index, program_hash, _, _, run = classified_item
            log_classified_program(program_index, program_hash, run)
    else:
        worker_count = min(jobs, len(program_work))
        log(f"  classifying oracle equivalence with {worker_count} workers")
        classified = []
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(classify_one, item) for item in program_work]
            for future in as_completed(futures):
                classified_item = future.result()
                classified.append(classified_item)
                program_index, program_hash, _, _, run = classified_item
                log_classified_program(program_index, program_hash, run)

    classified.sort(key=lambda item: item[0])
    vector_to_items: dict[tuple[Any, ...], list[tuple[int, str, Program]]] = {}
    oracle_program_runs = []
    for program_index, program_hash, program, outcome_vector, _ in classified:
        vector_to_items.setdefault(outcome_vector, []).append((program_index, program_hash, program))
    for _, program_hash, _, _, run in classified:
        run["program_hash"] = program_hash
        oracle_program_runs.append(run)

    representatives = []
    representative_metadata = {}
    for class_id, members in enumerate(vector_to_items.values(), start=1):
        representative = members[0]
        representatives.append(representative)
        representative_metadata[representative[1]] = {
            "oracle_equivalence_class_id": class_id,
            "oracle_equivalence_class_size": len(members),
        }

    log(
        "  oracle equivalence: "
        f"{len(representatives)}/{len(program_work)} classes "
        f"over {len(oracle_cases)} inputs"
    )
    return representatives, representative_metadata, oracle_program_runs


def coverage_row_from_run(
    task_id: str,
    selector_id: str,
    program_hash: str,
    run: dict[str, Any],
    input_source: str,
    num_inputs: int,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "selector_id": selector_id,
        "program_hash": program_hash,
        "input_source": input_source,
        "num_inputs": num_inputs,
        "covered_lines": run["covered_lines"],
        "total_lines": run["total_lines"],
        "line_coverage": run["line_coverage"],
        "success_count": run["success_count"],
        "error_count": run["error_count"],
        "timeout_count": run["timeout_count"],
        "panic_count": run["panic_count"],
    }


def compute_program_coverage(
    task_id: str,
    selector_id: str,
    program_hash: str,
    program: Program,
    inputs: list[Any],
    input_source: str,
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    judge_context_code: str | None = None,
) -> dict[str, Any]:
    run = run_program_inputs(
        program,
        inputs,
        interpreter=interpreter,
        timeout=timeout,
        judge_context_code=judge_context_code,
    )
    return coverage_row_from_run(
        task_id,
        selector_id,
        program_hash,
        run,
        input_source,
        len(inputs),
    )


def skipped_task_row(
    task_id: str,
    selector_id: str,
    reason: str,
    detail: str = "",
    input_source: str = "",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "selector_id": selector_id,
        "status": "skipped",
        "skip_reason": reason,
        "detail": detail,
        "input_source": input_source,
        "num_inputs": 0,
        "program_count": 0,
        "mean_line_coverage": "",
        "weighted_line_coverage": "",
        "covered_lines": 0,
        "total_lines": 0,
    }


def skipped_input_quality_row(
    task_id: str,
    selector_id: str,
    reason: str,
    detail: str = "",
    input_source: str = "",
    original_program_count: int = 0,
    oracle_equivalence_class_count: int = 0,
    oracle_equivalence_input_count: int = 0,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "selector_id": selector_id,
        "status": "skipped",
        "skip_reason": reason,
        "detail": detail,
        "input_source": input_source,
        "num_inputs": 0,
        "valid_input_count": 0,
        "program_count": 0,
        "original_program_count": original_program_count,
        "oracle_equivalence_class_count": oracle_equivalence_class_count,
        "oracle_equivalence_input_count": oracle_equivalence_input_count,
        "valid_program_input_count": 0,
        "differential_test_count": 0,
        "weighted_differential_test_rate": "",
        "mean_program_differential_test_rate": "",
        "valid_execution_rate": "",
        "distinguishing_input_count": 0,
        "distinguishing_input_rate": "",
        "mean_output_class_count": "",
        "success_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "panic_count": 0,
    }


def completed_task_row(
    task_id: str,
    selector_id: str,
    program_rows: list[dict[str, Any]],
    input_source: str,
    num_inputs: int,
) -> dict[str, Any]:
    covered = sum(int(row["covered_lines"]) for row in program_rows)
    total = sum(int(row["total_lines"]) for row in program_rows)
    mean = (
        sum(float(row["line_coverage"]) for row in program_rows) / len(program_rows)
        if program_rows
        else 0.0
    )
    weighted = covered / total if total else 0.0
    return {
        "task_id": task_id,
        "selector_id": selector_id,
        "status": "ok",
        "skip_reason": "",
        "detail": "",
        "input_source": input_source,
        "num_inputs": num_inputs,
        "program_count": len(program_rows),
        "mean_line_coverage": mean,
        "weighted_line_coverage": weighted,
        "covered_lines": covered,
        "total_lines": total,
    }


def completed_count_rows(
    task_id: str,
    selector_id: str,
    program_runs: list[dict[str, Any]],
    input_source: str,
    num_inputs: int,
    original_program_count: int,
    oracle_equivalence_input_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_class_counts = []
    distinguishing_inputs = []
    for input_index in range(num_inputs):
        output_keys = {
            result.output_key
            for run in program_runs
            for result in [run["input_results"][input_index]]
            if result.status == "success" and result.output_key is not None
        }
        output_class_count = len(output_keys)
        output_class_counts.append(output_class_count)
        distinguishing_inputs.append(output_class_count >= 2)

    program_rows = []
    for run in program_runs:
        input_results = run["input_results"]
        valid_input_count = sum(1 for result in input_results if result.status == "success")
        differential_test_count = sum(
            1
            for input_index, result in enumerate(input_results)
            if distinguishing_inputs[input_index] and result.status == "success"
        )
        program_rows.append(
            {
                "task_id": task_id,
                "selector_id": selector_id,
                "program_hash": run["program_hash"],
                "oracle_equivalence_class_id": run["oracle_equivalence_class_id"],
                "oracle_equivalence_class_size": run["oracle_equivalence_class_size"],
                "input_source": input_source,
                "num_inputs": num_inputs,
                "valid_input_count": valid_input_count,
                "differential_test_count": differential_test_count,
                "differential_test_rate": (
                    differential_test_count / valid_input_count if valid_input_count else 0.0
                ),
                "success_count": run["success_count"],
                "error_count": run["error_count"],
                "timeout_count": run["timeout_count"],
                "panic_count": run["panic_count"],
            }
        )

    total_runs = len(program_runs) * num_inputs
    total_success = sum(int(run["success_count"]) for run in program_runs)
    valid_input_count = sum(1 for count in output_class_counts if count > 0)
    valid_program_input_count = sum(int(row["valid_input_count"]) for row in program_rows)
    differential_test_count = sum(int(row["differential_test_count"]) for row in program_rows)
    distinguishing_input_count = sum(1 for is_distinguishing in distinguishing_inputs if is_distinguishing)
    task_row = {
        "task_id": task_id,
        "selector_id": selector_id,
        "status": "ok",
        "skip_reason": "",
        "detail": "",
        "input_source": input_source,
        "num_inputs": num_inputs,
        "valid_input_count": valid_input_count,
        "program_count": len(program_runs),
        "original_program_count": original_program_count,
        "oracle_equivalence_class_count": len(program_runs),
        "oracle_equivalence_input_count": oracle_equivalence_input_count,
        "valid_program_input_count": valid_program_input_count,
        "differential_test_count": differential_test_count,
        "weighted_differential_test_rate": (
            differential_test_count / valid_program_input_count
            if valid_program_input_count
            else 0.0
        ),
        "mean_program_differential_test_rate": (
            sum(float(row["differential_test_rate"]) for row in program_rows)
            / len(program_rows)
            if program_rows
            else 0.0
        ),
        "valid_execution_rate": total_success / total_runs if total_runs else 0.0,
        "distinguishing_input_count": distinguishing_input_count,
        "distinguishing_input_rate": (
            distinguishing_input_count / num_inputs if num_inputs else 0.0
        ),
        "mean_output_class_count": (
            sum(output_class_counts) / len(output_class_counts) if output_class_counts else 0.0
        ),
        "success_count": total_success,
        "error_count": sum(int(run["error_count"]) for run in program_runs),
        "timeout_count": sum(int(run["timeout_count"]) for run in program_runs),
        "panic_count": sum(int(run["panic_count"]) for run in program_runs),
    }
    return program_rows, task_row


def representative_count_runs(
    program_runs: list[dict[str, Any]],
    representative_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    count_program_runs = []
    for run in program_runs:
        metadata = representative_metadata.get(run["program_hash"])
        if metadata is None:
            continue
        count_run = dict(run)
        count_run.update(metadata)
        count_program_runs.append(count_run)
    return count_program_runs


def append_skipped_metric_rows(
    rows: MetricRows,
    metric: str,
    task_id: str,
    selector_id: str,
    reason: str,
    detail: str = "",
    input_source: str = "",
    include_oracle_outputs: bool = True,
) -> None:
    if wants_coverage(metric):
        rows.coverage_task_rows.append(
            skipped_task_row(task_id, selector_id, reason, detail, input_source)
        )
    if wants_count(metric):
        rows.input_quality_task_rows.append(
            skipped_input_quality_row(task_id, selector_id, reason, detail, input_source)
        )
        if include_oracle_outputs:
            rows.oracle_coverage_task_rows.append(
                skipped_task_row(
                    task_id,
                    selector_id,
                    reason,
                    detail,
                    ORACLE_COVERAGE_INPUT_SOURCE,
                )
            )
            rows.oracle_input_quality_task_rows.append(
                skipped_input_quality_row(
                    task_id,
                    selector_id,
                    reason,
                    detail,
                    ORACLE_COVERAGE_INPUT_SOURCE,
                )
            )


def compute_metric_rows(
    objects: list[dict[str, Any]],
    tasks_by_id: dict[str, Task],
    content_map: dict[str, str],
    selector_id: str,
    input_provider: Callable[[Task], tuple[list[Any], str]],
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    log_fn: Callable[[str], None] | None = None,
    max_programs: int | None = None,
    on_task_done: Callable[[MetricRows], None] | None = None,
    jobs: int = 1,
    metric: str = METRIC_COVERAGE,
    initial_coverage_program_rows: list[dict[str, Any]] | None = None,
    initial_coverage_task_rows: list[dict[str, Any]] | None = None,
    initial_differential_program_rows: list[dict[str, Any]] | None = None,
    initial_input_quality_task_rows: list[dict[str, Any]] | None = None,
    initial_oracle_coverage_program_rows: list[dict[str, Any]] | None = None,
    initial_oracle_coverage_task_rows: list[dict[str, Any]] | None = None,
    initial_oracle_differential_program_rows: list[dict[str, Any]] | None = None,
    initial_oracle_input_quality_task_rows: list[dict[str, Any]] | None = None,
) -> MetricRows:
    rows = MetricRows(
        coverage_program_rows=list(initial_coverage_program_rows or []),
        coverage_task_rows=list(initial_coverage_task_rows or []),
        differential_program_rows=list(initial_differential_program_rows or []),
        input_quality_task_rows=list(initial_input_quality_task_rows or []),
        oracle_coverage_program_rows=list(initial_oracle_coverage_program_rows or []),
        oracle_coverage_task_rows=list(initial_oracle_coverage_task_rows or []),
        oracle_differential_program_rows=list(initial_oracle_differential_program_rows or []),
        oracle_input_quality_task_rows=list(initial_oracle_input_quality_task_rows or []),
    )

    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    def checkpoint() -> None:
        if on_task_done is not None:
            on_task_done(rows)

    for index, obj in enumerate(objects, start=1):
        task_id = str(obj.get("task_id", ""))
        log(f"[{index}/{len(objects)}] task {task_id}")
        task = tasks_by_id.get(task_id)
        if task is None:
            log("  skipped: missing_dataset_task")
            append_skipped_metric_rows(
                rows,
                metric,
                task_id,
                selector_id,
                "missing_dataset_task",
            )
            checkpoint()
            continue

        try:
            execution_spec = task_execution_spec(task)
        except TaskExecutionError as error:
            log(f"  skipped: unsupported_test_semantics ({error})")
            append_skipped_metric_rows(
                rows,
                metric,
                task_id,
                selector_id,
                "unsupported_test_semantics",
                str(error),
            )
            checkpoint()
            continue

        program_hashes, error = extract_program_hashes(obj, selector_id)
        if error is not None:
            log(f"  skipped: {error}")
            append_skipped_metric_rows(rows, metric, task_id, selector_id, error)
            checkpoint()
            continue
        if max_programs is not None:
            original_count = len(program_hashes)
            program_hashes = program_hashes[:max_programs]
            if len(program_hashes) != original_count:
                log(f"  limiting programs: {len(program_hashes)}/{original_count}")

        missing = [program_hash for program_hash in program_hashes if program_hash not in content_map]
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... ({len(missing)} total)"
            log(f"  skipped: missing_content ({preview})")
            append_skipped_metric_rows(
                rows,
                metric,
                task_id,
                selector_id,
                "missing_content",
                preview,
            )
            checkpoint()
            continue

        program_work = []
        for program_index, program_hash in enumerate(program_hashes, start=1):
            program = program_from_content(task.requirements, content_map[program_hash])
            program_work.append((program_index, program_hash, program))

        oracle_cases = execution_spec.oracle_cases
        representative_metadata: dict[str, dict[str, Any]] = {}
        execution_program_work = program_work
        skip_count_reason = None
        oracle_metrics_completed = False

        if wants_count(metric):
            if not oracle_cases:
                skip_count_reason = "missing_oracle_input_output_tests"
                log("  count skipped: missing_oracle_input_output_tests")
                if not wants_coverage(metric):
                    rows.input_quality_task_rows.append(
                        skipped_input_quality_row(
                            task_id,
                            selector_id,
                            skip_count_reason,
                            "task.tests contains no InputOutput inputs",
                            original_program_count=len(program_work),
                        )
                    )
                    rows.oracle_coverage_task_rows.append(
                        skipped_task_row(
                            task_id,
                            selector_id,
                            skip_count_reason,
                            "task.tests contains no InputOutput inputs",
                            ORACLE_COVERAGE_INPUT_SOURCE,
                        )
                    )
                    rows.oracle_input_quality_task_rows.append(
                        skipped_input_quality_row(
                            task_id,
                            selector_id,
                            skip_count_reason,
                            "task.tests contains no InputOutput inputs",
                            ORACLE_COVERAGE_INPUT_SOURCE,
                            original_program_count=len(program_work),
                        )
                    )
                    checkpoint()
                    continue
            else:
                (
                    representatives,
                    representative_metadata,
                    oracle_program_runs,
                ) = oracle_equivalence_classes(
                    program_work,
                    oracle_cases,
                    interpreter=interpreter,
                    timeout=timeout,
                    jobs=jobs,
                    log_fn=log,
                    judge_context_code=execution_spec.judge_context_code,
                    oracle_case_description=execution_spec.oracle_case_description,
                )
                current_oracle_coverage_rows = [
                    coverage_row_from_run(
                        task_id,
                        selector_id,
                        run["program_hash"],
                        run,
                        ORACLE_COVERAGE_INPUT_SOURCE,
                        len(oracle_cases),
                    )
                    for run in oracle_program_runs
                ]
                rows.oracle_coverage_program_rows.extend(current_oracle_coverage_rows)
                rows.oracle_coverage_task_rows.append(
                    completed_task_row(
                        task_id,
                        selector_id,
                        current_oracle_coverage_rows,
                        ORACLE_COVERAGE_INPUT_SOURCE,
                        len(oracle_cases),
                    )
                )
                oracle_count_program_runs = representative_count_runs(
                    oracle_program_runs,
                    representative_metadata,
                )
                oracle_differential_rows, oracle_input_quality_row = completed_count_rows(
                    task_id,
                    selector_id,
                    oracle_count_program_runs,
                    ORACLE_COVERAGE_INPUT_SOURCE,
                    len(oracle_cases),
                    original_program_count=len(program_work),
                    oracle_equivalence_input_count=len(oracle_cases),
                )
                rows.oracle_differential_program_rows.extend(oracle_differential_rows)
                rows.oracle_input_quality_task_rows.append(oracle_input_quality_row)
                oracle_metrics_completed = True
                if not wants_coverage(metric):
                    execution_program_work = representatives

        try:
            log("  generating plurality inputs")
            inputs, input_source = input_provider(task)
            log(f"  inputs: {len(inputs)} ({input_source})")
        except Exception as e:
            log(f"  skipped: input_generation_failed ({type(e).__name__}: {e})")
            append_skipped_metric_rows(
                rows,
                metric,
                task_id,
                selector_id,
                "input_generation_failed",
                f"{type(e).__name__}: {e}",
                include_oracle_outputs=not oracle_metrics_completed,
            )
            checkpoint()
            continue

        def run_one(
            item: tuple[int, str, Program],
            inputs=inputs,
            judge_context_code=execution_spec.judge_context_code,
        ) -> tuple[int, dict[str, Any]]:
            program_index, program_hash, program = item
            run = run_program_inputs(
                program,
                inputs,
                interpreter=interpreter,
                timeout=timeout,
                capture_output_key=wants_count(metric),
                judge_context_code=judge_context_code,
            )
            run["program_hash"] = program_hash
            return program_index, run

        original_program_count = len(program_hashes)

        def log_program_run(
            program_index: int,
            run: dict[str, Any],
            program_count=original_program_count,
        ) -> None:
            if wants_coverage(metric):
                metric_text = f"coverage={float(run['line_coverage']):.4f}"
            else:
                metric_text = "executed"
            log(
                "  "
                f"program {program_index}/{program_count} "
                f"{run['program_hash'][:12]} {metric_text} "
                f"success={run['success_count']} error={run['error_count']} "
                f"timeout={run['timeout_count']} panic={run['panic_count']}"
            )

        if jobs == 1 or len(execution_program_work) <= 1:
            indexed_runs = [run_one(item) for item in execution_program_work]
            for program_index, run in indexed_runs:
                log_program_run(program_index, run)
        else:
            worker_count = min(jobs, len(execution_program_work))
            log(f"  executing programs with {worker_count} workers")
            indexed_runs = []
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(run_one, item) for item in execution_program_work]
                for future in as_completed(futures):
                    program_index, run = future.result()
                    indexed_runs.append((program_index, run))
                    log_program_run(program_index, run)

        current_program_runs = [
            run for _, run in sorted(indexed_runs, key=lambda indexed_run: indexed_run[0])
        ]
        done_parts = []

        if wants_coverage(metric):
            current_coverage_rows = [
                coverage_row_from_run(
                    task_id,
                    selector_id,
                    run["program_hash"],
                    run,
                    input_source,
                    len(inputs),
                )
                for run in current_program_runs
            ]
            rows.coverage_program_rows.extend(current_coverage_rows)
            coverage_task_row = completed_task_row(
                task_id,
                selector_id,
                current_coverage_rows,
                input_source,
                len(inputs),
            )
            rows.coverage_task_rows.append(coverage_task_row)
            done_parts.append(
                f"mean={float(coverage_task_row['mean_line_coverage']):.4f} "
                f"weighted={float(coverage_task_row['weighted_line_coverage']):.4f}"
            )

        if wants_count(metric):
            if skip_count_reason is not None:
                rows.input_quality_task_rows.append(
                    skipped_input_quality_row(
                        task_id,
                        selector_id,
                        skip_count_reason,
                        "task.tests contains no InputOutput inputs",
                        input_source,
                        original_program_count=len(program_work),
                    )
                )
                rows.oracle_coverage_task_rows.append(
                    skipped_task_row(
                        task_id,
                        selector_id,
                        skip_count_reason,
                        "task.tests contains no InputOutput inputs",
                        ORACLE_COVERAGE_INPUT_SOURCE,
                    )
                )
                rows.oracle_input_quality_task_rows.append(
                    skipped_input_quality_row(
                        task_id,
                        selector_id,
                        skip_count_reason,
                        "task.tests contains no InputOutput inputs",
                        ORACLE_COVERAGE_INPUT_SOURCE,
                        original_program_count=len(program_work),
                    )
                )
                done_parts.append("count_skipped=missing_oracle_input_output_tests")
                log("  done: " + " ".join(done_parts))
                checkpoint()
                continue

            count_program_runs = representative_count_runs(current_program_runs, representative_metadata)
            differential_rows, input_quality_row = completed_count_rows(
                task_id,
                selector_id,
                count_program_runs,
                input_source,
                len(inputs),
                original_program_count=len(program_work),
                oracle_equivalence_input_count=len(oracle_cases),
            )
            rows.differential_program_rows.extend(differential_rows)
            rows.input_quality_task_rows.append(input_quality_row)
            done_parts.append(
                "distinguishing="
                f"{input_quality_row['distinguishing_input_count']}/{len(inputs)} "
                f"valid_exec={float(input_quality_row['valid_execution_rate']):.4f}"
            )

        log("  done: " + " ".join(done_parts))
        checkpoint()

    return rows


def compute_coverage_rows(
    objects: list[dict[str, Any]],
    tasks_by_id: dict[str, Task],
    content_map: dict[str, str],
    selector_id: str,
    input_provider: Callable[[Task], tuple[list[Any], str]],
    interpreter: str | Path | None = None,
    timeout: int = EXECUTION_TIMEOUT_SECONDS,
    log_fn: Callable[[str], None] | None = None,
    max_programs: int | None = None,
    on_task_done: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    jobs: int = 1,
    initial_program_rows: list[dict[str, Any]] | None = None,
    initial_task_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def coverage_checkpoint(rows: MetricRows) -> None:
        if on_task_done is not None:
            on_task_done(rows.coverage_program_rows, rows.coverage_task_rows)

    metric_rows = compute_metric_rows(
        objects,
        tasks_by_id,
        content_map,
        selector_id,
        input_provider,
        interpreter=interpreter,
        timeout=timeout,
        log_fn=log_fn,
        max_programs=max_programs,
        on_task_done=coverage_checkpoint,
        jobs=jobs,
        metric=METRIC_COVERAGE,
        initial_coverage_program_rows=initial_program_rows,
        initial_coverage_task_rows=initial_task_rows,
    )
    return metric_rows.coverage_program_rows, metric_rows.coverage_task_rows


def build_summary(
    program_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    args,
    data_dir: Path,
    output_dir: Path,
    input_source: str | None = None,
) -> dict[str, Any]:
    ok_task_rows = [row for row in task_rows if row["status"] == "ok"]
    skipped_task_rows = [row for row in task_rows if row["status"] != "ok"]
    total_covered = sum(int(row["covered_lines"]) for row in program_rows)
    total_lines = sum(int(row["total_lines"]) for row in program_rows)
    return {
        "selector_id": args.selector_id,
        "input_source": input_source or args.plurality_input_source,
        "execution_semantics": getattr(args, "execution_semantics", RAW_OUTPUT_SEMANTICS),
        "model": args.model,
        "metric": getattr(args, "metric", METRIC_COVERAGE),
        "max_programs": getattr(args, "max_programs", None),
        "jobs": getattr(args, "jobs", None),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "task_count": len(task_rows),
        "computed_task_count": len(ok_task_rows),
        "skipped_task_count": len(skipped_task_rows),
        "program_count": len(program_rows),
        "mean_program_line_coverage": (
            sum(float(row["line_coverage"]) for row in program_rows) / len(program_rows)
            if program_rows
            else 0.0
        ),
        "mean_task_line_coverage": (
            sum(float(row["mean_line_coverage"]) for row in ok_task_rows) / len(ok_task_rows)
            if ok_task_rows
            else 0.0
        ),
        "weighted_line_coverage": total_covered / total_lines if total_lines else 0.0,
        "covered_lines": total_covered,
        "total_lines": total_lines,
        "skipped_reasons": {
            reason: sum(1 for row in skipped_task_rows if row["skip_reason"] == reason)
            for reason in sorted({row["skip_reason"] for row in skipped_task_rows})
        },
    }


def build_differential_summary(
    program_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    args,
    data_dir: Path,
    output_dir: Path,
    input_source: str | None = None,
) -> dict[str, Any]:
    ok_task_rows = [row for row in task_rows if row["status"] == "ok"]
    skipped_task_rows = [row for row in task_rows if row["status"] != "ok"]
    eligible_task_rows = [
        row
        for row in ok_task_rows
        if int(row.get("oracle_equivalence_class_count") or 0) >= 2
    ]
    eligible_task_ids = {str(row["task_id"]) for row in eligible_task_rows}
    eligible_program_rows = [
        row for row in program_rows if str(row.get("task_id")) in eligible_task_ids
    ]
    total_valid_inputs = sum(int(row["valid_input_count"]) for row in eligible_program_rows)
    total_differential_tests = sum(
        int(row["differential_test_count"]) for row in eligible_program_rows
    )
    total_success = sum(int(row["success_count"]) for row in eligible_task_rows)
    total_runs = sum(
        int(row["program_count"]) * int(row["num_inputs"]) for row in eligible_task_rows
    )
    total_inputs = sum(int(row["num_inputs"]) for row in eligible_task_rows)
    total_valid_input_level_inputs = sum(
        int(row.get("valid_input_count") or 0) for row in eligible_task_rows
    )
    total_original_programs = sum(
        int(row.get("original_program_count") or 0) for row in ok_task_rows
    )
    total_oracle_equivalence_classes = sum(
        int(row.get("oracle_equivalence_class_count") or 0) for row in ok_task_rows
    )
    total_oracle_equivalence_inputs = sum(
        int(row.get("oracle_equivalence_input_count") or 0) for row in ok_task_rows
    )
    total_distinguishing_inputs = sum(
        int(row["distinguishing_input_count"]) for row in eligible_task_rows
    )
    weighted_output_class_count = sum(
        float(row["mean_output_class_count"]) * int(row["num_inputs"])
        for row in eligible_task_rows
    )
    return {
        "selector_id": args.selector_id,
        "differential_count_schema_version": DIFFERENTIAL_COUNT_SCHEMA_VERSION,
        "execution_semantics": getattr(args, "execution_semantics", RAW_OUTPUT_SEMANTICS),
        "oracle_equivalence_mode": getattr(
            args,
            "oracle_equivalence_mode",
            ORACLE_EQUIVALENCE_MODE,
        ),
        "input_source": input_source or args.plurality_input_source,
        "model": args.model,
        "metric": getattr(args, "metric", METRIC_COUNT),
        "max_programs": getattr(args, "max_programs", None),
        "jobs": getattr(args, "jobs", None),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "task_count": len(task_rows),
        "computed_task_count": len(ok_task_rows),
        "skipped_task_count": len(skipped_task_rows),
        "program_count": len(program_rows),
        "eligible_task_count": len(eligible_task_rows),
        "ineligible_single_class_task_count": len(ok_task_rows) - len(eligible_task_rows),
        "eligible_program_count": len(eligible_program_rows),
        "original_program_count": total_original_programs,
        "oracle_equivalence_class_count": total_oracle_equivalence_classes,
        "oracle_equivalence_input_count": total_oracle_equivalence_inputs,
        "valid_input_count": total_valid_inputs,
        "differential_test_count": total_differential_tests,
        "mean_program_differential_test_rate": (
            sum(float(row["differential_test_rate"]) for row in eligible_program_rows)
            / len(eligible_program_rows)
            if eligible_program_rows
            else 0.0
        ),
        "weighted_differential_test_rate": (
            total_differential_tests / total_valid_inputs if total_valid_inputs else 0.0
        ),
        "mean_task_weighted_differential_test_rate": (
            sum(
                float(row.get("weighted_differential_test_rate") or 0.0)
                for row in eligible_task_rows
            )
            / len(eligible_task_rows)
            if eligible_task_rows
            else 0.0
        ),
        "mean_task_mean_program_differential_test_rate": (
            sum(
                float(row.get("mean_program_differential_test_rate") or 0.0)
                for row in eligible_task_rows
            )
            / len(eligible_task_rows)
            if eligible_task_rows
            else 0.0
        ),
        "valid_execution_rate": total_success / total_runs if total_runs else 0.0,
        "mean_task_valid_input_count": (
            total_valid_input_level_inputs / len(eligible_task_rows)
            if eligible_task_rows
            else 0.0
        ),
        "distinguishing_input_count": total_distinguishing_inputs,
        "input_count": total_inputs,
        "distinguishing_input_rate": (
            total_distinguishing_inputs / total_inputs if total_inputs else 0.0
        ),
        "mean_task_distinguishing_input_rate": (
            sum(float(row["distinguishing_input_rate"]) for row in eligible_task_rows)
            / len(eligible_task_rows)
            if eligible_task_rows
            else 0.0
        ),
        "mean_output_class_count": (
            weighted_output_class_count / total_inputs if total_inputs else 0.0
        ),
        "skipped_reasons": {
            reason: sum(1 for row in skipped_task_rows if row["skip_reason"] == reason)
            for reason in sorted({row["skip_reason"] for row in skipped_task_rows})
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp_path.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_resume_summary(
    summary_path: Path,
    args,
    data_dir: Path,
    require_differential_schema: bool = False,
    input_source: str | None = None,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "selector_id": args.selector_id,
        "input_source": input_source or args.plurality_input_source,
        "model": args.model,
        "data_dir": str(data_dir),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"Cannot resume: existing {key}={summary.get(key)!r} "
                f"does not match current {key}={value!r}."
            )
    expected_semantics = getattr(args, "execution_semantics", RAW_OUTPUT_SEMANTICS)
    existing_semantics = summary.get("execution_semantics")
    if existing_semantics is None:
        if expected_semantics != RAW_OUTPUT_SEMANTICS:
            raise ValueError(
                "Cannot resume: existing outputs do not record judge-label execution semantics."
            )
    elif existing_semantics != expected_semantics:
        raise ValueError(
            "Cannot resume: existing execution_semantics="
            f"{existing_semantics!r} does not match current "
            f"execution_semantics={expected_semantics!r}."
        )
    current_metric = getattr(args, "metric", METRIC_COVERAGE)
    if "metric" in summary and summary.get("metric") != current_metric:
        raise ValueError(
            f"Cannot resume: existing metric={summary.get('metric')!r} "
            f"does not match current metric={current_metric!r}."
        )
    current_max_programs = getattr(args, "max_programs", None)
    if "max_programs" in summary and summary.get("max_programs") != current_max_programs:
        raise ValueError(
            "Cannot resume: existing max_programs="
            f"{summary.get('max_programs')!r} does not match current "
            f"max_programs={current_max_programs!r}."
        )
    if require_differential_schema:
        schema_version = summary.get("differential_count_schema_version")
        allow_legacy_raw_output = (
            schema_version == 6
            and existing_semantics is None
            and expected_semantics == RAW_OUTPUT_SEMANTICS
        )
        if not allow_legacy_raw_output and schema_version != DIFFERENTIAL_COUNT_SCHEMA_VERSION:
            raise ValueError(
                "Cannot resume: existing differential outputs do not use "
                f"schema version {DIFFERENTIAL_COUNT_SCHEMA_VERSION}."
            )
        expected_mode = getattr(args, "oracle_equivalence_mode", ORACLE_EQUIVALENCE_MODE)
        if summary.get("oracle_equivalence_mode") != expected_mode:
            raise ValueError(
                "Cannot resume: existing oracle_equivalence_mode="
                f"{summary.get('oracle_equivalence_mode')!r} does not match "
                f"current oracle_equivalence_mode={expected_mode!r}."
            )


def read_resume_outputs(
    output_dir: Path,
    args,
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    program_path = output_dir / "coverage_per_program.csv"
    task_path = output_dir / "coverage_per_task.csv"
    summary_path = output_dir / "coverage_summary.json"

    if not program_path.exists() and not task_path.exists():
        return [], []
    if not program_path.exists() or not task_path.exists():
        raise FileNotFoundError(
            "Cannot resume from a partial output directory: expected both "
            "coverage_per_program.csv and coverage_per_task.csv."
        )

    if summary_path.exists():
        validate_resume_summary(summary_path, args, data_dir)

    return read_csv_rows(program_path), read_csv_rows(task_path)


def read_resume_differential_outputs(
    output_dir: Path,
    args,
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    program_path = output_dir / "differential_per_program.csv"
    task_path = output_dir / "input_quality_per_task.csv"
    summary_path = output_dir / "differential_summary.json"

    if not program_path.exists() and not task_path.exists():
        return [], []
    if not program_path.exists() or not task_path.exists():
        raise FileNotFoundError(
            "Cannot resume from a partial output directory: expected both "
            "differential_per_program.csv and input_quality_per_task.csv."
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            "Cannot resume differential outputs without differential_summary.json."
        )
    validate_resume_summary(
        summary_path,
        args,
        data_dir,
        require_differential_schema=True,
    )

    return read_csv_rows(program_path), read_csv_rows(task_path)


def read_resume_oracle_coverage_outputs(
    output_dir: Path,
    args,
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    program_path = output_dir / "oracle_coverage_per_program.csv"
    task_path = output_dir / "oracle_coverage_per_task.csv"
    summary_path = output_dir / "oracle_coverage_summary.json"

    if not program_path.exists() and not task_path.exists():
        return [], []
    if not program_path.exists() or not task_path.exists():
        raise FileNotFoundError(
            "Cannot resume from a partial output directory: expected both "
            "oracle_coverage_per_program.csv and oracle_coverage_per_task.csv."
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            "Cannot resume oracle coverage outputs without oracle_coverage_summary.json."
        )

    validate_resume_summary(
        summary_path,
        args,
        data_dir,
        input_source=ORACLE_COVERAGE_INPUT_SOURCE,
    )

    return read_csv_rows(program_path), read_csv_rows(task_path)


def read_resume_oracle_differential_outputs(
    output_dir: Path,
    args,
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    program_path = output_dir / "oracle_differential_per_program.csv"
    task_path = output_dir / "oracle_input_quality_per_task.csv"
    summary_path = output_dir / "oracle_differential_summary.json"

    if not program_path.exists() and not task_path.exists():
        return [], []
    if not program_path.exists() or not task_path.exists():
        raise FileNotFoundError(
            "Cannot resume from a partial output directory: expected both "
            "oracle_differential_per_program.csv and oracle_input_quality_per_task.csv."
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            "Cannot resume oracle differential outputs without "
            "oracle_differential_summary.json."
        )

    validate_resume_summary(
        summary_path,
        args,
        data_dir,
        require_differential_schema=True,
        input_source=ORACLE_COVERAGE_INPUT_SOURCE,
    )

    return read_csv_rows(program_path), read_csv_rows(task_path)


def read_resume_metric_outputs(output_dir: Path, args, data_dir: Path) -> MetricRows:
    metric = getattr(args, "metric", METRIC_COVERAGE)
    rows = MetricRows()
    if wants_coverage(metric):
        rows.coverage_program_rows, rows.coverage_task_rows = read_resume_outputs(
            output_dir,
            args,
            data_dir,
        )
    if wants_count(metric):
        (
            rows.differential_program_rows,
            rows.input_quality_task_rows,
        ) = read_resume_differential_outputs(output_dir, args, data_dir)
        (
            rows.oracle_coverage_program_rows,
            rows.oracle_coverage_task_rows,
        ) = read_resume_oracle_coverage_outputs(output_dir, args, data_dir)
        (
            rows.oracle_differential_program_rows,
            rows.oracle_input_quality_task_rows,
        ) = read_resume_oracle_differential_outputs(output_dir, args, data_dir)
        count_task_ids = completed_task_ids(rows.input_quality_task_rows)
        oracle_coverage_task_ids = completed_task_ids(rows.oracle_coverage_task_rows)
        oracle_count_task_ids = completed_task_ids(rows.oracle_input_quality_task_rows)
        if count_task_ids != oracle_coverage_task_ids or count_task_ids != oracle_count_task_ids:
            raise ValueError(
                "Cannot resume: differential, oracle coverage, and oracle "
                "differential outputs contain different completed task IDs."
            )

    if metric == METRIC_BOTH:
        coverage_task_ids = completed_task_ids(rows.coverage_task_rows)
        count_task_ids = completed_task_ids(rows.input_quality_task_rows)
        if coverage_task_ids != count_task_ids:
            raise ValueError(
                "Cannot resume: coverage and differential outputs contain "
                "different completed task IDs."
            )

    return rows


def completed_task_ids(task_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("task_id"))
        for row in task_rows
        if row.get("task_id") not in (None, "")
    }


def write_outputs(
    output_dir: Path,
    program_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "coverage_per_program.csv", PROGRAM_FIELDS, program_rows)
    write_csv(output_dir / "coverage_per_task.csv", TASK_FIELDS, task_rows)
    summary_path = output_dir / "coverage_summary.json"
    tmp_summary_path = summary_path.with_name(f"{summary_path.name}.tmp")
    with tmp_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_summary_path.replace(summary_path)


def write_differential_outputs(
    output_dir: Path,
    program_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "differential_per_program.csv", DIFFERENTIAL_PROGRAM_FIELDS, program_rows)
    write_csv(output_dir / "input_quality_per_task.csv", INPUT_QUALITY_FIELDS, task_rows)
    summary_path = output_dir / "differential_summary.json"
    tmp_summary_path = summary_path.with_name(f"{summary_path.name}.tmp")
    with tmp_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_summary_path.replace(summary_path)


def write_oracle_coverage_outputs(
    output_dir: Path,
    program_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "oracle_coverage_per_program.csv", PROGRAM_FIELDS, program_rows)
    write_csv(output_dir / "oracle_coverage_per_task.csv", TASK_FIELDS, task_rows)
    summary_path = output_dir / "oracle_coverage_summary.json"
    tmp_summary_path = summary_path.with_name(f"{summary_path.name}.tmp")
    with tmp_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_summary_path.replace(summary_path)


def write_oracle_differential_outputs(
    output_dir: Path,
    program_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "oracle_differential_per_program.csv",
        DIFFERENTIAL_PROGRAM_FIELDS,
        program_rows,
    )
    write_csv(output_dir / "oracle_input_quality_per_task.csv", INPUT_QUALITY_FIELDS, task_rows)
    summary_path = output_dir / "oracle_differential_summary.json"
    tmp_summary_path = summary_path.with_name(f"{summary_path.name}.tmp")
    with tmp_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_summary_path.replace(summary_path)


def build_metric_summaries(
    rows: MetricRows,
    args,
    data_dir: Path,
    output_dir: Path,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    metric = getattr(args, "metric", METRIC_COVERAGE)
    coverage_summary = None
    differential_summary = None
    oracle_coverage_summary = None
    oracle_differential_summary = None
    if wants_coverage(metric):
        coverage_summary = build_summary(
            rows.coverage_program_rows,
            rows.coverage_task_rows,
            args,
            data_dir,
            output_dir,
        )
    if wants_count(metric):
        oracle_coverage_summary = build_summary(
            rows.oracle_coverage_program_rows,
            rows.oracle_coverage_task_rows,
            args,
            data_dir,
            output_dir,
            input_source=ORACLE_COVERAGE_INPUT_SOURCE,
        )
    if wants_count(metric):
        differential_summary = build_differential_summary(
            rows.differential_program_rows,
            rows.input_quality_task_rows,
            args,
            data_dir,
            output_dir,
        )
        oracle_differential_summary = build_differential_summary(
            rows.oracle_differential_program_rows,
            rows.oracle_input_quality_task_rows,
            args,
            data_dir,
            output_dir,
            input_source=ORACLE_COVERAGE_INPUT_SOURCE,
        )
    return (
        coverage_summary,
        differential_summary,
        oracle_coverage_summary,
        oracle_differential_summary,
    )


def write_metric_outputs(
    output_dir: Path,
    rows: MetricRows,
    coverage_summary: dict[str, Any] | None,
    differential_summary: dict[str, Any] | None,
    oracle_coverage_summary: dict[str, Any] | None,
    oracle_differential_summary: dict[str, Any] | None,
    metric: str,
) -> None:
    if wants_coverage(metric):
        if coverage_summary is None:
            raise ValueError("Missing coverage summary for coverage output.")
        write_outputs(
            output_dir,
            rows.coverage_program_rows,
            rows.coverage_task_rows,
            coverage_summary,
        )
    if wants_count(metric):
        if differential_summary is None:
            raise ValueError("Missing differential summary for differential output.")
        write_differential_outputs(
            output_dir,
            rows.differential_program_rows,
            rows.input_quality_task_rows,
            differential_summary,
        )
        if oracle_coverage_summary is None:
            raise ValueError("Missing oracle coverage summary for oracle coverage output.")
        write_oracle_coverage_outputs(
            output_dir,
            rows.oracle_coverage_program_rows,
            rows.oracle_coverage_task_rows,
            oracle_coverage_summary,
        )
        if oracle_differential_summary is None:
            raise ValueError("Missing oracle differential summary for oracle differential output.")
        write_oracle_differential_outputs(
            output_dir,
            rows.oracle_differential_program_rows,
            rows.oracle_input_quality_task_rows,
            oracle_differential_summary,
        )


def summarize(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"task_count: {summary['task_count']}",
            f"computed_task_count: {summary['computed_task_count']}",
            f"skipped_task_count: {summary['skipped_task_count']}",
            f"program_count: {summary['program_count']}",
            f"mean_program_line_coverage: {summary['mean_program_line_coverage']}",
            f"weighted_line_coverage: {summary['weighted_line_coverage']}",
            f"output_dir: {summary['output_dir']}",
        ]
    )


def summarize_differential(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"task_count: {summary['task_count']}",
            f"computed_task_count: {summary['computed_task_count']}",
            f"skipped_task_count: {summary['skipped_task_count']}",
            f"program_count: {summary['program_count']}",
            f"eligible_task_count: {summary['eligible_task_count']}",
            "ineligible_single_class_task_count: "
            f"{summary['ineligible_single_class_task_count']}",
            "mean_program_differential_test_rate: "
            f"{summary['mean_program_differential_test_rate']}",
            f"weighted_differential_test_rate: {summary['weighted_differential_test_rate']}",
            "mean_task_weighted_differential_test_rate: "
            f"{summary['mean_task_weighted_differential_test_rate']}",
            "mean_task_mean_program_differential_test_rate: "
            f"{summary['mean_task_mean_program_differential_test_rate']}",
            f"mean_task_valid_input_count: {summary['mean_task_valid_input_count']}",
            f"distinguishing_input_rate: {summary['distinguishing_input_rate']}",
            f"valid_execution_rate: {summary['valid_execution_rate']}",
            f"output_dir: {summary['output_dir']}",
        ]
    )


def summarize_metric_outputs(
    coverage_summary: dict[str, Any] | None,
    differential_summary: dict[str, Any] | None,
    oracle_coverage_summary: dict[str, Any] | None,
    oracle_differential_summary: dict[str, Any] | None,
    metric: str,
) -> str:
    if metric == METRIC_COVERAGE:
        if coverage_summary is None:
            raise ValueError("Missing coverage summary.")
        return summarize(coverage_summary)
    if metric == METRIC_COUNT:
        if (
            differential_summary is None
            or oracle_coverage_summary is None
            or oracle_differential_summary is None
        ):
            raise ValueError("Missing count summaries.")
        return "\n\n".join(
            [
                "differential:\n" + summarize_differential(differential_summary),
                "oracle_coverage:\n" + summarize(oracle_coverage_summary),
                "oracle_differential:\n"
                + summarize_differential(oracle_differential_summary),
            ]
        )
    if (
        coverage_summary is None
        or differential_summary is None
        or oracle_coverage_summary is None
        or oracle_differential_summary is None
    ):
        raise ValueError("Missing summary for both metric output.")
    return "\n\n".join(
        [
            "coverage:\n" + summarize(coverage_summary),
            "differential:\n" + summarize_differential(differential_summary),
            "oracle_coverage:\n" + summarize(oracle_coverage_summary),
            "oracle_differential:\n" + summarize_differential(oracle_differential_summary),
        ]
    )


def make_logger(quiet: bool) -> Callable[[str], None] | None:
    if quiet:
        return None

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    return log


def load_tasks(args) -> list[Task]:
    if args.task_list:
        task_ids = [
            line.strip()
            for line in Path(args.task_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset = [
            list(load_dataset(Path(args.dataset) / f"{task_id}.json"))[0]
            for task_id in task_ids
        ]
    elif args.task:
        dataset = [list(load_dataset(Path(args.dataset) / f"{args.task}.json"))[0]]
    else:
        dataset = list(load_dataset(Path(args.dataset)))

    if args.only:
        dataset = [task for task in dataset if task.id == args.only]
    if args.only_list:
        ids = {
            line.strip()
            for line in Path(args.only_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        dataset = [task for task in dataset if task.id in ids]
    return dataset


def build_model(args):
    if args.num_left_samples:
        just_tri_it.config.NUM_LEFT_SAMPLES = args.num_left_samples

    model = {
        "gpt-4o": CloseAI("gpt-4o", 1.0, max_batch=just_tri_it.config.NUM_LEFT_SAMPLES),
        "deepseek-v3": XMCP("ds/deepseek-v3", 1.0),
        "gemini-2.5-flash": CloseAI("gemini-2.5-flash", 1.0),
    }[args.model]

    just_tri_it.utils.CURRENT_MODEL = args.model
    model = Repeatable(setup_cache(model, args))

    return model


def make_input_provider(args, model, executor):
    def input_provider(task: Task) -> tuple[list[Any], str]:
        if args.plurality_input_source == "oracle":
            if task_execution_spec(task).semantics == JUDGE_LABEL_SEMANTICS:
                raise ValueError(
                    "--plurality-input-source oracle is unsupported for TestFunction "
                    "datasets; use generated inputs."
                )
            return oracle_inputs_from_tests(task.tests), "oracle"
        return generate_inputs(model, task.requirements, executor), "generated"

    return input_provider


def resolve_interpreter(test_venv: str | None) -> str:
    if test_venv:
        return str((Path(test_venv).resolve() / "bin" / "python"))
    return sys.executable


def main(argv=None):
    init_random()
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data)
    output_dir = Path(args.output_dir)
    log = make_logger(args.quiet)

    model = build_model(args)
    executor = SubprocessExecutor(Path(args.test_venv)) if args.test_venv else PersistentWorkerExecutor()
    try:
        tasks = load_tasks(args)
        args.execution_semantics, args.oracle_equivalence_mode = dataset_execution_metadata(tasks)
        if args.plurality_input_source == "oracle" and args.execution_semantics != RAW_OUTPUT_SEMANTICS:
            raise ValueError(
                "--plurality-input-source oracle is only available for InputOutput datasets; "
                "use generated inputs for TestFunction datasets."
            )
        tasks_by_id = {task.id: task for task in tasks}
        objects = read_jsonl_objects(data_dir / "data.jsonl")
        content_map = read_content_map(data_dir / "content")
        input_provider = make_input_provider(args, model, executor)
        initial_rows = MetricRows()
        objects_to_process = objects

        if args.resume:
            initial_rows = read_resume_metric_outputs(output_dir, args, data_dir)
            if wants_coverage(args.metric):
                done_task_ids = completed_task_ids(initial_rows.coverage_task_rows)
            else:
                done_task_ids = completed_task_ids(initial_rows.input_quality_task_rows)
            objects_to_process = [
                obj
                for obj in objects
                if str(obj.get("task_id", "")) not in done_task_ids
            ]
            if log is not None:
                log(
                    "Resume enabled: "
                    f"loaded {len(done_task_ids)} completed tasks, "
                    f"{len(objects_to_process)} remaining"
                )

        def save_checkpoint(current_rows: MetricRows) -> None:
            (
                coverage_summary,
                differential_summary,
                oracle_coverage_summary,
                oracle_differential_summary,
            ) = build_metric_summaries(
                current_rows,
                args,
                data_dir,
                output_dir,
            )
            write_metric_outputs(
                output_dir,
                current_rows,
                coverage_summary,
                differential_summary,
                oracle_coverage_summary,
                oracle_differential_summary,
                args.metric,
            )
            if log is not None:
                if wants_coverage(args.metric):
                    task_count = len(current_rows.coverage_task_rows)
                else:
                    task_count = len(current_rows.input_quality_task_rows)
                log(f"  checkpoint saved: {task_count}/{len(objects)} tasks")

        if log is not None:
            input_model_text = (
                f"{model.alias} ({model.model_name})"
                if args.plurality_input_source == "generated"
                else "oracle inputs"
            )
            log(
                "Starting plurality input coverage: "
                f"tasks={len(objects)} selector={args.selector_id} "
                f"input_source={args.plurality_input_source} "
                f"metric={args.metric} jobs={args.jobs}"
            )
            log(f"Task execution semantics: {args.execution_semantics}")
            log(f"Experiment sample model: {model.alias} ({model.model_name})")
            log(f"Plurality input generation model: {input_model_text}")

        rows = compute_metric_rows(
            objects_to_process,
            tasks_by_id,
            content_map,
            args.selector_id,
            input_provider,
            interpreter=resolve_interpreter(args.test_venv),
            log_fn=log,
            max_programs=args.max_programs,
            on_task_done=save_checkpoint,
            jobs=args.jobs,
            metric=args.metric,
            initial_coverage_program_rows=initial_rows.coverage_program_rows,
            initial_coverage_task_rows=initial_rows.coverage_task_rows,
            initial_differential_program_rows=initial_rows.differential_program_rows,
            initial_input_quality_task_rows=initial_rows.input_quality_task_rows,
            initial_oracle_coverage_program_rows=initial_rows.oracle_coverage_program_rows,
            initial_oracle_coverage_task_rows=initial_rows.oracle_coverage_task_rows,
            initial_oracle_differential_program_rows=(
                initial_rows.oracle_differential_program_rows
            ),
            initial_oracle_input_quality_task_rows=initial_rows.oracle_input_quality_task_rows,
        )
        (
            coverage_summary,
            differential_summary,
            oracle_coverage_summary,
            oracle_differential_summary,
        ) = build_metric_summaries(
            rows,
            args,
            data_dir,
            output_dir,
        )
        if log is not None:
            log(f"Writing final {args.metric} outputs to {output_dir}")
        write_metric_outputs(
            output_dir,
            rows,
            coverage_summary,
            differential_summary,
            oracle_coverage_summary,
            oracle_differential_summary,
            args.metric,
        )
        print(
            summarize_metric_outputs(
                coverage_summary,
                differential_summary,
                oracle_coverage_summary,
                oracle_differential_summary,
                args.metric,
            )
        )

        skipped_task_count = 0
        if coverage_summary is not None:
            skipped_task_count = max(skipped_task_count, coverage_summary["skipped_task_count"])
        if differential_summary is not None:
            skipped_task_count = max(skipped_task_count, differential_summary["skipped_task_count"])
        if oracle_coverage_summary is not None:
            skipped_task_count = max(skipped_task_count, oracle_coverage_summary["skipped_task_count"])
        if oracle_differential_summary is not None:
            skipped_task_count = max(
                skipped_task_count,
                oracle_differential_summary["skipped_task_count"],
            )
        if args.strict and skipped_task_count > 0:
            raise SystemExit(
                "Strict mode failed: "
                f"{skipped_task_count} tasks were skipped."
            )
        if args.metric == METRIC_COVERAGE:
            return coverage_summary
        if args.metric == METRIC_COUNT:
            return {
                "differential": differential_summary,
                "oracle_coverage": oracle_coverage_summary,
                "oracle_differential": oracle_differential_summary,
            }
        return {
            "coverage": coverage_summary,
            "differential": differential_summary,
            "oracle_coverage": oracle_coverage_summary,
            "oracle_differential": oracle_differential_summary,
        }
    finally:
        executor.shutdown()


if __name__ == "__main__":
    main()
