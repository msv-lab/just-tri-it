#!/usr/bin/env python3
"""Split LCB Part 6 task IDs into reproducible random groups."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("datasets/lcb_part6.json")
DEFAULT_OUTPUT_PREFIX = Path("datasets/lcb_part6_task_ids_group")
DEFAULT_GROUPS = 7
DEFAULT_SEED = 42
DEFAULT_EXPECTED_COUNT = 175


def load_tasks(input_path: Path) -> list[Any]:
    if input_path.suffix == ".jsonl":
        tasks: list[Any] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    tasks.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{input_path}:{line_number} is not valid JSON.") from exc
        return tasks

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON array of tasks.")
    return data


def extract_task_ids(tasks: list[Any], id_field: str) -> list[str]:
    task_ids: list[str] = []
    seen: set[str] = set()

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not a JSON object.")
        if id_field not in task:
            raise ValueError(f"Task at index {index} does not contain field {id_field!r}.")

        task_id = str(task[id_field])
        if task_id in seen:
            raise ValueError(f"Duplicate task ID found: {task_id}")

        seen.add(task_id)
        task_ids.append(task_id)

    return task_ids


def split_task_ids(task_ids: list[str], groups: int, seed: int) -> list[list[str]]:
    if groups <= 0:
        raise ValueError("groups must be positive.")
    if len(task_ids) % groups != 0:
        raise ValueError(f"{len(task_ids)} task IDs cannot be split evenly into {groups} groups.")

    shuffled = task_ids[:]
    random.Random(seed).shuffle(shuffled)

    group_size = len(shuffled) // groups
    return [shuffled[index : index + group_size] for index in range(0, len(shuffled), group_size)]


def output_path_for_group(output_prefix: Path, group_number: int) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_{group_number}.txt")


def write_groups(groups: list[list[str]], output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for group_number, task_ids in enumerate(groups, start=1):
        output_path = output_path_for_group(output_prefix, group_number)
        output_path.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly split LCB Part 6 task IDs into txt files, one ID per line."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON or JSONL dataset. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output file prefix. Default: {DEFAULT_OUTPUT_PREFIX}",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=DEFAULT_GROUPS,
        help=f"Number of groups to write. Default: {DEFAULT_GROUPS}",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Field name used as task ID. Default: id",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible splits. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_COUNT,
        help=f"Expected number of tasks. Default: {DEFAULT_EXPECTED_COUNT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tasks = load_tasks(args.input)
    if len(tasks) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} tasks in {args.input}, found {len(tasks)}.")

    task_ids = extract_task_ids(tasks, args.id_field)
    groups = split_task_ids(task_ids, args.groups, args.seed)
    output_paths = write_groups(groups, args.output_prefix)

    group_sizes = ", ".join(str(len(group)) for group in groups)
    print(
        f"Wrote {len(task_ids)} task IDs into {len(groups)} groups "
        f"using seed {args.seed}. Group sizes: {group_sizes}."
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
