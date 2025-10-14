#!/usr/bin/env python3
"""
dataset_json_filesystem_converter.py

Usage:
  dataset_json_filesystem_converter.py <json_or_dir> <dir_or_json>

Behavior:
- If the first argument is a JSON file and the second is a directory,
  convert JSON -> filesystem layout.
- If the first argument is a directory and the second is a JSON file,
  convert filesystem layout -> JSON.

JSON schema (list of objects):
[
  {
    "id": <string>,
    "requirements": {
      "signature": <string>,
      "description": <string>
    },
    "judge": <string>,
    "tests": <json>,  # any JSON-serializable value
    "metadata": {
      "correct_solution": <string>,      # optional
      "incorrect_solution": <string>,    # optional
      <other fields>                     # arbitrary additional fields
    }
  },
  ...
]

Filesystem layout under root directory:
<id>/signature.txt
<id>/description.txt
<id>/judge.py
<id>/tests.json
<id>/correct_solution.py                 # empty file if field absent
<id>/incorrect_solution.py               # empty file if field absent
<id>/other_metadata.json                 # metadata except correct/incorrect_solution
"""

import sys
import os
import json
from typing import Any, Dict, List


SIG_FILE = "signature.txt"
DESC_FILE = "description.txt"
JUDGE_FILE = "judge.py"
TESTS_FILE = "tests.json"
CORRECT_FILE = "correct_solution.py"
INCORRECT_FILE = "incorrect_solution.py"
OTHER_META_FILE = "other_metadata.json"

REQ_FIELDS = ["signature", "description"]


def is_json_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".json")


def is_directory(path: str) -> bool:
    return os.path.isdir(path)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_to_fs(json_path: str, root_dir: str) -> None:
    data = read_json(json_path)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of items.")

    os.makedirs(root_dir, exist_ok=True)

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be an object.")
        id_ = item.get("id")
        if not isinstance(id_, str) or not id_:
            raise ValueError("Each item must have a non-empty string 'id'.")

        requirements = item.get("requirements") or {}
        if not isinstance(requirements, dict):
            raise ValueError(f"Item {id_}: 'requirements' must be an object.")
        missing_req = [f for f in REQ_FIELDS if f not in requirements or not isinstance(requirements[f], str)]
        if missing_req:
            raise ValueError(f"Item {id_}: missing or invalid requirements fields: {missing_req}")

        judge = item.get("judge")
        if not isinstance(judge, str):
            raise ValueError(f"Item {id_}: 'judge' must be a string.")

        tests = item.get("tests", None)  # any JSON-serializable

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Item {id_}: 'metadata' must be an object.")

        correct_solution = metadata.get("correct_solution")
        incorrect_solution = metadata.get("incorrect_solution")

        # Build other metadata excluding correct/incorrect_solution
        other_metadata = {k: v for k, v in metadata.items() if k not in ("correct_solution", "incorrect_solution")}

        item_dir = os.path.join(root_dir, id_)
        os.makedirs(item_dir, exist_ok=True)

        # Write required files
        write_text(os.path.join(item_dir, SIG_FILE), requirements["signature"])
        write_text(os.path.join(item_dir, DESC_FILE), requirements["description"])
        write_text(os.path.join(item_dir, JUDGE_FILE), judge)
        write_json(os.path.join(item_dir, TESTS_FILE), tests)

        # Write solution files (empty if absent)
        write_text(os.path.join(item_dir, CORRECT_FILE), correct_solution or "")
        write_text(os.path.join(item_dir, INCORRECT_FILE), incorrect_solution or "")

        # Write other metadata
        write_json(os.path.join(item_dir, OTHER_META_FILE), other_metadata)


def fs_to_json(root_dir: str, json_path: str) -> None:
    if not is_directory(root_dir):
        raise ValueError(f"Directory not found: {root_dir}")

    items: List[Dict[str, Any]] = []

    for entry in sorted(os.listdir(root_dir)):
        item_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(item_dir):
            continue  # skip files at root
        id_ = entry

        # Required files check
        sig_path = os.path.join(item_dir, SIG_FILE)
        desc_path = os.path.join(item_dir, DESC_FILE)
        judge_path = os.path.join(item_dir, JUDGE_FILE)
        tests_path = os.path.join(item_dir, TESTS_FILE)
        other_meta_path = os.path.join(item_dir, OTHER_META_FILE)
        correct_path = os.path.join(item_dir, CORRECT_FILE)
        incorrect_path = os.path.join(item_dir, INCORRECT_FILE)

        missing = [p for p in [sig_path, desc_path, judge_path, tests_path, other_meta_path] if not os.path.isfile(p)]
        if missing:
            raise ValueError(f"Item {id_}: missing required files: {', '.join(os.path.basename(m) for m in missing)}")

        # Read required content
        signature = read_text(sig_path).strip()
        description = read_text(desc_path)
        judge = read_text(judge_path)
        tests = read_json(tests_path)

        # Read solution files (empty file means field absent)
        correct_solution = read_text(correct_path) if os.path.isfile(correct_path) else ""
        incorrect_solution = read_text(incorrect_path) if os.path.isfile(incorrect_path) else ""

        # Interpret empty as absent
        metadata: Dict[str, Any] = read_json(other_meta_path)
        if not isinstance(metadata, dict):
            raise ValueError(f"Item {id_}: other_metadate.json must be a JSON object.")

        if correct_solution:
            metadata["correct_solution"] = correct_solution
        if incorrect_solution:
            metadata["incorrect_solution"] = incorrect_solution

        item = {
            "id": id_,
            "requirements": {
                "signature": signature,
                "description": description,
            },
            "judge": judge,
            "tests": tests,
            "metadata": metadata,
        }
        items.append(item)

    write_json(json_path, items)


def main():
    if len(sys.argv) != 3:
        print("Usage: dataset_json_filesystem_converter.py <json_or_dir> <dir_or_json>", file=sys.stderr)
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]

    src_is_json = is_json_file(src)
    src_is_dir = is_directory(src)
    dst_is_json = dst.lower().endswith(".json")
    dst_is_dir = dst.endswith(os.sep) or not dst_is_json

    # Normalize: if dst exists and is a directory, treat as directory
    if os.path.exists(dst) and os.path.isdir(dst):
        dst_is_dir = True
        dst_is_json = False

    if src_is_json and (dst_is_dir or not os.path.exists(dst)):
        # JSON -> FS
        # Ensure destination directory exists or can be created
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)
        elif not os.path.isdir(dst):
            raise ValueError(f"Destination must be a directory for JSON->FS: {dst}")
        json_to_fs(src, dst)
    elif src_is_dir and dst_is_json:
        # FS -> JSON
        fs_to_json(src, dst)
    else:
        raise ValueError("Ambiguous or invalid arguments. Provide either <json> <directory> or <directory> <json>.")

if __name__ == "__main__":
    main()
