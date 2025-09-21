from dataclasses import dataclass
import argparse
import sys
import json
import hashlib
from pathlib import Path
from itertools import islice
import jsonlines
from typing import List, Any, Dict

from just_tri_it.cached_llm import Model, Repeatable, AI302, XMCP
from just_tri_it.program import Program, Test
from just_tri_it.executor import Executor, Pass, Fail, Timeout
from just_tri_it.dataset import Dataset, load_dataset
from just_tri_it.utils import (
    print_annotated_hr,
    add_cache_options,
    setup_cache,
    replace_with_hash_and_update_map,
    ExperimentFailure,
    print_legend
)
from just_tri_it.config import init_selectors, NUM_LEFT_SAMPLES
from just_tri_it.code_generator import Vanilla
from just_tri_it.selection import Selected, Abstained


def parse_args():
    parser = argparse.ArgumentParser()
    add_cache_options(parser)
    parser.add_argument(
        "--test-venv",
        type=str,
        help="Set virtual environment for testing generated programs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input file containing the dataset."
    )
    parser.add_argument(
        "--task-list",
        type=str,
        help="File with task IDs to run (all by default)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM to use."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Store your experiment result."
    )
    return parser.parse_args()


SELECTOR_IDS = [
    "Plurality",
    "MaxTest_Assert",
    "MaxTest_IO",
    "CodeT_Assert",
    "CodeT_IO",
    "Syntactic",
    "OffByOne",
    "Postcondition",
    "FOR_INV",
    "FOR_FIB"
]


@dataclass
class Database:
    objects: List[Any]
    content: Dict[str, str]

    @staticmethod
    def load(directory: Path) -> 'Database':
        content_dir = directory / "content"
        object_file = directory / "data.jsonl"
        object_file.touch(exist_ok=True)
        with jsonlines.open(object_file) as reader:
            objects = list(reader)
        id_to_content: Dict[str, str] = {
            file_path.stem: file_path.read_text(encoding="utf-8")
            for file_path in content_dir.glob("*.txt")
        }
        return Database(objects, id_to_content)
                

    def save(self, directory: Path):
        content_dir = directory / "content"
        content_dir.mkdir(parents=True, exist_ok=True)
        object_file = directory / "data.jsonl"
        with jsonlines.open(object_file, mode='w', flush=True) as writer:
            writer.write_all(self.objects)

        for file_id, content in self.content.items():
            file_path = content_dir / f"{file_id}.txt"
            file_path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    
    model = AI302(args.model, 1.0)
    # model = XMCP(args.model, 1.0)

    model = setup_cache(model, args)

    # even without persistent cache, we need to ensure that we always
    # sample the same programs for a consistent database:
    model = Repeatable(model)
    
    executor = Executor(Path(args.test_venv))
    dataset = load_dataset(Path(args.dataset))
    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    database = Database.load(data_dir)

    if args.task_list:
        task_list_file = Path(args.task_list)
        with task_list_file.open("r", encoding="utf-8") as f:
            task_ids = [line.strip() for line in f]
            dataset = [t for t in dataset if t.id in task_ids]

    execute_experiment(model, executor, dataset, database, data_dir)


def execute_experiment(model, executor, dataset, db, data_dir):
    """Schema (for objects):
    {
        "task_id": ...,
        "requirements": ...,
        "sample_correctness": [
            [..., True, ['pass', 'pass', ...]],
            [..., False, ['pass', 'fail', ...]],
            ...
        ],
        "selectors": [
            {
                "id": ...,
                "outcome": ...,  # "selected" or "abstained"
                "selected": ...,
                "witnesses": ...,
                "raw_data": ...
            },
        ]
    }
    """

    print_legend()
    for task in dataset:
        if task.id in map(lambda o: o["task_id"], db.objects):
            continue

        obj = {
            "task_id": task.id,
            "requirements": task.requirements,
            "sample_correctness": [],
            "selectors": []
        }
        
        print()
        print_annotated_hr(f"Task {task.id}")

        print(f"\n[Sample correctness]", end="", file=sys.stderr, flush=True)
        samples = list(islice(Vanilla().generate(model, task.requirements, NUM_LEFT_SAMPLES), NUM_LEFT_SAMPLES))
        for s in samples:
            status, details = s.passes(executor, task.tests)
            obj["sample_correctness"].append((s, status, details))

        all_selectors = init_selectors(executor, Vanilla(), model)
        
        for selector_id in SELECTOR_IDS:
            print(f"\n[{selector_id}]", end="", file=sys.stderr, flush=True)
            
            selector_data = { "id": selector_id }
            try:
                selector = all_selectors[selector_id]

                if callable(selector):
                    s = selector(task)
                else:
                    s = selector
                outcome, raw_data = s.generate_and_select(model, task.requirements)
                selector_data["raw_data"] = raw_data
                match outcome:
                    case Selected(program, witnesses):
                        selector_data["outcome"] = "selected"
                        selector_data["selected"] = program
                        selector_data["witnesses"] = witnesses
                    case Abstained():
                        selector_data["outcome"] = "abstained"
                obj["selectors"].append(selector_data)
            except ExperimentFailure as e:
                print(f"\n{selector_id} failed on {task.id} with {e}", file=sys.stderr, flush=True)
    
        dbobj = replace_with_hash_and_update_map(obj, db.content)
        db.objects.append(dbobj)
        db.save(data_dir)
