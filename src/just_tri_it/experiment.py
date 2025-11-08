import json
from dataclasses import dataclass
import argparse
import sys
from pathlib import Path
from itertools import islice
import jsonlines
from typing import List, Any, Dict

from just_tri_it.cached_llm import Repeatable, AI302
from just_tri_it.executor import SubprocessExecutor, PersistentWorkerExecutor
from just_tri_it.dataset import load_dataset
from just_tri_it.utils import (
    print_annotated_hr,
    add_cache_options,
    setup_cache,
    replace_with_hash_and_update_map,
    ExperimentFailure,
    print_legend,
    init_random
)
import just_tri_it.utils
from just_tri_it.config import init_selectors
from just_tri_it.code_generator import Vanilla
from just_tri_it.selection import Selected, Abstained
from just_tri_it.input_generator import generate_inputs
import just_tri_it.config

GET_CORRECT = False


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
        "--task",
        type=str,
        help="task ID to run (all by default)."
    )
    parser.add_argument(
        "--only",
        type=str,
        help="task ID to run (all by default)."
    )
    parser.add_argument(
        "--only-list",
        type=str,
        help="file with task IDs to run"
    )
    parser.add_argument(
        "--num-left-samples",
        type=int,
        help="Number of left samples"
    )
    parser.add_argument(
        "--num-right-samples",
        type=int,
        help="Number of right samples"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM to use."
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Evaluate sample correctness."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Store your experiment result."
    )
    return parser.parse_args()


# to skip running some configurations:
SKIP_SELECTORS = [
    # "Plurality",
    # "MajorityVote",
    "MaxTest_Assert",
    "MaxTest_IO",
    # "CodeT_Assert",
    "CodeT_IO",
    # "Syntactic",
    # "OffByOne",
    # "Postcondition",
    # "FWD_INV",
    # "FWD_SINV",
    "ENUM_SINV"
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

    @staticmethod
    def load_ignore(directory: Path) -> 'Database':
        content_dir = directory / "content"
        object_file = directory / "data.jsonl"
        object_file.touch(exist_ok=True)

        objects = []
        with object_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Ignore invalid JSON lines {i} - {e}")
                    continue

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
    init_random()
    
    args = parse_args()

    if args.num_left_samples:
        just_tri_it.config.NUM_LEFT_SAMPLES = args.num_left_samples
    if args.num_right_samples:
        just_tri_it.config.NUM_RIGHT_SAMPLES = args.num_right_samples

    model = {
        "gpt-4o": AI302("gpt-4o", 1.0, max_batch=just_tri_it.config.NUM_LEFT_SAMPLES),
        "deepseek-v3": AI302("deepseek-v3.1", 1.0, alias="deepseek-v3"),
        "gemini-2.5-flash": AI302("gemini-2.5-flash", 1.0)
    }[args.model]
    just_tri_it.utils.CURRENT_MODEL = args.model
    
    # model = XMCP(args.model, 1.0)

    model = setup_cache(model, args)

    # even without persistent cache, we need to ensure that we always
    # sample the same programs for a consistent database:
    model = Repeatable(model)

    if args.test_venv:
        executor = SubprocessExecutor(Path(args.test_venv))
    else:
        executor = PersistentWorkerExecutor()

    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    database = Database.load_ignore(data_dir)

    if args.task_list:
        task_list_file = Path(args.task_list)
        with task_list_file.open("r", encoding="utf-8") as f:
            task_ids = [line.strip() for line in f]
            dataset = []
            for t_id in task_ids:
                file_path = Path(args.dataset) / (t_id + ".json")
                dataset.append(list(load_dataset(file_path))[0])
    elif args.task:
        file_path = Path(args.dataset) / (args.task + ".json")
        dataset = [list(load_dataset(file_path))[0]]
    else:
        dataset = load_dataset(Path(args.dataset))

    if args.only:
        dataset = [task for task in dataset if task.id == args.only]
    if args.only_list:
        path = Path(args.only_list)
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dataset = [task for task in dataset if task.id in ids]

    correctness_only = False
    if args.correctness:
        correctness_only= True

    execute_experiment(model, executor, dataset, database, data_dir, correctness_only)

    executor.shutdown()


def execute_experiment(model, executor, dataset, db, data_dir, correctness_only=False):
    """Schema (for objects):
    {
        "task_id": ...,
        "requirements": ...,
        "sample_correctness": [
            [..., True, ['Pass', 'Pass', ...]],
            [..., False, ['Pass', 'Fail', ...]],
            ...
        ],
        "selectors": [
            {
                "id": ...,
                "outcome": ...,  # "selected" or "abstained"
                "selected": ...,  # if selected
                "witnesses": ...,  # if selected
                "raw_data": ...
            },
        ]
    }
    """

    print_legend()
    task_num = len(dataset)

    for index, task in enumerate(dataset):
        if task.id == "HumanEval/129":
            continue
        if task.id in ["Mbpp/255", "Mbpp/260", "Mbpp/267", "Mbpp/300", "Mbpp/739", "Mbpp/765"]:
            continue
        if task.id in map(lambda o: o["task_id"], db.objects):
            continue
        just_tri_it.utils.CURRENT_TASK = task.id

        obj = {
            "task_id": task.id,
            "requirements": task.requirements,
            "sample_correctness": [],
            "selectors": []
        }
        
        print()
        print_annotated_hr(f"{index+1}/{task_num} —— Task {task.id}")

        print("\n[Sample correctness]", end="", file=sys.stderr, flush=True)
        if GET_CORRECT:
            samples = list(islice(Vanilla().generate(model, task.requirements, just_tri_it.config.NUM_LEFT_SAMPLES), just_tri_it.config.NUM_LEFT_SAMPLES))
            for s in samples:
                status, details = s.passes(executor, task.tests)
                obj["sample_correctness"].append((s, status, details))
        else:
            samples = list(islice(Vanilla().generate(model, task.requirements, just_tri_it.config.NUM_LEFT_SAMPLES), just_tri_it.config.NUM_LEFT_SAMPLES))
            for s in samples:
                obj["sample_correctness"].append((s, None, []))

        if not correctness_only:

            all_selectors = init_selectors(executor, Vanilla(), model)
            selector_ids = all_selectors.keys()

            for selector_id in selector_ids:
                if selector_id in SKIP_SELECTORS:
                    continue

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
