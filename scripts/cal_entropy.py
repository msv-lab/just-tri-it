import json
import math
import random
from dataclasses import dataclass
import argparse
import sys
from pathlib import Path
from itertools import islice
import jsonlines
from typing import List, Any, Dict
import os

from just_tri_it.input_generator import generate_inputs
from just_tri_it.executor import Executor, Success
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
from just_tri_it.program import InputOutput
from just_tri_it.plurality import UncertainOutput

GET_CORRECT = True


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
    return parser.parse_args()


# to skip running some configurations:
SKIP_SELECTORS = [
    # "Plurality",
    # "MajorityVote",
    # "MaxTest_Assert",
    # "MaxTest_IO",
    # "CodeT_Assert",
    # "CodeT_IO",
    # "Syntactic",
    # "OffByOne",
    # "Postcondition",
    # "FWD_INV",
    # "FWD_SINV",
    # "ENUM_SINV"
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

    execute_experiment(model, executor, dataset, data_dir)

    executor.shutdown()


def entropy_plugin(counts) -> float:
    counts = [c for c in counts if c > 0]
    n = sum(counts)
    if n == 0:
        return 0.0
    return -sum((c/n) * math.log((c/n), 2) for c in counts)

def entropy_millermadow(counts) -> float:
    counts = [c for c in counts if c > 0]
    n = sum(counts)
    if n == 0:
        return 0.0
    H_plug = entropy_plugin(counts)
    K_obs = len(counts)
    return H_plug + (K_obs - 1) / (2 * n * math.log(2))

def entropy_jeffreys(counts, alpha: float = 0.5) -> float:
    counts = [c for c in counts if c > 0]
    n = sum(counts)
    if n == 0:
        return 0.0
    K_obs = len(counts)
    denom = n + alpha * K_obs
    ps = [(c + alpha) / denom for c in counts]
    return -sum(p * math.log(p, 2) for p in ps)


def cal_entropy(model, executor, task, sample_num):
    generator = Vanilla()
    inputs = generate_inputs(model, task.requirements, executor)
    programs = list(islice(generator.generate(model, task.requirements, sample_num), sample_num))

    classes = []
    outputs = []
    for p in programs:
        results = []
        for i in inputs:
            match executor.run(p, i):
                case Success(v):
                    results.append(v)
                case _:
                    results.append(UncertainOutput())
        if len(classes) == 0:
            classes.append(0)
        else:
            found = False
            for i, outs in enumerate(outputs):
                if outs == results:
                    classes.append(classes[i])
                    found = True
                    break
            if not found:
                classes.append(max(classes) + 1)
        outputs.append(results)
    
    valid_class_to_programs = {}
    for i in range(len(programs)):
        class_id = classes[i]
        program = programs[i]
        output = outputs[i]
        if not all(isinstance(o, UncertainOutput) for o in output):
            if class_id not in valid_class_to_programs:
                valid_class_to_programs[class_id] = []
            valid_class_to_programs[class_id].append(program)

    counts = []
    for p_list in valid_class_to_programs.values():
       counts.append(len(p_list))
    
    H_plugin = entropy_plugin(counts)

    return H_plugin


def execute_experiment(model, executor, dataset, data_dir):
    print_legend()
    task_num = len(dataset)

    for sample_num in range(5, 51, 5):
        print("sample_num is " + str(sample_num) + "\n")
        task_dict = {}
        for index, task in enumerate(dataset):
            print("task: "+ task.id + "\n")
            try:
                entropy = cal_entropy(model, executor, task, sample_num)
                task_dict[task.id] = entropy
            except Exception as e:
                continue
        print("\n")
        try:
            jsonl_path = os.path.join(data_dir, "entropy.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                    line_data = {
                        "sample_num": sample_num,
                        "task_entropies": task_dict
                    }
                    json_line = json.dumps(line_data, ensure_ascii=False)
                    f.write(json_line + "\n")
            print(f"success write: {jsonl_path}")
        except Exception as e:
            print(f"fail write: {str(e)}")        
            

        
