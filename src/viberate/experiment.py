import argparse
import sys
import json
import hashlib
from pathlib import Path
from itertools import islice
import traceback
from typing import List

from viberate.cached_llm import Model, Persistent, AI302, XMCP
from viberate.metrics import all_metrics_abt
from viberate.program import Program, Test
from viberate.executor import Executor, Pass, Fail, Timeout, passes_tests
from viberate.dataset import Dataset, load_dataset
from viberate.code_generator import Vanilla, Generator, Selector, Abstained
from viberate.plurality import Plurality
from viberate.utils import print_annotated_hr
from viberate.codet import CodeT, MODE
from viberate.vb_selector import VibeRate


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
        help="Input file containing the dataset."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Identifier of task to run (all by default)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM to use."
    )
    parser.add_argument(
        "--experiment-result",
        type=str,
        required=True,
        help="Store your experiment result."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configurations of this experiment"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config_path = Path(args.config) 
        print(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model = AI302(config["model_name"], config["temp"])
    
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
            model.start_slicing(export_root)
                
        test_venv = Path(args.test_venv)
        executor = Executor(test_venv)

        dataset_path = Path(args.dataset)
        dataset = load_dataset(dataset_path)

        try:
            dataset_fullname = dataset_path.stem
            split_index = dataset_fullname.rfind('_')
            name = dataset_fullname[:split_index]
            batch = int(dataset_fullname[split_index + 1:])
        except:
            print("can't parse dataset name")
            name = "unknown"
            batch = "unknown"

        if args.task:
            dataset = [t for t in dataset if t.id == args.task]
        
        experiment_result = {
            "dataset": name,
            "batch": batch,
            "selectors": {},
            "generators": {}
        }

        if args.experiment_result:
            result_root = Path(args.experiment_result)
            # program directory store all generated programs during this experiment
            program_dir = result_root / "program"
            program_dir.mkdir(parents=True, exist_ok=True)
            # config.json store the configurations of this experiment
            with open(result_root / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            # program.json store all forward programs and their correctness by tests
            program_json_path = result_root / "program.json"
            if not program_json_path.exists():
                try:
                    result_root.mkdir(parents=True, exist_ok=True)
                    with open(program_json_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f, indent=4)
                    print(f"Created program.json at {program_json_path}")
                except Exception as e:
                    print(f"Error creating program.json: {e}")
            else:
                print(f"program.json already exists at {program_json_path}")
            
            global program_dict
            with open(program_json_path, 'r', encoding='utf-8') as f:
                program_dict = json.load(f) 

            if "generators" in config and len(config["generators"]):
                for gen in config["generators"]:
                    print_annotated_hr("Generator: " + gen)
                    match gen:
                        case "Vanilla":
                            num = config["generators"][gen]["number_of_programs"]
                            generator = Vanilla()
                            experiment_result["generators"][gen] = {"number_of_programs": num, 'results': []}
                            for task in dataset:
                                print_annotated_hr(f"Task {task.id}")
                                new_dict = {
                                    "task_id": task.id
                                }
                                programs = list(islice(generator.generate(model, task.requirements, program_dir), num))
                                new_dict.update({"generated_programs": [p.hash() for p in programs]})
                                program_dict = Selector.update_program_correctness(task.id, executor, programs, task.tests, program_dict)
                                experiment_result["generators"][gen]["results"].append(new_dict)
                        case _:
                            print("Unsupported generator", file=sys.stderr)
                            continue
                    
                            
            if "selectors" in config and len(config["selectors"]):
                for select in config["selectors"]:
                    print_annotated_hr("Selector: " + select)
                    match select:
                        case "Plurality":
                            num = config["selectors"][select]["number_of_programs"]
                            selector = Plurality(executor, Vanilla(), num)
                            init_dict = {
                                "number_of_programs": num,
                                "results": []
                            }
                        case "CodeT_assertion" | "CodeT_IOcompare":
                            num_p = config["selectors"][select]["number_of_programs"]
                            num_t = config["selectors"][select]["number_of_tests"]
                            mode = MODE.ASSERTION if select == "CodeT_assertion" else MODE.IO_COMPARE
                            selector = CodeT(executor, Vanilla(), mode, num_p, num_t)
                            init_dict = {
                                "number_of_programs": num_p,
                                "number_of_tests": num_t,
                                "results": []
                            }
                        case "VibeRate":
                            num_1 = config["selectors"][select]["number_of_program_1"]
                            num_2 = config["selectors"][select]["number_of_program_2"]
                            selector = VibeRate(executor, Vanilla(), num_1, num_2)
                            init_dict = {
                                "number_of_program_1": num_1,
                                "number_of_program_2": num_2,
                                "results": []
                            }
                        case _:
                            print("Unsupported selectors", file=sys.stderr)
                            continue
                    experiment_result["selectors"][select] = init_dict
                    for task in dataset:
                        print_annotated_hr(f"Task {task.id}")
                        new_dict = {
                            "task_id": task.id
                        }
                        # print(program_dict)
                        select_result = selector.generate_and_select(model, task, program_dir, program_dict)
                        new_dict.update(select_result[0])
                        program_dict = select_result[1]
                        experiment_result["selectors"][select]["results"].append(new_dict)
            # save into files
            try:
                # print(experiment_result)
                with open(result_root / f"{name}_batch_{batch}_raw.json", 'w', encoding='utf-8') as f:
                    json.dump(experiment_result, f, indent=4)
                with open(result_root / "program.json", 'w', encoding='utf-8') as f:
                    json.dump(program_dict, f, indent=4)
            except Exception as e:
                print(f"error when saving: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
