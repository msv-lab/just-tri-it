import argparse
import sys
from pathlib import Path
from itertools import islice
from typing import List

from viberate.cached_llm import Model, Persistent, AI302, XMCP
from viberate.metrics import all_metrics_abt
from viberate.program import Program, Test
from viberate.executor import Executor, Pass, Fail
from viberate.dataset import Dataset, load_dataset
from viberate.code_generator import Vanilla, Generator, Selector, Abstained
from viberate.plurality import Plurality
from viberate.utils import print_annotated_hr
from viberate.codet import CodeT
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
        "--generator",
        type=str,
        help="Code generator configuration to benchmark."
    )
    parser.add_argument(
        "--selector",
        type=str,
        help="Code selector configuration to benchmark."
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
        help="Store your experiment result."
    )
    return parser.parse_args()


def passes_tests(executor: Executor, program: Program, tests: List[Test]) -> bool:
    ok = True
    for test in tests:
        match executor.run_test(program, test):
            case Pass():
                pass
            case Fail():
                ok = False
                break
    return ok


def evaluate_generator(model: Model, executor: Executor, generator: Generator, dataset: Dataset):
    N = 5
    for task in dataset:
        programs = islice(generator.generate(model, task.requirements), N)
        results = []
        for program in programs:
            if passes_tests(executor, program, task.tests):
                results.append(1)
            else:
                results.append(0)
        print(f"Task {task.id} pass@1: {sum(results)/len(results)}")


def evaluate_class_selector(model: Model, executor: Executor, selector: Selector, dataset: Dataset):
    c_prob_lst = []
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for task in dataset:
        print_annotated_hr(f"Task {task.id}")
        program_nums, program_list, decision, classes = selector.generate_and_select(model, task.requirements)

        correct_num = []
        print_annotated_hr("Run tests for checking correctness")
        for index, program in enumerate(program_list):
            print_annotated_hr(f"Running program {index}")
            if passes_tests(executor, program, task.tests):
                correct_num.append(index)
        print_annotated_hr("Index of correct programs")
        print(correct_num)

        c_prob_denominator = 0
        c_prob_numerator = None
        if not isinstance(decision, Abstained):
            for class_num, pid_list in classes.items():
                c_prob_denominator += len(pid_list) * (len(pid_list) - 1)
                if c_prob_numerator is None:
                    for pid in pid_list:
                        if pid in correct_num:
                            c_prob_numerator = len(pid_list) * (len(pid_list) - 1)
                            break
            if c_prob_numerator is None:
                c_prob_numerator = 0
            if c_prob_denominator != 0:
                c_prob_lst.append(c_prob_numerator / c_prob_denominator)
        else:
            c_prob_lst.append(None)
        print_annotated_hr("conditional probability")
        print(c_prob_lst, file=sys.stderr)

        if len(correct_num) > 0:
            # GT: no abstention
            if c_prob_numerator and c_prob_numerator > 0:
                n1 += 1
            elif isinstance(decision, Abstained):
                n3 += 1
            else:
                n2 += 1
        else:
            # GT: abstention
            if isinstance(decision, Abstained):
                n5 += 1
            else:
                n4 += 1
    all_metrics_abt(n1, n2, n3, n4, n5)


def evaluate_pair_selector(model, executor, selector, dataset):
    c_prob_lst = []
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for task in dataset:
        print_annotated_hr(f"Task {task.id}")
        program_nums, program_list, decision, pairs = selector.generate_and_select(model, task.requirements)

        correct_num = []
        print_annotated_hr("Run tests for checking correctness")
        for index, program in enumerate(program_list):
            print_annotated_hr(f"Running program {index}")
            if passes_tests(executor, program, task.tests):
                correct_num.append(index)
        print_annotated_hr("Index of correct programs")
        print(correct_num)

        c_prob_denominator = 0
        c_prob_numerator = 0
        if not isinstance(decision, Abstained):
            print_annotated_hr(f"Selected")
            print(pairs)
            for key, value in pairs.items():
                c_prob_denominator += len(value)  # the number of resonating pairs
                for pair in value:
                    if pair[0] in correct_num:
                        c_prob_numerator += 1  # the number of pair that contains a correct answer
            if c_prob_denominator != 0:
                c_prob_lst.append(c_prob_numerator / c_prob_denominator)
        else:
            c_prob_lst.append(None)
        print_annotated_hr("conditional probability")
        print(c_prob_lst, file=sys.stderr)

        if len(correct_num) > 0:
            # GT: no abstention
            if c_prob_numerator and c_prob_numerator > 0:
                n1 += 1
            elif isinstance(decision, Abstained):
                n3 += 1
            else:
                n2 += 1
        else:
            # GT: abstention
            if isinstance(decision, Abstained):
                n5 += 1
            else:
                n4 += 1
    all_metrics_abt(n1, n2, n3, n4, n5)


def evaluate_simple_selector(model, executor, selector, dataset):
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for task in dataset:
        print_annotated_hr(f"Task {task.id}")
        program_num, program_list, decision = selector.generate_and_select(model, task.requirements)
        correct_num = []
        for index, program in enumerate(program_list):
            if passes_tests(executor, program, task.tests):
                correct_num.append(index)
        if len(correct_num) > 0:
            # GT: no abstention
            if program_num and program_num in correct_num:
                n1 += 1
            elif isinstance(decision, Abstained):
                n3 += 1
            else:
                n2 += 1
        else:
            # GT: abstention
            if isinstance(decision, Abstained):
                n5 += 1
            else:
                n4 += 1
    all_metrics_abt(n1, n2, n3, n4, n5)


def main():
    args = parse_args()
    
    model = AI302(args.model, 1.0)
    # model = XMCP(args.model, 1.0)

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

    dataset = load_dataset(Path(args.dataset))

    if args.task:
        dataset = [t for t in dataset if t.id == args.task]

    GENERATORS = {
        "Vanilla": Vanilla()
    }

    SELECTORS = {
        "Plurality": Plurality(executor, Vanilla(), 5),
        "CodeT": CodeT(executor, Vanilla(), 5, 5),
        "VibeRate": VibeRate(executor, Vanilla(), 5, 5)
    }

    if args.generator:
        evaluate_generator(model, executor, GENERATORS[args.generator], dataset)

    if args.selector:
        print_annotated_hr(args.selector)
        match args.selector:
            case "Plurality":
                evaluate_class_selector(model, executor, SELECTORS[args.selector], dataset)
            case "CodeT":
                evaluate_simple_selector(model, executor, SELECTORS[args.selector], dataset)
            case "VibeRate":
                evaluate_pair_selector(model, executor, SELECTORS[args.selector], dataset)

    if args.experiment_result:
        result_root = Path(args.experiment_result)

    # print_annotated_hr("VibeRate")
    # evaluate_pair_selector(model, executor, SELECTORS["VibeRate"], dataset)


if __name__ == "__main__":
    main()
