

import argparse
import json
from pathlib import Path
import sys
import traceback

from viberate.cached_llm import AI302
from viberate.metrics import all_metrics_abt
from viberate.utils import print_annotated_hr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-results",
        type=str,
        required=True,
        help="The directory where your input experiment results are stored."
    )
    return parser.parse_args()


def evaluate_selector_simple(results: list, p_dict: dict):
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for result in results:
        print_annotated_hr(f"Task {result['task_id']}")
        program_list = result["generated_programs"] # hash
        program_num = result["chosen_programs"] # index
        decision = result["decision"]

        correct_num = []
        for index, program in enumerate(program_list):
            if p_dict[program]:
                correct_num.append(index)

        if len(correct_num) > 0:
            # GT: no abstention
            if program_num and program_num[0] in correct_num:
                n1 += 1
            elif decision == "Abstained":
                n3 += 1
            else:
                n2 += 1
        else:
            # GT: abstention
            if decision == "Abstained":
                n5 += 1
            else:
                n4 += 1
    return len(results), [n1, n2, n3, n4, n5], None

def evaluate_selector_pair(results: list, p_dict: dict):
    c_prob_lst = []
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for result in results:
        print_annotated_hr(f"Task {result['task_id']}")
        program_list = result["programs"]["forward"] # hash
        decision = result["decision"]
        pairs = result["pairs"]

        correct_num = []
        for index, program in enumerate(program_list):
            if p_dict[program]:
                correct_num.append(index)

        c_prob_denominator = 0
        c_prob_numerator = 0
        if decision != "Abstained":
            for key, value in pairs.items():
                c_prob_denominator += len(value)  # the number of resonating pairs
                for pair in value:
                    if pair[0] in correct_num:
                        c_prob_numerator += 1  # the number of pair that contains a correct answer
            if c_prob_denominator != 0:
                c_prob_lst.append(c_prob_numerator / c_prob_denominator)
            else:
                c_prob_lst.append(None)
        else:
            c_prob_lst.append('Abstained')

        print_annotated_hr("conditional probability")
        print(c_prob_lst, file=sys.stderr)

        if len(correct_num) > 0:
            # GT: no abstention
            if c_prob_numerator and c_prob_numerator > 0:
                n1 += 1
            elif decision == "Abstained":
                n3 += 1
            else:
                n2 += 1
        else:
            # GT: abstention
            if decision == "Abstained":
                n5 += 1
            else:
                n4 += 1
    return len(results), [n1, n2, n3, n4, n5], c_prob_lst

def evaluate_selector_class(results: list, p_dict: dict):
    c_prob_lst = []
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    for result in results:
        print_annotated_hr(f"Task {result['task_id']}")
        program_num = result["chosen_programs"] # index
        program_list = result["generated_programs"] # hash
        decision = result["decision"]
        classes = result["classes"]

        correct_num = []
        for index, program in enumerate(program_list):
            if p_dict[program]:
                correct_num.append(index)

        c_prob_denominator = 0
        c_prob_numerator = None
        if decision != "Abstained":
            for p_class in classes:
                pid_list = p_class["program_indexes"]
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
                c_prob_lst.append('Abstained')
        else:
            c_prob_lst.append('Abstained')

        print_annotated_hr("conditional probability")
        print(c_prob_lst, file=sys.stderr)

        if len(correct_num) > 0:
            # GT: no abstention
            if c_prob_numerator and c_prob_numerator > 0:
                n1 += 1
            elif decision == "Abstained":
                n3 += 1
            else:
                n2 += 1
        else:
            # GT: abstention
            if decision == "Abstained":
                n5 += 1
            else:
                n4 += 1
    return len(results), [n1, n2, n3, n4, n5], c_prob_lst


def main():
    args = parse_args()
    try:
        input_dir = Path(args.experiment_results)
        
        program_dir = input_dir / "program.json"
        with open(program_dir, 'r', encoding='utf-8') as f:
            p_dict = json.load(f)

        batch_files = sorted(input_dir.glob("*_batch_*_raw.json"))
        metrics = {}
        for batch_file in batch_files:
            with open(batch_file, 'r', encoding='utf-8') as f:
                exp_results = json.load(f)
            select_dict = exp_results["selectors"]
            for selector in select_dict:
                print_annotated_hr(f"Evaluating selector: {selector}")
                if selector not in metrics:
                    metrics[selector] = {
                        "num": 0,
                        "n_lst": [0, 0, 0, 0, 0],
                        "c_prob_lst": []
                    }
                match selector:
                    case "Plurality":
                        num, n_lst, c_prob_lst = evaluate_selector_class(select_dict[selector]["results"], p_dict)
                    case "CodeT_IOcompare" | "CodeT_assertion":
                        num, n_lst, c_prob_lst = evaluate_selector_simple(select_dict[selector]["results"], p_dict)
                    case "VibeRate":
                        num, n_lst, c_prob_lst = evaluate_selector_pair(select_dict[selector]["results"], p_dict)
                    case _:
                        print(f"Unknown selector: {selector}")
                        continue
                metrics[selector]["num"] += num
                for i in range(5):
                    metrics[selector]["n_lst"][i] += n_lst[i]
                if c_prob_lst is not None:
                    metrics[selector]["c_prob_lst"] += c_prob_lst
        for selector in metrics:
            metrics[selector].update(all_metrics_abt(*metrics[selector]["n_lst"]))
            if metrics[selector]["c_prob_lst"]:
                filtered_c_prob_lst = [x for x in metrics[selector]["c_prob_lst"] if isinstance(x, (int, float))]
                if filtered_c_prob_lst:
                    metrics[selector]["aver_c_prob"] = sum(filtered_c_prob_lst) / len(filtered_c_prob_lst)
                else:
                    metrics[selector]["aver_c_prob"] = None
        
        with open(input_dir / "evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
                
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()