import argparse
import json
from pathlib import Path
import sys
import traceback
from collections import defaultdict
from statistics import mean
from typing import Dict

import matplotlib.pyplot as plt

from just_tri_it.experiment import Database
from just_tri_it.metrics import all_abstention_metrics
from just_tri_it.utils import print_annotated_hr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The directory where your input experiment results are stored."
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="The directory where your measures and plots are stored."
    )
    return parser.parse_args()


# def evaluate_selector_simple(results: list, p_dict: dict):
#     n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
#     for result in results:
#         print_annotated_hr(f"Task {result['task_id']}")
#         all_p_hash = result["generated_programs"]  # hash
#         chosen_p_hash = result["chosen_programs"]  # hash
#         decision = result["decision"]

#         correct_p_hash = []
#         for p_hash in all_p_hash:
#             if p_dict[result['task_id']][p_hash]['result']:
#                 correct_p_hash.append(p_hash)

#         if len(correct_p_hash) > 0:
#             # GT: no abstention
#             if chosen_p_hash and chosen_p_hash[0] in correct_p_hash:
#                 n1 += 1
#             elif decision == "Abstained":
#                 n3 += 1
#             else:
#                 n2 += 1
#         else:
#             # GT: abstention
#             if decision == "Abstained":
#                 n5 += 1
#             else:
#                 n4 += 1
#     return len(results), [n1, n2, n3, n4, n5], None

# def evaluate_selector_pair(results: list, p_dict: dict):
#     # unfinished for delete hash and reformating
#     overall_n = [0, 0, 0, 0, 0]
#     overall_c_lst = []
#     metrics_dict = {}
#     for result in results:
#         # for each task
#         print_annotated_hr(f"Task {result['task_id']}")
#         overall_c_denominator = 0
#         overall_c_numerator = 0

#         program_list = result["programs"]["forward"]  # hash
#         correct_num = []
#         for p_hash in program_list:
#             if p_dict[result['task_id']][p_hash]['result']:
#                 correct_num.append(p_hash)

#         for prop in result["property"]:
#             # for each property
#             c_denominator = 0
#             c_numerator = 0
#             if prop["name"] not in metrics_dict:
#                 metrics_dict[prop["name"]] = {
#                     "c_prob_lst": [],
#                     "n_lst": [0, 0, 0, 0, 0]
#                 }

#             if prop["decision"] != "Abstained":
#                 c_denominator = len(prop["pairs"])
#                 for pair in prop["pairs"]:
#                     if pair[0] in correct_num:
#                         c_numerator += 1

#             if len(correct_num) > 0:
#                 # GT: no abstention
#                 if c_numerator > 0:
#                     metrics_dict[prop["name"]]["n_lst"][0] += 1
#                     overall_n[0] += 1
#                 elif prop["decision"] == "Abstained":
#                     metrics_dict[prop["name"]]["n_lst"][2] += 1
#                     overall_n[2] += 1
#                 else:
#                     metrics_dict[prop["name"]]["n_lst"][1] += 1
#                     overall_n[1] += 1
#             else:
#                 # GT: abstention
#                 if prop["decision"] == "Abstained":
#                     metrics_dict[prop["name"]]["n_lst"][4] += 1
#                     overall_n[4] += 1
#                 else:
#                     metrics_dict[prop["name"]]["n_lst"][3] += 1
#                     overall_n[3] += 1

#             overall_c_numerator += c_numerator
#             overall_c_denominator += c_denominator
#             if c_denominator > 0:
#                 metrics_dict[prop["name"]]["c_prob_lst"].append(c_numerator / c_denominator)
#             else:
#                 metrics_dict[prop["name"]]["c_prob_lst"].append('Abstained')

#          if overall_c_numerator > 0:
#             overall_c_lst.append(overall_c_numerator / overall_c_denominator)
#         else:
#             overall_c_lst.append('Abstained')
#     return len(results), overall_n, overall_c_lst, metrics_dict

# def evaluate_selector_class(results: list, p_dict: dict):
#     c_prob_lst = []
#     n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
#     for result in results:
#         print_annotated_hr(f"Task {result['task_id']}")
#         program_list = result["generated_programs"]  # hash
#         decision = result["decision"]
#         classes = result["classes"]

#         correct_p_hash = []
#         for p_hash in program_list:
#             if p_dict[result['task_id']][p_hash]['result']:
#                 correct_p_hash.append(p_hash)

#         c_prob_denominator = 0
#         c_prob_numerator = None
#         if decision != "Abstained":
#             for p_class in classes:
#                 p_hash_list = p_class["program_hashes"]
#                 c_prob_denominator += len(p_hash_list) * (len(p_hash_list) - 1)
#                 if c_prob_numerator is None:
#                     for p_hash in p_hash_list:
#                         if p_hash in correct_p_hash:
#                             c_prob_numerator = len(p_hash_list) * (len(p_hash_list) - 1)
#                             break
#             if c_prob_numerator is None:
#                 c_prob_numerator = 0
#             if c_prob_denominator != 0:
#                 c_prob_lst.append(c_prob_numerator / c_prob_denominator)
#             else:
#                 c_prob_lst.append('Abstained')
#         else:
#             c_prob_lst.append('Abstained')

#         print_annotated_hr("conditional probability")
#         print(c_prob_lst, file=sys.stderr)

#         if len(correct_p_hash) > 0:
#             # GT: no abstention
#             if c_prob_numerator and c_prob_numerator > 0:
#                 n1 += 1
#             elif decision == "Abstained":
#                 n3 += 1
#             else:
#                 n2 += 1
#         else:
#             # GT: abstention
#             if decision == "Abstained":
#                 n5 += 1
#             else:
#                 n4 += 1
#     return len(results), [n1, n2, n3, n4, n5], c_prob_lst


def abstention_measures(db):
    """
    from "Know Your Limits: A Survey of Abstention in Large Language Models"
    +--------------+---------+-----------+-----------+
    | GT \\ Answer | Correct | Incorrect | Abstained |
    +--------------+---------+-----------+-----------+
    | Select       | N[1]    | N[2]      | N[3]      |
    +--------------+---------+-----------+-----------+
    | Abstain      |         | N[4]      | N[5]      |
    +--------------+---------+-----------+-----------+
    """
    matrix_per_method = {}

    for obj in db.objects:
        correct_samples = [ p for p, correct, _ in obj["sample_correctness"] if correct ]
        ground_truth_is_select = len(correct_samples) > 0
        for selector_data in obj["selectors"]:
            method = selector_data["id"]
            if method not in matrix_per_method:
                matrix_per_method[method] = [0, 0, 0, 0, 0]

            if ground_truth_is_select:
                if selector_data["outcome"] == "selected":
                    correctly_selected = selector_data["selected"] in correct_samples
                    if correctly_selected:
                        matrix_per_method[method][0] += 1
                    else:
                        matrix_per_method[method][1] += 1
                else:
                    assert selector_data["outcome"] == "abstained"
                    matrix_per_method[method][3] += 1
            else:
                if selector_data["outcome"] == "selected":
                    matrix_per_method[method][4] += 1
                else:
                    assert selector_data["outcome"] == "abstained"
                    matrix_per_method[method][5] += 1

    return { method: all_abstention_metrics(*matrix) for method, matrix in matrix_per_method.items() } 


def plot_sorted_percentages(probs, label, output_file):
    probs = {k: v for k, v in probs.items() if v is not None}
    if len(probs) == 0:
        return
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    methods, values = zip(*sorted_items)
    percentages = [v * 100 for v in values]

    colors = ["grey" if m == "unconditional" else "skyblue" for m in methods]    

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, percentages, color=colors, edgecolor='black')

    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45)
    plt.ylabel(f"{label} (%)")
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def prob_correct(db) -> float:
    probs = []
    for obj in db.objects:
        correct_samples = [ p for p, correct, _ in obj["sample_correctness"] if correct ]
        probs.append(len(correct_samples) / len(obj["sample_correctness"]))
                
    return mean(probs)


def prob_correct_under_agreement(db) -> Dict[str, float]:
    probs_per_agreement_method = defaultdict(list)
    
    for obj in db.objects:
        correct_samples = [ p for p, correct, _ in obj["sample_correctness"] if correct ]
        seen_methods = set()
        
        for selector_data in obj["selectors"]:
            if selector_data["outcome"] == "abstained":
                continue
            method = selector_data["raw_data"]["agreement_raw_data"]["method"]
            if method not in seen_methods:
                seen_methods.add(method)
                num_total_agreements = 0
                num_faithful_agreements = 0
                for (program, witnesses) in selector_data["raw_data"]["agreement"]:
                    num_total_agreements += len(witnesses)
                    if program in correct_samples:
                        num_faithful_agreements += len(witnesses)
                prob = num_faithful_agreements / num_total_agreements
                probs_per_agreement_method[method].append(prob)

    probs = { method: mean(probs) for method, probs in probs_per_agreement_method.items() }
    probs["unconditional"] = prob_correct(db)
                
    return probs


def main():
    args = parse_args()
    data_dir = Path(args.data)
    db = Database.load(data_dir)
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)

    method_to_measures = abstention_measures(db)

    # transposing table:
    measure_to_methods = {
        k2: {k1: method_to_measures[k1][k2] for k1 in method_to_measures}
        for k2 in next(iter(method_to_measures.values()))
    }
    measure_to_methods["prob_correct_under_agreement"] = prob_correct_under_agreement(db)

    for measure in measure_to_methods:
        plot_sorted_percentages(measure_to_methods[measure],
                                measure,
                                report_dir/f"{measure}.png")

    with (report_dir/f"metrics.json").open("w", encoding="utf-8") as f:
        json.dump(measure_to_methods, f, indent=4)


if __name__ == "__main__":
    main()
