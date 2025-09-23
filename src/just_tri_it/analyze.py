import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import Dict

import matplotlib.pyplot as plt

from just_tri_it.experiment import Database
from just_tri_it.metrics import all_abstention_metrics


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


def abstention_measures(db):
    """
    from "Know Your Limits: A Survey of Abstention in Large Language Models"
    +--------------+---------+-----------+-----------+
    | GT \\ Answer | Correct | Incorrect | Abstained |
    +--------------+---------+-----------+-----------+
    | Select       | N[0]    | N[1]      | N[2]      |
    +--------------+---------+-----------+-----------+
    | Abstain      |         | N[3]      | N[4]      |
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
                    matrix_per_method[method][2] += 1
            else:
                if selector_data["outcome"] == "selected":
                    matrix_per_method[method][3] += 1
                else:
                    assert selector_data["outcome"] == "abstained"
                    matrix_per_method[method][4] += 1

    return { method: all_abstention_metrics(*matrix) for method, matrix in matrix_per_method.items() } 


def plot_sorted_percentages(probs, label, output_file):
    probs = {k: v for k, v in probs.items() if v is not None}
    if len(probs) == 0:
        return
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    methods, values = zip(*sorted_items, strict=True)
    percentages = [v * 100 for v in values]

    colors = ["grey" if m == "unconditional" else "skyblue" for m in methods]    

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, percentages, color=colors, edgecolor='black')

    for bar, pct in zip(bars, percentages, strict=True):
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

    with (report_dir/"metrics.json").open("w", encoding="utf-8") as f:
        json.dump(measure_to_methods, f, indent=4)


if __name__ == "__main__":
    main()
