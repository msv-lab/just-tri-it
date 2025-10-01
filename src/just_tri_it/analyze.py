import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import Dict

import math
import matplotlib.pyplot as plt
import numpy as np

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
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
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

    return {method: all_abstention_metrics(*matrix) for method, matrix in matrix_per_method.items()}


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
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
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
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
        probs.append(len(correct_samples) / len(obj["sample_correctness"]))

    return mean(probs)


def get_num_inputs(obj):
    for selector_data in obj["selectors"]:
        method = selector_data["raw_data"]["agreement_raw_data"]["method"]
        if method == 'plurality_0.0':
            return len(selector_data["raw_data"]["agreement_raw_data"]["inputs"])
    return -1


def prob_correct_under_agreement(db) -> Dict[str, float]:
    agreements_per_method = defaultdict(list)

    for obj in db.objects:
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
        seen_methods = set()

        num_inputs = get_num_inputs(obj)
        if num_inputs < 10:
            print(f"{obj['task_id']}: has only {num_inputs} inputs")

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
                agreements_per_method[method].append((num_faithful_agreements, num_total_agreements))

    probs = {}

    for method, results in agreements_per_method.items():
        total_faithful = sum(f for f, t in results)
        total_pairs = sum(t for f, t in results)

        if total_pairs > 0:
            probs[method] = total_faithful / total_pairs
        else:
            probs[method] = None
                
    probs["unconditional"] = prob_correct(db)

    return probs


def to_interval(x):
    if not (0 <= x <= 1):
        raise ValueError("should be between 0 and 1")

    if x == 1.0:
        return 9

    left = math.floor(x * 10) / 10

    return int(left / 0.1)


def cal_class_size_info(db):
    interval_data = {}
    box_data = {}

    for obj in db.objects:
        # for each task
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
        all_length = len(obj["sample_correctness"])
        seen_methods = set()
        proportion = 0

        for selector_data in obj["selectors"]:
            if selector_data["outcome"] == "abstained":
                continue
            if selector_data["id"] == "Plurality":
                class_data_dict = selector_data["raw_data"]["agreement_raw_data"]["classes"]
                size_of_class = 0
                for key, value in class_data_dict.items():
                    for correct_sample in correct_samples:
                        if correct_sample in value:
                            size_of_class = len(value)
                            break
                proportion = size_of_class / all_length
            method = selector_data["raw_data"]["agreement_raw_data"]["method"]
            if method not in seen_methods:
                seen_methods.add(method)
                for (program, witnesses) in selector_data["raw_data"]["agreement"]:
                    if program in correct_samples:
                        if method not in interval_data:
                            interval_data[method] = [0] * 10
                            box_data[method] = []
                        interval_data[method][to_interval(proportion)] += 1
                        box_data[method].append(proportion)
                        break

    return interval_data, box_data


def plot_distribution(data, output_file):
    bins = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]
    x = np.arange(len(bins))

    n_methods = len(data)
    total_width = 0.8
    width = total_width / n_methods

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method, values) in enumerate(data.items()):
        offset = (i - n_methods / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45)

    ax.set_xlabel("Proportion of Correct Programs")
    ax.set_ylabel("Number of Correct Decision")
    ax.set_title("Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_box(data, output_file):
    sorted_keys = sorted(data.keys())
    sorted_values = [data[k] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(sorted_values, labels=sorted_keys, patch_artist=True)

    ax.set_xlabel("Class Size of Correct Decision")
    ax.set_ylabel("Values")
    ax.set_title("Boxplot")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    db = Database.load_ignore(data_dir)
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
                                report_dir / f"{measure}.png")

    interval_data, box_data = cal_class_size_info(db)
    if interval_data and box_data:
        plot_distribution(interval_data, report_dir / "distribution.png")
        plot_box(box_data, report_dir / "box.png")

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(measure_to_methods, f, indent=4)


if __name__ == "__main__":
    main()
