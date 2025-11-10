import argparse
import copy
import csv
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
    decisions_per_method_per_task = {}
    no_answer_prop_p = 0

    for obj in db.objects:
        decisions_per_method_per_task[obj["task_id"]] = {}
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
        if len(correct_samples) == 0:
            no_answer_prop_p += 1
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
                        decisions_per_method_per_task[obj["task_id"]][method] = "selected_correct"
                    else:
                        matrix_per_method[method][1] += 1
                        decisions_per_method_per_task[obj["task_id"]][method] = "selected_incorrect"
                else:
                    assert selector_data["outcome"] == "abstained"
                    matrix_per_method[method][2] += 1
                    decisions_per_method_per_task[obj["task_id"]][method] = "incorrectly_abstained"
            else:
                if selector_data["outcome"] == "selected":
                    matrix_per_method[method][3] += 1
                    decisions_per_method_per_task[obj["task_id"]][method] = "selected_instead_of_abstaining"
                else:
                    assert selector_data["outcome"] == "abstained"
                    matrix_per_method[method][4] += 1
                    decisions_per_method_per_task[obj["task_id"]][method] = "correctly_abstained"

    no_answer_prop_q = len(db.objects)
    prop = no_answer_prop_p / no_answer_prop_q
    return {method: all_abstention_metrics(*matrix) for method, matrix in
            matrix_per_method.items()}, decisions_per_method_per_task, prop


def tasks_where_jti_outperforms_majority(decisions):
    result = []
    for task, per_method in decisions.items():
        if "JUST-TRI-IT" in per_method and\
           "MajorityVote" in per_method and\
           per_method["JUST-TRI-IT"] == "selected_correct" and\
           per_method["MajorityVote"] != "selected_correct":
            result.append(task)
    return result


SELECTOR_PAPER_NAMES = {
    "JUST-TRI-IT": "JUST-TRI-IT",
    "JTI-MAJ": "JTI-MAJ",    
    "MajorityVote": "Majority0.5",
    "Plurality": "Plurality",
    "Postcondition": "Postcondition",
    "CodeT_Assert": "CodeT"
}


def plot_sorted_percentages_compress(probs, label, output_file, change_order, abs_prop=None):
    plt.rcParams.update({
        "text.usetex": True,  # LaTeX rendering
        "font.family": "serif",  # Academic serif font
        "axes.labelsize": 14,  # Axis label size
        "axes.titlesize": 14,  # Title size
        "legend.fontsize": 12,  # Legend size
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })
    probs = {k: v for k, v in probs.items() if v is not None}

    paper_skip = ["ENUM_SINV", "FWD_SINV", "FWD_INV", "OffByOne", "CodeT_IO", "MaxTest_Assert", "MaxTest_IO", "Syntactic"]

    if label in ["reliable_accuracy", "abstention_accuracy", "abstention_rate", "abstention_f1"]:
        plt.figure(figsize=(5, 4))
    else:
        plt.figure(figsize=(5.5, 5))
    
    if label in ["reliable_accuracy", "abstention_accuracy", "abstention_rate", "abstention_f1"]:
        probs = {SELECTOR_PAPER_NAMES[k]: v for k, v in probs.items() if k not in paper_skip}
        match label:
            case "reliable_accuracy":
                label = "Reliable Accuracy"
            case "abstention_accuracy":
                label = "Overall Accuracy"
            case "abstention_rate":
                label = "Abstention Rate"
            case "abstention_f1":
                label = "Abstention F1"
                
    if len(probs) == 0:
        return
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    methods, values = zip(*sorted_items, strict=True)
    percentages = [v * 100 for v in values]

    colors = ["lightsalmon" if m == "JUST-TRI-IT" else ("lightpink" if m == "JTI-MAJ" else "skyblue") for m in methods]

    if change_order:
        methods = methods[::-1]
        percentages = percentages[::-1]
        colors = colors[::-1]
       
    bars = plt.barh(methods, percentages, color=colors, edgecolor=colors, height=0.6)  # ★ 改为水平条形图

    for bar, pct in zip(bars, percentages, strict=True):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va='center', fontsize=12)  # ★ 调整文字位置为右侧

    if abs_prop is not None:
        abs_percentage = abs_prop * 100
        ax = plt.gca()        
        ax.axvline(x=abs_percentage, color='blue', linestyle='--', linewidth=2,
                   label=f'Ground Truth ({abs_percentage:.1f}\%)')  # ★ 改为垂直线
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel(f"{label} (\%)")  # ★ 改成 X 轴标签
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.close()


def plot_sorted_selected_correct_counts(probs_json, label, output_file, change_order):
    plt.rcParams.update({
        "text.usetex": True,   # LaTeX rendering
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    TARGET_LABEL = "selected_correct"

    # Aggregate counts per method from the JSON-like structure: {problem: {method: outcome_str}}
    method_counts = {}
    for problem, outcomes in probs_json.items():
        if not isinstance(outcomes, dict):
            continue
        for m, outcome in outcomes.items():
            if outcome == TARGET_LABEL:
                method_counts[m] = method_counts.get(m, 0) + 1

    # Early exit if nothing to plot
    if not method_counts:
        return

    paper_skip = ["ENUM_SINV", "FWD_SINV", "FWD_INV", "OffByOne", "CodeT_IO", "MaxTest_Assert", "MaxTest_IO", "Syntactic"]
    method_counts = {SELECTOR_PAPER_NAMES.get(k, k): v for k, v in method_counts.items() if k not in paper_skip}

    plt.figure(figsize=(5, 4))

    # Sort by count desc
    sorted_items = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
    methods, counts = zip(*sorted_items)

    # Colors consistent with your pastel palette rule (swap names as needed)
    colors = ["lightsalmon" if m == "JUST-TRI-IT" else ("lightpink" if m == "JTI-MAJ" else "skyblue") for m in methods]

    # Optionally reverse order (top-to-bottom)
    if change_order:
        methods = methods[::-1]
        counts = counts[::-1]
        colors = colors[::-1]

    # Plot horizontal bars
    bars = plt.barh(methods, counts, color=colors, edgecolor=colors, height=0.6)

    # Annotate counts to the right of each bar
    for bar, val in zip(bars, counts, strict=True):
        plt.text(bar.get_width() + max(1, 0.01 * max(counts)),  # small offset
                 bar.get_y() + bar.get_height() / 2,
                 f"{val}", va='center', fontsize=12)

    # Clean spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel(f"{label} (count)")
    # Nice x-limits: from 0 to a bit over max
    xmax = max(counts) if counts else 1
    plt.xlim(0, xmax * 1.1)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.close()   


AGREEMENT_PAPER_NAMES = {
    "enum-sinv": "ENUM-SINV",
    "fwd-inv": "FWD-INV",    
    "fwd-sinv": "FWD-SINV",
    "plurality_0.0": "Another solution",
    "postcondition": "Postcondition",
    "off-by-one": "OffByOne",
    "syntactic": "Syntactic",
    "unconditional": "Unconditional",
    "test_assert": "Test"
}


def plot_sorted_percentages(probs, label, output_file, abs_prop=None):
    plt.rcParams.update({
        "text.usetex": True,  # LaTeX rendering
        "font.family": "serif",  # Academic serif font
        "axes.labelsize": 14,  # Axis label size
        "axes.titlesize": 14,  # Title size
        "legend.fontsize": 12,  # Legend size
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    probs = {k: v for k, v in probs.items() if v is not None}

    if label == "prob_correct_under_agreement":
        probs = {AGREEMENT_PAPER_NAMES[k]: v for k, v in probs.items() if k not in ["plurality_0.5", "test_IO", "JUST-TRI-IT", "JTI-MAJ"]}
        label = "Probability"
    
    if len(probs) == 0:
        return
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    methods, values = zip(*sorted_items, strict=True)
    percentages = [v * 100 for v in values]

    colors = ["grey" if m == "Unconditional" else ("lightsalmon" if m in ["FWD-INV", "FWD-SINV", "ENUM-SINV"] else "skyblue") for m in methods]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(methods, percentages, color=colors, edgecolor=colors, width=0.7)

    for bar, pct in zip(bars, percentages, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha='center', va='bottom', fontsize=12)

    if abs_prop is not None:
        abs_percentage = abs_prop * 100
        plt.axhline(y=abs_percentage, color='blue', linestyle='--', linewidth=2,
                    label=f'Ground Truth ({abs_percentage:.1f}%)')
        plt.legend()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(rotation=45)
    plt.ylabel(f"{label} (\%)")
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.close()


def prob_correct(db) -> float:
    probs = []
    for obj in db.objects:
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
        probs.append(len(correct_samples) / len(obj["sample_correctness"]))

    return mean(probs)


def prob_correct_by_task(db) -> Dict[str, float]:
    probs = {}
    between_zero_and_half = 0
    for obj in db.objects:
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]
        p = (len(correct_samples) / len(obj["sample_correctness"]))
        if p > 0 and p < 0.5:
            between_zero_and_half += 1
        probs[obj["task_id"]] = p
    # print(f"difficult but feasible: {between_zero_and_half}/{len(probs)}")
    return probs


def plot_distribution_with_separate_zero(data, output_file):
    data = np.array(data)

    zeros = data[data == 0]
    low_probs = data[(data > 0) & (data < 0.5)]
    high_probs = data[data >= 0.5]

    # Define bins
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1

    plt.rcParams.update({
        "text.usetex": True,  # LaTeX rendering
        "font.family": "serif",  # Academic serif font
        "axes.labelsize": 12,  # Axis label size
        "axes.titlesize": 13,  # Title size
        "legend.fontsize": 10,  # Legend size
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    fig, ax = plt.subplots(figsize=(4.5, 3))

    # Plot histogram
    ax.hist(high_probs, bins=bins, color='seagreen', edgecolor='seagreen', label=r'$P \geq 0.5$')
    ax.hist(low_probs, bins=bins, color='skyblue', edgecolor='skyblue', label=r'$0 < P < 0.5$')
    ax.hist(zeros, bins=[-0.001, 0.001], color='darkred', edgecolor='darkred', alpha=0.8, label=r'$P = 0$')
    # Labels
    ax.set_xlabel(r"Probability", labelpad=6)
    ax.set_ylabel(r"Problem count", labelpad=6)
    # ax.set_title(r"Distribution of Probabilities of Correctness")

    # Academic look: clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)    

    # Ticks outward
    ax.tick_params(axis="both", direction="out", length=5)

    # Legend outside for clarity
    ax.legend(frameon=False, loc="upper right")

    # Save high-quality vector format
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()


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

        # num_inputs = get_num_inputs(obj)
        # if num_inputs < 10:
        #     print(f"{obj['task_id']}: has only {num_inputs} inputs")

        for selector_data in obj["selectors"]:
            if selector_data["outcome"] == "abstained":
                continue
            if "raw_data" not in selector_data:
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
        correct_samples = [p for p, correct, _ in obj["sample_correctness"] if correct]

        all_length = len(obj["sample_correctness"])
        len_dict = {}

        for selector_data in obj["selectors"]:
            if selector_data["outcome"] == "abstained":
                continue
            if selector_data["id"] == "Plurality":
                class_data_dict = selector_data["raw_data"]["agreement_raw_data"]["classes"]
                for key, value in class_data_dict.items():
                    for item in value:
                        len_dict[item] = len(value)
            if selector_data["selected"] in correct_samples:
                if selector_data["id"] not in interval_data:
                    interval_data[selector_data["id"]] = [0] * 10
                    box_data[selector_data["id"]] = []
                if len(len_dict):
                    proportion = len_dict[selector_data["selected"]] / all_length
                    interval_data[selector_data["id"]][to_interval(proportion)] += 1
                    box_data[selector_data["id"]].append(proportion)
    return interval_data, box_data


def plot_distribution(data, output_file):
    bins = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]
    x = np.arange(len(bins))

    custom_order = [
        'Plurality', 'MajorityVote',
        'MaxTest_Assert', 'MaxTest_IO', 'CodeT_Assert', 'CodeT_IO',
        'Syntactic', 'OffByOne', 'Postcondition',
        'FWD_INV', 'FWD_SINV', 'FOR_INV', 'FOR_FIB'
    ]

    sorted_methods = [method for method in custom_order if method in data]

    n_methods = len(sorted_methods)
    total_width = 0.8
    width = total_width / n_methods

    fig, ax = plt.subplots(figsize=(8, 5))

    color_groups = {
        'group1': ['Plurality', 'MajorityVote'],
        'group2': ['MaxTest_Assert', 'MaxTest_IO', 'CodeT_Assert', 'CodeT_IO'],
        'group3': ['Syntactic', 'OffByOne', 'Postcondition'],
        'group4': ['FWD_INV', 'FWD_SINV', 'FOR_INV', 'FOR_FIB']
    }

    group_colors = {
        'group1': ['#1f77b4', '#4c94c4'],
        'group2': ['#2ca02c', '#4cbf4c', '#6fdc6f', '#92f292'],
        'group3': ['#ff7f0e', '#ff9e3e', '#ffbd6e'],
        'group4': ['#9467bd', '#a885c9', '#bca3d5', '#d0c1e1']
    }

    color_map = {}
    for group, methods in color_groups.items():
        colors = group_colors[group]
        for i, method in enumerate(methods):
            if method in data:
                color_map[method] = colors[i % len(colors)]

    for i, method in enumerate(sorted_methods):
        values = data[method]
        offset = (i - n_methods / 2) * width + width / 2
        ax.bar(x + offset, values, width,
               label=method,
               color=color_map[method],
               edgecolor='white',
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=0)

    ax.set_xlabel("Proportion of Class Size for Correct Programs")
    ax.set_ylabel("Number of Correct Decisions")

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for method in sorted_methods:
        if method in labels:
            idx = labels.index(method)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    ax.legend(ordered_handles, ordered_labels,
              loc='upper left')

    plt.tight_layout(pad=2.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_box(data, output_file):
    custom_order = [
        'Plurality', 'MajorityVote',
        'MaxTest_Assert', 'MaxTest_IO', 'CodeT_Assert', 'CodeT_IO',
        'Syntactic', 'OffByOne', 'Postcondition',
        'FWD_INV', 'FWD_SINV', 'FOR_INV', 'FOR_FIB'
    ]

    sorted_keys = [method for method in custom_order if method in data]
    sorted_values = [data[k] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(sorted_values, labels=sorted_keys, patch_artist=True)

    ax.set_xlabel("Selector")
    ax.set_ylabel("Proportion of Class Size for Correct Programs")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def add_just_tri_it(db):

    for obj in db.objects:
        selection_by_method = {}
        selection_by_method["FWD_INV"] = None
        selection_by_method["FWD_SINV"] = None
        selection_by_method["ENUM_SINV"] = None
        selection_by_method["MajorityVote"] = None        
        
        for selector_data in obj["selectors"]:
            method = selector_data["id"]
            if method in selection_by_method:
                if selector_data["outcome"] == "selected":
                    selection_by_method[method] = selector_data["selected"]
                else:
                    assert selector_data["outcome"] == "abstained"

        just_tri_it_selection = None
        if selection_by_method["ENUM_SINV"] is not None:
            just_tri_it_selection = selection_by_method["ENUM_SINV"]
        elif selection_by_method["FWD_SINV"] is not None:
            just_tri_it_selection = selection_by_method["FWD_SINV"]
        elif selection_by_method["FWD_INV"] is not None:
            just_tri_it_selection = selection_by_method["FWD_INV"]
        just_tri_it_data = {
            "id": "JUST-TRI-IT",
            "witnesses": []
        }
        if just_tri_it_selection is None:
            just_tri_it_data["outcome"] = "abstained"
        else:
            just_tri_it_data["outcome"] = "selected"
            just_tri_it_data["selected"] = just_tri_it_selection

        obj["selectors"].append(just_tri_it_data)            

        jti_maj_selection = None
        if just_tri_it_selection is not None:
            jti_maj_selection = just_tri_it_selection
        elif selection_by_method["MajorityVote"] is not None:
            jti_maj_selection = selection_by_method["MajorityVote"]

        jti_maj_data = {
            "id": "JTI-MAJ",
            "witnesses": []
        }
        if jti_maj_selection is None:
            jti_maj_data["outcome"] = "abstained"
        else:
            jti_maj_data["outcome"] = "selected"
            jti_maj_data["selected"] = jti_maj_selection
        
        # obj["selectors"].append(jti_maj_data)

def main():
    args = parse_args()
    data_dir = Path(args.data)
    db = Database.load_ignore(data_dir)
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)

    corr_dist = prob_correct_by_task(db)
    with (report_dir / "prob_correct_per_task.csv").open(mode="w") as file:
        writer = csv.writer(file)
        for task_id, prob in corr_dist.items():
            writer.writerow([task_id, prob])

    plot_distribution_with_separate_zero(list(corr_dist.values()), report_dir / "prob_correct_distribution.pdf")

    add_just_tri_it(db)

    method_to_measures, decisions, abs_prop = abstention_measures(db)

    plot_sorted_selected_correct_counts(decisions, "Correct solutions", report_dir / "correct_solutions.pdf", False)    

    if len(method_to_measures) > 0:
        task_list = tasks_where_jti_outperforms_majority(decisions)
        
        with (report_dir / "tasks_where_jti_outperforms_majority.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(task_list))
        
        # transposing table:
        measure_to_methods = {
            k2: {k1: method_to_measures[k1][k2] for k1 in method_to_measures}
            for k2 in next(iter(method_to_measures.values()))
        }
        measure_to_methods["prob_correct_under_agreement"] = prob_correct_under_agreement(db)

        for measure in measure_to_methods:
            if measure == "prob_correct_under_agreement":
                plot_sorted_percentages(measure_to_methods[measure],
                                        measure,
                                        report_dir / f"{measure}.pdf")
            else:
                if measure != "abstention_rate":
                    plot_sorted_percentages_compress(measure_to_methods[measure],
                                                     measure,
                                                     report_dir / f"{measure}.pdf",
                                                     True)
                else:
                    plot_sorted_percentages_compress(measure_to_methods[measure],
                                                     measure,
                                                     report_dir / f"{measure}.pdf",
                                                     False,
                                                     abs_prop)

        interval_data, box_data = cal_class_size_info(db)
        if interval_data and box_data:
            plot_distribution(interval_data, report_dir / "prob_selected_distribution.png")
            plot_box(box_data, report_dir / "box.png")

        with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(measure_to_methods, f, indent=4)

        with (report_dir / "decisions.json").open("w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=4)


if __name__ == "__main__":
    main()
