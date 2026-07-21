"""Generate the three JSON reports used by the artifact analysis.

The script deliberately writes only:

* ``incorrect_plurality_entropy_summary.json``
* ``jti_l2_metrics_dedup.json``
* ``paired_bootstrap.json``
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean

import numpy as np

from just_tri_it.experiment import Database
from just_tri_it.metrics import all_abstention_metrics


DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CI_CONFIDENCE = 0.95
DEFAULT_CI_SEED = 0
CORE_PAIRED_BOOTSTRAP_METRICS = (
    "abstention_accuracy",
    "reliable_accuracy",
    "abstention_rate",
    "abstention_f1",
)
JTI_CLUSTER_TARGET_METHODS = (
    "fwd-inv",
    "fwd-sinv",
    "enum-sinv",
)
INCORRECT_PLURALITY_SELECTOR_ID = "Plurality"
INCORRECT_PLURALITY_METHOD = "plurality_0.0"
INCORRECT_PLURALITY_NORMALIZATION = "remaining_group_sizes_renormalized"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="One or more experiment result directories containing data.jsonl.",
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Directory in which the three JSON reports are written.",
    )
    return parser.parse_args()


def unique_preserving_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_data_source(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Data source directory does not exist: {path}")
    return Database.load_ignore(path)


def combine_databases(data_paths):
    """Combine runs while keeping source-specific task rows distinct."""
    data_paths = [Path(path) for path in data_paths]
    add_suffix = len(data_paths) > 1
    objects = []
    content = {}
    sources = []

    for source_index, data_path in enumerate(data_paths, start=1):
        source_db = load_data_source(data_path)
        for obj in source_db.objects:
            copied = deepcopy(obj)
            if add_suffix:
                copied["task_id"] = f"{copied['task_id']}_{source_index}"
            objects.append(copied)

        if add_suffix:
            content.update({
                f"{task_id}_{source_index}": value
                for task_id, value in source_db.content.items()
            })
        else:
            content.update(source_db.content)

        sources.append({
            "index": source_index,
            "path": str(data_path),
            "original_rows": len(source_db.objects),
            "filtered_rows": len(source_db.objects),
            "missing_tasks": 0,
            "missing_task_ids": [],
        })

    metadata = {
        "num_sources": len(data_paths),
        "task_id_suffix_rule": (
            "none" if not add_suffix else "{original_task_id}_{1_based_data_source_index}"
        ),
        "subset_requested_tasks": 0,
        "sources": sources,
        "total_rows": len(objects),
    }
    return Database(objects, content), metadata


def abstention_contribution(obj):
    matrix_per_method = {}
    correct_samples = {
        program
        for program, is_correct, _ in obj["sample_correctness"]
        if is_correct
    }
    ground_truth_is_select = bool(correct_samples)

    for selector_data in obj["selectors"]:
        method = selector_data["id"]
        matrix = matrix_per_method.setdefault(method, [0, 0, 0, 0, 0])

        if ground_truth_is_select:
            if selector_data["outcome"] == "selected":
                if selector_data["selected"] in correct_samples:
                    matrix[0] += 1
                else:
                    matrix[1] += 1
            else:
                assert selector_data["outcome"] == "abstained"
                matrix[2] += 1
        elif selector_data["outcome"] == "selected":
            matrix[3] += 1
        else:
            assert selector_data["outcome"] == "abstained"
            matrix[4] += 1

    return matrix_per_method


def add_vector_counts(target, source):
    for index, count in enumerate(source):
        target[index] += count


def paired_task_matrices(matrix_per_task, target="JUST-TRI-IT", baseline="MajorityVote"):
    return [
        (task_counts[target], task_counts[baseline])
        for task_counts in matrix_per_task
        if target in task_counts and baseline in task_counts
    ]


def aggregate_paired_matrices(paired_matrices, indices=None):
    target_matrix = [0, 0, 0, 0, 0]
    baseline_matrix = [0, 0, 0, 0, 0]
    if indices is None:
        indices = range(len(paired_matrices))

    for index in indices:
        target_counts, baseline_counts = paired_matrices[index]
        add_vector_counts(target_matrix, target_counts)
        add_vector_counts(baseline_matrix, baseline_counts)

    return target_matrix, baseline_matrix


def paired_metric_differences(target_matrix, baseline_matrix):
    target_metrics = all_abstention_metrics(*target_matrix)
    baseline_metrics = all_abstention_metrics(*baseline_matrix)
    differences = {}
    for metric in CORE_PAIRED_BOOTSTRAP_METRICS:
        target_estimate = target_metrics[metric]
        baseline_estimate = baseline_metrics[metric]
        difference = None
        if target_estimate is not None and baseline_estimate is not None:
            difference = target_estimate - baseline_estimate
        differences[metric] = (target_estimate, baseline_estimate, difference)
    return differences


def percentile_ci(values, confidence=DEFAULT_CI_CONFIDENCE):
    if not values:
        return None
    alpha = 1 - confidence
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def paired_bootstrap_comparison(
        matrix_per_task,
        target="JUST-TRI-IT",
        baseline="MajorityVote",
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=DEFAULT_CI_CONFIDENCE,
        seed=DEFAULT_CI_SEED):
    paired_matrices = paired_task_matrices(matrix_per_task, target, baseline)
    n = len(paired_matrices)
    target_matrix, baseline_matrix = aggregate_paired_matrices(paired_matrices)
    point_estimates = paired_metric_differences(target_matrix, baseline_matrix)
    bootstrap_samples = defaultdict(list)

    if n:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            sampled_indices = rng.integers(0, n, size=n)
            sampled_target, sampled_baseline = aggregate_paired_matrices(
                paired_matrices,
                sampled_indices,
            )
            for metric, (_, _, difference) in paired_metric_differences(
                    sampled_target,
                    sampled_baseline,
            ).items():
                if difference is not None:
                    bootstrap_samples[metric].append(difference)

    result = {}
    for metric, (target_estimate, baseline_estimate, difference) in point_estimates.items():
        bootstrap = percentile_ci(bootstrap_samples[metric], confidence)
        result[metric] = {
            "target": target,
            "baseline": baseline,
            "target_estimate": target_estimate,
            "baseline_estimate": baseline_estimate,
            "difference": difference,
            "bootstrap_ci": bootstrap,
            "ci_excludes_zero": (
                False if bootstrap is None else bootstrap[0] > 0 or bootstrap[1] < 0
            ),
            "n": n,
        }
    return result


def paired_bootstrap_comparisons(matrix_per_task, target="JUST-TRI-IT"):
    baselines = sorted({
        method
        for task_counts in matrix_per_task
        for method in task_counts
        if method != target
    })
    result = {}
    for baseline in baselines:
        comparison = paired_bootstrap_comparison(
            matrix_per_task,
            target=target,
            baseline=baseline,
        )
        if comparison[CORE_PAIRED_BOOTSTRAP_METRICS[0]]["n"]:
            result[baseline] = comparison
    return result


def agreement_method(selector_data):
    raw_data = selector_data.get("raw_data")
    if not isinstance(raw_data, dict):
        return None
    agreement_raw_data = raw_data.get("agreement_raw_data")
    if not isinstance(agreement_raw_data, dict):
        return None
    return agreement_raw_data.get("method")


def selector_for_agreement_method(obj, method):
    return next(
        (
            selector_data
            for selector_data in obj.get("selectors", [])
            if agreement_method(selector_data) == method
        ),
        None,
    )


def selector_for_plurality_entropy(obj):
    return next(
        (
            selector_data
            for selector_data in obj.get("selectors", [])
            if selector_data.get("id") == INCORRECT_PLURALITY_SELECTOR_ID
            and agreement_method(selector_data) == INCORRECT_PLURALITY_METHOD
        ),
        None,
    )


def normalized_entropy_from_counts(counts):
    counts = [count for count in counts if count > 0]
    if len(counts) <= 1:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return entropy / math.log(len(counts))


def incorrect_plurality_entropy_for_task(obj):
    total_generated_programs = len(obj.get("sample_correctness", []))
    if total_generated_programs == 0:
        return None

    selector_data = selector_for_plurality_entropy(obj)
    if selector_data is None:
        return None
    raw_data = selector_data.get("raw_data", {})
    agreement_raw_data = raw_data.get("agreement_raw_data", {}) if isinstance(raw_data, dict) else {}
    classes = agreement_raw_data.get("classes")
    if not isinstance(classes, dict):
        return None

    correctness = {}
    for sample in obj["sample_correctness"]:
        if len(sample) >= 2:
            program, is_correct = sample[:2]
            correctness[program] = correctness.get(program, False) or bool(is_correct)

    incorrect_group_sizes = []
    for class_programs in classes.values():
        if not isinstance(class_programs, list):
            continue
        if any(correctness.get(program, False) for program in class_programs):
            continue
        if class_programs:
            incorrect_group_sizes.append(len(class_programs))

    return normalized_entropy_from_counts(incorrect_group_sizes)


def incorrect_plurality_entropy_summary(db):
    values = [
        incorrect_plurality_entropy_for_task(obj)
        for obj in db.objects
    ]
    computed_values = [value for value in values if value is not None]
    return {
        "mean_normalized_entropy": mean(computed_values) if computed_values else None,
        "computed_task_count": len(computed_values),
        "skipped_task_count": len(values) - len(computed_values),
        "selector_id": INCORRECT_PLURALITY_SELECTOR_ID,
        "method": INCORRECT_PLURALITY_METHOD,
        "normalization": INCORRECT_PLURALITY_NORMALIZATION,
    }


def jti_generated_p_info(obj):
    multiplicity = Counter()
    correctness = {}
    for sample in obj.get("sample_correctness", []):
        if len(sample) < 2:
            continue
        program, is_correct = sample[:2]
        multiplicity[program] += 1
        if program not in correctness:
            correctness[program] = bool(is_correct)
    return multiplicity, correctness


def jti_collect_p_q_lists(agreement):
    p_order = []
    p_to_raw_qs = defaultdict(list)
    for item in agreement:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        program, witnesses = item[:2]
        if program not in p_to_raw_qs:
            p_order.append(program)
        if isinstance(witnesses, list):
            p_to_raw_qs[program].extend(witnesses)
        elif witnesses is not None:
            p_to_raw_qs[program].append(witnesses)
    return p_order, p_to_raw_qs


def jti_dedup_agreement_clusters(agreement):
    p_order, p_to_raw_qs = jti_collect_p_q_lists(agreement)
    p_to_unique_qs = {
        program: unique_preserving_order(p_to_raw_qs[program])
        for program in p_order
    }
    p_order = [program for program in p_order if p_to_unique_qs[program]]
    q_to_programs = defaultdict(list)
    for program in p_order:
        for witness in p_to_unique_qs[program]:
            q_to_programs[witness].append(program)

    visited = set()
    clusters = []
    for program in p_order:
        if program in visited:
            continue
        visited.add(program)
        stack = [program]
        component = set()
        while stack:
            current = stack.pop()
            component.add(current)
            for witness in p_to_unique_qs[current]:
                for neighbor in q_to_programs[witness]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        programs = [candidate for candidate in p_order if candidate in component]
        matching_q = unique_preserving_order(
            witness
            for candidate in programs
            for witness in p_to_unique_qs[candidate]
        )
        clusters.append({"programs": programs, "matching_q": matching_q})
    return clusters


def jti_missing_dedup_cluster_metric_row(obj, method):
    return {
        "task_id": obj["task_id"],
        "method": method,
        "status": "missing_selector",
        "correct_p_ratio": None,
        "correct_q_ratio": None,
        "incorrect_p_ratio": None,
        "incorrect_q_ratio": None,
        "all_generated_p": None,
        "all_generated_q": None,
        "num_clusters": 0,
        "denominator_source": None,
    }


def jti_dedup_cluster_metric_for_selector(obj, method, selector_data):
    _multiplicity, p_correctness = jti_generated_p_info(obj)
    raw_data = selector_data.get("raw_data", {})
    agreement = raw_data.get("agreement", []) if isinstance(raw_data, dict) else []
    clusters = jti_dedup_agreement_clusters(agreement)
    all_generated_p = sum(len(cluster["programs"]) for cluster in clusters)
    all_generated_q = len({
        witness
        for cluster in clusters
        for witness in cluster["matching_q"]
    })

    correct_p_count = correct_q_count = 0
    incorrect_p_count = incorrect_q_count = 0
    for cluster in clusters:
        programs = cluster["programs"]
        if not programs:
            continue
        p_count = len(programs)
        q_count = len(cluster["matching_q"])
        if p_correctness.get(programs[0], False):
            correct_p_count += p_count
            correct_q_count += q_count
        else:
            incorrect_p_count += p_count
            incorrect_q_count += q_count

    status = "ok" if all_generated_p and all_generated_q else "zero_denominator"
    return {
        "task_id": obj["task_id"],
        "method": method,
        "status": status,
        "correct_p_ratio": correct_p_count / all_generated_p if all_generated_p else None,
        "correct_q_ratio": correct_q_count / all_generated_q if all_generated_q else None,
        "incorrect_p_ratio": incorrect_p_count / all_generated_p if all_generated_p else None,
        "incorrect_q_ratio": incorrect_q_count / all_generated_q if all_generated_q else None,
        "all_generated_p": all_generated_p,
        "all_generated_q": all_generated_q,
        "num_clusters": len(clusters),
        "denominator_source": "matched_unique",
    }


def jti_cluster_metrics_dedup(db):
    rows = []
    for obj in db.objects:
        for method in JTI_CLUSTER_TARGET_METHODS:
            selector_data = selector_for_agreement_method(obj, method)
            if selector_data is None:
                rows.append(jti_missing_dedup_cluster_metric_row(obj, method))
            else:
                rows.append(jti_dedup_cluster_metric_for_selector(obj, method, selector_data))
    return rows


def jti_l2_score(pairs):
    denominator = sum(p * p + q * q for p, q in pairs)
    if denominator == 0:
        return None
    return 2 * sum((p - q) ** 2 for p, q in pairs) / denominator


def jti_l2_metric_row(method, rows):
    correct_pairs = []
    incorrect_pairs = []
    n_skipped = 0
    fields = (
        "correct_p_ratio",
        "correct_q_ratio",
        "incorrect_p_ratio",
        "incorrect_q_ratio",
    )
    for row in rows:
        if row["status"] != "ok" or any(row[field] is None for field in fields):
            n_skipped += 1
            continue
        correct_pairs.append((row["correct_p_ratio"], row["correct_q_ratio"]))
        incorrect_pairs.append((row["incorrect_p_ratio"], row["incorrect_q_ratio"]))

    correct_score = jti_l2_score(correct_pairs)
    incorrect_score = jti_l2_score(incorrect_pairs)
    gap = None if correct_score is None or incorrect_score is None else incorrect_score - correct_score
    return {
        "method": method,
        "correct_score": correct_score,
        "incorrect_score": incorrect_score,
        "gap": gap,
        "holds": None if gap is None else correct_score < incorrect_score,
        "n_used": len(correct_pairs),
        "n_skipped": n_skipped,
    }


def jti_l2_metrics_from_cluster_metrics(cluster_rows):
    rows_by_method = defaultdict(list)
    for row in cluster_rows:
        rows_by_method[row["method"]].append(row)
    metrics = [
        jti_l2_metric_row(method, rows_by_method[method])
        for method in JTI_CLUSTER_TARGET_METHODS
    ]
    metrics.append(jti_l2_metric_row("combined", cluster_rows))
    return metrics


def _agreement_witnesses(selector_data, program):
    raw_data = selector_data.get("raw_data")
    agreement = raw_data.get("agreement") if isinstance(raw_data, dict) else None
    if not isinstance(agreement, list):
        return set()
    witnesses = set()
    for entry in agreement:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        agreed_program, program_witnesses = entry[:2]
        if agreed_program != program or program_witnesses is None:
            continue
        if isinstance(program_witnesses, list):
            witnesses.update(program_witnesses)
        else:
            witnesses.add(program_witnesses)
    return witnesses


def _cross_mode_witness_agreement(selection_by_method, selector_data_by_method):
    methods = ("ENUM_SINV", "FWD_SINV", "FWD_INV")
    selected_methods = [method for method in methods if selection_by_method[method] is not None]
    for index, first_method in enumerate(selected_methods):
        first_program = selection_by_method[first_method]
        for second_method in selected_methods[index + 1:]:
            second_program = selection_by_method[second_method]
            if first_program == second_program:
                continue
            for method in (first_method, second_method):
                first_witnesses = _agreement_witnesses(
                    selector_data_by_method[method],
                    first_program,
                )
                second_witnesses = _agreement_witnesses(
                    selector_data_by_method[method],
                    second_program,
                )
                if first_witnesses & second_witnesses:
                    return True
    return False


def add_just_tri_it(db):
    jti_methods = ("FWD_INV", "FWD_SINV", "ENUM_SINV")
    for obj in db.objects:
        selection_by_method = {method: None for method in (*jti_methods, "MajorityVote")}
        selector_data_by_method = {}
        just_tri_it_did_not_crash = 0

        for selector_data in obj["selectors"]:
            method = selector_data["id"]
            if method in jti_methods:
                just_tri_it_did_not_crash += 1
            if method not in selection_by_method:
                continue
            selector_data_by_method[method] = selector_data
            if selector_data["outcome"] == "selected":
                selection_by_method[method] = selector_data["selected"]
            else:
                assert selector_data["outcome"] == "abstained"

        if just_tri_it_did_not_crash < len(jti_methods):
            continue

        selections = [
            selection_by_method[method]
            for method in ("ENUM_SINV", "FWD_SINV", "FWD_INV")
            if selection_by_method[method] is not None
        ]
        unique_selections = set(selections)
        if len(unique_selections) == 1 or (
                len(unique_selections) == 2
                and _cross_mode_witness_agreement(
                    selection_by_method,
                    selector_data_by_method,
                )):
            selection = selections[0]
        else:
            selection = None

        just_tri_it = {"id": "JUST-TRI-IT", "witnesses": []}
        if selection is None:
            just_tri_it["outcome"] = "abstained"
        else:
            just_tri_it.update({"outcome": "selected", "selected": selection})
        obj["selectors"].append(just_tri_it)


def write_json(report_dir, filename, data):
    with (report_dir / filename).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main():
    args = parse_args()
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)
    db, _metadata = combine_databases(args.data)
    add_just_tri_it(db)

    write_json(
        report_dir,
        "incorrect_plurality_entropy_summary.json",
        incorrect_plurality_entropy_summary(db),
    )
    deduplicated_clusters = jti_cluster_metrics_dedup(db)
    write_json(
        report_dir,
        "jti_l2_metrics_dedup.json",
        jti_l2_metrics_from_cluster_metrics(deduplicated_clusters),
    )
    matrix_per_task = [abstention_contribution(obj) for obj in db.objects]
    write_json(
        report_dir,
        "paired_bootstrap.json",
        paired_bootstrap_comparisons(matrix_per_task),
    )


if __name__ == "__main__":
    main()
