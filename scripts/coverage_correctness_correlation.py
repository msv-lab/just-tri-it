#!/usr/bin/env python3
"""Analyze correctness correlations with plurality-input quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SELECTOR_ID = "Plurality"
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_SEED = 0
SUMMARY_SCHEMA_VERSION = 3
METRIC_COVERAGE = "coverage"
METRIC_DISTINGUISHING = "distinguishing"
METRIC_BOTH = "both"

COVERAGE_PER_TASK_FIELDS = [
    "source_label",
    "task_id",
    "selector_id",
    "correct_sample_count",
    "sample_count",
    "probability_correct",
    "coverage_status",
    "coverage_input_source",
    "mean_line_coverage",
    "included_in_analysis",
    "exclusion_reason",
]

DISTINGUISHING_PER_TASK_FIELDS = [
    "source_label",
    "task_id",
    "selector_id",
    "correct_sample_count",
    "sample_count",
    "probability_correct",
    "input_quality_status",
    "input_source",
    "num_inputs",
    "oracle_equivalence_class_count",
    "distinguishing_input_rate",
    "included_in_analysis",
    "exclusion_reason",
]


class InputSource:
    def __init__(self, label: str, data: Path, result_path: Path):
        self.label = label
        self.data = data
        self.result_path = result_path


class AnalysisMetric:
    def __init__(
        self,
        name: str,
        input_path_name: str,
        value_field: str,
        filename: str,
        output_stem: str,
        x_axis_label: str,
        plot_title: str,
        per_task_fields: list[str],
    ):
        self.name = name
        self.input_path_name = input_path_name
        self.value_field = value_field
        self.filename = filename
        self.output_stem = output_stem
        self.x_axis_label = x_axis_label
        self.plot_title = plot_title
        self.per_task_fields = per_task_fields


COVERAGE_ANALYSIS = AnalysisMetric(
    name=METRIC_COVERAGE,
    input_path_name="coverage",
    value_field="mean_line_coverage",
    filename="coverage_per_task.csv",
    output_stem="coverage_correctness",
    x_axis_label="Mean line coverage",
    plot_title="Coverage and Sample Correctness by Task",
    per_task_fields=COVERAGE_PER_TASK_FIELDS,
)

DISTINGUISHING_ANALYSIS = AnalysisMetric(
    name=METRIC_DISTINGUISHING,
    input_path_name="input_quality",
    value_field="distinguishing_input_rate",
    filename="input_quality_per_task.csv",
    output_stem="distinguishing_correctness",
    x_axis_label="Distinguishing input rate",
    plot_title="Distinguishing Input Rate and Sample Correctness by Task",
    per_task_fields=DISTINGUISHING_PER_TASK_FIELDS,
)


def analysis_metric(name: str) -> AnalysisMetric:
    return {
        METRIC_COVERAGE: COVERAGE_ANALYSIS,
        METRIC_DISTINGUISHING: DISTINGUISHING_ANALYSIS,
    }[name]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate experiment sample-program correctness probability with "
            "plurality input coverage or distinguishing-input rate."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        help=(
            "Single experiment result directory containing data.jsonl, or a "
            "data.jsonl path. Must be paired with the CSV required by --metric."
        ),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        help="Single coverage_per_task.csv path. Must be paired with --data.",
    )
    parser.add_argument(
        "--input-quality",
        type=Path,
        help="Single input_quality_per_task.csv path. Must be paired with --data.",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        help=(
            "Coverage output directory containing coverage_per_task.csv and "
            "input_quality_per_task.csv. Must be paired with --data."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("DATA", "RESULT_PATH"),
        type=Path,
        help=(
            "Experiment data directory/file and either a matching metric CSV or "
            "a coverage output directory. A directory runs both analyses unless "
            "--metric selects one. Repeat to combine multiple data sets."
        ),
    )
    parser.add_argument(
        "--source-label",
        action="append",
        help=(
            "Optional label for one input source. Repeat in --input order; for "
            "legacy single-source flags, provide at most one label."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for the CSV, JSON summary, and PDF scatter plot.",
    )
    parser.add_argument(
        "--selector-id",
        default=DEFAULT_SELECTOR_ID,
        help=f"Coverage selector to analyze. Default: {DEFAULT_SELECTOR_ID}.",
    )
    parser.add_argument(
        "--metric",
        choices=[METRIC_COVERAGE, METRIC_DISTINGUISHING, METRIC_BOTH],
        help=(
            "Task-level test metric to correlate with correctness. "
            "The default is coverage for CSV input and both metrics for coverage directories."
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help=f"Two-sided permutation-test iterations. Default: {DEFAULT_PERMUTATIONS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for permutation tests. Default: {DEFAULT_SEED}.",
    )
    args = parser.parse_args(argv)
    if args.permutations <= 0:
        parser.error("--permutations must be positive.")
    has_legacy_input = any(
        value is not None
        for value in (args.data, args.coverage, args.input_quality, args.coverage_dir)
    )
    if args.input and has_legacy_input:
        parser.error("Use either --input or legacy single-source flags, not both.")
    if not args.input:
        legacy_result_paths = [
            path for path in (args.coverage, args.input_quality, args.coverage_dir) if path
        ]
        if args.data is None or len(legacy_result_paths) != 1:
            parser.error(
                "Provide --data with exactly one of --coverage, --input-quality, or "
                "--coverage-dir; alternatively provide one or more --input pairs."
            )
    source_count = len(args.input) if args.input else 1
    if args.source_label and len(args.source_label) != source_count:
        parser.error("Provide exactly one --source-label for each input source.")
    return args


def resolve_data_path(data: Path) -> Path:
    """Accept either an experiment result directory or its data.jsonl file."""
    path = data / "data.jsonl" if data.is_dir() else data
    if not path.is_file():
        raise FileNotFoundError(f"Experiment data.jsonl was not found: {path}")
    return path


def default_source_label(data_path: Path) -> str:
    """Derive a compact label from conventional results/<dataset>/<model>/data paths."""
    experiment_dir = data_path.parent.parent
    dataset_dir = experiment_dir.parent
    if experiment_dir.name and dataset_dir.name:
        return f"{dataset_dir.name}/{experiment_dir.name}"
    return str(data_path.parent)


def input_sources_from_args(args: argparse.Namespace) -> list[InputSource]:
    input_pairs = args.input or [
        [args.data, next(path for path in (args.coverage, args.input_quality, args.coverage_dir) if path)]
    ]
    sources: list[InputSource] = []
    for index, (data, result_path) in enumerate(input_pairs):
        data_path = resolve_data_path(data)
        label = (
            args.source_label[index]
            if args.source_label
            else default_source_label(data_path)
        )
        sources.append(InputSource(label=label, data=data_path, result_path=result_path))

    labels = [source.label for source in sources]
    if len(labels) != len(set(labels)):
        raise ValueError("Input source labels must be unique.")
    return sources


def selected_metrics(args: argparse.Namespace, sources: list[InputSource]) -> list[AnalysisMetric]:
    if args.metric == METRIC_BOTH:
        if not all(source.result_path.is_dir() for source in sources):
            raise ValueError(
                "--metric both requires every input result path to be a coverage directory."
            )
        return [COVERAGE_ANALYSIS, DISTINGUISHING_ANALYSIS]
    if args.metric is not None:
        return [analysis_metric(args.metric)]

    if all(source.result_path.is_dir() for source in sources):
        return [COVERAGE_ANALYSIS, DISTINGUISHING_ANALYSIS]
    if any(source.result_path.is_dir() for source in sources):
        raise ValueError(
            "Cannot infer both metrics from a mixture of coverage directories and CSV files. "
            "Use --metric coverage or --metric distinguishing."
        )
    if not args.input and args.input_quality is not None:
        return [DISTINGUISHING_ANALYSIS]
    return [COVERAGE_ANALYSIS]


def metric_path_for(source: InputSource, metric: AnalysisMetric) -> Path:
    if source.result_path.is_dir():
        return source.result_path / metric.filename
    return source.result_path


def validate_metric_paths(
    sources: list[InputSource], metrics: list[AnalysisMetric]
) -> None:
    missing = [
        f"{source.label}: {metric_path_for(source, metric)}"
        for source in sources
        for metric in metrics
        if not metric_path_for(source, metric).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required metric CSV files:\n" + "\n".join(missing)
        )


def read_data_objects(path: Path) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            task_id = obj.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"Missing task_id on {path}:{line_number}")
            if task_id in objects:
                raise ValueError(f"Duplicate task_id in data.jsonl: {task_id}")
            objects[task_id] = obj
    return objects


def correctness_probability(obj: dict[str, Any]) -> tuple[int, int, float] | None:
    """Match analyze.py's True/total definition while rejecting unknown statuses."""
    samples = obj.get("sample_correctness")
    if not isinstance(samples, list) or not samples:
        return None

    statuses: list[bool] = []
    for sample in samples:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            return None
        status = sample[1]
        if type(status) is not bool:
            return None
        statuses.append(status)

    correct_count = sum(statuses)
    sample_count = len(statuses)
    return correct_count, sample_count, correct_count / sample_count


def read_metric_rows(path: Path, metric: AnalysisMetric) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric CSV was not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"task_id", "selector_id", "status", metric.value_field}
        if metric.name == METRIC_DISTINGUISHING:
            required.add("oracle_equivalence_class_count")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Metric CSV is missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def metric_rows_by_task(
    rows: list[dict[str, str]], selector_id: str, metric: AnalysisMetric
) -> tuple[dict[str, dict[str, str]], set[str]]:
    """Return selector rows and task IDs that only have another selector."""
    selected: dict[str, dict[str, str]] = {}
    all_selectors_by_task: dict[str, set[str]] = {}
    for row in rows:
        task_id = (row.get("task_id") or "").strip()
        if not task_id:
            continue
        row_selector_id = row.get("selector_id") or ""
        all_selectors_by_task.setdefault(task_id, set()).add(row_selector_id)
        if row_selector_id != selector_id:
            continue
        if task_id in selected:
            row_description = "coverage" if metric.name == METRIC_COVERAGE else "input-quality"
            raise ValueError(
                f"Multiple {row_description} rows found for task_id="
                f"{task_id!r} and selector_id={selector_id!r}."
            )
        selected[task_id] = row

    selector_mismatch_task_ids = {
        task_id
        for task_id, selector_ids in all_selectors_by_task.items()
        if task_id not in selected and selector_ids
    }
    return selected, selector_mismatch_task_ids


def finite_float(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def missing_metric_reason(metric: AnalysisMetric) -> str:
    return "missing_coverage" if metric.name == METRIC_COVERAGE else "missing_input_quality"


def non_ok_metric_reason(metric: AnalysisMetric) -> str:
    return (
        "coverage_status_not_ok"
        if metric.name == METRIC_COVERAGE
        else "input_quality_status_not_ok"
    )


def oracle_equivalence_class_count(row: dict[str, str]) -> int | None:
    try:
        count = int(row["oracle_equivalence_class_count"])
    except (KeyError, TypeError, ValueError):
        return None
    return count if count >= 1 else None


def make_per_task_rows(
    data_objects: dict[str, dict[str, Any]],
    metric_rows: dict[str, dict[str, str]],
    selector_mismatch_task_ids: set[str],
    selector_id: str,
    source_label: str,
    metric: AnalysisMetric,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task_id in sorted(set(data_objects) | set(metric_rows)):
        obj = data_objects.get(task_id)
        metric_row = metric_rows.get(task_id)
        probability = correctness_probability(obj) if obj is not None else None
        correct_count = probability[0] if probability is not None else ""
        sample_count = probability[1] if probability is not None else ""
        probability_correct = probability[2] if probability is not None else ""
        metric_status = metric_row.get("status", "") if metric_row is not None else ""
        input_source = metric_row.get("input_source", "") if metric_row is not None else ""
        metric_value = (
            finite_float(metric_row.get(metric.value_field)) if metric_row is not None else None
        )
        exclusion_reason = ""

        if obj is None:
            exclusion_reason = "missing_experiment_data"
        elif probability is None:
            exclusion_reason = "correctness_not_evaluated"
        elif metric_row is None:
            exclusion_reason = (
                "selector_mismatch"
                if task_id in selector_mismatch_task_ids
                else missing_metric_reason(metric)
            )
        elif metric_status != "ok":
            exclusion_reason = non_ok_metric_reason(metric)
        elif metric_value is None:
            exclusion_reason = f"invalid_{metric.value_field}"
        elif metric.name == METRIC_DISTINGUISHING:
            class_count = oracle_equivalence_class_count(metric_row)
            if class_count is None:
                exclusion_reason = "invalid_oracle_equivalence_class_count"
            elif class_count == 1:
                exclusion_reason = "single_oracle_equivalence_class"

        common = {
            "source_label": source_label,
            "task_id": task_id,
            "selector_id": selector_id,
            "correct_sample_count": correct_count,
            "sample_count": sample_count,
            "probability_correct": probability_correct,
            "included_in_analysis": not exclusion_reason,
            "exclusion_reason": exclusion_reason,
        }
        if metric.name == METRIC_COVERAGE:
            result.append(
                {
                    **common,
                    "coverage_status": metric_status,
                    "coverage_input_source": input_source,
                    "mean_line_coverage": "" if metric_value is None else metric_value,
                }
            )
        else:
            result.append(
                {
                    **common,
                    "input_quality_status": metric_status,
                    "input_source": input_source,
                    "num_inputs": metric_row.get("num_inputs", "") if metric_row else "",
                    "oracle_equivalence_class_count": (
                        metric_row.get("oracle_equivalence_class_count", "")
                        if metric_row
                        else ""
                    ),
                    "distinguishing_input_rate": "" if metric_value is None else metric_value,
                }
            )
    return result


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    correlation = float(np.corrcoef(x, y)[0, 1])
    return correlation if math.isfinite(correlation) else None


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including ties, without SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson_correlation(average_ranks(x), average_ranks(y))


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_correlation: float | None,
    correlation_fn,
    permutations: int,
    seed: int,
) -> float | None:
    if observed_correlation is None:
        return None
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    observed_abs = abs(observed_correlation)
    for _ in range(permutations):
        permuted = correlation_fn(x, rng.permutation(y))
        if permuted is not None and abs(permuted) >= observed_abs:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (permutations + 1)


def correlation_statistics(
    per_task_rows: list[dict[str, Any]],
    metric: AnalysisMetric,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    included = [row for row in per_task_rows if row["included_in_analysis"]]
    x = np.asarray([float(row[metric.value_field]) for row in included])
    y = np.asarray([float(row["probability_correct"]) for row in included])
    pearson = pearson_correlation(x, y)
    spearman = spearman_correlation(x, y)
    return {
        "eligible_task_count": len(included),
        "pearson_r": pearson,
        "pearson_permutation_p_value": permutation_p_value(
            x, y, pearson, pearson_correlation, permutations, seed
        ),
        "spearman_rho": spearman,
        "spearman_permutation_p_value": permutation_p_value(
            x, y, spearman, spearman_correlation, permutations, seed
        ),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    temporary_path.replace(path)


def format_statistic(value: float | None) -> str:
    return "not computable" if value is None else f"{value:.4f}"


def write_scatter_plot(
    path: Path,
    statistics: dict[str, Any],
    rows: list[dict[str, Any]],
    metric: AnalysisMetric,
) -> None:
    included = [row for row in rows if row["included_in_analysis"]]
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    if included:
        source_rows: dict[str, list[dict[str, Any]]] = {}
        for row in included:
            source_rows.setdefault(row["source_label"], []).append(row)
        colors = plt.get_cmap("tab10")
        for index, (source_label, source_data) in enumerate(source_rows.items()):
            axis.scatter(
                [float(row[metric.value_field]) for row in source_data],
                [float(row["probability_correct"]) for row in source_data],
                color=colors(index % colors.N),
                edgecolor="#124d58",
                linewidth=0.5,
                alpha=0.85,
                label=source_label,
            )
        if len(source_rows) > 1:
            axis.legend(title="Source", loc="lower left", frameon=True)
    else:
        axis.text(
            0.5,
            0.5,
            "No eligible tasks for correlation analysis",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_xlabel(metric.x_axis_label)
    axis.set_ylabel("Sample-program correctness probability")
    axis.set_title(metric.plot_title)
    axis.grid(True, color="#d8dde0", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    annotation = "\n".join(
        [
            f"n = {statistics['eligible_task_count']}",
            "Pearson r = "
            f"{format_statistic(statistics['pearson_r'])}, p = "
            f"{format_statistic(statistics['pearson_permutation_p_value'])}",
            "Spearman rho = "
            f"{format_statistic(statistics['spearman_rho'])}, p = "
            f"{format_statistic(statistics['spearman_permutation_p_value'])}",
        ]
    )
    axis.text(
        0.02,
        0.98,
        annotation,
        ha="left",
        va="top",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#b8c2c7", "pad": 4},
    )
    figure.tight_layout()
    figure.savefig(path, format="pdf")
    plt.close(figure)


def source_summary(
    source: InputSource,
    metric_path: Path,
    data_objects: dict[str, dict[str, Any]],
    metric_csv_rows: list[dict[str, str]],
    selected_metric_rows: dict[str, dict[str, str]],
    per_task_rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    metric: AnalysisMetric,
) -> dict[str, Any]:
    exclusion_counts = Counter(
        row["exclusion_reason"]
        for row in per_task_rows
        if row["exclusion_reason"]
    )
    summary = {
        "source_label": source.label,
        "data_path": str(source.data),
        "data_task_count": len(data_objects),
        "per_task_row_count": len(per_task_rows),
        "excluded_task_count": len(per_task_rows) - statistics["eligible_task_count"],
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        **statistics,
    }
    summary[f"{metric.input_path_name}_path"] = str(metric_path)
    summary[f"{metric.input_path_name}_row_count"] = len(metric_csv_rows)
    summary[f"{metric.input_path_name}_selector_task_count"] = len(selected_metric_rows)
    return summary


def run_analysis(
    args: argparse.Namespace, sources: list[InputSource], metric: AnalysisMetric
) -> dict[str, Any]:
    all_per_task_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for source in sources:
        data_objects = read_data_objects(source.data)
        metric_path = metric_path_for(source, metric)
        metric_csv_rows = read_metric_rows(metric_path, metric)
        selected_metric_rows, selector_mismatch_task_ids = metric_rows_by_task(
            metric_csv_rows, args.selector_id, metric
        )
        per_task_rows = make_per_task_rows(
            data_objects,
            selected_metric_rows,
            selector_mismatch_task_ids,
            args.selector_id,
            source.label,
            metric,
        )
        statistics = correlation_statistics(
            per_task_rows, metric, args.permutations, args.seed
        )
        summaries.append(
            source_summary(
                source,
                metric_path,
                data_objects,
                metric_csv_rows,
                selected_metric_rows,
                per_task_rows,
                statistics,
                metric,
            )
        )
        all_per_task_rows.extend(per_task_rows)

    statistics = correlation_statistics(
        all_per_task_rows, metric, args.permutations, args.seed
    )
    exclusion_counts = Counter(
        row["exclusion_reason"]
        for row in all_per_task_rows
        if row["exclusion_reason"]
    )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "metric": metric.name,
        "metric_field": metric.value_field,
        "selector_id": args.selector_id,
        "permutations": args.permutations,
        "seed": args.seed,
        "source_count": len(sources),
        "data_task_count": sum(item["data_task_count"] for item in summaries),
        f"{metric.input_path_name}_row_count": sum(
            item[f"{metric.input_path_name}_row_count"] for item in summaries
        ),
        f"{metric.input_path_name}_selector_task_count": sum(
            item[f"{metric.input_path_name}_selector_task_count"] for item in summaries
        ),
        "per_task_row_count": len(all_per_task_rows),
        "excluded_task_count": len(all_per_task_rows) - statistics["eligible_task_count"],
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "sources": summaries,
        "per_source_statistics": [
            {
                "source_label": item["source_label"],
                "eligible_task_count": item["eligible_task_count"],
                "pearson_r": item["pearson_r"],
                "pearson_permutation_p_value": item["pearson_permutation_p_value"],
                "spearman_rho": item["spearman_rho"],
                "spearman_permutation_p_value": item[
                    "spearman_permutation_p_value"
                ],
            }
            for item in summaries
        ],
        **statistics,
    }
    if len(summaries) == 1:
        summary["data_path"] = summaries[0]["data_path"]
        summary[f"{metric.input_path_name}_path"] = summaries[0][
            f"{metric.input_path_name}_path"
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / f"{metric.output_stem}_per_task.csv",
        metric.per_task_fields,
        all_per_task_rows,
    )
    write_json(args.output_dir / f"{metric.output_stem}_summary.json", summary)
    write_scatter_plot(
        args.output_dir / f"{metric.output_stem}_scatter.pdf",
        statistics,
        all_per_task_rows,
        metric,
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    sources = input_sources_from_args(args)
    metrics = selected_metrics(args, sources)
    validate_metric_paths(sources, metrics)
    summaries = {
        metric.name: run_analysis(args, sources, metric)
        for metric in metrics
    }
    if len(summaries) == 1:
        return next(iter(summaries.values()))
    return {
        "metric": METRIC_BOTH,
        "source_count": len(sources),
        "metrics": summaries,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    if summary.get("metric") == METRIC_BOTH:
        for metric_name, metric_summary in summary["metrics"].items():
            print(
                f"{metric_name}-correctness correlation: "
                f"eligible_tasks={metric_summary['eligible_task_count']} "
                f"pearson_r={format_statistic(metric_summary['pearson_r'])} "
                f"spearman_rho={format_statistic(metric_summary['spearman_rho'])}"
            )
        return
    print(
        f"{summary['metric']}-correctness correlation: "
        f"eligible_tasks={summary['eligible_task_count']} "
        f"pearson_r={format_statistic(summary['pearson_r'])} "
        f"spearman_rho={format_statistic(summary['spearman_rho'])}"
    )


if __name__ == "__main__":
    main()
