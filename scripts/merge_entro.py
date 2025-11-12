#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple JSONL files by "sample_num", combine "task_entropies",
and plot mean task entropy vs. sample size.

Usage:
  python merge_and_plot_entropy.py data1.jsonl data2.jsonl -o merged_output.jsonl --plot entropy_plot.png --converge 20

Notes:
- Later files overwrite duplicate task keys for the same sample_num.
- The plot shows mean task entropy for each sample_num.
- If --converge is given, the script draws a vertical line at that sample size,
  shades the "convergence region" to the right, and shows the mean after convergence.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
from statistics import mean
import matplotlib.pyplot as plt


def merge_jsonl_files(input_files, output_file):
    """
    Merge multiple JSONL files by sample_num and combine task_entropies.

    Args:
        input_files (list[str]): Input JSONL file paths.
        output_file (str): Output JSONL path.
    Returns:
        OrderedDict[int, dict]: Merged records keyed by sample_num (sorted asc).
    """
    merged = defaultdict(lambda: {"sample_num": None, "task_entropies": {}})

    for file_path in input_files:
        p = Path(file_path)
        if not p.exists():
            print(f"Warning: file {p} does not exist. Skipped.")
            continue

        print(f"Processing: {p}")
        with p.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    data = json.loads(line)
                    if "sample_num" not in data:
                        print(f"Warning: {p} line {line_num} missing 'sample_num'. Skipped.")
                        continue
                    if "task_entropies" not in data or not isinstance(data["task_entropies"], dict):
                        print(f"Warning: {p} line {line_num} missing/invalid 'task_entropies'. Skipped.")
                        continue

                    sample_num = data["sample_num"]
                    task_entropies = data["task_entropies"]

                    rec = merged[sample_num]
                    rec["sample_num"] = sample_num
                    # Merge: later files overwrite duplicate keys
                    rec["task_entropies"].update(task_entropies)
                except json.JSONDecodeError as e:
                    print(f"Warning: {p} line {line_num} invalid JSON: {e}. Skipped.")
                except Exception as e:
                    print(f"Warning: error processing {p} line {line_num}: {e}. Skipped.")

    # Sort by sample_num and write out
    ordered = OrderedDict(sorted(merged.items(), key=lambda kv: kv[0]))
    outp = Path(output_file)
    with outp.open('w', encoding='utf-8') as f:
        for sample_num, record in ordered.items():
            # sort task_entropies by key for readability
            record = {
                "sample_num": sample_num,
                "task_entropies": dict(sorted(record["task_entropies"].items()))
            }
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

    print(f"\nMerged {len(ordered)} distinct sample_num record(s).")
    print(f"Output written to: {outp.resolve()}")
    return ordered


def compute_mean_entropies(merged_records):
    """
    Compute mean of task_entropies for each sample_num.

    Args:
        merged_records (OrderedDict[int, dict]): Output of merge_jsonl_files.
    Returns:
        tuple[list[int], list[float]]: (sample_nums, mean_entropies)
    """
    sample_nums = []
    mean_vals = []
    for sample_num, rec in merged_records.items():
        vals = list(rec["task_entropies"].values())
        if len(vals) == 0:
            continue
        sample_nums.append(sample_num)
        mean_vals.append(mean(vals))
    return sample_nums, mean_vals


def plot_entropy(sample_nums, mean_vals, plot_path=None, converge=None):
    """
    Plot mean task entropy vs sample size; optionally show convergence region.

    Args:
        sample_nums (list[int])
        mean_vals (list[float])
        plot_path (str|None): If set, save figure to this path; else show interactively.
        converge (int|None): sample_num at which we consider convergence starts.
    """
    if not sample_nums:
        print("Nothing to plot: no data points available.")
        return

    plt.rcParams.update({
        "text.usetex": True,  # LaTeX rendering
        "font.family": "serif",  # Academic serif font
        "axes.labelsize": 14,  # Axis label size
        "axes.titlesize": 14,  # Title size
        "legend.fontsize": 12,  # Legend size
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    # Line & markers
    ax.plot(sample_nums, mean_vals, marker='o', linewidth=2, label='Mean task entropy')

    # Value labels
    for x, y in zip(sample_nums, mean_vals):
        ax.text(x, y + 0.009 * max(mean_vals), f"{y:.5f}", ha='center', va='bottom')

    # Convergence decoration
    if converge is not None and any(x >= converge for x in sample_nums):
        # vertical line
        ax.axvline(converge, linestyle='--', linewidth=1, label='Convergence start')
        # shaded region to the right
        xmax = max(sample_nums)
        ax.axvspan(converge, xmax, alpha=0.1, label='Convergence region')
        # mean after convergence
        post_vals = [v for x, v in zip(sample_nums, mean_vals) if x >= converge]
        if post_vals:
            post_mean = mean(post_vals)
            ax.axhline(post_mean, linestyle='--', linewidth=1, label=f"Mean after converge: {post_mean:.6f}")

    ax.set_xlabel("Program Sample Size")
    ax.set_ylabel("Entropy")
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, linestyle=':')

    fig.tight_layout()

    if plot_path:
        outp = Path(plot_path).with_suffix(".pdf")
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, format='pdf', bbox_inches='tight')
        print(f"PDF plot saved to: {outp.resolve()}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge JSONL files by 'sample_num', compute mean task entropy, and plot."
    )
    parser.add_argument('input_files', nargs='+', help='Input JSONL file path(s).')
    parser.add_argument('-o', '--output', default='merged_output.jsonl',
                        help='Output merged JSONL (default: merged_output.jsonl).')
    parser.add_argument('--plot', default=None,
                        help='Optional output plot image path (e.g., entropy_plot.png).')
    parser.add_argument('--converge', type=int, default=None,
                        help='Optional sample_num at which convergence begins (e.g., 20).')
    return parser.parse_args()


def main():
    args = parse_args()
    merged = merge_jsonl_files(args.input_files, args.output)
    xs, ys = compute_mean_entropies(merged)
    print("\nSample sizes and mean entropies:")
    for x, y in zip(xs, ys):
        print(f"  {x}: {y:.6f}")
    plot_entropy(xs, ys, plot_path=args.plot, converge=args.converge)


if __name__ == "__main__":
    main()
