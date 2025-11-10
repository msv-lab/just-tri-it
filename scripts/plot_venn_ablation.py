#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "matplotlib_venn",
# ]
# ///
#

import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

TARGET_LABEL = "selected_correct"
METHODS = ["FWD_INV", "FWD_SINV", "ENUM_SINV"]

# Pastel colors for the three sets
PASTEL_COLORS = {
    "A": "#87CEEB",  # skyblue
    "B": "#90EE90",  # lightgreen
    "C": "#F08080",  # lightcoral
}

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python plot_venn.py <results.json> [output.pdf]")
        sys.exit(1)

    infile = Path(sys.argv[1])
    if not infile.exists():
        print(f"Input file not found: {infile}")
        sys.exit(1)

    outfile = Path(sys.argv[2]) if len(sys.argv) == 3 else infile.with_name(infile.stem + "_venn.pdf")

    with infile.open("r") as f:
        data = json.load(f)

    # Collect problem sets per method where outcome == TARGET_LABEL
    sets = {m: set() for m in METHODS}
    for problem, outcomes in data.items():
        if not isinstance(outcomes, dict):
            continue
        for m in METHODS:
            if m in outcomes and outcomes[m] == TARGET_LABEL:
                sets[m].add(problem)

    A = sets["FWD_INV"]
    B = sets["FWD_SINV"]
    C = sets["ENUM_SINV"]

    plt.rcParams.update({
        "text.usetex": True,  # LaTeX rendering
        "font.family": "serif",  # Academic serif font
        "axes.labelsize": 20,  # Axis label size
        "axes.titlesize": 20,  # Title size
        "legend.fontsize": 16,  # Legend size
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    })

    plt.figure(figsize=(4, 4))
    v = venn3(
        subsets=(A, B, C),
        set_labels=("FWD_INV", "FWD_SINV", "ENUM_SINV")
    )

    # Apply pastel colors to each main circle patch
    if v.get_patch_by_id("100"):
        v.get_patch_by_id("100").set_color(PASTEL_COLORS["A"])
        v.get_patch_by_id("100").set_alpha(0.6)
    if v.get_patch_by_id("010"):
        v.get_patch_by_id("010").set_color(PASTEL_COLORS["B"])
        v.get_patch_by_id("010").set_alpha(0.6)
    if v.get_patch_by_id("001"):
        v.get_patch_by_id("001").set_color(PASTEL_COLORS["C"])
        v.get_patch_by_id("001").set_alpha(0.6)

    # For overlaps, blend by lowering alpha
    for subset in ("110", "101", "011", "111"):
        patch = v.get_patch_by_id(subset)
        if patch:
            patch.set_alpha(0.4)

    # Make outlines subtle
    for subset in ("100", "010", "001", "110", "101", "011", "111"):
        patch = v.get_patch_by_id(subset)
        if patch:
            patch.set_edgecolor("gray")
            patch.set_linewidth(1.0)

    # plt.title("Venn diagram of problems 'selected_correct' by method")
    plt.tight_layout()
    plt.savefig(outfile, format="pdf")
    print(f"Saved Venn diagram to {outfile}")

if __name__ == "__main__":
    main()
