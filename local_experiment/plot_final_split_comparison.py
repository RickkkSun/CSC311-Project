#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/csc311-mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final train/validation/test accuracy and gaps from training_summary.json."
    )
    parser.add_argument(
        "--summary",
        default="artifacts/training_summary.json",
        help="Path to training_summary.json.",
    )
    parser.add_argument(
        "--output",
        default="analysis/final_split_comparison.png",
        help="Path to the output PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = json.load(summary_path.open("r", encoding="utf-8"))
    best = summary["tuning_summary"]["best_validation_result"]
    final_train = summary["final_train_metrics"]
    final_test = summary["final_test_metrics"]

    accuracy_labels = ["Train+Val\nFinal Train", "Outer Val\nTune Score", "Local Test\nFinal Eval"]
    accuracy_values = [
        float(final_train["train_accuracy"]),
        float(best["validation_accuracy"]),
        float(final_test["test_accuracy"]),
    ]
    gap_labels = ["Train - Val", "Train - Test"]
    gap_values = [
        float(best["accuracy_gap"]),
        float(final_test["train_test_accuracy_gap"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    acc_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    axes[0].bar(accuracy_labels, accuracy_values, color=acc_colors, alpha=0.9)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Final Split Accuracy Comparison")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(accuracy_values):
        axes[0].text(idx, value + 0.015, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    gap_colors = ["#d62728", "#9467bd"]
    axes[1].bar(gap_labels, gap_values, color=gap_colors, alpha=0.9)
    axes[1].set_ylim(0.0, max(0.12, max(gap_values) + 0.02))
    axes[1].set_ylabel("Gap")
    axes[1].set_title("Generalization Gaps")
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(gap_values):
        axes[1].text(idx, value + 0.004, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Strict Holdout MLP Summary", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Validation score is measured on the fixed outer validation split during tuning; train/test are from the final train+val refit.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
