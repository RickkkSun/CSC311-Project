#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/csc311-mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from train_mlp import (
    DEFAULT_SHARED_PREPROCESS,
    GROUP_COLUMN,
    LABEL_COLUMN,
    ModelConfig,
    load_shared_module,
    make_feature_matrices,
    make_model,
    split_raw_dataframe,
)


warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict-holdout overfitting diagnostics and learning-curve plots."
    )
    parser.add_argument(
        "--train-csv",
        default="/Users/rickkk0417/Downloads/training_data_202601 (1).csv",
        help="Path to the raw training CSV.",
    )
    parser.add_argument(
        "--shared-preprocess",
        default=str(DEFAULT_SHARED_PREPROCESS),
        help="Path to the shared first-pass preprocessing script.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory containing training_summary.json and mlp_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory where analysis CSV/JSON/plots will be saved.",
    )
    parser.add_argument(
        "--fractions",
        default="0.2,0.35,0.5,0.65,0.8,1.0",
        help="Comma-separated training-group fractions for the learning curve.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of random train-group subsamples per fraction.",
    )
    return parser.parse_args()


def load_selected_config(artifact_dir: Path) -> tuple[ModelConfig, dict, dict]:
    summary = json.load((artifact_dir / "training_summary.json").open("r", encoding="utf-8"))
    metadata = json.load((artifact_dir / "mlp_metadata.json").open("r", encoding="utf-8"))
    config_data = summary["selected_config"]
    config = ModelConfig(
        hidden_layer_sizes=tuple(config_data["hidden_layer_sizes"]),
        alpha=float(config_data["alpha"]),
        learning_rate_init=float(config_data["learning_rate_init"]),
        batch_size=int(config_data["batch_size"]),
        max_iter=int(config_data["max_iter"]),
        max_text_features=int(config_data["max_text_features"]),
        sublinear_tf=bool(config_data["sublinear_tf"]),
        random_state=int(config_data["random_state"]),
    )
    return config, summary, metadata


def score_split(model, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predictions = model.predict(features)
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro")),
    )


def summarize_records(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    grouped = df.groupby("fraction", as_index=False)
    return grouped.agg(
        train_accuracy_mean=("train_accuracy", "mean"),
        train_accuracy_std=("train_accuracy", "std"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        train_macro_f1_mean=("train_macro_f1", "mean"),
        train_macro_f1_std=("train_macro_f1", "std"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", "std"),
        train_validation_gap_mean=("train_validation_gap", "mean"),
        train_validation_gap_std=("train_validation_gap", "std"),
        train_test_gap_mean=("train_test_gap", "mean"),
        train_test_gap_std=("train_test_gap", "std"),
        n_groups_mean=("n_groups", "mean"),
        n_groups_std=("n_groups", "std"),
        n_rows_mean=("n_rows", "mean"),
        n_rows_std=("n_rows", "std"),
    )


def plot_learning_curve(summary_df: pd.DataFrame, output_path: Path) -> None:
    x = summary_df["fraction"].astype(float).to_numpy()
    plt.figure(figsize=(9, 5.5))
    line_specs = [
        ("train_accuracy_mean", "train_accuracy_std", "Train", "#1f77b4"),
        ("validation_accuracy_mean", "validation_accuracy_std", "Validation", "#ff7f0e"),
        ("test_accuracy_mean", "test_accuracy_std", "Test", "#2ca02c"),
    ]
    for mean_col, std_col, label, color in line_specs:
        y = summary_df[mean_col].astype(float).to_numpy()
        y_std = summary_df[std_col].fillna(0.0).astype(float).to_numpy()
        plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.15, color=color)

    plt.xlabel("Training Group Fraction")
    plt.ylabel("Accuracy")
    plt.title("Strict Holdout Learning Curve")
    plt.ylim(0.75, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_gap_curve(summary_df: pd.DataFrame, output_path: Path) -> None:
    x = summary_df["fraction"].astype(float).to_numpy()
    plt.figure(figsize=(9, 5.2))
    specs = [
        ("train_validation_gap_mean", "train_validation_gap_std", "Train - Validation", "#d62728"),
        ("train_test_gap_mean", "train_test_gap_std", "Train - Test", "#9467bd"),
    ]
    for mean_col, std_col, label, color in specs:
        y = summary_df[mean_col].astype(float).to_numpy()
        y_std = summary_df[std_col].fillna(0.0).astype(float).to_numpy()
        plt.plot(x, y, marker="o", linewidth=2, label=label, color=color)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.15, color=color)

    plt.xlabel("Training Group Fraction")
    plt.ylabel("Gap")
    plt.title("Generalization Gap vs Training Size")
    plt.ylim(0.0, max(0.14, float(summary_df["train_test_gap_mean"].max()) + 0.03))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = [float(item.strip()) for item in args.fractions.split(",") if item.strip()]
    if not fractions:
        raise ValueError("At least one training fraction is required.")

    config, training_summary, metadata = load_selected_config(artifact_dir)
    use_group_relative = bool(metadata.get("use_group_relative_features", True))

    shared_module = load_shared_module(Path(args.shared_preprocess))
    raw_df = pd.read_csv(args.train_csv)
    shared_module.validate_group_integrity(raw_df)

    split_seed = int(training_summary["split_summary"]["seed"])
    train_raw, val_raw, test_raw, split_summary = split_raw_dataframe(
        raw_df,
        shared_module=shared_module,
        seed=split_seed,
    )

    train_clean_df = shared_module.basic_row_clean(train_raw)
    val_clean_df = shared_module.basic_row_clean(val_raw)
    test_clean_df = shared_module.basic_row_clean(test_raw)

    label_encoder = LabelEncoder()
    label_encoder.fit(raw_df[LABEL_COLUMN].to_numpy())
    train_labels = label_encoder.transform(train_raw[LABEL_COLUMN].to_numpy())
    val_labels = label_encoder.transform(val_raw[LABEL_COLUMN].to_numpy())
    test_labels = label_encoder.transform(test_raw[LABEL_COLUMN].to_numpy())

    unique_train_groups = np.array(sorted(train_raw[GROUP_COLUMN].unique()))
    records: list[dict] = []

    for fraction in fractions:
        n_groups = max(1, int(round(len(unique_train_groups) * fraction)))
        for repeat in range(args.repeats):
            rng = np.random.default_rng(10_000 + repeat)
            selected_groups = np.sort(rng.choice(unique_train_groups, size=n_groups, replace=False))
            mask = train_raw[GROUP_COLUMN].isin(selected_groups).to_numpy()

            subset_train_clean_df = train_clean_df.loc[mask].reset_index(drop=True)
            subset_train_labels = train_labels[mask]

            train_matrix, val_matrix, _, _, _ = make_feature_matrices(
                cleaned_train_df=subset_train_clean_df,
                cleaned_test_df=val_clean_df,
                shared_module=shared_module,
                config=config,
                use_group_relative=use_group_relative,
            )
            _, test_matrix, _, _, _ = make_feature_matrices(
                cleaned_train_df=subset_train_clean_df,
                cleaned_test_df=test_clean_df,
                shared_module=shared_module,
                config=config,
                use_group_relative=use_group_relative,
            )

            model = make_model(config)
            model.fit(train_matrix, subset_train_labels)

            train_accuracy, train_macro_f1 = score_split(model, train_matrix, subset_train_labels)
            validation_accuracy, validation_macro_f1 = score_split(model, val_matrix, val_labels)
            test_accuracy, test_macro_f1 = score_split(model, test_matrix, test_labels)

            records.append(
                {
                    "fraction": fraction,
                    "repeat": repeat,
                    "n_groups": int(n_groups),
                    "n_rows": int(len(subset_train_clean_df)),
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": validation_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_macro_f1": train_macro_f1,
                    "validation_macro_f1": validation_macro_f1,
                    "test_macro_f1": test_macro_f1,
                    "train_validation_gap": train_accuracy - validation_accuracy,
                    "train_test_gap": train_accuracy - test_accuracy,
                }
            )

    records_df = pd.DataFrame(records).sort_values(["fraction", "repeat"]).reset_index(drop=True)
    summary_df = summarize_records(records)

    records_df.to_csv(output_dir / "learning_curve_records.csv", index=False)
    summary_df.to_csv(output_dir / "learning_curve_summary.csv", index=False)
    plot_learning_curve(summary_df, output_dir / "learning_curve_accuracy.png")
    plot_gap_curve(summary_df, output_dir / "learning_curve_gaps.png")

    diagnostic_summary = {
        "selected_config": asdict(config),
        "use_group_relative_features": use_group_relative,
        "split_summary": split_summary,
        "fractions": fractions,
        "repeats": args.repeats,
        "best_validation_accuracy": float(
            training_summary["tuning_summary"]["best_validation_result"]["validation_accuracy"]
        ),
        "best_validation_gap": float(training_summary["tuning_summary"]["best_validation_result"]["accuracy_gap"]),
        "final_local_test_accuracy": float(training_summary["final_test_metrics"]["test_accuracy"]),
        "final_train_test_gap": float(training_summary["final_test_metrics"]["train_test_accuracy_gap"]),
        "inner_cv_mean_accuracy": float(
            training_summary["tuning_summary"]["best_validation_result"]["inner_cv_mean_accuracy"]
        ),
        "inner_cv_accuracy_std": float(
            np.std(
                [
                    item["validation_accuracy"]
                    for item in training_summary["tuning_summary"]["best_validation_result"]["inner_cv_fold_metrics"]
                ],
                ddof=0,
            )
        ),
        "learning_curve_tail": summary_df.tail(2).to_dict(orient="records"),
    }
    with (output_dir / "overfit_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostic_summary, handle, indent=2, ensure_ascii=True)

    print(f"Saved {output_dir / 'learning_curve_records.csv'}")
    print(f"Saved {output_dir / 'learning_curve_summary.csv'}")
    print(f"Saved {output_dir / 'learning_curve_accuracy.png'}")
    print(f"Saved {output_dir / 'learning_curve_gaps.png'}")
    print(f"Saved {output_dir / 'overfit_diagnostics.json'}")


if __name__ == "__main__":
    main()
