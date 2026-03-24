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
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from train_mlp import (
    DEFAULT_SHARED_PREPROCESS,
    LABEL_COLUMN,
    ModelConfig,
    evaluate_holdout_config,
    fit_final_model,
    load_shared_module,
    make_feature_matrices,
    make_model,
    split_raw_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices for the strict outer validation split and local test split."
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
        help="Directory containing training_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory where confusion matrix outputs will be saved.",
    )
    return parser.parse_args()


def load_selected_config(artifact_dir: Path) -> tuple[ModelConfig, dict]:
    summary = json.load((artifact_dir / "training_summary.json").open("r", encoding="utf-8"))
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
    return config, summary


def normalize_confusion(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cm / row_sums


def plot_single_confusion(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
    *,
    normalize: bool,
) -> None:
    display = normalize_confusion(cm) if normalize else cm.astype(float)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    cmap = plt.cm.Blues
    im = ax.imshow(display, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    thresh = display.max() / 2.0 if display.size else 0.0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if normalize:
                text = f"{display[i, j]:.2f}\n({cm[i, j]})"
            else:
                text = f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if display[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, summary = load_selected_config(artifact_dir)
    use_group_relative = bool(
        summary.get("selection_policy", {}).get("use_group_relative_features", True)
        or summary.get("tuning_summary", {}).get("best_validation_result", {}).get("use_group_relative_features", True)
    )

    shared_module = load_shared_module(Path(args.shared_preprocess))
    raw_df = pd.read_csv(args.train_csv)
    shared_module.validate_group_integrity(raw_df)

    split_seed = int(summary["split_summary"]["seed"])
    train_raw, val_raw, test_raw, _ = split_raw_dataframe(raw_df, shared_module=shared_module, seed=split_seed)

    train_clean_df = shared_module.basic_row_clean(train_raw)
    val_clean_df = shared_module.basic_row_clean(val_raw)
    test_clean_df = shared_module.basic_row_clean(test_raw)

    label_encoder = LabelEncoder()
    label_encoder.fit(raw_df[LABEL_COLUMN].to_numpy())
    class_names = label_encoder.classes_.tolist()
    train_labels = label_encoder.transform(train_raw[LABEL_COLUMN].to_numpy())
    val_labels = label_encoder.transform(val_raw[LABEL_COLUMN].to_numpy())
    test_labels = label_encoder.transform(test_raw[LABEL_COLUMN].to_numpy())

    train_matrix, val_matrix, _, _, _ = make_feature_matrices(
        cleaned_train_df=train_clean_df,
        cleaned_test_df=val_clean_df,
        shared_module=shared_module,
        config=config,
        use_group_relative=use_group_relative,
    )
    outer_model = make_model(config)
    outer_model.fit(train_matrix, train_labels)
    val_predictions = outer_model.predict(val_matrix)

    trainval_clean_df = pd.concat([train_clean_df, val_clean_df], ignore_index=True)
    trainval_labels = np.concatenate([train_labels, val_labels])
    final_model, _, _, _, _, _ = fit_final_model(
        trainval_clean_df=trainval_clean_df,
        test_clean_df=test_clean_df,
        trainval_labels=trainval_labels,
        test_labels=test_labels,
        config=config,
        shared_module=shared_module,
        use_group_relative=use_group_relative,
    )
    _, test_matrix, _, _, _ = make_feature_matrices(
        cleaned_train_df=trainval_clean_df,
        cleaned_test_df=test_clean_df,
        shared_module=shared_module,
        config=config,
        use_group_relative=use_group_relative,
    )
    test_predictions = final_model.predict(test_matrix)

    val_cm = confusion_matrix(val_labels, val_predictions, labels=np.arange(len(class_names)))
    test_cm = confusion_matrix(test_labels, test_predictions, labels=np.arange(len(class_names)))

    pd.DataFrame(val_cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix_validation_counts.csv")
    pd.DataFrame(test_cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix_test_counts.csv")
    pd.DataFrame(
        normalize_confusion(val_cm), index=class_names, columns=class_names
    ).to_csv(output_dir / "confusion_matrix_validation_normalized.csv")
    pd.DataFrame(
        normalize_confusion(test_cm), index=class_names, columns=class_names
    ).to_csv(output_dir / "confusion_matrix_test_normalized.csv")

    plot_single_confusion(
        val_cm,
        class_names,
        "Outer Validation Confusion Matrix",
        output_dir / "confusion_matrix_validation.png",
        normalize=True,
    )
    plot_single_confusion(
        test_cm,
        class_names,
        "Local Test Confusion Matrix",
        output_dir / "confusion_matrix_test.png",
        normalize=True,
    )

    summary_payload = {
        "classes": class_names,
        "validation_counts": val_cm.tolist(),
        "test_counts": test_cm.tolist(),
        "validation_normalized": normalize_confusion(val_cm).tolist(),
        "test_normalized": normalize_confusion(test_cm).tolist(),
    }
    with (output_dir / "confusion_matrix_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)

    print(f"Saved {output_dir / 'confusion_matrix_validation.png'}")
    print(f"Saved {output_dir / 'confusion_matrix_test.png'}")
    print(f"Saved {output_dir / 'confusion_matrix_validation_counts.csv'}")
    print(f"Saved {output_dir / 'confusion_matrix_test_counts.csv'}")


if __name__ == "__main__":
    main()
