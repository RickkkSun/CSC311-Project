#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from mlp_preprocess import (
    TEXT_TOKEN_PATTERN,
    apply_shared_metadata,
    build_model_text,
    build_structured_features,
    fit_group_relative_metadata,
    structured_feature_columns,
)
from pandas.errors import PerformanceWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore", category=PerformanceWarning)

GROUP_COLUMN = "unique_id"
LABEL_COLUMN = "Painting"
DEFAULT_SHARED_PREPROCESS = Path(__file__).resolve().parent / "shared_preprocess_v2.py"


@dataclass(frozen=True)
class ModelConfig:
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    learning_rate_init: float
    batch_size: int
    max_iter: int
    max_text_features: int
    sublinear_tf: bool
    random_state: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a strict holdout MLP classifier for local reproducible experiments."
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
        "--output-dir",
        default="artifacts",
        help="Directory where exported weights and metadata will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the grouped train/val/test split.",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.005,
        help="Validation accuracy tolerance used before preferring lower-overfit models.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run a holdout hyperparameter search on train/validation before the final train+val fit.",
    )
    parser.add_argument(
        "--disable-group-relative",
        action="store_true",
        help="Disable unique_id-relative structured features for ablation while keeping the same strict split/tuning flow.",
    )
    return parser.parse_args()


def load_shared_module(path: Path) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("shared_preprocess_v2", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Could not load shared preprocess from {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def split_raw_dataframe(raw_df: pd.DataFrame, shared_module: ModuleType, seed: int):
    unique_ids = raw_df[GROUP_COLUMN].unique()
    train_ids, val_ids, test_ids = shared_module.make_group_split(unique_ids, seed=seed)

    train_raw = raw_df[raw_df[GROUP_COLUMN].isin(train_ids)].reset_index(drop=True)
    val_raw = raw_df[raw_df[GROUP_COLUMN].isin(val_ids)].reset_index(drop=True)
    test_raw = raw_df[raw_df[GROUP_COLUMN].isin(test_ids)].reset_index(drop=True)

    split_summary = {
        "seed": seed,
        "train_rows": int(len(train_raw)),
        "val_rows": int(len(val_raw)),
        "test_rows": int(len(test_raw)),
        "train_groups": int(train_raw[GROUP_COLUMN].nunique()),
        "val_groups": int(val_raw[GROUP_COLUMN].nunique()),
        "test_groups": int(test_raw[GROUP_COLUMN].nunique()),
    }
    return train_raw, val_raw, test_raw, split_summary


def build_text_features(
    train_text: pd.Series,
    test_text: pd.Series,
    *,
    max_text_features: int,
    sublinear_tf: bool,
    vocabulary: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    kwargs = {
        "lowercase": False,
        "preprocessor": None,
        "tokenizer": None,
        "token_pattern": TEXT_TOKEN_PATTERN,
        "ngram_range": (1, 2),
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": sublinear_tf,
        "dtype": np.float32,
    }

    if vocabulary is None:
        vectorizer = TfidfVectorizer(min_df=2, max_features=max_text_features, **kwargs)
        train_matrix = vectorizer.fit_transform(train_text).toarray()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocabulary, **kwargs)
        train_matrix = vectorizer.fit_transform(train_text).toarray()

    if len(test_text) == 0:
        test_matrix = np.empty((0, train_matrix.shape[1]), dtype=np.float32)
    else:
        test_matrix = vectorizer.transform(test_text).toarray()

    return train_matrix.astype(np.float32), test_matrix.astype(np.float32), vectorizer


def make_feature_matrices(
    cleaned_train_df: pd.DataFrame,
    cleaned_test_df: pd.DataFrame,
    *,
    shared_module: ModuleType,
    config: ModelConfig,
    shared_metadata: dict | None = None,
    text_vocabulary: dict[str, int] | None = None,
    use_group_relative: bool = True,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer, dict, list[str]]:
    if shared_metadata is None:
        shared_metadata = shared_module.fit_shared_metadata(cleaned_train_df)
        if use_group_relative:
            shared_metadata = fit_group_relative_metadata(cleaned_train_df, shared_metadata)

    train_shared = apply_shared_metadata(cleaned_train_df, shared_metadata)
    test_shared = apply_shared_metadata(cleaned_test_df, shared_metadata)

    train_text = build_model_text(train_shared)
    test_text = build_model_text(test_shared)
    text_train, text_test, vectorizer = build_text_features(
        train_text,
        test_text,
        max_text_features=config.max_text_features,
        sublinear_tf=config.sublinear_tf,
        vocabulary=text_vocabulary,
    )

    feature_columns = structured_feature_columns(shared_metadata)
    struct_train = build_structured_features(train_shared, feature_columns)
    struct_test = build_structured_features(test_shared, feature_columns)

    train_matrix = np.hstack([text_train, struct_train]).astype(np.float32)
    test_matrix = np.hstack([text_test, struct_test]).astype(np.float32)
    return train_matrix, test_matrix, vectorizer, shared_metadata, feature_columns


def make_model(config: ModelConfig) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        learning_rate_init=config.learning_rate_init,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        early_stopping=False,
        random_state=config.random_state,
    )


def fit_and_score(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    eval_matrix: np.ndarray,
    eval_labels: np.ndarray,
    config: ModelConfig,
) -> tuple[MLPClassifier, dict]:
    model = make_model(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_matrix, train_labels)

    train_predictions = model.predict(train_matrix)
    eval_predictions = model.predict(eval_matrix)
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)

    train_accuracy = float(accuracy_score(train_labels, train_predictions))
    eval_accuracy = float(accuracy_score(eval_labels, eval_predictions))
    train_macro_f1 = float(f1_score(train_labels, train_predictions, average="macro"))
    eval_macro_f1 = float(f1_score(eval_labels, eval_predictions, average="macro"))

    metrics = {
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "validation_accuracy": eval_accuracy,
        "validation_macro_f1": eval_macro_f1,
        "accuracy_gap": train_accuracy - eval_accuracy,
        "macro_f1_gap": train_macro_f1 - eval_macro_f1,
        "n_iter": int(model.n_iter_),
        "convergence_warning": bool(convergence_warning),
    }
    return model, metrics


def evaluate_holdout_config(
    train_clean_df: pd.DataFrame,
    val_clean_df: pd.DataFrame,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    config: ModelConfig,
    *,
    shared_module: ModuleType,
    use_group_relative: bool,
) -> dict:
    train_matrix, val_matrix, _, _, _ = make_feature_matrices(
        cleaned_train_df=train_clean_df,
        cleaned_test_df=val_clean_df,
        shared_module=shared_module,
        config=config,
        use_group_relative=use_group_relative,
    )
    _, metrics = fit_and_score(
        train_matrix=train_matrix,
        train_labels=train_labels,
        eval_matrix=val_matrix,
        eval_labels=val_labels,
        config=config,
    )
    return {"config": asdict(config), **metrics}


def evaluate_inner_cv_config(
    train_clean_df: pd.DataFrame,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    config: ModelConfig,
    *,
    shared_module: ModuleType,
    use_group_relative: bool,
    n_splits: int = 5,
) -> dict:
    splitter = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for fold_idx, (inner_train_idx, inner_eval_idx) in enumerate(
        splitter.split(train_clean_df, train_labels, train_groups),
        start=1,
    ):
        inner_train = train_clean_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_eval = train_clean_df.iloc[inner_eval_idx].reset_index(drop=True)
        train_matrix, eval_matrix, _, _, _ = make_feature_matrices(
            cleaned_train_df=inner_train,
            cleaned_test_df=inner_eval,
            shared_module=shared_module,
            config=config,
            use_group_relative=use_group_relative,
        )
        _, metrics = fit_and_score(
            train_matrix=train_matrix,
            train_labels=train_labels[inner_train_idx],
            eval_matrix=eval_matrix,
            eval_labels=train_labels[inner_eval_idx],
            config=config,
        )
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    return {
        "inner_cv_mean_accuracy": float(np.mean([item["validation_accuracy"] for item in fold_metrics])),
        "inner_cv_mean_macro_f1": float(np.mean([item["validation_macro_f1"] for item in fold_metrics])),
        "inner_cv_mean_gap": float(np.mean([item["accuracy_gap"] for item in fold_metrics])),
        "inner_cv_warning_rate": float(np.mean([item["convergence_warning"] for item in fold_metrics])),
        "inner_cv_fold_metrics": fold_metrics,
    }


def choose_best_config(evaluations: list[dict], tolerance: float) -> dict:
    best_inner_cv = max(item["inner_cv_mean_accuracy"] for item in evaluations)
    shortlist = [item for item in evaluations if item["inner_cv_mean_accuracy"] >= best_inner_cv - tolerance]
    shortlist.sort(
        key=lambda item: (
            item["convergence_warning"],
            -item["validation_accuracy"],
            item["accuracy_gap"],
            -item["inner_cv_mean_accuracy"],
            item["inner_cv_mean_gap"],
            -item["validation_macro_f1"],
            item["config"]["max_text_features"],
        )
    )
    return shortlist[0]


def fit_final_model(
    trainval_clean_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    trainval_labels: np.ndarray,
    test_labels: np.ndarray,
    config: ModelConfig,
    *,
    shared_module: ModuleType,
    use_group_relative: bool,
):
    train_matrix, test_matrix, vectorizer, shared_metadata, feature_columns = make_feature_matrices(
        cleaned_train_df=trainval_clean_df,
        cleaned_test_df=test_clean_df,
        shared_module=shared_module,
        config=config,
        use_group_relative=use_group_relative,
    )

    model = make_model(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_matrix, trainval_labels)

    train_predictions = model.predict(train_matrix)
    test_predictions = model.predict(test_matrix)
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)

    train_metrics = {
        "train_accuracy": float(accuracy_score(trainval_labels, train_predictions)),
        "train_macro_f1": float(f1_score(trainval_labels, train_predictions, average="macro")),
        "n_iter": int(model.n_iter_),
        "convergence_warning": bool(convergence_warning),
    }
    test_metrics = {
        "test_accuracy": float(accuracy_score(test_labels, test_predictions)),
        "test_macro_f1": float(f1_score(test_labels, test_predictions, average="macro")),
        "train_test_accuracy_gap": float(
            accuracy_score(trainval_labels, train_predictions) - accuracy_score(test_labels, test_predictions)
        ),
        "train_test_macro_f1_gap": float(
            f1_score(trainval_labels, train_predictions, average="macro")
            - f1_score(test_labels, test_predictions, average="macro")
        ),
    }
    return model, vectorizer, shared_metadata, feature_columns, train_metrics, test_metrics


def export_artifacts(
    output_dir: Path,
    config: ModelConfig,
    label_encoder: LabelEncoder,
    model: MLPClassifier,
    vectorizer: TfidfVectorizer,
    shared_metadata: dict,
    feature_columns: list[str],
    split_summary: dict,
    tuning_summary: dict | None,
    final_train_metrics: dict,
    final_test_metrics: dict,
    use_group_relative: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": 3,
        "label_classes": label_encoder.classes_.tolist(),
        "text_token_pattern": TEXT_TOKEN_PATTERN,
        "text_sublinear_tf": config.sublinear_tf,
        "leakage_patterns": shared_metadata["text_leakage_patterns"],
        "vocabulary": {token: int(idx) for token, idx in vectorizer.vocabulary_.items()},
        "scale_cols": shared_metadata["scale_cols"],
        "medians": shared_metadata["medians"],
        "clip_lower": shared_metadata["clip_lower"],
        "clip_upper": shared_metadata["clip_upper"],
        "means": shared_metadata["means"],
        "stds": shared_metadata["stds"],
        "group_relative_base_columns": shared_metadata.get("group_relative_base_columns", []),
        "group_relative_scale_columns": shared_metadata.get("group_relative_scale_columns", []),
        "fixed_multi_select_categories": shared_metadata["fixed_multi_select_categories"],
        "structured_feature_columns": feature_columns,
        "config": asdict(config),
        "use_group_relative_features": use_group_relative,
        "fit_scope": "train_plus_validation",
        "local_split_summary": split_summary,
        "train_metrics": final_train_metrics,
    }
    with (output_dir / "mlp_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)

    payload = {"idf": vectorizer.idf_.astype(np.float32)}
    for idx, coef in enumerate(model.coefs_):
        payload[f"coef_{idx}"] = coef.astype(np.float32)
    for idx, intercept in enumerate(model.intercepts_):
        payload[f"intercept_{idx}"] = intercept.astype(np.float32)
    np.savez_compressed(output_dir / "mlp_weights.npz", **payload)

    summary_payload = {
        "split_summary": split_summary,
        "selection_policy": {
            "type": "strict_train_val_test_holdout",
            "selection_tolerance": tuning_summary["selection_tolerance"] if tuning_summary else None,
            "use_group_relative_features": use_group_relative,
            "rule": (
                "Keep a fixed outer train/validation/test split. Within train, keep models whose "
                "inner grouped CV accuracy is within tolerance of the best candidate, then prefer "
                "higher outer validation accuracy, lower train-validation gap, and no convergence warning."
            ),
        },
        "tuning_summary": tuning_summary,
        "selected_config": asdict(config),
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)


def candidate_configs() -> list[ModelConfig]:
    return [
        ModelConfig((96,), 1.5e-3, 8.0e-4, 64, 100, 1500, True, 42),
        ModelConfig((128,), 1.2e-3, 8.0e-4, 64, 110, 1800, True, 42),
        ModelConfig((128, 48), 1.2e-3, 8.0e-4, 64, 100, 1800, True, 42),
        ModelConfig((128, 64), 1.0e-3, 8.0e-4, 64, 100, 1800, True, 42),
        ModelConfig((112, 48), 1.2e-3, 7.0e-4, 64, 110, 1700, True, 42),
        ModelConfig((144,), 1.0e-3, 7.0e-4, 64, 110, 2000, True, 42),
        ModelConfig((136, 48), 1.2e-3, 7.0e-4, 64, 110, 1900, True, 42),
        ModelConfig((144, 48), 1.0e-3, 7.0e-4, 64, 100, 2000, True, 42),
        ModelConfig((144, 48), 1.0e-3, 6.0e-4, 64, 120, 2000, True, 42),
        ModelConfig((144, 64), 1.0e-3, 8.0e-4, 64, 90, 2000, True, 42),
        ModelConfig((160, 48), 1.1e-3, 7.0e-4, 64, 100, 2000, True, 42),
        ModelConfig((160, 64), 8.0e-4, 8.0e-4, 64, 90, 2000, True, 42),
        ModelConfig((160, 64), 1.0e-3, 7.0e-4, 64, 100, 2200, True, 42),
        ModelConfig((176, 64), 1.0e-3, 6.0e-4, 64, 100, 2200, True, 42),
        ModelConfig((160, 64), 8.0e-4, 8.0e-4, 64, 90, 2000, False, 42),
    ]


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    output_dir = Path(args.output_dir)
    shared_path = Path(args.shared_preprocess)

    shared_module = load_shared_module(shared_path)
    raw_df = pd.read_csv(train_csv)
    shared_module.validate_group_integrity(raw_df)

    train_raw, val_raw, test_raw, split_summary = split_raw_dataframe(
        raw_df,
        shared_module=shared_module,
        seed=args.seed,
    )

    train_clean_df = shared_module.basic_row_clean(train_raw)
    val_clean_df = shared_module.basic_row_clean(val_raw)
    test_clean_df = shared_module.basic_row_clean(test_raw)

    label_encoder = LabelEncoder()
    label_encoder.fit(raw_df[LABEL_COLUMN].to_numpy())
    train_labels = label_encoder.transform(train_raw[LABEL_COLUMN].to_numpy())
    val_labels = label_encoder.transform(val_raw[LABEL_COLUMN].to_numpy())
    test_labels = label_encoder.transform(test_raw[LABEL_COLUMN].to_numpy())
    train_groups = train_raw[GROUP_COLUMN].to_numpy()

    selected_config = candidate_configs()[7]
    tuning_summary = None
    use_group_relative = not args.disable_group_relative

    if args.search:
        evaluations = []
        for config in candidate_configs():
            outer_metrics = evaluate_holdout_config(
                train_clean_df=train_clean_df,
                val_clean_df=val_clean_df,
                train_labels=train_labels,
                val_labels=val_labels,
                config=config,
                shared_module=shared_module,
                use_group_relative=use_group_relative,
            )
            inner_cv_metrics = evaluate_inner_cv_config(
                train_clean_df=train_clean_df,
                train_labels=train_labels,
                train_groups=train_groups,
                config=config,
                shared_module=shared_module,
                use_group_relative=use_group_relative,
            )
            evaluations.append({**outer_metrics, **inner_cv_metrics})
        best = choose_best_config(evaluations, tolerance=args.selection_tolerance)
        tuning_summary = {
            "selection_tolerance": args.selection_tolerance,
            "evaluations": evaluations,
            "best_validation_result": best,
        }
        best_config = best["config"]
        selected_config = ModelConfig(
            hidden_layer_sizes=tuple(best_config["hidden_layer_sizes"]),
            alpha=best_config["alpha"],
            learning_rate_init=best_config["learning_rate_init"],
            batch_size=best_config["batch_size"],
            max_iter=best_config["max_iter"],
            max_text_features=best_config["max_text_features"],
            sublinear_tf=best_config["sublinear_tf"],
            random_state=best_config["random_state"],
        )

    trainval_clean_df = pd.concat([train_clean_df, val_clean_df], ignore_index=True)
    trainval_labels = np.concatenate([train_labels, val_labels])

    model, vectorizer, shared_metadata, feature_columns, final_train_metrics, final_test_metrics = fit_final_model(
        trainval_clean_df=trainval_clean_df,
        test_clean_df=test_clean_df,
        trainval_labels=trainval_labels,
        test_labels=test_labels,
        config=selected_config,
        shared_module=shared_module,
        use_group_relative=use_group_relative,
    )
    export_artifacts(
        output_dir=output_dir,
        config=selected_config,
        label_encoder=label_encoder,
        model=model,
        vectorizer=vectorizer,
        shared_metadata=shared_metadata,
        feature_columns=feature_columns,
        split_summary=split_summary,
        tuning_summary=tuning_summary,
        final_train_metrics=final_train_metrics,
        final_test_metrics=final_test_metrics,
        use_group_relative=use_group_relative,
    )

    print(f"Saved artifacts to {output_dir.resolve()}")
    print(f"Split summary: {split_summary}")
    print(f"Selected config: {selected_config}")
    if tuning_summary is not None:
        best = tuning_summary["best_validation_result"]
        print(f"Inner CV accuracy: {best['inner_cv_mean_accuracy']:.4f}")
        print(f"Validation accuracy: {best['validation_accuracy']:.4f}")
        print(f"Validation macro-F1: {best['validation_macro_f1']:.4f}")
        print(f"Train-validation gap: {best['accuracy_gap']:.4f}")
    print(f"Final train+val accuracy: {final_train_metrics['train_accuracy']:.4f}")
    print(f"Local test accuracy: {final_test_metrics['test_accuracy']:.4f}")
    print(f"Train+val to test gap: {final_test_metrics['train_test_accuracy_gap']:.4f}")
    print(f"Exported feature count: {model.coefs_[0].shape[0]}")


if __name__ == "__main__":
    main()
