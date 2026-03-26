from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlp_preprocess import (
    GROUP_COLUMN,
    apply_shared_metadata,
    basic_row_clean,
    build_model_text,
    build_structured_features,
    build_text_features_from_vocabulary,
)

_CACHE: dict[str, dict] = {}


def _artifact_dir(use_fallback: bool = False) -> Path:
    dirname = "artifacts_no_group_relative" if use_fallback else "artifacts"
    return Path(__file__).resolve().parent / dirname


def _load_artifacts(use_fallback: bool = False) -> dict:
    cache_key = "fallback" if use_fallback else "primary"
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    artifact_dir = _artifact_dir(use_fallback=use_fallback)
    metadata_path = artifact_dir / "mlp_metadata.json"
    weights_path = artifact_dir / "mlp_weights.npz"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing exported MLP artifacts. Expected {metadata_path} and {weights_path}."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    weights = np.load(weights_path, allow_pickle=False)

    coefs = []
    intercepts = []
    layer_idx = 0
    while f"coef_{layer_idx}" in weights and f"intercept_{layer_idx}" in weights:
        coefs.append(weights[f"coef_{layer_idx}"].astype(np.float32))
        intercepts.append(weights[f"intercept_{layer_idx}"].astype(np.float32))
        layer_idx += 1
    if not coefs:
        raise ValueError("No exported MLP layers found in the weight file.")

    artifacts = {
        "metadata": metadata,
        "idf": weights["idf"].astype(np.float32),
        "coefs": coefs,
        "intercepts": intercepts,
    }
    _CACHE[cache_key] = artifacts
    return artifacts


def _maybe_load_fallback_artifacts() -> dict | None:
    artifact_dir = _artifact_dir(use_fallback=True)
    metadata_path = artifact_dir / "mlp_metadata.json"
    weights_path = artifact_dir / "mlp_weights.npz"
    if not metadata_path.exists() or not weights_path.exists():
        return None
    return _load_artifacts(use_fallback=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=x)


def _forward(features: np.ndarray, artifacts: dict) -> np.ndarray:
    activations = features
    last_idx = len(artifacts["coefs"]) - 1
    for layer_idx, (coef, intercept) in enumerate(zip(artifacts["coefs"], artifacts["intercepts"])):
        activations = activations @ coef + intercept
        if layer_idx != last_idx:
            activations = _relu(activations)
    return activations


def _should_use_group_relative(raw_df: pd.DataFrame, metadata: dict) -> bool:
    if not metadata.get("use_group_relative_features", True):
        return False
    if not metadata.get("group_relative_base_columns"):
        return False
    if GROUP_COLUMN not in raw_df.columns:
        return False

    group_ids = raw_df[GROUP_COLUMN]
    if group_ids.isna().any():
        return False

    expected_group_size = int(metadata.get("expected_group_size", 3))
    counts = group_ids.value_counts(sort=False)
    if counts.empty:
        return False
    return bool((counts == expected_group_size).all())


def _predict_with_artifacts(raw_df: pd.DataFrame, artifacts: dict, use_group_relative: bool) -> list[str]:
    metadata = artifacts["metadata"]
    cleaned_df = basic_row_clean(
        raw_df,
        patterns=metadata["leakage_patterns"],
        fixed_categories=metadata["fixed_multi_select_categories"],
    )
    shared_df = apply_shared_metadata(
        cleaned_df,
        metadata,
        use_group_relative=use_group_relative,
    )

    texts = build_model_text(shared_df).tolist()
    text_features = build_text_features_from_vocabulary(
        texts,
        metadata["vocabulary"],
        artifacts["idf"],
        metadata.get("text_sublinear_tf", False),
    )
    structured_features = build_structured_features(shared_df, metadata["structured_feature_columns"])
    features = np.hstack([text_features, structured_features]).astype(np.float32)

    logits = _forward(features, artifacts)
    label_indices = logits.argmax(axis=1)
    return [metadata["label_classes"][int(idx)] for idx in label_indices]


def predict_all(csv_filename: str) -> list[str]:
    raw_df = pd.read_csv(csv_filename)

    primary_artifacts = _load_artifacts(use_fallback=False)
    if _should_use_group_relative(raw_df, primary_artifacts["metadata"]):
        return _predict_with_artifacts(raw_df, primary_artifacts, use_group_relative=True)

    fallback_artifacts = _maybe_load_fallback_artifacts()
    if fallback_artifacts is not None:
        return _predict_with_artifacts(raw_df, fallback_artifacts, use_group_relative=False)

    return _predict_with_artifacts(raw_df, primary_artifacts, use_group_relative=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pred.py <csv_filename>")
    for prediction in predict_all(sys.argv[1]):
        print(prediction)
