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


def _metadata_path() -> Path:
    return Path(__file__).resolve().parent / "mlp_metadata.json"


def _weights_path() -> Path:
    return Path(__file__).resolve().parent / "mlp_weights.npz"


def _load_layer_params(weights: np.lib.npyio.NpzFile, prefix: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    coefs = []
    intercepts = []
    layer_idx = 0
    while (
        f"{prefix}coef_{layer_idx}" in weights
        and f"{prefix}intercept_{layer_idx}" in weights
    ):
        coefs.append(weights[f"{prefix}coef_{layer_idx}"].astype(np.float32))
        intercepts.append(weights[f"{prefix}intercept_{layer_idx}"].astype(np.float32))
        layer_idx += 1
    return coefs, intercepts


def _load_artifacts(model_key: str = "primary") -> dict:
    cached = _CACHE.get(model_key)
    if cached is not None:
        return cached

    metadata_path = _metadata_path()
    weights_path = _weights_path()
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing exported MLP artifacts. Expected {metadata_path} and {weights_path}."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        raw_metadata = json.load(handle)
    weights = np.load(weights_path, allow_pickle=False)

    if isinstance(raw_metadata, dict) and "primary" in raw_metadata and "fallback" in raw_metadata:
        if model_key not in raw_metadata:
            raise KeyError(f"Missing metadata block for model '{model_key}'.")
        metadata = raw_metadata[model_key]
        prefix = f"{model_key}_"
    else:
        if model_key != "primary":
            raise KeyError(f"Combined fallback artifacts are not available for model '{model_key}'.")
        metadata = raw_metadata
        prefix = ""

    idf_key = f"{prefix}idf"
    if idf_key not in weights:
        raise ValueError(f"Missing TF-IDF idf array '{idf_key}' in {weights_path}.")

    coefs, intercepts = _load_layer_params(weights, prefix)
    if not coefs:
        raise ValueError(f"No exported MLP layers found for model '{model_key}' in {weights_path}.")

    artifacts = {
        "metadata": metadata,
        "idf": weights[idf_key].astype(np.float32),
        "coefs": coefs,
        "intercepts": intercepts,
    }
    _CACHE[model_key] = artifacts
    return artifacts


def _has_fallback_artifacts() -> bool:
    with _metadata_path().open("r", encoding="utf-8") as handle:
        raw_metadata = json.load(handle)
    return isinstance(raw_metadata, dict) and "fallback" in raw_metadata


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

    primary_artifacts = _load_artifacts(model_key="primary")
    if _should_use_group_relative(raw_df, primary_artifacts["metadata"]):
        return _predict_with_artifacts(raw_df, primary_artifacts, use_group_relative=True)

    if _has_fallback_artifacts():
        fallback_artifacts = _load_artifacts(model_key="fallback")
        return _predict_with_artifacts(raw_df, fallback_artifacts, use_group_relative=False)

    return _predict_with_artifacts(raw_df, primary_artifacts, use_group_relative=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pred.py <csv_filename>")
    for prediction in predict_all(sys.argv[1]):
        print(prediction)
