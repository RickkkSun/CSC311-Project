from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlp_preprocess import (
    apply_shared_metadata,
    basic_row_clean,
    build_model_text,
    build_structured_features,
    build_text_features_from_vocabulary,
)

_CACHE: dict | None = None


def _metadata_path() -> Path:
    return Path(__file__).resolve().parent / "mlp_metadata.json"


def _weights_path() -> Path:
    return Path(__file__).resolve().parent / "mlp_weights.npz"


def _load_artifacts() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    metadata_path = _metadata_path()
    weights_path = _weights_path()
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
        raise ValueError("No exported MLP layers found in mlp_weights.npz.")

    _CACHE = {
        "metadata": metadata,
        "idf": weights["idf"].astype(np.float32),
        "coefs": coefs,
        "intercepts": intercepts,
    }
    return _CACHE

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


def predict_all(csv_filename: str) -> list[str]:
    artifacts = _load_artifacts()
    metadata = artifacts["metadata"]
    raw_df = pd.read_csv(csv_filename)

    cleaned_df = basic_row_clean(
        raw_df,
        patterns=metadata["leakage_patterns"],
        fixed_categories=metadata["fixed_multi_select_categories"],
    )
    shared_df = apply_shared_metadata(cleaned_df, metadata)

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pred.py <csv_filename>")
    for prediction in predict_all(sys.argv[1]):
        print(prediction)
