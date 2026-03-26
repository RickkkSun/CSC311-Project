from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


TEXT_MAP = {
    "Describe how this painting makes you feel.": "feel_text",
    "If this painting was a food, what would be?": "food_text",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "soundtrack_text",
}

NUMERIC_MAP = {
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
    "How many prominent colours do you notice in this painting?": "colour_count",
    "How many objects caught your eye in the painting?": "object_count",
}

LIKERT_MAP = {
    "This art piece makes me feel sombre.": "sombre",
    "This art piece makes me feel content.": "content",
    "This art piece makes me feel calm.": "calm",
    "This art piece makes me feel uneasy.": "uneasy",
}

MULTI_MAP = {
    "If you could purchase this painting, which room would you put that painting in?": "room",
    "If you could view this art in person, who would you want to view it with?": "view_with",
    "What season does this art piece remind you of?": "season",
}

PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"
GROUP_COLUMN = "unique_id"
LABEL_COLUMN = "Painting"
TEXT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
MULTI_SHORT_NAMES = ["room", "view_with", "season"]

MISSING_COLUMNS = [
    "feel_text_missing",
    "food_text_missing",
    "soundtrack_text_missing",
    "combined_text_missing",
    "emotion_intensity_missing",
    "colour_count_missing",
    "object_count_missing",
    "sombre_missing",
    "content_missing",
    "calm_missing",
    "uneasy_missing",
    "willingness_to_pay_missing",
    "room_missing",
    "view_with_missing",
    "season_missing",
]

UNKNOWN_COUNT_COLUMNS = [
    "room_unknown_count",
    "view_with_unknown_count",
    "season_unknown_count",
]

GROUP_RELATIVE_BASE_COLUMNS = [
    "emotion_intensity_value",
    "colour_count_value",
    "object_count_value",
    "sombre_value",
    "content_value",
    "calm_value",
    "uneasy_value",
    "willingness_to_pay_log",
    "feel_text_len",
    "food_text_len",
    "soundtrack_text_len",
    "combined_text_len",
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: object, patterns: list[str]) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text == "":
        return ""

    text = strip_accents(text).lower()
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_likert(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*([1-5])", str(value))
    if match:
        return float(match.group(1))
    return np.nan


def parse_money(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if text == "":
        return np.nan
    text = text.replace(",", "")

    scale_map = {
        "k": 1_000,
        "thousand": 1_000,
        "m": 1_000_000,
        "million": 1_000_000,
        "b": 1_000_000_000,
        "billion": 1_000_000_000,
    }

    match = re.search(
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(k|m|b|thousand|million|billion)?\b",
        text,
    )
    if match:
        amount = float(match.group(1))
        suffix = match.group(2)
        if suffix:
            amount *= scale_map[suffix]
        return amount

    match = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(k|m|b|thousand|million|billion)\b",
        text,
    )
    if match:
        amount = float(match.group(1))
        amount *= scale_map[match.group(2)]
        return amount

    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", text)
    if match:
        return float(match.group(1))

    return np.nan


def split_multi_select(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def join_nonempty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def basic_row_clean(raw_df: pd.DataFrame, patterns: list[str], fixed_categories: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_df.index)
    if GROUP_COLUMN in raw_df.columns:
        out[GROUP_COLUMN] = raw_df[GROUP_COLUMN]
    if LABEL_COLUMN in raw_df.columns:
        out[LABEL_COLUMN] = raw_df[LABEL_COLUMN]

    cleaned_text_cols = []
    for src_col, short_name in TEXT_MAP.items():
        clean_col = f"{short_name}_clean"
        missing_col = f"{short_name}_missing"
        length_col = f"{short_name}_len"

        out[clean_col] = raw_df[src_col].fillna("").map(lambda value: normalize_text(value, patterns))
        out[missing_col] = (out[clean_col] == "").astype(int)
        out[length_col] = out[clean_col].str.len()
        cleaned_text_cols.append(clean_col)

    out["combined_text_clean"] = out[cleaned_text_cols].apply(
        lambda row: join_nonempty(row.tolist()),
        axis=1,
    )
    out["combined_text_missing"] = (out["combined_text_clean"] == "").astype(int)
    out["combined_text_len"] = out["combined_text_clean"].str.len()

    for src_col, short_name in NUMERIC_MAP.items():
        raw_col = f"{short_name}_raw"
        out[raw_col] = pd.to_numeric(raw_df[src_col], errors="coerce")
        out[f"{short_name}_missing"] = out[raw_col].isna().astype(int)

    for src_col, short_name in LIKERT_MAP.items():
        raw_col = f"{short_name}_raw"
        out[raw_col] = raw_df[src_col].map(parse_likert)
        out[f"{short_name}_missing"] = out[raw_col].isna().astype(int)

    out["willingness_to_pay_raw"] = raw_df[PRICE_COL].map(parse_money)
    out["willingness_to_pay_missing"] = out["willingness_to_pay_raw"].isna().astype(int)

    for src_col, short_name in MULTI_MAP.items():
        token_lists = raw_df[src_col].map(split_multi_select)
        allowed = set(fixed_categories[short_name])
        out[f"{short_name}_tokens"] = token_lists.map(lambda values: " | ".join(values))
        out[f"{short_name}_missing"] = token_lists.map(lambda values: int(len(values) == 0))
        out[f"{short_name}_unknown_count"] = token_lists.map(
            lambda values, allowed_set=allowed: sum(token not in allowed_set for token in values)
        )

    return out


def _group_relative_feature_names(base_columns: list[str]) -> list[str]:
    names: list[str] = []
    for column in base_columns:
        names.append(f"{column}_group_centered")
        names.append(f"{column}_group_rank")
    return names


def _add_group_relative_raw_features(df: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    out = df.copy()

    if GROUP_COLUMN not in out.columns:
        for column in base_columns:
            out[f"{column}_group_centered"] = 0.0
            out[f"{column}_group_rank"] = 0.0
        return out

    group_sizes = out.groupby(GROUP_COLUMN)[GROUP_COLUMN].transform("size").astype(float)
    group_denominator = np.maximum(group_sizes - 1.0, 1.0)

    for column in base_columns:
        group_mean = out.groupby(GROUP_COLUMN)[column].transform("mean")
        rank = out.groupby(GROUP_COLUMN)[column].rank(method="average")
        out[f"{column}_group_centered"] = out[column] - group_mean
        out[f"{column}_group_rank"] = (rank - 1.0) / group_denominator

    return out


def _add_zero_group_relative_raw_features(df: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in base_columns:
        out[f"{column}_group_centered"] = 0.0
        out[f"{column}_group_rank"] = 0.0
    return out


def fit_group_relative_metadata(train_df: pd.DataFrame, metadata: dict) -> dict:
    updated = {
        key: (value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value)
        for key, value in metadata.items()
    }
    tmp = apply_shared_metadata(train_df, updated)
    tmp = _add_group_relative_raw_features(tmp, GROUP_RELATIVE_BASE_COLUMNS)

    relative_columns = _group_relative_feature_names(GROUP_RELATIVE_BASE_COLUMNS)
    for column in relative_columns:
        lower = float(tmp[column].quantile(0.01))
        upper = float(tmp[column].quantile(0.99))
        if pd.isna(lower):
            lower = 0.0
        if pd.isna(upper):
            upper = 0.0
        if upper < lower:
            upper = lower

        clipped = tmp[column].clip(lower=lower, upper=upper)
        mean = float(clipped.mean())
        std = float(clipped.std())
        if pd.isna(std) or std == 0.0:
            std = 1.0

        updated["clip_lower"][column] = lower
        updated["clip_upper"][column] = upper
        updated["means"][column] = mean
        updated["stds"][column] = std

    updated["group_relative_base_columns"] = GROUP_RELATIVE_BASE_COLUMNS
    updated["group_relative_scale_columns"] = relative_columns
    updated["scale_cols"] = list(updated["scale_cols"]) + relative_columns
    return updated


def apply_shared_metadata(
    df: pd.DataFrame,
    metadata: dict,
    use_group_relative: bool | None = None,
) -> pd.DataFrame:
    out = df.copy()

    raw_to_value = [
        "emotion_intensity_raw",
        "colour_count_raw",
        "object_count_raw",
        "sombre_raw",
        "content_raw",
        "calm_raw",
        "uneasy_raw",
    ]

    for raw_col in raw_to_value:
        value_col = raw_col.replace("_raw", "_value")
        out[value_col] = out[raw_col].fillna(metadata["medians"][raw_col])

    out["willingness_to_pay_value"] = (
        out["willingness_to_pay_raw"]
        .fillna(metadata["medians"]["willingness_to_pay_raw"])
        .clip(lower=0)
    )
    out["willingness_to_pay_log"] = np.log1p(out["willingness_to_pay_value"])

    group_relative_base_columns = metadata.get("group_relative_base_columns")
    if group_relative_base_columns:
        if use_group_relative is False:
            out = _add_zero_group_relative_raw_features(out, group_relative_base_columns)
        else:
            out = _add_group_relative_raw_features(out, group_relative_base_columns)

    for column in metadata["scale_cols"]:
        clipped_col = f"{column}_clipped"
        z_col = f"{column}_z"
        out[clipped_col] = out[column].clip(
            lower=metadata["clip_lower"][column],
            upper=metadata["clip_upper"][column],
        )
        out[z_col] = (out[clipped_col] - metadata["means"][column]) / metadata["stds"][column]

    for short_name, categories in metadata["fixed_multi_select_categories"].items():
        token_sets = out[f"{short_name}_tokens"].fillna("").map(
            lambda value: {token.strip() for token in value.split("|") if token.strip()}
        )
        multi_hot_block = {
            f"{short_name}__{slugify(category)}": token_sets.map(
                lambda values, target=category: int(target in values)
            )
            for category in categories
        }
        out = pd.concat([out, pd.DataFrame(multi_hot_block, index=out.index)], axis=1)

    return out


def build_model_text(shared_df: pd.DataFrame) -> pd.Series:
    return (
        "feel_desc "
        + shared_df["feel_text_clean"].fillna("").map(str)
        + " food_desc "
        + shared_df["food_text_clean"].fillna("").map(str)
        + " soundtrack_desc "
        + shared_df["soundtrack_text_clean"].fillna("").map(str)
    )


def structured_feature_columns(metadata: dict) -> list[str]:
    scaled_columns = [f"{column}_z" for column in metadata["scale_cols"]]
    multi_hot_columns: list[str] = []
    for short_name in MULTI_SHORT_NAMES:
        for category in metadata["fixed_multi_select_categories"][short_name]:
            multi_hot_columns.append(f"{short_name}__{slugify(category)}")
    return scaled_columns + MISSING_COLUMNS + UNKNOWN_COUNT_COLUMNS + multi_hot_columns


def build_structured_features(shared_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return shared_df.loc[:, feature_columns].fillna(0.0).astype(np.float32).to_numpy()


def tokenize_with_bigrams(text: str) -> list[str]:
    tokens = re.findall(TEXT_TOKEN_PATTERN, text)
    if not tokens:
        return []
    features = list(tokens)
    features.extend(f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1))
    return features


def build_text_features_from_vocabulary(
    texts: list[str],
    vocabulary: dict[str, int],
    idf: np.ndarray,
    sublinear_tf: bool,
) -> np.ndarray:
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        counts: dict[int, float] = {}
        for token in tokenize_with_bigrams(text):
            token_idx = vocabulary.get(token)
            if token_idx is not None:
                counts[token_idx] = counts.get(token_idx, 0.0) + 1.0
        if not counts:
            continue
        indices = np.fromiter(counts.keys(), dtype=np.int32)
        values = np.fromiter(counts.values(), dtype=np.float32)
        if sublinear_tf:
            values = 1.0 + np.log(values)
        values *= idf[indices]
        norm = float(np.linalg.norm(values))
        if norm > 0.0:
            values /= norm
        matrix[row_idx, indices] = values
    return matrix
