from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


GROUP_COL = "unique_id"
LABEL_COL = "Painting"

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

EXPECTED_LABELS = {
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
}

FIXED_MULTI_SELECT_CATEGORIES = {
    "room": [
        "Bedroom",
        "Bathroom",
        "Office",
        "Living room",
        "Dining room",
    ],
    "view_with": [
        "Friends",
        "Family members",
        "Coworkers/Classmates",
        "Strangers",
        "By yourself",
    ],
    "season": [
        "Spring",
        "Summer",
        "Fall",
        "Winter",
    ],
}

TEXT_LEAKAGE_PATTERNS = [
    r"\bthe persistence of memory\b",
    r"\bpersistence of memory\b",
    r"\bthe starry night\b",
    r"\bstarry night\b",
    r"\bthe water lily pond\b",
    r"\bwater lily pond\b",
    r"\bwater lilies\b",
    r"\bwater lily\b",
    r"\blily pond\b",
    r"\bsalvador\b",
    r"\bdali\b",
    r"\bdal[ií]\b",
    r"\bmonet\b",
    r"\bclaude\b",
    r"\bvincent\b",
    r"\bvan gogh\b",
    r"\bgogh\b",
    r"\bstarry\b",
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""

    text = str(x).strip()
    if text == "":
        return ""

    text = strip_accents(text).lower()
    for pattern in TEXT_LEAKAGE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_likert(x: object) -> float:
    if pd.isna(x):
        return np.nan
    match = re.match(r"\s*([1-5])", str(x))
    if match:
        return float(match.group(1))
    return np.nan


def parse_price(x: object) -> float:
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()
    if s == "":
        return np.nan

    s = s.replace(",", "")
    scale_map = {
        "k": 1_000,
        "thousand": 1_000,
        "m": 1_000_000,
        "million": 1_000_000,
        "b": 1_000_000_000,
        "billion": 1_000_000_000,
    }

    p1 = re.search(
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(k|m|b|thousand|million|billion)?\b",
        s,
    )
    if p1:
        value = float(p1.group(1))
        suffix = p1.group(2)
        if suffix:
            value *= scale_map[suffix]
        return value

    p2 = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(k|m|b|thousand|million|billion)\b",
        s,
    )
    if p2:
        value = float(p2.group(1))
        value *= scale_map[p2.group(2)]
        return value

    p3 = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", s)
    if p3:
        return float(p3.group(1))

    return np.nan


def split_multi_select(x: object) -> list[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "":
        return []
    return [part.strip() for part in s.split(",") if part.strip()]


def join_nonempty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def make_group_split(unique_ids, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_ids = np.array(sorted(unique_ids))
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    return train_ids, val_ids, test_ids


def validate_group_integrity(raw_df: pd.DataFrame) -> None:
    counts = raw_df.groupby(GROUP_COL).size()
    bad_count_ids = counts[counts != 3]
    if not bad_count_ids.empty:
        raise ValueError(
            "Found unique_id groups that do not have exactly 3 rows. "
            f"Examples: {bad_count_ids.head().to_dict()}"
        )

    label_sets = raw_df.groupby(GROUP_COL)[LABEL_COL].agg(lambda x: set(x))
    bad_label_ids = label_sets[label_sets != EXPECTED_LABELS]
    if not bad_label_ids.empty:
        example = {int(k): sorted(list(v)) for k, v in bad_label_ids.head().to_dict().items()}
        raise ValueError(
            "Found unique_id groups that do not cover all 3 expected labels. "
            f"Examples: {example}"
        )


def basic_row_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out[GROUP_COL] = raw_df[GROUP_COL]
    out[LABEL_COL] = raw_df[LABEL_COL]

    cleaned_text_cols = []
    for src_col, short_name in TEXT_MAP.items():
        clean_col = f"{short_name}_clean"
        out[clean_col] = raw_df[src_col].fillna("").map(normalize_text)
        out[f"{short_name}_missing"] = (out[clean_col] == "").astype(int)
        out[f"{short_name}_len"] = out[clean_col].str.len()
        cleaned_text_cols.append(clean_col)

    out["combined_text_clean"] = out[cleaned_text_cols].apply(lambda row: join_nonempty(row.tolist()), axis=1)
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

    out["willingness_to_pay_raw"] = raw_df[PRICE_COL].map(parse_price)
    out["willingness_to_pay_missing"] = out["willingness_to_pay_raw"].isna().astype(int)

    for src_col, short_name in MULTI_MAP.items():
        token_lists = raw_df[src_col].map(split_multi_select)
        allowed = set(FIXED_MULTI_SELECT_CATEGORIES[short_name])
        out[f"{short_name}_tokens"] = token_lists.map(lambda xs: " | ".join(xs))
        out[f"{short_name}_missing"] = token_lists.map(lambda xs: int(len(xs) == 0))
        out[f"{short_name}_unknown_count"] = token_lists.map(
            lambda xs, allowed_set=allowed: sum(token not in allowed_set for token in xs)
        )

    return out


def fit_shared_metadata(train_df: pd.DataFrame) -> dict:
    impute_cols = [
        "emotion_intensity_raw",
        "colour_count_raw",
        "object_count_raw",
        "sombre_raw",
        "content_raw",
        "calm_raw",
        "uneasy_raw",
        "willingness_to_pay_raw",
    ]

    medians = {}
    for col in impute_cols:
        median = train_df[col].median()
        if pd.isna(median):
            median = 0.0
        medians[col] = float(median)

    train_tmp = train_df.copy()

    value_cols = []
    for col in [
        "emotion_intensity_raw",
        "colour_count_raw",
        "object_count_raw",
        "sombre_raw",
        "content_raw",
        "calm_raw",
        "uneasy_raw",
    ]:
        value_col = col.replace("_raw", "_value")
        train_tmp[value_col] = train_tmp[col].fillna(medians[col])
        value_cols.append(value_col)

    train_tmp["willingness_to_pay_value"] = (
        train_tmp["willingness_to_pay_raw"]
        .fillna(medians["willingness_to_pay_raw"])
        .clip(lower=0)
    )
    train_tmp["willingness_to_pay_log"] = np.log1p(train_tmp["willingness_to_pay_value"])

    scale_cols = value_cols + [
        "willingness_to_pay_log",
        "feel_text_len",
        "food_text_len",
        "soundtrack_text_len",
        "combined_text_len",
    ]

    clip_lower = {}
    clip_upper = {}
    means = {}
    stds = {}
    for col in scale_cols:
        lower = float(train_tmp[col].quantile(0.01))
        upper = float(train_tmp[col].quantile(0.99))
        if pd.isna(lower):
            lower = 0.0
        if pd.isna(upper):
            upper = 0.0
        if upper < lower:
            upper = lower

        clipped = train_tmp[col].clip(lower=lower, upper=upper)
        mean = float(clipped.mean())
        std = float(clipped.std())
        if pd.isna(std) or std == 0.0:
            std = 1.0

        clip_lower[col] = lower
        clip_upper[col] = upper
        means[col] = mean
        stds[col] = std

    return {
        "seed": 42,
        "expected_labels": sorted(EXPECTED_LABELS),
        "text_leakage_patterns": TEXT_LEAKAGE_PATTERNS,
        "fixed_multi_select_categories": FIXED_MULTI_SELECT_CATEGORIES,
        "medians": medians,
        "scale_cols": scale_cols,
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
        "means": means,
        "stds": stds,
    }
