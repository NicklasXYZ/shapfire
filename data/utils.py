from typing import Iterable, Optional, Dict, Sequence, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def _normalize_text_columns(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Lowercase/trim only string-like columns (or a provided subset).
    Uses .casefold() which is more robust than .lower() for text.
    """
    out = df.copy()
    cols = list(columns) if columns is not None else out.select_dtypes(include=["object", "string"]).columns
    for c in cols:
        out[c] = out[c].astype("string").str.strip().str.casefold()
    return out


def _replace_null_sentinels(df: pd.DataFrame, sentinels: Sequence[str] = ("#null!", "nan")) -> pd.DataFrame:
    """
    Convert common null *strings* to actual NaN.
    """
    out = df.copy()
    to_replace = set(s.lower() for s in sentinels) | {"#null!", "#NULL!", "nan", "NaN"}
    return out.replace(list(to_replace), np.nan)


def _coerce_decimal_comma(series: pd.Series) -> pd.Series:
    """
    Convert strings with comma decimal separators to floats.
    No-op for already numeric series.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def cast_features(
    df: pd.DataFrame,
    numerical_features: Optional[Iterable[str]],
    categorical_features: Optional[Iterable[str]],
    *,
    cast_categorical_to_category: bool = True,
) -> pd.DataFrame:
    """
    Vectorized casting by data *type*, not statistical role.

    - Numerical features → float (handles decimal commas and coercion).
    - Categorical features → map Danish yes/no/unknown ('nej','ja','ved ikke') to 0/1/NaN
      when given as text; then optionally cast to pandas 'category'.
    """
    out = df.copy()

    # Numerical
    if numerical_features:
        for col in numerical_features:
            if col in out.columns:
                out[col] = _coerce_decimal_comma(out[col])

    # Categorical (binary example: Danish yes/no/unknown)
    mapping = {"nej": 0, "ja": 1, "ved ikke": np.nan}
    if categorical_features:
        for col in categorical_features:
            if col in out.columns:
                # normalize text only if not already numeric
                s = out[col]
                if not pd.api.types.is_numeric_dtype(s):
                    s = s.astype("string").str.strip().str.casefold()
                    s = s.map(mapping).astype("Float64")  # nullable float for NaN support
                else:
                    s = pd.to_numeric(s, errors="coerce")
                out[col] = s.astype("category") if cast_categorical_to_category else s

    return out


def organize_input_data(
    df: pd.DataFrame,
    drop_features: Optional[Iterable[str]] = None,
    numerical_features: Optional[Iterable[str]] = None,
    categorical_features: Optional[Iterable[str]] = None,
    *,
    create_bmi: bool = True,
    bmi_cols: Tuple[str, str] = ("weight", "height"),
    cast_categorical_to_category: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize your raw dataframe.

    Steps:
      1) Drop unwanted columns.
      2) Normalize text-like columns (trim + casefold).
      3) Replace common null sentinels with NaN.
      4) Cast categorical/numerical columns by *data type*:
         - categorical: map 'nej'/'ja'/'ved ikke' → 0/1/NaN, optional pandas 'category'
         - numerical: handle decimal commas and coerce to float
      5) Create BMI if weight/height available and valid.
    """
    out = df.copy()

    # 1) Drop columns
    if drop_features:
        out = out.drop(columns=[c for c in drop_features if c in out.columns], errors="ignore")

    # 2) Normalize string-like columns only
    out = _normalize_text_columns(out)

    # 3) Standardize null tokens to NaN
    out = _replace_null_sentinels(out, sentinels=("#NULL!", "#null!", "nan", "NaN"))

    # 4) Casts by declared data type
    out = cast_features(
        out,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        cast_categorical_to_category=cast_categorical_to_category,
    )

    # 5) BMI (kg / (m^2)) if both columns exist and height > 0
    if create_bmi:
        weight_col, height_col = bmi_cols
        if weight_col in out.columns and height_col in out.columns:
            w = pd.to_numeric(out[weight_col], errors="coerce")
            h_cm = pd.to_numeric(out[height_col], errors="coerce")
            h_m = h_cm / 100.0
            valid = (w.notna()) & (h_m.notna()) & (h_m > 0)
            bmi = pd.Series(np.nan, index=out.index, dtype="float64")
            bmi[valid] = w[valid] / (h_m[valid] ** 2)
            out["bmi"] = bmi

    return out


def import_and_preprocess_data(
    numerical_features: list[str],
    categorical_features: list[str],
    target_variable: str,
    *,
    dataset_path: Optional[str | Path] = None,
    drop_features: Optional[list[str]] = None,
    organized_subgroups_path: str | Path = "data/original_dataset_with_subgroups_column.csv",
    split_test_size: float = 0.50,
    split_random_state: int = 0,
    stratify_on_target: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    If `dataset_path` is provided (and exists), use the original dataset:
      - load -> organize -> split
      - add a 'subgroup' column (0 = preprocess split, 1 = final split)
      - save to `organized_subgroups_path` for future runs
      - return (X_preprocess, X_final, y_preprocess, y_final)

    Otherwise, load the previously saved subgroup-based dataset
    from `organized_subgroups_path` and return the same four objects.
    """

    def _ensure_columns(df: pd.DataFrame, must_exist: Iterable[str], context: str) -> None:
        missing = [c for c in must_exist if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s) in {context}: {missing}")

    # Normalize paths
    dataset_path = Path(dataset_path) if dataset_path is not None else None
    organized_subgroups_path = Path(organized_subgroups_path)

    # -------- Mode A: original dataset provided --------
    if dataset_path is not None:
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")

        # Load raw, then organize using your helpers
        df = pd.read_csv(dataset_path, sep=",", header="infer", index_col=0)

        ndf = organize_input_data(
            df=df,
            drop_features=drop_features or [],
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        _ensure_columns(ndf, [target_variable], "organized dataframe")

        y = ndf[target_variable]
        X = ndf.drop(columns=[target_variable])

        stratify = y if stratify_on_target else None
        X_preprocess, X_final, y_preprocess, y_final = train_test_split(
            X,
            y,
            test_size=split_test_size,
            random_state=split_random_state,
            stratify=stratify,
        )

        # Persist a subgrouped dataset for future runs
        subgroup_df = ndf.copy()
        subgroup_df["subgroup"] = 0
        subgroup_df.loc[X_final.index, "subgroup"] = 1
        # Ensure target + subgroup exist in saved file
        _ensure_columns(subgroup_df, [target_variable, "subgroup"], "subgroup dataframe")
        organized_subgroups_path.parent.mkdir(parents=True, exist_ok=True)
        subgroup_df.to_csv(organized_subgroups_path)

        return X_preprocess, X_final, y_preprocess, y_final

    # -------- Mode B: load previously created subgroup dataset --------
    if not organized_subgroups_path.exists():
        raise FileNotFoundError(
            f"organized_subgroups_path not found: {organized_subgroups_path}. "
            f"Run this function once with `dataset_path=...` to create it."
        )

    ndf = pd.read_csv(organized_subgroups_path, sep=",", header="infer", index_col=0)
    _ensure_columns(ndf, [target_variable, "subgroup"], "organized subgroup dataframe")

    # Optional: coerce dtypes again
    for col in numerical_features or []:
        if col in ndf.columns:
            ndf[col] = pd.to_numeric(ndf[col], errors="coerce")
    for col in categorical_features or []:
        if col in ndf.columns:
            ndf[col] = ndf[col].astype("category")

    if not set(ndf["subgroup"].unique()).issuperset({0, 1}):
        raise ValueError("`subgroup` column must contain at least the values 0 and 1.")

    cols_to_drop = [target_variable, "subgroup"]
    X_preprocess = ndf[ndf["subgroup"] == 0].drop(columns=[c for c in cols_to_drop if c in ndf.columns])
    y_preprocess = ndf.loc[ndf["subgroup"] == 0, target_variable]
    X_final = ndf[ndf["subgroup"] == 1].drop(columns=[c for c in cols_to_drop if c in ndf.columns])
    y_final = ndf.loc[ndf["subgroup"] == 1, target_variable]

    return X_preprocess, X_final, y_preprocess, y_final