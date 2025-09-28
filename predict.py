import argparse, json, sys, gzip, datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import lightgbm as lgb


def _die(msg: str, code: int = 2):
    print(msg, file=sys.stderr)
    sys.exit(code)

def _read_table(path: str, assume_header: bool = True) -> pd.DataFrame:
    """
    Read CSV (also supports .gz) by default. If file ends with .parquet/.pq, use parquet.
    """
    p = Path(path)
    if not p.exists():
        _die(f"[ERROR] Input file not found: {path}")

    if p.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except Exception as e:
            _die(f"[ERROR] Failed reading Parquet: {e}")
    # CSV (optionally gz)
    try:
        if p.suffix.lower() == ".gz":
            with gzip.open(p, "rt") as f:
                return pd.read_csv(f)
        else:
            return pd.read_csv(p)
    except Exception as e:
        _die(f"[ERROR] Failed reading CSV: {e}")

    # unreachable
    return pd.DataFrame()


def load_artifacts(export_dir: str):
    p = Path(export_dir)
    try:
        booster = lgb.Booster(model_file=str(p / "model.txt"))
    except Exception as e:
        _die(f"[ERROR] Failed to load LightGBM model: {e}")

    try:
        schema = json.loads((p / "schema.json").read_text())
    except Exception as e:
        _die(f"[ERROR] Failed to read schema.json: {e}")

    try:
        meta = json.loads((p / "params.json").read_text())
    except Exception:
        # params are nice to have (best_iteration) but not strictly required
        meta = {}

    # Pull schema fields
    try:
        feat_names: List[str] = schema["feature_names"]
        dtypes_map: Dict[str, str] = schema["dtypes"]
    except KeyError as e:
        _die(f"[ERROR] schema.json missing key: {e}")

    # label policy & best iteration (optional)
    label_policy = schema.get("label_policy", {"type": "threshold", "threshold": 0.5})
    best_iteration = meta.get("best_iteration", None)

    # Return only what the caller actually uses
    return booster, feat_names, dtypes_map, label_policy, best_iteration


def _ensure_required_columns(df: pd.DataFrame, feat_names: List[str]):
    # Fail on duplicate columns (ambiguous mapping)
    dupe_cols = df.columns[df.columns.duplicated()].tolist()
    if dupe_cols:
        _die(f"[ERROR] Duplicate column names in input: {dupe_cols}")

    # Strict presence check
    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        msg = (
            "[ERROR] Missing required columns from input.\n"
            f"Required (schema order): {feat_names}\n"
            f"Missing: {missing}\n"
            f"Present: {df.columns.tolist()}"
        )
        _die(msg)


def coerce_dataframe(
    df: pd.DataFrame,
    feat_names: List[str],
    dtypes_map: Dict[str, str],
):
    rep = {"extra_columns": [], "dtype_changes": {}, "n_rows": len(df)}

    # ignore extras by subsetting
    rep["extra_columns"] = [c for c in df.columns if c not in feat_names]
    df = df[feat_names]

    # dtype coercion
    for c in feat_names:
        tgt = dtypes_map.get(c)
        if tgt == "category":
            # Make sure dtype is CategoricalDtype (levels not enforced here)
            if not pd.api.types.is_categorical_dtype(df[c].dtype):
                rep["dtype_changes"][c] = f"{df[c].dtype} -> category"
                df[c] = df[c].astype("category")
        elif tgt == "float64":
            if df[c].dtype != "float64":
                rep["dtype_changes"][c] = f"{df[c].dtype} -> float64"
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        else:
            _die(f"[ERROR] Unsupported dtype in schema for '{c}': {tgt}")

    return df, rep


def _predict(
    booster: lgb.Booster,
    X: pd.DataFrame,
    num_iter: Optional[int],
    num_threads: Optional[int] = 1,
):
    kwargs = {}
    if num_threads is not None:
        kwargs["num_threads"] = int(num_threads)
    # validate_features=True (default) checks order/names; we already enforce, so keep default
    return booster.predict(X, num_iteration=num_iter, **kwargs)


def _write_log(log_path: Path, payload: dict):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception as e:
        _die(f"[ERROR] Failed to write log file '{log_path}': {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Apply a LightGBM model exported as model.txt + schema.json."
    )
    ap.add_argument(
        "--export-dir",
        required=True,
        help="Directory containing model.txt, schema.json, params.json",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input table: CSV (optionally .gz) or Parquet",
    )
    ap.add_argument(
        "--prob-only",
        action="store_true",
        help="Only probabilities (no labels)",
    )
    ap.add_argument(
        "--output", required=True, help="Output CSV with predictions"
    )
    ap.add_argument(
        "--log-output",
        default=None,
        help="Path to a JSON log with run metadata (defaults to <output>.log.json)",
    )
    args = ap.parse_args()

    (
        booster,
        feat_names,
        dtypes_map,
        label_policy,
        best_iter,
    ) = load_artifacts(args.export_dir)

    df_in = _read_table(args.input)
    df_in = df_in.loc[:, ~df_in.columns.str.match(r"Unnamed")]

    # Strict presence + duplicate checks
    _ensure_required_columns(df_in, feat_names)

    # Coercion & subsetting (ignore extras)
    X, rep = coerce_dataframe(df_in, feat_names, dtypes_map)

    # Predict with early-stopping iteration (if present)
    num_iter = int(best_iter) if (best_iter := best_iter) is not None else None

    proba = _predict(booster, X, num_iter)

    # Expect binary: 1D scores. If not, fail explicitly
    if proba.ndim != 1:
        _die(
            "[ERROR] Model appears to be multiclass (2D probability array). "
            "This script currently supports binary models only."
        )

    # Clean CSV: only predictions (+ optional labels)
    out = pd.DataFrame({"pred_proba": proba})
    if not args.prob_only and label_policy.get("type") == "threshold":
        thr = float(label_policy.get("threshold", 0.5))
        out["pred_label"] = (proba >= thr).astype(int)

    try:
        out.to_csv(args.output, index=False)
    except Exception as e:
        _die(f"[ERROR] Failed to write output CSV: {e}")

    # Build metadata log payload
    log_payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "export_dir": str(Path(args.export_dir).resolve()),
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "prob_only": bool(args.prob_only),
        "label_policy": label_policy,
        "best_iteration": None if num_iter is None else int(num_iter),
        "n_rows": rep["n_rows"],
        "extra_columns": rep["extra_columns"],
        "dtype_changes": rep["dtype_changes"],
    }
    log_path = Path(args.log_output) if args.log_output else Path(args.output).with_suffix(Path(args.output).suffix + ".log.json")
    _write_log(log_path, log_payload)

    print(f"[OK] wrote {args.output}")
    print(f"[OK] wrote log {log_path}")
    if rep["extra_columns"]:
        print(f"[INFO] Ignored extra columns: {rep['extra_columns']}")
    if rep["dtype_changes"]:
        print(f"[INFO] Coerced dtypes: {rep['dtype_changes']}")


if __name__ == "__main__":
    main()
