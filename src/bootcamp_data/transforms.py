import pandas as pd
import re

# =========================
# Datetime helpers (Day 3)
# =========================

def parse_datetime(df: pd.DataFrame, col: str, utc: bool = True) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce", utc=utc)
    return out

def add_time_parts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[col]
    out[f"{col}_date"] = dt.dt.date
    out[f"{col}_year"] = dt.dt.year
    out[f"{col}_month"] = dt.dt.month
    out[f"{col}_dow"] = dt.dt.dayofweek
    out[f"{col}_hour"] = dt.dt.hour
    return out

# =========================
# Outlier helpers (Day 3)
# =========================

def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = float(q1 - k * iqr)
    hi = float(q3 + k * iqr)
    return lo, hi

def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.clip(lower=lo, upper=hi)

def add_outlier_flag(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:
    lo, hi = iqr_bounds(df[col], k=k)
    x = pd.to_numeric(df[col], errors="coerce")
    return df.assign(is_outlier=(x < lo) | (x > hi))

# =========================
# Schema & quality helpers
# =========================

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        order_id=df["order_id"].astype("string"),
        user_id=df["user_id"].astype("string"),
        amount=pd.to_numeric(df["amount"], errors="coerce").astype("Float64"),
        quantity=pd.to_numeric(df["quantity"], errors="coerce").astype("Int64"),
    )

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna().sum()
        .rename("n_missing")
        .to_frame()
        .assign(p_missing=lambda t: t["n_missing"] / len(df))
        .sort_values("p_missing", ascending=False)
    )

def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}_is_missing"] = out[c].isna()
    return out

# =========================
# Text normalization
# =========================

_ws = re.compile(r"\s+")

def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(_ws, " ", regex=True)
    )

def apply_mapping(s: pd.Series, mapping: dict[str, str]) -> pd.Series:
    s_norm = normalize_text(s)
    return s_norm.map(lambda x: mapping.get(x, x))

# =========================
# Deduplication helper
# =========================

def dedupe_keep_latest(df: pd.DataFrame, key_cols: list[str], ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    return (
        out.sort_values(ts_col)
           .drop_duplicates(subset=key_cols, keep="last")
           .reset_index(drop=True)
    )
