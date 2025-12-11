import pandas as pd
import re
from datetime import datetime
from dateutil import tz

EU_DT_NO_TIME = ["%d/%m/%y", "%d/%m/%Y"]
EU_DT_WITH_TIME = ["%d/%m/%y %H:%M", "%d/%m/%Y %H:%M"]

def strip_value(x: str) -> str:
    if pd.isna(x):
        return x
    x = str(x).replace("\ufeff", "")
    return x.strip().strip("$").strip()

def read_pipe_dollar_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append([strip_value(t) for t in line.split("|")])
    if not rows:
        return pd.DataFrame()
    header = [strip_value(h) for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header)
    return df

def parse_eu_datetime(s: str):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    for fmt in EU_DT_WITH_TIME + EU_DT_NO_TIME:
        try:
            return pd.to_datetime(datetime.strptime(s, fmt))
        except Exception:
            continue
    return pd.NaT

def coerce_numeric(df, cols, int_cols=None):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if int_cols and c in int_cols:
                df[c] = df[c].astype("Int64")
    return df

def ensure_bounds(df, col, lo=None, hi=None):
    if col not in df.columns: return df
    if lo is not None:
        df.loc[df[col] < lo, col] = pd.NA
    if hi is not None:
        df.loc[df[col] > hi, col] = pd.NA
    return df

def add_flag(df, name, value):
    df[name] = value
    return df
