from pathlib import Path
import json
import numpy as np
import pandas as pd

# ------------------------- chemins & sorties -------------------------
PROC = Path("data/processed")
OUT_PARQUET = PROC / "consumables_forecasts.parquet"
OUT_PREVIEW = PROC / "consumables_forecasts_preview.csv"

RESETS_PATH = PROC / "consumables_with_resets.parquet"
SERIAL_REL = PROC / "serial_relations.parquet"
RULES_YAML = Path("configs/rules.yaml")

EVAL = PROC / "eval"
OFFSETS_COLOR_JSON = EVAL / "offsets_by_color.json"
OFFSETS_COLMAKER_JSON = EVAL / "offsets_by_color_maker.json"
OFFSETS_GLOBAL_TXT = EVAL / "offset_global.txt"
OFFSETS_COMPANY_COLOR_JSON = EVAL / "offsets_by_company_color.json"


# ------------------------- utils -------------------------
def _to_dt(s, dayfirst=True):
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def _normalize_serial(s):
    return str(s).strip().upper() if pd.notna(s) else ""


def _read_rules():
    try:
        import yaml

        if RULES_YAML.exists():
            with open(RULES_YAML, "r", encoding="utf-8") as f:
                rules = yaml.safe_load(f) or {}
        else:
            rules = {}
    except Exception:
        rules = {}

    rules.setdefault("lead_time_days", 5)
    rules.setdefault("buffer_days", 3)
    rules.setdefault("horizon_cap_days", 300)

    c = rules.setdefault("consumables", {})
    c.setdefault("lookback_days", 30)
    c.setdefault("alt_lookback_days", 45)
    c.setdefault("min_points_for_slope", 8)
    c.setdefault("low_threshold_pct", 10)
    c.setdefault("min_abs_slope_pct_per_day", 0.08)
    return rules


def _linear_regression_days_pct(df):
    x = (df["date_update"] - df["date_update"].min()).dt.total_seconds() / 86400.0
    y = df["pct"].astype(float)
    n = len(df)
    if n < 2:
        return np.nan, n
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0:
        return np.nan, n
    slope = num / den
    return float(slope), n


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _make_empty_forecast_df() -> pd.DataFrame:
    cols = [
        "serial",
        "serial_norm",
        "company",
        "manufacturer",
        "relation_type",
        "color",
        "cycle_id",
        "points_used",
        "slope_pct_per_day",
        "level_now_pct",
        "days_left_est",
        "last_seen",
        "stockout_date",
        "offset_days",
        "reorder_date",
        "should_order_now",
    ]
    return pd.DataFrame(columns=cols)


# ------------------------- coeur : calcul sur un groupe -------------------------
def compute_forecast_one_group(
    g: pd.DataFrame,
    lookback_days: int,
    alt_lookback_days: int,
    min_points_for_slope: int,
    low_threshold: float,
    min_abs_slope: float,
    horizon_cap: int,
):
    g = g.sort_values("date_update")
    last_date = g["date_update"].max()

    # fenêtre lookback standard
    w = g[g["date_update"] >= (last_date - pd.Timedelta(days=lookback_days))].copy()
    w = w.dropna(subset=["pct"])
    w = w[(w["pct"] >= 0) & (w["pct"] <= 100)]

    # élargit si pas assez de points
    if len(w) < min_points_for_slope and len(g) >= min_points_for_slope:
        w = g[g["date_update"] >= (last_date - pd.Timedelta(days=alt_lookback_days))].copy()
        w = w.dropna(subset=["pct"])
        w = w[(w["pct"] >= 0) & (w["pct"] <= 100)]

    if len(w) < min_points_for_slope:
        level_now = float(w["pct"].iloc[-1]) if len(w) else np.nan
        return {
            "points_used": int(len(w)),
            "slope_pct_per_day": np.nan,
            "level_now_pct": round(level_now, 1) if pd.notna(level_now) else np.nan,
            "days_left_est": np.nan,
            "stockout_date": pd.NaT,
        }

    slope, _ = _linear_regression_days_pct(w)
    level_now = float(w["pct"].iloc[-1])

    if (not pd.notna(slope)) or (slope >= 0) or (abs(slope) < min_abs_slope):
        return {
            "points_used": int(len(w)),
            "slope_pct_per_day": np.nan,
            "level_now_pct": round(level_now, 1),
            "days_left_est": np.nan,
            "stockout_date": pd.NaT,
        }

    effective_drop = max(level_now - low_threshold, 0.0)
    days_left = 0.0 if effective_drop == 0 else (effective_drop / abs(slope))
    days_left = float(min(days_left, float(horizon_cap)))
    stockout = (last_date + pd.Timedelta(days=days_left)).normalize()

    return {
        "points_used": int(len(w)),
        "slope_pct_per_day": float(round(slope, 4)),
        "level_now_pct": float(round(level_now, 1)),
        "days_left_est": float(round(days_left, 1)),
        "stockout_date": stockout,
    }


# ------------------------- main -------------------------
def main():
    if not RESETS_PATH.exists():
        raise SystemExit(f"Missing {RESETS_PATH} — run detect_resets first.")

    # 1) paramètres
    rules = _read_rules()
    lead_time = _safe_int(rules.get("lead_time_days", 5), 5)
    buffer = _safe_int(rules.get("buffer_days", 3), 3)
    horizon_cap = _safe_int(rules.get("horizon_cap_days", 300), 300)

    c = rules.get("consumables", {})
    lookback_days = _safe_int(c.get("lookback_days", 30), 30)
    alt_lookback_days = _safe_int(c.get("alt_lookback_days", 45), 45)
    min_pts_for_slope = _safe_int(c.get("min_points_for_slope", 8), 8)
    low_threshold = float(c.get("low_threshold_pct", 10))
    min_abs_slope = float(c.get("min_abs_slope_pct_per_day", 0.08))

    # 2) offsets appris
    offsets_by_company_color = {}
    offsets_by_color = {}
    offsets_by_colmaker = {}
    offset_global = 0

    try:
        if OFFSETS_COMPANY_COLOR_JSON.exists():
            offsets_by_company_color = json.loads(
                OFFSETS_COMPANY_COLOR_JSON.read_text(encoding="utf-8")
            )
    except Exception as e:
        print("[compute_slopes] WARN: cannot read offsets_by_company_color:", e)

    try:
        if OFFSETS_COLOR_JSON.exists():
            offsets_by_color = json.loads(OFFSETS_COLOR_JSON.read_text(encoding="utf-8"))
    except Exception as e:
        print("[compute_slopes] WARN: cannot read offsets_by_color:", e)

    try:
        if OFFSETS_COLMAKER_JSON.exists():
            offsets_by_colmaker = json.loads(
                OFFSETS_COLMAKER_JSON.read_text(encoding="utf-8")
            )
    except Exception as e:
        print("[compute_slopes] WARN: cannot read offsets_by_color_maker:", e)

    try:
        if OFFSETS_GLOBAL_TXT.exists():
            offset_global = _safe_int(
                OFFSETS_GLOBAL_TXT.read_text(encoding="utf-8").strip(), 0
            )
    except Exception as e:
        print("[compute_slopes] WARN: cannot read offset_global:", e)

    # 3) charge data resets
    df = pd.read_parquet(RESETS_PATH)

    # 🔥 CAS 0 LIGNE : on écrit un parquet vide et on sort proprement
    if df.empty:
        print("[compute_slopes] WARN: resets file is empty -> no forecasts.")
        fc = _make_empty_forecast_df()
        fc.to_parquet(OUT_PARQUET, index=False)
        fc.head(0).to_csv(OUT_PREVIEW, index=False, encoding="utf-8")
        print("[compute_slopes] forecasts saved. Rows: 0 (should_order_now True: 0)")
        print(f"Preview: {OUT_PREVIEW}")
        return

    need = {"serial", "color", "date_update", "pct", "cycle_id"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[compute_slopes] Missing columns in resets: {miss}")

    df["date_update"] = _to_dt(df["date_update"])
    df = df.dropna(subset=["serial", "color", "date_update"]).copy()
    df["serial_norm"] = df["serial"].map(_normalize_serial)
    df["color"] = df["color"].astype(str).str.strip().str.lower()

    # enrichissement via serial_relations
    if SERIAL_REL.exists():
        rel = pd.read_parquet(SERIAL_REL)

        rename_map = {
            "No. serie": "serial",
            "No serie": "serial",
            "Societe": "company",
        }
        for k, v in rename_map.items():
            if k in rel.columns and v not in rel.columns:
                rel = rel.rename(columns={k: v})

        rel["serial_norm"] = rel["serial"].map(_normalize_serial)
        keep = [
            c
            for c in ["serial_norm", "company", "relation_type", "manufacturer"]
            if c in rel.columns
        ]
        rel = rel[keep].drop_duplicates("serial_norm")
        df = df.merge(rel, on="serial_norm", how="left")

    # 5) périmètre contrats avec fallback
    if "relation_type" in df.columns:
        before = len(df)
        mask_contract = df["relation_type"].eq("contrat_adex")
        n_contract = int(mask_contract.sum())

        if n_contract > 0:
            df = df[mask_contract].copy()
            print(
                f"[compute_slopes] contract filter: {before} -> {len(df)} rows (contrat_adex only)"
            )
        else:
            print(
                f"[compute_slopes] WARN: no contrat_adex rows found in resets "
                f"({before} lignes). On garde tout le monde."
            )

    # 🔥 CAS : après filtrage, plus rien
    if df.empty:
        print("[compute_slopes] WARN: no data left after filtering -> no forecasts.")
        fc = _make_empty_forecast_df()
        fc.to_parquet(OUT_PARQUET, index=False)
        fc.head(0).to_csv(OUT_PREVIEW, index=False, encoding="utf-8")
        print("[compute_slopes] forecasts saved. Rows: 0 (should_order_now True: 0)")
        print(f"Preview: {OUT_PREVIEW}")
        return

    # 6) groupby
    group_cols = ["serial_norm", "color", "cycle_id"]
    if "company" in df.columns:
        group_cols.insert(1, "company")

    records = []
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("date_update")

        out = compute_forecast_one_group(
            g,
            lookback_days=lookback_days,
            alt_lookback_days=alt_lookback_days,
            min_points_for_slope=min_pts_for_slope,
            low_threshold=low_threshold,
            min_abs_slope=min_abs_slope,
            horizon_cap=horizon_cap,
        )

        rec = dict(zip(group_cols, keys))
        rec["serial"] = g["serial"].iloc[-1]

        if "manufacturer" in df.columns:
            rec["manufacturer"] = (
                g["manufacturer"].dropna().iloc[-1]
                if g["manufacturer"].notna().any()
                else None
            )
        if "relation_type" in df.columns:
            rec["relation_type"] = (
                g["relation_type"].dropna().iloc[-1]
                if g["relation_type"].notna().any()
                else None
            )
        if "company" in df.columns:
            rec["company"] = (
                g["company"].dropna().iloc[-1]
                if g["company"].notna().any()
                else None
            )

        rec.update(out)
        rec["last_seen"] = g["date_update"].max()
        records.append(rec)

    # 🔥 CAS : aucun groupe n’a donné de forecast exploitable
    if not records:
        print("[compute_slopes] WARN: no records produced -> empty forecast.")
        fc = _make_empty_forecast_df()
        fc.to_parquet(OUT_PARQUET, index=False)
        fc.head(0).to_csv(OUT_PREVIEW, index=False, encoding="utf-8")
        print("[compute_slopes] forecasts saved. Rows: 0 (should_order_now True: 0)")
        print(f"Preview: {OUT_PREVIEW}")
        return

    fc = pd.DataFrame(records)

    # 7) offsets -> offset_days
    def _resolve_offset(row):
        col = str(row.get("color", "")).lower()
        comp = str(row.get("company", "")).strip()
        maker = str(row.get("manufacturer", "")).strip()

        if comp and comp in offsets_by_company_color and col in offsets_by_company_color[comp]:
            return _safe_int(offsets_by_company_color[comp][col], offset_global)

        if maker and maker in offsets_by_colmaker and col in offsets_by_colmaker[maker]:
            return _safe_int(offsets_by_colmaker[maker][col], offset_global)

        if col in offsets_by_color:
            return _safe_int(offsets_by_color[col], offset_global)

        return offset_global

    fc["offset_days"] = fc.apply(_resolve_offset, axis=1).astype("Int64")

    # 8) reorder_date & should_order_now
    fc["reorder_date"] = pd.NaT
    mask_valid = fc["stockout_date"].notna() & fc["offset_days"].notna()
    fc.loc[mask_valid, "reorder_date"] = (
        fc.loc[mask_valid, "stockout_date"]
        - pd.to_timedelta(fc.loc[mask_valid, "offset_days"].astype(int), unit="D")
    )

    fc["last_seen"] = pd.to_datetime(fc["last_seen"], errors="coerce")

    fc["should_order_now"] = (
        fc["reorder_date"].notna()
        & fc["last_seen"].notna()
        & (
            fc["reorder_date"]
            <= (fc["last_seen"] + pd.to_timedelta(lead_time + buffer, unit="D"))
        )
        & (fc["level_now_pct"].fillna(101) <= 60)
    )

    # 9) ordre des colonnes
    ordered_cols = [
        "serial",
        "serial_norm",
        "company" if "company" in fc.columns else None,
        "manufacturer" if "manufacturer" in fc.columns else None,
        "relation_type" if "relation_type" in fc.columns else None,
        "color",
        "cycle_id",
        "points_used",
        "slope_pct_per_day",
        "level_now_pct",
        "days_left_est",
        "last_seen",
        "stockout_date",
        "offset_days",
        "reorder_date",
        "should_order_now",
    ]
    ordered_cols = [c for c in ordered_cols if c and c in fc.columns]
    fc = fc[ordered_cols].sort_values(["serial_norm", "color", "cycle_id"])

    # 10) sauvegarde
    fc.to_parquet(OUT_PARQUET, index=False)
    fc.head(100).to_csv(OUT_PREVIEW, index=False, encoding="utf-8")

    n_true = int(fc["should_order_now"].sum()) if "should_order_now" in fc.columns else 0
    print(
        f"[compute_slopes] forecasts saved. Rows: {len(fc)}  (should_order_now True: {n_true})"
    )
    print(f"Preview: {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
