import pandas as pd
from pathlib import Path
import numpy as np

DATA = Path("data/processed")
OUT = DATA / "ml/dataset_v21.parquet"


# -----------------------
# LOAD FUNCTIONS
# -----------------------
def load_forecasts():
    return pd.read_parquet(DATA / "consumables_forecasts.parquet")


def load_resets():
    return pd.read_parquet(DATA / "consumables_with_resets.parquet")


def load_meters():
    return pd.read_parquet(DATA / "meters.parquet")


def load_ledger():
    return pd.read_parquet(DATA / "item_ledger.parquet")


def load_serial_rel():
    return pd.read_parquet(DATA / "serial_relations.parquet")


# -----------------------
# FEATURE BLOCKS
# -----------------------
def feat_cycle_stats(resets: pd.DataFrame) -> pd.DataFrame:
    """
    Stats de cycle par (serial_norm, color) basées sur consumables_with_resets :
      - rs_n_cycles
      - rs_mean_cycle_len_days
      - rs_max_cycle_len_days
      - rs_days_since_last_reset
      - rs_has_reset
    """
    if resets is None or len(resets) == 0:
        print("[v21] feat_cycle_stats: resets vide -> aucun feature.")
        return pd.DataFrame()

    df = resets.copy()

    # safe types
    df["serial"] = df["serial"].astype(str)
    df["color"] = df["color"].astype(str)
    df["serial_norm"] = df["serial"].str.upper().str.strip()
    df["date_update"] = pd.to_datetime(df["date_update"], errors="coerce")

    # on garde uniquement les lignes valides
    df = df.dropna(subset=["serial_norm", "color", "date_update"])

    # ---- 1) Durée de cycle par (serial_norm, color, cycle_id)
    cycle_grp = df.groupby(["serial_norm", "color", "cycle_id"], dropna=False)

    cycle_agg = (
        cycle_grp["date_update"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "cycle_start", "max": "cycle_end", "count": "n_points"})
    )

    cycle_agg["cycle_len_days"] = (
        (cycle_agg["cycle_end"] - cycle_agg["cycle_start"]).dt.days.clip(lower=0) + 1
    )

    # ---- 2) Agrégation par (serial_norm, color)
    sc_grp = cycle_agg.groupby(["serial_norm", "color"], dropna=False)

    agg_cycles = sc_grp.agg(
        rs_n_cycles=("cycle_id", "nunique"),
        rs_mean_cycle_len_days=("cycle_len_days", "mean"),
        rs_max_cycle_len_days=("cycle_len_days", "max"),
    ).reset_index()

    # ---- 3) Dernier reset + jours depuis le dernier reset
    # last_seen par (serial_norm, color)
    last_seen = (
        df.groupby(["serial_norm", "color"])["date_update"]
        .max()
        .reset_index()
        .rename(columns={"date_update": "rs_last_seen"})
    )

    # dernière date de reset (is_reset = True)
    mask_reset = df.get("is_reset", pd.Series(False, index=df.index))
    df_reset = df[mask_reset == True]

    if len(df_reset) > 0:
        last_reset = (
            df_reset.groupby(["serial_norm", "color"])["date_update"]
            .max()
            .reset_index()
            .rename(columns={"date_update": "rs_last_reset"})
        )
    else:
        last_reset = pd.DataFrame(
            columns=["serial_norm", "color", "rs_last_reset"]
        )

    # merge last_seen + last_reset
    tmp = last_seen.merge(last_reset, on=["serial_norm", "color"], how="left")

    tmp["rs_days_since_last_reset"] = (
        (tmp["rs_last_seen"] - tmp["rs_last_reset"])
        .dt.days.replace({np.nan: np.nan})
    )

    # flag binaire
    tmp["rs_has_reset"] = tmp["rs_last_reset"].notna().astype(int)

    # on ne garde que les colonnes utiles
    tmp = tmp[
        ["serial_norm", "color", "rs_last_seen", "rs_last_reset", "rs_days_since_last_reset", "rs_has_reset"]
    ]

    # ---- 4) Merge cycles + resets
    out = agg_cycles.merge(tmp, on=["serial_norm", "color"], how="left")

    return out


def feat_usage_speed(resets: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la vitesse moyenne réelle d'utilisation par (serial_norm, color).
    Sortie :
      - us_mean_daily_pct_drop
      - us_median_daily_pct_drop
      - us_peak_daily_pct_drop
      - us_mean_pct_drop_per_cycle
      - us_has_usage_data
    """
    if resets is None or len(resets) == 0:
        print("[v21] feat_usage_speed: resets vide → aucun feature.")
        return pd.DataFrame()

    df = resets.copy()
    df["serial_norm"] = df["serial"].astype(str).str.upper().str.strip()
    df["color"] = df["color"].astype(str)
    df["date_update"] = pd.to_datetime(df["date_update"], errors="coerce")
    df = df.dropna(subset=["serial_norm", "color", "date_update", "pct"])

    # tri
    df = df.sort_values(["serial_norm", "color", "date_update"])

    # --- daily delta (pct lost / day)
    df["pct_shift"] = df.groupby(["serial_norm", "color"])["pct"].shift(1)
    df["date_shift"] = df.groupby(["serial_norm", "color"])["date_update"].shift(1)

    df["day_delta"] = (df["date_update"] - df["date_shift"]).dt.days
    df["pct_delta"] = df["pct_shift"] - df["pct"]  # perte de % : positive = consommation

    # on ne garde que les deltas positifs et plausibles
    valid = df[
        (df["day_delta"] > 0) &
        (df["day_delta"] <= 14) &          # sécurité : exclure les très gros trous
        (df["pct_delta"] >= 0) &
        (df["pct_delta"] <= 100)
    ].copy()

    # vitesse quotidienne
    valid["pct_per_day"] = valid["pct_delta"] / valid["day_delta"]

    # --- agreg par (serial, color)
    grp = valid.groupby(["serial_norm", "color"])

    usage = grp.agg(
        us_mean_daily_pct_drop=("pct_per_day", "mean"),
        us_median_daily_pct_drop=("pct_per_day", "median"),
        us_peak_daily_pct_drop=("pct_per_day", "max"),
        us_n_points=("pct_per_day", "count"),
    ).reset_index()

    # Cycle-based consumption (optionnel mais utile)
    cycle_grp = df.groupby(["serial_norm", "color", "cycle_id"])
    cycle_cons = (
        cycle_grp["pct"]
        .agg(["first", "last"])
        .reset_index()
        .rename(columns={"first": "pct_start", "last": "pct_end"})
    )
    cycle_cons["cycle_pct_used"] = (cycle_cons["pct_start"] - cycle_cons["pct_end"]).clip(lower=0)

    cycle_stats = (
        cycle_cons.groupby(["serial_norm", "color"])["cycle_pct_used"]
        .mean()
        .reset_index()
        .rename(columns={"cycle_pct_used": "us_mean_pct_drop_per_cycle"})
    )

    # merge usage + cycle stats
    out = usage.merge(cycle_stats, on=["serial_norm", "color"], how="left")

    # flag des machines avec au moins 2 points valides
    out["us_has_usage_data"] = (out["us_n_points"] >= 2).astype(int)

    return out[
        [
            "serial_norm",
            "color",
            "us_mean_daily_pct_drop",
            "us_median_daily_pct_drop",
            "us_peak_daily_pct_drop",
            "us_mean_pct_drop_per_cycle",
            "us_has_usage_data",
        ]
    ]



def feat_meters(meters: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features pages imprimées par serial_norm.
      - volume last 7/30 days
      - daily mean / std
      - ratios (color, A3)
    """
    if meters is None or len(meters) == 0:
        print("[v21] feat_meters: meters vide → aucun feature.")
        return pd.DataFrame()

    df = meters.copy()

    # ---------- 1) Trouver / construire serial_norm ----------
    cols_lower = {c: c.lower() for c in df.columns}

    # si serial_norm déjà là, on la garde
    if "serial_norm" in df.columns:
        serial_src = "serial_norm"
    else:
        # sinon on cherche une colonne type "No serie", "$No serie$", etc.
        candidates = [
            c for c in df.columns
            if "serie" in c.lower() and ("no" in c.lower() or "n°" in c.lower() or "n " in c.lower())
        ]
        if not candidates:
            print("[v21] feat_meters: aucune colonne série trouvée → skip block meters.")
            return pd.DataFrame()
        serial_src = candidates[0]

    df["serial_norm"] = df[serial_src].astype(str).str.upper().str.strip()

    # ---------- 2) Date de relevé ----------
    # on cherche une colonne de date type "Date releve"
    date_col = None
    for c in df.columns:
        n = c.lower()
        if "releve" in n and "date" in n:
            date_col = c
            break
    if date_col is None:
        print("[v21] feat_meters: aucune colonne de date de relevé trouvée → skip block meters.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["serial_norm", "date"])

    # ---------- 3) Création des deltas de pages ----------
    meter_pairs = [
        ("Debut A4 NB", "Fin A4 NB", "pages_a4_nb"),
        ("Debut A4 CO", "Fin A4 CO", "pages_a4_color"),
        ("Debut A3 NB", "Fin A3 NB", "pages_a3_nb"),
        ("Debut A3 CO", "Fin A3 CO", "pages_a3_color"),
    ]

    for deb, fin, newcol in meter_pairs:
        if deb in df.columns and fin in df.columns:
            df[newcol] = (
                pd.to_numeric(df[fin], errors="coerce")
                - pd.to_numeric(df[deb], errors="coerce")
            ).clip(lower=0)
        else:
            df[newcol] = 0

    df["pages_total"] = (
        df["pages_a4_nb"]
        + df["pages_a4_color"]
        + df["pages_a3_nb"]
        + df["pages_a3_color"]
    )

    df["ratio_color"] = df["pages_a4_color"] / df["pages_total"].replace(0, np.nan)
    df["ratio_a3"] = (df["pages_a3_nb"] + df["pages_a3_color"]) / df["pages_total"].replace(0, np.nan)

    grp = df.groupby("serial_norm")

    out = grp.agg(
        m_total_pages=("pages_total", "sum"),
        m_n_records=("pages_total", "count"),
        m_ratio_color=("ratio_color", "mean"),
        m_ratio_a3=("ratio_a3", "mean"),
        m_daily_mean=("pages_total", "mean"),
        m_daily_std=("pages_total", "std"),
    ).reset_index()

    max_date = df["date"].max()
    df_30 = df[df["date"] >= max_date - pd.Timedelta(days=30)]
    df_7 = df[df["date"] >= max_date - pd.Timedelta(days=7)]

    out30 = df_30.groupby("serial_norm")["pages_total"].sum().reset_index().rename(
        columns={"pages_total": "m_total_pages_last30"}
    )
    out7 = df_7.groupby("serial_norm")["pages_total"].sum().reset_index().rename(
        columns={"pages_total": "m_total_pages_last7"}
    )

    out = out.merge(out30, on="serial_norm", how="left")
    out = out.merge(out7, on="serial_norm", how="left")

    return out




def feat_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """(sera rempli ensuite)"""
    return pd.DataFrame()


def feat_kpax_quality(resets: pd.DataFrame) -> pd.DataFrame:
    """(sera rempli ensuite)"""
    return pd.DataFrame()


def feat_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month / dayofweek / week based on last_seen."""
    d = df.copy()
    d["last_seen"] = pd.to_datetime(d["last_seen"], errors="coerce")
    d["month"] = d["last_seen"].dt.month
    d["dayofweek"] = d["last_seen"].dt.dayofweek
    d["week"] = d["last_seen"].dt.isocalendar().week.astype(int)
    # on met aussi serial_norm ici au cas où
    d["serial_norm"] = d["serial"].str.upper().str.strip()
    return d[["serial_norm", "color", "month", "dayofweek", "week"]]


# -----------------------
# BUILD MAIN DATASET
# -----------------------
def main():
    print("[v21] Loading sources…")

    fc = load_forecasts().copy()
    rs = load_resets()
    mt = load_meters()
    led = load_ledger()
    rel = load_serial_rel()


    print(f"[v21] forecasts: {len(fc)}")
    print(f"[v21] resets: {len(rs)}")
    print(f"[v21] meters: {len(mt)}")
    print(f"[v21] ledger: {len(led)}")

    # Normalisation du serial
    fc["serial_norm"] = fc["serial"].str.upper().str.strip()
    fc["last_seen"] = pd.to_datetime(fc["last_seen"], errors="coerce")

    # ------------------------
    # MERGE 1 : Time features
    # ------------------------
    time_features = feat_time_features(fc)
    df = fc.merge(time_features, on=["serial_norm", "color"], how="left")

    # ------------------------
    # MERGE 2 : Cycle stats
    # ------------------------
    cycle_stats = feat_cycle_stats(rs)
    if len(cycle_stats) > 0:
        df = df.merge(cycle_stats, on=["serial_norm", "color"], how="left")

    # ------------------------
    # MERGE 3 : Usage speed (à venir)
    # ------------------------
    usage = feat_usage_speed(rs)
    if len(usage) > 0:
        df = df.merge(usage, on=["serial_norm", "color"], how="left")

    # ------------------------
    # MERGE 4 : Meters (à venir)
    # ------------------------
    meters_f = feat_meters(mt)
    if len(meters_f) > 0:
        df = df.merge(meters_f, on="serial_norm", how="left")

    # ------------------------
    # MERGE 5 : Ledger (à venir)
    # ------------------------
    ledger_f = feat_ledger(led)
    if len(ledger_f) > 0:
        df = df.merge(ledger_f, on="serial_norm", how="left")

    # ------------------------
    # MERGE 6 : KPAX Quality (à venir)
    # ------------------------
    kpax_q = feat_kpax_quality(rs)
    if len(kpax_q) > 0:
        df = df.merge(kpax_q, on=["serial_norm", "color"], how="left")

    print(f"[v21] Final dataset shape: {df.shape}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[v21] Saved -> {OUT}")


if __name__ == "__main__":
    main()
