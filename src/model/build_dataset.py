from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/processed")
EVAL = DATA / "eval"
OUT = DATA / "ml"
OUT.mkdir(parents=True, exist_ok=True)


def norm_serial(s):
    if not isinstance(s, str):
        return None
    return s.strip().upper()


# -----------------------------
# KPAX time-series features
# -----------------------------
def build_kpax_ts_features():
    kpax = pd.read_parquet(DATA / "consumables_with_resets.parquet").copy()

    # noms EXACTS confirmés par toi
    kpax["serial_norm"] = kpax["serial"].map(norm_serial)
    kpax["color"] = kpax["color"].astype(str).str.lower()

    # on prend date_import (ou date_update si besoin)
    kpax["date"] = pd.to_datetime(kpax["date_import"], errors="coerce")
    kpax["pct"] = pd.to_numeric(kpax["pct"], errors="coerce")

    kpax = kpax.dropna(subset=["serial_norm", "color", "date", "pct"])
    kpax = kpax.sort_values(["serial_norm", "color", "date"])

    # delta / jour
    kpax["pct_diff"] = kpax.groupby(["serial_norm", "color"])["pct"].diff()
    kpax["delta_days"] = kpax.groupby(["serial_norm", "color"])["date"].diff().dt.days

    kpax["rate_pct_per_day"] = kpax["pct_diff"] / kpax["delta_days"].replace(0, np.nan)

    def roll_mean(s, w):
        return s.rolling(w, min_periods=max(3, w // 3)).mean()

    # slopes récentes
    kpax["slope_7d"]  = kpax.groupby(["serial_norm", "color"])["rate_pct_per_day"].transform(lambda s: roll_mean(s, 7))
    kpax["slope_14d"] = kpax.groupby(["serial_norm", "color"])["rate_pct_per_day"].transform(lambda s: roll_mean(s, 14))
    kpax["slope_30d"] = kpax.groupby(["serial_norm", "color"])["rate_pct_per_day"].transform(lambda s: roll_mean(s, 30))

    # variance récente
    kpax["var_14d"] = kpax.groupby(["serial_norm", "color"])["rate_pct_per_day"].transform(
        lambda s: s.rolling(14, min_periods=5).var()
    )

    # âge du cycle + nb points
    kpax["cycle_age_days"] = kpax.groupby(["serial_norm", "color", "cycle_id"])["date"].transform(
        lambda s: (s.max() - s.min()).days
    )
    kpax["points_in_cycle"] = kpax.groupby(["serial_norm", "color", "cycle_id"])["date"].transform("count")

    # days since last reset
    last_reset_date = (
        kpax[kpax["is_reset"] == 1]
        .groupby(["serial_norm", "color"])["date"]
        .max()
    )
    kpax["last_reset_date"] = kpax.set_index(["serial_norm", "color"]).index.map(last_reset_date)
    kpax["days_since_reset"] = (kpax["date"] - kpax["last_reset_date"]).dt.days

    # dernière ligne par série/couleur
    last = kpax.groupby(["serial_norm", "color"]).tail(1)

    feats = last[[
        "serial_norm", "color",
        "slope_7d", "slope_14d", "slope_30d",
        "var_14d",
        "cycle_age_days", "points_in_cycle",
        "days_since_reset",
        "pct"  # niveau actuel
    ]].copy()

    return feats


# -----------------------------
# Ledger history features
# -----------------------------
def build_ledger_features():
    led = pd.read_parquet(DATA / "item_ledger.parquet").copy()

    # --- détecter la colonne du numéro de série ---
    serial_col_candidates = ["serial", "No. serie", "no_serie", "no.serie", "no_serie_norm"]
    serial_col = None
    for c in led.columns:
        if c.lower().replace(" ", "") in [s.lower().replace(" ", "") for s in serial_col_candidates]:
            serial_col = c
            break
    if serial_col is None:
        raise ValueError(f"Aucune colonne série trouvée dans ledger: {led.columns}")

    led["serial_norm"] = led[serial_col].astype(str).map(norm_serial)

    # --- détecter la date de compta ---
    date_col_candidates = ["doc_date", "Date compta", "date_compta"]
    date_col = None
    for c in led.columns:
        if c.lower().replace(" ", "") in [s.lower().replace(" ", "") for s in date_col_candidates]:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Aucune colonne date trouvée dans ledger: {led.columns}")

    led["doc_date"] = pd.to_datetime(led[date_col], errors="coerce", dayfirst=True)

    # --- type consommable -> couleur ---
    type_col_candidates = ["consumable_type", "Type conso"]
    type_col = None
    for c in led.columns:
        if c.lower().replace(" ", "") in [s.lower().replace(" ", "") for s in type_col_candidates]:
            type_col = c
            break

    if type_col is None:
        raise ValueError(f"Aucune colonne type conso trouvée: {led.columns}")

    t = led[type_col].astype(str).str.lower()

    led["color"] = np.select(
        [
            t.str.contains("noir|black"),
            t.str.contains("cyan"),
            t.str.contains("magenta"),
            t.str.contains("jaune|yellow"),
        ],
        ["black", "cyan", "magenta", "yellow"],
        default=None,
    )

    # nettoyage minimal
    led = led.dropna(subset=["serial_norm", "doc_date", "color"])
    led = led.sort_values(["serial_norm", "color", "doc_date"])

    # intervalle entre commandes
    led["prev_ship"] = led.groupby(["serial_norm", "color"])["doc_date"].shift(1)
    led["ship_interval"] = (led["doc_date"] - led["prev_ship"]).dt.days

    agg = led.groupby(["serial_norm", "color"]).agg(
        last_ship_date=("doc_date", "max"),
        mean_ship_interval=("ship_interval", "mean"),
        n_ships_180d=("doc_date", lambda s: (s >= (s.max() - pd.Timedelta(days=180))).sum()),
    ).reset_index()

    return agg



def main():
    forecasts = pd.read_parquet(DATA / "consumables_forecasts.parquet").copy()
    forecasts["serial_norm"] = forecasts["serial"].map(norm_serial)
    forecasts["color"] = forecasts["color"].astype(str).str.lower()

    # backtest results (target)
    bt = pd.read_csv(EVAL / "forecast_backtest_results.csv")
    bt["serial_norm"] = bt["serial"].map(norm_serial)
    bt["color"] = bt["color"].astype(str).str.lower()

    for c in bt.columns:
        if "date" in c.lower():
            bt[c] = pd.to_datetime(bt[c], errors="coerce", dayfirst=True)

    if "err_days" not in bt.columns:
        raise ValueError("Backtest doit contenir err_days (actual - predicted).")

    bt = bt[bt["err_days"].notna()].copy()

    merge_cols = ["serial_norm", "color"]

    df = forecasts.merge(bt[merge_cols + ["err_days"]], on=merge_cols, how="inner")
    print(f"[build_dataset] after merge forecasts+backtest: {df.shape}")

    # NEW: KPAX TS feats
    kpax_ts = build_kpax_ts_features()
    df = df.merge(kpax_ts, on=merge_cols, how="left")

    # NEW: Ledger feats
    led_feats = build_ledger_features()
    df = df.merge(led_feats, on=merge_cols, how="left")

    print(f"[build_dataset] after KPAX_ts + ledger_feats: {df.shape}")

    # select numeric features
    target_col = "err_days"
    id_cols = ["serial", "serial_norm", "color"]
    date_cols = [c for c in df.columns if "date" in c.lower() or "last_seen" in c.lower()]

    num_cols = []
    for c in df.columns:
        if c in id_cols or c == target_col or c in date_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    print(f"[build_dataset] numeric features: {len(num_cols)}")

    dataset = df[id_cols + date_cols + num_cols + [target_col]].copy()

    dataset.to_parquet(OUT / "dataset_xgb.parquet", index=False)
    dataset.head(200).to_csv(OUT / "dataset_xgb_preview.csv", index=False)

    print(f"[build_dataset] saved -> {OUT/'dataset_xgb.parquet'} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
