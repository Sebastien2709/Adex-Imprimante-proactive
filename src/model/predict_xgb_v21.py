from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# On réutilise les fonctions de feature engineering de build_dataset_v21
from .build_dataset_v21 import (
    feat_time_features,
    feat_cycle_stats,
    feat_usage_speed,
    feat_meters,
    feat_ledger,
    feat_kpax_quality,
)

def norm_serial(s):
    if s is None:
        return ""
    return str(s).upper().strip()


DATA_PROCESSED = Path("data/processed")
ML_DATA = DATA_PROCESSED / "ml"
MODEL_DIR = Path("models")


def load_forecasts_current() -> pd.DataFrame:
    """
    Charge les forecasts V1 (pentes) actuels sur lesquels on veut appliquer le modèle V2.1.
    On part de consumables_forecasts.parquet (output de compute_slopes).
    """
    path = DATA_PROCESSED / "consumables_forecasts.parquet"
    df = pd.read_parquet(path).copy()
    if "serial_norm" not in df.columns:
        df["serial_norm"] = df["serial"].astype(str).str.upper().str.strip()
    return df


def main():
    # 1) Charger les données nécessaires
    fc = load_forecasts_current()
    resets = pd.read_parquet(DATA_PROCESSED / "consumables_with_resets.parquet")
    meters = pd.read_parquet(DATA_PROCESSED / "meters.parquet")
    ledger = pd.read_parquet(DATA_PROCESSED / "item_ledger.parquet")

    print("[predict_xgb_v21] forecasts rows:", len(fc))
    print("[predict_xgb_v21] resets rows:", len(resets))
    print("[predict_xgb_v21] meters rows:", len(meters))
    print("[predict_xgb_v21] ledger rows:", len(ledger))

    # --- 2) Features avancées (même logique que dataset_v21) ---

    # Time-features (mois, jour_semaine, etc.) basés sur last_seen
    time_feats = feat_time_features(fc)

    # Stats de cycles & vitesse d’usage à partir des resets
    cycle_stats = feat_cycle_stats(resets)
    usage_speed = feat_usage_speed(resets)

    # Features issus des compteurs de pages (meters)
    mt_feats = feat_meters(meters)

    # Features issus de l’historique des commandes (item_ledger)
    ld_feats = feat_ledger(ledger)

    # Qualité KPAX (machine “fiable” ou pas)
    kpax_q = feat_kpax_quality(resets)

    # --- 3) Merge de toutes les features sur les forecasts ---

    df = fc.copy()

    # sécurité : s'assurer que serial_norm existe et est propre
    if "serial_norm" not in df.columns:
        df["serial_norm"] = df["serial"].astype(str).str.upper().str.strip()

    # Mêmes joins que dans build_dataset_v21 (mais sans backtest)
    df = df.merge(cycle_stats, on=["serial_norm", "color"], how="left")
    df = df.merge(usage_speed, on=["serial_norm", "color"], how="left")
    df = df.merge(mt_feats, on="serial_norm", how="left")

    # MERGE ledger features (optionnel)
    if ld_feats is not None and not ld_feats.empty and "serial_norm" in ld_feats.columns:
        df = df.merge(ld_feats, on="serial_norm", how="left")
        print(f"[predict_xgb_v21] ledger features merged: {ld_feats.shape[1] - 1} cols")
    else:
        print("[predict_xgb_v21] WARN: no ledger features available, skipping ledger merge.")

    df = df.merge(time_feats, on=["serial_norm", "color"], how="left")

    # MERGE KPAX quality features (safety check)
    if (
        kpax_q is not None
        and not kpax_q.empty
        and "serial_norm" in kpax_q.columns
        and "color" in kpax_q.columns
    ):
        df = df.merge(kpax_q, on=["serial_norm", "color"], how="left")
        print(f"[predict_xgb_v21] kpax quality merged: {kpax_q.shape[1]} cols")
    else:
        print("[predict_xgb_v21] WARN: no KPAX quality data available → skipping merge.")

    print("[predict_xgb_v21] shape after feature merge:", df.shape)

    # --- 4) Charger le modèle V2.1 et préparer X ---

    bundle_path = MODEL_DIR / "xgb_offset_model_v21.pkl"
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Créer les colonnes manquantes éventuelles (au cas où)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].copy()

    # Cast en numérique + remplacement NaN
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- 5) Prédire les offsets ---

    print("[predict_xgb_v21] prédiction des offsets V2.1…")
    df["ml_offset_days_v21"] = model.predict(X)

    # --- 6) Calcul des dates de rupture + days_left ML ---

    # Base pour la date de rupture :
    # - si tu as stockout_date ou stockout_date_base, on l'utilise
    # - sinon fallback : last_seen + days_left_est
    base_col = None
    for candidate in ["stockout_date_base", "stockout_date"]:
        if candidate in df.columns:
            base_col = candidate
            break

    if base_col is None:
        # fallback
        df["last_seen_dt"] = pd.to_datetime(df["last_seen"], errors="coerce")
        if "days_left_est" in df.columns:
            base_base_days = pd.to_numeric(df["days_left_est"], errors="coerce").fillna(0)
        else:
            base_base_days = 0
        df["stockout_date_base_tmp"] = df["last_seen_dt"] + pd.to_timedelta(
            base_base_days, unit="D"
        )
        base_col = "stockout_date_base_tmp"
    else:
        df["last_seen_dt"] = pd.to_datetime(df["last_seen"], errors="coerce")

    df[base_col] = pd.to_datetime(df[base_col], errors="coerce")

    # Nouvelle date de rupture ML V2.1
    df["stockout_date_ml_v21"] = df[base_col] + pd.to_timedelta(
        df["ml_offset_days_v21"], unit="D"
    )

    # Recalcul des days_left depuis last_seen
    df["days_left_ml_v21"] = (
        df["stockout_date_ml_v21"] - df["last_seen_dt"]
    ).dt.days
    df["days_left_ml_v21"] = df["days_left_ml_v21"].clip(lower=0)

    # --- 6.bis) Snapshot + recommandations métier ---

    today = pd.Timestamp.today().normalize()
    df["reference_date"] = df["last_seen_dt"].fillna(today)

    # On ne modifie pas df pour la sauvegarde détaillée : on travaille sur une copie triée
    df_sorted = df.sort_values(["serial_norm", "color", "reference_date"])
    snapshot = df_sorted.groupby(["serial_norm", "color"]).tail(1).copy()

    # Calcul des jours restants “aujourd’hui” (vision métier)
    snapshot["days_left_v21"] = (
        snapshot["stockout_date_ml_v21"] - today
    ).dt.days
    snapshot["days_left_v21"] = snapshot["days_left_v21"].clip(lower=0)

    # --- 7) Sauvegardes ---

    ML_DATA.mkdir(parents=True, exist_ok=True)

    # a) Toutes les prédictions détaillées (historisées)
    out_parq = ML_DATA / "consumables_forecasts_ml_v21.parquet"
    out_csv = ML_DATA / "consumables_forecasts_ml_v21_preview.csv"

    df.to_parquet(out_parq, index=False)
    df.head(500).to_csv(out_csv, index=False)

    print(f"[predict_xgb_v21] saved -> {out_parq}")
    print(f"[predict_xgb_v21] preview -> {out_csv}")
    print(f"[predict_xgb_v21] lignes: {len(df)}")

    # b) Snapshot (une ligne par toner / machine)
    snapshot_path = ML_DATA / "consumables_forecasts_ml_v21_snapshot.parquet"
    snapshot.to_parquet(snapshot_path, index=False)
    print(f"[predict_xgb_v21] snapshot saved -> {snapshot_path} ({len(snapshot)} lignes)")

    # c) Fichier métier : recommandations à envoyer (ex: <= 10 jours restants)
    THRESHOLD_DAYS = 10  # à ajuster avec ton tuteur métier
    recommend = snapshot[snapshot["days_left_v21"] <= THRESHOLD_DAYS].copy()

    reco_csv = DATA_PROCESSED / "replenishments_to_create_v21.csv"
    recommend.to_csv(reco_csv, sep=";", index=False)
    print(f"[predict_xgb_v21] recommandations V2.1 -> {reco_csv} ({len(recommend)} lignes)")


if __name__ == "__main__":
    main()
