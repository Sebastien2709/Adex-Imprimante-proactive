# src/model/predict_xgb.py
# --------------------------------------------------
# Applique le modèle XGBoost d'offsets sur les
# forecasts actuels (consumables_forecasts.parquet).
# - Charge le bundle modèle (model + feature_cols)
# - Reconstruit X avec toutes les colonnes attendues
#   (celles manquantes sont mises à 0.0)
# - Prédit ml_offset_days
# - Calcule stockout_date_ml et days_left_ml
# - Sauvegarde dans data/processed/ml/...
# --------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import joblib


DATA = Path("data/processed")
ML_DATA = DATA / "ml"
MODEL_DIR = Path("models")


def _load_forecasts() -> pd.DataFrame:
    fc_path = DATA / "consumables_forecasts.parquet"
    if not fc_path.exists():
        raise SystemExit(f"[predict_xgb] ERREUR: fichier introuvable: {fc_path}")
    df = pd.read_parquet(fc_path).copy()
    # normalisation de base
    if "serial" in df.columns and "serial_norm" not in df.columns:
        df["serial_norm"] = df["serial"].astype(str).str.strip().str.upper()
    if "last_seen" in df.columns:
        df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce")
    return df


def _load_model_bundle():
    bundle_path = MODEL_DIR / "xgb_offset_model.pkl"
    if not bundle_path.exists():
        raise SystemExit(
            f"[predict_xgb] ERREUR: modèle non trouvé: {bundle_path}.\n"
            "Lance d'abord: python -m src.model.train_xgb (une fois)."
        )
    bundle = joblib.load(bundle_path)
    if "model" not in bundle or "feature_cols" not in bundle:
        raise SystemExit("[predict_xgb] ERREUR: bundle modèle invalide (clé 'model' ou 'feature_cols' manquante).")
    return bundle["model"], list(bundle["feature_cols"])


def _build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Construit une matrice X avec exactement les colonnes dans feature_cols.
    - Si une colonne existe dans df: cast numérique + NaN -> 0.0
    - Si elle n'existe pas: colonne pleine de 0.0
    """
    X_dict = {}
    n = len(df)
    for col in feature_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            X_dict[col] = s.fillna(0.0)
        else:
            # colonne manquante -> 0
            X_dict[col] = pd.Series(0.0, index=df.index)
    X = pd.DataFrame(X_dict, index=df.index)
    return X


def _compute_baseline_stockout(df: pd.DataFrame) -> pd.Series:
    """
    Définit une date de rupture de base (stockout_date_base) à partir de:
    - stockout_date si disponible
    - sinon last_seen + days_left_est
    """
    # 1) si stockout_date existe, on la prend
    if "stockout_date" in df.columns:
        base = pd.to_datetime(df["stockout_date"], errors="coerce")
    else:
        base = pd.Series(pd.NaT, index=df.index)

    # 2) fallback avec last_seen + days_left_est
    need_fallback = base.isna()
    if need_fallback.any():
        if "last_seen" in df.columns:
            last_seen = pd.to_datetime(df["last_seen"], errors="coerce")
        else:
            # si vraiment rien: on met "aujourd'hui"
            last_seen = pd.Series(pd.Timestamp("today").normalize(), index=df.index)

        if "days_left_est" in df.columns:
            dle = pd.to_numeric(df["days_left_est"], errors="coerce").fillna(0)
        else:
            dle = pd.Series(0, index=df.index)

        fallback = last_seen + pd.to_timedelta(dle, unit="D")
        base[need_fallback] = fallback[need_fallback]

    return base


def main():
    ML_DATA.mkdir(parents=True, exist_ok=True)

    # 1) charge forecasts actuels
    df = _load_forecasts()

    # 2) charge modèle + liste des features
    model, feature_cols = _load_model_bundle()

    # 3) construit X aligné sur feature_cols
    X = _build_feature_matrix(df, feature_cols)

    print("[predict_xgb] prédiction des offsets...")
    offset_pred = model.predict(X)
    df["ml_offset_days"] = pd.to_numeric(offset_pred, errors="coerce")

    # 4) stockout_date de base
    df["stockout_date_base"] = _compute_baseline_stockout(df)

    # 5) date de rupture ML = base + offset ML
    #    (offset peut être négatif si le modèle anticipe une rupture plus tôt)
    df["stockout_date_ml"] = df["stockout_date_base"] + pd.to_timedelta(
        df["ml_offset_days"].fillna(0), unit="D"
    )

    # 6) days_left_ml = (stockout_date_ml - last_seen).days, avec garde-fous
    if "last_seen" in df.columns:
        last_seen = pd.to_datetime(df["last_seen"], errors="coerce")
    else:
        last_seen = pd.Series(pd.Timestamp("today").normalize(), index=df.index)

    delta_days = (df["stockout_date_ml"] - last_seen).dt.days
    # Nettoyage : NaN -> 0, clamp [0, 730]
    delta_days = delta_days.fillna(0).clip(lower=0, upper=365 * 2)
    df["days_left_ml"] = delta_days.astype("Int64")

    # 7) sauvegarde
    out_parq = ML_DATA / "consumables_forecasts_ml.parquet"
    out_csv = ML_DATA / "consumables_forecasts_ml_preview.csv"
    df.to_parquet(out_parq, index=False)
    df.head(500).to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[predict_xgb] sauvegardé -> {out_parq}")
    print(f"[predict_xgb] preview -> {out_csv}")
    print(f"[predict_xgb] lignes: {len(df)}")


if __name__ == "__main__":
    main()
