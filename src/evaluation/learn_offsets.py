# ==============================================================
# learn_offsets.py
# ==============================================================
# Apprend les offsets (en jours) pour recentrer la médiane d'erreur à 0.
# Calculé sur les résultats du backtest : forecast_backtest_results.csv
# ==============================================================
import json
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------
# Chemins
# --------------------------------------------------------------
PROC_DIR = Path("data/processed")
EVAL_DIR = PROC_DIR / "eval"
IN_CSV   = EVAL_DIR / "forecast_backtest_results.csv"
OUT_COLOR = EVAL_DIR / "offsets_by_color.json"
OUT_CM    = EVAL_DIR / "offsets_by_color_maker.json"
OUT_TXT   = EVAL_DIR / "offset_global.txt"
OUT_CC = EVAL_DIR / "offsets_by_company_color.json"

# --------------------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------------------
def safe_offset(series: pd.Series) -> int:
    """
    Offset = median(err_days), borné pour éviter les extrêmes.
    err_days = ship_date_real - ref_date.
    Raison: err' = err - offset -> choisir offset = median(err) => median(err') ≈ 0.
    """
    s = series.dropna()
    if s.empty:
        return 0
    val = float(s.median())
    # bornes prudentes (à ajuster si besoin)
    return int(np.clip(round(val), -60, 60))

# --------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------
def main():
    if not IN_CSV.exists():
        raise SystemExit(f"❌ Missing {IN_CSV}. Run: python -m src.evaluation.backtest first.")

    # Lecture du CSV
    df = pd.read_csv(
        IN_CSV,
        parse_dates=["ship_date_real", "stockout_date", "reorder_date", "ref_date"],
        dtype={"color": "string"}
    )
    df = df[df["ship_date_real"].notna()].copy()
    if df.empty:
        print("[learn_offsets] ⚠️ No matched shipments in backtest CSV.")
        json.dump({}, open(OUT_COLOR, "w"))
        OUT_TXT.write_text("0")
        return

    # Déterminer la date de référence (priorité : ref_date > reorder_date > stockout_date)
    ref = df["ref_date"]
    if "reorder_date" in df.columns:
        ref = ref.where(ref.notna(), df["reorder_date"])
    ref = ref.where(ref.notna(), df["stockout_date"])

    # Calcul erreur en jours
    df["err_days"] = (df["ship_date_real"] - ref).dt.days
    print(f"[learn_offsets] matched rows: {len(df)}")

    # ----------------------------------------------------------
    # Offsets par couleur
    # ----------------------------------------------------------
    by_color = df.groupby(df["color"].astype(str).str.lower().fillna("unknown"))["err_days"] \
             .apply(safe_offset).to_dict()
    print("✅ Learned offsets (days) by color:", by_color)

    # ----------------------------------------------------------
    # Offset global
    # ----------------------------------------------------------
    global_off = safe_offset(df["err_days"])
    print("✅ Global offset (days):", global_off)

    # ----------------------------------------------------------
    # Offsets par couleur × fabricant (si disponible)
    # ----------------------------------------------------------
    by_cm = {}
    if "manufacturer" in df.columns:
        for (m, c), g in df.groupby(["manufacturer", "color"], dropna=False):
            by_cm.setdefault(m, {})
            by_cm[m][str(c).lower()] = safe_offset(g["err_days"])
        print(f"✅ Offsets by color×maker learned for {len(by_cm)} makers")
    
    # Offsets par company × color (si colonne présente)
    by_cc = {}
    if "company" in df.columns:
        for (comp, col), g in df.groupby(["company", "color"], dropna=False):
            by_cc.setdefault(str(comp), {})
            by_cc[str(comp)][str(col).lower()] = safe_offset(g["err_days"])
        print(f"✅ Offsets by company×color learned for {len(by_cc)} companies")


    # ----------------------------------------------------------
    # Sauvegarde
    # ----------------------------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_COLOR, "w", encoding="utf-8") as f:
        json.dump(by_color, f, ensure_ascii=False, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(str(global_off))
    if by_cm:
        with open(OUT_CM, "w", encoding="utf-8") as f:
            json.dump(by_cm, f, ensure_ascii=False, indent=2)
    if by_cc:
        with open(OUT_CC, "w", encoding="utf-8") as f:
            json.dump(by_cc, f, ensure_ascii=False, indent=2)


    print("💾 Saved:")
    print("   ", OUT_COLOR)
    print("   ", OUT_TXT)
    if by_cm:
        print("   ", OUT_CM)


if __name__ == "__main__":
    main()
