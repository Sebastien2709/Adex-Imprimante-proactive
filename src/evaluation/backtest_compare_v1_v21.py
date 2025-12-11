from pathlib import Path
import pandas as pd


DATA = Path("data/processed")
ML = DATA / "ml"


def normalize_serial(s):
    """Normalise le numéro de série pour les jointures."""
    if not isinstance(s, str):
        return None
    return str(s).strip().upper().replace(" ", "")


def pick_first_existing(df, candidates):
    """Retourne le premier nom de colonne existant dans df parmi candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_kpis(df, base_n=3100):
    """Retourne les KPI standardisés pour un tableau de backtest."""
    if len(df) == 0:
        return {
            "n_forecasts": 0,
            "n_matched": 0,
            "coverage_%": 0,
            "mae_days": None,
            "median_err": None,
            "jit_rate_%": 0,
            "early_%": 0,
            "late_%": 0,
        }

    df = df.copy()
    df["err"] = (df["real_delivery_date"] - df["forecast_date"]).dt.days

    mae = df["err"].abs().mean()
    med = df["err"].median()
    n = len(df)

    jit = ((df["err"].abs() <= 2).sum() / n) * 100
    early = (df["err"] < -2).sum() / n * 100
    late = (df["err"] > 2).sum() / n * 100

    return {
        "n_forecasts": int(n),
        "n_matched": int(n),
        "coverage_%": round(n / base_n * 100, 1) if base_n > 0 else 0,
        "mae_days": round(mae, 2),
        "median_err": float(med),
        "jit_rate_%": round(jit, 1),
        "early_%": round(early, 1),
        "late_%": round(late, 1),
    }


def load_ledger():
    """Charge item_ledger.parquet de façon robuste (serial + dates)."""
    led_path = DATA / "item_ledger.parquet"
    if not led_path.exists():
        print(f"[compare] WARN: fichier ledger introuvable: {led_path}")
        return pd.DataFrame()

    led = pd.read_parquet(led_path)
    print(f"[compare] ledger brut: {len(led)} lignes, colonnes = {list(led.columns)}")

    # --- Serial ---
    serial_col = pick_first_existing(
        led,
        ["serial_norm", "serial", "$No. serie$", "No. serie", "no_serie"],
    )
    if serial_col is None:
        print("[compare] WARN: aucune colonne de n° de série trouvée dans ledger → pas de backtest.")
        return pd.DataFrame()

    led["serial_norm"] = led[serial_col].map(normalize_serial)

    # --- Date comptable / livraison ---
    date_col = pick_first_existing(
        led,
        ["doc_date", "$Date compta$", "Date compta", "date", "posting_date", "Posting Date"],
    )
    if date_col is None:
        print("[compare] WARN: aucune colonne de date trouvée dans ledger → pas de backtest.")
        return pd.DataFrame()

    raw = led[date_col].astype(str)

    # 🔥 Nettoyage des valeurs : on enlève les $ et les espaces parasites
    raw_clean = (
        raw.str.replace("$", "", regex=False)
           .str.strip()
    )

    print(f"[compare] date_col choisi dans ledger: {date_col} (dtype={led[date_col].dtype})")
    # Debug léger : quelques exemples
    print("[compare] exemples de dates brutes:", list(raw.head(3)))
    print("[compare] exemples de dates nettoyées:", list(raw_clean.head(3)))

    # 1) tentative générique avec dayfirst=True
    dt = pd.to_datetime(raw_clean, errors="coerce", dayfirst=True)

    # 2) si trop de NaT, on tente quelques formats classiques
    if dt.isna().mean() > 0.95:
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
            dt_try = pd.to_datetime(raw_clean, format=fmt, errors="coerce")
            if dt_try.isna().mean() < dt.isna().mean():
                dt = dt_try

    led["doc_date"] = dt

    nat_ratio = float(led["doc_date"].isna().mean())
    print(f"[compare] ledger doc_date NaT ratio: {nat_ratio:.3f}")

    # On drop uniquement les lignes sans date OU sans serial
    before = len(led)
    led = led.dropna(subset=["serial_norm", "doc_date"])
    after = len(led)
    print(f"[compare] ledger après dropna(serial_norm, doc_date): {before} -> {after} lignes")

    return led



def main():
    print("[compare_backtest_v1_v21] loading…")

    # === Load base forecasts V1 (pentes)
    v1_path = DATA / "consumables_forecasts.parquet"
    if not v1_path.exists():
        print(f"[compare] WARN: {v1_path} introuvable → V1 indisponible.")
        v1 = pd.DataFrame()
    else:
        v1 = pd.read_parquet(v1_path)
        v1["serial_norm"] = v1["serial"].map(normalize_serial)
        print(f"[compare] v1 rows: {len(v1)}")

    # === Load V2.1 ML predictions
    v21_path = ML / "consumables_forecasts_ml_v21.parquet"
    if not v21_path.exists():
        raise SystemExit(f"[compare] ERREUR: {v21_path} introuvable (lance d'abord predict_xgb_v21).")

    v21 = pd.read_parquet(v21_path)
    # Dans v21, on a normalement déjà serial_norm, mais on sécurise
    serial_col_v21 = "serial_norm" if "serial_norm" in v21.columns else "serial"
    v21["serial_norm"] = v21[serial_col_v21].map(normalize_serial)
    print(f"[compare] v21 rows: {len(v21)}")

    # === Load ledger (réalité)
    led = load_ledger()
    print(f"[compare] scope v1={len(v1)} v21={len(v21)} ledger={len(led)}")

    # Si ledger vide → on sort proprement
    if len(led) == 0:
        print("\n[compare] Ledger vide → impossible de calculer des KPI réels pour V1/V2.1.")
        print("Vérifie item_ledger.parquet ou la colonne de date comptable.")
        # On écrit quand même des fichiers vides pour ne pas casser de dépendances
        out_dir = DATA / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "v1_backtest_detailed.csv").write_text("")
        (out_dir / "v21_backtest_detailed.csv").write_text("")
        pd.DataFrame(
            [{"model": "V1"}, {"model": "V2.1"}]
        ).to_csv(out_dir / "compare_backtest_v1_v21_summary.csv", index=False)
        return

    # === Construction date de forecast

    # V1: last_seen + days_left_est
    if not v1.empty:
        v1 = v1.copy()
        v1["last_seen"] = pd.to_datetime(v1["last_seen"], errors="coerce")
        v1["days_left_est"] = pd.to_numeric(v1["days_left_est"], errors="coerce").fillna(0)
        v1["forecast_date"] = v1["last_seen"] + pd.to_timedelta(
            v1["days_left_est"].astype(int), unit="D"
        )

        # garder dernière observation par (serial_norm, color)
        v1_latest = v1.sort_values("last_seen").groupby(["serial_norm", "color"]).tail(1)
    else:
        v1_latest = pd.DataFrame(columns=["serial_norm", "color", "forecast_date"])

    # V2.1 : stockout ML direct
    v21 = v21.copy()
    # chercher colonne de date ML
    stockout_col = None
    for c in ["stockout_date_ml_v21", "stockout_date_ml", "stockout_date"]:
        if c in v21.columns:
            stockout_col = c
            break

    if stockout_col is None:
        raise SystemExit("Impossible de trouver une colonne stockout_date_ml*_ dans consumables_forecasts_ml_v21.parquet")

    v21["forecast_date"] = pd.to_datetime(v21[stockout_col], errors="coerce")
    # dernière observation par (serial_norm, color)
    if "last_seen" in v21.columns:
        v21["last_seen"] = pd.to_datetime(v21["last_seen"], errors="coerce")
        v21_latest = v21.sort_values("last_seen").groupby(["serial_norm", "color"]).tail(1)
    else:
        v21_latest = v21

    # === merge with ledger ===
    bt_v1 = v1_latest.merge(
        led[["serial_norm", "doc_date"]],
        on="serial_norm",
        how="left",
        validate="m:m",
    ) if not v1_latest.empty else v1_latest.copy()

    bt_v21 = v21_latest.merge(
        led[["serial_norm", "doc_date"]],
        on="serial_norm",
        how="left",
        validate="m:m",
    )

    bt_v1 = bt_v1.rename(columns={"doc_date": "real_delivery_date"})
    bt_v21 = bt_v21.rename(columns={"doc_date": "real_delivery_date"})

    # Ne garder que ceux qui ont une livraison réelle
    bt_v1 = bt_v1.dropna(subset=["real_delivery_date"]) if not bt_v1.empty else bt_v1
    bt_v21 = bt_v21.dropna(subset=["real_delivery_date"])

    # === KPI ===
    base_n = len(v1_latest) if not v1_latest.empty else len(v21_latest)

    kpi_v1 = compute_kpis(bt_v1, base_n=base_n)
    kpi_v21 = compute_kpis(bt_v21, base_n=base_n)

    print("\n===== KPI V1 (pentes) =====")
    for k, v in kpi_v1.items():
        print(f"{k:<15} : {v}")

    print("\n===== KPI V2.1 (ML Boosté) =====")
    for k, v in kpi_v21.items():
        print(f"{k:<15} : {v}")

    # === Save results ===
    out_dir = DATA / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_path_out = out_dir / "v1_backtest_detailed.csv"
    v21_path_out = out_dir / "v21_backtest_detailed.csv"
    summary_path = out_dir / "compare_backtest_v1_v21_summary.csv"

    bt_v1.to_csv(v1_path_out, index=False)
    bt_v21.to_csv(v21_path_out, index=False)

    pd.DataFrame(
        [
            {"model": "V1", **kpi_v1},
            {"model": "V2.1", **kpi_v21},
        ]
    ).to_csv(summary_path, index=False)

    print("\n💾 Saved:")
    print(" -", v1_path_out)
    print(" -", v21_path_out)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
