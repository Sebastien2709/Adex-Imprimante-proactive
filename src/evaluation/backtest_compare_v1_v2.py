from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data/processed")
ML_DATA = DATA / "ml"
OUT_DIR = DATA / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOL_DAYS = 120  # fenêtre de matching +/- 120 jours autour de la prédiction


def norm_serial(s):
    if not isinstance(s, str):
        return None
    return s.strip().upper()


def ledger_color_from_type(x: str):
    if not isinstance(x, str):
        return None
    t = x.lower()
    if "noir" in t or "black" in t:
        return "black"
    if "cyan" in t:
        return "cyan"
    if "magenta" in t:
        return "magenta"
    if "jaune" in t or "yellow" in t:
        return "yellow"
    return None


def prep_forecasts_v1():
    f1 = pd.read_parquet(DATA / "consumables_forecasts.parquet").copy()
    f1["serial_norm"] = f1["serial"].map(norm_serial)

    # dates
    if "last_seen" in f1.columns:
        f1["last_seen"] = pd.to_datetime(f1["last_seen"], errors="coerce")
    if "stockout_date" in f1.columns:
        f1["pred_date_v1"] = pd.to_datetime(f1["stockout_date"], errors="coerce")
    elif "days_left_est" in f1.columns and "last_seen" in f1.columns:
        f1["pred_date_v1"] = f1["last_seen"] + pd.to_timedelta(
            f1["days_left_est"].round().astype("Int64"), unit="D"
        )
    else:
        raise ValueError("Impossible de construire pred_date_v1 (stockout_date/days_left_est manquants).")

    return f1


def prep_forecasts_v2():
    f2 = pd.read_parquet(ML_DATA / "consumables_forecasts_ml.parquet").copy()
    f2["serial_norm"] = f2["serial"].map(norm_serial)

    if "last_seen" in f2.columns:
        f2["last_seen"] = pd.to_datetime(f2["last_seen"], errors="coerce")

    if "stockout_date_ml" in f2.columns:
        f2["pred_date_v2"] = pd.to_datetime(f2["stockout_date_ml"], errors="coerce")
    elif "days_left_ml" in f2.columns and "last_seen" in f2.columns:
        days_int = pd.to_numeric(f2["days_left_ml"], errors="coerce").fillna(0).clip(0, 365*2)
        f2["pred_date_v2"] = f2["last_seen"] + pd.to_timedelta(days_int.round().astype("Int64"), unit="D")
    else:
        raise ValueError("Impossible de construire pred_date_v2 (stockout_date_ml/days_left_ml manquants).")

    return f2


def prep_ledger():
    led = pd.read_parquet(DATA / "item_ledger.parquet").copy()

    # noms normalisés vus dans tes logs: serial, doc_date, consumable_type
    if "serial" not in led.columns:
        # fallback si ancien nom FR
        if "No. serie" in led.columns:
            led["serial"] = led["No. serie"]
        elif "No serie" in led.columns:
            led["serial"] = led["No serie"]
        else:
            raise ValueError("Colonne serial introuvable dans item_ledger.parquet")

    led["serial_norm"] = led["serial"].map(norm_serial)

    # date commande
    if "doc_date" not in led.columns:
        if "Date compta" in led.columns:
            led["doc_date"] = led["Date compta"]
        else:
            raise ValueError("Colonne doc_date/Date compta introuvable dans item_ledger.parquet")
    led["doc_date"] = pd.to_datetime(led["doc_date"], errors="coerce", dayfirst=True)

    # couleur depuis consumable_type
    if "consumable_type" in led.columns:
        led["color"] = led["consumable_type"].map(ledger_color_from_type)
    elif "Type conso" in led.columns:
        led["color"] = led["Type conso"].map(ledger_color_from_type)
    else:
        led["color"] = None

    led = led[led["serial_norm"].notna() & led["doc_date"].notna() & led["color"].notna()].copy()
    return led


def match_forecasts(forecasts: pd.DataFrame, led: pd.DataFrame, pred_col: str, tag: str):
    out_rows = []
    # index ledger par serial+color pour matcher vite
    grp = led.groupby(["serial_norm", "color"])["doc_date"].apply(list).to_dict()

    for r in forecasts.itertuples(index=False):
        serial = getattr(r, "serial_norm")
        color = getattr(r, "color")
        pred_date = getattr(r, pred_col)

        key = (serial, color)
        actual_dates = grp.get(key, [])

        matched_date = pd.NaT
        err_days = np.nan

        if pd.notna(pred_date) and actual_dates:
            # nearest actual date
            ad = pd.Series(pd.to_datetime(actual_dates))
            diffs = (ad - pred_date).dt.days.abs()
            i = diffs.idxmin()
            if diffs.loc[i] <= TOL_DAYS:
                matched_date = ad.loc[i]
                err_days = (matched_date - pred_date).days

        out_rows.append({
            "serial_norm": serial,
            "color": color,
            "pred_date": pred_date,
            "actual_date": matched_date,
            "err_days": err_days,
            "model": tag
        })

    return pd.DataFrame(out_rows)


def kpis(df_res: pd.DataFrame):
    n_forecasts = len(df_res)
    matched = df_res["err_days"].notna()
    n_matched = matched.sum()
    coverage = 100 * n_matched / n_forecasts if n_forecasts else 0

    mae = df_res.loc[matched, "err_days"].abs().mean() if n_matched else np.nan
    median_err = df_res.loc[matched, "err_days"].median() if n_matched else np.nan
    jit = (df_res.loc[matched, "err_days"].abs() <= 2).mean() * 100 if n_matched else np.nan
    early = (df_res.loc[matched, "err_days"] < -2).mean() * 100 if n_matched else np.nan
    late = (df_res.loc[matched, "err_days"] > 2).mean() * 100 if n_matched else np.nan

    return {
        "n_forecasts": n_forecasts,
        "n_matched": int(n_matched),
        "coverage_%": round(coverage, 1),
        "mae_days": round(mae, 2) if pd.notna(mae) else np.nan,
        "median_err": round(median_err, 2) if pd.notna(median_err) else np.nan,
        "jit_rate_%": round(jit, 1) if pd.notna(jit) else np.nan,
        "early_%": round(early, 1) if pd.notna(early) else np.nan,
        "late_%": round(late, 1) if pd.notna(late) else np.nan,
    }


def main():
    print("[compare_backtest] loading...")
    f1 = prep_forecasts_v1()
    f2 = prep_forecasts_v2()
    led = prep_ledger()

    # même scope serials/couleurs
    scope = set(zip(led["serial_norm"], led["color"]))
    f1 = f1[f1.apply(lambda x: (x["serial_norm"], x["color"]) in scope, axis=1)].copy()
    f2 = f2[f2.apply(lambda x: (x["serial_norm"], x["color"]) in scope, axis=1)].copy()

    print(f"[compare_backtest] scope v1={len(f1)} v2={len(f2)} ledger={len(led)}")

    res_v1 = match_forecasts(f1, led, "pred_date_v1", "v1_slopes")
    res_v2 = match_forecasts(f2, led, "pred_date_v2", "v2_xgb")

    k1 = kpis(res_v1)
    k2 = kpis(res_v2)

    print("\n===== KPI V1 (pentes) =====")
    for k, v in k1.items():
        print(f"{k:12s}: {v}")

    print("\n===== KPI V2 (XGBoost) =====")
    for k, v in k2.items():
        print(f"{k:12s}: {v}")

    # save detailed + summary
    res = pd.concat([res_v1, res_v2], ignore_index=True)
    res.to_csv(OUT_DIR / "compare_backtest_v1_v2_results.csv", index=False)

    summary = pd.DataFrame([{"model": "v1_slopes", **k1}, {"model": "v2_xgb", **k2}])
    summary.to_csv(OUT_DIR / "compare_backtest_v1_v2_summary.csv", index=False)

    print(f"\n💾 Saved:")
    print(f" - {OUT_DIR/'compare_backtest_v1_v2_results.csv'}")
    print(f" - {OUT_DIR/'compare_backtest_v1_v2_summary.csv'}")


if __name__ == "__main__":
    main()
