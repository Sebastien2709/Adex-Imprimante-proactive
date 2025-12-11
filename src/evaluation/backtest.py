# ==============================================================
# backtest.py — évalue les prévisions vs expéditions réelles
# - Référence = reorder_date si dispo, sinon stockout_date
# - Matching = "après d'abord, sinon le plus proche" (avec distance max)
# - Garde-fous: fenêtre temporelle + distance max d'appariement
# - Sortie: data/processed/eval/forecast_backtest_results.csv
# ==============================================================

import re
from pathlib import Path
import numpy as np
import pandas as pd

print("[backtest] version=2025-10-09T3")

# ------------------ chemins ------------------
PROC = Path("data/processed")
EVAL = PROC / "eval"
EVAL.mkdir(parents=True, exist_ok=True)

FORECASTS_PATH = PROC / "consumables_forecasts.parquet"
LEDGER_PATH    = PROC / "item_ledger.parquet"
OUT_CSV        = EVAL / "forecast_backtest_results.csv"

# ---------------- paramètres -----------------
WINDOW_DAYS     = 180     # fenêtre globale pour chercher un envoi autour de la date de réf
MAX_MATCH_DAYS  = 75     # distance max autorisée entre date de réf et envoi retenu (en valeur absolue)
JIT_TOL         = 5       # "juste à temps" = |erreur| <= 5 jours

# ==============================================================#
# Utils
# ==============================================================#
def _norm(s) -> str:
    return (str(s) if pd.notna(s) else "").strip().lower()

EU_DT_FORMATS = ("%d/%m/%Y", "%d/%m/%y")

def parse_eu_date_series(series: pd.Series) -> pd.Series:
    """Parse sans warning: essaie formats EU connus, sinon dayfirst=True."""
    parsed_any = None
    for fmt in EU_DT_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            parsed_any = parsed
            break
    if parsed_any is None:
        parsed_any = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed_any.dt.normalize()

# mapping SKU -> couleur (à étendre si besoin)
SKU_COLOR_MAP = {
    r"\bBK\b": "black", r"\bBLACK\b": "black", r"\bK($|\b)": "black",
    r"\bCYAN\b": "cyan", r"\bC($|\b)": "cyan",
    r"\bMAGENTA\b": "magenta", r"\bM($|\b)": "magenta",
    r"\bYELLOW\b": "yellow", r"\bJAUNE\b": "yellow", r"\bY($|\b)": "yellow",
}
 
def _infer_color_from_text(txt: str) -> str:
    t = _norm(txt).upper()
    # mots complets FR/EN
    if " NOIR" in " "+t or " BLACK" in " "+t or re.search(r"\bK($|\b)", t): return "black"
    if " CYAN" in " "+t or re.search(r"\bC($|\b)", t): return "cyan"
    if " MAGENTA" in " "+t or re.search(r"\bM($|\b)", t): return "magenta"
    if " JAUNE" in " "+t or " YELLOW" in " "+t or re.search(r"\bY($|\b)", t): return "yellow"
    # patterns génériques
    for pat, col in SKU_COLOR_MAP.items():
        if re.search(pat, t, re.I):
            return col
    return ""

def infer_color_from_ledger_row(row) -> str:
    # priorité: article_no, puis consumable_type, puis designation
    for col in ("article_no", "consumable_type", "designation"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            c = _infer_color_from_text(str(row[col]))
            if c:
                return c
    return ""

# ==============================================================#
# Chargement & préparation
# ==============================================================#
def load_data():
    if not FORECASTS_PATH.exists():
        raise SystemExit(f"Missing {FORECASTS_PATH} — run features first.")
    if not LEDGER_PATH.exists():
        raise SystemExit(f"Missing {LEDGER_PATH} — run cleaning first.")

    # ---------- Forecasts ----------
    f = pd.read_parquet(FORECASTS_PATH)

    need_f = {"serial", "color", "stockout_date"}
    missing_f = need_f - set(f.columns)
    if missing_f:
        raise SystemExit(f"[backtest] Missing columns in forecasts: {missing_f}")

    f["stockout_date"] = pd.to_datetime(f["stockout_date"], errors="coerce").dt.normalize()
    if "reorder_date" in f.columns:
        f["reorder_date"] = pd.to_datetime(f["reorder_date"], errors="coerce").dt.normalize()
    else:
        f["reorder_date"] = pd.NaT
    if "last_seen" in f.columns:
        f["last_seen"] = pd.to_datetime(f["last_seen"], errors="coerce")

    # normalisations
    f = f[(f["serial"].notna()) & (f["color"].notna())].copy()
    f["serial_norm"] = f["serial"].astype(str).str.strip().str.upper()
    f["color"] = f["color"].astype(str).str.strip().str.lower()

    # ref_date = reorder_date si présente, sinon stockout_date
    f["ref_date"] = f["reorder_date"].where(f["reorder_date"].notna(), f["stockout_date"])

    # ---------- Ledger ----------
    led = pd.read_parquet(LEDGER_PATH)
    print("[backtest] ledger columns (before rename):", list(led.columns))

    # --- charger les serials sous contrat (si dispo) ---
    CONTRACTED = PROC / "serial_relations.parquet"
    contracted_serials = None
    if CONTRACTED.exists():
        sr = pd.read_parquet(CONTRACTED)
        if "serial" not in sr.columns and "No. serie" in sr.columns:
            sr = sr.rename(columns={"No. serie": "serial"})
        sr["serial_norm"] = sr["serial"].astype(str).str.strip().str.upper()
        contracted_serials = set(sr["serial_norm"].unique())


    rename_map = {
        "No. serie": "serial", "No serie": "serial",
        "No. contrat": "contract_no", "No contrat": "contract_no",
        "Designation": "designation",
        "Type conso": "consumable_type",
        "Quantite": "qty",
        "No article": "article_no",
        "Date compta": "doc_date",
        "Societe": "company",
        "No document": "doc_no",
    }
    led = led.rename(columns={k: v for k, v in rename_map.items() if k in led.columns})
    print("[backtest] ledger columns (after rename):", list(led.columns))

    # colonne de date -> ship_date (sans warning)
    date_col = next((c for c in ["doc_datetime", "doc_date"] if c in led.columns), None)
    if date_col is None:
        print("[backtest] ⚠️ no date col (doc_datetime/doc_date). ship_date=NaT.")
        led["ship_date"] = pd.NaT
    else:
        led["ship_date"] = parse_eu_date_series(led[date_col])

    # normalisations
    if "serial" not in led.columns:
        led["serial"] = pd.NA
    led["serial_norm"] = led["serial"].astype(str).str.strip().str.upper()

    # si on a le référentiel contrat, ne garder que ces séries
    if contracted_serials is not None:
        led = led[led["serial_norm"].isin(contracted_serials)].copy()

    if "relation_type" not in led.columns:
        has_contract = led["contract_no"].notna() if "contract_no" in led.columns else False
        has_serial = led["serial"].notna()
        led["relation_type"] = np.where(
            has_contract & has_serial, "contrat_adex",
            np.where(has_serial, "machine_hors_contrat", "commande_libre")
        )

    # couleur estimée
    led["ledger_color"] = led.apply(infer_color_from_ledger_row, axis=1)

    # --------- Diagnostics ---------
    print("[diag] forecasts dates: ", f["ref_date"].min(), "→", f["ref_date"].max())
    print("[diag] ledger ship_date:", led["ship_date"].min(), "→", led["ship_date"].max())
    if "relation_type" in led.columns:
        print("[diag] ledger relation_type counts:", led["relation_type"].value_counts(dropna=False).to_dict())
    f_serials  = set(f["serial_norm"].unique())
    l_serials  = set(led["serial_norm"].dropna().unique())
    inter = f_serials & l_serials
    print("[diag] unique serials — forecasts:", len(f_serials), " | ledger:", len(l_serials), " | intersect:", len(inter))
    print("[diag] ledger rows on intersect serials:", int(led[led["serial_norm"].isin(inter)].shape[0]))

    return f, led

# ==============================================================#
# Matching prévisions ↔ expéditions
# ==============================================================#
def attach_next_shipment_date(forecasts: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    led = ledger.copy()

    # Filtrer sur contrat uniquement si le ledger en contient réellement
    if "relation_type" in led.columns and (led["relation_type"] == "contrat_adex").any():
        led = led[led["relation_type"] == "contrat_adex"]

    led = led[led["serial_norm"].notna() & led["ship_date"].notna()].copy()

    keep_cols = ["doc_no", "designation", "consumable_type", "qty"]
    for c in keep_cols:
        if c not in led.columns:
            led[c] = pd.NA

    led_sc = led[led["ledger_color"] != ""].copy()
    led_sc = led_sc[["serial_norm", "ledger_color", "ship_date"] + keep_cols].rename(columns={"ledger_color": "color"})
    led_s  = led[["serial_norm", "ship_date"] + keep_cols].copy()

    # tri déterministe
    led_sc = led_sc.sort_values(["serial_norm", "color", "ship_date"])
    led_s  = led_s.sort_values(["serial_norm", "ship_date"])

    recs = []
    for _, r in forecasts.iterrows():
        serial = r.get("serial_norm")
        color  = r.get("color")
        refd   = r.get("ref_date")

        base = {
            "serial": r.get("serial"),
            "company": r.get("company", None),
            "color": color,
            "level_now_pct": r.get("level_now_pct", np.nan),
            "slope_pct_per_day": r.get("slope_pct_per_day", np.nan),
            "days_left_est": r.get("days_left_est", np.nan),
            "stockout_date": r.get("stockout_date"),
            "reorder_date": r.get("reorder_date"),
            "ref_date": refd,
            "should_order_now": bool(r.get("should_order_now", False)),
            "relation_type": r.get("relation_type", ""),
            "ship_date_real": pd.NaT,
            "doc_no": pd.NA,
            "ledger_designation": pd.NA,
            "ledger_type": pd.NA,
            "ledger_qty": pd.NA,
        }

        if pd.isna(serial) or pd.isna(refd):
            recs.append(base)
            continue

        lo = refd - pd.Timedelta(days=WINDOW_DAYS)
        hi = refd + pd.Timedelta(days=WINDOW_DAYS)

        # 1) (serial + couleur) — "après d'abord"
        cand = led_sc[(led_sc["serial_norm"] == serial) & (led_sc["color"] == color)]
        cand = cand[(cand["ship_date"] >= lo) & (cand["ship_date"] <= hi)]
        row = None
        cand_after = cand[cand["ship_date"] >= refd]
        if not cand_after.empty:
            row = cand_after.iloc[0]
        elif not cand.empty:
            # sinon, le plus proche en absolu (avec MAX_MATCH_DAYS)
            idx_nearest = (cand["ship_date"] - refd).abs().idxmin()
            near = cand.loc[idx_nearest]
            if abs((near["ship_date"] - refd).days) <= MAX_MATCH_DAYS:
                row = near

        # 2) fallback: serial seul, même logique
        if row is None:
            cand2 = led_s[(led_s["serial_norm"] == serial) & (led_s["ship_date"] >= lo) & (led_s["ship_date"] <= hi)]
            cand2_after = cand2[cand2["ship_date"] >= refd]
            if not cand2_after.empty:
                row = cand2_after.iloc[0]
            elif not cand2.empty:
                idx2 = (cand2["ship_date"] - refd).abs().idxmin()
                near2 = cand2.loc[idx2]
                if abs((near2["ship_date"] - refd).days) <= MAX_MATCH_DAYS:
                    row = near2

        if row is not None:
            base.update({
                "ship_date_real": row["ship_date"],
                "doc_no": row.get("doc_no"),
                "ledger_designation": row.get("designation"),
                "ledger_type": row.get("consumable_type"),
                "ledger_qty": row.get("qty"),
            })

        recs.append(base)

    df = pd.DataFrame(recs)

    # colonnes garanties
    for col, default in [
        ("ship_date_real", pd.NaT),
        ("ref_date", pd.NaT),
        ("stockout_date", pd.NaT),
        ("reorder_date", pd.NaT),
        ("color", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    return df

# ==============================================================#
# KPI
# ==============================================================#
def compute_metrics(df: pd.DataFrame):
    matched = df[df["ship_date_real"].notna()].copy()
    matched["err_days"] = (matched["ship_date_real"] - matched["ref_date"]).dt.days
    kpi = {
        "n_forecasts": len(df),
        "n_matched": len(matched),
        "coverage_%": round(100 * len(matched) / max(len(df), 1), 1),
    }
    if not matched.empty:
        kpi.update({
            "mae_days": round(abs(matched["err_days"]).mean(), 2),
            "median_err": round(matched["err_days"].median(), 2),
            "jit_rate_%": round(100 * (abs(matched["err_days"]) <= JIT_TOL).mean(), 1),
            "early_%": round(100 * (matched["err_days"] < -JIT_TOL).mean(), 1),
            "late_%": round(100 * (matched["err_days"] > JIT_TOL).mean(), 1),
        })
    else:
        kpi.update({"mae_days": None, "median_err": None, "jit_rate_%": None,
                    "early_%": None, "late_%": None})
    return matched, kpi

# ==============================================================#
# Main
# ==============================================================#
def main():
    print("[backtest] loading data…")
    f, led = load_data()

    # Limiter les forecasts à la période couverte par le ledger ± fenêtre (sur ref_date)
    if led["ship_date"].notna().any():
        led_min = led["ship_date"].min()
        led_max = led["ship_date"].max()
        if pd.notna(led_min) and pd.notna(led_max):
            lo = led_min - pd.Timedelta(days=WINDOW_DAYS)
            hi = led_max + pd.Timedelta(days=WINDOW_DAYS)
            before_len = len(f)
            f = f[(f["ref_date"] >= lo) & (f["ref_date"] <= hi)].copy()
            print(f"[diag] trimmed forecasts to ledger window: {before_len} -> {len(f)} "
                  f"({lo.date()} .. {hi.date()})")

    # périmètre contrat si dispo
    if "relation_type" in f.columns and (f["relation_type"] == "contrat_adex").any():
        f = f[f["relation_type"].eq("contrat_adex")]

    # dédoublonnage
    f = f.sort_values(["serial_norm", "color", "ref_date"]).drop_duplicates(subset=["serial_norm","color","ref_date"])
    print(f"[backtest] forecasts scope: {len(f)} rows")

    print("[backtest] matching with ledger…")
    bt = attach_next_shipment_date(f, led)

    # garde-fous colonnes
    for c in ["ship_date_real", "ref_date"]:
        if c not in bt.columns:
            print(f"[backtest] WARN: missing column {c} in matched dataframe. Creating default.")
            bt[c] = pd.NaT

    # KPI
    matched, kpi = compute_metrics(bt)

    bt["err_days"] = (bt["ship_date_real"] - bt["ref_date"]).dt.days
    bt.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[backtest] saved results: {OUT_CSV} ({len(bt)} rows)\n")

    print("===== KPI Backtest =====")
    for k, v in kpi.items():
        print(f"{k:15s}: {v}")

    if not matched.empty:
        print(f"[backtest] ✅ matched: {len(matched)} rows ({kpi['coverage_%']}% coverage)")
    else:
        print("[backtest] ⚠️ no matched shipments — check color mapping or window size.")

if __name__ == "__main__":
    main()
