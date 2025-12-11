import os, yaml, pandas as pd

PROC = "data/processed"
RULES_PATH = "configs/rules.yaml"

DEFAULT_RULES = {
    "dates": {"allow_future": False, "timezone": "Europe/Paris"},
    "consumables": {"pct_min": 0, "pct_max": 100, "min_points_for_slope": 8, "low_threshold_pct": 10},
    "meters": {"allow_negative_deltas": False},
    "joins": {"keys": ["serial", "company"]},
    "quality_thresholds": {"max_all_zero_ratio": 0.20},
}

def load_rules():
    if not os.path.exists(RULES_PATH):
        print(f"[WARN] {RULES_PATH} manquant, utilisation des valeurs par défaut.")
        return DEFAULT_RULES
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        yml = yaml.safe_load(f)
    if not isinstance(yml, dict) or not yml:
        print(f"[WARN] {RULES_PATH} vide/invalide, utilisation des valeurs par défaut.")
        return DEFAULT_RULES
    # merge superficiel: yml > defaults
    rules = DEFAULT_RULES.copy()
    for k, v in yml.items():
        if isinstance(v, dict) and k in rules:
            merged = rules[k].copy()
            merged.update(v)
            rules[k] = merged
        else:
            rules[k] = v
    return rules

def check_future_dates(df, cols):
    errs = []
    today = pd.Timestamp.today().normalize()
    for c in cols:
        if c in df.columns:
            # dates peuvent être date/datetime/str
            s = pd.to_datetime(df[c], errors="coerce")
            fut = s.dropna() > today
            if fut.any():
                errs.append(f"{c}: {int(fut.sum())} future values")
    return errs

def check_pct_bounds(df, cols, lo, hi):
    errs = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            bad = s.dropna().pipe(lambda x: (x < lo) | (x > hi)).sum()
            if bad:
                errs.append(f"{c}: {int(bad)} values out of [{lo},{hi}]")
    return errs

def check_non_negative(df, cols):
    errs = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            bad = (s.dropna() < 0).sum()
            if bad:
                errs.append(f"{c}: {int(bad)} negatives")
    return errs

def main():
    RULES = load_rules()
    errors = []

    # Consumables
    cons_path = os.path.join(PROC, "kpax_consumables.parquet")
    if os.path.exists(cons_path):
        cons = pd.read_parquet(cons_path)
        errors += check_future_dates(cons, ["date_import","date_update"])
        errors += check_pct_bounds(cons,
                                   ["black_pct","cyan_pct","magenta_pct","yellow_pct"],
                                   RULES["consumables"]["pct_min"],
                                   RULES["consumables"]["pct_max"])

    # Meters
    meters_path = os.path.join(PROC, "meters.parquet")
    if os.path.exists(meters_path):
        m = pd.read_parquet(meters_path)
        errors += check_future_dates(m, ["start_date","end_date","read_date"])
        delta_cols = [c for c in m.columns if c.endswith("_delta")]
        if not RULES["meters"].get("allow_negative_deltas", False) and delta_cols:
            errors += check_non_negative(m, delta_cols)

    # Item ledger
    ledger_path = os.path.join(PROC, "item_ledger.parquet")
    if os.path.exists(ledger_path):
        led = pd.read_parquet(ledger_path)
        errors += check_future_dates(led, ["doc_datetime"])

    if errors:
        print("❌ Validation FAILED:")
        for e in errors: print(" -", e)
        raise SystemExit(1)
    else:
        print("✅ Validation PASSED")

if __name__ == "__main__":
    main()
