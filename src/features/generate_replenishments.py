# src/features/generate_replenishments.py
import os
import pandas as pd
import yaml

PROC = "data/processed"
IN_FC = os.path.join(PROC, "consumables_forecasts.parquet")
REL   = os.path.join(PROC, "serial_relations.parquet")
OUT_CSV = os.path.join(PROC, "replenishments_to_create.csv")

def load_rules():
    with open("configs/rules.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    if not os.path.exists(IN_FC):
        raise SystemExit("Run compute_slopes first.")

    rules = load_rules()
    _lead_time   = rules.get("lead_time_days", 5)
    _buffer_days = rules.get("buffer_days", 3)

    # 1) Charger les prévisions
    fc = pd.read_parquet(IN_FC)

    # 2) Normaliser SERIAL (trim + upper) pour fiabiliser les jointures
    if "serial" in fc.columns:
        fc["serial"] = fc["serial"].astype(str).str.strip().str.upper()

    # 3) Joindre le type de relation (CONTRAT / HORS CONTRAT / LIBRE) via une colonne RENOMMÉE
    if os.path.exists(REL):
        rel = pd.read_parquet(REL)  # attendu: ['serial','company','relation_type']
        if "serial" in rel.columns:
            rel_small = rel[["serial","relation_type"]].drop_duplicates("serial").copy()
            rel_small["serial"] = rel_small["serial"].astype(str).str.strip().str.upper()
            rel_small = rel_small.rename(columns={"relation_type": "relation_type_rel"})  # éviter _x/_y
            fc = fc.merge(rel_small, on="serial", how="left")
        else:
            fc["relation_type_rel"] = pd.NA
    else:
        fc["relation_type_rel"] = pd.NA

    # 3b) Coalescer proprement dans la colonne finale 'relation_type'
    if "relation_type" in fc.columns:
        # si la colonne existe déjà (écrite par compute_slopes), on complète les NaN avec la version référentiel
        fc["relation_type"] = fc["relation_type"].where(fc["relation_type"].notna(), fc["relation_type_rel"])
    else:
        fc["relation_type"] = fc["relation_type_rel"]
    if "relation_type_rel" in fc.columns:
        fc.drop(columns=["relation_type_rel"], inplace=True)

    # 4) Filtre: should_order_now + valeurs clés + CONTRAT uniquement
    should_col = "should_order_now"
    mask = (
        (fc[should_col] == True if should_col in fc.columns else False)
        & (fc["stockout_date"].notna() if "stockout_date" in fc.columns else False)
        & (fc["level_now_pct"].notna() if "level_now_pct" in fc.columns else False)
        & (fc["relation_type"] == "contrat_adex")
    )

    # 5) Colonnes de sortie
    desired_cols = ["serial","company","color","level_now_pct","days_left_est","stockout_date","relation_type"]
    present_cols = [c for c in desired_cols if c in fc.columns]
    todo = fc.loc[mask, present_cols].copy()

    # 6) Champs ERP placeholders
    todo["sku"] = ""                 # à mapper plus tard
    todo["qty"] = 1                  # 1 cartouche par défaut
    todo["priority"] = "HIGH"
    todo["action"] = "CREATE_ORDER"

    sort_cols = [c for c in ["stockout_date","serial","color"] if c in todo.columns]
    if sort_cols:
        todo = todo.sort_values(sort_cols)

    # Logs utiles
    rel_counts = fc["relation_type"].value_counts(dropna=False).to_dict() if "relation_type" in fc.columns else {}
    should_true = int(fc[should_col].sum()) if should_col in fc.columns else 0
    print(f"[replenishments] relation_type counts: {rel_counts} | should_order_now=True: {should_true}")
    print(f"[replenishments] kept rows (contracts only): {len(todo)} / {len(fc)}")

    os.makedirs(PROC, exist_ok=True)
    todo.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {OUT_CSV} ({len(todo)} rows)")

if __name__ == "__main__":
    main()
