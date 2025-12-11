import os, pandas as pd

PROC = "data/processed"
FC   = os.path.join(PROC, "consumables_forecasts.parquet")
REL  = os.path.join(PROC, "serial_relations.parquet")

def norm_ser(s):
    return str(s).strip().upper()

def main():
    fc = pd.read_parquet(FC)
    rel = pd.read_parquet(REL)

    # Normaliser
    fc_ser  = fc["serial"].astype(str).map(norm_ser)
    rel_ser = rel["serial"].astype(str).map(norm_ser)

    # Stats
    print("FC rows:", len(fc), "| REL rows:", len(rel))
    print("REL relation_type counts:", rel["relation_type"].value_counts(dropna=False).to_dict())

    # Intersections
    s_fc  = set(fc_ser.unique())
    s_rel = set(rel_ser.unique())
    inter = s_fc & s_rel
    print("Unique serials -> FC:", len(s_fc), "REL:", len(s_rel), "INTERSECTION:", len(inter))

    # Part de FC couverte par REL
    print("Coverage FC by REL: {:.1f}%".format(100 * len(inter) / max(1,len(s_fc))))

    # Aperçu de quelques serials non appariés
    missing = list(s_fc - s_rel)
    print("Examples not matched (up to 10):", missing[:10])

    # Quels relation_type sur l'intersection (rapide)
    rel_norm = rel.copy()
    rel_norm["serial"] = rel_norm["serial"].astype(str).map(norm_ser)
    has_contrat = rel_norm[rel_norm["relation_type"]=="contrat_adex"]["serial"].nunique()
    print("Serials 'contrat_adex' in REL:", has_contrat)

if __name__ == "__main__":
    main()
