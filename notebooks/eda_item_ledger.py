from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

PROC = Path("data/processed")
OUT = Path("data/processed/eda/item_ledger")
OUT.mkdir(parents=True, exist_ok=True)

def infer_color_from_text(txt: str) -> str:
    """Reconnaît la couleur en FR/EN + abréviations usuelles (K/C/M/Y)."""
    if not isinstance(txt, str):
        return "unknown"
    t = txt.lower()
    # sécuriser les mots entiers + abréviations fréquentes
    if re.search(r"\b(noir|black|bk|k)\b", t):     return "black"
    if re.search(r"\b(cyan|c)\b", t):              return "cyan"
    if re.search(r"\b(magenta|m)\b", t):           return "magenta"
    if re.search(r"\b(jaune|yellow|y)\b", t):      return "yellow"
    # quelques motifs de désignations (ex: CEXV49, C879 … avec lettre finale couleur)
    if re.search(r"\bcex?v?\s*49\b.*\bnoir\b", t): return "black"
    if re.search(r"\bcex?v?\s*49\b.*\bcyan\b", t): return "cyan"
    if re.search(r"\bcex?v?\s*49\b.*\bmagenta\b", t): return "magenta"
    if re.search(r"\bcex?v?\s*49\b.*\b(yellow|jaune)\b", t): return "yellow"
    # fallback
    return "unknown"

def main():
    df = pd.read_parquet(PROC / "item_ledger.parquet")

    # --- colonnes présentes chez toi
    # ['No article','Date compta','No document','Quantite','No. contrat','No. serie',
    #  'Designation','Fabricant','Type conso','Type conso general','Capacite','Source conso','Societe','relation_type']

    # parse date (format vu: 04/01/23)
    df["Date compta"] = pd.to_datetime(df["Date compta"], errors="coerce", dayfirst=True)

    # couleur : priorité à "Type conso" (Toner Noir/Cyan/…); sinon "Designation"
    if "Type conso" in df.columns:
        col_type = df["Type conso"].fillna("")
        color = col_type.map(infer_color_from_text)
    else:
        color = pd.Series(["unknown"] * len(df))
    # si inconnu via Type conso, on tente la designation
    mask_unk = color.eq("unknown")
    if mask_unk.any() and "Designation" in df.columns:
        color.loc[mask_unk] = df.loc[mask_unk, "Designation"].fillna("").map(infer_color_from_text)
    df["color"] = color

    # === Graph 1: volume par couleur
    vc = df["color"].value_counts()
    plt.figure()
    vc.plot(kind="bar", rot=0)
    plt.title("Volume d'envois par couleur")
    plt.ylabel("Nombre d'envois")
    plt.tight_layout()
    plt.savefig(OUT / "bar_shipments_by_color.png")
    plt.close()

    # === Graph 2: top sociétés (si dispo)
    if "Societe" in df.columns:
        top_comp = df["Societe"].fillna("NA").value_counts().head(15)
        plt.figure()
        top_comp.sort_values().plot(kind="barh")
        plt.title("Top 15 sociétés par nombre d'envois")
        plt.xlabel("Nb d'envois")
        plt.tight_layout()
        plt.savefig(OUT / "bar_top_companies.png")
        plt.close()

    # === Graph 3: timeline hebdo
    weekly = (
        df.set_index("Date compta")
          .sort_index()
          .resample("W")
          .size()
    )
    plt.figure()
    weekly.plot()
    plt.title("Volume d'envois réels (hebdomadaire)")
    plt.ylabel("Nombre d'envois")
    plt.xlabel("Date compta")
    plt.tight_layout()
    plt.savefig(OUT / "ts_weekly_shipments.png")
    plt.close()

    print(f"[EDA Ledger] OK -> {OUT}")

if __name__ == "__main__":
    main()
