from pathlib import Path

ROOT = Path("data/processed/eda")
OUT_MD = ROOT / "EDA_Report.md"

sections = [
    ("Prévisions (Forecasts)", ROOT / "forecasts"),
    ("Envois réels (Item Ledger)", ROOT / "item_ledger"),
    ("Activité imprimantes (Meters)", ROOT / "meters"),
]

def img_md(path):
    return f"![{path.name}]({path.as_posix()})\n"

with open(OUT_MD, "w", encoding="utf-8") as md:
    md.write("# 📊 EDA Report – AdexGroup Project\n\n")
    for title, folder in sections:
        if not folder.exists():
            continue
        md.write(f"## {title}\n\n")
        for img in sorted(folder.glob("*.png")):
            md.write(img_md(img))
        md.write("\n")

print(f"[EDA report] Markdown généré -> {OUT_MD}")
