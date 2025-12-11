import pandas as pd

df = pd.read_parquet("data/processed/item_ledger.parquet")
print(df.columns.tolist())
print(df.head(10).to_string())

df2 = pd.read_parquet("data/processed/meters.parquet")
print(df2.columns.tolist())
print(df2.head(10).to_string())
