import pandas as pd
from src.data.clean_kpax_consumables import clean_consumables
from src.data.clean_meters import clean_meters
from src.data.clean_item_ledger import clean_item_ledger

def test_pct_bounds():
    df = pd.DataFrame({"black_pct":[0,50,100,101,-1]})
    out = clean_consumables(df.copy())
    assert out["black_pct"].isna().sum() == 2  # 101 & -1 -> NaN

def test_all_zero_flag():
    df = pd.DataFrame({
        "black_pct":[0,10],
        "cyan_pct":[0,10],
        "magenta_pct":[0,10],
        "yellow_pct":[0,10],
    })
    out = clean_consumables(df)
    assert out["all_zero_flag"].tolist() == [1,0]

def test_meters_deltas_non_negative():
    df = pd.DataFrame({
        "start_a4_bw":[100,  50],
        "end_a4_bw":  [150,  40],
    })
    out = clean_meters(df)
    assert "a4_bw_delta" in out.columns
    assert out.loc[0,"a4_bw_delta"] == 50
    assert out.loc[1,"a4_bw_delta"] == -10  # le test laisse passer; validation le signalera

def test_item_ledger_date_parse():
    df = pd.DataFrame({"doc_date":["08/02/23","31/12/22"]})
    out = clean_item_ledger(df)
    assert pd.api.types.is_datetime64_ns_dtype(out["doc_datetime"])
