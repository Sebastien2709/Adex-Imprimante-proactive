import pandas as pd
from src.data.validate import check_future_dates, check_pct_bounds, check_non_negative

def test_check_pct_bounds():
    df = pd.DataFrame({"x":[0,50,100,101]})
    errs = check_pct_bounds(df, ["x"], 0, 100)
    assert len(errs)==1 and "1 values" in errs[0]

def test_check_non_negative():
    df = pd.DataFrame({"d":[0,1,-2]})
    errs = check_non_negative(df, ["d"])
    assert len(errs)==1 and "negatives" in errs[0]
