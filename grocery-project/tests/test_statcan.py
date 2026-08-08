import io

import pandas as pd

from statcan import load_food_cpi

CPI_CSV = """REF_DATE,GEO,DGUID,Products and product groups,UOM,UOM_ID,SCALAR_FACTOR,SCALAR_ID,VECTOR,COORDINATE,VALUE,STATUS,SYMBOL,TERMINATED,DECIMALS
2024-02,Canada,2016A000011124,All-items,2002=100,17,units,0,v41690973,2.2,158.8,,,,1
2024-02,Canada,2016A000011124,Food,2002=100,17,units,0,v41691234,2.5,188.1,,,,1
2024-03,Canada,2016A000011124,Food,2002=100,17,units,0,v41691234,2.5,187.8,,,,1
2024-02,Ontario,2016A000235,Food,2002=100,17,units,0,v41699999,2.5,190.0,,,,1
"""


def test_filters_to_food_and_canada():
    result = load_food_cpi(io.StringIO(CPI_CSV))

    assert len(result) == 2
    assert set(result["cpi"]) == {188.1, 187.8}


def test_parses_ref_date_to_datetime():
    result = load_food_cpi(io.StringIO(CPI_CSV))

    assert result["date"].iloc[0] == pd.Timestamp("2024-02-01")
    assert result["date"].iloc[1] == pd.Timestamp("2024-03-01")


def test_sorted_by_date():
    reversed_csv = CPI_CSV.replace(
        "2024-02,Canada,2016A000011124,Food,2002=100,17,units,0,v41691234,2.5,188.1,,,,1\n"
        "2024-03,Canada,2016A000011124,Food,2002=100,17,units,0,v41691234,2.5,187.8,,,,1\n",
        "2024-03,Canada,2016A000011124,Food,2002=100,17,units,0,v41691234,2.5,187.8,,,,1\n"
        "2024-02,Canada,2016A000011124,Food,2002=100,17,units,0,v41691234,2.5,188.1,,,,1\n",
    )
    result = load_food_cpi(io.StringIO(reversed_csv))

    assert list(result["date"]) == sorted(result["date"])
