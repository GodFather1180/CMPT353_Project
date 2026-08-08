"""Load the StatCan Food CPI series, table 18-10-0004-01."""

import pandas as pd


def load_food_cpi(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    food = df[(df["Products and product groups"] == "Food") & (df["GEO"] == "Canada")].copy()
    food["date"] = pd.to_datetime(food["REF_DATE"], format="%Y-%m")
    food = food[["date", "VALUE"]].rename(columns={"VALUE": "cpi"})
    return food.sort_values("date").reset_index(drop=True)
