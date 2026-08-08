"""Build our own monthly food price index from Hammer data.

Products constantly enter and leave the catalog (see shrinkflation.py),
so a fixed base month would lose most products within a few months of
it. Instead this chains together month over month price relatives,
using whichever products have data in each pair of adjacent months,
the same idea StatCan itself uses to handle a changing product basket.

Each link is the geometric mean of the relatives (a Jevons index, what
StatCan uses at this level). The arithmetic mean drifts upward whenever
prices bounce on sales: halve then double averages to +25% instead of
netting out to zero.
"""

import numpy as np
import pandas as pd

from data_loading import MIN_PLAUSIBLE_PRICE_PER_UNIT
from package_parser import parse_package_sizes


def build_price_index(products: pd.DataFrame, prices: pd.DataFrame, start_month: str = None) -> pd.DataFrame:
    """start_month (e.g. "2024-06") skips earlier months entirely, not
    just rebases to them. Hammer's scraping was still ramping up in its
    first few months (Feb 2024 had 2,296 price observations, June 2024
    had 1.37 million), so an early base month is a tiny, unreliable
    anchor that the whole chained index would inherit.
    """
    sizes = parse_package_sizes(products["units"])
    sizes.index = products["id"]

    joined = prices.join(sizes["quantity"], on="product_id")
    joined = joined[joined["quantity"] > 0].copy()
    joined["price_per_unit"] = joined["current_price"] / joined["quantity"]
    joined = joined[joined["price_per_unit"] >= MIN_PLAUSIBLE_PRICE_PER_UNIT]
    joined["month"] = joined["nowtime"].dt.to_period("M")

    if start_month is not None:
        joined = joined[joined["month"] >= pd.Period(start_month, "M")]

    monthly = joined.groupby(["product_id", "month"])["price_per_unit"].mean().unstack("month")
    monthly = monthly.sort_index(axis=1)
    months = monthly.columns

    # loop is over ~30 months, not the data
    link_relatives = []
    for i in range(1, len(months)):
        prev_month, curr_month = monthly[months[i - 1]], monthly[months[i]]
        both_present = prev_month.notna() & curr_month.notna() & (prev_month > 0)
        relatives = curr_month[both_present] / prev_month[both_present]
        link_relatives.append(np.exp(np.log(relatives).mean()))

    index_values = [100.0] + list(100.0 * np.cumprod(link_relatives))
    return pd.DataFrame({"month": months.to_timestamp(), "price_index": index_values})
