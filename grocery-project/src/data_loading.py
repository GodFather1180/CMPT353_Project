"""Load and clean Project Hammer price data.

There are two source files (see jacobfilipp.com/hammer). product.csv has
one row per product, keyed by `id`, with columns id, concatted, vendor,
product_name, units, brand, detail_url, sku, upc. The `concatted` column
is just a `vendor~product_name@units^brand` packing of the other columns,
so it doesn't add anything and gets dropped on load.

raw.csv has one row per scrape observation, keyed by `product_id`, with
columns nowtime, current_price, old_price, price_per_unit, other,
product_id. It's about 75 million rows and 3.5GB, so we load it once with
tight dtypes and cache the result as parquet, since re-parsing that much
CSV every run would be a waste of time.

Both files have a small number of corrupted rows, probably from a field
like a comma inside a product name shifting the columns over. That shows
up as a URL or timestamp sitting in the `vendor` field, or non-numeric
text in `current_price`. We detect these and drop them, printing how many
so the scale of the problem stays visible instead of just disappearing.
"""

from pathlib import Path

import pandas as pd

KNOWN_VENDORS = {
    "Walmart", "Metro", "Loblaws", "SaveOnFoods",
    "Voila", "NoFrills", "TandT", "Galleria",
}

# Project Hammer scraping started 2024-02-28 (jacobfilipp.com/hammer).
# A handful of `nowtime` values in raw.csv show up as the Julian day zero
# sentinel (-4713-11-24) instead of a real timestamp, so anything before
# the project's start date gets treated as invalid rather than a real scrape.
EARLIEST_VALID_DATE = pd.Timestamp("2024-01-01")


def load_products(path) -> pd.DataFrame:
    """Load product.csv: vendor/name/units metadata, keyed by `id`."""
    df = pd.read_csv(
        path,
        dtype={
            "id": "int64",
            "concatted": "string",
            "vendor": "string",
            "product_name": "string",
            "units": "string",
            "brand": "string",
            "detail_url": "string",
            "sku": "string",
            "upc": "string",
        },
    )
    df = df.drop(columns=["concatted"])

    valid_vendor = df["vendor"].isin(KNOWN_VENDORS)
    n_bad = (~valid_vendor).sum()
    if n_bad:
        print(f"load_products: dropping {n_bad} rows with an unrecognized vendor value")

    return df[valid_vendor].reset_index(drop=True)


def load_raw_prices(csv_path, cache_path=None) -> pd.DataFrame:
    """Load raw.csv: per-scrape price observations, keyed by `product_id`.

    Cached as parquet at `cache_path` after the first load.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)

    # We read `other` as plain strings instead of `category` and cast it
    # after the full file is loaded. Reading category dtype directly hits a
    # pandas bug where chunked CSV parsing tries to union each chunk's
    # category dtype, and one chunk with mis-split rows (probably from
    # unescaped newlines inside values like "sale\n$3.50 MIN 2") ends up
    # with an incompatible dtype partway through the file.
    df = pd.read_csv(
        csv_path,
        dtype={
            "nowtime": "string",
            "current_price": "string",
            "old_price": "string",
            "price_per_unit": "string",
            "other": "string",
            "product_id": "string",
        },
    )

    nowtime = pd.to_datetime(df["nowtime"], errors="coerce")
    current_price = pd.to_numeric(df["current_price"], errors="coerce")
    old_price = pd.to_numeric(df["old_price"], errors="coerce")
    product_id = pd.to_numeric(df["product_id"], errors="coerce")

    valid = (
        nowtime.notna()
        & (nowtime >= EARLIEST_VALID_DATE)
        & product_id.notna()
    )
    n_bad = (~valid).sum()
    if n_bad:
        print(f"load_raw_prices: dropping {n_bad} rows with invalid nowtime or product_id")

    df = df.loc[valid, ["price_per_unit", "other"]].copy()
    df["other"] = df["other"].astype("category")
    df["nowtime"] = nowtime[valid]
    df["current_price"] = current_price[valid].astype("float32")
    df["old_price"] = old_price[valid].astype("float32")
    df["product_id"] = product_id[valid].astype("int64")
    df = df.reset_index(drop=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

    return df


def merge_prices_with_products(prices: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Join price observations to their product metadata."""
    return prices.merge(
        products, left_on="product_id", right_on="id", how="inner", suffixes=("", "_product")
    )
