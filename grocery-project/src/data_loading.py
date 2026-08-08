"""Load and clean Project Hammer price data.

product.csv is one row per product, keyed by `id`. `concatted` is dropped,
it's just a repacking of the other columns. raw.csv is one row per scrape,
keyed by `product_id`, about 75 million rows, so it's cached as parquet
after the first load.

Both files have a small number of corrupted rows, probably from a shifted
column, a URL in `vendor`, non-numeric text in `current_price`. Those get
dropped, with a printed count so it's not silent.
"""

from pathlib import Path

import pandas as pd

from package_parser import parse_package_sizes

KNOWN_VENDORS = {
    "Walmart", "Metro", "Loblaws", "SaveOnFoods",
    "Voila", "NoFrills", "TandT", "Galleria",
}

# Hammer scraping started 2024-02-28. A handful of `nowtime` values show
# up as -4713-11-24 (a Julian day zero sentinel), so anything earlier is
# treated as invalid.

EARLIEST_VALID_DATE = pd.Timestamp("2024-01-01")

# below this a price_per_unit is corrupted data, not a real price
# (a gram of food for a thirtieth of a cent)
MIN_PLAUSIBLE_PRICE_PER_UNIT = 0.001


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

    # "error" shows up as a literal product_name on scraper failures, not a real product
    not_error = df["product_name"] != "error"
    n_error = (~not_error).sum()
    if n_error:
        print(f"load_products: dropping {n_error} rows with product_name 'error'")

    return df[valid_vendor & not_error].reset_index(drop=True)


def load_raw_prices(csv_path, cache_path=None) -> pd.DataFrame:
    """Load raw.csv: per-scrape price observations, keyed by `product_id`.

    Cached as parquet at `cache_path` after the first load.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)

    # `other` is read as a string and cast to category after loading.
    # reading it as category directly hits a pandas bug with mis-split
    # rows (unescaped newlines in values like "sale\n$3.50 MIN 2").
    
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


def fix_cents_bug(prices: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Some rows store current_price in cents instead of dollars.

    Cross-checked against price_per_unit ($X.XX/100g or /100ml) and the
    product's parsed size. A row where current_price is ~100x what that
    implies gets divided back down.
    """
    sizes = parse_package_sizes(products["units"])
    sizes.index = products["id"]

    joined = prices.join(sizes[["quantity", "unit"]], on="product_id")

    # price_per_unit shows up as either "$X.XX/100g" or, just as often,
    # "Y¢/100g" with a cents sign and no decimal point at all.
    dollar_ppu = prices["price_per_unit"].str.extract(r"^\$(?P<rate>\d+\.\d+)\s*/\s*100(?P<denom_unit>g|ml)$")
    cents_ppu = prices["price_per_unit"].str.extract(r"^(?P<rate>\d+)¢\s*/\s*100(?P<denom_unit>g|ml)$")

    rate = pd.to_numeric(dollar_ppu["rate"], errors="coerce")
    rate = rate.fillna(pd.to_numeric(cents_ppu["rate"], errors="coerce") / 100)
    denom_unit = dollar_ppu["denom_unit"].fillna(cents_ppu["denom_unit"])

    implied_price = rate * joined["quantity"] / 100
    ratio = joined["current_price"] / implied_price

    is_bugged = (joined["unit"] == denom_unit) & implied_price.notna() & ratio.between(90, 110)
    n_bad = is_bugged.sum()
    if n_bad:
        print(f"fix_cents_bug: correcting {n_bad} rows scaled by 100x")

    fixed = prices.copy()
    fixed.loc[is_bugged, "current_price"] = fixed.loc[is_bugged, "current_price"] / 100
    fixed.loc[is_bugged, "old_price"] = fixed.loc[is_bugged, "old_price"] / 100
    return fixed
