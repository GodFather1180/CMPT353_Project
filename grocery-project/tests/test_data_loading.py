import pandas as pd

from data_loading import load_products, load_raw_prices, merge_prices_with_products

PRODUCTS_CSV = """id,concatted,vendor,product_name,units,brand,detail_url,sku,upc
1,"Voila~Milk 2%@4L^",Voila,Milk 2%,4L,,,,
2,"junk~@^",https://www.example.com/bad-vendor-scrape,,,,,,
"""

# This mirrors real corruption we saw in the Hammer files: a Julian epoch
# sentinel date, plus a non-numeric value in current_price, both caused by
# shifted columns in the source scrape.
RAW_CSV = """nowtime,current_price,old_price,price_per_unit,other,product_id
2024-06-22 10:35:40,0.99,1.29,$0.99/item,SALE,1
-4713-11-24 12:00:00,1.50,1.50,$1.50/item,,1
2024-06-22 10:36:00,not_a_number,2.00,$2.00/item,,1
"""


def test_load_products_drops_unrecognized_vendor(tmp_path):
    path = tmp_path / "product.csv"
    path.write_text(PRODUCTS_CSV)

    df = load_products(path)

    assert list(df["id"]) == [1]
    assert list(df["vendor"]) == ["Voila"]


def test_load_products_drops_redundant_concatted_column(tmp_path):
    path = tmp_path / "product.csv"
    path.write_text(PRODUCTS_CSV)

    df = load_products(path)

    assert "concatted" not in df.columns


def test_load_raw_prices_drops_invalid_dates(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(RAW_CSV)

    df = load_raw_prices(path)

    assert len(df) == 2
    assert df["nowtime"].min() >= pd.Timestamp("2024-01-01")
    assert pd.api.types.is_datetime64_any_dtype(df["nowtime"])
    assert pd.api.types.is_integer_dtype(df["product_id"])


def test_load_raw_prices_coerces_non_numeric_price_to_nan(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(RAW_CSV)

    df = load_raw_prices(path)

    assert df["current_price"].isna().sum() == 1
    assert df["current_price"].notna().sum() == 1


def test_load_raw_prices_caches_to_parquet(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(RAW_CSV)
    cache_path = tmp_path / "raw.parquet"

    first = load_raw_prices(csv_path, cache_path=cache_path)
    assert cache_path.exists()

    csv_path.unlink()  # cached read must not touch the CSV again
    second = load_raw_prices(csv_path, cache_path=cache_path)

    # check_categorical=False because parquet round trips `other`'s
    # categories from pandas' nullable StringDtype to plain object dtype.
    # Same values, just different backing storage, not a real difference.
    pd.testing.assert_frame_equal(first, second, check_categorical=False)


def test_merge_prices_with_products(tmp_path):
    products_path = tmp_path / "product.csv"
    products_path.write_text(PRODUCTS_CSV)
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(RAW_CSV)

    products = load_products(products_path)
    prices = load_raw_prices(raw_path)
    merged = merge_prices_with_products(prices, products)

    assert (merged["vendor"] == "Voila").all()
    assert len(merged) == len(prices)
