import pandas as pd
import pytest

from data_loading import fix_cents_bug, load_products, load_raw_prices, merge_prices_with_products

PRODUCTS_CSV = """id,concatted,vendor,product_name,units,brand,detail_url,sku,upc
1,"Voila~Milk 2%@4L^",Voila,Milk 2%,4L,,,,
2,"junk~@^",https://www.example.com/bad-vendor-scrape,,,,,,
3,"Metro~error@^",Metro,error,,,,,
"""

# mirrors real corruption from the Hammer files: a bad date, a bad price
RAW_CSV = """nowtime,current_price,old_price,price_per_unit,other,product_id
2024-06-22 10:35:40,0.99,1.29,$0.99/item,SALE,1
-4713-11-24 12:00:00,1.50,1.50,$1.50/item,,1
2024-06-22 10:36:00,not_a_number,2.00,$2.00/item,,1
"""


def test_load_products_drops_unrecognized_vendor(tmp_path):
    path = tmp_path / "product.csv"
    path.write_text(PRODUCTS_CSV)

    df = load_products(path)

    assert 2 not in list(df["id"])


def test_load_products_drops_error_placeholder_name(tmp_path):
    path = tmp_path / "product.csv"
    path.write_text(PRODUCTS_CSV)

    df = load_products(path)

    assert 3 not in list(df["id"])


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

    # check_categorical=False: parquet changes the categories' backing
    # dtype on round trip, same values though
    pd.testing.assert_frame_equal(first, second, check_categorical=False)


CENTS_BUG_PRODUCTS_CSV = """id,concatted,vendor,product_name,units,brand,detail_url,sku,upc
1,"Voila~Bugged Item@500g^",Voila,Bugged Item,500g,,,,
2,"Voila~Clean Item@200g^",Voila,Clean Item,200g,,,,
3,"Voila~Cents Notation Item@473ml^",Voila,Cents Notation Item,473ml,,,,
"""

# product 1's real price is $27.50, ($5.50/100g x 500g), but current_price
# is stored as 2750.0, a 100x cents scale. product 2 is already correct.
# product 3 mirrors a real Hammer row: price_per_unit uses the cents sign
# instead of a dollar sign ("74c/100ml"), real price is $3.48, stored as 348.0.
CENTS_BUG_RAW_CSV = """nowtime,current_price,old_price,price_per_unit,other,product_id
2024-06-22 10:35:40,2750.0,,$5.50/100g,,1
2024-06-22 10:35:40,6.00,,$3.00/100g,,2
2024-06-22 10:35:40,348.0,,74¢/100ml,,3
"""


def test_fix_cents_bug_corrects_scaled_price(tmp_path):
    products_path = tmp_path / "product.csv"
    products_path.write_text(CENTS_BUG_PRODUCTS_CSV)
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(CENTS_BUG_RAW_CSV)

    products = load_products(products_path)
    prices = load_raw_prices(raw_path)
    fixed = fix_cents_bug(prices, products)

    bugged = fixed[fixed["product_id"] == 1]
    assert bugged["current_price"].iloc[0] == pytest.approx(27.50)


def test_fix_cents_bug_leaves_correct_price_alone(tmp_path):
    products_path = tmp_path / "product.csv"
    products_path.write_text(CENTS_BUG_PRODUCTS_CSV)
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(CENTS_BUG_RAW_CSV)

    products = load_products(products_path)
    prices = load_raw_prices(raw_path)
    fixed = fix_cents_bug(prices, products)

    clean = fixed[fixed["product_id"] == 2]
    assert clean["current_price"].iloc[0] == pytest.approx(6.00)


def test_fix_cents_bug_handles_cents_sign_notation(tmp_path):
    products_path = tmp_path / "product.csv"
    products_path.write_text(CENTS_BUG_PRODUCTS_CSV)
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(CENTS_BUG_RAW_CSV)

    products = load_products(products_path)
    prices = load_raw_prices(raw_path)
    fixed = fix_cents_bug(prices, products)

    bugged = fixed[fixed["product_id"] == 3]
    assert bugged["current_price"].iloc[0] == pytest.approx(3.48)


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
