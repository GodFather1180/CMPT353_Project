import math

import pandas as pd
import pytest

from package_parser import parse_package_sizes

# Each case is (input text, expected quantity, expected unit, expected pack count).
# expected values are None where we want the row to come back unparsed (NaN
# in every column). Quantity is always the total amount for the price, so a
# multipack like "10x88ml" should come out as 880.0 grams of stuff, not 88.

CLEAN_METRIC = [
    ("500g", 500.0, "g", 1),
    ("120g", 120.0, "g", 1),
    ("204g", 204.0, "g", 1),
    ("1.5l", 1500.0, "ml", 1),
    ("1.36 litre", 1360.0, "ml", 1),
    ("4L", 4000.0, "ml", 1),
]

COUNT_BASED = [
    ("12ea", 12.0, "each", 12),
    ("5 each", 5.0, "each", 5),
    ("6 each", 6.0, "each", 6),
    ("4 per pack", 4.0, "each", 4),
]

MULTIPACK = [
    ("10x88ml", 880.0, "ml", 10),
    ("2 x 500g", 1000.0, "g", 2),
]

IMPERIAL = [
    ("0.6 ounce", pytest.approx(17.01, abs=0.01), "g", 1),
]

EMBEDDED_IN_PRODUCT_NAME = [
    ("Milk 2% 4L", 4000.0, "ml", 1),
    ("Cheese 450g", 450.0, "g", 1),
]

# These are real values pulled straight from product.csv's `units` column,
# not made up. "foil wrapped4 each" and the two "$.../1kg $.../1lb" strings
# showed up when we sampled the data earlier.
REAL_MESSY_VALUES = [
    ("foil wrapped4 each", 4.0, "each", 4),
    ("$46.28/1kg $21.00/1lb", None, None, None),
    ("$1.96/1kg $0.89/1lb", None, None, None),
    ("betty crocker super moist lemon cake mix", None, None, None),
]

ALL_CASES = (
    CLEAN_METRIC
    + COUNT_BASED
    + MULTIPACK
    + IMPERIAL
    + EMBEDDED_IN_PRODUCT_NAME
    + REAL_MESSY_VALUES
)


@pytest.mark.parametrize("text, expected_quantity, expected_unit, expected_pack_count", ALL_CASES)
def test_parse_single_value(text, expected_quantity, expected_unit, expected_pack_count):
    result = parse_package_sizes(pd.Series([text]))

    if expected_quantity is None:
        assert math.isnan(result["quantity"].iloc[0])
        assert pd.isna(result["unit"].iloc[0])
        assert math.isnan(result["pack_count"].iloc[0])
    else:
        assert result["quantity"].iloc[0] == expected_quantity
        assert result["unit"].iloc[0] == expected_unit
        assert result["pack_count"].iloc[0] == expected_pack_count


def test_empty_string_is_unparsed():
    result = parse_package_sizes(pd.Series([""]))
    assert math.isnan(result["quantity"].iloc[0])


def test_missing_value_is_unparsed():
    result = parse_package_sizes(pd.Series([None]))
    assert math.isnan(result["quantity"].iloc[0])


def test_result_keeps_input_index():
    # so parsed columns can be assigned straight back onto product.csv
    series = pd.Series(["500g", "12ea"], index=[7, 42])
    result = parse_package_sizes(series)
    assert list(result.index) == [7, 42]


def test_operates_on_whole_series_at_once():
    # a stand-in for the "no explicit loops" course rule: this just checks
    # the function accepts a multi-row Series and returns one row per input,
    # not that it's fast, but a real per-row loop would still pass this
    result = parse_package_sizes(pd.Series(["500g", "1.5l", "not a size"]))
    assert len(result) == 3
