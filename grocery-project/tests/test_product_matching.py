import pandas as pd
import pytest

from product_matching import is_match

# real product name pairs pulled from product.csv, not made up

TRUE_MATCHES = [
    # reordered words, same product
    ("Old El Paso Thick N' Chunky Salsa Hot 650 ml", "Old El Paso Salsa Thick N' Chunky Hot 650 ml"),
    # capitalization only
    ("MADE GOOD - Sea Salt Star Puffed Crackers, 121 Gram", "Made Good - Sea Salt Star Puffed Crackers, 121 Gram"),
    # same line, different size, the actual shrinkflation case
    ("ASSI CUT CRAB 2L", "ASSI CUT CRAB 3L"),
    # hyphen vs space
    ("1% Lactose Free Milk", "1% Lactose-Free Milk"),
    # trivial spacing difference in the size
    ("Kraft Peanut Butter Crunchy 500g", "Kraft Peanut Butter Crunchy 500 g"),
]

FALSE_MATCHES = [
    # same brand, same size, real different products
    ("Kraft Peanut Butter Crunchy 500g", "Kraft Peanut Butter Smooth 500g"),
    # a second qualifier word pair, not just crunchy/smooth
    ("President's Choice Salsa Original 400ml", "President's Choice Salsa Spicy 400ml"),
    # unrelated products
    ("Kraft Peanut Butter Smooth 500g", "Old El Paso Salsa Hot 650ml"),
]

ALL_CASES = [(a, b, True) for a, b in TRUE_MATCHES] + [(a, b, False) for a, b in FALSE_MATCHES]


@pytest.mark.parametrize("name_a, name_b, expected", ALL_CASES)
def test_is_match(name_a, name_b, expected):
    result = is_match(pd.Series([name_a]), pd.Series([name_b]))
    assert result.iloc[0] == expected


def test_operates_on_whole_series_at_once():
    a = pd.Series(["Kraft Peanut Butter Crunchy 500g", "Old El Paso Salsa Hot 650ml"])
    b = pd.Series(["Kraft Peanut Butter Smooth 500g", "Old El Paso Salsa Hot 650 ml"])
    result = is_match(a, b)
    assert list(result) == [False, True]


def test_result_keeps_input_index():
    a = pd.Series(["Old El Paso Salsa Hot 650ml"], index=[7])
    b = pd.Series(["Old El Paso Salsa Hot 650 ml"], index=[7])
    result = is_match(a, b)
    assert list(result.index) == [7]
