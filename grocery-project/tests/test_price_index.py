import pandas as pd
import pytest

from price_index import build_price_index


def _products(rows):
    return pd.DataFrame(rows, columns=["id", "units"])


def _prices(rows):
    df = pd.DataFrame(rows, columns=["product_id", "current_price", "nowtime"])
    df["nowtime"] = pd.to_datetime(df["nowtime"])
    return df


def test_index_tracks_a_clean_price_increase():
    products = _products([
        (1, "500g"),
        (2, "200g"),
    ])
    prices = _prices([
        (1, 5.00, "2024-01-15"),
        (1, 5.50, "2024-02-15"),
        (2, 2.00, "2024-01-20"),
        (2, 2.20, "2024-02-20"),
    ])

    result = build_price_index(products, prices)

    assert result["price_index"].iloc[0] == 100.0
    assert result["price_index"].iloc[1] == pytest.approx(110.0)


def test_index_excludes_implausible_price_per_unit():
    products = _products([
        (1, "500g"),
        (2, "200g"),
        (3, "500g"),
    ])
    prices = _prices([
        (1, 5.00, "2024-01-15"),
        (1, 5.50, "2024-02-15"),
        (2, 2.00, "2024-01-20"),
        (2, 2.20, "2024-02-20"),
        # corrupted: $0.0001 for 500g is far below the plausibility floor,
        # a real value like this would swing a mean by orders of magnitude
        (3, 0.0001, "2024-01-10"),
        (3, 5.00, "2024-02-10"),
    ])

    result = build_price_index(products, prices)

    # same +10% result as the clean case, product 3's corrupted month is
    # dropped rather than distorting the mean
    assert result["price_index"].iloc[1] == pytest.approx(110.0)


def test_index_starts_at_100():
    products = _products([(1, "500g")])
    prices = _prices([
        (1, 5.00, "2024-01-15"),
        (1, 6.00, "2024-02-15"),
    ])

    result = build_price_index(products, prices)

    assert result["price_index"].iloc[0] == 100.0
    assert result["month"].iloc[0] == pd.Timestamp("2024-01-01")


def test_index_uses_geometric_mean_so_bounces_net_out():
    # one product halves while the other doubles: geometric mean of the
    # relatives is 1, arithmetic would wrongly report +25%
    products = _products([
        (1, "500g"),
        (2, "500g"),
    ])
    prices = _prices([
        (1, 4.00, "2024-01-15"),
        (1, 2.00, "2024-02-15"),
        (2, 2.00, "2024-01-15"),
        (2, 4.00, "2024-02-15"),
    ])

    result = build_price_index(products, prices)

    assert result["price_index"].iloc[1] == pytest.approx(100.0)


def test_start_month_skips_earlier_months():
    products = _products([(1, "500g")])
    prices = _prices([
        (1, 5.00, "2024-01-15"),
        (1, 999.00, "2024-02-15"),  # a wild early month we want excluded
        (1, 6.00, "2024-03-15"),
        (1, 6.60, "2024-04-15"),
    ])

    result = build_price_index(products, prices, start_month="2024-03")

    assert result["month"].iloc[0] == pd.Timestamp("2024-03-01")
    assert result["price_index"].iloc[0] == 100.0
    assert result["price_index"].iloc[1] == pytest.approx(110.0)
