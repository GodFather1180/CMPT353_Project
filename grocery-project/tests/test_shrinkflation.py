import numpy as np
import pandas as pd

from shrinkflation import (
    fit_price_change_models,
    find_shrinkflation_candidates,
    find_stable_pairs,
    run_significance_test,
    run_vendor_comparison,
)

PRODUCTS_CSV = """id,concatted,vendor,product_name,units,brand,detail_url,sku,upc
1,x,Voila,Chips Original,500g,,,,
2,x,Voila,Chips Original,400g,,,,
3,x,Voila,Salsa Hot,650ml,,,,
4,x,Voila,Salsa Hot,650ml,,,,
5,x,Voila,Egg Whites,1g,,,,
6,x,Voila,Egg Whites,500g,,,,
7,x,Voila,Rice Bag,500g,,,,
8,x,Voila,Rice Bag,500g,,,,
"""

RAW_CSV = """nowtime,current_price,old_price,price_per_unit,other,product_id
2024-01-01 10:00:00,3.00,,$0.60/100g,,1
2024-01-10 10:00:00,3.00,,$0.75/100g,,2
2024-01-01 10:00:00,4.00,,$0.62/100ml,,3
2024-06-01 10:00:00,4.00,,$0.62/100ml,,3
2024-01-01 10:00:00,4.00,,$0.62/100ml,,4
2024-06-01 10:00:00,4.00,,$0.62/100ml,,4
2024-01-01 10:00:00,1.00,,$100.00/100g,,5
2024-01-05 10:00:00,1.00,,$100.00/100g,,5
2024-01-01 10:00:00,5.00,,$1.00/100g,,6
2024-01-01 10:00:00,0.0001,,$0.00/100g,,7
2024-01-05 10:00:00,5.00,,$1.00/100g,,8
2024-01-05 10:00:00,5.00,,$1.00/100g,,6
"""


def _load():
    import io

    from data_loading import load_products, load_raw_prices

    products = load_products(io.StringIO(PRODUCTS_CSV))
    prices = load_raw_prices(io.StringIO(RAW_CSV))
    return products, prices


def test_finds_a_real_shrinkflation_pair():
    products, prices = _load()
    result = find_shrinkflation_candidates(products, prices)

    pair = result[(result["first_id"] == 1) & (result["second_id"] == 2)]
    assert len(pair) == 1
    assert pair["size_change_pct"].iloc[0] < 0


def test_excludes_coexisting_duplicates():
    products, prices = _load()
    result = find_shrinkflation_candidates(products, prices)

    # ids 3 and 4 are the same size and scraped almost simultaneously,
    # not a replacement
    coexisting = result[
        result["first_id"].isin([3, 4]) & result["second_id"].isin([3, 4])
    ]
    assert len(coexisting) == 0


def test_excludes_implausible_size_ratio():
    products, prices = _load()
    result = find_shrinkflation_candidates(products, prices)

    # ids 5 and 6 are the "1g egg whites" source data error, a 500x ratio
    garbage = result[
        result["first_id"].isin([5, 6]) & result["second_id"].isin([5, 6])
    ]
    assert len(garbage) == 0


def test_find_stable_pairs_finds_same_size_match():
    products, prices = _load()
    result = find_stable_pairs(products, prices)

    pair = result[result["first_id"].isin([3, 4]) & result["second_id"].isin([3, 4])]
    assert len(pair) == 1
    assert pair["size_change_pct"].iloc[0] == 0


def test_find_stable_pairs_excludes_the_shrink_pair():
    products, prices = _load()
    result = find_stable_pairs(products, prices)

    shrink_pair = result[result["first_id"].isin([1, 2]) & result["second_id"].isin([1, 2])]
    assert len(shrink_pair) == 0


def test_excludes_implausible_price_per_unit():
    products, prices = _load()
    result = find_stable_pairs(products, prices)

    # ids 7 and 8: $0.0001 for 500g is a corrupted near-zero price, real
    # data produced a 999,900% change from a pair shaped just like this
    garbage = result[result["first_id"].isin([7, 8]) & result["second_id"].isin([7, 8])]
    assert len(garbage) == 0


def test_significance_uses_mannwhitney_for_non_normal_data():
    rng = np.random.default_rng(0)
    shrink = rng.exponential(scale=2.0, size=200) + 5
    control = rng.exponential(scale=2.0, size=200)

    result = run_significance_test(shrink, control)

    assert result["test"] == "mannwhitneyu"
    assert result["significant"] is True


def test_significance_uses_ttest_for_normal_data():
    rng = np.random.default_rng(1)
    shrink = rng.normal(loc=20, scale=5, size=200)
    control = rng.normal(loc=0, scale=5, size=200)

    result = run_significance_test(shrink, control)

    assert result["test"] == "ttest_ind"
    assert result["significant"] is True


def test_significance_not_significant_when_no_real_difference():
    rng = np.random.default_rng(2)
    shrink = rng.normal(loc=0, scale=5, size=200)
    control = rng.normal(loc=0, scale=5, size=200)

    result = run_significance_test(shrink, control)

    assert result["significant"] is False


def test_fit_price_change_models_identifies_size_change_as_predictive():
    rng = np.random.default_rng(0)
    n = 300
    size_change_pct = rng.uniform(-80, 0, n)
    noise = rng.normal(0, 5, n)
    price_per_unit_change_pct = -1.5 * size_change_pct + noise

    pairs = pd.DataFrame({
        "size_change_pct": size_change_pct,
        "price_per_unit_change_pct": price_per_unit_change_pct,
        "vendor": rng.choice(["A", "B", "C"], n),
        "unit": rng.choice(["g", "ml"], n),
        "first_quantity": rng.uniform(100, 1000, n),
        "first_price_per_unit": rng.uniform(0.01, 1.0, n),
    })

    result = fit_price_change_models(pairs)

    assert result["linear_test_r2"] > 0.5
    top_feature = max(result["forest_importances"], key=result["forest_importances"].get)
    assert top_feature == "size_change_pct"


def _make_clean_pairs(rng, n=300):
    size_change_pct = rng.uniform(-80, 0, n)
    noise = rng.normal(0, 5, n)
    price_per_unit_change_pct = -1.5 * size_change_pct + noise
    return pd.DataFrame({
        "size_change_pct": size_change_pct,
        "price_per_unit_change_pct": price_per_unit_change_pct,
        "vendor": rng.choice(["A", "B", "C"], n),
        "unit": rng.choice(["g", "ml"], n),
        "first_quantity": rng.uniform(100, 1000, n),
        "first_price_per_unit": rng.uniform(0.01, 1.0, n),
    })


def test_fit_price_change_models_trims_extreme_outlier():
    rng = np.random.default_rng(3)
    pairs = _make_clean_pairs(rng)
    # one row with a wildly implausible target, like the confirmed
    # measurement-error cases found in the real data
    outlier = pd.DataFrame([{
        "size_change_pct": -50.0,
        "price_per_unit_change_pct": 50000.0,
        "vendor": "A",
        "unit": "g",
        "first_quantity": 500.0,
        "first_price_per_unit": 0.01,
    }])
    pairs = pd.concat([pairs, outlier], ignore_index=True)

    result = fit_price_change_models(pairs)

    assert result["n"] < len(pairs)
    assert result["linear_test_r2"] > 0.5


def test_fit_price_change_models_regularizes_forest():
    rng = np.random.default_rng(4)
    pairs = _make_clean_pairs(rng)

    result = fit_price_change_models(pairs, forest_max_depth=4, forest_min_samples_leaf=10)

    assert result["forest_params"]["max_depth"] == 4
    assert result["forest_params"]["min_samples_leaf"] == 10


def test_vendor_comparison_finds_real_difference():
    rng = np.random.default_rng(5)
    # non-normal (exponential), and vendor C is shifted well above the rest
    candidates = pd.concat([
        pd.DataFrame({
            "vendor": v,
            "price_per_unit_change_pct": rng.exponential(scale=5.0, size=50) + shift,
        })
        for v, shift in [("A", 0), ("B", 0), ("C", 40)]
    ], ignore_index=True)

    result = run_vendor_comparison(candidates)

    assert result["test"] == "kruskal"
    assert result["significant"] is True


def test_vendor_comparison_no_difference():
    rng = np.random.default_rng(6)
    candidates = pd.concat([
        pd.DataFrame({
            "vendor": v,
            "price_per_unit_change_pct": rng.exponential(scale=5.0, size=50),
        })
        for v in ["A", "B", "C"]
    ], ignore_index=True)

    result = run_vendor_comparison(candidates)

    assert result["significant"] is False
