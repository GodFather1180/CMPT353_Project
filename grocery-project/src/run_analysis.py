"""Run the actual analysis and print the findings: the significance
test, the regression, the vendor comparison, and the price index vs
StatCan's Food CPI.

python3 src/run_analysis.py
"""

import pandas as pd

from data_loading import fix_cents_bug, load_products, load_raw_prices
from price_index import build_price_index
from shrinkflation import (
    find_shrinkflation_candidates,
    find_stable_pairs,
    fit_price_change_models,
    run_significance_test,
    run_vendor_comparison,
)
from statcan import load_food_cpi


def main():
    products = load_products("data/hammer-5-csv/hammer-4-product.csv")
    prices = load_raw_prices(
        "data/hammer-5-csv/hammer-4-raw.csv",
        cache_path="data/hammer-5-csv/raw_cache.parquet",
    )
    prices = fix_cents_bug(prices, products)

    candidates = find_shrinkflation_candidates(products, prices)
    control = find_stable_pairs(products, prices)
    print(f"\nshrinkflation candidates: {len(candidates):,}")
    print(f"control pairs (no size change): {len(control):,}")

    print("\n--- significance test: shrink vs control price_per_unit_change_pct ---")
    result = run_significance_test(
        candidates["price_per_unit_change_pct"], control["price_per_unit_change_pct"]
    )
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n--- regression: what predicts price_per_unit_change_pct ---")
    combined = pd.concat([candidates, control], ignore_index=True)
    model_result = fit_price_change_models(combined)
    print(f"  n: {model_result['n']:,}")
    print(f"  linear R2 (train/test): {model_result['linear_train_r2']:.4f} / {model_result['linear_test_r2']:.4f}")
    print(f"  forest R2 (train/test): {model_result['forest_train_r2']:.4f} / {model_result['forest_test_r2']:.4f}")
    print("  forest feature importances:")
    for feature, importance in sorted(model_result["forest_importances"].items(), key=lambda kv: -kv[1]):
        print(f"    {feature}: {importance:.4f}")

    print("\n--- does the effect differ by vendor? ---")
    vendor_result = run_vendor_comparison(candidates)
    print(f"  test: {vendor_result['test']}, p_value: {vendor_result['p_value']}, significant: {vendor_result['significant']}")
    print("  median price_per_unit_change_pct by vendor:")
    for vendor, median in sorted(vendor_result["medians_by_vendor"].items(), key=lambda kv: -kv[1]):
        n = vendor_result["n_by_vendor"][vendor]
        print(f"    {vendor}: {median:.2f}% (n={n})")

    print("\n--- our price index vs StatCan Food CPI ---")
    our_index = build_price_index(products, prices, start_month="2024-06")
    cpi = load_food_cpi("data/1810000401_databaseLoadingData-2.csv")
    cpi["month"] = cpi["date"].dt.to_period("M").dt.to_timestamp()
    merged = our_index.merge(cpi, on="month", how="inner")
    merged["cpi_rebased"] = merged["cpi"] / merged["cpi"].iloc[0] * 100
    print(merged[["month", "price_index", "cpi_rebased"]].to_string(index=False))


if __name__ == "__main__":
    main()
