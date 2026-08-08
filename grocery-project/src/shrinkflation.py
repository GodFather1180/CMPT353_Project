"""Find shrinkflation candidates: same product, smaller size, replaced in place.

product.csv's `id` never changes size, so a real shrinkflation event has to
show up as one id disappearing and a different, smaller-size id for the
same product taking over, within the same vendor. That's what this looks
for: same-vendor name matches, a real size decrease, and a clean handoff
in time rather than two listings that just happened to coexist.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_loading import MIN_PLAUSIBLE_PRICE_PER_UNIT
from package_parser import parse_package_sizes
from product_matching import find_best_matches

_FEATURES = ["size_change_pct", "first_quantity", "first_price_per_unit", "vendor", "unit"]
_TARGET = "price_per_unit_change_pct"


def _build_matched_pairs(products: pd.DataFrame, prices: pd.DataFrame, match_threshold: float) -> pd.DataFrame:
    """Same-vendor, cross-id name matches with size, timing, and price
    joined in. No shrink/timing filtering yet, that's the caller's job.
    """
    products = products.join(parse_package_sizes(products["units"]))
    products = products.dropna(subset=["quantity", "unit", "product_name"])

    active = prices.groupby("product_id")["nowtime"].agg(first_active="min", last_active="max")
    avg_price = prices.groupby("product_id")["current_price"].mean().rename("avg_price")

    pairs = []
    for vendor, group in products.groupby("vendor"):
        names = group.set_index("id")["product_name"]
        matches = find_best_matches(names, names, threshold=match_threshold, exclude_self_index=True)
        matches = matches.dropna(subset=["match_index"])
        matches["a_id"] = matches.index
        matches["b_id"] = matches["match_index"].astype("int64")
        matches["vendor"] = vendor
        pairs.append(matches[["a_id", "b_id", "score", "vendor"]])

    if not pairs:
        return _empty_result()
    pairs = pd.concat(pairs, ignore_index=True)

    # a_id/b_id show up in both directions (a->b and b->a), keep one
    pair_key = pairs[["a_id", "b_id"]].min(axis=1).astype(str) + "-" + pairs[["a_id", "b_id"]].max(axis=1).astype(str)
    pairs = pairs.loc[~pair_key.duplicated()]

    sizes = products.set_index("id")[["quantity", "unit"]]
    avg_price_df = avg_price.to_frame()
    for side in ("a", "b"):
        pairs = pairs.join(sizes.add_suffix(f"_{side}"), on=f"{side}_id")
        pairs = pairs.join(active.add_suffix(f"_{side}"), on=f"{side}_id")
        pairs = pairs.join(avg_price_df.add_suffix(f"_{side}"), on=f"{side}_id")

    pairs = pairs.dropna(
        subset=["quantity_a", "quantity_b", "avg_price_a", "avg_price_b", "first_active_a", "first_active_b"]
    )
    pairs = pairs[pairs["unit_a"] == pairs["unit_b"]]
    pairs = pairs[(pairs["avg_price_a"] > 0) & (pairs["avg_price_b"] > 0)]

    if pairs.empty:
        return _empty_result()

    # order chronologically: "first" is whichever id started being scraped earlier
    b_is_earlier = pairs["first_active_b"] < pairs["first_active_a"]
    result = pd.DataFrame(index=pairs.index)
    for field in ("id", "quantity", "first_active", "last_active", "avg_price"):
        a_col = "a_id" if field == "id" else f"{field}_a"
        b_col = "b_id" if field == "id" else f"{field}_b"
        result[f"first_{field}"] = np.where(b_is_earlier, pairs[b_col], pairs[a_col])
        result[f"second_{field}"] = np.where(b_is_earlier, pairs[a_col], pairs[b_col])
    result["score"] = pairs["score"]
    result["vendor"] = pairs["vendor"]
    result["unit"] = pairs["unit_a"]  # unit_a == unit_b, already filtered above

    result["gap_days"] = (
        pd.to_datetime(result["second_first_active"]) - pd.to_datetime(result["first_last_active"])
    ).dt.days
    result["size_change_pct"] = (
        (result["second_quantity"] - result["first_quantity"]) / result["first_quantity"] * 100
    )
    result["first_price_per_unit"] = result["first_avg_price"] / result["first_quantity"]
    result["second_price_per_unit"] = result["second_avg_price"] / result["second_quantity"]

    plausible_price = (
        (result["first_price_per_unit"] >= MIN_PLAUSIBLE_PRICE_PER_UNIT)
        & (result["second_price_per_unit"] >= MIN_PLAUSIBLE_PRICE_PER_UNIT)
    )
    result = result[plausible_price]

    result["price_per_unit_change_pct"] = (
        (result["second_price_per_unit"] - result["first_price_per_unit"]) / result["first_price_per_unit"] * 100
    )
    return result


def find_shrinkflation_candidates(
    products: pd.DataFrame,
    prices: pd.DataFrame,
    match_threshold: float = 0.9,
    gap_days_threshold: float = 14,
    size_ratio_threshold: float = 5.0,
) -> pd.DataFrame:
    result = _build_matched_pairs(products, prices, match_threshold)
    if result.empty:
        return result

    size_ratio = result["second_quantity"] / result["first_quantity"]
    plausible_ratio = size_ratio.between(1 / size_ratio_threshold, size_ratio_threshold)
    clean_handoff = result["gap_days"].abs() <= gap_days_threshold
    is_shrink = result["size_change_pct"] < 0

    return result[plausible_ratio & clean_handoff & is_shrink].reset_index(drop=True)


def find_stable_pairs(
    products: pd.DataFrame, prices: pd.DataFrame, match_threshold: float = 0.9
) -> pd.DataFrame:
    """Matched pairs with no real size difference, a control group for
    comparing against find_shrinkflation_candidates.
    """
    result = _build_matched_pairs(products, prices, match_threshold)
    if result.empty:
        return result
    return result[result["size_change_pct"] == 0].reset_index(drop=True)


def run_significance_test(shrink_change_pct, control_change_pct, alpha: float = 0.05) -> dict:
    """Is price_per_unit_change_pct higher for the shrink group than control?

    Checks normality (normaltest) and equal variance (levene) first, then
    picks ttest_ind or the non-parametric Mann-Whitney U accordingly.
    """
    shrink = pd.Series(shrink_change_pct).replace([np.inf, -np.inf], np.nan).dropna()
    control = pd.Series(control_change_pct).replace([np.inf, -np.inf], np.nan).dropna()

    shrink_normal = stats.normaltest(shrink).pvalue > alpha
    control_normal = stats.normaltest(control).pvalue > alpha
    equal_variance = stats.levene(shrink, control).pvalue > alpha

    if shrink_normal and control_normal:
        test_name = "ttest_ind"
        outcome = stats.ttest_ind(shrink, control, equal_var=equal_variance, alternative="greater")
    else:
        test_name = "mannwhitneyu"
        outcome = stats.mannwhitneyu(shrink, control, alternative="greater")

    return {
        "test": test_name,
        "statistic": outcome.statistic,
        "p_value": outcome.pvalue,
        "significant": bool(outcome.pvalue < alpha),
        "shrink_median": shrink.median(),
        "control_median": control.median(),
        "shrink_n": len(shrink),
        "control_n": len(control),
    }


def fit_price_change_models(
    pairs: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 0,
    forest_max_depth: int = 6,
    forest_min_samples_leaf: int = 20,
) -> dict:
    """Fit LinearRegression and RandomForestRegressor on what predicts
    price_per_unit_change_pct: size_change_pct, starting size/price, and
    vendor/unit as one-hot categories.

    Rows past the standard box plot bounds (1.5x IQR) on the target are
    dropped.
    """
    data = pairs[_FEATURES + [_TARGET]].replace([np.inf, -np.inf], np.nan).dropna()

    q1, q3 = data[_TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    data = data[data[_TARGET].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

    # get_dummies returns bool columns, which cause overflow warnings
    # once mixed with the float columns in sklearn's matrix math
    X = pd.get_dummies(data[_FEATURES], columns=["vendor", "unit"], drop_first=True).astype("float64")
    y = data[_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    forest = RandomForestRegressor(
        max_depth=forest_max_depth,
        min_samples_leaf=forest_min_samples_leaf,
        random_state=random_state,
    )
    forest.fit(X_train, y_train)

    return {
        "n": len(data),
        "features": list(X.columns),
        "forest_params": {"max_depth": forest_max_depth, "min_samples_leaf": forest_min_samples_leaf},
        "linear_train_r2": linear.score(X_train, y_train),
        "linear_test_r2": linear.score(X_test, y_test),
        "linear_coefficients": dict(zip(X.columns, linear.coef_)),
        "forest_train_r2": forest.score(X_train, y_train),
        "forest_test_r2": forest.score(X_test, y_test),
        "forest_importances": dict(zip(X.columns, forest.feature_importances_)),
    }


def run_vendor_comparison(candidates: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Does price_per_unit_change_pct differ by vendor?

    Checks normality per vendor first, then picks one-way ANOVA if every
    group looks normal, otherwise Kruskal-Wallis.
    """
    data = candidates[["vendor", "price_per_unit_change_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = {vendor: g["price_per_unit_change_pct"].to_numpy() for vendor, g in data.groupby("vendor")}
    # normaltest needs at least 8 values
    groups = {vendor: values for vendor, values in groups.items() if len(values) >= 8}

    all_normal = all(stats.normaltest(values).pvalue > alpha for values in groups.values())

    if all_normal:
        test_name = "f_oneway"
        outcome = stats.f_oneway(*groups.values())
    else:
        test_name = "kruskal"
        outcome = stats.kruskal(*groups.values())

    return {
        "test": test_name,
        "statistic": outcome.statistic,
        "p_value": outcome.pvalue,
        "significant": bool(outcome.pvalue < alpha),
        "medians_by_vendor": {vendor: float(np.median(values)) for vendor, values in groups.items()},
        "n_by_vendor": {vendor: len(values) for vendor, values in groups.items()},
    }


def _empty_result() -> pd.DataFrame:
    columns = [
        "first_id", "second_id", "first_quantity", "second_quantity",
        "gap_days", "size_change_pct", "price_per_unit_change_pct", "score", "vendor", "unit",
    ]
    return pd.DataFrame(columns=columns)
