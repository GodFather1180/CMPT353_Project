"""Build the EDA plots for the price and package size data.

Run directly to regenerate every plot into results/plots:

python src/generate_plots.py

Each function here takes the already loaded data and an output directory,
so they can also be called on their own. Each also takes a log_scale
flag, since the linear view is worth seeing.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading import fix_cents_bug, load_products, load_raw_prices
from package_parser import parse_package_sizes
from price_index import build_price_index
from shrinkflation import find_shrinkflation_candidates, find_stable_pairs
from statcan import load_food_cpi

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"


def _bins(values, n, log_scale):
    if log_scale:
        return np.logspace(np.log10(values.min()), np.log10(values.max()), n)
    return np.linspace(values.min(), values.max(), n)


def plot_price_distribution(prices: pd.DataFrame, out_dir: Path, log_scale: bool = True) -> Path:
    positive_price = prices.loc[prices["current_price"] > 0, "current_price"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = _bins(positive_price, 100, log_scale)
    scale_label = "log scale" if log_scale else "linear"

    axes[0].hist(positive_price, bins=bins)
    axes[1].boxplot(positive_price, vert=False)
    for ax in axes:
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(f"current_price ($, {scale_label})")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"current_price distribution ({scale_label})")
    axes[1].set_title(f"current_price box plot ({scale_label})")

    fig.tight_layout()
    suffix = "" if log_scale else "_linear"
    out_path = out_dir / f"current_price_distribution{suffix}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_quantity_distribution(
    parsed_sizes: pd.DataFrame, out_dir: Path, log_scale: bool = True
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    categories = [("g", "grams"), ("ml", "millilitres")]
    scale_label = "log scale" if log_scale else "linear"

    for ax, (category, label) in zip(axes, categories):
        values = parsed_sizes.loc[parsed_sizes["unit"] == category, "quantity"]
        values = values[values > 0]
        ax.hist(values, bins=_bins(values, 80, log_scale))
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(f"quantity ({label}, {scale_label})")
        ax.set_ylabel("count")
        ax.set_title(f"parsed package size: {label}")

    fig.tight_layout()
    suffix = "" if log_scale else "_linear"
    out_path = out_dir / f"parsed_quantity_distribution{suffix}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_raw_vs_cleaned_price(
    raw_csv_path, cleaned_prices: pd.DataFrame, out_dir: Path, log_scale: bool = True
) -> Path:
    raw = pd.read_csv(raw_csv_path, usecols=["current_price"], dtype="string")
    raw_price = pd.to_numeric(raw["current_price"], errors="coerce")
    raw_price = raw_price[raw_price > 0]

    cleaned_price = cleaned_prices.loc[cleaned_prices["current_price"] > 0, "current_price"]
    scale_label = "log scale" if log_scale else "linear"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    combined = pd.concat([raw_price, cleaned_price])
    bins = _bins(combined, 100, log_scale)

    axes[0].hist(raw_price, bins=bins, color="tab:red")
    axes[1].hist(cleaned_price, bins=bins, color="tab:blue")
    for ax in axes:
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(f"current_price ($, {scale_label})")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"raw, uncleaned ({len(raw_price):,} rows)")
    axes[1].set_title(f"after load_raw_prices ({len(cleaned_price):,} rows)")

    fig.tight_layout()
    suffix = "" if log_scale else "_linear"
    out_path = out_dir / f"raw_vs_cleaned_price{suffix}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_shrink_vs_control(candidates: pd.DataFrame, control: pd.DataFrame, out_dir: Path) -> Path:
    """price_per_unit_change_pct for shrink candidates vs the same-size
    control group, the companion picture to the Mann-Whitney result.
    Clipped to -100..300 since a handful of extreme values would
    otherwise flatten the rest of the histogram.
    """
    shrink = candidates["price_per_unit_change_pct"].replace([np.inf, -np.inf], np.nan).dropna()
    stable = control["price_per_unit_change_pct"].replace([np.inf, -np.inf], np.nan).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(-100, 300, 80)

    axes[0].hist(shrink.clip(-100, 300), bins=bins, alpha=0.6, label="shrink candidates", color="tab:red")
    axes[0].hist(stable.clip(-100, 300), bins=bins, alpha=0.6, label="control (no size change)", color="tab:blue")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("price_per_unit_change_pct (clipped to -100..300)")
    axes[0].set_ylabel("count")
    axes[0].set_title("shrink vs control")
    axes[0].legend()

    axes[1].boxplot([shrink, stable], vert=False, tick_labels=["shrink", "control"])
    axes[1].set_xlabel("price_per_unit_change_pct, full range")
    axes[1].set_title("shrink vs control box plot")

    fig.tight_layout()
    out_path = out_dir / "shrink_vs_control.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_price_index_vs_cpi(our_index: pd.DataFrame, cpi: pd.DataFrame, out_dir: Path) -> Path:
    """Our own chained price index against StatCan's Food CPI, both
    rebased to 100 at the first shared month so the trends are
    comparable even though the raw index scales aren't.
    """
    cpi = cpi.copy()
    cpi["month"] = cpi["date"].dt.to_period("M").dt.to_timestamp()
    merged = our_index.merge(cpi, on="month", how="inner")
    merged["cpi_rebased"] = merged["cpi"] / merged["cpi"].iloc[0] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged["month"], merged["price_index"], label="our price index (Hammer data)", color="tab:red", linewidth=2)
    ax.plot(merged["month"], merged["cpi_rebased"], label="StatCan Food CPI (rebased)", color="tab:blue", linewidth=2)
    ax.axhline(100, color="black", linewidth=0.5)

    # GST/HST holiday: tax-inclusive CPI dips here, tax-exclusive shelf
    # prices do not
    ax.axvspan(
        pd.Timestamp("2024-12-14"), pd.Timestamp("2025-02-15"),
        color="gray", alpha=0.15, label="GST/HST holiday (tax-inclusive CPI dips)",
    )
    ax.set_ylabel("index (first month = 100)")
    ax.set_title("Our price index vs official Food CPI")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = out_dir / "price_index_vs_cpi.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_raw_prices(
        "data/hammer-5-csv/hammer-4-raw.csv",
        cache_path="data/hammer-5-csv/raw_cache.parquet",
    )
    products = load_products("data/hammer-5-csv/hammer-4-product.csv")
    parsed_sizes = parse_package_sizes(products["units"])

    raw_csv_path = "data/hammer-5-csv/hammer-4-raw.csv"
    for log_scale in (True, False):
        price_path = plot_price_distribution(prices, out_dir, log_scale)
        print(f"wrote {price_path}")
        quantity_path = plot_quantity_distribution(parsed_sizes, out_dir, log_scale)
        print(f"wrote {quantity_path}")
        raw_vs_cleaned_path = plot_raw_vs_cleaned_price(raw_csv_path, prices, out_dir, log_scale)
        print(f"wrote {raw_vs_cleaned_path}")

    cleaned_prices = fix_cents_bug(prices, products)
    candidates = find_shrinkflation_candidates(products, cleaned_prices)
    control = find_stable_pairs(products, cleaned_prices)
    shrink_vs_control_path = plot_shrink_vs_control(candidates, control, out_dir)
    print(f"wrote {shrink_vs_control_path}")

    # 2024-06: the first month with a stable, full-scale observation
    # count, earlier months are Hammer's scraping ramp-up period
    our_index = build_price_index(products, cleaned_prices, start_month="2024-06")
    cpi = load_food_cpi("data/1810000401_databaseLoadingData-2.csv")
    price_index_path = plot_price_index_vs_cpi(our_index, cpi, out_dir)
    print(f"wrote {price_index_path}")


if __name__ == "__main__":
    main()
