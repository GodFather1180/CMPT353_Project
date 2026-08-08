# Shrinkflation in Canadian groceries: CMPT 353 project

**Question:** Are there products whose package size shrank while price held
steady or rose ("shrinkflation")?
Does the effect differ by grocery chain?
How does our own price index compare to StatCan's
official food CPI over the same period?

## Data

- **Project Hammer** (https://jacobfilipp.com/hammer/): historical scraped
  daily prices from Loblaws, No Frills, Metro, Voila, T&T, Walmart Canada,
  Save-On-Foods, and Galleria. Two files: `product.csv` (one row per
  product listing) and `raw.csv` (one row per price observation, ~70
  million rows, 3.5GB).
- **StatCan table 18-10-0004-01**: monthly Consumer Price Index, used as
  an external benchmark for our own price index. Downloaded from StatCan
  with the Food product group and Canada geography included.

The full data is too large to commit. Small real samples in the exact
format the code expects are in [`sample_data/`](sample_data/README.md),
with a README describing what each file demonstrates.

## Requirements

Python 3.11+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

(pandas, numpy, scipy, scikit-learn, matplotlib, pyarrow, pytest)

## Setup

Download the Hammer CSVs and the StatCan table, then place them so the
tree looks like:

```
data/
  hammer-5-csv/
    hammer-4-product.csv
    hammer-4-raw.csv
  1810000401_databaseLoadingData-2.csv
```

The StatCan filename is what their site's "download selected data" option
produces for table 18-10-0004-01.

## Running

All commands run from this directory (`grocery-project/`).

The full analysis (candidate detection, significance test, regression,
vendor comparison, and the price index vs CPI table), printed to stdout:

```bash
python3 src/run_analysis.py
```

The first run parses the 3.5GB `raw.csv` and caches it as parquet next to
the CSV, which takes a few minutes and a few GB of RAM. Later runs load
the cache and are much faster.

All plots, written to `results/plots/`:

```bash
python3 src/generate_plots.py
```

Tests:

```bash
python3 -m pytest tests/
```

