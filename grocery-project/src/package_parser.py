"""Parse package size out of messy Hammer `units` / product name text.

The plan is to take a pandas Series of raw text (from product.csv's
`units` column, or a product name string) and return a DataFrame with one
row per input, giving the total normalized quantity, the unit it's in,
and how many sub-units make up a multipack.

quantity is the total amount you get for the price, already normalized,
so "10x88ml" becomes 880.0, not 88.0. unit is one of "g", "ml", or
"each". pack_count is how many individual pieces are in the package, so
2 for "2 x 500g" and 1 for a plain "500g". Rows that don't look like a
real package size at all (garbage text, empty strings) come back as NaN
across all three columns rather than raising, so a caller can just check
how many rows failed to parse with isna() instead of writing error
handling around every call.

Not implemented yet. Tests are written first so the edge cases are
pinned down before any parsing logic exists.
"""

import pandas as pd


def parse_package_sizes(units: pd.Series) -> pd.DataFrame:
    raise NotImplementedError("tests are written, implementation comes next")
