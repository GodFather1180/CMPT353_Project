"""Parse package size out of messy Hammer `units` / product name text.

Takes a pandas Series of raw text (from product.csv's `units` column, or a
product name string) and returns a DataFrame with one row per input,
giving the total normalized quantity, the unit it's in, and how many
sub-units make up a multipack.

quantity is the total amount you get for the price, already normalized,
so "10x88ml" becomes 880.0, not 88.0. unit is one of "g", "ml", or
"each". pack_count is how many individual pieces are in the package, so
2 for "2 x 500g" and 1 for a plain "500g". Rows that don't look like a
real package size at all (garbage text, empty strings) come back as NaN
across all three columns rather than raising, so a caller can just check
how many rows failed to parse with isna() instead of writing error
handling around every call.

Everything below works on the whole Series at once with regex extraction
and dict based lookups, not a Python loop over rows, since the course
grades on vectorized pandas and 300,000+ rows through a per row loop
would also just be slow.
"""

import re

import numpy as np
import pandas as pd

# token -> (normalized unit, how many of the normalized unit it's worth).
# oz and lb are treated as weight here since that's what shows up on food
# and personal care packaging in this data (a lip balm listed as
# "0.6 ounce" is clearly a weight, not fluid ounces). If fluid ounces ever
# turn up in the data we will need a separate "fl oz" entry for volume.
_UNIT_INFO = {
    "g": ("g", 1.0),
    "gram": ("g", 1.0),
    "grams": ("g", 1.0),
    "gm": ("g", 1.0),
    "kg": ("g", 1000.0),
    "kilogram": ("g", 1000.0),
    "kilograms": ("g", 1000.0),
    "mg": ("g", 0.001),
    "milligram": ("g", 0.001),
    "milligrams": ("g", 0.001),
    "oz": ("g", 28.349523125),
    "ounce": ("g", 28.349523125),
    "ounces": ("g", 28.349523125),
    "lb": ("g", 453.59237),
    "lbs": ("g", 453.59237),
    "pound": ("g", 453.59237),
    "pounds": ("g", 453.59237),
    "ml": ("ml", 1.0),
    "millilitre": ("ml", 1.0),
    "millilitres": ("ml", 1.0),
    "milliliter": ("ml", 1.0),
    "milliliters": ("ml", 1.0),
    "l": ("ml", 1000.0),
    "litre": ("ml", 1000.0),
    "litres": ("ml", 1000.0),
    "liter": ("ml", 1000.0),
    "liters": ("ml", 1000.0),
    "each": ("each", 1.0),
    "ea": ("each", 1.0),
    "count": ("each", 1.0),
    "ct": ("each", 1.0),
    "unit": ("each", 1.0),
    "units": ("each", 1.0),
    "per pack": ("each", 1.0),
}

_TOKEN_TO_CATEGORY = {token: info[0] for token, info in _UNIT_INFO.items()}
_TOKEN_TO_FACTOR = {token: info[1] for token, info in _UNIT_INFO.items()}

# "per pack" is the only token with a space in it, so it needs its own
# piece of the pattern with \s+ instead of a literal space. Everything
# else can just be escaped and joined with |. Longest tokens first so a
# token like "grams" gets a chance to match before the shorter "g" does.
_SIMPLE_TOKENS = sorted((t for t in _UNIT_INFO if " " not in t), key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(t) for t in _SIMPLE_TOKENS) + r"|per\s+pack"
_UNIT_GROUP = rf"(?:{_UNIT_PATTERN})"

# multipack has to be checked before the single quantity pattern, because
# something like "10x88ml" would otherwise still match the single pattern
# on its own (it would find "88ml" and miss the "10x" multiplier entirely).
_MULTIPACK_RE = rf"(?P<count>\d+)\s*[x×]\s*(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_GROUP})\b"
_SINGLE_RE = rf"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_GROUP})\b"


def parse_package_sizes(units: pd.Series) -> pd.DataFrame:
    text = units.astype("string")
    lower = text.str.lower()

    # a couple of rows in product.csv have retailer price info like
    # "$46.28/1kg $21.00/1lb" leaked into the units column instead of a
    # real package size. There's a number and a unit token in there, but
    # trusting it would just be guessing, so anything with a dollar sign
    # gets treated as unparseable rather than parsed.
    has_dollar_sign = lower.str.contains(r"\$", regex=True, na=False)

    multipack = lower.str.extract(_MULTIPACK_RE)
    single = lower.str.extract(_SINGLE_RE)

    multipack_unit = multipack["unit"].str.replace(r"\s+", " ", regex=True)
    single_unit = single["unit"].str.replace(r"\s+", " ", regex=True)

    multipack_category = multipack_unit.map(_TOKEN_TO_CATEGORY)
    multipack_factor = multipack_unit.map(_TOKEN_TO_FACTOR)
    single_category = single_unit.map(_TOKEN_TO_CATEGORY)
    single_factor = single_unit.map(_TOKEN_TO_FACTOR)

    multipack_count = pd.to_numeric(multipack["count"], errors="coerce")
    multipack_qty = pd.to_numeric(multipack["qty"], errors="coerce")
    single_qty = pd.to_numeric(single["qty"], errors="coerce")

    multipack_valid = multipack_count.notna() & multipack_qty.notna() & multipack_category.notna()
    single_valid = single_qty.notna() & single_category.notna()

    multipack_quantity = multipack_count * multipack_qty * multipack_factor
    multipack_pack_count = multipack_count

    single_quantity = single_qty * single_factor
    # for a pure count like "5 each", the pack count is just the quantity
    # itself. for anything else (a plain "500g"), it's a single item.
    single_pack_count = single_quantity.where(single_category == "each", other=1.0)

    # multipack wins over single when both matched, since a multipack
    # string like "10x88ml" also happens to contain a valid looking single
    # match ("88ml") that we don't want.
    quantity = single_quantity.where(single_valid, other=np.nan)
    quantity = quantity.where(~multipack_valid, other=multipack_quantity)
    quantity = quantity.where(~has_dollar_sign, other=np.nan)

    unit = single_category.where(single_valid, other=np.nan)
    unit = unit.where(~multipack_valid, other=multipack_category)
    unit = unit.where(~has_dollar_sign, other=np.nan)

    pack_count = single_pack_count.where(single_valid, other=np.nan)
    pack_count = pack_count.where(~multipack_valid, other=multipack_pack_count)
    pack_count = pack_count.where(~has_dollar_sign, other=np.nan)

    # pd.to_numeric on a nullable "string" dtype column hands back a
    # nullable Float64 column (pd.NA for missing) instead of a plain
    # float64 one (np.nan for missing). Casting to plain float64 here
    # keeps the output an ordinary numpy float column either way.
    quantity = quantity.astype("float64")
    pack_count = pack_count.astype("float64")

    return pd.DataFrame({"quantity": quantity, "unit": unit, "pack_count": pack_count})
