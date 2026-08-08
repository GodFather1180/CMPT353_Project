"""Parse package size out of messy Hammer `units` / product name text.

Takes a Series of raw text and returns a DataFrame with quantity, unit
("g", "ml", or "each"), and pack_count. quantity is already normalized,
so "10x88ml" becomes 880.0. Multipacks can be written either order,
"2 x 500g" or "60gx6". Rows that don't look like a real size just come
back as NaN instead of raising.
"""

import re

import numpy as np
import pandas as pd

# token -> (normalized unit, factor to convert to it).
# oz and lb are weight here, not fluid ounces, that's what the data shows.

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

# "per pack" is the only multi-word token, so it gets its own \s+ piece.
# longest tokens first so "grams" matches before the shorter "g" does.

_SIMPLE_TOKENS = sorted((t for t in _UNIT_INFO if " " not in t), key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(t) for t in _SIMPLE_TOKENS) + r"|per\s+pack"

UNIT_GROUP = rf"(?:{_UNIT_PATTERN})"

# multipack is checked before single, since "10x88ml" would otherwise
# match single as just "88ml" and lose the multiplier.

_MULTIPACK_RE = re.compile(
    rf"(?P<count>\d+)\s*[x×]\s*(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\b"
)
# reversed: "148gx4" instead of "4x148g"

_REVERSED_MULTIPACK_RE = re.compile(
    rf"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\s*[x×]\s*(?P<count>\d+)\b"
)
_SINGLE_RE = re.compile(rf"(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_GROUP})\b")


def _extract_match(lower: pd.Series, pattern: re.Pattern, has_count: bool):
    """Match one pattern, return validity, quantity, unit, pack_count.

    has_count is True for the multipack patterns, which have their own
    count group. False for single, where pack_count comes from whether
    the unit is "each" instead.
    """
    match = lower.str.extract(pattern)
    unit = match["unit"].str.replace(r"\s+", " ", regex=True)
    category = unit.map(_TOKEN_TO_CATEGORY)
    factor = unit.map(_TOKEN_TO_FACTOR)
    qty = pd.to_numeric(match["qty"], errors="coerce")

    if has_count:
        count = pd.to_numeric(match["count"], errors="coerce")
        valid = qty.notna() & count.notna() & category.notna()
        quantity = qty * factor * count
        pack_count = count
    else:
        valid = qty.notna() & category.notna()
        quantity = qty * factor
        # "5 each" -> pack_count is the quantity itself. "500g" -> just 1.
        pack_count = quantity.where(category == "each", other=1.0)

    return valid, quantity, category, pack_count


def parse_package_sizes(units: pd.Series) -> pd.DataFrame:
    text = units.astype("string")
    lower = text.str.lower()

    # a few rows have leaked price info like "$46.28/1kg" instead of a
    # real size. anything with a dollar sign is treated as unparseable.

    has_dollar_sign = lower.str.contains(r"\$", regex=True, na=False)

    multipack_valid, multipack_quantity, multipack_category, multipack_pack_count = (
        _extract_match(lower, _MULTIPACK_RE, has_count=True)
    )
    reversed_valid, reversed_quantity, reversed_category, reversed_pack_count = (
        _extract_match(lower, _REVERSED_MULTIPACK_RE, has_count=True)
    )
    single_valid, single_quantity, single_category, single_pack_count = _extract_match(
        lower, _SINGLE_RE, has_count=False
    )

    quantity = single_quantity.where(single_valid, other=np.nan)
    quantity = quantity.where(~reversed_valid, other=reversed_quantity)
    quantity = quantity.where(~multipack_valid, other=multipack_quantity)
    quantity = quantity.where(~has_dollar_sign, other=np.nan)

    unit = single_category.where(single_valid, other=np.nan)
    unit = unit.where(~reversed_valid, other=reversed_category)
    unit = unit.where(~multipack_valid, other=multipack_category)
    unit = unit.where(~has_dollar_sign, other=np.nan)

    pack_count = single_pack_count.where(single_valid, other=np.nan)
    pack_count = pack_count.where(~reversed_valid, other=reversed_pack_count)
    pack_count = pack_count.where(~multipack_valid, other=multipack_pack_count)
    pack_count = pack_count.where(~has_dollar_sign, other=np.nan)

    # to_numeric on a "string" dtype column gives back nullable Float64
    # (pd.NA), not plain float64 (np.nan). cast so it's ordinary either way.
    
    quantity = quantity.astype("float64")
    pack_count = pack_count.astype("float64")

    return pd.DataFrame({"quantity": quantity, "unit": unit, "pack_count": pack_count})
