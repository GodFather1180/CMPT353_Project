"""Load raw Project Hammer price data into a consistent DataFrame.

Left as a stub until we've seen the actual column names and a few sample
rows — the Hammer scrapes are known to vary in format across chains and
scrape batches, so guessing the schema here would just mean rewriting it.
"""

import pandas as pd


def load_hammer_data(path: str) -> pd.DataFrame:
    raise NotImplementedError("Waiting on a sample of the actual data before implementing this.")
