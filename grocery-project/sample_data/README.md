# Sample data

The real data is too large to commit (`raw.csv` alone is 3.5GB), so these
are small, real data in the exact format the code uses. They're not
made up, every row here came out of the actual Hammer/StatCan files.

- `product_sample.csv`: 9 rows from `product.csv`. Includes a few clean
  rows plus the messy cases the parser and loader are built to handle: a
  multipack unit (`45gx8`), an `"error"` placeholder row (a scraper
  failure, not a real product), a garbled units field
  (`foil wrapped4 each`), and a leaked price string (`$8.98`) sitting in
  the `units` column instead of a real size.
- `raw_sample.csv`: 18 rows from `raw.csv`, mostly a couple of scrapes
  per sample product, plus one row with the cents-vs-dollars bug
  (`current_price = 648.0` alongside `price_per_unit = "$324.00/100ml"`)
  and one with the Julian-epoch bad date sentinel (`-4713-11-24`).
- `statcan_cpi_sample.csv`: 12 rows from the StatCan Food CPI table
  (18-10-0004-01), a few months across a few product groups.

Running the loaders against these files shows the cleaning actually
working:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from data_loading import load_products, load_raw_prices
from statcan import load_food_cpi
print(load_products('sample_data/product_sample.csv'))
print(load_raw_prices('sample_data/raw_sample.csv'))
print(load_food_cpi('sample_data/statcan_cpi_sample.csv'))
"
```
