# Shrinkflation in Canadian groceries — CMPT 353 project

**Question:** Are there products whose package size shrank while price held
steady or rose ("shrinkflation")? 
Does the effect differ by grocery chain orproduct category? 
How does our own price index compare to StatCan's
official food CPI over the same period?

## Data

- **Project Hammer** — historical scraped daily prices from Loblaws, No
  Frills, Metro, Voila, T&T, Walmart Canada, Save-On-Foods, and Galleria.
  Raw files go in `data/` (gitignored — see below).
- **StatCan** monthly average retail prices / food CPI, used as an external
  benchmark in the final stage. Source and table IDs TBD once we get there.

Raw data is not committed (`data/*` is gitignored except `.gitkeep`) — it's
large, scraped, and not ours to redistribute. Anyone reproducing this
project needs to drop the Hammer CSVs into `data/` themselves.
