# Market Basket Analysis — Association Rule Learning

Finds product associations in a grocery store transaction dataset. The classic example: "customers who buy bread also buy butter." Both Apriori and ECLAT are implemented here and compared side by side.

## The dataset

`Market_Basket_Optimisation.csv` contains 7,500 transactions from a French grocery store, each with up to 20 items. There's no header row — just comma-separated product names per transaction.

## How the algorithms differ

**Apriori** generates rules with support, confidence, and lift. You care about all three: support tells you how common the pair is, confidence tells you how reliable the rule is, and lift tells you how much better than random chance it is.

**ECLAT** is simpler — it only looks at support (co-occurrence frequency), not confidence or lift. It's faster because it uses vertical data representation, but gives you less information per rule.

Both are implemented using `apyori` here, which means ECLAT is effectively Apriori filtered to support-only output. A true ECLAT implementation would use depth-first search on a transaction-ID list, but for this dataset size the difference in speed is negligible.

## Expected output

Top rules by lift typically include things like:
- `herb & pepper → ground beef` (lift ~3.3)
- `whole wheat pasta → olive oil` (lift ~3.1)
- `tomato sauce → ground beef` (lift ~3.1)

## How to run

```bash
python main.py
```

Prints top 10 rules for both methods. Saves two CSV files to `results/`:
- `apriori_top10.csv` — top 10 rules sorted by lift
- `eclat_top10.csv` — top 10 item pairs sorted by support

## Code structure

```
BasketAnalyzer
├── load_transactions()   → reads CSV into a list of lists (one list per transaction)
├── _parse_rules()        → static method, extracts antecedent/consequent/metrics from apyori output
├── run_apriori()         → runs apriori, returns DataFrame sorted by lift
├── run_eclat()           → same but returns only support columns, sorted by support
└── save_results()        → writes top-10 CSVs to results/
```

## Parameters

Default thresholds: `min_support=0.003`, `min_confidence=0.2`, `min_lift=3.0`. These work well for this dataset. Lowering `min_support` below 0.002 can generate thousands of rules — probably not useful.
