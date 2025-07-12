
[![PyPI version](https://badge.fury.io/py/rank-validation.svg)](https://badge.fury.io/py/rank-validation)
[![Downloads](https://static.pepy.tech/badge/rank-validation)](https://pepy.tech/project/rank-validation)
[![Build](https://github.com/akashkdubey/ranking_validation/actions/workflows/ci.yml/badge.svg)](https://github.com/akashkdubey/ranking_validation/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# **rank-validation**

> **📊 One-liner ranking evaluation for search, recommendation & IR.**

`rank-validation` converts a dataframe of truth ✨ *vs* prediction 🔮 into two ready-to-export reports—per-query and overall—complete with industry-standard metrics at any cut-off **k**.

---

## ✨ Key features

|               | Why it matters |
|---------------|----------------|
| **Simple API** | `get_metrics_report()` returns familiar **pandas** DataFrames. |
| **Metrics baked-in** | nDCG, Recall, Kendall’s τ-b, τ-ap, RBO (easily extensible). |
| **Any cut-off** | Evaluate at @1, @5, @20… whatever your product cares about. |
| **Automatic grade alignment** | Helper utilities map prediction lists onto graded relevance. |
| **Vectorised core** | Pure **NumPy/Pandas ≥1.4**—no external C extensions, scales to millions of queries on a laptop. |
| **CLI & Python ≥3.8** | Use in notebooks **or** on the command line: `rank-validation results.csv --metrics ndcg recall`. |

---

## 🚀 Installation

```bash
pip install rank-validation
```

The wheel is lightweight (< 30 KB) and pulls in only **numpy**, **pandas**, **scipy**, and **rbo**.

---

## ⚡ Quick start (Python)

```python
import pandas as pd
from rank_validation.validation_generator import get_metrics_report

df = pd.DataFrame({
    "query": ["q1", "q2"],
    "truth_items":  [["A","B","C","D"],   ["X","Y","Z"]],
    "truth_scores": [[3,2,1,0],           [2,1,0]],
    "pred_items":   [["B","A","E","C"],   ["Y","X","Z"]],
})

metrics  = ["ndcg", "recall", "kendall_tau", "tau_ap", "rbo"]
cutoffs  = [3, 5]

query_report, overall_report = get_metrics_report(
    df,
    truth_item_col="truth_items",
    truth_score_col="truth_scores",
    pred_item_col="pred_items",
    metric_list=metrics,
    cutoff_list=cutoffs,
)

print(query_report)        # only `query` + metric columns
print(overall_report)      # mean, std, min, max, …
```

Typical `query_report`:

```
  query  ndcg@3  recall@3  kendall_tau@3  tau_ap@3  rbo@3  ndcg@5  recall@5  kendall_tau@5  tau_ap@5  rbo@5
0    q1    0.91      0.67           0.33      0.40   0.79    0.90      1.00           0.33      0.46   0.79
1    q2    1.00      0.67           0.67      0.80   1.00    1.00      1.00           0.67      0.80   1.00
```

---

## 🧮 Supported metrics

| Metric | What it measures | Reference |
| ------ | ---------------- | --------- |
| **nDCG@k** | Graded relevance with log-discounted gain, normalised by ideal ranking | Järvelin & Kekäläinen (2002) |
| **Recall@k** | Proportion of relevant items retrieved in top k | – |
| **Kendall’s τ-b@k** | Rank correlation, tie-adjusted | Kendall (1938) |
| **Kendall’s τ-ap@k** | **Top-weighted** rank correlation | Yilmaz et al. (2008) |
| **RBO@k** | Rank-biased overlap, emphasises early ranks | Webber et al. (2010) |

> **Heads-up:** RBO requires the two lists to have unique items and equal lengths. If you hit `RankingSimilarity` errors, drop duplicates or omit RBO.

---

## 🛠️ API reference

```python
def get_metrics_report(
    df: pd.DataFrame,
    truth_item_col: str,
    truth_score_col: str,
    pred_item_col: str,
    metric_list: list[str],
    cutoff_list: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]
```

| Parameter | Description |
|-----------|-------------|
| **df** | DataFrame with at least the three list-columns below. |
| **truth_item_col** | Column holding ground-truth item IDs. |
| **truth_score_col** | Column with relevance grades (same order & length). |
| **pred_item_col** | Column holding system-predicted ranked lists. |
| **metric_list** | Any subset of the keys in `rank_validation.metrics.METRIC_REGISTRY`. |
| **cutoff_list** | Integers, e.g. `[1, 3, 10]`. Each yields a `metric@k` column. |

Returns `(query_report, overall_report)` where:

* `query_report` – **only** the `query` column (if present) and metric columns.  
* `overall_report` – `query_report.describe()`.

---

## ⚙️ Performance tips

* The core is fully vectorised; multi-core Pandas handles millions of rows out-of-the-box.  
* For very long truth lists (> 1 K items) evaluate in chunks to cap memory.  
* τ-ap is O(k²); keep `k` ≤ 50 for interactive latency.

---

## 🖥️ Command-line

```bash
rank-validation my_results.csv   --truth-items truth_items --truth-scores truth_scores   --pred-items pred_items   --metrics ndcg recall tau_ap   --cutoffs 1 5 10   --output reports/
```

Generates `query_report.csv` and `overall_report.csv` in the target folder.

---

## 🤝 Contributing

Bug / feature ideas → **Issues**.  
PRs welcome — please add tests and run `pre-commit run -a`.

---

## 🛣️ Roadmap

- [ ] Mean Average Precision (MAP)  
- [ ] Mean Reciprocal Rank (MRR)  
- [ ] Statistical significance tests (paired randomisation)  
- [ ] GPU path via cuDF / RAPIDS  

---

## 📝 License

MIT © 2025 Akash Dubey

---

## 🔗 Links & citation

| | |
|---|---|
| **Docs & examples** | <https://github.com/akashkdubey/ranking_validation> |
| **PyPI** | <https://pypi.org/project/rank-validation/> |

```bibtex
@software{Dubey_2025_rank_validation,
  author = {Dubey, Akash},
  title  = {rank-validation: A lightweight toolkit for ranking evaluation},
  year   = {2025},
  url    = {https://github.com/akashkdubey/ranking_validation}
}
```

<sub>Built with ❤️, Pandas, and SciPy.</sub>
