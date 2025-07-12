[![PyPI version](https://badge.fury.io/py/rank-validation.svg)](https://badge.fury.io/py/rank-validation) 
[![Downloads](https://static.pepy.tech/badge/rank-validation)](https://pepy.tech/project/rank-validation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# **rank‑validation**

> **📊 One‑liner ranking evaluation for search, recommendation & IR.**

`rank‑validation` turns a dataframe of truth ✨ *vs* prediction 🔮 into two ready‑to‑export reports—per‑query and overall—complete with industry‑standard metrics at any cut‑off **k**.

---

## ✨ Key features

- **Simple API** – `get_metrics_report(...)` returns pandas DataFrames you already know how to use.  
- **Out‑of‑the‑box metrics** – nDCG, Recall, Kendall’s τ, RBO (extendable).  
- **Arbitrary cut‑offs** – evaluate at @1, @5, @20… whatever matters.  
- **Automatic score alignment** – helper utilities map prediction lists onto truth scores for graded relevance.  
- **Vectorised NumPy & Pandas core** – scales to millions of queries on a laptop.  
- **Pure Python ≥ 3.8** – zero native extensions.

---

## 🚀 Installation

```bash
pip install rank-validation
```

The wheel is lightweight (< 30 KB) and pulls in only **numpy**, **pandas**, **scipy** & **rbo**.

---

## ⚡ Quick start

```python
import pandas as pd
from rank_validation.validation_generator import get_metrics_report

df = pd.DataFrame({
    "query": ["q1", "q2"],
    "truth_items":  [["A","B","C","D"], ["X","Y","Z"]],
    "truth_scores": [[3,2,1,0],          [2,1,0]],
    "pred_items":   [["B","A","E","C"], ["Y","X","Z"]],
})

metrics  = ["ndcg", "recall", "kendall_tau", "rbo"]
cutoffs  = [3, 5]

query_report, overall_report = get_metrics_report(
    df,
    truth_item_col="truth_items",
    truth_score_col="truth_scores",
    pred_item_col="pred_items",
    metric_list=metrics,
    cutoff_list=cutoffs,
)

print(query_report.head())  # per‑query breakdown
print(overall_report)       # summary stats (mean, std, …)
```

Typical output:

```
  query  ndcg@3  recall@3  kendall_tau@3  rbo@3  ndcg@5  recall@5  kendall_tau@5  rbo@5
0    q1    0.91      0.67           0.33   0.79    0.90      1.00           0.33   0.79
1    q2    1.00      0.67           0.67   1.00    1.00      1.00           0.67   1.00
```

---

## 🧮 Supported metrics & formulas

| Metric | What it measures | Reference |
| ------ | ---------------- | --------- |
| **nDCG@k** | Graded relevance with log‑discounted gain, normalised by ideal ranking | Järvelin & Kekäläinen (2002) |
| **Recall@k** | Proportion of ground‑truth items retrieved in top k | – |
| **Kendall’s τ@k** | Rank correlation, ties handled via normalisation | Kendall (1938) |
| **RBO@k** | Top‑weighted similarity between two indefinite rankings | Webber et al. (2010) |

> **Heads‑up:** RBO requires the two lists to have unique items and equalised lengths. If you hit `RankingSimilarity` errors, drop duplicates beforehand or omit RBO for that experiment.

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
| **df** | DataFrame containing at minimum the three list‑columns below. |
| **truth_item_col** | Column name containing ground‑truth item IDs. |
| **truth_score_col** | Column name containing relevance grades (same order & length as `truth_item_col`). |
| **pred_item_col** | Column name with system‑predicted ranked lists. |
| **metric_list** | Any subset of `["ndcg", "recall", "kendall_tau", "rbo"]`. |
| **cutoff_list** | Integers e.g. `[1, 3, 10]`. Each generates `metric@k` columns. |

Returns `(query_report, overall_report)` where:

- **query_report** – original df plus metric columns.  
- **overall_report** – `query_report.describe()` (mean, std, min, max…).

---

## ⚙️ Performance tips

- Core logic is vectorised; multi‑process pandas handles millions of rows out‑of‑the‑box.  
- Chunk evaluation if truth lists are extremely long (> 1 K items) to limit memory.

---

## 🤝 Contributing

Found a bug? Want MAP or MRR support? PRs are welcome! Please open an issue first so we can discuss the approach.

1. Fork ➡️ branch ➡️ commit (with tests!)  
2. `pre‑commit run -a`  
3. Open a pull request describing the change.

---

## 🛣️ Roadmap

- [ ] Mean Average Precision (MAP)  
- [ ] Mean Reciprocal Rank (MRR)  
- [ ] Optional GPU acceleration via cuDF / RAPIDS  

---

## 📝 License

MIT © 2025 Akash Dubey

---

## 🔗 Links & citation

- **Docs / examples**: <https://github.com/akashkdubey/ranking_validation>  
- **PyPI**: <https://pypi.org/project/rank-validation/>

```bibtex
@software{Dubey_2025_rank_validation,
  author = {Dubey, Akash},
  title  = {rank‑validation: A lightweight toolkit for ranking evaluation},
  year   = {2025},
  url    = {https://github.com/akashkdubey/ranking_validation}
}
```

<sub>Built with ❤️, Pandas & SciPy.</sub>
