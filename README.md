[![PyPI version](https://img.shields.io/pypi/v/rank-validation?label=PyPI)](https://pypi.org/project/rank-validation/)
[![Downloads](https://static.pepy.tech/badge/rank-validation)](https://pepy.tech/project/rank-validation)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)




# **rank‑validation**

> **📊 One‑liner ranking evaluation for search, recommendation & IR.**

`rank‑validation` turns a dataframe of truth ✨ *vs* prediction 🔮 into two ready‑to‑export reports—per‑query and overall—complete with industry‑standard metrics at any cut‑off **k**.

---

## ✨ Key features

- **Simple API** – `get_metrics_report(...)` returns pandas DataFrames you already know how to use.  
- **Out‑of‑the‑box metrics** – nDCG, Recall, Kendall’s τ‑b, Kendall’s τ‑ap, RBO (extendable).  
- **Arbitrary cut‑offs** – evaluate at @1, @5, @20… whatever matters.  
- **Automatic score alignment** – helper utilities map prediction lists onto truth scores for graded relevance.  
- **Vectorised NumPy & Pandas core** – scales to millions of queries on a laptop.  
- **Pure Python ≥ 3.8** – zero native extensions.

---

## 📢 What’s new *(v1.2.0)*

* **Edge‑safe metrics**
  * nDCG copes with queries shorter than *k*.
  * τ‑ap avoids out‑of‑range indexing for tiny lists.
  * RBO returns `0.0` (not `1.0`) for two empty lists.
* **Robust helpers** &mdash; Utility functions align truth/prediction lists and zero‑pad scores.
* **Better docs** &mdash; Input schema, edge‑case semantics and result interpretation are now documented.

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

print(query_report.head())  # per‑query breakdown
print(overall_report)       # summary stats (mean, std, …)
```

Typical `query_report`:

```
  query  ndcg@3  recall@3  kendall_tau@3  tau_ap@3  rbo@3  ndcg@5  recall@5  kendall_tau@5  tau_ap@5  rbo@5
0    q1    0.91      0.67           0.33      0.40   0.79    0.90      1.00           0.33      0.46   0.79
1    q2    1.00      0.67           0.67      0.80   1.00    1.00      1.00           0.67      0.80   1.00
```

Typical `overall_report`:

```
       ndcg@3  recall@3  kendall_tau@3  tau_ap@3  rbo@3  ndcg@5  recall@5  kendall_tau@5  tau_ap@5  rbo@5
mean     0.96     0.67           0.50      0.60   0.90    0.95     1.00           0.50      0.63   0.90
std      0.06     0.00           0.24      0.28   0.15    0.05     0.00           0.24      0.24   0.15
```

---

## 🧮 Supported metrics & formulas

| Metric | What it measures | Reference |
| ------ | ---------------- | --------- |
| **nDCG@k** | Graded relevance with log‑discounted gain, normalised by ideal ranking | Järvelin & Kekäläinen (2002) |
| **Recall@k** | Proportion of ground‑truth items retrieved in top k | – |
| **Kendall’s τ‑b@k** | Rank correlation, tie‑adjusted | Kendall (1938) |
| **Kendall’s τ‑ap@k** | **Top‑weighted** rank correlation | Yilmaz et al. (2008) |
| **RBO@k** | Top‑weighted similarity between two indefinite rankings | Webber et al. (2010) |

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
| **df** | DataFrame with at least the three list‑columns below. |
| **truth_item_col** | Column holding ground‑truth item IDs. |
| **truth_score_col** | Column with relevance grades (same order & length). |
| **pred_item_col** | Column holding system‑predicted ranked lists. |
| **metric_list** | Any subset of `METRIC_REGISTRY` keys, e.g. `ndcg`, `tau_ap`. |
| **cutoff_list** | Integers e.g. `[1, 3, 10]`. Each yields `metric@k` columns. |

Returns `(query_report, overall_report)` where:

* **query_report** – original df plus metric columns.  
* **overall_report** – `query_report.describe()`.

---

### Edge‑case semantics

* **Empty `truth_items`** → all metrics `0.0` for that query.  
* **Empty `pred_items`** → recall `0.0`; correlation/similarity metrics also `0.0`.  
* **Lists shorter than *k*** → missing ranks are treated as zero gain/irrelevant.

---

## ⚙️ Performance tips

- Core logic is vectorised; multi‑process pandas handles millions of rows out‑of‑the‑box.  
- Chunk evaluation if truth lists are extremely long (> 1 K items) to limit memory.

---

## 🤝 Contributing

Bug report? New metric? Glad to have you! Please:

1. Open an issue outlining the proposal.  
2. Fork → branch → **add unit tests**.  
3. Run `pre‑commit run -a` & `pytest`.  
4. Submit a pull request.

---

## 🛣️ Roadmap

- [ ] Mean Average Precision (MAP)  
- [ ] Mean Reciprocal Rank (MRR)  
- [ ] Precision@k & F1@k  
- [ ] Expected Reciprocal Rank (ERR)  
- [ ] GPU acceleration via cuDF / RAPIDS  

---

## 📝 License

Apache License 2.0 © 2025 Akash Dubey

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

<sub>Built with ❤️, Pandas & SciPy.</sub>
