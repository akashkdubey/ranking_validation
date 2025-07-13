[![PyPI version](https://img.shields.io/pypi/v/rank-validation?label=PyPI)](https://pypi.org/project/rank-validation/)
[![Downloads](https://static.pepy.tech/badge/rank-validation)](https://pepy.tech/project/rank-validation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# **rank‑validation**

> **📊 One‑liner ranking evaluation for search, recommendation & IR**

`rank‑validation` turns a dataframe of **ground‑truth** ✨ *vs* **prediction** 🔮 into ready‑to‑export **per‑query** and **overall** reports equipped with industry‑standard metrics – all in a single call.

---

## ✨ What’s new (v1.2)

* **Robust edge handling** – all metrics now cope with *shorter‑than‑k* lists and return 0 instead of crashing fileciteturn4file2  
* **τ‑ap & RBO fixes** – no more index errors; two empty lists now yield RBO = 0 fileciteturn4file6  
* **Cleaner RBO API** – optional `p` lets you tune top‑weighting steepness  
* **Extensible registry** – register a metric with one line:  
  ```python
  METRIC_REGISTRY["my_metric"] = my_func
  ``` fileciteturn4file17  

---

## 🔑 Key features

- **Simple pandas API** – `get_metrics_report()` returns familiar `DataFrame`s.  
- **Edge‑safe implementations** – never raises on zero‑division or short lists.  
- **Out‑of‑the‑box metrics** – nDCG, Recall, *Kendall’s τ‑b*, *Kendall’s τ‑ap*, RBO.  
- **Arbitrary cut‑offs** – evaluate at @1, @5, @20… whatever matters.  
- **Vectorised NumPy & Pandas core** – millions of queries fit on a laptop.  
- **Pure Python ≥ 3.8** – zero native extensions.

---

## 🚀 Installation

```bash
pip install rank-validation
```

The wheel is tiny (< 30 KB) and depends only on **numpy**, **pandas**, **scipy** & **rbo**.

---

## ⚡ Quick start

```python
import pandas as pd
from rank_validation.validation_generator import get_metrics_report

df = pd.DataFrame({
    "query": ["q1", "q2"],
    "truth_items":  [["A","B","C","D"], ["X","Y","Z"]],
    "truth_scores": [[3,2,1,0],          [2,1,0]],
    "pred_items":   [["B","A","E","C"],  ["Y","X","Z"]],
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

print(query_report.head())   # per‑query breakdown
print(overall_report.loc["mean"])  # summary row
```

---

## 🧮 Supported metrics

| Metric | Measures | Notes |
| ------ | -------- | ----- |
| **nDCG@k** | Graded relevance with log₂ discount | Järvelin & Kekäläinen 2002 |
| **Recall@k** | Share of relevant items retrieved | – |
| **Kendall τ‑b@k** | Rank correlation (tie‑aware) | Kendall 1938 |
| **Kendall τ‑ap@k** | **Top‑weighted** rank correlation | Yılmaz et al. 2008 |
| **RBO@k (p = 0.9)** | Top‑weighted list overlap | Webber et al. 2010 |

> **Note** RBO now returns 0.0 when both lists are empty and allows custom `p`.

---

## 🛠️ Minimal API

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

*Returns `(query_report, overall_report)` where*  
• **query_report** – original rows + metric columns  
• **overall_report** – `query_report.describe()` (use the *mean* row for system‑level scores)

---

## ⚙️ Performance tips

- The heavy lifting is vectorised; still, for **very** deep lists (> 1 K) slice `cutoff_list` to what you care about.  
- Evaluation is embarrassingly parallel; chunk `df` and `concat` if working with 10‑million‑query logs.

---

## 🤝 Contributing

Bug, idea or metric missing? Open an issue or PR! Please add unit tests – metrics are mission‑critical.

---

## 🛣️ Roadmap

- [ ] **MAP** – Mean Average Precision  
- [ ] **MRR** – Mean Reciprocal Rank  
- [ ] **Precision@k** & **F1@k**  
- [ ] **ERR** – Expected Reciprocal Rank  
- [ ] Optional multi‑process evaluation  
- [ ] GPU acceleration via cuDF / RAPIDS  

---

## 📝 License

MIT © 2025 Akash Dubey

---

## 🔗 Links & citation

* Docs / examples: <https://github.com/akashkdubey/ranking_validation>  
* PyPI: <https://pypi.org/project/rank-validation/>

```bibtex
@software{Dubey_2025_rank_validation,
  author = {Dubey, Akash},
  title  = {rank‑validation: A lightweight toolkit for ranking evaluation},
  year   = {2025},
  url    = {https://github.com/akashkdubey/ranking_validation}
}
```

<sub>Built with ❤️ Pandas, NumPy & SciPy.</sub>
