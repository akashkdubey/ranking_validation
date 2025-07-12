"""
High-level public API:

    get_metrics_report(df, ...)

It produces
-----------
query_report   : per-query metric DataFrame
overall_report : df.describe() summary of those metrics

Implementation notes
--------------------
* All heavy per-row work happens once in `prepare_relevance`.
* Metric functions are fetched from METRIC_REGISTRY, so adding
  new metrics is a one-liner registration.
* No swifter dependency – pandas ≥1.4 vectorisation is faster
  for large frames and removes a non-essential requirement.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Dict

import pandas as pd

from .metrics import METRIC_REGISTRY
from .normalise import normalise_relevance


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _prepare_relevance_row(
    truth_items: Sequence[str],
    truth_scores: Sequence[float],
    pred_items: Sequence[str],
) -> Tuple[List[str], List[float], List[str], List[float]]:
    truth_rel, pred_rel = normalise_relevance(truth_items, truth_scores, pred_items)
    t_items, t_scores = map(list, zip(*truth_rel))
    p_items, p_scores = map(list, zip(*pred_rel))
    return t_items, t_scores, p_items, p_scores


def _add_metric_columns(
    df: pd.DataFrame,
    metrics: List[str],
    cutoffs: List[int],
    col_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Vectorised column-wise metric evaluation.

    Parameters
    ----------
    df        : DataFrame with four pre-split list columns
    metrics   : subset of METRIC_REGISTRY keys
    cutoffs   : list[int], e.g. [3, 5, 10]
    col_map   : mapping logical name → column in df
    """
    t_items = df[col_map["truth_items"]]
    t_scores = df[col_map["truth_scores"]]
    p_items = df[col_map["pred_items"]]
    p_scores = df[col_map["pred_scores"]]

    for m in metrics:
        func = METRIC_REGISTRY[m]
        for k in cutoffs:
            col = f"{m}@{k}"
            if m == "ndcg":
                df[col] = [func(ts, ps, k) for ts, ps in zip(t_scores, p_scores)]
            elif m == "recall":
                df[col] = [func(ti, pi, k) for ti, pi in zip(t_items, p_items)]
            else:  # kendall_tau / rbo_sim operate on item IDs only
                df[col] = [func(ti, pi, k) for ti, pi in zip(t_items, p_items)]

    return df


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_metrics_report(
    df: pd.DataFrame,
    truth_item_col: str,
    truth_score_col: str,
    pred_item_col: str,
    metric_list: List[str],
    cutoff_list: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ranking-metric reports.

    Parameters
    ----------
    df
        Input frame – one row per query with list-columns.
    truth_item_col
        Column containing ground-truth item IDs.
    truth_score_col
        Column with relevance grades aligned to `truth_item_col`.
    pred_item_col
        Column with system-predicted ranked lists.
    metric_list
        Subset of ``METRIC_REGISTRY`` keys.
    cutoff_list
        Cut-offs (depths) at which to evaluate each metric.

    Returns
    -------
    query_report
        Per-query DataFrame: only the ``"query"`` column (if present) and metric columns.
    overall_report
        `query_report.describe()` – aggregate stats.
    """
    if not set(metric_list) <= METRIC_REGISTRY.keys():
        unknown = set(metric_list) - METRIC_REGISTRY.keys()
        raise ValueError(f"Unknown metrics requested: {unknown!r}")

    # --------------------------------------------------------------------- #
    # 1  Prepare relevance lists & scores                                   #
    # --------------------------------------------------------------------- #
    prep_cols = ["_t_items", "_t_scores", "_p_items", "_p_scores"]
    df[prep_cols] = df.apply(
        lambda row: _prepare_relevance_row(
            row[truth_item_col], row[truth_score_col], row[pred_item_col]
        ),
        axis=1,
        result_type="expand",
    )

    # --------------------------------------------------------------------- #
    # 2  Compute metric columns                                             #
    # --------------------------------------------------------------------- #
    col_map = {
        "truth_items": "_t_items",
        "truth_scores": "_t_scores",
        "pred_items": "_p_items",
        "pred_scores": "_p_scores",
    }
    df_metrics = _add_metric_columns(df, metric_list, cutoff_list, col_map)

    # --------------------------------------------------------------------- #
    # 3  Return clean reports                                               #
    # --------------------------------------------------------------------- #
    metric_cols = [c for c in df_metrics.columns if "@" in c]
    base_cols = ["query"] if "query" in df_metrics.columns else []
    query_report = df_metrics[base_cols + metric_cols]

    overall_report = query_report.describe()

    return query_report, overall_report
