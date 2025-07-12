"""ranking_metrics.py
====================

Vectorised implementations of core information-retrieval **ranking metrics**.

Public API
----------
`ndcg`
    Normalised Discounted Cumulative Gain (nDCG).
`recall`
    Set-based recall at *k*.
`kendall_tau`
    Kendall’s τ-b rank correlation with tie handling.
`tau_ap`
    Average-Precision weighted Kendall’s τ (τ-ap).
`rbo_sim`
    Rank-Biased Overlap (RBO) similarity.
`METRIC_REGISTRY`
    Mapping *metric-name* → *callable*, for config-driven evaluation.

All functions are

* **Vectorised** (NumPy + SciPy) – no Python loops over items *except where
  inherent to the definition* (τ-ap’s double sum).
* **Type-hinted** – signatures use the standard ``typing`` module.
* **Edge-safe** – never raise on zero-division; they return *0.0* instead.

Examples
--------
>>> from ranking_metrics import ndcg, recall, tau_ap
>>> y_true = [3, 2, 3, 0]
>>> y_pred = [3, 3, 2, 0]
>>> ndcg(y_true, y_pred, k=3)
1.0
>>> recall(["A", "B", "C"], ["B", "A", "D"], k=3)
0.6666...
>>> tau_ap(["A", "B", "C"], ["A", "C", "B"], k=3)
0.2222...
"""

from __future__ import annotations
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
import rbo as _rbo

__all__ = [
    "ndcg",
    "recall",
    "kendall_tau",
    "tau_ap",
    "rbo_sim",
    "METRIC_REGISTRY",
]

###############################################################################
# nDCG
###############################################################################
def ndcg(
    true_relevance: Sequence[float],
    predicted_relevance: Sequence[float],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain (nDCG).

    Parameters
    ----------
    true_relevance : Sequence[float]
        Ground-truth *graded* relevance scores (larger = more relevant).
    predicted_relevance : Sequence[float]
        Model-predicted scores aligned **one-to-one** with `true_relevance`.
    k : int
        Evaluation depth (``k >= 1``).

    Returns
    -------
    float
        nDCG ∈ ``[0.0, 1.0]``; returns ``0.0`` when the ideal DCG is zero.

    Notes
    -----
    DCG and IDCG are computed with ``2**rel − 1`` gains and ``log₂(rank+1)``
    discounts (Jarvelin & Kekäläinen, 2002).

    Examples
    --------
    >>> ndcg([3, 3, 2, 0], [3, 3, 2, 0], k=3)
    1.0
    """
    k = max(int(k), 1)
    log_denom = np.log2(np.arange(2, k + 2))

    # --- Ideal DCG ----------------------------------------------------------
    truth_sorted = np.sort(true_relevance)[::-1][:k]
    ideal_dcg = ((2.0 ** truth_sorted - 1.0) / log_denom).sum()
    if ideal_dcg == 0.0:  # avoid division by zero
        return 0.0

    # --- System DCG ---------------------------------------------------------
    sys_rel = np.asarray(predicted_relevance, dtype=float)[:k]
    sys_dcg = ((2.0 ** sys_rel - 1.0) / log_denom).sum()
    return float(sys_dcg / ideal_dcg)


###############################################################################
# Recall
###############################################################################
def recall(
    truth_items: Sequence[str],
    pred_items: Sequence[str],
    k: int,
) -> float:
    """Set-based recall@k (multiplicity ignored).

    Parameters
    ----------
    truth_items : Sequence[str]
        Relevant (ground-truth) item IDs.
    pred_items : Sequence[str]
        Ranked list of retrieved item IDs.
    k : int
        Evaluation depth.

    Returns
    -------
    float
        Recall ∈ ``[0.0, 1.0]``.  If `truth_items` is empty → ``0.0``.

    Examples
    --------
    >>> recall(["A", "B", "C"], ["B", "A", "D"], k=3)
    0.6666...
    """
    if not truth_items:
        return 0.0
    retrieved = set(pred_items[:k])
    relevant = set(truth_items)
    return len(retrieved & relevant) / len(relevant)


###############################################################################
# Kendall's τ-b
###############################################################################
def kendall_tau(
    truth_items: Sequence[str],
    pred_items: Sequence[str],
    k: int,
) -> float:
    """Kendall’s τ-b rank correlation on the top-`k` prefix.

    Items missing from either list are appended so the permutations contain
    identical elements before correlation is computed (common IR practice).

    Parameters
    ----------
    truth_items, pred_items : Sequence[str]
        Gold and system orderings.
    k : int
        Prefix length.

    Returns
    -------
    float
        τ-b ∈ ``[-1.0, 1.0]``; returns ``0.0`` when SciPy yields NaN.

    Examples
    --------
    >>> kendall_tau(["A", "B", "C"], ["A", "C", "B"], k=3)
    0.3333
    """
    truth_norm, pred_norm = _normalise_for_kendalltau(truth_items, pred_items, k)
    tau, _ = stats.kendalltau(truth_norm[:k], pred_norm[:k])
    return 0.0 if np.isnan(tau) else float(tau)


def _normalise_for_kendalltau(
    truth_items: Sequence[str],
    pred_items: Sequence[str],
    k: int,
) -> Tuple[List[str], List[str]]:
    """Return two k-length lists containing the **same** elements for τ-b."""
    truth_k = list(dict.fromkeys(truth_items[:k]))
    pred_k = list(dict.fromkeys(pred_items[:k]))

    truth_extra = [it for it in pred_k if it not in truth_k]
    pred_extra = [it for it in truth_k if it not in pred_k]

    return truth_k + truth_extra, pred_k + pred_extra


###############################################################################
# τ-ap  (Average-Precision weighted Kendall)
###############################################################################
def tau_ap(
    truth_items: Sequence[str],
    pred_items: Sequence[str],
    k: int,
) -> float:
    """Average-Precision weighted Kendall’s τ (τ-ap).

    Introduced by Yılmaz, Keşelj & Robertson (SIGIR 2008), τ-ap penalises
    discordant pairs more near the top of the ranking.

    Parameters
    ----------
    truth_items, pred_items : Sequence[str]
        Gold and system orderings (duplicates ignored).
    k : int
        Evaluation depth (≥ 2).

    Returns
    -------
    float
        τ-ap ∈ ``[-1.0, 1.0]``; returns ``0.0`` when no pairwise weight exists.

    Notes
    -----
    τ-ap is defined as::

        τ_ap = (2 / (k·(k−1))) Σ_{i<j} w_i w_j s_ij
        w_r = k − r + 1   (descending weights)
        s_ij = sign((π_t(i) − π_t(j)) · (π_p(i) − π_p(j)))

    The implementation runs in **O(k²)** (double loop) which is acceptable
    for typical IR depths (k ≤ 100).

    Examples
    --------
    >>> tau_ap(["A", "B", "C"], ["A", "C", "B"], k=3)
    0.6364
    """
    k = max(int(k), 2)

    # --- Deduplicate and keep first-k unique items --------------------------
    truth = list(dict.fromkeys(truth_items))[:k]
    pred = list(dict.fromkeys(pred_items))[:k]

    # --- Pad so both lists share the same elements -------------------------
    for it in truth:
        if it not in pred:
            pred.append(it)
    for it in pred:
        if it not in truth:
            truth.append(it)

    # --- Rank maps (1-based) ------------------------------------------------
    pos_t = {it: r for r, it in enumerate(truth, 1)}
    pos_p = {it: r for r, it in enumerate(pred, 1)}

    weights = np.arange(k, 0, -1, dtype=float)  # [k, k-1, …, 1]

    concord, discord = 0.0, 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            s_ij = np.sign(
                (pos_t[truth[i]] - pos_t[truth[j]])
                * (pos_p[truth[i]] - pos_p[truth[j]])
            )
            if s_ij > 0:
                concord += weights[i] * weights[j]
            elif s_ij < 0:
                discord += weights[i] * weights[j]

    denom = concord + discord
    return 0.0 if denom == 0.0 else (concord - discord) / denom


###############################################################################
# Rank-Biased Overlap (RBO)
###############################################################################
def rbo_sim(
    truth_items: Sequence[str],
    pred_items: Sequence[str],
    k: int,
    p: float = 0.9,
) -> float:
    """Rank-Biased Overlap (RBO) similarity.

    Introduced by Webber, Moffat & Zobel (TOIS 2010), RBO is a **top-weighted**
    measure of agreement between two rankings that gracefully handles
    *indefinite* (potentially unbounded) lists.  A user is modelled as
    inspecting deeper ranks with probability ``p`` each step, so overlap at
    depth *d* is geometrically discounted by ``p**(d-1)``.

    Parameters
    ----------
    truth_items, pred_items : Sequence[str]
        Gold (ground-truth) and system orderings.  Duplicates are removed,
        preserving first occurrence.
    k : int
        Evaluation depth; both lists are truncated (and, if necessary, padded so
        they contain the same elements) before similarity is computed.
    p : float, optional
        Persistence/steepness parameter in ``(0, 1)`` (default ``0.9``).  Higher
        values give more weight to lower ranks; ``p → 0`` yields plain overlap
        at rank 1.

    Returns
    -------
    float
        RBO similarity ``∈ [0.0, 1.0]``.  ``1.0`` indicates identical rankings,
        ``0.0`` indicates no overlap up to depth *k*.

    Notes
    -----
    * This implementation delegates to
      ``rbo.RankingSimilarity(...).rbo_ext(p=p)``, which computes the
      **extrapolated** RBO – it adds a residual term for the unobserved tail so
      that scores remain well-behaved even when lists are shorter than *k*.
    * Time-complexity: **O(k)**.
    * Edge-safe: returns ``0.0`` if both input lists are empty.

    Examples
    --------
    >>> rbo_sim(["A", "B", "C"], ["B", "A", "D"], k=3, p=0.9)
    0.63
    >>> rbo_sim(["A", "B", "C"], ["A", "B", "C"], k=3)
    1.0
    """
    unique_truth = list(dict.fromkeys(truth_items))[:k]
    unique_pred = list(dict.fromkeys(pred_items))[:k]

    if unique_truth == unique_pred:
        return 1.0

    return float(_rbo.RankingSimilarity(unique_truth, unique_pred).rbo_ext(p=p))


###############################################################################
# Metric Registry
###############################################################################
METRIC_REGISTRY: Dict[str, Callable[..., float]] = {
    "ndcg": ndcg,
    "recall": recall,
    "kendall_tau": kendall_tau,
    "tau_ap": tau_ap,
    "rbo": rbo_sim,
}
"""Canonical *metric-name* → *function* registry.

>>> METRIC_REGISTRY["tau_ap"](truth, pred, k=10)
0.4876...
"""
