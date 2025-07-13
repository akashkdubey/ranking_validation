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



def _to_fixed_length(relevance: Sequence[float], k: int) -> np.ndarray:
    """
    Convert any relevance list/array to a NumPy vector of length *k*.

    * Truncates to the first *k* items.
    * Pads the tail with zeros if the input is shorter than *k*.

    Parameters
    ----------
    relevance
        Relevance scores (arbitrary numeric scale).
    k
        Cut-off depth.

    Returns
    -------
    np.ndarray
        A 1-D float vector of exactly length *k*.
    """
    k = max(int(k), 1)
    vec = np.asarray(relevance, dtype=float)[:k]
    if vec.size < k:
        vec = np.pad(vec, (0, k - vec.size), "constant")
    return vec

###############################################################################
# nDCG
###############################################################################
def ndcg(true_relevance: Sequence[float],
         predicted_relevance: Sequence[float],
         k: int) -> float:
    """
    Normalised Discounted Cumulative Gain (nDCG@k).

    * **Robust to short lists** – missing ranks are treated as zero gain.
    * Uses log₂ discounts and gains 2^rel − 1 (graded relevance).

    Parameters
    ----------
    true_relevance
        Ground-truth relevance scores (graded). Order need **not** be sorted;
        the function derives the ideal ranking internally.
    predicted_relevance
        Predicted relevance scores aligned to the same items as
        `true_relevance`.
    k
        Cut-off depth (top-*k*).

    Returns
    -------
    float
        nDCG value in `[0, 1]`. Returns `0.0` when the ideal DCG is zero.

    Examples
    --------
    >>> ndcg([3, 2], [2, 3], 5)  # short lists, k larger than lengths
    0.8339912323981488
    >>> ndcg([], [], 5)          # completely empty query
    0.0
    """
    k = max(int(k), 1)

    # --- Prepare fixed-length relevance vectors ----------------------------
    truth_rel = _to_fixed_length(true_relevance, k)
    pred_rel  = _to_fixed_length(predicted_relevance, k)

    log_denom = np.log2(np.arange(2, k + 2))            # length-k

    # --- Ideal DCG ---------------------------------------------------------
    ideal_sorted = np.sort(truth_rel)[::-1]             # length-k
    ideal_dcg = ((2.0 ** ideal_sorted - 1.0) / log_denom).sum()
    if ideal_dcg == 0.0:
        return 0.0

    # --- System DCG --------------------------------------------------------
    sys_dcg = ((2.0 ** pred_rel - 1.0) / log_denom).sum()
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

    Duplicates are ignored (both truth and prediction are converted to sets)
    and an empty truth list returns 0.0 by design.

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
    >>> recall([], ['X', 'Y'], 5)   # design choice: defined as 0.0
    0.0
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

    If the effective top-`k` window contains fewer than 2 unique items,
    τ-ap is defined as 0.0 (no pairwise comparisons possible).

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

    # 0. Trivial early exit                                                  
    if k < 2:
        return 0.0


    # 1. Deduplicate while preserving order                                
    truth = list(dict.fromkeys(truth_items))
    pred  = list(dict.fromkeys(pred_items))

    
    # 2. Pad so both lists contain the same items (full union)           
    for it in truth:
        if it not in pred:
            pred.append(it)
    for it in pred:
        if it not in truth:
            truth.append(it)

    union_size = len(truth)             # == len(pred) by construction
    if union_size < 2:
        return 0.0

    
    # 3. Choose effective depth n = min(k, union_size)                   
    n = min(k, union_size)
    top_truth = truth[:n]               # the AP-weighted window

    
    # 4. Build 1-based rank maps from the *full* padded lists            
    #    (guarantees every lookup succeeds)                              
    
    pos_t = {it: r for r, it in enumerate(truth, 1)}
    pos_p = {it: r for r, it in enumerate(pred, 1)}

    # AP weights w_i = n, n-1, …, 1
    weights = np.arange(n, 0, -1, dtype=float)

    concord = discord = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            u, v = top_truth[i], top_truth[j]
            s = np.sign((pos_t[u] - pos_t[v]) * (pos_p[u] - pos_p[v]))
            if s > 0:
                concord += weights[i] * weights[j]
            elif s < 0:
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

    # Handle the degenerate case of two empty rankings
    if not unique_truth and not unique_pred:
        return 0.0
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
