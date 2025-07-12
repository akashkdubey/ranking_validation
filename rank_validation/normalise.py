"""
Utility helpers for aligning truth/prediction lists and scores.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple


def normalise_relevance(
    truth_items: Sequence[str],
    truth_scores: Sequence[float],
    pred_items: Sequence[str],
) -> Tuple[
    List[Tuple[str, float]],
    List[Tuple[str, float]],
]:
    """
    Aligns prediction with ground-truth grading.

    Any item not present in the truth set is assigned score 0.

    Returns
    -------
    norm_truth : list[(item, score)]
    norm_pred  : list[(item, score)]
    """
    truth_map = dict(zip(truth_items, truth_scores))
    norm_truth = list(truth_map.items())
    norm_pred = [(it, truth_map.get(it, 0.0)) for it in pred_items]
    return norm_truth, norm_pred
