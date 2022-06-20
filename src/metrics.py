import rbo
import numpy as np

from scipy import stats
from typing import List

from normalise import normalise_for_kendalltau


def ndcg(true_relevance: List[float],
         predicted_relevance: List[float],
         cutoff: int) -> float:

    def cumm_gain(relevance_scores: List[float], log_vals: List[int], cutoff: int):
        cumm_sum = 0
        for index, relevance_value in enumerate(relevance_scores[:cutoff]):
            num = 2 ** relevance_value - 1
            den = log_vals[index]
            cumm_sum += num / den
        return cumm_sum

    log_vals = np.log2([index + 1 for index in range(1, cutoff + 2)])
    dcg = cumm_gain(predicted_relevance, log_vals, cutoff)
    ideal_dcg = cumm_gain(true_relevance, log_vals, cutoff)
    ndcg_score = dcg / ideal_dcg
    return ndcg_score


def recall(true_relevance: List[str],
           predicted_relevance: List[str],
           cutoff: int) -> float:

    intersection = set(true_relevance[:cutoff]) & set(predicted_relevance[:cutoff])
    recall_score = len(intersection) / len(true_relevance)
    return recall_score


def kendall_tau(true_relevance:  List[str],
                predicted_relevance:  List[str],
                cutoff: int) -> float:

    true_relevance, predicted_relevance = normalise_for_kendalltau(true_relevance, predicted_relevance, cutoff)
    tau, _ = stats.kendalltau(true_relevance[:cutoff], predicted_relevance[:cutoff])
    return tau


def rbo_sim(true_relevance:  List[str],
            predicted_relevance:  List[str],
            cutoff: int) -> float:

    rbo_score = rbo.RankingSimilarity(true_relevance[:cutoff], predicted_relevance[:cutoff]).rbo()
    return rbo_score
