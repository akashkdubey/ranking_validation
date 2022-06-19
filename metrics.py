import rbo
from scipy import stats
from typing import Optional
import numpy as np


def ndcg(true_relevance: list, predicted_relevance: list, cutoff: int):
    def cumm_gain(relevance_scores: list, log_vals: list, cutoff: int):
        cumm_sum = 0
        for index, relevance_value in enumerate(relevance_scores[:cutoff]):
            num = 2 ** relevance_value - 1
            den = log_vals[index]
            cumm_sum += num / den
        return cumm_sum

    log_vals = np.log2([index + 1 for index in range(1, len(true_relevance) + 1)])
    dcg = cumm_gain(predicted_relevance, log_vals, cutoff)
    ideal_dcg = cumm_gain(true_relevance, log_vals, cutoff)
    ndcg_score = dcg / ideal_dcg
    return ndcg_score


def recall(true_relevance: list, predicted_relevance: list, cutoff: int):
    intersection = set(true_relevance[:cutoff]) & set(predicted_relevance[:cutoff])
    recall_score = len(intersection) / len(true_relevance)
    return recall_score


def kendall_tau(true_relevance: list, predicted_relevance: list, cutoff: int):
    tau, _ = stats.kendalltau(true_relevance[:cutoff], predicted_relevance[:cutoff])
    return tau


def rbo_sim(true_relevance: list, predicted_relevance: list, cutoff: int):
    rbo_score = rbo.RankingSimilarity(true_relevance[:cutoff], predicted_relevance[:cutoff]).rbo()
    return rbo_score
