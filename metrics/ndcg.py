import math


def cumm_gain(relevance_scores: list, log_vals: list, cutoff: int) -> int:
    cumm_sum: int = 0
    for index, relevance_value in enumerate(relevance_scores[:cutoff]):
        num = 2 ** relevance_value - 1
        den = log_vals[index]
        cumm_sum += num / den
    return cumm_sum


def ndcg(true_relevance: list, predicted_relevance: list, cutoff: int) -> int:
    log_vals = [math.log(i + 1) for i in range(1, len(true_relevance))]
    dcg = cumm_gain(predicted_relevance, log_vals, cutoff)
    ideal_dcg = cumm_gain(true_relevance, log_vals, cutoff)
    ndcg_score = dcg/ideal_dcg
    return ndcg_score


