from scipy import stats


def kendall_tau(true_relevance: list, predicted_relevance: list, cutoff: int):
    tau, _ = stats.kendalltau(true_relevance[:cutoff], predicted_relevance[:cutoff])
    return tau
