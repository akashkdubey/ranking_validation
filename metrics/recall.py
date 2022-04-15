def recall(true_relevance: list, predicted_relevance: list, cutoff: int):
    intersection = set(true_relevance[:cutoff]) & set(predicted_relevance[:cutoff])
    recall_score = len(intersection)/len(true_relevance)
    return recall_score

