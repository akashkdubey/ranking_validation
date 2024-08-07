import collections
from collections.abc import Set
from typing import List, Tuple


class OrderedSet(Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, item):
        return item in self.d

    def __iter__(self):
        return iter(self.d)


def normalise_for_kendalltau(true_relevance: List[str],
                             predicted_relevance: List[str],
                             cutoff: int) -> Tuple[List[str], List[str]]:
    """
    :param true_relevance: list of ground truth items (doc ids).
    :param predicted_relevance: list of predicted items (doc ids).
    :param cutoff: The depth of the list to consider for rank validation.
    :return: Normalised true relevance items (doc ids) and normalised predicted relevance items (doc ids)
    """

    add_to_true_rel = list(OrderedSet(predicted_relevance[:cutoff]) - OrderedSet(true_relevance[:cutoff]))
    add_to_pred_rel = list(OrderedSet(true_relevance[:cutoff]) - OrderedSet(predicted_relevance[:cutoff]))
    norm_true_rel = list(OrderedSet(true_relevance[:cutoff])) + add_to_true_rel
    norm_pred_rel = list(OrderedSet(predicted_relevance[:cutoff])) + add_to_pred_rel
    return norm_true_rel, norm_pred_rel


def normalise_relevance(true_relevance_items: List[str],
                        true_relevance_scores: List[float],
                        predicted_relevance_items: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    :param true_relevance_items: list of ground truth items (doc ids)
    :param true_relevance_scores: list of relevancy scores, corresponding to the items(doc ids) in truth_items
    :param predicted_relevance_items: list of predicted lists of the items(doc ids)
    :return: Two list of tuples with (item, score), each for ground truth and the prediction
    """

    normalised_true_relevance = list(zip(true_relevance_items, true_relevance_scores))
    true_relevance_dict = dict(normalised_true_relevance)
    normalised_predicted_relevance = []  # list of tuples
    for predicted_item in predicted_relevance_items:
        if predicted_item in true_relevance_dict:
            normalised_predicted_relevance.append((predicted_item, true_relevance_dict[predicted_item]))
        else:
            normalised_predicted_relevance.append((predicted_item, 0))
    return normalised_true_relevance, normalised_predicted_relevance
