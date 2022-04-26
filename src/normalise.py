from utils import OrderedSet


def normalise_for_kendalltau(true_relevance, predicted_relevance):
    add_to_true_rel = list(OrderedSet(predicted_relevance) - OrderedSet(true_relevance))
    add_to_pred_rel = list(OrderedSet(true_relevance) - OrderedSet(predicted_relevance))
    norm_true_rel = list(OrderedSet(true_relevance)) + add_to_true_rel
    norm_pred_rel = list(OrderedSet(predicted_relevance)) + add_to_pred_rel
    return norm_true_rel, norm_pred_rel
