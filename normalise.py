import collections


class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, item):
        return item in self.d

    def __iter__(self):
        return iter(self.d)


def normalise_for_kendalltau(true_relevance, predicted_relevance):
    add_to_true_rel = list(OrderedSet(predicted_relevance) - OrderedSet(true_relevance))
    add_to_pred_rel = list(OrderedSet(true_relevance) - OrderedSet(predicted_relevance))
    norm_true_rel = list(OrderedSet(true_relevance)) + add_to_true_rel
    norm_pred_rel = list(OrderedSet(predicted_relevance)) + add_to_pred_rel
    return norm_true_rel, norm_pred_rel


def normalise_relevance(true_relevance_items: list, true_relevance_scores: list,
                        predicted_relevance_items: list):
    normalised_true_relevance = list(zip(true_relevance_items, true_relevance_scores))

    true_relevance_dict = dict(normalised_true_relevance)
    normalised_predicted_relevance = []  # list of tuples
    for predicted_item in predicted_relevance_items:
        if predicted_item in true_relevance_dict:
            normalised_predicted_relevance.append((predicted_item, true_relevance_dict[predicted_item]))
        else:
            normalised_predicted_relevance.append((predicted_item, 0))

    return normalised_true_relevance, normalised_predicted_relevance
