class ValidationGenerator:

    def __init__(self, true_relevance_items=None, true_relevance_scores=None, predicted_relevance_items=None):
        self.true_relevance_items = true_relevance_items
        self.true_relevance_scores = true_relevance_scores
        self.predicted_relevance_items = predicted_relevance_items

    @staticmethod
    def prep_relevances(true_relevance_items: list, true_relevance_scores: list,
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
