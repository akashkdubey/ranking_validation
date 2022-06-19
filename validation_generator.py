import pandas as pd

from ranking_validation.normalise import normalise_relevance
from ranking_validation.metrics import *
import swifter


class ValidationGenerator:

    def __init__(self):
        pass

    @staticmethod
    def prepare_relevance(truth_items: list, truth_scores: list, pred_items: list):

        truth_rel, pred_rel = normalise_relevance(truth_items, truth_scores, pred_items)

        truth_rel_items, truth_rel_scores = list(map(list, zip(*truth_rel)))
        pred_rel_items, pred_rel_scores = list(map(list, zip(*pred_rel)))

        return pd.Series([truth_rel_items, truth_rel_scores, pred_rel_items, pred_rel_scores])

    @staticmethod
    def create_metric_cols(df, truth_item_col, truth_score_col, pred_item_col, pred_score_col, metric_list,
                           cutoff_list):

        if "ndcg" in metric_list:
            for cutoff in cutoff_list:
                col_name = "ndcg@" + str(cutoff)
                df[col_name] = df.swifter.apply(lambda x: ndcg(x[truth_score_col], x[pred_score_col], cutoff), axis=1)

        if "recall" in metric_list:
            for cutoff in cutoff_list:
                col_name = "recall@" + str(cutoff)
                df[col_name] = df.swifter.apply(lambda x: recall(x[truth_item_col], x[pred_item_col], cutoff), axis=1)

        if "kendall_tau" in metric_list:
            for cutoff in cutoff_list:
                col_name = "kendall_tau@" + str(cutoff)
                df[col_name] = df.swifter.apply(lambda x: kendall_tau(x[truth_item_col], x[pred_item_col], cutoff), axis=1)

        if "rbo" in metric_list:
            for cutoff in cutoff_list:
                col_name = "rbo@" + str(cutoff)
                df[col_name] = df.swifter.apply(lambda x: rbo_sim(x[truth_item_col], x[pred_item_col], cutoff), axis=1)

        return df

    @staticmethod
    def get_metrics_report(df, truth_item_col, truth_score_col, pred_item_col, metric_list, cutoff_list):
        df[[truth_item_col, truth_score_col, pred_item_col, "pred_score_col"]] = df.swifter.apply(
            lambda x: ValidationGenerator.prepare_relevance(x[truth_item_col], x[truth_score_col], x[pred_item_col]), axis=1)
        df_scored = ValidationGenerator.create_metric_cols(df, truth_item_col, truth_score_col,
                                                           pred_item_col, "pred_score_col", metric_list, cutoff_list)
        df_scored = df_scored.drop([truth_item_col, truth_score_col, pred_item_col, "pred_score_col"], axis=1)
        return df_scored


if __name__ == "__main__":
    pass
