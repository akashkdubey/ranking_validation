import pandas as pd
from typing import List, Tuple
import swifter
from metrics import ndcg, recall, kendall_tau, rbo_sim
from normalise import normalise_relevance


def prepare_relevance(truth_items: List[str],
                      truth_scores: List[float],
                      pred_items: List[str]) -> pd.Series:
    """
    :param truth_items:  list of ground truth items (doc ids)
    :param truth_scores: list of relevancy scores, corresponding to the items(doc ids) in truth_items
    :param pred_items:   list of predicted lists of the items(doc ids)
    :return: normalised and prepared ground truth items, scores corresponding to the ground truth items(doc ids), predicted list of ranked items,
             scores of items(doc ids) corresponding to the predicted list of items(doc ids).
    """
    truth_rel, pred_rel = normalise_relevance(truth_items, truth_scores, pred_items)
    truth_rel_items, truth_rel_scores = list(map(list, zip(*truth_rel)))
    pred_rel_items, pred_rel_scores = list(map(list, zip(*pred_rel)))
    return pd.Series([truth_rel_items, truth_rel_scores, pred_rel_items, pred_rel_scores])


def create_metric_cols(df: pd.DataFrame,
                       truth_item_col: pd.Series,
                       truth_score_col: pd.Series,
                       pred_item_col: pd.Series,
                       pred_score_col: pd.Series,
                       metric_list: List[str],
                       cutoff_list: List[int]) -> pd.DataFrame:
    """
    :param df: A dataframe with columns : Query, truth_item_col, truth_score_col, pred_item_col, pred_score_col
    :param truth_item_col: Dataframe column of ground truth items(doc ids) (Each entry of the column should be a list)
    :param truth_score_col: Dataframe column of ground truth scores (Each entry of the column should be a list)
    :param pred_item_col: Dataframe column of predicted items(doc ids) (Each entry of the column should be a list)
    :param pred_score_col: Dataframe column of predicted scores (Each entry of the column should be a list)
    :param metric_list: List of metrics to be chose from ['ndcg', 'kendall_tau', 'rbo', 'recall']
    :param cutoff_list: List of different cutoffs (corresponding to the depth of the list) for which we want the validation metrics.
    :return: A dataframe with the desired metric report
    """
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


def get_metrics_report(df: pd.DataFrame,
                       truth_item_col: pd.Series,
                       truth_score_col: pd.Series,
                       pred_item_col: pd.Series,
                       metric_list: List[str],
                       cutoff_list: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param df: A dataframe with columns : Query, truth_item_col, truth_score_col, pred_item_col, pred_score_col.
    :param truth_item_col: A dataframe column with ground truth list of items(doc ids).
    :param truth_score_col: A dataframe column with list of scores, corresponding to the items(doc ids) in ground truth col
    :param pred_item_col: A dataframe column with predicted ranked lists.
    :param metric_list: List of metrics to be chosen from ['ndcg', 'kendall_tau', 'rbo', 'recall'].
    :param cutoff_list: List of different cutoffs (corresponding to the depth of the list) for which we want the validation metrics.
    :return: Two dataframes with query level report and overall summary report.
    """

    df[[truth_item_col, truth_score_col, pred_item_col, "pred_score_col"]] = df.swifter.apply(
        lambda x: prepare_relevance(x[truth_item_col],
                                    x[truth_score_col],
                                    x[pred_item_col]),
        axis=1)
    df_scored = create_metric_cols(df, truth_item_col, truth_score_col, pred_item_col, "pred_score_col",
                                   metric_list, cutoff_list)
    query_level_report = df_scored.drop([truth_item_col, truth_score_col, pred_item_col, "pred_score_col"], axis=1)
    overall_summary_report = query_level_report.describe()
    return query_level_report, overall_summary_report
