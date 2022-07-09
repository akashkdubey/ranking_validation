# Ranking Validation
This python package generates two ranking validation reports, one is a query level report that would contain information
on the metrics provided as input and the overall metrics report which is a summarised version of query level report. 


# Introduction
The packages essentially compares two ranked lists at its core but scales it to a dataframe and automates the process of 
comparing two ranked lists for different queries for different set of metrics and at different cutoffs.

# Usage

## Installation using pip

`pip install rank-validation`

## Getting the validation report
```
from rank_validation.validation_generator import get_metrics_report
query_report, overall_report = get_metrics_report(df, truth item column, truth score column, pred item col, metrics, cutoff)
```
<br>
Current version supports the following metrics: <br>
<br>
- ndcg <br>
- kendall_tau <br>
- recall <br>
- rbo <br>
<br >

**Note** : rbo might run into errors if : `len(set(ground_truth_items)) != len(set(predicted_items))`, so if rbo gives an
error, try generating the report with other metrics available.






