import csv
import os
import pandas as pd
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from ..Utils.resources import SharedResources


def compute_fold_average(folder, data=None, metrics=[], suffix=''):
    """

    """
    classes = SharedResources.getInstance().validation_class_names
    results = None
    if data is None:
        results_filename = os.path.join(folder, 'Validation', 'all_scores.csv')
        tmp = pd.read_csv(results_filename)
        use_cols = list(tmp.columns)[0:SharedResources.getInstance().upper_default_metrics_index] + metrics
        results = pd.read_csv(results_filename, usecols=use_cols)
    else:
        results = deepcopy(data)

    results.replace('inf', np.nan, inplace=True)
    results.replace(float('inf'), np.nan, inplace=True)
    results.replace('', np.nan, inplace=True)
    results.replace(' ', np.nan, inplace=True)
    unique_folds = np.unique(results['Fold'])
    nb_folds = len(unique_folds)
    metrics_per_fold = []
    classwise_per_fold = []

    metric_names = list(results.columns[3:])
    # if metrics is not None:
    #     metric_names.extend(metrics)

    fold_average_columns = ['Fold', '# samples', '# TP', '# FP', '# FN', '# TN', "Avg TP probability",
                            "Recall", "Precision", "Specificity", "F1-score", "Accuracy", "Balanced accuracy"]
    # for m in metric_names:
    #     fold_average_columns.extend([m + ' (Mean)', m + ' (Std)'])

    for f in unique_folds:
        multiclass_results, classwise_metrics = compute_classwise_fold_metrics(results, f, classes=classes)
        # fold_average_metrics, fold_std_metrics = compute_singe_fold_average_metrics(classwise_metrics)
        fold_metrics = []
        # for m in range(len(fold_average_metrics)):
        #     fold_metrics.append(fold_average_metrics[m])
        #     fold_metrics.append(fold_std_metrics[m])
        fold_average = [f] + multiclass_results + fold_metrics
        metrics_per_fold.append(fold_average)
        classwise_per_fold.append(classwise_metrics)

    metrics_per_fold_df = pd.DataFrame(data=metrics_per_fold, columns=fold_average_columns)
    study_filename = os.path.join(folder, 'Validation', 'folds_multiclass_metrics_average.csv') if suffix == '' else os.path.join(folder,
                                                                                                 'Validation',
                                                                                                 '_folds_multiclass_metrics_average_' + suffix + '.csv')
    metrics_per_fold_df.to_csv(study_filename, index=False)

    #Saving the classwise results to file
    for c in classes:
        class_filename = os.path.join(folder, 'Validation',
                                      f'folds_{c}_metrics_average.csv') if suffix == '' else os.path.join(folder,
                                                                                                                'Validation',
                                                                                                                f'_folds_{c}_metrics_average_' + suffix + '.csv')
        res = []
        for cf in range(len(classwise_per_fold)):
            res.append([cf] + classwise_per_fold[cf][c])
        class_metrics_per_fold_df = pd.DataFrame(data=res, columns=fold_average_columns)
        class_metrics_per_fold_df.to_csv(class_filename, index=False)


    ####### Averaging the results from the different folds ###########
    total_samples = metrics_per_fold_df['# samples'].sum()
    total_fold_metrics_to_average = metrics_per_fold_df.values[:, 6:]

    # Performing pooled estimates (taking into account the sample size for each fold) when relevant
    pooled_fold_averaged_results = [len(unique_folds), total_samples]
    for m in range(total_fold_metrics_to_average.shape[1]):
        mean_final = 0
        for f in range(len(unique_folds)):
            fold_val = total_fold_metrics_to_average[f, m]
            fold_sample_size = list(metrics_per_fold_df['# samples'])[f]
            mean_final = mean_final + (fold_val * fold_sample_size)
        mean_final = mean_final / total_samples
        std_final = np.std(total_fold_metrics_to_average[:, m])
        pooled_fold_averaged_results.extend([mean_final, std_final])

    overall_average_columns = ['Fold', '# samples']
    for m in ['TP proba avg', 'Patient-wise recall', 'Patient-wise precision', 'Patient-wise specificity', 'Patient-wise F1',
              'Patient-wise Accuracy', 'Patient-wise Balanced accuracy']:
        overall_average_columns.extend([m + ' (Mean)', m + ' (Std)'])

    # for m in metric_names:
    #     overall_average_columns.extend([m + ' (Mean)', m + ' (Std)'])
    pooled_fold_averaged_results_df = pd.DataFrame(data=np.asarray(pooled_fold_averaged_results).reshape(1, len(overall_average_columns)), columns=overall_average_columns)
    study_filename = os.path.join(folder, 'Validation', 'multiclass_overall_metrics_average.csv') if suffix == '' else os.path.join(folder,
                                                                                                 'Validation',
                                                                                                 'multiclass_overall_metrics_average_' + suffix + '.csv')
    pooled_fold_averaged_results_df.to_csv(study_filename, index=False)


def compute_singe_fold_average_metrics(results, fold_number, best_threshold, best_overlap, metric_names,
                                       positive_detected_state=False):
    fold_results = results.loc[results['Fold'] == fold_number]
    thresh_index = (np.round(fold_results['Threshold'], 1) == best_threshold)
    all_for_thresh = fold_results.loc[thresh_index]
    if len(all_for_thresh) == 0:
        # Empty fold? Can indicate something went wrong, or was not computed properly beforehand
        return None
    if positive_detected_state:
        all_for_thresh = all_for_thresh.loc[all_for_thresh['PiW Dice'] >= best_overlap]
    upper_default_metrics_index = SharedResources.getInstance().upper_default_metrics_index
    default_metrics_average = list(np.mean(all_for_thresh.values[:, 3:upper_default_metrics_index], axis=0))
    default_metrics_std = [np.std(all_for_thresh.values[:, x], axis=0) for x in range(3, upper_default_metrics_index)]

    extra_metrics_average = []
    extra_metrics_std = []
    for m in metric_names:
        if m in fold_results.columns.values:
            if m in ['HD95', 'ASSD', 'RAVD', 'VC', 'OASSD']:
                avg = fold_results.loc[thresh_index][fold_results.loc[thresh_index][m] != -1.0][m].dropna().astype(
                    'float32').mean()
                std = fold_results.loc[thresh_index][fold_results.loc[thresh_index][m] != -1.0][m].dropna().astype(
                    'float32').std(ddof=0)
            else:
                avg = all_for_thresh[m].dropna().astype('float32').mean()
                std = all_for_thresh[m].dropna().astype('float32').std(ddof=0)
            extra_metrics_average.append(avg)
            extra_metrics_std.append(std)

    return default_metrics_average + extra_metrics_average, default_metrics_std + extra_metrics_std


def compute_classwise_fold_metrics(results, fold_number, classes):
    fold_results = results.loc[results['Fold'] == fold_number]
    classwise_results = {}

    if len(fold_results) == 0:
        # Empty fold? Can indicate something went wrong, or was not computed properly beforehand
        for c in classes:
            classwise_results[c] = [-1., -1., -1., -1., -1., -1., -1., -1.]
        return classwise_results

    for c in classes:
        class_results = fold_results.loc[results['GT'] == c]
        nb_samples = len(class_results)
        true_positives = fold_results.loc[(fold_results['GT'] == c) & (fold_results['Prediction'] == c)]
        avg_true_positive_pred = fold_results.loc[(fold_results['GT'] == c) & (fold_results['Prediction'] == c)]["Proba "+c].mean()
        false_positives = fold_results.loc[(fold_results['GT'] != c) & (fold_results['Prediction'] == c)]
        false_negatives = fold_results.loc[(fold_results['GT'] == c) & (fold_results['Prediction'] != c)]
        true_negatives = fold_results.loc[(fold_results['GT'] != c) & (fold_results['Prediction'] != c)]
        # avg_false_negative_pred = fold_results.loc[(fold_results['GT'] == c) & (fold_results['Prediction'] != c)]
        class_recall = len(true_positives) / (len(true_positives) + len(false_negatives) + 1e-6)
        class_precision = len(true_positives) / (len(true_positives) + len(false_positives) + 1e-6)
        class_specificity = 100. if len(true_negatives) + len(false_positives) == 0 else len(true_negatives) / (
                    len(true_negatives) + len(false_positives) + 1e-6)
        class_f1 = 2 * len(true_positives) / (
                    (2 * len(true_positives)) + len(false_positives) + len(false_negatives) + 1e-6)
        accuracy = (len(true_positives) + len(true_negatives)) / (
                    len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives))
        balanced_accuracy = (class_recall + class_specificity) / 2
        classwise_results[c] = [nb_samples, len(true_positives), len(false_positives), len(false_negatives),
                                len(true_negatives), avg_true_positive_pred, class_recall, class_precision,
                                class_specificity, class_f1, accuracy, balanced_accuracy]

    multiclass_results = [len(fold_results)]
    for i in range(1, 5):
        sum = 0
        for c in classes:
            sum = sum + classwise_results[c][i]
        multiclass_results.append(sum)
    for i in range(5, 12):
        macro_average = 0
        for c in classes:
            macro_average = macro_average + (classwise_results[c][0] * classwise_results[c][i])
        macro_average = macro_average / len(fold_results)
        multiclass_results.append(macro_average)
    return multiclass_results, classwise_results

