import csv
import os
import pandas as pd
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from Utils.latex_converter import *


def best_segmentation_probability_threshold_analysis(folder, detection_overlap_thresholds=None):
    """
    The best threshold probability and object overlap are determined based on a combination of overall DICE
    performance and F1-score. The recall here is not object-wise (i.e., each tumor part not considered individually),
    but at a patient-level based on overall Dice and overlap cut-off.
    :param folder: main validation directory containing the all_dice_scores.csv file.
    :param detection_overlap_thresholds: list of threshold values (float) to use in the range [0., 1.].
    :return: optimal probability threshold and Dice cut-off.
    """
    patient_dices_filename = os.path.join(folder, 'Validation', 'all_dice_scores.csv')
    study_filename = os.path.join(folder, 'Validation', 'optimal_dice_study.csv')
    study_file = open(study_filename, 'w')
    study_writer = csv.writer(study_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    study_writer.writerow(
        ['Detection threshold', 'Dice threshold', 'Mean dice', 'Nb found', 'Nb total', 'Recall', 'Precision', 'F1'])

    all_dices = pd.read_csv(patient_dices_filename)
    object_detection_dice_thresholds = [0.]
    if detection_overlap_thresholds is not None and type(detection_overlap_thresholds) is list:
        object_detection_dice_thresholds = detection_overlap_thresholds

    thresholds = list(np.unique([np.round(x, 1) for x in list(np.unique(all_dices['Threshold'].values))]))
    nb_thresh = len(thresholds)
    recall_precision_results = []

    max_global_metrics_value = 0
    max_threshold = None
    max_overlap = None
    for obj in object_detection_dice_thresholds:
        for thr in range(nb_thresh):
            dices = all_dices['Dice'].values[thr::nb_thresh]
            # using x <= obj brings up results since more leniant
            mean = np.ma.masked_array(dices, [np.round(x, 3) < obj for x in dices]).mean()
            detection = [np.round(x, 3) > obj for x in dices]
            found = np.count_nonzero(detection)

            gt = all_dices['#GT'].values[thr::nb_thresh]
            relevant = np.count_nonzero(gt)
            recall = found / relevant
            #recall = found / len(dices)
            #
            # det = all_dices['#Det'].values[thr::nb_thresh]
            # selected = np.count_nonzero(det)
            # if selected > 0:
            #     precision = found / selected
            # else:
            #     precision = 1
            all_for_thresh = all_dices.values[thr::nb_thresh]
            precisions = []
            # The proper overall precision computation can be debated?
            for x, xval in enumerate(all_for_thresh):
                tmp = pd.DataFrame(xval.reshape((1, len(xval))), columns=all_dices.columns)
                if np.round(tmp['Dice'].values[0], 3) > obj and tmp['#GT'].values[0] > 0:
                    precisions.append(tmp['Inst Precision'].values[0])

            precision = np.mean(precisions)

            F1 = 2 * ((precision * recall) / (precision + recall))
            recall_precision_results.append([obj, thresholds[thr], mean, found, len(dices), recall, precision, F1])
            study_writer.writerow([obj, thresholds[thr], mean, found, len(dices), recall, precision, F1])

            global_result = (mean + F1) / 2.
            if global_result > max_global_metrics_value:
                max_global_metrics_value = global_result
                max_threshold = thresholds[thr]
                max_overlap = obj

    # Small trick to comply with further computation. Otherwise, patients with a 0% Dice detection would still be
    # considered as true positives, which is a behaviour to avoid.
    if max_overlap == 0.:
        max_overlap = 0.01
    study_writer.writerow(['', '', '', '', '', '', '', ''])
    study_writer.writerow([max_overlap, max_threshold, '', '', '', '', '', ''])
    study_file.close()
    print('Selected values (Overlap: {}, Threshold: {}) for global metric of {}.'.format(max_overlap, max_threshold,
                                                                                         max_global_metrics_value))

    recall_precision_results = np.asarray(recall_precision_results)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    for o, obj in enumerate(object_detection_dice_thresholds):
        threshs = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 1]
        dices = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 2]
        F1s = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 7]
        ax.plot(threshs, dices, label=str(obj * 100) + '% overlap')
        ax2.plot(threshs, F1s, label=str(obj * 100) + '% overlap')
        ax3.plot(dices, F1s, label=str(obj * 100) + '% overlap')
        ax4.scatter(threshs, dices, label=str(obj * 100) + '% overlap' + '_Dice', marker='x')
        ax4.scatter(threshs, F1s, label=str(obj * 100) + '% overlap' + '_F1s', marker='o')

    ax.set(xlabel='Network probability threshold', ylabel='Dice')  # title='Dice over network probability.'
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.grid(linestyle='--')
    ax.legend()
    ax2.set(xlabel='Network probability threshold', ylabel='F1-score')  # , title='F1-score over network probability.'
    ax2.set_xlim(0., 1.)
    ax2.set_ylim(0., 1.)
    ax2.grid(linestyle='--')
    ax2.legend()
    ax3.set(xlabel='Dices', ylabel='F1-score')  # , title='F1-score over dice evolution.'
    ax3.set_xlim(0., 1.)
    ax3.set_ylim(0., 1.)
    ax3.grid(linestyle='--')
    ax3.legend()
    ax4.set(xlabel='Network probability threshold',
            ylabel='Probability')  # , title='Combined dice/F1-score over network probability.'
    ax4.set_xlim(0., 1.)
    ax4.set_ylim(0., 1.)
    ax4.grid(linestyle='--')
    ax4.legend(loc='lower center')
    # plt.show()

    os.makedirs(os.path.join(folder, 'Validation', 'OptimalSearch'), exist_ok=True)
    fig.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'dice_over_threshold.png'), dpi=300,
                bbox_inches="tight")
    fig2.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'F1_over_threshold.png'), dpi=300,
                 bbox_inches="tight")
    fig3.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'F1_over_dice.png'), dpi=300,
                 bbox_inches="tight")
    fig4.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'metrics_scatter_over_threshold.png'), dpi=300,
                 bbox_inches="tight")

    return max_overlap, max_threshold


def best_segmentation_probability_threshold_analysis_postop(folder, detection_overlap_thresholds=None):
    """
    The best threshold probability and object overlap are determined based on a combination of overall DICE
    performance and F1-score. The recall here is not object-wise (i.e., each tumor part not considered individually),
    but at a patient-level based on overall Dice and overlap cut-off.
    :param folder: main validation directory containing the all_dice_scores.csv file.
    :param detection_overlap_thresholds: list of threshold values (float) to use in the range [0., 1.].
    :return: optimal probability threshold and Dice cut-off.
    """
    patient_dices_filename = os.path.join(folder, 'Validation', 'all_dice_scores.csv')
    study_filename = os.path.join(folder, 'Validation', 'optimal_dice_study.csv')
    study_file = open(study_filename, 'w')
    study_writer = csv.writer(study_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    study_writer.writerow(
        ['Detection threshold', 'Dice threshold', 'Mean dice', 'Nb true_positives', 'Nb total', 'Recall', 'Precision', 'F1'])

    all_dices = pd.read_csv(patient_dices_filename)
    object_detection_dice_thresholds = [0.]
    if detection_overlap_thresholds is not None and type(detection_overlap_thresholds) is list:
        object_detection_dice_thresholds = detection_overlap_thresholds

    thresholds = list(np.unique([np.round(x, 1) for x in list(np.unique(all_dices['Threshold'].values))]))
    nb_thresh = len(thresholds)
    recall_precision_results = []

    max_global_metrics_value = 0
    max_threshold = None
    max_overlap = None
    for obj in object_detection_dice_thresholds:
        for thr in range(nb_thresh):
            dices = all_dices['Dice'].values[thr::nb_thresh]
            gt = all_dices['#GT'].values[thr::nb_thresh]
            det = all_dices['#Det'].values[thr::nb_thresh]

            # using x <= obj brings up results since more leniant
            mean = np.ma.masked_array(dices, [x < obj for x in dices]).mean()
            detection = [dices[i] > obj and det[i] > 0 and gt[i] > 0 for i in range(len(dices))]
            true_positives = np.count_nonzero(detection)

            relevant = np.count_nonzero(gt)
            recall = true_positives / relevant
            #recall = true_positives / len(dices)
            #
            # det = all_dices['#Det'].values[thr::nb_thresh]
            # selected = np.count_nonzero(det)
            # if selected > 0:
            #     precision = true_positives / selected
            # else:
            #     precision = 1
            # all_for_thresh = all_dices.values[thr::nb_thresh]
            # precisions = []
            # # The proper overall precision computation can be debated?
            # for x, xval in enumerate(all_for_thresh):
            #     tmp = pd.DataFrame(xval.reshape((1, len(xval))), columns=all_dices.columns)
            #     if tmp['Dice'].values[0] > obj and tmp['#GT'].values[0] > 0:
            #         precisions.append(tmp['Inst Precision'].values[0])

            det_positive = [dices[i] > obj and det[i] > 0 for i in range(len(dices))]
            precision = true_positives / np.count_nonzero(det_positive)

            F1 = 2 * ((precision * recall) / (precision + recall))
            recall_precision_results.append([obj, thresholds[thr], mean, true_positives, len(dices), recall, precision, F1])
            study_writer.writerow([obj, thresholds[thr], mean, true_positives, len(dices), recall, precision, F1])

            global_result = (mean + F1) / 2.
            if global_result > max_global_metrics_value:
                max_global_metrics_value = global_result
                max_threshold = thresholds[thr]
                max_overlap = obj

    # Small trick to comply with further computation. Otherwise, patients with a 0% Dice detection would still be
    # considered as true positives, which is a behaviour to avoid.
    if max_overlap == 0.:
        max_overlap = 0.01
    study_writer.writerow(['', '', '', '', '', '', '', ''])
    study_writer.writerow([max_overlap, max_threshold, '', '', '', '', '', ''])
    study_file.close()
    print('Selected values (Overlap: {}, Threshold: {}) for global metric of {}.'.format(max_overlap, max_threshold,
                                                                                         max_global_metrics_value))

    recall_precision_results = np.asarray(recall_precision_results)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    for o, obj in enumerate(object_detection_dice_thresholds):
        threshs = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 1]
        dices = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 2]
        F1s = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 7]
        ax.plot(threshs, dices, label=str(obj * 100) + '% overlap')
        ax2.plot(threshs, F1s, label=str(obj * 100) + '% overlap')
        ax3.plot(dices, F1s, label=str(obj * 100) + '% overlap')
        ax4.scatter(threshs, dices, label=str(obj * 100) + '% overlap' + '_Dice', marker='x')
        ax4.scatter(threshs, F1s, label=str(obj * 100) + '% overlap' + '_F1s', marker='o')

    ax.set(xlabel='Network probability threshold', ylabel='Dice')  # title='Dice over network probability.'
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.grid(linestyle='--')
    ax.legend()
    ax2.set(xlabel='Network probability threshold', ylabel='F1-score')  # , title='F1-score over network probability.'
    ax2.set_xlim(0., 1.)
    ax2.set_ylim(0., 1.)
    ax2.grid(linestyle='--')
    ax2.legend()
    ax3.set(xlabel='Dices', ylabel='F1-score')  # , title='F1-score over dice evolution.'
    ax3.set_xlim(0., 1.)
    ax3.set_ylim(0., 1.)
    ax3.grid(linestyle='--')
    ax3.legend()
    ax4.set(xlabel='Network probability threshold',
            ylabel='Probability')  # , title='Combined dice/F1-score over network probability.'
    ax4.set_xlim(0., 1.)
    ax4.set_ylim(0., 1.)
    ax4.grid(linestyle='--')
    ax4.legend(loc='lower center')
    # plt.show()

    os.makedirs(os.path.join(folder, 'Validation', 'OptimalSearch'), exist_ok=True)
    fig.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'dice_over_threshold.png'), dpi=300,
                bbox_inches="tight")
    fig2.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'F1_over_threshold.png'), dpi=300,
                 bbox_inches="tight")
    fig3.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'F1_over_dice.png'), dpi=300,
                 bbox_inches="tight")
    fig4.savefig(os.path.join(folder, 'Validation', 'OptimalSearch', 'metrics_scatter_over_threshold.png'), dpi=300,
                 bbox_inches="tight")

    return max_overlap, max_threshold


def compute_fold_average(folder, data=None, best_threshold=0.5, best_overlap=0.0, metrics=[], suffix='', dice_fixed_metric=False,
                         output_folder=None):
    """
    :param folder: Main study folder where the results will be dumped (assuming inside a Validation sub-folder)
    :param best_threshold:
    :param best_overlap:
    :param metric_names:
    :return:
    """
    if output_folder is None:
        output_folder = os.path.join(folder, 'Validation')

    results = None
    if data is None:
        results_filename = os.path.join(output_folder, 'all_dice_scores.csv')
        results = pd.read_csv(results_filename)
    else:
        results = deepcopy(data)

    results.replace(np.inf, np.nan, inplace=True)
    results.replace('inf', np.nan, inplace=True)
    results.replace('', np.nan, inplace=True)
    results.replace(' ', np.nan, inplace=True)
    metric_names = ['Dice', 'Dice-TP', 'Dice-P', 'Dice-N']
    if metrics is not None:
        metric_names.extend(metrics)
        # 'Inst DICE', 'Inst Recall', 'Inst Precision', 'VS', 'IOU', 'MI', 'ARI', 'Jaccard', 'TPR',
        #             'TNR', 'FPR', 'FNR', 'PPV', 'AUC', 'MCC', 'CKS', 'HD95', 'ASSD', 'RAVD', 'VC', 'OASSD']
    unique_folds = np.unique(results['Fold'])
    nb_folds = len(unique_folds)
    metrics_per_fold = []
    fold_average_columns = ['Fold', '# samples', 'Patient-wise recall', 'Patient-wise precision', 'Patient-wise F1',
                            'FPPP', 'Object-wise recall', 'Object-wise precision', 'Object-wise F1',
                            'Patient-wise recall postop', 'Patient-wise precision postop', 'Patient-wise F1 postop',
                            'Accuracy', 'Balanced accuracy', 'Positive rate', 'Negative rate', 'Specificity']
    for m in metric_names:
        if m in results.columns.values or 'Dice' in m:
            fold_average_columns.extend([m + '_mean', m + '_std'])
        if m == 'Absolute volume error':
            fold_average_columns.extend([m + '_median'])

    # Regarding the overlap threshold, should the patient discarded for recall be
    # used for other metrics computation?
    for f in unique_folds:
        fold_results = results.loc[results['Fold'] == f]
        thresh_index = (np.round(fold_results['Threshold'], 1) == best_threshold)
        all_for_thresh = fold_results.loc[thresh_index] #fold_results.loc[fold_results['Threshold'] == best_threshold]
        if len(all_for_thresh) == 0: # Empty fold? Can indicate something went wrong, or was not computed properly beforehand
            fold_average = [f, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            metrics_per_fold.append(fold_average)
            continue

        nb_missed_tumors = len(fold_results.loc[thresh_index & (fold_results['#GT'] > 0) & (fold_results['Dice'] < best_overlap)])
        nb_tumors = len(fold_results.loc[thresh_index & (fold_results['#GT'] > 0)])
        patient_wise_recall = 1 - (nb_missed_tumors / nb_tumors) #1 - (nb_missed_tumors/len(all_for_thresh))
        patient_wise_precision = fold_results.loc[thresh_index & (fold_results['#Det'] > 0)]['Inst Precision'].mean() # all_for_thresh.loc[all_for_thresh['#Det'] > 0]['Inst Precision'].mean()
        fppp = np.subtract(all_for_thresh['#Det'].values,
                           np.multiply(all_for_thresh['Inst Precision'].values,
                                       all_for_thresh['#Det'].values)).mean()
        objects_total = all_for_thresh['#GT'].values.sum()
        objects_found = np.multiply(all_for_thresh['Inst Recall'].values, all_for_thresh['#GT'].values).sum()
        object_wise_recall = objects_found / objects_total
        objects_false_positive_total = np.subtract(all_for_thresh['#Det'].values,
                                        np.multiply(all_for_thresh['Inst Precision'].values,
                                                    all_for_thresh['#Det'].values)).sum()
        object_wise_precision = 1 - (objects_false_positive_total / all_for_thresh['#Det'].sum())
        patient_wise_F1 = 2 * ((patient_wise_precision * patient_wise_recall) / (patient_wise_precision + patient_wise_recall))
        object_wise_F1 = 2 * ((object_wise_precision * object_wise_recall) / (object_wise_precision + object_wise_recall))

        true_positives = len(all_for_thresh.loc[(all_for_thresh['#Det'] > 0) & (all_for_thresh['#GT'] > 0)])
        false_negatives = len(all_for_thresh.loc[(all_for_thresh['#Det'] == 0) & (all_for_thresh['#GT'] > 0)])
        false_positives = len(all_for_thresh.loc[(all_for_thresh['#Det'] > 0) & (all_for_thresh['#GT'] == 0)])
        true_negatives = len(all_for_thresh.loc[(all_for_thresh['#Det'] == 0) & (all_for_thresh['#GT'] == 0)])

        patient_wise_recall_postop = true_positives / (true_positives + false_negatives)
        patient_wise_precision_postop = true_positives / (true_positives + false_positives)
        patient_wise_F1_postop = 2 * ((patient_wise_recall_postop * patient_wise_precision_postop) / (patient_wise_precision_postop + patient_wise_recall_postop))
        accuracy = (true_positives + true_negatives ) / (true_negatives + true_positives + false_negatives + false_positives)
        specificity = 1 if (true_negatives + false_positives) == 0 else true_negatives / (true_negatives + false_positives)
        balanced_accuracy = (patient_wise_recall_postop + specificity) / 2

        fold_average = [f, len(np.unique(fold_results['Patient'].values)), patient_wise_recall, patient_wise_precision,
                        patient_wise_F1, fppp, object_wise_recall, object_wise_precision, object_wise_F1,
                        patient_wise_recall_postop, patient_wise_precision_postop, patient_wise_F1_postop, accuracy, balanced_accuracy,
                        (true_positives + false_negatives)/len(all_for_thresh), (false_positives + true_negatives)/len(all_for_thresh), specificity]
        for m in metric_names:
            if m in fold_results.columns.values:
                if m in ['HD95', 'ASSD', 'RAVD', 'VC', 'OASSD']:
                    # avg = all_for_thresh[all_for_thresh[m] != -1.0][m].dropna().astype('float32').mean()
                    # std = all_for_thresh[all_for_thresh[m] != -1.0][m].dropna().astype('float32').std(ddof=0)
                    avg = fold_results.loc[thresh_index][fold_results.loc[thresh_index][m] != -1.0][m].dropna().astype('float32').mean()
                    std = fold_results.loc[thresh_index][fold_results.loc[thresh_index][m] != -1.0][m].dropna().astype('float32').std(ddof=0)
                else:
                    avg = all_for_thresh[m].dropna().astype('float32').mean()
                    std = all_for_thresh[m].dropna().astype('float32').std(ddof=0)
                fold_average.extend([avg, std])

            elif m == 'Dice-TP':
                true_positives = all_for_thresh.loc[(all_for_thresh['Dice'] >= best_overlap) &
                                                    (all_for_thresh['#GT'] > 0)]
                avg = true_positives['Dice'].astype('float32').mean()
                std = true_positives['Dice'].astype('float32').std(ddof=0)
                fold_average.extend([avg, std])
            elif m == 'Dice-P':
                positives = all_for_thresh.loc[(all_for_thresh['#GT'] > 0)]
                avg = positives['Dice'].astype('float32').mean()
                std = positives['Dice'].astype('float32').std()
                fold_average.extend([avg, std])
            elif m == 'Dice-N':
                negatives = all_for_thresh.loc[(all_for_thresh['#GT'] == 0)]
                avg = negatives['Dice'].astype('float32').mean()
                std = negatives['Dice'].astype('float32').std()
                fold_average.extend([avg, std])

            if m == 'Absolute volume error':
                median = all_for_thresh[m].dropna().astype('float32').median()
                fold_average.extend([median])

        metrics_per_fold.append(fold_average)

    metrics_per_fold_df = pd.DataFrame(data=metrics_per_fold, columns=fold_average_columns)
    study_filename = os.path.join(output_folder, 'folds_metrics_average.csv') if suffix == '' else os.path.join(output_folder, 'folds_metrics_average_' + suffix + '.csv')
    metrics_per_fold_df.to_csv(study_filename)
    export_df_to_latex(output_folder, data=metrics_per_fold_df, suffix='folds_metrics_average' + suffix)
    export_df_to_latex_paper(output_folder, data=metrics_per_fold_df, suffix='folds_metrics_average' + suffix)

    ####### Averaging the results from the different folds ###########
    total_samples = metrics_per_fold_df['# samples'].sum()
    fold_averaged_results = [total_samples]
    fixed_metrics = ['Patient-wise recall', 'Patient-wise precision', 'Patient-wise F1', 'FPPP', 'Object-wise recall',
                     'Object-wise precision', 'Object-wise F1', 'Patient-wise recall postop', 'Patient-wise precision postop',
                     'Patient-wise F1 postop', 'Accuracy', 'Balanced accuracy', 'Positive rate', 'Negative rate', 'Specificity']

    if dice_fixed_metric:
        fixed_metrics = fixed_metrics + ['Dice_mean', 'Dice-TP_mean', 'Dice-P_mean', 'Dice-N_mean']

    fold_averaged_results_df_columns = ['Fold']
    # Performing classical averaging first on the relevant metrics
    for fm in fixed_metrics:
        mean_final = 0
        std_final = metrics_per_fold_df[fm].values.std()
        for f in unique_folds:
            fold_val = metrics_per_fold_df.loc[metrics_per_fold_df['Fold'] == f]
            mean_final = mean_final + (fold_val[fm].values[0] * fold_val['# samples'].values[0])
        mean_final = mean_final / total_samples
        fold_averaged_results.extend([mean_final, std_final])
        fold_averaged_results_df_columns.extend([fm + '_mean', fm + '_std'])

    # Performing pooled estimates (taking into account the sample size for each fold) when relevant
    for m in metric_names:
        if m in [x.split('_')[0] for x in metrics_per_fold_df.columns.values]:
            mean_final = 0
            std_final = 0
            for f in unique_folds:
                fold_val = metrics_per_fold_df.loc[metrics_per_fold_df['Fold'] == f]
                mean_final = mean_final + (fold_val[m + '_mean'].values[0] * fold_val['# samples'].values[0])
                std_final = std_final + ((fold_val['# samples'].values[0] - 1) * math.pow(fold_val[m + '_std'].values[0], 2) + (fold_val['# samples'].values[0]) * math.pow(fold_val[m + '_mean'].values[0], 2))
            mean_final = mean_final / total_samples
            std_final = math.sqrt((1 / (total_samples - 1)) * (std_final - (total_samples * math.pow(mean_final, 2))))
            fold_averaged_results.extend([mean_final, std_final])
            fold_averaged_results_df_columns.extend([m + '_mean', m + '_std'])
            if m == 'Absolute volume error':
                # Take the median over all samples
                thresh_index = (np.round(results['Threshold'], 1) == best_threshold)
                all_for_thresh = results.loc[thresh_index]
                median = all_for_thresh[m].dropna().astype('float32').median()
                fold_averaged_results.extend([median])
                fold_averaged_results_df_columns.extend([m + '_median'])

    fold_averaged_results_df = pd.DataFrame(np.asarray(fold_averaged_results).reshape((1, len(fold_averaged_results))),
                                            columns=fold_averaged_results_df_columns)
    output_filename = os.path.join(output_folder, 'overall_metrics_average.csv') if suffix == '' else os.path.join(output_folder, 'overall_metrics_average_' + suffix + '.csv')
    fold_averaged_results_df.to_csv(output_filename, index=False)
    # export_mean_std_df_to_latex(output_folder, data=fold_averaged_results_df, suffix='overall_metrics_average_' + suffix)
    # export_mean_std_df_to_latex_paper(output_folder, data=fold_averaged_results_df, suffix='overall_metrics_average_' + suffix)
