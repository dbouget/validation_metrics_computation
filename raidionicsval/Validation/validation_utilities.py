import csv
import os
import pandas as pd
import numpy as np
import math
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
from ..Utils.resources import SharedResources


def best_segmentation_probability_threshold_analysis(folder, detection_overlap_thresholds=None):
    classes = SharedResources.getInstance().validation_class_names
    class_optimal = {}
    for c in classes:
        class_optimal[c] = {}
        class_optimal[c]['All'] = []
        class_optimal[c]['True Positive'] = []
        optimal_overlap, optimal_threshold = best_segmentation_probability_threshold_analysis_inner(folder,
                                                                                                    detection_overlap_thresholds,
                                                                                                    c, False)
        class_optimal[c]['All'] = [optimal_overlap, optimal_threshold]
        optimal_overlap_tp, optimal_threshold_tp = best_segmentation_probability_threshold_analysis_inner(folder,
                                                                                                          detection_overlap_thresholds,
                                                                                                          c, True)
        class_optimal[c]['True Positive'] = [optimal_overlap_tp, optimal_threshold_tp]

    return class_optimal


def best_segmentation_probability_threshold_analysis_inner(folder, detection_overlap_thresholds, class_name,
                                                           true_positive_state):
    """
    The best threshold probability and object overlap are determined based on a combination of overall DICE
    performance and F1-score. The recall here is not object-wise (i.e., each tumor part not considered individually),
    but at a patient-level based on overall Dice and overlap cut-off.
    :param folder: main validation directory containing the all_dice_scores.csv file.
    :param detection_overlap_thresholds: list of threshold values (float) to use in the range [0., 1.].
    :return: optimal probability threshold and Dice cut-off.
    """
    suffix = "_tp" if true_positive_state else ""
    patient_dices_filename = os.path.join(folder, class_name + '_dice_scores.csv')
    study_filename = os.path.join(folder, class_name + '_optimal_dice_study' + suffix + '.csv')
    study_file = open(study_filename, 'w')
    study_writer = csv.writer(study_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    study_writer.writerow(
        ['Detection threshold', 'Dice threshold', 'Dice', 'PiW Recall', 'PiW Precision', 'PiW F1', 'Found', 'Total'])

    all_dices = pd.read_csv(patient_dices_filename)
    object_detection_dice_thresholds = [0.]
    if detection_overlap_thresholds is not None and type(detection_overlap_thresholds) is list:
        object_detection_dice_thresholds = detection_overlap_thresholds

    thresholds = [np.round(x, 2) for x in list(np.unique(all_dices['Threshold'].values))]
    nb_thresh = len(thresholds)
    recall_precision_results = []

    max_global_metrics_value = 0
    max_recall = 0
    max_mean_dice = 0
    max_threshold = None
    max_overlap = None
    if true_positive_state:
        all_dices = all_dices.loc[all_dices["True Positive"] == True]

    for obj in object_detection_dice_thresholds:
        for thr in range(nb_thresh):
            thr_data = all_dices[thr::nb_thresh]
            thr_data_found = thr_data.loc[np.round(thr_data['PiW Dice'], 3) >= obj]
            mean_dice = thr_data_found['PiW Dice'].values.mean()
            mean_recall = thr_data_found['PiW Recall'].values.mean()
            mean_precision = thr_data_found['PiW Precision'].values.mean()
            mean_f1 = thr_data_found['PiW F1'].values.mean()

            nb_gt = len(thr_data)
            nb_found = len(thr_data_found)
            global_recall = nb_found / nb_gt

            recall_precision_results.append([obj, thresholds[thr], mean_dice, mean_recall, mean_precision, mean_f1, nb_found, nb_gt])
            study_writer.writerow([obj, thresholds[thr], mean_dice, mean_recall, mean_precision, mean_f1, nb_found, nb_gt])

            global_result = (mean_dice + global_recall) / 2
            if global_result > max_global_metrics_value:
                max_global_metrics_value = global_result
                max_threshold = thresholds[thr]
                max_overlap = obj
                max_recall = global_recall
                max_mean_dice = mean_dice

    # # Small trick to comply with further computation. Otherwise, patients with a 0% Dice detection would still be
    # # considered as true positives, which is a behaviour to avoid.
    # if max_overlap == 0.:
    #     max_overlap = 0.01
    study_writer.writerow(['', '', '', '', '', '', '', ''])
    study_writer.writerow([max_overlap, max_threshold, '', '', '', '', '', ''])
    study_file.close()
    logging.info('Class \'{}\' - Selected values (Overlap: {}, Threshold: {}) for global metric of {:.3f} (Dice: {:.2f}% and Recall: {:.2f}%).'
          ' True positive case: {}'.format(class_name, max_overlap, max_threshold, max_global_metrics_value,
                                           max_mean_dice * 100. , max_recall * 100., true_positive_state))

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
        F1s = recall_precision_results[o * nb_thresh:(o + 1) * nb_thresh, 5]
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

    os.makedirs(os.path.join(folder, 'OptimalSearch'), exist_ok=True)
    fig.savefig(os.path.join(folder, 'OptimalSearch', 'dice_over_threshold.png'), dpi=300,
                bbox_inches="tight")
    fig2.savefig(os.path.join(folder, 'OptimalSearch', 'F1_over_threshold.png'), dpi=300,
                 bbox_inches="tight")
    fig3.savefig(os.path.join(folder, 'OptimalSearch', 'F1_over_dice.png'), dpi=300,
                 bbox_inches="tight")
    fig4.savefig(os.path.join(folder, 'OptimalSearch', 'metrics_scatter_over_threshold.png'), dpi=300,
                 bbox_inches="tight")

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    return max_overlap, max_threshold


def compute_fold_average(folder, data=None, class_optimal={}, metrics=[], suffix='', condition='All'):
    """
    @TODO. Issue if running without external when they exist in the file (ValueError: 80 columns passed, passed data had 36 columns)
    """
    classes = SharedResources.getInstance().validation_class_names
    optimal_tag = 'All' if condition=='All' else 'True Positive'
    for c in classes:
        optimal_values = class_optimal[c][optimal_tag]
        compute_fold_average_inner(folder, data=data, class_name=c, best_threshold=optimal_values[1],
                                   best_overlap=optimal_values[0], metrics=metrics, suffix=suffix,condition=condition)


def compute_fold_average_inner(folder, class_name, data=None, best_threshold=0.5, best_overlap=0.0, metrics=[],
                               suffix='', condition='All'):
    """
    @TODO. Should add the #Found in addition to #samples to know how many patients/images are positively detected.

    :param folder: Main study folder where the results will be dumped (assuming inside a Validation sub-folder)
    :param best_threshold:
    :param best_overlap:
    :param metric_names:
    :return:
    """
    results = None
    if data is None:
        results_filename = os.path.join(folder, class_name + '_dice_scores.csv')
        tmp = pd.read_csv(results_filename)
        use_cols = list(tmp.columns)[0:SharedResources.getInstance().upper_default_metrics_index] + metrics
        results = pd.read_csv(results_filename, usecols=use_cols)
    else:
        results = deepcopy(data)

    suffix = "Positive" + suffix if condition=='Positive' else suffix
    suffix = "TP" + suffix if condition=='TP' else suffix
    results.replace('inf', np.nan, inplace=True)
    results.replace(float('inf'), np.nan, inplace=True)
    results.replace('', np.nan, inplace=True)
    results.replace(' ', np.nan, inplace=True)
    unique_folds = np.unique(results['Fold'])
    nb_folds = len(unique_folds)
    metrics_per_fold = []
    volume_threshold = 0.
    if len(SharedResources.getInstance().validation_true_positive_volume_thresholds) == 1:
        volume_threshold = SharedResources.getInstance().validation_true_positive_volume_thresholds[0]
    else:
        index_cl = SharedResources.getInstance().validation_class_names.index(class_name)
        volume_threshold = SharedResources.getInstance().validation_true_positive_volume_thresholds[index_cl]

    metric_names = list(results.columns[3:])
    # if metrics is not None:
    #     metric_names.extend(metrics)

    fold_average_columns = ['Fold', '# samples', 'Patient-wise recall', 'Patient-wise precision',
                            'Patient-wise specificity', 'Patient-wise F1', 'Patient-wise Accuracy',
                            'Patient-wise Balanced accuracy']
    for m in metric_names:
        fold_average_columns.extend([m + ' (Mean)', m + ' (Std)'])

    # Regarding the overlap threshold, should the patient discarded for recall be
    # used for other metrics computation?
    for f in unique_folds:
        patientwise_metrics = compute_patientwise_fold_metrics(results, f, best_threshold, best_overlap,
                                                               volume_threshold, condition)
        fold_average_metrics, fold_std_metrics = compute_singe_fold_average_metrics(results, f, best_threshold,
                                                                                    best_overlap, metrics, volume_threshold, condition)
        fold_metrics = []
        for m in range(len(fold_average_metrics)):
            fold_metrics.append(fold_average_metrics[m])
            fold_metrics.append(fold_std_metrics[m])
        fold_average = [f, len(np.unique(results.loc[results['Fold'] == f]['Patient'].values))] + patientwise_metrics + fold_metrics
        metrics_per_fold.append(fold_average)

    metrics_per_fold_df = pd.DataFrame(data=metrics_per_fold, columns=fold_average_columns)
    study_filename = os.path.join(folder, class_name + '_folds_metrics_average.csv') if suffix == '' else os.path.join(folder,
                                                                                                 class_name + '_folds_metrics_average_' + suffix + '.csv')
    metrics_per_fold_df.to_csv(study_filename, index=False)

    ####### Averaging the results from the different folds ###########
    total_samples = metrics_per_fold_df['# samples'].sum()
    patientwise_fold_metrics_to_average = metrics_per_fold_df.values[:, 2:8]
    fold_metrics_to_average = metrics_per_fold_df.values[:, 8:][:, 0::2]
    fold_std_metrics_to_average = metrics_per_fold_df.values[:, 8:][:, 1::2]
    total_fold_metrics_to_average = np.concatenate((patientwise_fold_metrics_to_average, fold_metrics_to_average), axis=1)
    fold_metrics_average = np.mean(total_fold_metrics_to_average, axis=0)
    fold_metrics_std = np.std(total_fold_metrics_to_average, axis=0)
    fold_averaged_results = [total_samples]
    for m in range(len(fold_metrics_average)):
        fold_averaged_results.append(fold_metrics_average[m])
        fold_averaged_results.append(fold_metrics_std[m])

    # Performing pooled estimates (taking into account the sample size for each fold) when relevant
    pooled_fold_averaged_results = [len(unique_folds), total_samples]
    pw_index = 6  # Length of the initial fold_average_columns, without the first two elements.
    for m in range(total_fold_metrics_to_average.shape[1]):
        mean_final = 0
        std_final = 0
        for f in range(len(unique_folds)):
            fold_val = total_fold_metrics_to_average[f, m]
            fold_sample_size = list(metrics_per_fold_df['# samples'])[f]
            mean_final = mean_final + (fold_val * fold_sample_size)
            # For patient-wise metrics, there is no std value for within each fold
            if m < pw_index:
                std_final = std_final
            else:
                std_final = std_final + ((fold_sample_size - 1) * math.pow(fold_std_metrics_to_average[f, m-pw_index], 2) + (fold_sample_size) * math.pow(fold_val, 2))
        mean_final = mean_final / total_samples
        if m >= pw_index:
            std_final = math.sqrt((1 / (total_samples - 1)) * (std_final - (total_samples * math.pow(mean_final, 2))))
        else:
            std_final = np.std(total_fold_metrics_to_average[:, m])
        pooled_fold_averaged_results.extend([mean_final, std_final])

    overall_average_columns = ['Fold', '# samples']
    for m in ['Patient-wise recall', 'Patient-wise precision', 'Patient-wise specificity', 'Patient-wise F1',
              'Patient-wise Accuracy', 'Patient-wise Balanced accuracy']:
        overall_average_columns.extend([m + ' (Mean)', m + ' (Std)'])

    for m in metric_names:
        overall_average_columns.extend([m + ' (Mean)', m + ' (Std)'])
    pooled_fold_averaged_results_df = pd.DataFrame(data=np.asarray(pooled_fold_averaged_results).reshape(1, len(overall_average_columns)), columns=overall_average_columns)
    study_filename = os.path.join(folder, class_name + '_overall_metrics_average.csv') if suffix == '' else os.path.join(folder,
                                                                                                 class_name + '_overall_metrics_average_' + suffix + '.csv')
    pooled_fold_averaged_results_df.to_csv(study_filename, index=False)


def compute_singe_fold_average_metrics(results, fold_number, best_threshold, best_overlap, metric_names, volume_threshold, condition):
    fold_results = results.loc[results['Fold'] == fold_number]
    thresh_index = (np.round(fold_results['Threshold'], 1) == best_threshold)
    all_for_thresh = fold_results.loc[thresh_index]
    if len(all_for_thresh) == 0:
        # Empty fold? Can indicate something went wrong, or was not computed properly beforehand
        return None
    if condition == 'Positive':
        all_for_thresh = all_for_thresh.loc[all_for_thresh['GT volume (ml)'] > volume_threshold]
    elif condition == 'TP':
        all_for_thresh = all_for_thresh.loc[(all_for_thresh['GT volume (ml)'] > volume_threshold) & (all_for_thresh['PiW Dice'] > best_overlap)]
    upper_default_metrics_index = SharedResources.getInstance().upper_default_metrics_index
    if len(all_for_thresh) == 0:
        # Returning a list to avoid crashing, even if the selection is empty.
        return [0.] * (upper_default_metrics_index - 3) + [0.] * len(metric_names) , [0.] * (upper_default_metrics_index - 3) + [0.] * len(metric_names)
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


def compute_patientwise_fold_metrics(results, fold_number, best_threshold, best_overlap, det_vol_thr, condition):
    fold_results = results.loc[results['Fold'] == fold_number]
    thresh_index = (np.round(fold_results['Threshold'], 1) == best_threshold)
    all_for_thresh = fold_results.loc[thresh_index]
    if len(all_for_thresh) == 0:
        # Empty fold? Can indicate something went wrong, or was not computed properly beforehand
        fold_average = [-1., -1., -1., -1., -1., -1.]
        return fold_average

    if condition == 'All':
        true_positives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] > 0) & (fold_results['GT volume (ml)'] > 0)]
        false_positives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] > 0) & (fold_results['GT volume (ml)'] <= 0)]
        true_negatives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] <= 0) & (fold_results['GT volume (ml)'] <= 0)]
        false_negatives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] <= 0) & (fold_results['GT volume (ml)'] > 0)]

    elif condition == 'Positive':
        true_positives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] > det_vol_thr) & (fold_results['GT volume (ml)'] > det_vol_thr)]
        false_positives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] > det_vol_thr) & (fold_results['GT volume (ml)'] <= det_vol_thr)]
        true_negatives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] <= det_vol_thr) & (fold_results['GT volume (ml)'] <= det_vol_thr)]
        false_negatives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] <= det_vol_thr) & (fold_results['GT volume (ml)'] > det_vol_thr)]

    elif condition == 'TP':
        true_positives = fold_results.loc[thresh_index & (fold_results['PiW Dice'] > best_overlap) & (fold_results['GT volume (ml)'] > det_vol_thr)]
        false_positives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] > det_vol_thr) & (fold_results['GT volume (ml)'] <= det_vol_thr)]
        true_negatives = fold_results.loc[thresh_index & (fold_results['Detection volume (ml)'] <= det_vol_thr) & (fold_results['GT volume (ml)'] <= det_vol_thr)]
        false_negatives = fold_results.loc[thresh_index & (fold_results['PiW Dice'] <= best_overlap) & (fold_results['GT volume (ml)'] > det_vol_thr)]

    patient_wise_recall = len(true_positives) / (len(true_positives) + len(false_negatives) + 1e-6)
    patient_wise_precision = len(true_positives) / (len(true_positives) + len(false_positives) + 1e-6)
    patient_wise_specificity = 1. if len(true_negatives) + len(false_positives) == 0 else len(true_negatives) / (len(true_negatives) + len(false_positives) + 1e-6)
    patient_wise_f1 = 2 * len(true_positives) / ((2 * len(true_positives)) + len(false_positives) + len(false_negatives) + 1e-6)
    accuracy = (len(true_positives) + len(true_negatives)) / (len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives))
    balanced_accuracy = (patient_wise_recall + patient_wise_specificity) / 2

    # fold_number, len(np.unique(fold_results['Patient'].values))
    patientwise_metrics = [patient_wise_recall, patient_wise_precision, patient_wise_specificity, patient_wise_f1,
                           accuracy, balanced_accuracy]
    return patientwise_metrics

