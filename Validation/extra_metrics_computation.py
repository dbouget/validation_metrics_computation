import math
import os
import traceback
import itertools
import multiprocessing
from copy import deepcopy
import pandas as pd
import nibabel as nib
import numpy as np
from typing import List
from tqdm import tqdm
from medpy.metric.binary import hd95, volume_correlation, assd, ravd, obj_assd
from sklearn.metrics import mutual_info_score, jaccard_score, normalized_mutual_info_score, adjusted_rand_score,\
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics.cluster import rand_score
from Utils.io_converters import get_fold_from_file
from Utils.resources import SharedResources


def compute_patient_extra_metrics(patient_object, class_index, optimal_threshold, metrics: List[str] = []):
    extra_metrics_results = []
    try:
        metric_values = [x[1] for x in patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:]]
        if None not in metric_values:
            return patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:]

        ground_truth_ni = nib.load(patient_object._ground_truth_filepaths[class_index])
        detection_ni = nib.load(patient_object._prediction_filepaths[class_index])

        gt = ground_truth_ni.get_data()
        gt[gt >= 1] = 1
        gt = gt.astype('uint8')

        detection = detection_ni.get_data()[:]
        detection[detection < optimal_threshold] = 0
        detection[detection >= optimal_threshold] = 1
        detection = detection.astype('uint8')

        tp_array = np.zeros(detection.shape)
        fp_array = np.zeros(detection.shape)
        tn_array = np.zeros(detection.shape)
        fn_array = np.zeros(detection.shape)

        tp_array[(gt == 1) & (detection == 1)] = 1
        fp_array[(gt == 0) & (detection == 1)] = 1
        tn_array[(gt == 0) & (detection == 0)] = 1
        fn_array[(gt == 1) & (detection == 0)] = 1

        # N-B: Sometimes unstable: it will hang forever if the image is too large it seems...
        # If so, just use 1 process
        # @TODO. Have to investigate how to fix or bypass the issue, should we resample to [1,1,1] to compute the metrics
        if SharedResources.getInstance().number_processes > 1:
            try:
                pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                extra_metrics_results = pool.map(parallel_metric_computation, zip(metrics, metric_values,
                                                                                  itertools.repeat(gt),
                                                                                  itertools.repeat(detection),
                                                                                  itertools.repeat(detection_ni.header),
                                                                                  itertools.repeat(ground_truth_ni.header),
                                                                                  itertools.repeat(tp_array),
                                                                                  itertools.repeat(tn_array),
                                                                                  itertools.repeat(fp_array),
                                                                                  itertools.repeat(fn_array)))
                pool.close()
                pool.join()
            except Exception as e:
                print("Issue computing metrics for patient {} in the multiprocessing loop.".format(patient_object.unique_id))
                print(traceback.format_exc())
        else:
            for metric in metrics:
                try:
                    metric_value = compute_specific_metric_value(metric=metric, gt=gt, detection=detection,
                                                                 tp=np.sum(tp_array), tn=np.sum(tn_array),
                                                                 fp=np.sum(fp_array), fn=np.sum(fn_array),
                                                                 gt_ni_header=ground_truth_ni.header,
                                                                 det_ni_header=detection_ni.header)
                    extra_metrics_results.extend([metric, metric_value])
                except Exception as e:
                    print('Issue computing metric {} for patient {}'.format(metric, patient_object.unique_id))
                    print(traceback.format_exc())
    except Exception as e:
        print('Global issue computing metrics for patient {}'.format(patient_object.unique_id))
        print(traceback.format_exc())

    return extra_metrics_results


def parallel_metric_computation(args):
    """
    Metrics computation method linked to the multiprocessing strategy. Effectively where the call to compute is made.
    :param args: list of arguments split from the lists given to the multiprocessing.Pool call.
    :return: list with metric name and computed metric value.
    """
    metric = args[0]
    metric_value = args[1][1]
    gt = args[2]
    detection = args[3]
    det_header = args[4]
    gt_header = args[5]
    tp_array = args[6]
    tn_array = args[7]
    fp_array = args[8]
    fn_array = args[9]

    if metric_value == metric_value and metric_value is not None:
        return [metric, metric_value]

    try:
        metric_value = compute_specific_metric_value(metric=metric, gt=gt, detection=detection,
                                                     tp=np.sum(tp_array), tn=np.sum(tn_array),
                                                     fp=np.sum(fp_array), fn=np.sum(fn_array),
                                                     gt_ni_header=gt_header,
                                                     det_ni_header=det_header)
    except Exception as e:
        print('Computing {} gave an exception'.format(metric))
        pass

    return [metric, metric_value]


def compute_specific_metric_value(metric, gt, detection, tp, tn, fp, fn, gt_ni_header, det_ni_header):
    metric_value = None
    if metric == 'VS':
        metric_value = math.inf
        den = (2 * tp) + fp + fn
        if den != 0:
            metric_value = 1 - ((abs(fn - fp)) / ((2 * tp) + fp + fn))
    elif metric == 'GCE':
        if (tp + fn) != 0 and (tn + fp) != 0 and (tp + fp) != 0 and (tn + fn) != 0:
            param11 = (fn * (fn + (2 * tp))) / (tp + fn)
            param12 = (fp * (fp + (2 * tn))) / (tn + fp)
            param21 = (fp * (fp + (2 * tp))) / (tp + fp)
            param22 = (fn * (fn + (2 * tn))) / (tn + fn)
            metric_value = (1 / np.prod(gt_ni_header.get_data_shape()[0:3])) * min(param11 + param12, param21 + param22)
        else:
            metric_value = math.inf
    elif metric == 'MI':
        metric_value = normalized_mutual_info_score(gt.flatten(), detection.flatten())
    elif metric == 'RI':
        metric_value = 0.
        a = 0.5 * ((tp * (tp - 1)) + (fp * (fp - 1)) + (tn * (tn - 1)) + (fn * (fn - 1)))
        b = 0.5 * ((math.pow(tp + fn, 2) + math.pow(tn + fp, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        c = 0.5 * ((math.pow(tp + fp, 2) + math.pow(tn + fn, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        d = np.prod(gt_ni_header.get_data_shape()[0:3]) * (np.prod(gt_ni_header.get_data_shape()[0:3]) - 1) / 2 - (a + b + c)
        num = a + b
        den = a + b + c + d
        if den != 0:
            metric_value = num / den
    elif metric == 'ARI':
        metric_value = 0.
        a = 0.5 * ((tp * (tp - 1)) + (fp * (fp - 1)) + (tn * (tn - 1)) + (fn * (fn - 1)))
        b = 0.5 * ((math.pow(tp + fn, 2) + math.pow(tn + fp, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        c = 0.5 * ((math.pow(tp + fp, 2) + math.pow(tn + fn, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        d = np.prod(gt_ni_header.get_data_shape()[0:3]) * (np.prod(gt_ni_header.get_data_shape()[0:3]) - 1) / 2 - (a + b + c)
        num = 2 * (a * d - b * c)
        den = math.pow(c, 2) + math.pow(b, 2) + 2 * a * d + (a + d) * (c + b)
        if den != 0:
            metric_value = num / den
    elif metric == 'VOI':
        fn_tp = fn + tp
        fp_tp = fp + tp
        total = np.prod(gt_ni_header.get_data_shape()[0:3])

        if fn_tp == 0 or (fn_tp / total) == 1 or fp_tp == 0 or (fp_tp / total) == 1:
            metric_value = math.inf
        else:
            h1 = -((fn_tp / total) * math.log2(fn_tp / total) + (1 - fn_tp / total) * math.log2(1 - fn_tp / total))
            h2 = -((fp_tp / total) * math.log2(fp_tp / total) + (1 - fp_tp / total) * math.log2(1 - fp_tp / total))

            p00 = 1 if tn == 0 else (tn / total)
            p01 = 1 if fn == 0 else (fn / total)
            p10 = 1 if fp == 0 else (fp / total)
            p11 = 1 if tp == 0 else (tp / total)

            h12 = -((tn / total) * math.log2(p00) + (fn / total) * math.log2(p01) + (fp / total) * math.log2(p10) + (tp / total) * math.log2(p11))
            mi = h1 + h2 - h12
            metric_value = h1 + h2 - (2 * mi)
    elif metric == 'Jaccard':
        metric_value = jaccard_score(gt.flatten(), detection.flatten())
    elif metric == 'IOU':
        metric_value = math.inf
        intersection = (gt == 1) & (detection == 1)
        union = (gt == 1) | (detection == 1)
        if np.count_nonzero(union) != 0:
            metric_value = np.count_nonzero(intersection) / np.count_nonzero(union)
    elif metric == 'TPR':
        metric_value = math.inf
        if (tp + fn) != 0:
            metric_value = tp / (tp + fn)
    elif metric == 'TNR':
        metric_value = math.inf
        if (tn + fp) != 0:
            metric_value = tn / (tn + fp)
    elif metric == 'FPR':
        metric_value = math.inf
        if (fp + tn) != 0:
            metric_value = fp / (fp + tn)
    elif metric == 'FNR':
        metric_value = math.inf
        if (fn + tp) != 0:
            metric_value = fn / (fn + tp)
    elif metric == 'PPV':
        metric_value = math.inf
        if (tp + fp) != 0:
            metric_value = tp / (tp + fp)
    elif metric == 'AUC':
        metric_value = roc_auc_score(gt.flatten(), detection.flatten())
    elif metric == 'MCC':
        metric_value = math.inf
        num = (tp * tn) - (fp * fn)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if den != 0:
            metric_value = num / den
        # metric_value = matthews_corrcoef(gt.flatten(), detection.flatten())
    elif metric == 'CKS':
        metric_value = cohen_kappa_score(gt.flatten(), detection.flatten())
    elif metric == 'HD95':
        metric_value = math.inf
        if np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = hd95(detection, gt, voxelspacing=det_ni_header.get_zooms(), connectivity=1)
    elif metric == 'ASSD':
        metric_value = math.inf
        if np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = assd(detection, gt, voxelspacing=det_ni_header.get_zooms(), connectivity=1)
    elif metric == 'OASSD':
        metric_value = math.inf
        if np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = obj_assd(detection, gt, voxelspacing=det_ni_header.get_zooms(), connectivity=1)
    elif metric == 'RAVD':
        metric_value = math.inf
        if np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = ravd(detection, gt)
    elif metric == 'VC':
        metric_value = math.inf
        if np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value, pval = volume_correlation(detection, gt)
    elif metric == 'MahaD':
        metric_value = math.inf
        gt_n = np.count_nonzero(detection)
        seg_n = np.count_nonzero(gt)

        if gt_n != 0 and seg_n != 0:
            gt_indices = np.flip(np.where(gt == 1), axis=0)
            gt_mean = gt_indices.mean(axis=1)
            gt_cov = np.cov(gt_indices)

            seg_indices = np.flip(np.where(detection == 1), axis=0)
            seg_mean = seg_indices.mean(axis=1)
            seg_cov = np.cov(seg_indices)

            # calculate common covariance matrix
            common_cov = (gt_n * gt_cov + seg_n * seg_cov) / (gt_n + seg_n)
            common_cov_inv = np.linalg.inv(common_cov)

            mean = gt_mean - seg_mean
            metric_value = math.sqrt(mean.dot(common_cov_inv).dot(mean.T))
    elif metric == 'ProbD':
        # metric_value = -1.
        metric_value = math.inf
        gt_flat = gt.flatten().astype(np.int8)
        det_flat = detection.flatten().astype(np.int8)

        probability_difference = np.absolute(gt_flat - det_flat).sum()
        probability_joint = (gt_flat * det_flat).sum()

        if probability_joint != 0:
            metric_value = probability_difference / (2. * probability_joint)

    return metric_value


def compute_overall_metrics_correlation(folder, data=None, best_threshold=0.5, best_overlap=0.0):
    """

    :param folder:
    :param metric_names:
    :param data:
    :param best_threshold:
    :param best_overlap:
    :return:
    """
    results = None
    if data is None:
        results_filename = os.path.join(folder, 'Validation', 'all_dice_scores.csv')
        results = pd.read_csv(results_filename)
    else:
        results = deepcopy(data)

    results.replace('inf', np.nan, inplace=True)
    optimal_results = results.loc[results['Threshold'] == best_threshold]
    # results_for_matrix_df = optimal_results.drop(['Patient', 'Fold', 'Threshold', '#Det', 'Inst DICE', 'Inst Recall',
    #                                               'Inst Precision', 'Largest foci Dice'], axis=1)
    # results_for_matrix_df = results_for_matrix_df[['Dice', 'IOU', 'Jaccard', 'AUC', 'TPR', 'TNR', 'FPR', 'FNR', 'PPV',
    #                                                'VS', 'VC', 'RAVD', 'GCE', 'MI', 'MCC', 'CKS', 'VOI', 'HD95',
    #                                                'MahaD', 'ProbD','ASSD', 'ARI', 'OASSD', '#GT']]
    results_for_matrix_df = optimal_results.drop(['Patient', 'Fold', 'Threshold', '#Det', 'Inst DICE', 'Inst Recall',
                                                   'Inst Precision', 'Largest foci Dice', '#GT', 'FPR', 'FNR'], axis=1)
    results_for_matrix_df = results_for_matrix_df[['Dice', 'TPR', 'TNR', 'PPV', 'IOU', 'GCE',
                                                   'VS', 'RAVD', 'MI', 'VOI', 'CKS', 'AUC', 'VC', 'MCC', 'ProbD',
                                                   'HD95', 'MahaD', 'ASSD', 'ARI', 'OASSD']]
    results_for_matrix_df = results_for_matrix_df.dropna()
    results_for_matrix_df = results_for_matrix_df.apply(pd.to_numeric)
    corr_matrix = results_for_matrix_df.corr()
    # print(corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2).render())
    export_correlation_matrix_to_latex(folder, corr_matrix)


def export_correlation_matrix_to_latex(folder, matrix):
    matrix_filename = os.path.join(folder, 'Validation', 'correlation_matrix.txt')
    pfile = open(matrix_filename, 'w')

    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\caption{Metrics confusion matrix. The color intensity of each cell represents the strength of'
                ' the correlation, where blue denotes direct correlation and red denotes inverse correlation.}\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write('\\begin{tabular}{l'+('r' * matrix.shape[0]) + '}\n')
    pfile.write('\\toprule\n')
    header_line = '{}'
    for elem in matrix.axes[0].values:
        header_line = header_line + ' & ' + elem
    pfile.write(header_line + '\\tabularnewline\n')
    for r in range(matrix.shape[0]):
        line = matrix.axes[1].values[r]
        for c in range(matrix.shape[1]):
            if matrix.values[r, c] == matrix.values[r, c]:
                num_value = float(matrix.values[r, c])
                line = line + ' & ' + latex_colorcode_from_values(num_value) + str(np.round(num_value, 2))
            else:
                line = line + ' & NaN'
        pfile.write(line + '\\tabularnewline\n')
    pfile.write('\\bottomrule\n')
    pfile.write('\\end{tabular}\n')
    pfile.write('}\n')
    pfile.write('\\end{table}')
    pfile.close()


def latex_colorcode_from_values(value):
    color_code = ''
    color_val = int(abs(value * 100))
    if value < 0:
        color_code = '\cellcolor{red!' + str(color_val) + '}'
    elif value >= 0:
        color_code = '\cellcolor{blue!' + str(color_val) + '}'

    return color_code
