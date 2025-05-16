import logging
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
# from medpy.metric.binary import hd95, volume_correlation, assd, ravd, obj_assd
from sklearn.metrics import jaccard_score, normalized_mutual_info_score, roc_auc_score, cohen_kappa_score
from ..Utils.resources import SharedResources
from ..Utils.io_converters import open_image_file, save_image_file
from ..Computation.medpy_metrics import (compute_hd95, compute_assd, compute_ravd, compute_volume_correlation,
                                         compute_object_assd)
from ..Validation.instance_segmentation_validation import *


def compute_patient_extra_metrics(patient_object, class_index, optimal_threshold, metrics: List[str] = []):
    extra_metrics_results = []
    try:
        if (patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold) is not None and
                patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:] is not None):
            metric_values = [x[1] for x in patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:]]
            if 'objectwise' in SharedResources.getInstance().validation_metric_spaces and 'patientwise' in SharedResources.getInstance().validation_metric_spaces:
                if False not in [x == x for x in metric_values]:
                    # If all metric values have been computed, i.e., no nan or None etc...
                    return patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:]
            elif 'patientwise' not in SharedResources.getInstance().validation_metric_spaces:
                if False not in [x == x for x in metric_values[1::2]]:
                    # If all metric values have been computed, i.e., no nan or None etc...
                    return patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:]
            elif 'objectwise' not in SharedResources.getInstance().validation_metric_spaces:
                if False not in [x == x for x in metric_values[0::2]]:
                    # If all metric values have been computed, i.e., no nan or None etc...
                    return patient_object.get_optimal_class_extra_metrics(class_index, optimal_threshold)[1:]
        else:
            metric_values = [None] * len(metrics)

        # ground_truth_ni = nib.load(patient_object._ground_truth_filepaths[class_index])
        # detection_ni = nib.load(patient_object._prediction_filepaths[class_index])
        # gt = ground_truth_ni.get_fdata()
        # detection = detection_ni.get_fdata()[:]
        gt, _, gt_input_specs = open_image_file(patient_object.ground_truth_filepaths[class_index])
        detection, _, det_input_specs = open_image_file(patient_object.prediction_filepaths[class_index])
        gt[gt >= 1] = 1
        gt = gt.astype('uint8')

        detection[detection < optimal_threshold] = 0
        detection[detection >= optimal_threshold] = 1
        detection = detection.astype('uint8')

        # # Cleaning the too small objects that might be noise in the detection
        # if np.count_nonzero(detection) > 0:
        #     detection_labels = measurements.label(detection)[0]
        #     # print('Found {} objects.'.format(np.max(self.detection_labels)))
        #     refined_image = deepcopy(detection)
        #     for c in range(1, np.max(detection_labels) + 1):
        #         if np.count_nonzero(detection_labels == c) < SharedResources.getInstance().validation_tiny_objects_removal_threshold:
        #             refined_image[refined_image == c] = 0
        #     refined_image[refined_image != 0] = 1
        #     detection = refined_image

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
                                                                                  itertools.repeat(det_input_specs[1]),
                                                                                  itertools.repeat(gt_input_specs[1]),
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
                                                                 gt_spacing=gt_input_specs[1],
                                                                 det_spacing=det_input_specs[1])
                    extra_metrics_results.append([metric, metric_value])
                except Exception as e:
                    print('Issue computing metric {} for patient {}'.format(metric, patient_object.unique_id))
                    print(traceback.format_exc())
        extra_metrics_results = [[f'PiW {x[0]}', x[1]] for x in extra_metrics_results]

        if "objectwise" in SharedResources.getInstance().validation_metric_spaces:
            obj_val = InstanceSegmentationValidation(gt_image=gt, detection_image=detection,
                                                     tiny_objects_removal_threshold=SharedResources.getInstance().validation_tiny_objects_removal_threshold)

            obj_val.spacing = gt_input_specs[1]
            obj_val.run()
            # Computing all metrics in an object-wise fashion
            all_instance_results = []
            for g, go in enumerate(obj_val.gt_candidates):
                gt_label = g + 1
                if gt_label in np.asarray(obj_val.matching_results)[:, 0]:
                    indices = np.where(np.asarray(obj_val.matching_results)[:, 0] == gt_label)[0]
                    if len(indices) > 1:
                        # Should not happen anymore
                        print(f"Warning - Entering a use-case which should not be possible!")
                        pass
                    det_label = np.asarray(obj_val.matching_results)[indices[0]][1]
                    # det_label = obj_val.matching_results[list(np.asarray(obj_val.matching_results)[:, 0]).index(gt_label)][1]
                    instance_gt_array = np.zeros(gt.shape, dtype="uint8")
                    instance_det_array = np.zeros(detection.shape, dtype="uint8")
                    instance_gt_array[obj_val.gt_labels == gt_label] = 1
                    instance_det_array[obj_val.detection_labels == det_label] = 1

                    tp_array = np.zeros(instance_gt_array.shape)
                    fp_array = np.zeros(instance_gt_array.shape)
                    tn_array = np.zeros(instance_gt_array.shape)
                    fn_array = np.zeros(instance_gt_array.shape)

                    tp_array[(instance_gt_array == 1) & (instance_det_array == 1)] = 1
                    fp_array[(instance_gt_array == 0) & (instance_det_array == 1)] = 1
                    tn_array[(instance_gt_array == 0) & (instance_det_array == 0)] = 1
                    fn_array[(instance_gt_array == 1) & (instance_det_array == 0)] = 1
                    instance_results = []
                    if SharedResources.getInstance().number_processes > 1:
                        try:
                            pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                            instance_results = pool.map(parallel_metric_computation, zip(metrics, metric_values,
                                                                                              itertools.repeat(instance_gt_array),
                                                                                              itertools.repeat(instance_det_array),
                                                                                              itertools.repeat(
                                                                                                  det_input_specs[1]),
                                                                                              itertools.repeat(
                                                                                                  gt_input_specs[1]),
                                                                                              itertools.repeat(tp_array),
                                                                                              itertools.repeat(tn_array),
                                                                                              itertools.repeat(fp_array),
                                                                                              itertools.repeat(fn_array)))
                            pool.close()
                            pool.join()
                        except Exception as e:
                            print("Issue computing metrics for patient {} in the multiprocessing loop.".format(
                                patient_object.unique_id))
                            print(traceback.format_exc())
                    else:
                        for metric in metrics:
                            try:
                                metric_value = compute_specific_metric_value(metric=metric, gt=instance_gt_array, detection=instance_det_array,
                                                                             tp=np.sum(tp_array), tn=np.sum(tn_array),
                                                                             fp=np.sum(fp_array), fn=np.sum(fn_array),
                                                                             gt_spacing=gt_input_specs[1],
                                                                             det_spacing=det_input_specs[1])
                                instance_results.append([metric, metric_value])
                            except Exception as e:
                                print('Issue computing metric {} for patient {}'.format(metric, patient_object.unique_id))
                                print(traceback.format_exc())
                    all_instance_results.append(instance_results)

            if len(all_instance_results) != 0:
                for m in metrics:
                    all_values = []
                    tp_values = []
                    for i in range(len(all_instance_results)):
                        for j in range(len(all_instance_results[i])):
                            if all_instance_results[i][j][0] == m:
                                if all_instance_results[i][j][1] != math.inf:
                                    tp_values.append(all_instance_results[i][j][1])
                                else:
                                    all_values.append(all_instance_results[i][j][1])
                                break
                    all_mean = np.mean(all_values)
                    all_std = np.std(all_values)
                    tp_mean = np.mean(tp_values)
                    tp_std = np.std(tp_values)
                    extra_metrics_results.append([f'OW {m}', tp_mean])
            else:
                for m in metrics:
                    extra_metrics_results.append([f'OW {m}', -999.])
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
    metric_value = args[1] #args[1][1]
    gt = args[2]
    detection = args[3]
    det_extra = args[4]
    gt_extra = args[5]
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
                                                     gt_spacing=gt_extra[1],
                                                     det_spacing=det_extra[1])
    except Exception as e:
        print('Computing {} gave an exception'.format(metric))
        pass

    return [metric, metric_value]


def compute_specific_metric_value(metric, gt, detection, tp, tn, fp, fn, gt_spacing, det_spacing):
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
            # metric_value = (1 / np.prod(gt_ni_header.get_data_shape()[0:3])) * min(param11 + param12, param21 + param22)
            metric_value = (1 / np.prod(gt.shape)) * min(param11 + param12, param21 + param22)
        else:
            metric_value = math.inf
    elif metric == 'MI':
        metric_value = normalized_mutual_info_score(gt.flatten(), detection.flatten())
    elif metric == 'RI':
        metric_value = 0.
        a = 0.5 * ((tp * (tp - 1)) + (fp * (fp - 1)) + (tn * (tn - 1)) + (fn * (fn - 1)))
        b = 0.5 * ((math.pow(tp + fn, 2) + math.pow(tn + fp, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        c = 0.5 * ((math.pow(tp + fp, 2) + math.pow(tn + fn, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        # d = np.prod(gt_ni_header.get_data_shape()[0:3]) * (np.prod(gt_ni_header.get_data_shape()[0:3]) - 1) / 2 - (a + b + c)
        d = np.prod(gt.shape) * (np.prod(gt.shape) - 1) / 2 - (a + b + c)
        num = a + b
        den = a + b + c + d
        if den != 0:
            metric_value = num / den
    elif metric == 'ARI':
        metric_value = 0.
        a = 0.5 * ((tp * (tp - 1)) + (fp * (fp - 1)) + (tn * (tn - 1)) + (fn * (fn - 1)))
        b = 0.5 * ((math.pow(tp + fn, 2) + math.pow(tn + fp, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        c = 0.5 * ((math.pow(tp + fp, 2) + math.pow(tn + fn, 2)) - (math.pow(tp, 2) + math.pow(tn, 2) + math.pow(fp, 2) + math.pow(fn, 2)))
        # d = np.prod(gt_ni_header.get_data_shape()[0:3]) * (np.prod(gt_ni_header.get_data_shape()[0:3]) - 1) / 2 - (a + b + c)
        d = np.prod(gt.shape) * (np.prod(gt.shape) - 1) / 2 - (a + b + c)
        num = 2 * (a * d - b * c)
        den = math.pow(c, 2) + math.pow(b, 2) + 2 * a * d + (a + d) * (c + b)
        if den != 0:
            metric_value = num / den
    elif metric == 'VOI':
        fn_tp = fn + tp
        fp_tp = fp + tp
        # total = np.prod(gt_ni_header.get_data_shape()[0:3])
        total = np.prod(gt.shape)

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
    elif metric == 'CKS':
        metric_value = cohen_kappa_score(gt.flatten(), detection.flatten())
    elif metric == 'HD95':
        metric_value = math.inf
        if np.max(gt) == 1 and np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = compute_hd95(detection, gt, voxelspacing=det_spacing, connectivity=1)
    elif metric == 'ASSD':
        metric_value = math.inf
        if np.max(gt) == 1 and np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = compute_assd(detection, gt, voxel_spacing=det_spacing)
    elif metric == 'OASSD':
        metric_value = math.inf
        if np.max(gt) == 1 and np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = compute_object_assd(detection, gt, voxel_spacing=det_spacing)
    elif metric == 'RAVD':
        metric_value = math.inf
        if np.max(gt) == 1 and np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value = compute_ravd(detection, gt)
    elif metric == 'VC':
        metric_value = math.inf
        if np.max(gt) == 1 and np.max(detection) == 1:  # Computation does not work if no binary object in the array
            metric_value, pval = compute_volume_correlation(detection, gt)
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
    else:
        logging.warning("Metric with name {} has not been implemented!".format(metric))
    return metric_value


def compute_overall_metrics_correlation(input_folder, output_folder, data=None, class_name=None,
                                        best_threshold=0.5, best_overlap=0.0, suffix=None):
    """

    :param input_folder:
    :param output_folder:
    :param class_name:
    :param data:
    :param best_threshold:
    :param best_overlap:
    :param suffix
    :return:
    """
    results = None
    if data is None:
        results_filename = os.path.join(input_folder, class_name + '_dice_scores.csv')
        results = pd.read_csv(results_filename)
    else:
        results = deepcopy(data)

    results.replace('inf', np.nan, inplace=True)
    optimal_results = results.loc[results['Threshold'] == best_threshold]
    results_for_matrix_df = optimal_results.drop(['Fold', 'Patient', 'Threshold'], axis=1)
    results_for_matrix_df = results_for_matrix_df.dropna()
    results_for_matrix_df = results_for_matrix_df.apply(pd.to_numeric)
    corr_matrix = results_for_matrix_df.corr()
    export_correlation_matrix_to_latex(output_folder, corr_matrix, suffix='_' + class_name + suffix)
    export_correlation_matrix_to_html(output_folder, corr_matrix, suffix='_' + class_name + suffix)


def export_correlation_matrix_to_html(output_folder, matrix, suffix=""):
    """
    Converts and saves the correlation matrix as an html table

    ...
    Attributes
    ----------
    output_folder: str
        Destination folder where the latex table will be saved under correlation_matrix.html
    matrix: pd.DataFrame
        Correlation matrix object as a pandas DataFrame
    suffix: (optional) str
        Name to append to the end of the destination file
    """
    matrix_filename = os.path.join(output_folder, 'correlation_matrix' + suffix + '.html')
    matrix.style.background_gradient(cmap='coolwarm').to_html(matrix_filename)


def export_correlation_matrix_to_latex(output_folder, matrix, suffix=""):
    """
    Converts and saves the correlation matrix as a latex table
    For the latex code to compile without errors, the following packages must be added at the top of the document:
        usepackage[table]{xcolor}
        usepackage{adjustbox}

    ...
    Attributes
    ----------
    output_folder: str
        Destination folder where the latex table will be saved under correlation_matrix.txt
    matrix: pd.DataFrame
        Correlation matrix object as a pandas DataFrame
    suffix: (optional) str
        Name to append to the end of the destination file
    """

    matrix_filename = os.path.join(output_folder, 'correlation_matrix' + suffix + '.txt')
    pfile = open(matrix_filename, 'w')

    pfile.write('\\begin{table}[h]\n')
    pfile.write('\\caption{Metrics confusion matrix. The color intensity of each cell represents the strength of'
                ' the correlation, where blue denotes direct correlation and red denotes inverse correlation.}\n')
    pfile.write('\\adjustbox{max width=\\textwidth}{\n')
    pfile.write('\\begin{tabular}{l'+('r' * matrix.shape[0]) + '}\n')
    pfile.write('\\toprule\n')
    header_line = '{}'
    for elem in matrix.axes[0].values:
        if '#' in elem:
            elem = elem.replace('#', 'Num ')
        header_line = header_line + ' & ' + elem
    pfile.write(header_line + '\\tabularnewline\n')
    for r in range(matrix.shape[0]):
        line = matrix.axes[1].values[r]
        if '#' in line:
            line = line.replace('#', 'Num ')
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
