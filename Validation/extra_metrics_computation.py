import os
import traceback
import itertools
import multiprocessing
from copy import deepcopy
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
from medpy.metric.binary import hd95, volume_correlation, assd, ravd, obj_assd
from sklearn.metrics import mutual_info_score, jaccard_score, normalized_mutual_info_score, adjusted_rand_score,\
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
from Utils.io_converters import get_fold_from_file
from Utils.resources import SharedResources


def compute_extra_metrics(data_root, study_folder, nb_folds, split_way, optimal_threshold, metrics: list = [],
                          prediction_files_suffix=''):
    """
    Compute a bunch of metrics for the current validation study, assuming a binary case for now. All the computed
    results will be stored inside a specific extra_metrics_results_per_patient.csv and then merged with the main
    all_dice_scores.csv when all patients have been processed.
    The results are saved after each patient, making it possible to resume the computation if a crash occurred.
    :param data_root:
    :param study_folder:
    :param nb_folds:
    :param split_way:
    :param optimal_threshold:
    :param metrics:
    :return:
    """
    cross_validation_file = os.path.join(study_folder, 'cross_validation_folds.txt')
    all_results_file = os.path.join(study_folder, 'Validation', 'all_dice_scores.csv')
    all_results_df = pd.read_csv(all_results_file)
    all_results_df['Threshold'] = all_results_df['Threshold'].round(decimals=2)

    # Creating a specific output file for the given optimal probability threshold, even though the threshold should
    # not change over time.
    output_filename = os.path.join(study_folder, 'Validation', 'extra_metrics_results_per_patient_thr' +
                                   str(int(optimal_threshold * 100.)) + '.csv')
    if not os.path.exists(output_filename):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        results_df = pd.DataFrame(columns=['Fold', 'UID'] + metrics)
    else:
        results_df = pd.read_csv(output_filename)
        if results_df.columns[0] != 'Fold':
            results_df = pd.read_csv(output_filename, index_col=0)
        # Check if not more metrics have been included since last iteration
        missing_metrics = [x for x in metrics if not x in list(results_df.columns)[1:]]
        for m in missing_metrics:
            results_df[m] = None

    for fold in range(0, nb_folds):
        if split_way == 'two-way':
            test_set, _ = get_fold_from_file(filename=cross_validation_file, fold_number=fold)
        else:
            val_set, test_set = get_fold_from_file(filename=cross_validation_file, fold_number=fold)
        print('\nComputing metrics for fold {}/{}.\n'.format(fold, nb_folds-1))
        for i, patient in enumerate(tqdm(test_set)):
            try:
                # @TODO. Hard-coded, have to decide on naming convention....
                uid = patient.split('_')[1]
                sub_folder_index = patient.split('_')[0]
                patient_extended = '_'.join(patient.split('_')[1:-1]).strip()

                if len(results_df.loc[results_df['UID'] == int(uid)]) == 0:
                    buff_array = np.full((1, results_df.shape[1]), None)
                    buff_array[0, 0] = fold
                    buff_array[0, 1] = int(uid)
                    buff_df = pd.DataFrame(buff_array, columns=list(results_df.columns))
                    results_df = results_df.append(buff_df, ignore_index=True)
                elif not None in results_df.loc[results_df['UID'] == int(uid)].values[0] and not np.isnan(np.sum(results_df.loc[results_df['UID'] == int(uid)].values[0])):
                    continue

                patient_image_base = os.path.join(data_root, sub_folder_index, uid, 'volumes', patient_extended)
                patient_image_filename = None
                for _, _, files in os.walk(os.path.dirname(patient_image_base)):
                    for f in files:
                        if os.path.basename(patient_image_base) in f:
                            patient_image_filename = os.path.join(os.path.dirname(patient_image_base), f)
                    break

                ground_truth_base = os.path.join(data_root, sub_folder_index, uid, 'segmentations', patient_extended)
                ground_truth_filename = None
                for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                    for f in files:
                        if os.path.basename(ground_truth_base) in f:
                            ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                    break
                detection_filename = os.path.join(study_folder, 'predictions', str(fold), sub_folder_index + '_' + uid,
                                                  os.path.basename(patient_image_filename).split('.')[0] + '-' +
                                                  prediction_files_suffix)

                if not os.path.exists(detection_filename):
                    continue

                ground_truth_ni = nib.load(ground_truth_filename)
                if len(ground_truth_ni.shape) == 4:
                    ground_truth_ni = nib.four_to_three(ground_truth_ni)[0]
                detection_ni = nib.load(detection_filename)

                if detection_ni.shape != ground_truth_ni.shape:
                    continue

                gt = ground_truth_ni.get_data()
                gt[gt >= 1] = 1

                detection = detection_ni.get_data()[:]
                detection[detection < optimal_threshold] = 0
                detection[detection >= optimal_threshold] = 1

                tp_array = np.zeros(detection.shape)
                fp_array = np.zeros(detection.shape)
                tn_array = np.zeros(detection.shape)
                fn_array = np.zeros(detection.shape)

                tp_array[(gt == 1) & (detection == 1)] = 1
                fp_array[(gt == 0) & (detection == 1)] = 1
                tn_array[(gt == 0) & (detection == 0)] = 1
                fn_array[(gt == 1) & (detection == 0)] = 1

                # N-B: Sometimes unstable: it will crash if the image is too large... If so, just uncomment/comment the
                # two following lines to perform the task without multiprocessing.
                # if False:
                if SharedResources.getInstance().number_processes > 1:
                    metric_values = []
                    for metric in metrics:
                        metric_value = results_df.loc[results_df['UID'] == int(uid)][metric].values[0]
                        metric_values.append(metric_value)

                    pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                    pat_results = pool.map(parallel_metric_computation, zip(metrics, metric_values, itertools.repeat(gt),
                                                                            itertools.repeat(detection),
                                                                            itertools.repeat(detection_ni.header.get_zooms()),
                                                                            itertools.repeat(ground_truth_ni.header.get_zooms()),
                                                                            itertools.repeat(tp_array), itertools.repeat(tn_array),
                                                                            itertools.repeat(fp_array), itertools.repeat(fn_array)))
                    pool.close()
                    pool.join()

                    for res in pat_results:
                        results_df.loc[results_df['UID'] == int(uid), res[0]] = res[1]
                else:
                    for metric in metrics:
                        metric_value = results_df.loc[results_df['UID'] == int(uid)][metric].values[0]
                        if metric_value == metric_value and metric_value is not None:
                            continue

                        if metric == 'VS':
                            metric_value = 1 - ((abs(np.sum(fn_array) - np.sum(fp_array)))
                                                / ((2 * np.sum(tp_array)) + np.sum(fp_array) + np.sum(fn_array)))
                        elif metric == 'GCE':
                            param11 = (np.sum(fn_array) * (np.sum(fn_array) + (2 * np.sum(tp_array)))) / (np.sum(tp_array) + np.sum(fn_array))
                            param12 = (np.sum(fp_array) * (np.sum(fp_array) + (2 * np.sum(tn_array)))) / (np.sum(tn_array) + np.sum(fp_array))
                            param21 = (np.sum(fp_array) * (np.sum(fp_array) + (2 * np.sum(tp_array)))) / (np.sum(tp_array) + np.sum(fp_array))
                            param22 = (np.sum(fn_array) * (np.sum(fn_array) + (2 * np.sum(tn_array)))) / (np.sum(tn_array) + np.sum(fn_array))
                            metric_value = (1/np.prod(ground_truth_ni.header.get_zooms())) * min(param11+param12, param21+param22)
                        elif metric == 'MI':
                            metric_value = normalized_mutual_info_score(gt.flatten(), detection.flatten())
                        elif metric == 'ARI':
                            metric_value = adjusted_rand_score(gt.flatten(), detection.flatten())
                        elif metric == 'Jaccard':
                            metric_value = jaccard_score(gt.flatten(), detection.flatten())
                        elif metric == 'IOU':
                            intersection = (gt == 1) & (detection == 1)
                            union = (gt == 1) | (detection == 1)
                            metric_value = np.count_nonzero(intersection) / (np.count_nonzero(union) + 1e-5)
                        elif metric == 'TPR':
                            metric_value = np.sum(tp_array) / (np.sum(tp_array) + np.sum(fn_array) + 1e-5)
                        elif metric == 'TNR':
                            metric_value = np.sum(tn_array) / (np.sum(tn_array) + np.sum(fp_array) + 1e-5)
                        elif metric == 'FPR':
                            metric_value = np.sum(fp_array) / (np.sum(fp_array) + np.sum(tn_array) + 1e-5)
                        elif metric == 'FNR':
                            metric_value = np.sum(fn_array) / (np.sum(fn_array) + np.sum(tp_array) + 1e-5)
                        elif metric == 'PPV':
                            metric_value = np.sum(tp_array) / (np.sum(tp_array) + np.sum(fp_array) + 1e-5)
                        elif metric == 'AUC':
                            metric_value = roc_auc_score(gt.flatten(), detection.flatten())
                        elif metric == 'MCC':
                            metric_value = matthews_corrcoef(gt.flatten(), detection.flatten())
                        elif metric == 'CKS':
                            metric_value = cohen_kappa_score(gt.flatten(), detection.flatten())
                        elif metric == 'HD95':
                            metric_value = -1.
                            if np.max(detection) == 1:  # Computation does not work if no binary object in the array
                                metric_value = hd95(detection, gt, voxelspacing=detection_ni.header.get_zooms(), connectivity=1)
                        elif metric == 'ASSD':
                            metric_value = -1.
                            if np.max(detection) == 1:  # Computation does not work if no binary object in the array
                                metric_value = assd(detection, gt, voxelspacing=detection_ni.header.get_zooms(), connectivity=1)
                        elif metric == 'OASSD':
                            metric_value = -1.
                            if np.max(detection) == 1:  # Computation does not work if no binary object in the array
                                metric_value = obj_assd(detection, gt, voxelspacing=detection_ni.header.get_zooms(), connectivity=1)
                        elif metric == 'RAVD':
                            metric_value = -1.
                            if np.max(detection) == 1:  # Computation does not work if no binary object in the array
                                metric_value = ravd(detection, gt)
                        elif metric == 'VC':
                            metric_value = -1.
                            if np.max(detection) == 1:  # Computation does not work if no binary object in the array
                                metric_value, pval = volume_correlation(detection, gt)

                        results_df.at[results_df.loc[results_df['UID'] == int(uid)].index.values[0], metric] = metric_value
                results_df.to_csv(output_filename, index=False)
            except Exception as e:
                print(traceback.format_exc())
                continue

    results_df.to_csv(output_filename, index=False)

    # Attaching the extra metrics to the main dice scores file
    results_df = pd.read_csv(output_filename)
    valid_thresholds = np.unique(all_results_df['Threshold'])
    valid_thresholds = [np.round(x, 2) for x in valid_thresholds]

    existing_uids = np.unique(all_results_df['Patient'].values)
    for metric in metrics:
        if metric not in all_results_df.columns:
            matching_length_metric_values = [' '] * len(all_results_df)
            all_results_df = all_results_df.join(pd.DataFrame(np.asarray(matching_length_metric_values), columns=[metric]))

        for uid in existing_uids:
            index_all = all_results_df.loc[(all_results_df['Patient'] == uid) & (all_results_df['Threshold'] == optimal_threshold)][metric].index
            metric_value = ''
            if uid in results_df['UID'].values:
                metric_value = results_df.loc[results_df['UID'] == uid][metric].values[0]
            if metric_value != '':
                all_results_df.at[index_all, metric] = metric_value
    all_results_df.to_csv(all_results_file, index=False)


def parallel_metric_computation(args):
    """
    Metrics computation method linked to the multiprocessing strategy. Effectively where the call to compute is made.
    :param args: list of arguments split from the lists given to the multiprocessing.Pool call.
    :return: list with metric name and computed metric value.
    """
    metric = args[0]
    metric_value = args[1]
    gt = args[2]
    detection = args[3]
    det_spacing = args[4]
    gt_spacing = args[5]
    tp_array = args[6]
    tn_array = args[7]
    fp_array = args[8]
    fn_array = args[9]

    if metric_value == metric_value and metric_value is not None:
        return [metric, metric_value]

    try:
        if metric == 'VS':
            metric_value = 1 - ((abs(np.sum(fn_array) - np.sum(fp_array)))
                                / ((2 * np.sum(tp_array)) + np.sum(fp_array) + np.sum(fn_array)))
        elif metric == 'GCE':
            param11 = (np.sum(fn_array) * (np.sum(fn_array) + (2 * np.sum(tp_array)))) / (
                        np.sum(tp_array) + np.sum(fn_array))
            param12 = (np.sum(fp_array) * (np.sum(fp_array) + (2 * np.sum(tn_array)))) / (
                        np.sum(tn_array) + np.sum(fp_array))
            param21 = (np.sum(fp_array) * (np.sum(fp_array) + (2 * np.sum(tp_array)))) / (
                        np.sum(tp_array) + np.sum(fp_array))
            param22 = (np.sum(fn_array) * (np.sum(fn_array) + (2 * np.sum(tn_array)))) / (
                        np.sum(tn_array) + np.sum(fn_array))
            metric_value = (1 / np.prod(gt_spacing)) * min(param11 + param12, param21 + param22)
        elif metric == 'MI':
            metric_value = normalized_mutual_info_score(gt.flatten(), detection.flatten())
        elif metric == 'ARI':
            metric_value = adjusted_rand_score(gt.flatten(), detection.flatten())
        elif metric == 'Jaccard':
            metric_value = jaccard_score(gt.flatten(), detection.flatten())
        elif metric == 'IOU':
            intersection = (gt == 1) & (detection == 1)
            union = (gt == 1) | (detection == 1)
            metric_value = np.count_nonzero(intersection) / (np.count_nonzero(union) + 1e-5)
        elif metric == 'TPR':
            metric_value = np.sum(tp_array) / (np.sum(tp_array) + np.sum(fn_array) + 1e-5)
        elif metric == 'TNR':
            metric_value = np.sum(tn_array) / (np.sum(tn_array) + np.sum(fp_array) + 1e-5)
        elif metric == 'FPR':
            metric_value = np.sum(fp_array) / (np.sum(fp_array) + np.sum(tn_array) + 1e-5)
        elif metric == 'FNR':
            metric_value = np.sum(fn_array) / (np.sum(fn_array) + np.sum(tp_array) + 1e-5)
        elif metric == 'PPV':
            metric_value = np.sum(tp_array) / (np.sum(tp_array) + np.sum(fp_array) + 1e-5)
        elif metric == 'AUC':
            metric_value = roc_auc_score(gt.flatten(), detection.flatten())
        elif metric == 'MCC':
            metric_value = matthews_corrcoef(gt.flatten(), detection.flatten())
        elif metric == 'CKS':
            metric_value = cohen_kappa_score(gt.flatten(), detection.flatten())
        elif metric == 'HD95':
            metric_value = -1.
            if np.max(detection) == 1:  # Hausdorff computation does not work if no binary object in the array
                metric_value = hd95(detection, gt, voxelspacing=det_spacing, connectivity=1)
        elif metric == 'ASSD':
            metric_value = -1.
            if np.max(detection) == 1:
                metric_value = assd(detection, gt, voxelspacing=det_spacing, connectivity=1)
        elif metric == 'OASSD':
            metric_value = -1.
            if np.max(detection) == 1:
                metric_value = obj_assd(detection, gt, voxelspacing=det_spacing, connectivity=1)
        elif metric == 'RAVD':
            metric_value = -1.
            if np.max(detection) == 1:
                metric_value = ravd(detection, gt)
        elif metric == 'VC':
            metric_value = -1.
            if np.max(detection) == 1:
                metric_value, pval = volume_correlation(detection, gt)
    except Exception as e:
        print('Computing {} gave an exception'.format(metric))
        pass

    return [metric, metric_value]


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

    optimal_results = results.loc[results['Threshold'] == best_threshold]
    results_for_matrix_df = optimal_results.drop(['Patient', 'Fold', 'Threshold', '#GT', '#Det'], axis=1)
    results_for_matrix_df = results_for_matrix_df.apply(pd.to_numeric)
    corr_matrix = results_for_matrix_df.corr()
    # print(corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2).render())
    export_correlation_matrix_to_latex(folder, corr_matrix)


def export_correlation_matrix_to_latex(folder, matrix):
    matrix_filename = os.path.join(folder, 'Validation', 'correlation_matrix.txt')
    pfile = open(matrix_filename, 'w')

    pfile.write('\\begin{table}[h]\n')
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
