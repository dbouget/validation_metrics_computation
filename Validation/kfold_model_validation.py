import os
import subprocess, shutil
import multiprocessing
import itertools
import traceback

import numpy as np
import csv
import time
import nibabel as nib
from nibabel import four_to_three
from copy import deepcopy
import pandas as pd
from math import ceil

from Validation.instance_segmentation_validation import *
from Utils.resources import SharedResources
from Utils.io_converters import get_fold_from_file
from Utils.volume_utilities import compute_tumor_volume
from Validation.validation_utilities import best_segmentation_probability_threshold_analysis,\
    best_segmentation_probability_threshold_analysis_postop, compute_fold_average
from Validation.extra_metrics_computation import compute_extra_metrics, compute_overall_metrics_correlation
from tqdm import tqdm

def compute_dice(volume1, volume2, epsilon=0.1):
    dice = (np.sum(volume1[volume2 == 1]) * 2.0 + epsilon) / (np.sum(volume1) + np.sum(volume2) + epsilon)
    return dice


def separate_dice_computation(args):
    """
    Dice computation method linked to the multiprocessing strategy. Effectively where the call to compute is made.
    :param args: list of arguments split from the lists given to the multiprocessing.Pool call.
    :return: list with the computed results for the current patient, at the given probability threshold.
    """
    t = args[0]
    fold_number = args[1]
    gt = args[2]
    detection_ni = args[3]
    patient_id = args[4]
    results = []

    detection = deepcopy(detection_ni.get_data())
    detection[detection < t] = 0
    detection[detection >= t] = 1
    detection = detection.astype('uint8')
    dice = compute_dice(gt, detection)
    voxel_size = np.prod(detection_ni.header.get_zooms()[0:3])
    volume_seg_ml = compute_tumor_volume(detection, voxel_size)

    obj_val = InstanceSegmentationValidation(gt_image=gt, detection_image=detection)
    try:
        # obj_val.set_trace_parameters(self.output_folder, fold_number, patient, t)
        obj_val.spacing = detection_ni.header.get_zooms()
        obj_val.run()
    except Exception as e:
        print('Issue computing instance segmentation parameters for patient {}'.format(patient_id))
        print(traceback.format_exc())

    instance_results = obj_val.instance_detection_results
    results.append([fold_number, patient_id, t, dice] + instance_results + [len(obj_val.gt_candidates),
                                                                            len(obj_val.detection_candidates)] \
                   + [volume_seg_ml])

    return results


class ModelValidation:
    """
    Compute performances metrics after k-fold cross-validation from sets of inference.
    The results will be stored inside a Validation sub-folder placed within the provided destination directory.
    """
    def __init__(self):
        self.data_root = SharedResources.getInstance().data_root
        self.input_folder = SharedResources.getInstance().validation_input_folder
        base_output_folder = SharedResources.getInstance().validation_output_folder

        if base_output_folder is not None and base_output_folder != "":
            self.output_folder = os.path.join(base_output_folder, 'Validation')
        else:
            self.output_folder = os.path.join(self.input_folder, 'Validation')
        os.makedirs(self.output_folder, exist_ok=True)

        self.fold_number = SharedResources.getInstance().validation_nb_folds
        self.split_way = SharedResources.getInstance().validation_split_way
        self.metric_names = ['Dice', 'Inst DICE', 'Inst Recall', 'Inst Precision', 'Largest foci Dice']
        self.metric_names.extend(SharedResources.getInstance().validation_metric_names)
        self.detection_overlap_thresholds = SharedResources.getInstance().validation_detection_overlap_thresholds
        print("Detection overlap: ", self.detection_overlap_thresholds)
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix

    def run(self):
        self.__generate_dice_scores()
        optimal_overlap, optimal_threshold = best_segmentation_probability_threshold_analysis_postop(self.input_folder,
                                                                                              detection_overlap_thresholds=self.detection_overlap_thresholds)

        compute_fold_average(self.input_folder, best_threshold=optimal_threshold, best_overlap=optimal_overlap)
        compute_extra_metrics(self.data_root, self.input_folder, nb_folds=self.fold_number, split_way=self.split_way,
                              optimal_threshold=optimal_threshold, metrics=self.metric_names[5:],
                              gt_files_suffix=self.gt_files_suffix,
                              prediction_files_suffix=self.prediction_files_suffix)
        compute_overall_metrics_correlation(self.input_folder, best_threshold=optimal_threshold)

    def __generate_dice_scores(self):
        """
        Generate the Dice scores (and default instance detection metrics) for all the patients and 10 probability
        thresholds equally-spaced. All the computed results will be stored inside all_dice_scores.csv.
        The results are saved after each patient, making it possible to resume the computation if a crash occurred.
        @TODO. Include an override flag to recompute anyway.
        :return:
        """
        cross_validation_description_file = os.path.join(self.input_folder, 'cross_validation_folds.txt')
        self.results_df = []
        self.dice_output_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
        if not os.path.exists(self.dice_output_filename):
            self.results_df = pd.DataFrame(columns=['Fold', 'Patient', 'Threshold', 'Dice', 'Inst DICE', 'Inst Recall',
                                                    'Inst Precision', 'Largest foci Dice', '#GT', '#Det', 'Volume segmentation'])
        else:
            self.results_df = pd.read_csv(self.dice_output_filename)
            if self.results_df.columns[0] != 'Fold':
                self.results_df = pd.read_csv(self.dice_output_filename, index_col=0)
        self.results_df['Patient'] = self.results_df.Patient.astype(str)

        results_per_folds = []
        for fold in range(0, self.fold_number):
            print('\nProcessing fold {}/{}.\n'.format(fold, self.fold_number - 1))
            if self.split_way == 'two-way':
                test_set, _ = get_fold_from_file(filename=cross_validation_description_file, fold_number=fold)
            else:
                val_set, test_set = get_fold_from_file(filename=cross_validation_description_file, fold_number=fold)
            results = self.__generate_dice_scores_for_fold(data_list=test_set, fold_number=fold)
            results_per_folds.append(results)

    def __generate_dice_scores_for_fold(self, data_list, fold_number):
        for i, patient in enumerate(tqdm(data_list)):
            uid = None
            try:
                # @TODO. Hard-coded, have to decide on naming convention....
                # start = time.time()
                uid = patient.split('_')[1]
                sub_folder_index = patient.split('_')[0]
                patient_extended = '_'.join(patient.split('_')[1:-1]).strip()

                # Checking if values have already been computed for the current patient to skip it if so.
                # In case values were not properly computed for the core part (i.e. first 10 columns without
                # extra-metrics), a recompute will be triggered.
                if len(self.results_df.loc[self.results_df['Patient'] == uid]) != 0:
                    if not None in self.results_df.loc[self.results_df['Patient'] == uid].values[0] and not np.isnan(
                            np.sum(self.results_df.loc[self.results_df['Patient'] == uid].values[0][3:10])):
                        continue

                # Annoying, but independent of extension
                # @TODO. must load images with SimpleITK to be completely generic.
                patient_image_base = os.path.join(self.data_root, sub_folder_index, uid, 'volumes', patient_extended)
                patient_image_filename = None
                for _, _, files in os.walk(os.path.dirname(patient_image_base)):
                    for f in files:
                        if os.path.basename(patient_image_base) in f:
                            patient_image_filename = os.path.join(os.path.dirname(patient_image_base), f)
                    break

                ground_truth_base = os.path.join(self.data_root, sub_folder_index, uid, 'segmentations', patient_extended)
                ground_truth_filename = None
                for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                    for f in files:
                        if os.path.basename(ground_truth_base) in f and self.gt_files_suffix in f:
                            ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                    break
                detection_filename = os.path.join(self.input_folder, 'predictions', str(fold_number),
                                                  sub_folder_index + '_' + uid,
                                                  os.path.basename(patient_image_filename).split('.')[0] +
                                                  '-' + self.prediction_files_suffix)
                if not os.path.exists(detection_filename):
                    continue

                file_stats = os.stat(detection_filename)
                ground_truth_ni = nib.load(ground_truth_filename)
                if len(ground_truth_ni.shape) == 4:
                    ground_truth_ni = nib.four_to_three(ground_truth_ni)[0]

                if file_stats.st_size == 0:
                    nib.save(nib.Nifti1Image(np.zeros(ground_truth_ni.get_shape), affine=ground_truth_ni.affine),
                             detection_filename)

                detection_ni = nib.load(detection_filename)
                if detection_ni.shape != ground_truth_ni.shape:
                    continue

                gt = ground_truth_ni.get_data()
                gt[gt >= 1] = 1

                pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                thr_range = np.arange(0.1, 1.1, 0.1)
                pat_results = pool.map(separate_dice_computation, zip(thr_range,
                                                                      itertools.repeat(fold_number),
                                                                      itertools.repeat(gt),
                                                                      itertools.repeat(detection_ni),
                                                                      itertools.repeat(uid)
                                                                      )
                                       )
                pool.close()
                pool.join()

                for ind, th in enumerate(thr_range):
                    sub_df = self.results_df.loc[(self.results_df['Patient'] == uid) & (self.results_df['Fold'] == fold_number) & (self.results_df['Threshold'] == th)]
                    ind_values = np.asarray(pat_results).reshape((len(thr_range), len(self.results_df.columns)))[ind, :]
                    buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df.columns)),
                                           columns=list(self.results_df.columns))
                    if len(sub_df) == 0:
                        self.results_df = self.results_df.append(buff_df, ignore_index=True)
                    else:
                        self.results_df.loc[sub_df.index.values[0], :] = list(ind_values)
                self.results_df.to_csv(self.dice_output_filename, index=False)
            except Exception as e:
                print('Issue processing patient {}\n'.format(uid))
                print(traceback.format_exc())
                continue
        return 0
