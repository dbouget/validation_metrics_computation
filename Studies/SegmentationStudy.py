import os
import traceback
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from Studies.AbstractStudy import AbstractStudy
from Utils.resources import SharedResources
from Validation.custom_airways_metrics_computation import *


class SegmentationStudy(AbstractStudy):

    def __init__(self):
        super().__init__()

    def run(self):
        self.__airways_specific()
        for c in self.class_names:
            super().compute_and_plot_overall(c, category='All')
            super().compute_and_plot_overall(c, category='True Positive')
            # compute_overall_metrics_correlation(self.input_folder, best_threshold=self.optimal_threshold)  # Not tested yet
            if self.extra_patient_parameters is not None:
                self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice', metric2='Volume',
                                                                    metric2_cutoffs=[2.], category='All')
                self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice', metric2='SpacZ',
                                                                    metric2_cutoffs=[2.], category='All')

    def __airways_specific(self):
        database_root = "/home/mnt/data/Lungs/LungsDatabase"
        groundtruth_skeleton_root = os.path.join(SharedResources.getInstance().studies_input_folder, "skeleton_gt")
        predictions_folder = os.path.join(SharedResources.getInstance().validation_input_folder, 'predictions')
        folds = []
        all_metrics = []
        all_metrics_columns = ['Fold', 'Patient', 'Precision', 'Sensitivity', 'Specificity', 'FNR', 'FPR', 'Dice',
                               'Tree length', 'Total branches', 'Detected branches', 'Branch detection ratio']

        # Collecting all the existing folds folder for the current k-fold cross-validation
        for _, dirs, _ in os.walk(predictions_folder):
            for d in dirs:
                folds.append(d)
            break

        # Iterating through each fold
        for fold in folds:
            print("Processing fold {}".format(fold))
            fold_folder = os.path.join(predictions_folder, fold)
            patients = []

            # Collecting all the patients inside the current fold
            for _, dirs, _ in os.walk(fold_folder):
                for d in dirs:
                    patients.append(d)
                break

            # Iterating through all the patients from the current fold
            for pat in tqdm(patients):
                pat_folder = os.path.join(fold_folder, pat)
                try:
                    uid = pat.split('_')[1]
                    pat_skeleton_folder = os.path.join(pat_folder, 'skeleton_gt')
                    # if not os.path.exists(pat_skeleton_folder):
                    #     print("No skeleton predictions for patient {}. Please compute beforehand!".format(pat))
                    #     continue

                    # Collecting the necessary ground truth files belonging for the current patient
                    pat_skel_groundtruth_folder = os.path.join(groundtruth_skeleton_root, uid)
                    gt_skeleton_filename = None
                    gt_label_filename = None
                    gt_parse_filename = None
                    for _, _, files in os.walk(pat_skel_groundtruth_folder):
                        for f in files:
                            if 'skel' in f:
                                gt_skeleton_filename = os.path.join(pat_skel_groundtruth_folder, f)
                            elif 'label' in f:
                                gt_label_filename = os.path.join(pat_skel_groundtruth_folder, f)
                            elif 'parse' in f:
                                gt_parse_filename = os.path.join(pat_skel_groundtruth_folder, f)
                        break
                    pat_groundtruth_filename = os.path.join(database_root, pat.split('_')[0], uid, 'segmentations',
                                                            os.path.basename(gt_skeleton_filename).replace('_skel.nii.gz', '_label_airways.nii.gz'))

                    # Collecting the necessary files belonging to the model predictions for the current patient
                    pat_prediction_filename = None
                    for _, _, files in os.walk(pat_folder):
                        for f in files:
                            if 'pred' in f:
                                pat_prediction_filename = os.path.join(pat_folder, f)
                        break
                    pred_skeleton_filename = None
                    pred_label_filename = None
                    pred_parse_filename = None
                    for _, _ , files in os.walk(pat_skeleton_folder):
                        for f in files:
                            if 'skel' in f:
                                pred_skeleton_filename = os.path.join(pat_skeleton_folder, f)
                            elif 'label' in f:
                                pred_label_filename = os.path.join(pat_skeleton_folder, f)
                            elif 'parse' in f:
                                pred_parse_filename = os.path.join(pat_skeleton_folder, f)
                        break

                    # Loading all nifti files as numpy arrays
                    pred_binary = nib.load(pat_prediction_filename).get_data()[:]
                    # pred_skeleton = nib.load(pred_skeleton_filename).get_data()[:]
                    # pred_label = nib.load(pred_label_filename).get_data()[:]
                    # pred_parse = nib.load(pred_parse_filename).get_data()[:]

                    gt_binary = nib.load(pat_groundtruth_filename).get_data()[:]
                    gt_skeleton = nib.load(gt_skeleton_filename).get_data()[:]
                    gt_label = nib.load(gt_label_filename).get_data()[:]
                    gt_parse = nib.load(gt_parse_filename).get_data()[:]

                    # Computing the metrics
                    precision = precision_calculation(pred_binary, gt_binary)
                    specificity = specificity_calculation(pred_binary, gt_binary)
                    sensitivity = sensitivity_calculation(pred_binary, gt_binary)
                    fnr = false_negative_rate_calculation(pred_binary, gt_binary)
                    fpr = false_positive_rate_calculation(pred_binary, gt_binary)
                    dice = dice_coefficient_score_calculation(pred_binary, gt_binary)
                    tree_length = tree_length_calculation(pred_binary, gt_skeleton)
                    branch_detection = branch_detected_calculation(pred_binary.astype('uint8'), gt_parse, gt_skeleton)
                    total_branches = branch_detection[0]
                    detected_branches = branch_detection[1]
                    branch_detection_ratio = branch_detection[2]
                    pat_metrics = [fold, pat, precision, specificity, sensitivity, fnr, fpr, dice, tree_length,
                                   total_branches, detected_branches, branch_detection_ratio]
                    all_metrics.append(pat_metrics)
                except Exception as e:
                    print("Error computing the airways-specific metrics for patient {}.".format(pat))
                    print(traceback.format_exc())
        all_metrics_df = pd.DataFrame(all_metrics, columns=all_metrics_columns)
        destination_filename = os.path.join(self.output_folder, SharedResources.getInstance().studies_study_name + '_airways-specific-metrics.csv')
        all_metrics_df.to_csv(destination_filename, index=False)
