import multiprocessing
import itertools
import os.path

import time

import numpy as np
import pandas as pd
from math import ceil

from tqdm import tqdm

from ..Computation.dice_computation_instance import separate_dice_computation
from ..Validation.instance_segmentation_validation import *
from ..Utils.resources import SharedResources
from ..Utils.PatientMetricsStructure import PatientMetrics
from ..Utils.io_converters import get_fold_from_file
from ..Validation.validation_utilities_classification import compute_fold_average
from ..Validation.extra_metrics_computation import compute_patient_extra_metrics


class ClassificationModelValidation:
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
        self.metric_names = []
        self.metric_names.extend(SharedResources.getInstance().validation_metric_names)
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.gt_master_file = "/home/dbouget/Data/Studies/SequenceClassifier/seqclassifier_patients_parameters.csv"  # @TODO. Should be in the config file
        self.gt_master_file_df = pd.read_csv(self.gt_master_file)
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix
        self.patients_metrics = {}

    def run(self):
        self.__compute_metrics()
        # if len(SharedResources.getInstance().validation_metric_names) != 0:
        #     self.__compute_extra_metrics(class_optimal=class_optimal)
        compute_fold_average(self.input_folder, metrics=self.metric_names)

    def __compute_metrics(self):
        """

        """
        cross_validation_description_file = os.path.join(self.input_folder, 'cross_validation_folds.txt')
        self.results_df = []
        self.all_scores_output_filename = os.path.join(self.output_folder, 'all_scores.csv')
        self.results_df_base_columns = ['Fold', 'Patient']
        for c in SharedResources.getInstance().validation_class_names:
            self.results_df_base_columns.extend(["Proba " + c])
        self.results_df_base_columns.extend(["Prediction", "GT"])
        self.results_df_base_columns.extend(SharedResources.getInstance().validation_metric_names)

        if not os.path.exists(self.all_scores_output_filename):
            self.results_df = pd.DataFrame(columns=self.results_df_base_columns)
        else:
            self.results_df = pd.read_csv(self.all_scores_output_filename)
            if self.results_df.columns[0] != 'Fold':
                self.results_df = pd.read_csv(self.all_scores_output_filename, index_col=0)
            missing_metrics = [x for x in SharedResources.getInstance().validation_metric_names if
                               not x in list(self.results_df.columns)[1:]]
            for m in missing_metrics:
                self.results_df[m] = None
        self.results_df['Patient'] = self.results_df.Patient.astype(str)

        results_per_folds = []
        for fold in range(0, self.fold_number):
            print('\nProcessing fold {}/{}.\n'.format(fold + 1, self.fold_number))
            if self.split_way == 'two-way':
                test_set, _ = get_fold_from_file(filename=cross_validation_description_file, fold_number=fold)
            else:
                val_set, test_set = get_fold_from_file(filename=cross_validation_description_file, fold_number=fold)
            results = self.__compute_metrics_for_fold(data_list=test_set, fold_number=fold)
            results_per_folds.append(results)

    def __compute_metrics_for_fold(self, data_list, fold_number):
        for i, patient in enumerate(tqdm(data_list)):
            uid = None
            try:
                # Option1. Working for files using the original naming conventions.
                if SharedResources.getInstance().validation_use_index_naming_convention:
                    pid = patient.split('_')[1]
                    sub_folder_index = str(ceil(int(pid) / 200))  # patient.split('_')[0]
                    pid = pid + '_' + patient.split('_')[3] + '_' + patient.split('_')[4]
                else:
                    # Option2. For files not following the original naming conventions
                    pid = patient
                    sub_folder_index = None

                uid = str(fold_number) + '_' + pid
                # Placeholder for holding all metrics for the current patient
                patient_metrics = PatientMetrics(id=uid, patient_id=pid, fold_number=fold_number,
                                                 class_names=SharedResources.getInstance().validation_class_names,
                                                 objective="classification")
                patient_metrics.init_from_file(self.output_folder)

                success = self.__identify_patient_files(patient_metrics, sub_folder_index, fold_number, self.gt_master_file_df)
                self.patients_metrics[uid] = patient_metrics

                # Checking if values have already been computed for the current patient to skip it if so.
                if patient_metrics.is_complete():
                    continue
                if not success:
                    print('Input files not found for patient {}\n'.format(uid))
                    continue

                self.__generate_scores_for_patient(patient_metrics, fold_number)
            except Exception as e:
                print('Issue processing patient {}\n'.format(uid))
                print(traceback.format_exc())
                continue
        return 0

    def __identify_patient_files(self, patient_metrics, folder_index, fold_number, masterfile=None):
        """
        Asserts the existence of the raw files on disk for computing the metrics for the current patient.
        :return:
        """
        use_internal_convention = SharedResources.getInstance().validation_use_index_naming_convention
        uid = patient_metrics.patient_id
        if use_internal_convention:
            uid = patient_metrics.patient_id.split('_')[0]
        classes = SharedResources.getInstance().validation_class_names
        pred_suffix = self.prediction_files_suffix[0]

        detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number), uid)
        if folder_index is not None:
           detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number),
                                               folder_index + '_' + uid)

        detection_filenames = []
        for _, _, files in os.walk(detection_image_base):
            for f in files:
                if pred_suffix in f:
                    detection_filenames.append(os.path.join(detection_image_base, f))
            break
        if len(detection_filenames) == 0:
            print("No detection file found in patient {}".format(patient_metrics.unique_id))
            return False
        elif len(detection_filenames) == 1:
            detection_filename = detection_filenames[0]
        else:
            ts = patient_metrics.patient_id.split('_')[1]
            for f in detection_filenames:
                if ts in os.path.basename(f.split('.')[0]):
                    detection_filename = f
                    break

        gt_results_filename = os.path.join(detection_image_base,
                                           os.path.basename(detection_filename).split(pred_suffix)[0] + 'gt.csv')

        # ts = os.path.basename(detection_filename).split('.')[0].split('_')[-3]
        # image_uid = uid + '_' + ts
        # gt_class = masterfile.loc[masterfile["Patient"] == image_uid]["Sequence"].values[0]
        gt_class = os.path.basename(detection_filename).split('.')[0].split('_')[-4]
        gt_result = np.zeros(len(classes)).astype('uint8')
        gt_result[classes.index(gt_class)] = 1
        np.savetxt(gt_results_filename, gt_result, delimiter=";")

        patient_filenames = [gt_results_filename, detection_filename]
        patient_metrics.set_patient_filenames(patient_filenames)
        return True

    def __generate_scores_for_patient(self, patient_metrics, fold_number):
        """
        Compute the basic metrics for all classes of the current patient
        :return:
        """
        uid = patient_metrics.patient_id
        classes = SharedResources.getInstance().validation_class_names
        nb_classes = len(classes)
        patient_filenames = {}

        gt_filename, det_filename = patient_metrics.get_class_filenames(0)
        gt_array = np.genfromtxt(gt_filename, delimiter=";")
        det_array = np.genfromtxt(det_filename, delimiter=";")
        gt_class = classes[np.argmax(gt_array)]
        det_class = classes[np.argmax(det_array)]
        pat_results = []
        pat_results.extend(det_array)
        pat_results.extend([det_class, gt_class])

        # # Filling in the csv files on disk for faster resume
        # results_filename = self.all_scores_output_filename
        # sub_df = self.class_results_df[classes[c]].loc[
        #     (self.class_results_df[classes[c]]['Patient'] == uid) & (self.class_results_df[classes[c]]['Fold'] == fold_number) & (
        #             self.class_results_df[classes[c]]['Threshold'] == th)]
        # if len(sub_df) == 0:
        #     extra_metrics = [None] * len(SharedResources.getInstance().validation_metric_names)
        #     ind_values = np.asarray(pat_results[ind][0] + extra_metrics)
        #     buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df_base_columns)),
        #                            columns=list(self.results_df_base_columns))
        #     self.class_results_df[classes[c]] = pd.concat([self.class_results_df[classes[c]], buff_df],
        #                                                   ignore_index=True)
        # else:
        #     ind_values = pat_results[ind][0] + list(self.class_results_df[classes[c]].loc[sub_df.index.values[0], :].values[len(pat_results[ind][0]):])
        #     self.class_results_df[classes[c]].loc[sub_df.index.values[0], :] = ind_values
        # self.results_df.to_csv(self.all_scores_output_filename, index=False)

        # Filling in the csv files on disk for faster resume
        sub_df = self.results_df.loc[
            (self.results_df['Patient'] == uid) & (self.results_df['Fold'] == fold_number)]
        if len(sub_df) == 0:
            ind_values = np.asarray([fold_number, uid] + list(pat_results))
            buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df_base_columns)),
                                   columns=list(self.results_df_base_columns))
            self.results_df = pd.concat([self.results_df, buff_df], ignore_index=True)
        else:
            ind_values = [fold_number, uid] + list(pat_results)
            self.results_df.loc[sub_df.index.values[0], :] = ind_values
        self.results_df.to_csv(self.all_scores_output_filename, index=False)

    def __compute_extra_metrics(self, class_optimal: dict = {}):
        """

        """
        print("Computing extra metrics for all patients.\n")
        classes = SharedResources.getInstance().validation_class_names
        for c in classes:
            optimal_values = class_optimal[c]['All']
            for p in tqdm(self.patients_metrics):
                # Initializing/completing the list which will hold the extra metrics
                self.patients_metrics[p].setup_extra_metrics(self.metric_names)
                pat_metrics = compute_patient_extra_metrics(self.patients_metrics[p], classes.index(c), optimal_values[1],
                                                            SharedResources.getInstance().validation_metric_names)
                self.patients_metrics[p].set_optimal_class_extra_metrics(classes.index(c), optimal_values[1], pat_metrics)

                # Filling in the overall dataframe and dumping results to csv after each patient
                for pm in pat_metrics:
                    metric_name = pm[0]
                    metric_value = pm[1]
                    self.class_results_df[c].at[self.class_results_df[c].loc[(self.class_results_df[c]['Patient'] == self.patients_metrics[p].patient_id) & (self.class_results_df[c]['Threshold'] == optimal_values[1])].index.values[0], metric_name] = metric_value
                self.class_results_df[c].to_csv(self.class_dice_output_filenames[c], index=False)
