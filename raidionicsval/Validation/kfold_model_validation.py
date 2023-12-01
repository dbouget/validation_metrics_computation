import multiprocessing
import itertools

import time
import pandas as pd
from math import ceil

from tqdm import tqdm

from ..Computation.dice_computation import separate_dice_computation
from ..Validation.instance_segmentation_validation import *
from ..Utils.resources import SharedResources
from ..Utils.PatientMetricsStructure import PatientMetrics
from ..Utils.io_converters import get_fold_from_file
from ..Validation.validation_utilities import best_segmentation_probability_threshold_analysis, compute_fold_average
from ..Validation.extra_metrics_computation import compute_patient_extra_metrics


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
        self.metric_names = []
        self.metric_names.extend(SharedResources.getInstance().validation_metric_names)
        self.detection_overlap_thresholds = SharedResources.getInstance().validation_detection_overlap_thresholds
        print("Detection overlap: ", self.detection_overlap_thresholds)
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix
        self.patients_metrics = {}

    def run(self):
        self.__compute_metrics()
        class_optimal = best_segmentation_probability_threshold_analysis(self.input_folder,
                                                                         detection_overlap_thresholds=self.detection_overlap_thresholds)
        if len(SharedResources.getInstance().validation_metric_names) != 0:
            self.__compute_extra_metrics(class_optimal=class_optimal)
        compute_fold_average(self.input_folder, class_optimal=class_optimal, metrics=self.metric_names,
                             true_positive_state=False)
        compute_fold_average(self.input_folder, class_optimal=class_optimal, metrics=self.metric_names,
                             true_positive_state=True)

    def __compute_metrics(self):
        """
        Generate the Dice scores (and default instance detection metrics) for all the patients and 10 probability
        thresholds equally-spaced. All the computed results will be stored inside all_dice_scores.csv.
        The results are saved after each patient, making it possible to resume the computation if a crash occurred.
        @TODO. Include an override flag to recompute anyway.
        :return:
        """
        cross_validation_description_file = os.path.join(self.input_folder, 'cross_validation_folds.txt')
        self.results_df = []
        self.class_results_df = {}
        self.dice_output_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
        self.class_dice_output_filenames = {}
        for c in SharedResources.getInstance().validation_class_names:
            self.class_dice_output_filenames[c] = os.path.join(self.output_folder, c + '_dice_scores.csv')
            self.class_results_df[c] = []
        self.results_df_base_columns = ['Fold', 'Patient', 'Threshold']
        self.results_df_base_columns.extend(["PiW Dice", "PiW Recall", "PiW Precision", "PiW F1"])
        # self.results_df_base_columns.extend(["PaW Dice", "PaW Recall", "PaW Precision", "PaW F1"])
        self.results_df_base_columns.extend(["GT volume (ml)", "True Positive", "Detection volume (ml)"])
        self.results_df_base_columns.extend(["OW Dice", "OW Recall", "OW Precision", "OW F1", "OW Dice Largest Object", '#GT', '#Det'])
        self.results_df_base_columns.extend(SharedResources.getInstance().validation_metric_names)

        if not os.path.exists(self.dice_output_filename):
            self.results_df = pd.DataFrame(columns=self.results_df_base_columns)
        else:
            self.results_df = pd.read_csv(self.dice_output_filename)
            if self.results_df.columns[0] != 'Fold':
                self.results_df = pd.read_csv(self.dice_output_filename, index_col=0)
            missing_metrics = [x for x in SharedResources.getInstance().validation_metric_names if
                               not x in list(self.results_df.columns)[1:]]
            for m in missing_metrics:
                self.results_df[m] = None

        for c in SharedResources.getInstance().validation_class_names:
            if not os.path.exists(self.class_dice_output_filenames[c]):
                self.class_results_df[c] = pd.DataFrame(columns=self.results_df_base_columns)
            else:
                self.class_results_df[c] = pd.read_csv(self.class_dice_output_filenames[c])
                if self.class_results_df[c].columns[0] != 'Fold':
                    self.class_results_df[c] = pd.read_csv(self.class_dice_output_filenames[c], index_col=0)
                missing_metrics = [x for x in SharedResources.getInstance().validation_metric_names if
                                   not x in list(self.class_results_df[c].columns)[1:]]
                for m in missing_metrics:
                    self.class_results_df[c][m] = None

        self.results_df['Patient'] = self.results_df.Patient.astype(str)
        for c in SharedResources.getInstance().validation_class_names:
            self.class_results_df[c]['Patient'] = self.class_results_df[c].Patient.astype(str)

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
                else:
                    # Option2. For files not following the original naming conventions
                    pid = patient
                    sub_folder_index = None

                uid = str(fold_number) + '_' + pid
                # Placeholder for holding all metrics for the current patient
                patient_metrics = PatientMetrics(id=uid, patient_id=pid, fold_number=fold_number,
                                                 class_names=SharedResources.getInstance().validation_class_names)
                patient_metrics.init_from_file(self.output_folder)

                success = self.__identify_patient_files(patient_metrics, sub_folder_index, fold_number)
                self.patients_metrics[uid] = patient_metrics

                # Checking if values have already been computed for the current patient to skip it if so.
                if patient_metrics.is_complete():
                    continue
                if not success:
                    print('Input files not found for patient {}\n'.format(uid))
                    continue

                self.__generate_dice_scores_for_patient(patient_metrics, fold_number)
            except Exception as e:
                print('Issue processing patient {}\n'.format(uid))
                print(traceback.format_exc())
                continue
        return 0

    def __identify_patient_files(self, patient_metrics, folder_index, fold_number):
        """
        Asserts the existence of the raw files on disk for computing the metrics for the current patient.
        :return:
        """
        uid = patient_metrics.patient_id
        classes = SharedResources.getInstance().validation_class_names
        nb_classes = len(classes)
        patient_filenames = {}

        # Iterating over all classes, where independent files are expected
        for c in range(nb_classes):
            patient_filenames[classes[c]] = []
            gt_suffix = self.gt_files_suffix[c]
            pred_suffix = self.prediction_files_suffix[c]

            # Annoying, but independent of extension
            # @TODO. must load images with SimpleITK to be completely generic.
            detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number), uid)
            if folder_index is not None:
               detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number),
                                                   folder_index + '_' + uid)

            detection_filename = None
            for _, _, files in os.walk(detection_image_base):
                for f in files:
                    if pred_suffix in f:
                        detection_filename = os.path.join(detection_image_base, f)
                break
            if not os.path.exists(detection_filename):
                print("No detection file found for class {} in patient {}".format(c, patient_metrics.unique_id))
                return False

            # @TODO. Second piece added to make it work when names are wrong in the cross validation file.
            patient_extended = uid
            patient_image_base = os.path.join(self.data_root, uid, patient_extended)
            if folder_index is not None:
                patient_extended = os.path.basename(detection_filename).split(pred_suffix)[0][:-1]
                patient_image_base = os.path.join(self.data_root, folder_index, uid, 'volumes', patient_extended)

            patient_image_filename = None
            for _, _, files in os.walk(os.path.dirname(patient_image_base)):
                for f in files:
                    if os.path.basename(patient_image_base) in f:
                        patient_image_filename = os.path.join(os.path.dirname(patient_image_base), f)
                break

            ground_truth_base = os.path.join(self.data_root, uid, patient_extended)
            if folder_index is not None:
                ground_truth_base = os.path.join(self.data_root, folder_index, uid, 'segmentations', patient_extended)

            ground_truth_filename = None
            for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                for f in files:
                    if os.path.basename(ground_truth_base) in f and gt_suffix in f:
                        ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                break

            # Specific actions for remapping BraTS results to match the whole tumor and tumor core categories
            if SharedResources.getInstance().validation_use_brats_data and (classes[c] == 'whole' or classes[c] == 'core'):
                ground_truth_filename = os.path.join(detection_image_base, patient_extended + '_' + gt_suffix)
            # The ground truth for the BraTS images is stored a bit differently
            elif SharedResources.getInstance().validation_use_brats_data and classes[c] == 'tumor':
                raw_gt = nib.load(ground_truth_filename).get_fdata()[:]
                ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_tumor.nii.gz")
                if not os.path.exists(ground_truth_filename):
                    new_gt = np.zeros(detection_ni.get_fdata().shape)
                    new_gt[raw_gt == 1] = 1
                    nib.save(nib.Nifti1Image(new_gt, detection_ni.affine), ground_truth_filename)
                tmp_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_necrosis.nii.gz")
                if not os.path.exists(tmp_filename):
                    new_gt = np.zeros(detection_ni.get_fdata().shape)
                    new_gt[raw_gt == 2] = 1
                    new_gt[raw_gt == 1] = 0
                    nib.save(nib.Nifti1Image(new_gt, detection_ni.affine), tmp_filename)
            elif SharedResources.getInstance().validation_use_brats_data and classes[c] == 'necrosis':
                ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_necrosis.nii.gz")

            detection_ni = nib.load(detection_filename)
            # If there's no ground truth, we assume the class to be empty for this patient and create an
            # empty ground truth volume.
            if ground_truth_filename is None or not os.path.exists(ground_truth_filename):
                ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_" + classes[c] + ".nii.gz")
                if not os.path.exists(ground_truth_filename):
                    empty_gt = np.zeros(detection_ni.get_fdata().shape)
                    nib.save(nib.Nifti1Image(empty_gt, detection_ni.affine), ground_truth_filename)
            else:
                file_stats = os.stat(detection_filename)
                ground_truth_ni = nib.load(ground_truth_filename)
                if len(ground_truth_ni.shape) == 4:
                    ground_truth_ni = nib.four_to_three(ground_truth_ni)[0]

                if file_stats.st_size == 0:
                    nib.save(nib.Nifti1Image(np.zeros(ground_truth_ni.get_shape), affine=ground_truth_ni.affine),
                             detection_filename)

                if detection_ni.shape != ground_truth_ni.shape:
                    return False

            patient_filenames[classes[c]] = [ground_truth_filename, detection_filename]
        patient_metrics.set_patient_filenames(patient_filenames)
        return True

    def __generate_dice_scores_for_patient(self, patient_metrics, fold_number):
        """
        Compute the basic metrics for all classes of the current patient
        :return:
        """
        uid = patient_metrics.patient_id
        classes = SharedResources.getInstance().validation_class_names
        nb_classes = len(classes)
        patient_filenames = {}
        thr_range = np.arange(0.1, 1.1, 0.1)

        # Iterating over all classes, where independent files are expected
        for c in range(nb_classes):
            gt_filename, det_filename = patient_metrics.get_class_filenames(c)
            ground_truth_ni = nib.load(gt_filename)
            detection_ni = nib.load(det_filename)

            gt = ground_truth_ni.get_fdata()
            gt[gt >= 1] = 1

            class_tp_threshold = SharedResources.getInstance().validation_true_positive_volume_thresholds[c]
            gt_volume = np.count_nonzero(gt) * np.prod(ground_truth_ni.header.get_zooms()) * 1e-3
            tp_state = True if gt_volume > class_tp_threshold else False
            extra = [np.round(gt_volume, 4), tp_state]
            pat_results = []
            if SharedResources.getInstance().number_processes > 1:
                pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                pat_results = pool.map(separate_dice_computation, zip(thr_range,
                                                                      itertools.repeat(fold_number),
                                                                      itertools.repeat(gt),
                                                                      itertools.repeat(detection_ni),
                                                                      itertools.repeat(uid),
                                                                      itertools.repeat(extra)
                                                                      )
                                       )
                pool.close()
                pool.join()
            else:
                for thr_value in thr_range:
                    thr_res = separate_dice_computation([thr_value, fold_number, gt, detection_ni, uid, extra])
                    pat_results.append(thr_res)

            patient_metrics.set_class_regular_metrics(classes[c], pat_results)
            # Filling in the csv files on disk for faster resume
            class_results_filename = self.class_dice_output_filenames[classes[c]]
            for ind, th in enumerate(thr_range):
                th = np.round(th, 2)
                sub_df = self.class_results_df[classes[c]].loc[
                    (self.class_results_df[classes[c]]['Patient'] == uid) & (self.class_results_df[classes[c]]['Fold'] == fold_number) & (
                            self.class_results_df[classes[c]]['Threshold'] == th)]
                # ind_values = np.asarray(pat_results[ind])
                # buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df_base_columns)),
                #                        columns=list(self.results_df_base_columns))
                if len(sub_df) == 0:
                    extra_metrics = [None] * len(SharedResources.getInstance().validation_metric_names)
                    ind_values = np.asarray(pat_results[ind][0] + extra_metrics)
                    buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df_base_columns)),
                                           columns=list(self.results_df_base_columns))
                    # self.class_results_df[classes[c]] = self.class_results_df[classes[c]].append(buff_df,
                    #                                                                              ignore_index=True)
                    self.class_results_df[classes[c]] = pd.concat([self.class_results_df[classes[c]], buff_df],
                                                                  ignore_index=True)
                else:
                    ind_values = pat_results[ind][0] + list(self.class_results_df[classes[c]].loc[sub_df.index.values[0], :].values[len(pat_results[ind][0]):])
                    self.class_results_df[classes[c]].loc[sub_df.index.values[0], :] = ind_values
            self.class_results_df[classes[c]].to_csv(class_results_filename, index=False)

        # Should compute the class macro-average results if multiple classes
        class_averaged_results = None
        class_results = []
        for c in classes:
            pat_class_results = patient_metrics.get_class_metrics(c)
            pat_class_extra_metrics = patient_metrics.get_class_extra_metrics_without_header(c)
            final_pat_class_res = [pat_class_results[x] for x in range(len(thr_range))]
            if len(SharedResources.getInstance().validation_metric_names) != 0:
                final_pat_class_res = [pat_class_results[x] + pat_class_extra_metrics[x] for x in range(len(thr_range))]
            class_results.append(final_pat_class_res)
        class_averaged_results = np.average(np.asarray(class_results).astype('float32')[:, :, 1:], axis=0)

        # Filling in the csv files on disk for faster resume
        for ind, th in enumerate(thr_range):
            th = np.round(th, 2)
            sub_df = self.results_df.loc[
                (self.results_df['Patient'] == uid) & (self.results_df['Fold'] == fold_number) & (
                            self.results_df['Threshold'] == th)]
            # ind_values = np.asarray([fold_number, uid, np.round(th, 2)] + list(class_averaged_results[ind]))
            # buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df_base_columns)),
            #                        columns=list(self.results_df_base_columns))
            if len(sub_df) == 0:
                ind_values = np.asarray([fold_number, uid, np.round(th, 2)] + list(class_averaged_results[ind]))
                buff_df = pd.DataFrame(ind_values.reshape(1, len(self.results_df_base_columns)),
                                       columns=list(self.results_df_base_columns))
                # self.results_df = self.results_df.append(buff_df, ignore_index=True)
                self.results_df = pd.concat([self.results_df, buff_df], ignore_index=True)
            else:
                ind_values = [fold_number, uid, np.round(th, 2)] + list(class_averaged_results[ind])
                self.results_df.loc[sub_df.index.values[0], :] = ind_values
        self.results_df.to_csv(self.dice_output_filename, index=False)

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
