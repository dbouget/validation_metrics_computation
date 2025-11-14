import multiprocessing
import itertools
import os.path
import time
import numpy as np
import pandas as pd
from math import ceil
import traceback
from tqdm import tqdm

from ..Computation.dice_computation_instance import separate_dice_computation
from ..Utils.resources import SharedResources
from ..Utils.PatientMetricsStructure import PatientMetrics
from ..Utils.io_converters import get_fold_from_file, is_valid_extension, open_image_file
from ..Computation.generative_metrics import parallel_metric_computation, compute_specific_metric_value


class GenerativeModelValidation:
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
        self.prediction_files_suffix = SharedResources.getInstance().validation_generative_prediction_file_suffix
        self.patients_metrics = {}

    def run(self):
        self.__compute_metrics()
        # if len(SharedResources.getInstance().validation_metric_names) != 0:
        #     self.__compute_extra_metrics(class_optimal=class_optimal)
        # compute_fold_average(self.input_folder, metrics=self.metric_names)

    def __compute_metrics(self):
        """

        """
        cross_validation_description_file = os.path.join(self.input_folder, 'cross_validation_folds.txt')
        self.results_df = []
        self.all_scores_output_filename = os.path.join(self.output_folder, 'all_scores.csv')
        self.results_df_base_columns = ['Fold', 'Patient', 'Slice']
        if "slicewise" in SharedResources.getInstance().validation_metric_spaces:
            self.results_df_base_columns.extend([f"SW {x}" for x in SharedResources.getInstance().validation_generative_metrics])
        if "patientwise" in SharedResources.getInstance().validation_metric_spaces:
            self.results_df_base_columns.extend([f"PW {x}" for x in SharedResources.getInstance().validation_generative_metrics])
            if len(SharedResources.getInstance().validation_generative_temporal_metrics) != 0:
                # @TODO. Better way to make it generic?
                if "consistency" in SharedResources.getInstance().validation_generative_temporal_metrics:
                    self.results_df_base_columns.extend(["PW Consistency (Orig) - NCC", "PW Consistency (Gen) - NCC",
                                                         "PW Consistency (Orig) - SSIM", "PW Consistency (Gen) - SSIM",
                                                         "PW Consistency (Orig) - GradientCorr", "PW Consistency (Gen) - GradientCorr"])
                if "flicker" in SharedResources.getInstance().validation_generative_temporal_metrics:
                    self.results_df_base_columns.extend(["PW Flicker (Orig) - SDE", "PW Flicker (Gen) - SDE",
                                                         "PW Flicker (Orig) - L2", "PW Flicker (Gen) - L2",
                                                         "PW Flicker (Orig) - HFER", "PW Flicker (Gen) - HFER",
                                                         "PW Flicker (Orig) - TSTD", "PW Flicker (Gen) - TSTD",
                                                         "PW Flicker (Orig) - CS", "PW Flicker (Gen) - CS",
                                                         "SW Flicker (Orig) - L2", "SW Flicker (Gen) - L2",
                                                         "SW Flicker (Orig) - SDE", "SW Flicker (Gen) - SDE"])

        if not os.path.exists(self.all_scores_output_filename):
            self.results_df = pd.DataFrame(columns=self.results_df_base_columns)
        else:
            self.results_df = pd.read_csv(self.all_scores_output_filename)
            if self.results_df.columns[0] != 'Fold':
                self.results_df = pd.read_csv(self.all_scores_output_filename, index_col=0)

            missing_metrics = [x for x in self.results_df_base_columns if not x in list(self.results_df.columns)[1:]]
            for m in missing_metrics:
                self.results_df[m] = None

            # if "slicewise" in SharedResources.getInstance().validation_metric_spaces:
            #     missing_metrics = [f"SW {x}" for x in SharedResources.getInstance().validation_generative_metrics if
            #                        not f"SW {x}" in list(self.results_df.columns)[1:]]
            #     for m in missing_metrics:
            #         self.results_df[m] = None
            # if "patientwise" in SharedResources.getInstance().validation_metric_spaces:
            #     missing_metrics = [f"SW {x}" for x in SharedResources.getInstance().validation_generative_metrics if
            #                        not f"SW {x}" in list(self.results_df.columns)[1:]]
            #     for m in missing_metrics:
            #         self.results_df[m] = None
            #
            #     missing_metrics = [f"PW {x}" for x in SharedResources.getInstance().validation_generative_temporal_metrics if
            #                        not f"PW {x}" in list(self.results_df.columns)[1:]]
            #     for m in missing_metrics:
            #         self.results_df[m] = None
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
                                                 class_names=[], objective="generative")
                patient_metrics.init_from_file(self.output_folder)

                success = self.__identify_patient_files(patient_metrics, sub_folder_index, fold_number)
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
        pred_suffix = self.prediction_files_suffix

        detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number), uid)
        if folder_index is not None:
            detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number),
                                                folder_index + '_' + uid)

        detection_filename = None
        for _, _, files in os.walk(detection_image_base):
            for f in files:
                if pred_suffix in f and is_valid_extension(fn=f,
                                                           extensions=SharedResources.getInstance().valid_file_extensions):
                    if use_internal_convention and patient_metrics.patient_id.split('_')[1] in f.split('_'):
                        detection_filename = os.path.join(detection_image_base, f)
                    elif not use_internal_convention:
                        detection_filename = os.path.join(detection_image_base, f)
            break
        if detection_filename is None or not os.path.exists(detection_filename):
            print(f"No generative file found for patient {patient_metrics.unique_id}")
            return False

        # Identification of the ground truth filename
        patient_extended = uid
        if folder_index is not None:
            patient_extended = os.path.basename(detection_filename).split(pred_suffix)[0][:-1]

        ground_truth_base = os.path.join(self.data_root, uid, patient_extended)
        if SharedResources.getInstance().validation_use_index_naming_convention and folder_index is not None:
            ground_truth_base = os.path.join(self.data_root, folder_index, uid, 'volumes', patient_extended)

        ground_truth_filename = None
        for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
            for f in files:
                if os.path.basename(f).split('.')[0] in ground_truth_base and is_valid_extension(fn=f,
                                                                                   extensions=SharedResources.getInstance().valid_file_extensions):
                    ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
            break
        # The ground truth filename inside the folder does not match the folder name, looking for the first eligible file if any
        if ground_truth_filename is None:
            for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                for f in files:
                    if is_valid_extension(fn=f, extensions=SharedResources.getInstance().valid_file_extensions):
                        ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                break

        patient_filenames = [ground_truth_filename, detection_filename]
        patient_metrics.set_patient_filenames(patient_filenames)
        return True

    def __generate_scores_for_patient(self, patient_metrics, fold_number):
        """
        Compute the basic metrics for all classes of the current patient
        :return:
        """
        uid = patient_metrics.patient_id
        gt_filename = patient_metrics.ground_truth_filepaths
        gen_filename = patient_metrics.prediction_filepaths

        gt, _, gt_specs = open_image_file(gt_filename)
        generative, _, det_specs = open_image_file(gen_filename)

        # Normalize the original file to [0, 1]
        gt_norm = gt / np.max(gt)

        extra_metrics_results = []
        # @TODO. Check that the input has 3D for slice-wise to run? if len(generative.shape) < 3:
        if "slicewise" in SharedResources.getInstance().validation_metric_spaces:
            # @TODO. Should retrieve this from the patient params, based off the config file.
            slicewise_metrics_results = []
            metrics_values = [None] * len(SharedResources.getInstance().validation_generative_metrics)
            sw_metrics = SharedResources.getInstance().validation_generative_metrics
            if SharedResources.getInstance().number_processes > 1:
                try:
                    gt_slices = [gt_norm[:, :, i] for i in range(gt_norm.shape[2])]
                    gen_slices = [generative[:, :, i] for i in range(generative.shape[2])]
                    pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                    slicewise_metrics_results = pool.map(parallel_metric_computation, zip(gt_slices, gen_slices,
                                                                                      itertools.repeat(sw_metrics),
                                                                                      itertools.repeat(metrics_values)))
                    pool.close()
                    pool.join()
                    for sli, slim in enumerate(slicewise_metrics_results):
                        curr_pat_metrics = [["Slice", (sli+1)]]
                        for mm in slim:
                            curr_pat_metrics.extend([[f"SW {mm[0]}", mm[1]]])
                        extra_metrics_results.append(curr_pat_metrics)
                except Exception as e:
                    print(f'Issue computing metrics for patient {uid}: {e}')
                    print(traceback.format_exc())
            else:
                for z in range(generative.shape[2]):
                    for metric in sw_metrics:
                        try:
                            metric_value = compute_specific_metric_value(metric=metric, gt=gt_norm[:, :, z],
                                                                         generative=generative[:, :, z])
                            slicewise_metrics_results.append([metric, metric_value])
                        except Exception as e:
                            print(f'Issue computing metric {metric} for patient {uid}')
                            print(traceback.format_exc())
                    extra_metrics_results.append([["Slice", z]] + [[f'SW {x[0]}', x[1]] for x in slicewise_metrics_results])

        if "patientwise" in SharedResources.getInstance().validation_metric_spaces:
            patientwise_metrics_results = []
            metrics_values = [None] * len(SharedResources.getInstance().validation_generative_metrics)
            sw_metrics = SharedResources.getInstance().validation_generative_metrics
            if SharedResources.getInstance().number_processes > 1:
                try:
                    pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                    metrics_results = pool.map(parallel_metric_computation, zip(itertools.repeat(gt_norm),
                                                                                          itertools.repeat(generative),
                                                                                          sw_metrics,
                                                                                          metrics_values))
                    pool.close()
                    pool.join()
                    curr_pat_metrics = [["Slice", -1]]
                    for mm in metrics_results:
                        curr_pat_metrics.extend([[f"PW {mm[0]}", mm[1]]])
                    extra_metrics_results.append(curr_pat_metrics)
                except Exception as e:
                    print(f'Issue computing metrics for patient {uid}: {e}')
                    print(traceback.format_exc())
            else:
                for metric in sw_metrics:
                    try:
                        metric_value = compute_specific_metric_value(metric=metric, gt=gt_norm, generative=generative)
                        patientwise_metrics_results.append([metric, metric_value])
                    except Exception as e:
                        print(f'Issue computing metric {metric} for patient {uid}')
                        print(traceback.format_exc())
                extra_metrics_results.append([["Slice", -1]] + [[f'PW {x[0]}', x[1]] for x in patientwise_metrics_results])

            tm_metrics = SharedResources.getInstance().validation_generative_temporal_metrics
            for tm in tm_metrics:
                try:
                    metric_value = compute_specific_metric_value(metric=tm, gt=gt_norm, generative=generative)
                    patientwise_metrics_results.append([tm, metric_value])
                except Exception as e:
                    print(f'Issue computing metric {tm} for patient {uid}')
                    print(traceback.format_exc())
            extra_metrics_results.append([["Slice", -1]] + [[f'PW {x[0]}', x[1]] for x in patientwise_metrics_results])


        # Filling in the overall dataframe and dumping results to csv after each patient
        for res in extra_metrics_results:
            sub_df = self.results_df.loc[
                (self.results_df['Patient'] == uid) & (
                            self.results_df['Fold'] == fold_number) & (
                        self.results_df['Slice'] == res[0][1])]
            if len(sub_df) == 0:
                pat_metrics = np.asarray([None] * len(self.results_df_base_columns))
                pat_metrics[0] = fold_number
                pat_metrics[1] = uid
                pat_metrics[2] = res[0][1]
                buff_df = pd.DataFrame(pat_metrics.reshape(1, len(self.results_df_base_columns)),
                                       columns=list(self.results_df_base_columns))
                self.results_df = pd.concat([self.results_df, buff_df], ignore_index=True)
            for pm in res[1:]:
                metric_name = pm[0]
                metric_value = pm[1]
                self.results_df.at[self.results_df.loc[
                    (self.results_df['Patient'] == uid) & (
                            self.results_df['Fold'] == fold_number) & (
                                self.results_df['Slice'] == res[0][1])].index.values[
                    0], metric_name] = metric_value
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
