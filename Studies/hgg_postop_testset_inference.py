import sys
import os
import multiprocessing
import itertools
import traceback
import pandas as pd
import numpy as np
from copy import deepcopy
import traceback
from Utils.io_converters import reload_optimal_validation_parameters
from Plotting.metric_versus_binned_boxplot import compute_binned_metric_over_metric_boxplot_postop
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from Validation.instance_segmentation_validation import *
from Utils.io_converters import get_fold_from_file
from Utils.volume_utilities import compute_tumor_volume
from Validation.validation_utilities import compute_fold_average
from Validation.extra_metrics_computation import compute_extra_metrics, compute_overall_metrics_correlation
from Validation.kfold_model_validation import separate_dice_computation
from Studies.hgg_postop_segmentation import threshold_volume_and_compute_classification_metrics, plot_classification_metrics_volume_cutoffs
from pyCompare import blandAltman


class HGGPostopTestsetInference:
    """
    Study for residual tumor segmentation from high-grade gliomas (the core tumor) in T1 MRIs.
    """
    def __init__(self):
        self.data_root = SharedResources.getInstance().data_root
        self.study_name = SharedResources.getInstance().studies_study_name

        self.input_folder = SharedResources.getInstance().test_input_folder
        self.base_output_folder = SharedResources.getInstance().test_output_folder
        if self.base_output_folder is not None and self.base_output_folder != "":
            self.output_folder = os.path.join(self.base_output_folder, 'Test')
        else:
            self.output_folder = os.path.join(self.input_folder, 'Test')
        os.makedirs(self.output_folder, exist_ok=True)

        self.fold_number = SharedResources.getInstance().test_nb_folds
        self.metric_names = ['Dice', 'Inst DICE', 'Inst Recall', 'Inst Precision', 'Largest foci Dice']
        self.metric_names.extend(SharedResources.getInstance().test_metric_names)
        self.detection_overlap_thresholds = SharedResources.getInstance().test_detection_overlap_thresholds
        print("Detection overlap: ", self.detection_overlap_thresholds)
        self.gt_files_suffix = SharedResources.getInstance().test_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().test_prediction_files_suffix
        self.exclude_ids = SharedResources.getInstance().test_exclude_ids
        if len(self.exclude_ids) > 0:
            try:
                exclude_ids = [int(i) for i in self.exclude_ids]
                self.exclude_ids = exclude_ids
            except Exception as e:
                print(f"Convert patient IDs to int failed, error {e}")

        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            try:
                self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(int).astype(str)
            except Exception as e:
                print(f"Convert patient IDs to int failed, error {e}")
            self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(str)
        self.optimal_overlap = None
        self.optimal_threshold = None

    def run(self):
        self.__generate_dice_scores()
        self.__retrieve_optimum_values()
        self.__read_results()
        self.__drop_exclude_ids()
        self.__compute_and_plot_overall()

        if self.extra_patient_parameters is not None:
            self.__compute_and_plot_metric_over_metric_categories(data=self.results, metric1='Dice', metric2='True postop volume', metric2_cutoffs=[1.])
            volume_figure_fname = Path(self.output_folder, 'volume_cutoff.png')
            self.__compute_and_plot_volume_cutoff_results(volume_cutoff_range=[0., 0.5], optimal_cutoff=0.175, save_fname=volume_figure_fname)
            results_cutoff = self.__compute_results_cutoff_volume(cutoff_volume=0.175)
            results_cutoff.to_csv(Path(self.output_folder, 'all_dice_scores_volume_cutoff.csv'), index=False)
            compute_fold_average(self.input_folder, results_cutoff, best_threshold=self.optimal_threshold,
                                 best_overlap=self.optimal_overlap,
                                 suffix='volume_cutoff', output_folder=self.output_folder)
            results_cutoff = self.__compute_EOR(results_cutoff, crop_to_zero=True)
            self.__study_volume_and_EOR(results_cutoff)

    def __retrieve_optimum_values(self):
        study_filename = os.path.join(self.input_folder, 'Validation', 'optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this.')

        self.optimal_overlap, self.optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)

    def __read_results(self):
        try:
            results_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
            results = pd.read_csv(results_filename)
            dice_thresholds = [np.round(x, 1) for x in list(np.unique(results['Threshold'].values))]
            nb_thresholds = len(dice_thresholds)
            optimal_threshold_index = dice_thresholds.index(self.optimal_threshold)
            optimal_results_per_patient = results[optimal_threshold_index::nb_thresholds]
            #best_dices_per_patient = results['Dice'].values[optimal_threshold_index::nb_thresholds]

            if self.extra_patient_parameters is not None:
                optimal_results_per_patient.loc[:, 'Patient'] = optimal_results_per_patient.Patient.astype(str)
                optimal_results_per_patient = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                                       on="Patient", how='left')

            self.results = results
            self.optimal_results = optimal_results_per_patient
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __drop_exclude_ids(self):
        drop_index_res = self.results[self.results['Patient'].isin(self.exclude_ids)].index
        self.results.drop(drop_index_res, axis=0, inplace=True)

        drop_index_opt = self.optimal_results[self.optimal_results['Patient'].isin(self.exclude_ids)].index
        self.optimal_results.drop(drop_index_opt, axis=0, inplace=True)

    def __compute_and_plot_overall(self):
        """
        Generate average results across all folds and per fold for all the computed metrics.
        :return:
        """
        try:
            # results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
            # results_df = pd.read_csv(results_filename)
            results_df = deepcopy(self.results)
            columns_to_drop = ['Fold', 'Patient', 'Threshold', 'Dice', '#GT', '#Det']
            columns = results_df.columns
            for elem in columns_to_drop:
                if elem in columns.values:
                    columns = columns.drop(elem)
            self.metric_names = list(columns.values)
            compute_fold_average(self.input_folder, data=results_df, best_threshold=self.optimal_threshold,
                                 best_overlap=self.optimal_overlap, metrics=self.metric_names,
                                 output_folder=self.output_folder)
            self.__compute_dice_confidence_intervals(data=results_df)

            if self.extra_patient_parameters is not None:
                self.__compute_results_metric_over_metric(data=results_df, metric1='Dice', metric2='True postop volume', suffix='')
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __compute_dice_confidence_intervals(self, data=None, suffix=''):
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 7:
            from Plotting.confidence_intervals_plot import compute_dice_confidence_intervals
            try:
                if data is None:
                    results_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
                    results = pd.read_csv(results_filename)
                else:
                    results = deepcopy(data)
                dice_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
                nb_tresholds = len(dice_thresholds)
                optimal_thresold_index = dice_thresholds.index(self.optimal_threshold)
                best_dices_per_patient = results['Dice'].values[optimal_thresold_index::nb_tresholds]
                compute_dice_confidence_intervals(folder=self.input_folder, dices=best_dices_per_patient,
                                                  postfix='_overall' + suffix, best_overlap=self.optimal_overlap)
            except Exception as e:
                print('{}'.format(traceback.format_exc()))
        else:
            print('Confidence intervals can only be computed with a Python version > 3.7.0, current version is {}.\n'.format(str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '.' + str(sys.version_info[2])))

    def __compute_results_metric_over_metric(self, data=None, metric1='Dice', metric2='Volume', suffix=''):
        try:
            if data is None:
                results_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
                results = pd.read_csv(results_filename)
            else:
                results = deepcopy(data)

            if self.extra_patient_parameters is None:
                return

            total_thresholds = [np.round(x, 1) for x in list(np.unique(results['Threshold'].values))]
            nb_thresholds = len(np.unique(results['Threshold'].values))
            optimal_thresold_index = total_thresholds.index(self.optimal_threshold)
            optimal_results_per_patient = results[optimal_thresold_index::nb_thresholds]
            # Not elegant, but either the two files have been merged before or not, so this test should be sufficient.
            if True in [x not in list(results.columns) for x in list(self.extra_patient_parameters.columns)]:
                optimal_results_per_patient.loc[:, 'Patient'] = optimal_results_per_patient.Patient.astype(str)
                optimal_results_per_patient = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                                       on="Patient", how='left')
            results = deepcopy(self.results)
            optimal_results_per_patient = deepcopy(self.optimal_results)
            folder = os.path.join(self.output_folder, metric2 + '-Wise')
            os.makedirs(folder, exist_ok=True)
            compute_binned_metric_over_metric_boxplot_postop(folder=folder, data=optimal_results_per_patient,
                                                      metric1=metric1, metric2=metric2, criterion1=self.optimal_overlap,
                                                      postfix='_overall' + suffix, number_bins=10)

            # Fold-wise analysis #
            fold_base_folder = os.path.join(folder, 'fold_analysis')
            os.makedirs(fold_base_folder, exist_ok=True)

            existing_folds = np.unique(results['Fold'].values)
            for f, fold in enumerate(existing_folds):
                results_fold = results.loc[results['Fold'] == fold]
                optimal_results_per_patient = results_fold[optimal_thresold_index::nb_thresholds]
                if True in [x not in list(results.columns) for x in list(self.extra_patient_parameters.columns)]:
                    optimal_results_per_patient.loc[:, 'Patient'] = optimal_results_per_patient.Patient.astype(str)
                    # Trick to only keep extra information for patients from the current fold with the 'how' attribute
                    fold_optimal_results = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                                    on="Patient", how='left')
                else:
                    fold_optimal_results = optimal_results_per_patient

                fold_folder = os.path.join(fold_base_folder, str(f))
                os.makedirs(fold_folder, exist_ok=True)
                compute_binned_metric_over_metric_boxplot_postop(folder=fold_folder, data=fold_optimal_results,
                                                          metric1=metric1, metric2=metric2,
                                                          criterion1=self.optimal_overlap,
                                                          postfix='_fold' + str(f) + suffix, number_bins=10)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __compute_and_plot_metric_over_metric_categories(self, data=None, metric1='Dice', metric2='Volume',
                                                         metric2_cutoffs=None, suffix=''):
        try:
            if data is None:
                results_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
                results = pd.read_csv(results_filename)
            else:
                results = deepcopy(data)
            total_thresholds = [np.round(x, 1) for x in list(np.unique(results['Threshold'].values))]
            nb_thresholds = len(np.unique(results['Threshold'].values))
            optimal_thresold_index = total_thresholds.index(self.optimal_threshold)
            optimal_results_per_patient = results[optimal_thresold_index::nb_thresholds]
            optimal_results_per_patient['Patient'] = optimal_results_per_patient.Patient.astype(str)
            total_optimal_results = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                             on="Patient", how='left')

            if not metric1 in list(total_optimal_results.columns) or not metric2 in list(total_optimal_results.columns):
                print('The required metric is missing from the DataFrame with either {} or {}. Skipping.\n'.format(metric1, metric2))
                return

            optimal_results_per_cutoff = {}
            for c, cutoff in enumerate(metric2_cutoffs):
                if c == 0:
                    cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] <= cutoff]
                    optimal_results_per_cutoff['<=' + str(cutoff)] = cat_optimal_results
                else:
                    cat_optimal_results = total_optimal_results.loc[metric2_cutoffs[c-1] < total_optimal_results[metric2] <= cutoff]
                    optimal_results_per_cutoff[']' + str(metric2_cutoffs[c-1]) + ',' + str(cutoff) + ']'] = cat_optimal_results
            cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] > metric2_cutoffs[-1]]
            optimal_results_per_cutoff['>' + str(metric2_cutoffs[-1])] = cat_optimal_results

            for category in optimal_results_per_cutoff.keys():
                self.__compute_dice_confidence_intervals(data=optimal_results_per_cutoff[category],
                                                         suffix=suffix + '_' + metric2 + '_' + category)
                self.__compute_results_metric_over_metric(data=optimal_results_per_cutoff[category], metric1=metric1,
                                                          metric2=metric2,
                                                          suffix=suffix + '_' + metric2 + '_' + category)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __compute_and_plot_volume_cutoff_results(self, volume_cutoff_range=[0., 5.], n_cutoff_steps=100,
                                                 cutoff_gt = True, optimal_cutoff = None, save_fname=None):
        try:
            cutoff_range = np.arange(volume_cutoff_range[0], volume_cutoff_range[1], (volume_cutoff_range[1]-volume_cutoff_range[0])/n_cutoff_steps)
            results = {}
            for cutoff in cutoff_range:
                res = threshold_volume_and_compute_classification_metrics(self.optimal_results, cutoff, cutoff if cutoff_gt else 0.0)
                if not len(results):
                    for k, v in res.items():
                        results[k] = [v]
                else:
                    for k, v in res.items():
                        results[k].append(v)

                output = f"Cutoff = {cutoff}, recall = {res['Recall']}, precision = {res['Precision']}, "\
                         f"F1 = {res['F1']}, accuracy = {res['Accuracy']}, tnr = {res['Specificity']}, "\
                         f"positive rate = {res['Positive rate']}, negative rate = {res['Negative rate']}"
                print(output)

            plot_classification_metrics_volume_cutoffs(results, cutoff_range, 'first_plot', None,
                                                       metrics_to_plot=['Recall', 'Precision', 'F1', 'Accuracy',
                                                                        'Dice Positive', 'Dice True Positive'],
                                                       optimal_cutoff=optimal_cutoff,
                                                       metrics_to_maximize=['Accuracy', 'F1', 'Recall'],
                                                       save_fname=save_fname)


        except Exception as e:
            print('{}'.format(traceback.format_exc()))

        return

    def __compute_results_cutoff_volume(self, cutoff_volume, cutoff_gt=True):
        data = deepcopy(self.optimal_results)
        data.loc[(data['Predicted postop volume'] <= cutoff_volume), 'Predicted postop volume'] = 0.0
        data.loc[(data['Predicted postop volume'] <= cutoff_volume), '#Det'] = 0

        if cutoff_gt:
            data.loc[(data['True postop volume'] <= cutoff_volume), 'True postop volume'] = 0.0
            data.loc[(data['True postop volume'] <= 0), '#GT'] = 0

        # Drop columns missing volume
        data.dropna(axis=0, subset=['True postop volume'], inplace=True)

        return data

    def __generate_dice_scores(self):
        """
        Generate the Dice scores (and default instance detection metrics) for all the patients and 10 probability
        thresholds equally-spaced. All the computed results will be stored inside all_dice_scores.csv.
        The results are saved after each patient, making it possible to resume the computation if a crash occurred.
        @TODO. Include an override flag to recompute anyway.
        :return:
        """
        cross_validation_description_file = os.path.join(self.input_folder, 'VUmc_external_test_set_fivefolds.txt')
        self.results_df = []
        self.dice_output_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
        if not os.path.exists(self.dice_output_filename):
            self.results_df = pd.DataFrame(columns=['Fold', 'Patient', 'Threshold', 'Dice', 'Inst DICE', 'Inst Recall',
                                                    'Inst Precision', 'Largest foci Dice', '#GT', '#Det',
                                                    'Predicted postop volume'])
        else:
            self.results_df = pd.read_csv(self.dice_output_filename)
            if self.results_df.columns[0] != 'Fold':
                self.results_df = pd.read_csv(self.dice_output_filename, index_col=0)
        self.results_df['Patient'] = self.results_df.Patient.astype(str)

        results_per_folds = []

        print('\nProcessing test set (fold 0), generating dice scores.\n')
        test_set, _ = get_fold_from_file(filename=cross_validation_description_file, fold_number=0)

        results = self.__generate_dice_scores_for_fold(data_list=test_set, fold_number=0)
        results_per_folds.append(results)

    def __generate_dice_scores_for_fold(self, data_list, fold_number):
        data_list = [d.replace('\n', '') for d in data_list]
        for i, patient in enumerate(tqdm(data_list)):
            uid = None
            try:
                # @TODO. Hard-coded, have to decide on naming convention....
                # start = time.time()
                uid = patient.split('_')[1]
                sub_folder_index = patient.split('_')[0]
                patient_extended = '_'.join(patient.split('_')[1:]).strip().replace('_sample', '')

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

                ground_truth_base = os.path.join(self.data_root, sub_folder_index, uid, 'segmentations',
                                                 patient_extended)
                ground_truth_filename = None
                for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                    for f in files:
                        if os.path.basename(ground_truth_base) in f and self.gt_files_suffix in f:
                            ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                    break
                detection_filename = os.path.join(self.input_folder, 'test_predictions', str(fold_number),
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
                    sub_df = self.results_df.loc[
                        (self.results_df['Patient'] == uid) & (self.results_df['Fold'] == fold_number) & (
                                    self.results_df['Threshold'] == th)]
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

    def __compute_EOR(self, results, crop_to_zero=False):
        data = deepcopy(results)

        true_EOR = np.array((results.loc[:, 'True preop volume'] - results.loc[:, 'True postop volume']) / results.loc[:, 'True preop volume'])
        data['True EOR'] = true_EOR
        #print(true_EOR[np.where(true_EOR<0)])
        #print(data.loc[np.where(true_EOR < 0), 'Patient'].values)

        predicted_EOR_type1 = np.array((results.loc[:, 'True preop volume'] - results.loc[:, 'Predicted postop volume']) / results.loc[:, 'True preop volume'])
        data['Predicted EOR type 1'] = predicted_EOR_type1
        #print(predicted_EOR_type1[np.where(predicted_EOR_type1 < 0)])
        #print(data.loc[np.where(predicted_EOR_type1 < 0), 'Patient'].values)

        if crop_to_zero:
            data.loc[(data['True EOR'] < 0), 'True EOR'] = 0.0
            data.loc[(data['Predicted EOR type 1'] < 0), 'Predicted EOR type 1'] = 0.0

        return data

    def __study_volume_and_EOR(self, results):
        data = deepcopy(results)
        output_folder = Path(self.output_folder, 'Volume-EOR')
        output_folder.mkdir(exist_ok=True)
        sns.set_style('ticks')

        # EOR
        save_fname = str(Path(output_folder, f'EOR_scatter_{self.study_name}_Test.png'))
        plt.figure()
        ax = sns.scatterplot(data=data*100, x='True EOR', y='Predicted EOR type 1')
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1")
        ax.set(title=f'True vs predicted EOR', xlabel='True EOR (%)', ylabel='Predicted EOR (%)')
        plt.savefig(save_fname)

        save_fname = str(Path(output_folder, f'EOR_bland_altman_{self.study_name}_Test.png'))
        blandAltman(data['True EOR'], data['Predicted EOR type 1'],
                    title=f'Bland-Altman of true vs predicted EOR', savePath=save_fname)

        # VOLUME
        save_fname = str(Path(output_folder, f'volume_scatter_{self.study_name}_Test.png'))
        plt.figure()
        plt.xscale('symlog')
        plt.yscale('symlog')
        ax = sns.scatterplot(data=data, x='True postop volume', y='Predicted postop volume')
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1")
        #ax.title('True vs predicted postop volume (symlog scale)')
        ax.set(title=f'True vs predicted postop volume (symlog scale)',
               xlabel='True postop volume (ml)', ylabel='Predicted postop volume (ml)')
        plt.savefig(save_fname)

        save_fname = str(Path(output_folder, f'volume_bland_altman_{self.study_name}_Test.png'))
        blandAltman(data['True postop volume'], data['Predicted postop volume'],
                    title=f'Bland-Altman of true vs predicted postop volume', savePath=save_fname)
