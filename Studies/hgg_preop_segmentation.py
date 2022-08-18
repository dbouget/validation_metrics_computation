import sys
import os
import pandas as pd
import numpy as np
from copy import deepcopy
import traceback
from Utils.resources import SharedResources
from Utils.io_converters import reload_optimal_validation_parameters
from Validation.validation_utilities import compute_fold_average
from Plotting.metric_versus_binned_boxplot import compute_binned_metric_over_metric_boxplot


class HGGPreopSegmentationStudy:
    """
    Study for segmenting high-grade gliomas (the core tumor) in T1 MRIs.
    """
    def __init__(self):
        self.study_name = SharedResources.getInstance().studies_study_name
        self.input_folder = os.path.join(SharedResources.getInstance().studies_input_folder, self.study_name)
        self.output_folder = SharedResources.getInstance().studies_output_folder
        self.metric_names = []
        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            self.extra_patient_parameters['Patient'] = self.extra_patient_parameters.Patient.astype(int).astype(str)

        if not os.path.exists(self.output_folder):
            self.output_folder = self.input_folder

        self.optimal_overlap = None
        self.optimal_threshold = None

    def run(self):
        self.__retrieve_optimum_values()
        self.__compute_and_plot_overall()
        if self.extra_patient_parameters is not None:
            self.__compute_and_plot_metric_over_metric_categories(metric1='Dice', metric2='Volume', metric2_cutoffs=[30.])

    def __retrieve_optimum_values(self):
        study_filename = os.path.join(self.input_folder, 'Validation', 'optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this.')

        self.optimal_overlap, self.optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)

    def __compute_and_plot_overall(self):
        """
        Generate average results across all folds and per fold for all the computed metrics.
        :return:
        """
        try:
            results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
            results_df = pd.read_csv(results_filename)
            results_df.replace('inf', np.nan, inplace=True)
            columns_to_drop = ['Fold', 'Patient', 'Threshold', 'Dice', '#GT', '#Det']
            columns = results_df.columns
            for elem in columns_to_drop:
                if elem in columns.values:
                    columns = columns.drop(elem)
            self.metric_names = list(columns.values)
            compute_fold_average(self.input_folder, data=results_df, best_threshold=self.optimal_threshold,
                                 best_overlap=self.optimal_overlap, metrics=self.metric_names)
            self.__compute_dice_confidence_intervals(data=results_df)

            if self.extra_patient_parameters is not None:
                self.__compute_results_metric_over_metric(data=results_df, metric1='Dice', metric2='Volume', suffix='')
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __compute_dice_confidence_intervals(self, data=None, suffix=''):
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 7:
            from Plotting.confidence_intervals_plot import compute_dice_confidence_intervals
            try:
                if data is None:
                    results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
                    results = pd.read_csv(results_filename)
                    results.replace('inf', np.nan, inplace=True)
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
                results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
                results = pd.read_csv(results_filename)
                results.replace('inf', np.nan, inplace=True)
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
                                                       on="Patient", how='left') #how='outer'

            folder = os.path.join(self.input_folder, 'Validation', metric2 + '-Wise')
            os.makedirs(folder, exist_ok=True)
            compute_binned_metric_over_metric_boxplot(folder=folder, data=optimal_results_per_patient,
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
                compute_binned_metric_over_metric_boxplot(folder=fold_folder, data=fold_optimal_results,
                                                          metric1=metric1, metric2=metric2,
                                                          criterion1=self.optimal_overlap,
                                                          postfix='_fold' + str(f) + suffix, number_bins=10)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __compute_and_plot_metric_over_metric_categories(self, data=None, metric1='Dice', metric2='Volume',
                                                         metric2_cutoffs=None, suffix=''):
        try:
            if data is None:
                results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
                results = pd.read_csv(results_filename)
                results.replace('inf', np.nan, inplace=True)
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
