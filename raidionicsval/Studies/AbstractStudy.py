import sys
import os
import logging
import pandas as pd
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import List
import traceback
import math
from ..Utils.resources import SharedResources
from ..Utils.io_converters import reload_optimal_validation_parameters
from ..Plotting.metric_versus_binned_boxplot import compute_binned_metric_over_metric_boxplot
from ..Validation.validation_utilities import compute_patientwise_fold_metrics, compute_singe_fold_average_metrics
from ..Validation.extra_metrics_computation import compute_overall_metrics_correlation
from ..Utils.latex_converter import export_segmentation_df_to_latex_paper


class AbstractStudy(ABC):
    """

    """
    _class_names = None
    _classes_optimal = {}  # Optimal probability threshold and overlap values, for each class, based on the analysis from the validation round

    def __init__(self):
        self.input_folder = SharedResources.getInstance().studies_input_folder
        self.output_folder = SharedResources.getInstance().studies_output_folder
        self.metric_names = []
        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            self.extra_patient_parameters['Patient'] = self.extra_patient_parameters.Patient.astype(str)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self._class_names = SharedResources.getInstance().studies_class_names
        self._classes_optimal = {}

        if SharedResources.getInstance().overall_objective == "segmentation":
            for c in self.class_names:
                self.__retrieve_optimum_values(c)

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def classes_optimal(self) -> dict:
        return self._classes_optimal

    @abstractmethod
    def run(self):
        """

        """
        pass

    def __retrieve_optimum_values(self, class_name: str):
        study_filename = os.path.join(self.input_folder, 'Validation', class_name + '_optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        self.classes_optimal[class_name] = {}
        self.classes_optimal[class_name]['All'] = [optimal_overlap, optimal_threshold]

        study_filename = os.path.join(self.input_folder, 'Validation', class_name + '_optimal_dice_study_tp.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study_tp file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        self.classes_optimal[class_name]['True Positive'] = [optimal_overlap, optimal_threshold]

    def compute_and_plot_overall(self, class_name: str, category: str = 'All') -> None:
        """
        Generate average results across all folds and per fold for all the computed metrics.

        :param class_name: Name of the class of interest
        :param category: Population to patients to focus on, from ['All', 'True Positive']. The threshold for
        disambiguating between true and false positive patients was set during the validation phase.

        :return:
        """
        try:
            results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
            results_df = pd.read_csv(results_filename)
            results_df.replace('inf', np.nan, inplace=True)
            optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
            if category != 'All':
                results_df = results_df.loc[results_df["True Positive"] == True]
                if category == "TP":
                    optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else self.classes_optimal[class_name]['True Positive'][0]
                    results_df = results_df.loc[(results_df["Threshold"] == optimal_threshold) & (results_df["PiW Dice"] >= optimal_overlap)]
                else:
                    results_df = results_df.loc[results_df["Threshold"] == optimal_threshold]
            else:
                results_df = results_df.loc[results_df["Threshold"] == optimal_threshold]

            self.metric_names = list(results_df.columns)[17:]  # Hard-coded, needs to be improved.
            self.__compute_dice_confidence_intervals(data=results_df, class_name=class_name, category=category)
            self.__compute_results_metric_over_metric(data=results_df, class_name=class_name, metric1='PiW Dice',
                                                      metric2='GT volume (ml)', category=category, suffix='_')
            self.__compute_agreement_blant_altman(data=results_df, class_name=class_name, category=category)
            export_segmentation_df_to_latex_paper(folder=self.input_folder, class_name=class_name, suffix=category)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def compute_and_plot_metrics_correlation_matrix(self, class_name: str, category: str = 'All') -> None:
        """

        :param class_name: Name of the class of interest
        :param category: Population to patients to focus on, from ['All', 'True Positive']. The threshold for
        disambiguating between true and false positive patients was set during the validation phase.

        :return:
        """
        try:
            results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
            results_df = pd.read_csv(results_filename)
            results_df.replace('inf', np.nan, inplace=True)
            suffix = '' if category == "All" else '_' + category
            optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
                self.classes_optimal[class_name]['True Positive'][0]
            optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
                self.classes_optimal[class_name]['True Positive'][1]
            if category != 'All':
                if category != 'All':
                    results_df = results_df.loc[results_df["True Positive"] == True]
                    if category == "TP":
                        results_df = results_df.loc[(results_df["Threshold"] == optimal_threshold) & (
                                results_df["PiW Dice"] >= optimal_overlap)]
                    else:
                        results_df = results_df.loc[results_df["Threshold"] == optimal_threshold]
                else:
                    results_df = results_df.loc[results_df["Threshold"] == optimal_threshold]

            compute_overall_metrics_correlation(self.input_folder, self.output_folder, data=results_df,
                                                class_name=class_name, best_threshold=optimal_threshold,
                                                best_overlap=optimal_overlap, suffix=suffix)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __compute_dice_confidence_intervals(self, class_name: str, data=None, category: str = 'All',
                                            study_name: str = "", suffix=''):
        """

        :param class_name:
        :param data:
        :param category:
        :param suffix:
        :return:
        """
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 7:
            from raidionicsval.Plotting.confidence_intervals_plot import compute_dice_confidence_intervals
            try:
                filename_extra = '' if category == 'All' else '_' + category
                if data is None:
                    results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
                    results = pd.read_csv(results_filename)
                    results.replace('inf', np.nan, inplace=True)
                    optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
                        self.classes_optimal[class_name]['True Positive'][1]
                    if category != 'All':
                        results = results.loc[results["True Positive"] == True]
                        if category == "TP":
                            optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
                            self.classes_optimal[class_name]['True Positive'][0]
                            results = results.loc[(results["Threshold"] == optimal_threshold) & (
                                        results["PiW Dice"] >= optimal_overlap)]
                    else:
                        results = results.loc[results["Threshold"] == optimal_threshold]
                else:
                    results = deepcopy(data)

                # dice_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
                # nb_tresholds = len(dice_thresholds)
                # optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
                # optimal_threshold_index = dice_thresholds.index(optimal_threshold)
                best_dices_per_patient = results['PiW Dice'].values #[optimal_threshold_index::nb_tresholds]
                optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else self.classes_optimal[class_name]['True Positive'][0]
                output_folder = self.output_folder
                if study_name != "":
                    output_folder = os.path.join(self.output_folder, study_name)
                compute_dice_confidence_intervals(folder=output_folder, dices=best_dices_per_patient,
                                                  postfix='_overall' + suffix + '_' + class_name + filename_extra,
                                                  best_overlap=optimal_overlap)
            except Exception as e:
                print('{}'.format(traceback.format_exc()))
        else:
            print('Confidence intervals can only be computed with a Python version > 3.7.0, current version is {}.\n'.format(str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '.' + str(sys.version_info[2])))

    def __compute_agreement_blant_altman(self, class_name: str, data=None, category: str = 'All', suffix=''):
        """

        :param class_name:
        :param data:
        :param category:
        :param suffix:
        :return:
        """
        from raidionicsval.Plotting.agreement_plot import compute_agreement_plot
        try:
            filename_extra = '' if category == 'All' else '_' + category
            if data is None:
                results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
                results = pd.read_csv(results_filename)
                results.replace('inf', np.nan, inplace=True)
                optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
                    self.classes_optimal[class_name]['True Positive'][1]
                if category != 'All':
                    results = results.loc[results["True Positive"] == True]
                    if category == "TP":
                        optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
                            self.classes_optimal[class_name]['True Positive'][0]
                        results = results.loc[(results["Threshold"] == optimal_threshold) & (
                                results["PiW Dice"] >= optimal_overlap)]
                else:
                    results = results.loc[results["Threshold"] == optimal_threshold]
            else:
                results = deepcopy(data)
            # dice_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
            # nb_tresholds = len(dice_thresholds)
            # optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
            # optimal_threshold_index = dice_thresholds.index(optimal_threshold)

            gt_volumes = results['GT volume (ml)'].values #[optimal_threshold_index::nb_tresholds]
            det_volumes = results['Detection volume (ml)'].values #[optimal_threshold_index::nb_tresholds]
            compute_agreement_plot(folder=self.output_folder, array1=gt_volumes, array2=det_volumes,
                                              postfix='_overall' + suffix + '_' + class_name + filename_extra)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))


    def __compute_results_metric_over_metric(self, class_name: str, data=None, metric1='PiW Dice',
                                             metric2='GT Volume (ml)', category: str = 'All', study_name: str = "",
                                             suffix=''):
        try:
            filename_extra = '' if category == 'All' else '_' + category
            if data is None:
                results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
                results = pd.read_csv(results_filename)
                results.replace('inf', np.nan, inplace=True)
                optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
                    self.classes_optimal[class_name]['True Positive'][1]
                if category != 'All':
                    results = results.loc[results["True Positive"] == True]
                    if category == "TP":
                        optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
                            self.classes_optimal[class_name]['True Positive'][0]
                        results = results.loc[(results["Threshold"] == optimal_threshold) & (
                                results["PiW Dice"] >= optimal_overlap)]
                else:
                    results = results.loc[results["Threshold"] == optimal_threshold]
            else:
                results = deepcopy(data)

            # if self.extra_patient_parameters is None:
            #     return

            # if category != 'All':
            #     results = results.loc[results["True Positive"] == True]

            number_bins = 10
            if metric2 == "SpacZ":
                number_bins = 5
            # total_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
            # nb_thresholds = len(np.unique(results['Threshold'].values))
            # optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
            # optimal_thresold_index = total_thresholds.index(optimal_threshold)
            # optimal_results_per_patient = results[optimal_thresold_index::nb_thresholds]
            optimal_results_per_patient = results
            # Not elegant, but either the two files have been merged before or not, so this test should be sufficient.
            if self.extra_patient_parameters is not None:
                if True in [x not in list(results.columns) for x in list(self.extra_patient_parameters.columns)]:
                    optimal_results_per_patient['Patient'] = optimal_results_per_patient.Patient.astype(str)
                    optimal_results_per_patient = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                                           on="Patient", how='left') #how='outer'

            if study_name == "":
                study_name = metric1.replace(" ", "") + '_' + metric2.replace(" ", "") + '-Wise'
            folder = os.path.join(self.output_folder, study_name)
            os.makedirs(folder, exist_ok=True)
            optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else self.classes_optimal[class_name]['True Positive'][0]
            compute_binned_metric_over_metric_boxplot(folder=folder, data=optimal_results_per_patient,
                                                      metric1=metric1, metric2=metric2,
                                                      criterion1=optimal_overlap,
                                                      postfix='_overall' + suffix + '_' + class_name + filename_extra,
                                                      number_bins=number_bins)

            # Fold-wise analysis #
            fold_base_folder = os.path.join(folder, 'fold_analysis')
            os.makedirs(fold_base_folder, exist_ok=True)

            existing_folds = np.unique(results['Fold'].values)
            for f, fold in enumerate(existing_folds):
                results_fold = results.loc[results['Fold'] == fold]
                optimal_results_per_patient = results_fold #[optimal_thresold_index::nb_thresholds]
                if self.extra_patient_parameters is not None:
                    if True in [x not in list(results.columns) for x in list(self.extra_patient_parameters.columns)]:
                        optimal_results_per_patient['Patient'] = optimal_results_per_patient.Patient.astype(str)
                        # Trick to only keep extra information for patients from the current fold with the 'how' attribute
                        fold_optimal_results = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                                        on="Patient", how='left')
                    else:
                        fold_optimal_results = optimal_results_per_patient
                else:
                    fold_optimal_results = optimal_results_per_patient

                fold_folder = os.path.join(fold_base_folder, str(f))
                os.makedirs(fold_folder, exist_ok=True)
                compute_binned_metric_over_metric_boxplot(folder=fold_folder, data=fold_optimal_results,
                                                          metric1=metric1, metric2=metric2,
                                                          criterion1=optimal_overlap,
                                                          postfix='_fold' + str(f) + suffix + '_' + class_name + filename_extra,
                                                          number_bins=number_bins)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def compute_and_plot_metric_over_metric_categories(self, class_name: str, metric1='PiW Dice',
                                                       metric2='Volume', metric2_cutoffs=None, category='All',
                                                       suffix='') -> None:
        """
        Performs the computation and plotting of a dense metric against another dense metric.

        :param class_name: current class of the segmentation object of interest
        :param data: subset of data to only consider, i.e., a pd.DataFrame already reduced
        :param metric1: Name of the first metric (dense), as appearing in the class_dice_scores.csv file
        :param metric2: Name of the second metric (dense), as appearing in either the class_dice_scores.csv file or the studies_extra_parameters_filename
        :param metric2_cutoffs:
        :param category: subset to consider from [All, True Positive]
        :param suffix: text to be appended to the corresponding result filenames
        :return: Nothing is returned, and the corresponding results are saved on disk.
        """
        try:
            print("Computing and plotting {} over {} with the following cut-off values [{}].\n".format(metric1,
                                                                                                    metric2,
                                                                                                    metric2_cutoffs))
            # if data is None:
            results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
            results = pd.read_csv(results_filename)
            results.replace('inf', np.nan, inplace=True)
            optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
                self.classes_optimal[class_name]['True Positive'][1]
            if category != 'All':
                results = results.loc[results["True Positive"] == True]
                if category == "TP":
                    optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
                        self.classes_optimal[class_name]['True Positive'][0]
                    results = results.loc[(results["Threshold"] == optimal_threshold) & (
                            results["PiW Dice"] >= optimal_overlap)]
                else:
                    results = results.loc[results["Threshold"] == optimal_threshold]
            else:
                results = results.loc[results["Threshold"] == optimal_threshold]
            # total_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
            # nb_thresholds = len(np.unique(results['Threshold'].values))
            # optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
            # optimal_thresold_index = total_thresholds.index(optimal_threshold)
            optimal_results_per_patient = results #[optimal_thresold_index::nb_thresholds]
            optimal_results_per_patient['Patient'] = optimal_results_per_patient.Patient.astype(str)
            if self.extra_patient_parameters is not None:
                total_optimal_results = pd.merge(optimal_results_per_patient, self.extra_patient_parameters, on="Patient")
            else:
                total_optimal_results = optimal_results_per_patient

            if not metric1 in list(total_optimal_results.columns) or not metric2 in list(total_optimal_results.columns):
                print('The required metric is missing from the DataFrame with either {} or {}. Skipping.\n'.format(metric1, metric2))
                return

            optimal_results_per_cutoff = {}
            if metric2_cutoffs is None or len(metric2_cutoffs) == 0:
                optimal_results_per_cutoff['All'] = total_optimal_results
            else:
                for c, cutoff in enumerate(metric2_cutoffs):
                    if c == 0:
                        cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] <= cutoff]
                        optimal_results_per_cutoff['<=' + str(cutoff)] = cat_optimal_results
                    else:
                        cat_optimal_results = total_optimal_results.loc[metric2_cutoffs[c-1] < total_optimal_results[metric2] <= cutoff]
                        optimal_results_per_cutoff[']' + str(metric2_cutoffs[c-1]) + ',' + str(cutoff) + ']'] = cat_optimal_results
                cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] > metric2_cutoffs[-1]]
                optimal_results_per_cutoff['>' + str(metric2_cutoffs[-1])] = cat_optimal_results

            study_name = metric1.replace(" ", "") + "_" + metric2.replace(" ", "") + '-Wise'
            study_output_folder = os.path.join(self.output_folder, study_name)
            os.makedirs(study_output_folder, exist_ok=True)
            for cat in optimal_results_per_cutoff.keys():
                # @TODO. Must include a new fold average specific for the studies, with mean and std values as input,
                # which is different from the inputs to the computation in the validation part....
                if len(optimal_results_per_cutoff[cat]) == 0:
                    print("Skipping analysis for {} {}. Collected pd.DataFrame is empty.\n".format(metric2, cat))
                    return
                self.compute_fold_average(folder=study_output_folder, data=optimal_results_per_cutoff[cat],
                                          class_optimal=self.classes_optimal, metrics=self.metric_names,
                                          suffix=suffix + '_' + metric2 + '_' + cat,
                                          class_names=SharedResources.getInstance().studies_class_names,
                                          condition=category)
                self.__compute_dice_confidence_intervals(class_name=class_name,
                                                         category=category,
                                                         data=optimal_results_per_cutoff[cat],
                                                         study_name=study_name,
                                                         suffix=suffix + '_' + metric2 + '_' + cat)
                self.__compute_results_metric_over_metric(class_name=class_name,
                                                          data=optimal_results_per_cutoff[cat], metric1=metric1,
                                                          metric2=metric2,
                                                          category=category,
                                                          study_name=study_name,
                                                          suffix=suffix + '_' + metric2 + '_' + cat)
            suffix = category + '_' + metric2 + '_'
            export_segmentation_df_to_latex_paper(folder=self.output_folder, class_name=class_name, study=study_name,
                                                  categories=list(optimal_results_per_cutoff.keys()), suffix=suffix)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def compute_and_plot_categorical_metric_over_metric_categories(self, class_name: str, metric1,
                                                                   metric2, metric2_cutoffs=None,
                                                                   category='All', suffix='') -> None:
        """
        Performs the computation and plotting of a dense metric against a categorical metric.
        The categorical metric is expected to be expressed as strings inside the studies_extra_parameters_filename.
        For example, for a metric called MR_sequence, the values are expected to be [T1, T2, FLAIR].

        :param class_name: current class of the segmentation object of interest
        :param data: subset of data to only consider, i.e., a pd.DataFrame already reduced
        :param metric1: Name of the first metric (dense), as appearing in the class_dice_scores.csv file
        :param metric2: Name of the second metric (categorical), as appearing in the studies_extra_parameters_filename
        :param metric2_cutoffs:
        :param category: subset to consider from [All, True Positive]
        :param suffix: text to be appended to the corresponding result filenames
        :return: Nothing is returned, and the corresponding results are saved on disk.
        """
        try:
            print("Computing and plotting {} over {} with the following cut-off values [{}].\n".format(metric1,
                                                                                                       metric2,
                                                                                                       metric2_cutoffs))
            # if data is None:
            results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
            results = pd.read_csv(results_filename)
            results.replace('inf', np.nan, inplace=True)
            optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
                self.classes_optimal[class_name]['True Positive'][1]
            # if category != 'All':
            #     results = results.loc[results["True Positive"] == True]
            #     if category == "TP":
            #         optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
            #             self.classes_optimal[class_name]['True Positive'][0]
            #         results = results.loc[(results["Threshold"] == optimal_threshold) & (
            #                 results["PiW Dice"] >= optimal_overlap)]
            #     else:
            #         results = results.loc[results["Threshold"] == optimal_threshold]
            # else:
            #     results = results.loc[results["Threshold"] == optimal_threshold]

            # total_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
            # nb_thresholds = len(np.unique(results['Threshold'].values))
            # optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
            # optimal_thresold_index = total_thresholds.index(optimal_threshold)
            optimal_results_per_patient = results #[optimal_thresold_index::nb_thresholds]
            optimal_results_per_patient['Patient'] = optimal_results_per_patient.Patient.astype(str)
            if self.extra_patient_parameters is not None:
                total_optimal_results = pd.merge(optimal_results_per_patient, self.extra_patient_parameters, on="Patient")
            else:
                total_optimal_results = optimal_results_per_patient

            if not metric1 in list(total_optimal_results.columns) or not metric2 in list(total_optimal_results.columns):
                print('The required metric is missing from the DataFrame with either {} or {}. Skipping.\n'.format(metric1, metric2))
                return

            total_patients_nb = len(total_optimal_results)
            total_optimal_results = total_optimal_results.dropna(subset=[metric2])
            total_patients_nb_selected = len(total_optimal_results)
            if total_patients_nb != total_patients_nb_selected:
                print(f"A total of {total_patients_nb-total_patients_nb_selected} patients has been discarded for "
                      f"the analysis over {metric2} because missing info from the extra_parameters.csv file!")
            optimal_results_per_cutoff = {}
            if metric2_cutoffs is None or len(metric2_cutoffs) == 0:
                metric2_cutoffs = list(np.unique(total_optimal_results[metric2].values))

            for c, cutoff in enumerate(metric2_cutoffs):
                cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] == cutoff]
                optimal_results_per_cutoff[cutoff] = cat_optimal_results

            study_name = metric1.replace(" ", "-") + "_Versus_" + metric2.replace(" ", "-")
            study_output_folder = os.path.join(self.output_folder, study_name)
            os.makedirs(study_output_folder, exist_ok=True)
            for cat in optimal_results_per_cutoff.keys():
                # @TODO. Must include a new fold average specific for the studies, with mean and std values as input,
                # which is different from the inputs to the computation in the validation part....
                if len(optimal_results_per_cutoff[cat]) == 0:
                    print("Skipping analysis for {} {}. Collected pd.DataFrame is empty.\n".format(metric2, cat))
                    return
                metric2 = "GT volume (ml)"
                self.compute_fold_average(folder=study_output_folder, data=optimal_results_per_cutoff[cat],
                                          class_optimal=self.classes_optimal, metrics=self.metric_names,
                                          suffix=suffix + '_' + metric2 + '_' + cat, class_names=SharedResources.getInstance().studies_class_names,
                                          condition=category)
                self.__compute_dice_confidence_intervals(class_name=class_name,
                                                         category=category,
                                                         data=optimal_results_per_cutoff[cat],
                                                         study_name=study_name,
                                                         suffix=suffix + '_' + metric2 + '_' + cat)
                self.__compute_results_metric_over_metric(class_name=class_name,
                                                          data=optimal_results_per_cutoff[cat], metric1=metric1,
                                                          metric2=metric2,
                                                          category=category,
                                                          study_name=study_name,
                                                          suffix=suffix + '_' + metric2 + '_' + cat)
            suffix = category + '_' + metric2 + '_'
            export_segmentation_df_to_latex_paper(folder=self.output_folder, class_name=class_name, study=study_name,
                                                  categories=list(optimal_results_per_cutoff.keys()), suffix=suffix)
        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def compute_fold_average(self, folder, data=None, class_optimal={}, metrics=[], suffix='', class_names=None,
                             condition='All'):
        # @TODO. Should not collect the classes from validation_class_names, as it might differ from the studied classes.
        if class_names is None:
            classes = SharedResources.getInstance().validation_class_names
        else:
            classes = class_names
        optimal_tag = 'All' if condition == 'All' else 'True Positive'
        for c in classes:
            optimal_values = class_optimal[c][optimal_tag]
            self.compute_fold_average_inner(folder, data=data, class_name=c, best_threshold=optimal_values[1],
                                       best_overlap=optimal_values[0], metrics=metrics, suffix=suffix, condition=condition)

    def compute_fold_average_inner(self, folder, class_name, data=None, best_threshold=0.5, best_overlap=0.0, metrics=[],
                                   suffix='', condition='All'):
        """
        :param folder: Destination folder where the results will be dumped (as specified in the configuration file)
        :param best_threshold:
        :param best_overlap:
        :param metric_names:
        :return:
        """
        try:
            results = None
            if data is None:
                results_filename = os.path.join(folder, class_name + '_dice_scores.csv')
                results = pd.read_csv(results_filename)
                # optimal_overlap = self.classes_optimal[class_name]['All'][0] if condition == 'All' else \
                #     self.classes_optimal[class_name]['True Positive'][0]
                # optimal_threshold = self.classes_optimal[class_name]['All'][1] if condition == 'All' else \
                #     self.classes_optimal[class_name]['True Positive'][1]
                # if condition != 'All':
                #     if condition != 'All':
                #         results = results.loc[results["True Positive"] == True]
                #         if condition == "TP":
                #             results = results.loc[(results["Threshold"] == optimal_threshold) & (
                #                     results["PiW Dice"] >= optimal_overlap)]
                #     else:
                #         results = results.loc[results["Threshold"] == optimal_threshold]
            else:
                results = deepcopy(data)

            suffix = condition + suffix
            results.replace('inf', np.nan, inplace=True)
            results.replace(float('inf'), np.nan, inplace=True)
            results.replace('', np.nan, inplace=True)
            results.replace(' ', np.nan, inplace=True)
            unique_folds = np.unique(results['Fold'])
            nb_folds = len(unique_folds)
            metrics_per_fold = []
            tp_volume_threshold = 0.
            if len(SharedResources.getInstance().validation_true_positive_volume_thresholds) == 1:
                tp_volume_threshold = SharedResources.getInstance().validation_true_positive_volume_thresholds[0]
            else:
                index_cl = SharedResources.getInstance().validation_class_names.find(class_name)
                tp_volume_threshold = SharedResources.getInstance().validation_true_positive_volume_thresholds[index_cl]

            metric_names = list(results.columns[3:list(results.columns).index("#Det") + 1])
            if metrics is not None:
                metric_names.extend(metrics)

            fold_average_columns = ['Fold', '# samples', 'Patient-wise recall', 'Patient-wise precision',
                                    'Patient-wise specificity',
                                    'Patient-wise F1', 'Patient-wise Accuracy', 'Patient-wise Balanced accuracy']
            for m in metric_names:
                fold_average_columns.extend([m + ' (Mean)', m + ' (Std)'])

            # Regarding the overlap threshold, should the patient discarded for recall be
            # used for other metrics computation?
            for f in unique_folds:
                patientwise_metrics = compute_patientwise_fold_metrics(results, f, best_threshold, best_overlap,
                                                                       tp_volume_threshold, condition)
                fold_average_metrics, fold_std_metrics = compute_singe_fold_average_metrics(results, f, best_threshold,
                                                                                            best_overlap, metrics, tp_volume_threshold, condition)
                fold_metrics = []
                for m in range(len(fold_average_metrics)):
                    fold_metrics.append(fold_average_metrics[m])
                    fold_metrics.append(fold_std_metrics[m])
                fold_average = [f, len(np.unique(
                    results.loc[results['Fold'] == f]['Patient'].values))] + patientwise_metrics + fold_metrics
                metrics_per_fold.append(fold_average)

            metrics_per_fold_df = pd.DataFrame(data=metrics_per_fold, columns=fold_average_columns)
            study_filename = os.path.join(folder, class_name + '_folds_metrics_average.csv') if suffix == '' else\
                os.path.join(folder, class_name + '_folds_metrics_average_' + suffix + '.csv')
            metrics_per_fold_df.to_csv(study_filename, index=False)

            ####### Averaging the results from the different folds ###########
            total_samples = metrics_per_fold_df['# samples'].sum()
            patientwise_fold_metrics_to_average = metrics_per_fold_df.values[:, 2:8]
            fold_metrics_to_average = metrics_per_fold_df.values[:, 8:][:, 0::2]
            fold_std_metrics_to_average = metrics_per_fold_df.values[:, 8:][:, 1::2]
            total_fold_metrics_to_average = np.concatenate((patientwise_fold_metrics_to_average, fold_metrics_to_average),
                                                           axis=1)
            fold_metrics_average = np.mean(total_fold_metrics_to_average, axis=0)
            fold_metrics_std = np.std(total_fold_metrics_to_average, axis=0)
            fold_averaged_results = [total_samples]
            for m in range(len(fold_metrics_average)):
                fold_averaged_results.append(fold_metrics_average[m])
                fold_averaged_results.append(fold_metrics_std[m])

            # Performing pooled estimates (taking into account the sample size for each fold) when relevant
            pooled_fold_averaged_results = [len(unique_folds), total_samples]
            pw_index = 6  # Length of the initial fold_average_columns, without the first two elements.
            for m in range(total_fold_metrics_to_average.shape[1]):
                mean_final = 0
                std_final = 0
                for f in range(len(unique_folds)):
                    fold_val = total_fold_metrics_to_average[f, m]
                    fold_sample_size = list(metrics_per_fold_df['# samples'])[f]
                    mean_final = mean_final + (fold_val * fold_sample_size)
                    # For patient-wise metrics, there is no std value for within each fold
                    if m < pw_index:
                        std_final = std_final
                    else:
                        std_final = std_final + (
                                    (fold_sample_size - 1) * math.pow(fold_std_metrics_to_average[f, m - pw_index], 2) + (
                                fold_sample_size) * math.pow(fold_val, 2))
                mean_final = mean_final / total_samples
                if m >= pw_index:
                    std_final = math.sqrt(
                        (1 / (total_samples - 1)) * (std_final - (total_samples * math.pow(mean_final, 2))))
                else:
                    std_final = np.std(total_fold_metrics_to_average[:, m])
                pooled_fold_averaged_results.extend([mean_final, std_final])

            overall_average_columns = ['Fold', '# samples']
            for m in ['Patient-wise recall', 'Patient-wise precision', 'Patient-wise specificity', 'Patient-wise F1',
                      'Patient-wise Accuracy', 'Patient-wise Balanced accuracy']:
                overall_average_columns.extend([m + ' (Mean)', m + ' (Std)'])

            for m in metric_names:
                overall_average_columns.extend([m + ' (Mean)', m + ' (Std)'])
            pooled_fold_averaged_results_df = pd.DataFrame(
                data=np.asarray(pooled_fold_averaged_results).reshape(1, len(overall_average_columns)),
                columns=overall_average_columns)
            study_filename = os.path.join(folder, class_name + '_overall_metrics_average.csv') if suffix == ''\
                else os.path.join(folder, class_name + '_overall_metrics_average_' + suffix + '.csv')
            pooled_fold_averaged_results_df.to_csv(study_filename, index=False)
        except Exception as e:
            print("Issue arose for class: {}.".format(class_name))
            print(traceback.format_exc())

    def compute_and_plot_metric_over_metric_cascading_categories(self, class_name: str, category: str = 'All'):
        results_filename = os.path.join(self.input_folder, 'Validation', class_name + '_dice_scores.csv')
        results = pd.read_csv(results_filename)
        results.replace('inf', np.nan, inplace=True)
        optimal_overlap = self.classes_optimal[class_name]['All'][0] if category == 'All' else \
            self.classes_optimal[class_name]['True Positive'][0]
        optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else \
            self.classes_optimal[class_name]['True Positive'][1]
        # if category != 'All':
        #     if category != 'All':
        #         results = results.loc[results["True Positive"] == True]
        #         if category == "TP":
        #             results = results.loc[(results["Threshold"] == optimal_threshold) & (
        #                     results["PiW Dice"] >= optimal_overlap)]
        #         else:
        #             results = results.loc[results["Threshold"] == optimal_threshold]
        #     else:
        #         results = results.loc[results["Threshold"] == optimal_threshold]

        # total_thresholds = [np.round(x, 2) for x in list(np.unique(results['Threshold'].values))]
        # nb_thresholds = len(np.unique(results['Threshold'].values))
        # optimal_threshold = self.classes_optimal[class_name]['All'][1] if category == 'All' else self.classes_optimal[class_name]['True Positive'][1]
        # optimal_thresold_index = total_thresholds.index(optimal_threshold)
        optimal_results_per_patient = results #[optimal_thresold_index::nb_thresholds]
        optimal_results_per_patient['Patient'] = optimal_results_per_patient.Patient.astype(str)
        if self.extra_patient_parameters is not None:
            total_optimal_results = pd.merge(optimal_results_per_patient, self.extra_patient_parameters, on="Patient")
        else:
            total_optimal_results = optimal_results_per_patient

        data_per_complete_selection = {}
        for s in SharedResources.getInstance().studies_selections_categorical:
            parsing = s.split(',')
            metric1 = parsing[0].strip()
            metric2 = parsing[1].strip()
            if parsing[2].strip() != '':
                metric2_cutoffs = [x for x in parsing[2].split('-')]
            else:
                metric2_cutoffs = None

            total_optimal_results = total_optimal_results.dropna(subset=[metric2])
            if metric2_cutoffs is None or len(metric2_cutoffs) == 0:
                metric2_cutoffs = list(np.unique(total_optimal_results[metric2].values))

            new_data_per_complete_selection = {}
            if len(data_per_complete_selection.keys()) == 0:
                for c, cutoff in enumerate(metric2_cutoffs):
                    cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] == cutoff]
                    new_data_per_complete_selection[cutoff] = cat_optimal_results
            else:
                for sel in list(data_per_complete_selection.keys()):
                    results_pool = data_per_complete_selection[sel]
                    for c, cutoff in enumerate(metric2_cutoffs):
                        cat_results = results_pool.loc[results_pool[metric2] == cutoff]
                        new_data_per_complete_selection[sel + ';' + cutoff] = cat_results
            data_per_complete_selection = new_data_per_complete_selection

        for s in SharedResources.getInstance().studies_selections_dense:
            parsing = s.split(',')
            metric1 = parsing[0].strip()
            metric2 = parsing[1].strip()
            if parsing[2].strip() != '':
                metric2_cutoffs = [float(x) for x in parsing[2].split('-')]
            else:
                metric2_cutoffs = None

            if metric2_cutoffs is None or len(metric2_cutoffs) == 0:
                continue

            total_optimal_results = total_optimal_results.dropna(subset=[metric2])
            new_data_per_complete_selection = {}
            if len(data_per_complete_selection.keys()) == 0:
                for c, cutoff in enumerate(metric2_cutoffs):
                    if c == 0:
                        cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] <= cutoff]
                        new_data_per_complete_selection[metric2 + '<=' + str(cutoff)] = cat_optimal_results
                    else:
                        cat_optimal_results = total_optimal_results.loc[metric2_cutoffs[c-1] < total_optimal_results[metric2] <= cutoff]
                        new_data_per_complete_selection[metric2 + ']' + str(metric2_cutoffs[c-1]) + ',' + str(cutoff) + ']'] = cat_optimal_results
                cat_optimal_results = total_optimal_results.loc[total_optimal_results[metric2] > metric2_cutoffs[-1]]
                new_data_per_complete_selection[metric2 + '>' + str(metric2_cutoffs[-1])] = cat_optimal_results
            else:
                for sel in list(data_per_complete_selection.keys()):
                    results_pool = data_per_complete_selection[sel]
                    for c, cutoff in enumerate(metric2_cutoffs):
                        if c == 0:
                            cat_optimal_results = results_pool.loc[results_pool[metric2] <= cutoff]
                            new_data_per_complete_selection[sel + ';' + metric2 + '<=' + str(cutoff)] = cat_optimal_results
                        else:
                            cat_optimal_results = results_pool.loc[
                                metric2_cutoffs[c - 1] < results_pool[metric2] <= cutoff]
                            new_data_per_complete_selection[sel + ';' + metric2 +
                                ']' + str(metric2_cutoffs[c - 1]) + ',' + str(cutoff) + ']'] = cat_optimal_results
                    cat_optimal_results = results_pool.loc[
                        results_pool[metric2] > metric2_cutoffs[-1]]
                    new_data_per_complete_selection[sel + ';' + metric2 + '>' + str(metric2_cutoffs[-1])] = cat_optimal_results
            data_per_complete_selection = new_data_per_complete_selection

        for sel in list(data_per_complete_selection.keys()):
            combined_df = data_per_complete_selection[sel]
            if len(combined_df) != 0:
                # self.__compute_results_metric_over_metric(data=combined_df, class_name=class_name, metric1='PiW Dice',
                #                                           metric2='GT volume (ml)', category=category,
                #                                           study_name='Cascaded_Selection/' + sel, suffix='_' + sel)
                dest_folder = os.path.join(self.output_folder, 'Cascaded_Selection', sel)
                os.makedirs(dest_folder, exist_ok=True)
                self.compute_fold_average(folder=dest_folder,
                                          data=combined_df,
                                          class_optimal=self.classes_optimal, metrics=self.metric_names,
                                          suffix='_' + sel,
                                          class_names=SharedResources.getInstance().studies_class_names,
                                          condition=category
                                          )
                export_segmentation_df_to_latex_paper(folder=dest_folder, class_name=class_name, study=sel,
                                                      input_csv_filename=os.path.join(dest_folder, class_name + '_overall_metrics_average_' + category + '_' + sel + '.csv'))
            else:
                print("No results for the following combination: {}. Skipping...".format(sel))