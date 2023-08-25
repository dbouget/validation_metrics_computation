import sys
import os
import pandas as pd
import numpy as np
from copy import deepcopy
import traceback
from Utils.resources import SharedResources
from Utils.io_converters import reload_optimal_validation_parameters
from Validation.validation_utilities import compute_fold_average
from Validation.extra_metrics_computation import compute_extra_metrics
from Plotting.metric_versus_binned_boxplot import compute_binned_metric_over_metric_boxplot_postop
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyCompare import blandAltman


class HGGPostopSegmentationStudy:
    """
    Study for residual tumor segmentation from high-grade gliomas (the core tumor) in T1 MRIs.
    """
    def __init__(self):
        self.study_name = SharedResources.getInstance().studies_study_name
        self.input_folder = Path(SharedResources.getInstance().studies_input_folder, self.study_name)
        self.base_output_folder = Path(SharedResources.getInstance().studies_output_folder, self.study_name)
        if self.base_output_folder is not None and str(self.base_output_folder) != "":
            self.output_folder = Path(self.base_output_folder, 'Validation')
        else:
            self.output_folder = Path(self.input_folder, 'Validation')
        self.output_folder.mkdir(exist_ok=True)

        self.metric_names = []
        self.extra_metric_names = SharedResources.getInstance().validation_metric_names
        self.extra_patient_parameters = None
        if Path(SharedResources.getInstance().studies_extra_parameters_filename).exists():
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            try:
                self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(
                    int).astype(str)
            except Exception as e:
                print(f"Convert patient IDs to int failed, keep as string, error {e}")
            self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(str)

        self.optimal_overlap = None
        self.optimal_threshold = None

        self.convert_ids = SharedResources.getInstance().postop_convert_ids
        self.exclude_ids = SharedResources.getInstance().postop_exclude_ids
        if len(self.exclude_ids) > 0:
            try:
                exclude_ids = [int(i) for i in self.exclude_ids]
                self.exclude_ids = exclude_ids
            except Exception as e:
                print(f"Convert exclude patient IDs to int failed, keep as string, error {e}")

    def run(self):
        self.__retrieve_optimum_values()
        self.__read_results()

        # self.export_data_validation_study()
        # self.__compute_and_plot_overall()
        if self.extra_patient_parameters is not None:
            # self.__compute_and_plot_metric_over_metric_categories(data=self.results, metric1='Dice', metric2='True postop volume', metric2_cutoffs=[1.])
            # volume_figure_fname = Path(self.input_folder, 'Validation', 'volume_cutoff.png')
            # self.__compute_and_plot_volume_cutoff_results(volume_cutoff_range=[0., 0.5], optimal_cutoff=0.175, save_fname=volume_figure_fname)
            results_cutoff = self.__compute_results_cutoff_volume(cutoff_volume=0.175)
            results_cutoff = self.compute_volume_error(results_cutoff)
            results_cutoff.to_csv(Path(self.input_folder, 'Validation', 'all_dice_scores_volume_cutoff.csv'), index=False)
            compute_fold_average(self.input_folder, results_cutoff, best_threshold=self.optimal_threshold,
                                 best_overlap=self.optimal_overlap, metrics=['HD95', 'Absolute volume error', 'True preop volume',
                                                                             'True postop volume', 'Predicted postop volume'],
                                 suffix='volume_cutoff', output_folder=str(self.output_folder))
            # results_cutoff = self.__compute_EOR(results_cutoff, crop_to_zero=True)
            # self.__study_volume_and_EOR(results_cutoff)

    def __retrieve_optimum_values(self):
        study_filename = os.path.join(self.input_folder, 'Validation', 'optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this.')

        self.optimal_overlap, self.optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)

    def __read_results(self):
        try:
            results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
            results = pd.read_csv(results_filename)

            if self.convert_ids:
                results = self.__convert_patient_ids(results)

            extra_metrics_filepath = Path(self.input_folder, 'Validation', 'extra_metrics_results_per_patient_thr' +
                                          str(int(self.optimal_threshold * 100.)) + '.csv')

            if extra_metrics_filepath.exists():
                extra_metrics = pd.read_csv(extra_metrics_filepath)
                extra_metrics['Threshold'] = self.optimal_threshold * np.ones(shape=len(extra_metrics))
                extra_metrics = extra_metrics[["Patient", "Threshold"] + self.extra_metric_names]
                for em in self.extra_metric_names:
                    if em in results.columns:
                        results.drop(em, axis=1, inplace=True)
                cols = results.columns.tolist() + self.extra_metric_names
                results = pd.merge(results, extra_metrics, on=["Patient", "Threshold"], how='left')
                results = results[cols]

            dice_thresholds = [np.round(x, 1) for x in list(np.unique(results['Threshold'].values))]
            nb_thresholds = len(dice_thresholds)
            optimal_threshold_index = dice_thresholds.index(self.optimal_threshold)
            optimal_results_per_patient = results[optimal_threshold_index::nb_thresholds]

            if self.extra_patient_parameters is not None:
                try:
                    optimal_results_per_patient.loc[:, 'Patient'] = optimal_results_per_patient.Patient.astype(int)
                except Exception as e:
                    print("Conversion of patient ID to int failed, keep as str")
                optimal_results_per_patient.loc[:, 'Patient'] = optimal_results_per_patient.Patient.astype(str)
                optimal_results_per_patient = pd.merge(optimal_results_per_patient, self.extra_patient_parameters,
                                                       on="Patient", how='left')

            self.results = results
            self.optimal_results = optimal_results_per_patient

            if len(self.exclude_ids) > 0:
                self.__drop_exclude_ids()

            # Save new results file after ID conversion and exclusion
            self.results.to_csv(Path(self.output_folder, 'all_dice_scores_clean.csv'), index=False)

        except Exception as e:
            print('{}'.format(traceback.format_exc()))

    def __convert_patient_ids(self, results):
        id_mapping_file = SharedResources.getInstance().postop_id_mapping
        base_id_column = SharedResources.getInstance().postop_base_id_column
        new_id_column = SharedResources.getInstance().postop_new_id_column

        if not Path(id_mapping_file).exists():
            print(f"Error converting patient IDs, ID mapping file {id_mapping_file} not found")
            return results

        id_mapping = pd.read_csv(id_mapping_file)
        if base_id_column not in id_mapping.columns or new_id_column not in id_mapping.columns:
            print(f"Error converting patient IDs, base id or new id columns {base_id_column}, {new_id_column} missing from id mapping")

        # Create ID dict
        self.id_dict = {id_row[base_id_column]: id_row[new_id_column] for row_name, id_row in id_mapping.iterrows()}

        # Update results
        results['Patient_old_IDs'] = deepcopy(results['Patient'])
        if 'HGG' in results['Patient'][0]:
            results['Patient'] = results['Patient'].map(lambda x: x.split('_')[1])
        else:
            results['Patient'] = results['Patient'].map(self.id_dict)

        results['Patient'] = results['Patient'].astype(int)
        results.dropna(axis='index', inplace=True)
        return results

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
            columns_to_drop = ['Fold', 'Patient', 'Patient_old_IDs', 'Threshold', 'Dice', '#GT', '#Det']
            columns = results_df.columns
            for elem in columns_to_drop:
                if elem in columns.values:
                    columns = columns.drop(elem)
            self.metric_names = list(columns.values)
            compute_fold_average(self.input_folder, data=results_df, best_threshold=self.optimal_threshold,
                                 best_overlap=self.optimal_overlap, metrics=self.metric_names)
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
                    results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
                    results = pd.read_csv(results_filename)
                else:
                    results = deepcopy(data)
                dice_thresholds = [np.round(x, 1) for x in list(np.unique(results['Threshold'].values))]
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
            if self.extra_patient_parameters is None:
                return

            results = deepcopy(self.results)
            optimal_results_per_patient = deepcopy(self.optimal_results)
            folder = os.path.join(self.input_folder, 'Validation', metric2 + '-Wise')
            os.makedirs(folder, exist_ok=True)
            compute_binned_metric_over_metric_boxplot_postop(folder=folder, data=optimal_results_per_patient,
                                                      metric1=metric1, metric2=metric2, criterion1=self.optimal_overlap,
                                                      postfix='_overall' + suffix, number_bins=10)

            # Fold-wise analysis #
            fold_base_folder = os.path.join(folder, 'fold_analysis')
            os.makedirs(fold_base_folder, exist_ok=True)

            existing_folds = np.unique(results['Fold'].values)
            for f, fold in enumerate(existing_folds):
                # results_fold = results.loc[results['Fold'] == fold]
                # optimal_results_per_patient = results_fold[optimal_thresold_index::nb_thresholds]
                optimal_results_per_patient_fold = optimal_results_per_patient[optimal_results_per_patient['Fold'] == fold]
                # if True in [x not in list(results.columns) for x in list(self.extra_patient_parameters.columns)]:
                #     optimal_results_per_patient_fold.loc[:, 'Patient'] = optimal_results_per_patient_fold.Patient.astype(str)
                #     # Trick to only keep extra information for patients from the current fold with the 'how' attribute
                #     fold_optimal_results = pd.merge(optimal_results_per_patient_fold, self.extra_patient_parameters,
                #                                     on="Patient", how='left')
                # else:
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
                results_filename = os.path.join(self.input_folder, 'Validation', 'all_dice_scores.csv')
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
                # print(output)

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
        data.loc[(data['Predicted postop volume'] == cutoff_volume), '#Det'] = 0

        if cutoff_gt:
            data.loc[(data['True postop volume'] <= cutoff_volume), 'True postop volume'] = 0.0
            data.loc[(data['True postop volume'] == 0), '#GT'] = 0

        # Drop columns missing volume
        data.dropna(axis=0, subset=['True postop volume'], inplace=True)

        return data

    def __compute_EOR(self, results, crop_to_zero=False):
        data = deepcopy(results)

        true_EOR = np.array((results.loc[:, 'True preop volume'] - results.loc[:, 'True postop volume']) / results.loc[:, 'True preop volume'])
        data['True EOR'] = true_EOR

        predicted_EOR_type1 = np.array((results.loc[:, 'True preop volume'] - results.loc[:, 'Predicted postop volume']) / results.loc[:, 'True preop volume'])
        data['Predicted EOR type 1'] = predicted_EOR_type1

        if crop_to_zero:
            data.loc[(data['True EOR'] < 0), 'True EOR'] = 0.0
            data.loc[(data['Predicted EOR type 1'] < 0), 'Predicted EOR type 1'] = 0.0

        return data

    def compute_volume_error(self, results):
        data = deepcopy(results)
        abs_volume_error = np.array(np.abs(results.loc[:, 'True postop volume'] - results.loc[:, 'Predicted postop volume']))
        data['Absolute volume error'] = abs_volume_error

        # Need to handle exceptions with division by zero correctly to compute the relative error as we have a lot of
        # small volumes - just keeping the absolute error for now
        # rel_volume_error = np.array(np.divide(abs_volume_error, results.loc[:, 'True postop volume']))
        # index_rel_vol_err = np.where((abs_volume_error > 0) & (results.loc[:, 'True postop volume'] == 0))
        # rel_volume_error[np.where((abs_volume_error > 0) & (results.loc[:, 'True postop volume'] == 0))] = 1

        return data

    def __study_volume_and_EOR(self, results):
        data = deepcopy(results)
        output_folder = Path(self.output_folder, 'Volume-EOR')
        output_folder.mkdir(exist_ok=True)
        sns.set_style('ticks')

        # EOR
        save_fname = str(Path(output_folder, f'EOR_scatter_{self.study_name}_Validation.png'))
        plt.figure()
        ax = sns.scatterplot(data=data*100, x='True EOR', y='Predicted EOR type 1')
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1")
        ax.set(title=f'True vs predicted EOR',
               xlabel='True EOR (%)', ylabel='Predicted EOR (%)')
        plt.savefig(save_fname)

        save_fname = str(Path(output_folder, f'EOR_bland_altman_{self.study_name}_Validation.png'))
        blandAltman(data['True EOR'], data['Predicted EOR type 1'],
                    title=f'Bland-Altman of true vs predicted EOR',
                    savePath=save_fname)

        # VOLUME
        save_fname = str(Path(output_folder, f'volume_scatter_{self.study_name}_Validation.png'))
        plt.figure()
        plt.xscale('symlog')
        plt.yscale('symlog')
        ax = sns.scatterplot(data=data, x='True postop volume', y='Predicted postop volume')
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1")
        #ax.title('True vs predicted postop volume (symlog scale)')
        ax.set(title=f'True vs predicted postop volume (symlog scale)',
               xlabel='True postop volume (ml)', ylabel='Predicted postop volume (ml)')
        plt.savefig(save_fname)

        save_fname = str(Path(output_folder, f'volume_bland_altman_{self.study_name}_Validation.png'))
        blandAltman(data['True postop volume'], data['Predicted postop volume'],
                    title=f'Bland-Altman of true vs predicted postop volume',
                    savePath=save_fname)

    def export_data_validation_study(self):
        data_base_dir = Path(self.input_folder).parent.parent
        opids_postop_consent = list(pd.read_csv(Path(data_base_dir, 'opids_postop_consent.csv')).columns)
        opids_postop_consent = np.unique(list(map(lambda x: int(x), opids_postop_consent)))
        print(data_base_dir)


def threshold_volume_and_compute_classification_metrics(optimal_results, cutoff_seg_vol, cutoff_true_vol=0.01):

    true_positives = len(
        optimal_results.loc[(optimal_results['Predicted postop volume'] > cutoff_seg_vol) & (optimal_results['True postop volume'] > cutoff_true_vol)])
    false_negatives = len(
        optimal_results.loc[(optimal_results['Predicted postop volume'] <= cutoff_seg_vol) & (optimal_results['True postop volume'] > cutoff_true_vol)])
    false_positives = len(
        optimal_results.loc[(optimal_results['Predicted postop volume'] > cutoff_seg_vol) & (optimal_results['True postop volume'] <= cutoff_true_vol)])
    true_negatives = len(
        optimal_results.loc[(optimal_results['Predicted postop volume'] <= cutoff_seg_vol) & (optimal_results['True postop volume'] <= cutoff_true_vol)])
    total = true_positives + false_negatives + false_positives + true_negatives

    # print(true_positives, false_negatives, false_positives, true_negatives, total)
    # print((true_positives +false_negatives) / total)
    # print((true_negatives + false_positives) / total)
    dice_pos = np.mean(optimal_results.loc[(optimal_results['True postop volume'] > cutoff_true_vol), 'Dice'])
    dice_true_pos = np.mean(optimal_results.loc[(optimal_results['Predicted postop volume'] > cutoff_seg_vol) & \
                                                (optimal_results['True postop volume'] > cutoff_true_vol), 'Dice'])

    results = {}
    results['Recall'] = true_positives / (true_positives + false_negatives)
    results['Precision'] = true_positives / (true_positives + false_positives)
    results['F1'] = 2 * ((results['Recall'] * results['Precision']) / (results['Precision'] + results['Recall']))
    results['Accuracy'] = (true_positives + true_negatives) / (true_negatives + true_positives + false_negatives + false_positives)
    results['Specificity'] = true_negatives / (true_negatives + false_positives)
    results['Balanced accuracy'] = (results['Recall'] + results['Specificity']) / 2
    results['Positive rate'] = (true_positives + false_negatives) / total
    results['Negative rate'] = (true_negatives + false_positives) / total
    results['Dice Positive'] = dice_pos
    results['Dice True Positive'] = dice_true_pos

    return results

def plot_classification_metrics_volume_cutoffs(results, cutoffs, title, output_path,
                                               metrics_to_maximize=['Accuracy'],
                                               metrics_to_plot=None, optimal_cutoff=None, save_fname=None):
    sns.set_theme()
    palette = sns.color_palette('colorblind', 10)
    fig = plt.figure()
    plt.ylim(0, 1)
    maxim_metrics = []
    colors = {}
    legend = []
    if metrics_to_plot is None:
        metrics_to_plot = results.keys()

    for i, m in enumerate(metrics_to_plot):
        plt.plot(cutoffs, results[m], c=palette[i])
        colors[m] = palette[i]
        if m in metrics_to_maximize:
            maxim_metrics.append(results[m])

    if optimal_cutoff is None:
        arg_cutoff = np.argmax(np.sum(np.array(maxim_metrics), axis=0))
    else:
        arg_cutoff = np.where(np.round(cutoffs, 3) == optimal_cutoff)[0][0]


    for m in metrics_to_maximize:
        plt.scatter(cutoffs[arg_cutoff], results[m][arg_cutoff], c=colors[m], marker='x')
        plt.text(cutoffs[arg_cutoff], results[m][arg_cutoff], f"{m} = {results[m][arg_cutoff]:.3f}", fontsize=10)

    #print(maxim_combined, cutoffs[maxim_combined])
    plt.legend(metrics_to_plot)
    if save_fname is not None:
        plt.savefig(save_fname)
    plt.show(block=False)
