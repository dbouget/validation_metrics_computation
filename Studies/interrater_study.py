import subprocess
import shutil
import os
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import SimpleITK as sitk
from Utils.io_converters import reload_optimal_validation_parameters
from Utils.resources import SharedResources
from Validation.validation_utilities import *
from Validation.validation_utilities import compute_fold_average
from PIL import Image
from pathlib import Path
import seaborn as sns
from .hgg_postop_segmentation import threshold_volume_and_compute_classification_metrics


class ComparePostopSegmentationStudy:
    """
    Study for segmenting tumors (all types?) in T1 MRIs (only?).
    """
    def __init__(self):
        self.override = False  # Will recompute everything (the transform and corresponding registrations) if activated
        self.input_folder = Path(SharedResources.getInstance().studies_input_folder)
        self.output_folder = Path(SharedResources.getInstance().studies_output_folder)
        self.study_name = Path(SharedResources.getInstance().studies_study_name)

        if not self.output_folder.exists():
            raise ValueError('No [\'Studies\'][\'output_folder\'] provided for the postop segmentation study.')

        self.output_dir = Path(self.output_folder, self.study_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        #Path(self.output_dir, 'Validation').mkdir(exist_ok=True)

        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            try:
                self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(int).astype(str)
            except Exception as e:
                print(f"Convert patient IDs to int failed, error {e}")
            self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(str)

        #self.studies = ['Ams_Trd_T1c', 'Ams_Trd_T1c_T1w', 'Ams_Trd_T1c_T1w_FLAIR', 'All_T1c_T1w_preop']
        #self.studies = ['run2_exp1_T1c', 'run2_exp2_T1c_T1w', 'run2_exp3_T1c_T1w_flair', 'run2_exp4_T1c_T1w_preop', 'run2_exp5_T1c_T1w_flair_preop']
        self.studies = ['501', '502', '503', '505']
        self.studies_description = ['T1ce', 'T1ce+T1w', 'T1ce+T1w+FLAIR',  'T1ce+T1w+FLAIR+Preop T1ce'] #'T1ce+T1w+Preop T1ce',

        #self.studies = ['Ams_T1c', 'Ams_Trd_T1c', 'Ams_T1c_T1w', 'Ams_Trd_T1c_T1w'] #
        #self.studies_description = ['Ams, T1c', 'Ams + Trd, T1c', 'Ams, T1c + T1w', 'Ams + Trd, T1c + T1w']#, 'Ams v1 + Trd, T1c']

        #self.studies = ['Ams_T1c', 'Ams_Trd_T1c', 'Ams_T1c_T1w', 'Ams_Trd_T1c_T1w'] #
        #self.studies_description = ['Ams, T1c', 'Ams + Trd, T1c', 'Ams, T1c + T1w', 'Ams + Trd, T1c + T1w']#, 'Ams v1 + Trd, T1c']
        # self.studies_description = ['Ams, T1c', 'Ams, T1c + T1w']  # , 'Ams v1 + Trd, T1c']
        # self.studies = ['Ams_T1c', 'Ams_T1c_T1w']  #

        #self.studies = ['Postop_Finetune_MRI_HGG_Preop_BrainMask', 'Postop_3Seq_v2', 'Postop_t1_w_t1woc']
        #self.studies_short = ['T1c', 'T1c + T2 + FLAIR', 'T1c + T1']

        #self.studies = ['Ams_T1c']#, 'Ams_Trd_T1c']
        #self.studies_short = ['Ams T1c']

        #self.studies = ['ams_data_stuyd']
        #self.studies_short = ['AMS Inference only']

        #sns.set_theme()
        self.palette = sns.color_palette('Accent', 10)
        sns.set_palette(sns.color_palette('Accent', 10))
        self.colors = [self.palette[1], self.palette[4], self.palette[3], self.palette[6], self.palette[7]]

    def run(self):
        #self._run_subdir('Validation')
        self._run_subdir('Test')
        return

    def _run_subdir(self, subdir):
        output_dir = Path(self.output_dir, subdir)
        output_dir.mkdir(exist_ok=True)

        self.read_global_results(subdir=subdir)
        self.compute_results_minimal_dataset_overlap_cutoff(subdir=subdir)
        self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Patient-wise F1 postop', 'Patient-wise recall postop',
                                                 'Patient-wise precision postop'],
                                        suffix='diceP-diceTP-F1-Rec-Prec', subdir=subdir)
        self.plot_all_metrics(subdir=subdir)

    def read_global_results(self, subdir=''):
        self.overall_metrics_average_list = []
        self.overall_metrics_average_cutoff_list = []
        self.results_list = []
        self.results_cutoff_list = []
        self.results_filtered_minimal = []

        for study in self.studies:
            metrics_filepath = Path(self.input_folder, study, subdir, 'overall_metrics_average.csv')
            metrics = pd.read_csv(str(metrics_filepath))
            self.overall_metrics_average_list.append(metrics)

            metrics_filepath = Path(self.input_folder, study, subdir, 'overall_metrics_average_volume_cutoff.csv')
            metrics = pd.read_csv(str(metrics_filepath))
            self.overall_metrics_average_cutoff_list.append(metrics)

            results_filepath = Path(self.input_folder, study, subdir, 'all_dice_scores.csv')
            results = pd.read_csv(str(results_filepath))
            self.results_list.append(results)

            results_cutoff_filepath = Path(self.input_folder, study, subdir, 'all_dice_scores_volume_cutoff.csv')
            results_cutoff = pd.read_csv(str(results_cutoff_filepath))
            self.results_cutoff_list.append(results_cutoff)

            #optimal_overlap, optimal_threshold

    def compute_results_minimal_dataset_overlap_cutoff(self, subdir=''):
        all_pids = [res.Patient for res in self.results_list]
        all_pids_cutoff = [res.Patient for res in self.results_cutoff_list]
        results_minimal_filtered = []
        results_minimal_filtered_cutoff = []

        for i, (res, res_cutoff) in enumerate(zip(self.results_list, self.results_cutoff_list)):
            results_filtered = deepcopy(res)
            results_filtered_cutoff = deepcopy(res_cutoff)

            for pids_list in all_pids:
                results_filtered = results_filtered[results_filtered.Patient.isin(pids_list)]

            for pids_list in all_pids_cutoff:
                results_filtered_cutoff = results_filtered_cutoff[results_filtered_cutoff.Patient.isin(pids_list)]

            results_minimal_filtered.append(results_filtered)
            optimal_overlap, optimal_threshold = self.__retrieve_optimum_values(self.studies[i])
            compute_fold_average(self.output_dir, data=results_filtered, best_threshold=optimal_threshold,
                                 best_overlap=optimal_overlap,
                                 suffix=f'{self.studies[i]}_minimal_overlap', dice_fixed_metric=True,
                                 output_folder=str(Path(self.output_dir, subdir)))

            results_minimal_filtered_cutoff.append(results_filtered_cutoff)
            compute_fold_average(self.output_dir, data=results_filtered_cutoff, best_threshold=optimal_threshold,
                                 best_overlap=optimal_overlap,
                                 suffix=f'{self.studies[i]}_minimal_overlap_cutoff', dice_fixed_metric=True,
                                 output_folder=str(Path(self.output_dir, subdir)))

            #results_min_filtered_thresh =

    def write_cutoff_results_latex(self, metrics=['Dice', 'Dice-TP', 'Dice-P', 'Patient-wise recall postop',
                                                 'Patient-wise precision postop', 'Patient-wise F1 postop'],
                                    suffix='', subdir=''):
        latex_table_fname = Path(self.output_dir, subdir, 'key_metrics_after_cutoff_latex.txt') if suffix == '' else Path(self.output_dir,
                                                                                                                          subdir,
                                                                                                                          f'results_{suffix}_latex.txt')
        pfile = open(latex_table_fname, 'w+')

        for i, study in enumerate(self.studies):
            output_string = self.studies_description[i]
            fname = Path(self.output_dir, subdir, f'overall_metrics_average_{study}_minimal_overlap_cutoff.csv')
            results = pd.read_csv(fname)

            for m in metrics:
                if results.loc[0, m+'_std'] > 0:
                    output_string += f" & {results.loc[0, m + '_mean'] * 100:.2f}$\pm${results.loc[0, m + '_std'] * 100:.2f}"
                else:
                    output_string += f" & {results.loc[0, m+'_mean']*100:.2f}"

            pfile.write(output_string + "\\\ \n")

        pfile.close()

    def plot_all_metrics(self, subdir=''):
        dice_metrics = ['Dice', 'Dice-P', 'Dice-TP', 'Dice-N']
        self.plot_metrics(dice_metrics,
                          'dice_metrics.png', f'Postop DSC - {subdir}', figsize=(10, 7),
                          subdir=subdir)

        classification_metrics = ['Patient-wise recall postop', 'Patient-wise precision postop',
                                  'Patient-wise F1 postop', 'Accuracy',
                                  'Balanced accuracy', 'Positive rate', 'Specificity']
        self.plot_metrics(classification_metrics,
                          'res_tumor_classification.png',
                          f'Metrics for binary classification: residual tumor / gross total resection - {subdir}',
                          figsize=(18, 10),
                          subdir=subdir)

        specific_classification_metrics = ['Patient-wise recall postop', 'Patient-wise precision postop',
                                           'Patient-wise F1 postop', 'Object-wise recall', 'Object-wise precision',
                                           'Object-wise F1']
        self.plot_metrics(specific_classification_metrics,
                          'specific_classification_metrics.png',
                          f'Patient-wise and object-wise classification metrics - {subdir}',
                          figsize=(16, 10),
                          subdir=subdir)

        metrics_cutoff = ['Patient-wise recall postop', 'Patient-wise precision postop',
                          'Patient-wise F1 postop', 'Accuracy', 'Balanced accuracy',
                          'Positive rate', 'Specificity', 'Dice-P', 'Dice-TP']
        self.plot_metrics(metrics_cutoff,
                          'metrics_volume_cutoff.png',
                          f'Classification metrics and DSC after volume cutoff - {subdir}',
                          figsize=(18, 10), cutoff=True, subdir=subdir)
        self.plot_metrics(metrics_cutoff,
                          'metrics_before_volume_cutoff.png',
                          f'Classification metrics and DSC before volume cutoff - {subdir}',
                          figsize=(18, 10), cutoff=False, subdir=subdir)

        key_metrics = ['Patient-wise recall postop', 'Patient-wise precision postop',
                       'Patient-wise F1 postop', 'Dice-P', 'Dice-TP']
        self.plot_metrics(key_metrics,
                          'key_metrics_volume_cutoff.png',
                          f'Metrics for post-operative segmentation performance',
                          figsize=(16, 10), cutoff=True, subdir=subdir)


    def __retrieve_optimum_values(self, study):
        study_filename = os.path.join(self.input_folder, study, 'Validation', 'optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this.')

        return reload_optimal_validation_parameters(study_filename=study_filename)

    def plot_metrics(self, metric_names, file_name, plot_title, figsize=(14, 10), cutoff=False, subdir=''):
        index = metric_names
        std_err = []
        scores = []

        for i in range(len(self.studies)):
            if not cutoff:
                scores.append([self.overall_metrics_average_list[i][ind + '_mean'][0] for ind in index])
                std_err.append([self.overall_metrics_average_list[i][ind + '_std'][0] for ind in index])
            else:
                scores.append([self.overall_metrics_average_cutoff_list[i][ind + '_mean'][0] for ind in index])
                std_err.append([self.overall_metrics_average_cutoff_list[i][ind + '_std'][0] for ind in index])

        fig, ax = plt.subplots(figsize=figsize)
        rects = []
        ind = np.arange(len(index))
        width = 0.8 / len(self.studies)

        for i in range(len(self.studies)):
            rect = ax.bar(ind + width * i, scores[i], width, color=self.colors[i], yerr=std_err[i])
            rects.append(rect)

        ax.set_ylabel('Scores', fontsize=18)
        ax.set_title(plot_title, fontsize=20)
        ax.set_xticks(ind + (len(self.studies) - 1) * width / 2)
        ax.set_xticklabels(index, fontsize=18, rotation=15)
        ax.set_ylim([0, 1])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)

        for i, rect in enumerate(rects):
            for j, r in enumerate(rect):
                height = r.get_height()
                text = round(scores[i][j], 2)
                ax.text(r.get_x() + r.get_width() / 2, height + 0.01,
                        text, ha='center', va='bottom', fontsize=16)

        if not cutoff:
            legends = (name + f", N={int(data['Fold'][0])}" for name, data in zip(self.studies_description, self.overall_metrics_average_list))
        else:
            legends = (name + f", N={int(data['Fold'][0])}" for name, data in
                       zip(self.studies_description, self.overall_metrics_average_cutoff_list))
        ax.legend((r[0] for r in rects), (l for l in legends), loc=1, fontsize=18)
        plot_output_path = Path(self.output_dir, subdir, file_name)
        print(f"Saving figure at {plot_output_path}")
        plt.tight_layout()
        plt.savefig(plot_output_path)


