import subprocess
import shutil
import os
import csv
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.interpolate import interp1d
import SimpleITK as sitk
from Utils.io_converters import reload_optimal_validation_parameters
from Utils.resources import SharedResources
from Validation.validation_utilities import *
from Validation.validation_utilities import compute_fold_average
from Validation.kfold_model_validation import compute_dice, compute_dice_uncertain, compute_tumor_volume
from PIL import Image
from pathlib import Path
import seaborn as sns
import re
from .hgg_postop_segmentation import threshold_volume_and_compute_classification_metrics
from scipy.stats import norm, mannwhitneyu, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import random
from Utils.statistics import pooled_std, pooled_mean
from copy import deepcopy

class CompareArchitecturesStudy:
    """
    Study for segmenting tumors (all types?) in T1 MRIs (only?).
    """
    def __init__(self):
        self.override = False  # Will recompute everything (the transform and corresponding registrations) if activated
        self.data_root = Path(SharedResources.getInstance().data_root)
        self.input_folder = Path(SharedResources.getInstance().studies_input_folder)
        self.output_folder = Path(SharedResources.getInstance().studies_output_folder)
        self.study_name = Path(SharedResources.getInstance().studies_study_name)

        if not self.output_folder.exists():
            raise ValueError('No [\'Studies\'][\'output_folder\'] provided for the postop segmentation study.')

        self.output_dir = Path(self.output_folder, self.study_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.arch_names = ['nnU-Net', 'AGU-Net']
        # self.input_study_dirs = [Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/Alexandros_experiments/compare_nnUNet_501-505'),
        #                          Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/TRD_experiments/AGU-Net_compare_exp1_5')]
        self.input_study_dirs = [Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/Alexandros_experiments'),
                                 Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/TRD_experiments')]

        #self.studies = ['Ams_Trd_T1c', 'Ams_Trd_T1c_T1w', 'Ams_Trd_T1c_T1w_FLAIR', 'All_T1c_T1w_preop']
        self.studies_lists = [['901', '902', '903', '904', '905'], #'504',
            ['run2_exp1_T1c', 'run2_exp2_T1c_T1w', 'run2_exp3_T1c_T1w_flair', 'run2_exp4_T1c_T1w_preop', 'run2_exp5_T1c_T1w_flair_preop']]
        # self.studies_lists = [['505'], ['run2_exp5_T1c_T1w_flair_preop']]
        # self.studies_description = ['T1ce', 'T1ce+T1w', 'T1ce+T1w+FLAIR', 'T1ce+T1w+Preop T1ce', 'T1ce+T1w+FLAIR+Preop T1ce'] #
        self.studies_description = ['A', 'B', 'C', 'D', 'E']
        # self.studies_description = ['E']
        self.subdirs = ['Validation', 'Test']
        self.subdir_labels = ['Val', 'Test']

        palette_names = ['cubehelix', 'cubehelix']
        palette_names = ['flare', 'crest']
        palette_names = ["light:salmon", "light:b"]
        self.palettes = [sns.color_palette(palette_names[0], n_colors=6), sns.color_palette(palette_names[1], n_colors=7, desat=1)]
        self.colors = [[pal[i+1] for i in range(len(self.studies_lists[0]))] for pal in self.palettes]

        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(int).astype(
                str)

        self.interrater_study_filepath = Path(self.output_dir, 'interrater_study.csv')

    def run(self):
        # for subdir in self.subdirs:
        #     self.run_subdir(subdir)
        # self._create_tables_segmentation()
        # self._create_tables_classification()
        # self.create_interrater_consensus_segmentations()
        # self.interrater_study()
        # self.interrater_study_summary()
        # self.visualize_interrater_results()

        ### Statistical tests ###
        # Segmentation
        # self.mannwhitneyutest_segmentation(study_number_nnunet=1, study_number_agunet=1)
        # self.tukey(0)
        # self.tukey(1)
        ## TODO: add confints on valset
        # self.confint_segmentation_cv()

        # self.classification_confints_bootstrapping_testset()
        # self.confint_classification_cv()

        self.equivalence_tests_interrater_study()
        return

    def _create_tables_segmentation(self):
        self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Object-wise recall',
                                                 'Object-wise precision', 'Object-wise F1'],
                                        suffix='seg_scores_DiceP_Dice_TP_obj-wise_Rec_Prec_F1', subdirs=self.subdirs)
        self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Patient-wise recall',
                                                 'Patient-wise precision postop', 'HD95', 'Absolute volume error_median'],
                                        suffix='seg_scores_diceP-diceTP-Rec-Prec-HD95-AVEmedian', subdirs=self.subdirs)

    def _create_tables_classification(self):
        self.write_cutoff_results_latex(
            metrics=['Patient-wise recall postop', 'Patient-wise precision postop', 'Specificity',
                     'Patient-wise F1 postop', 'Balanced accuracy'],
            suffix='classif_scores-Rec-Prec-Spec-F1-bAcc', subdirs=self.subdirs)

    def confint_segmentation_cv(self):
        print("\nConf ints for segmentation on cross-validation set")
        for arch, arch_name in enumerate(self.arch_names):
            print(f"Confidence intervals for architecture {arch_name}:")
            for study, study_name in enumerate(self.studies_lists[arch]):
                results = self.load_optimal_results(arch=arch, study=study_name, subdir='Validation')
                positive_index = (results['#GT'].values >= 1)
                results = results.loc[positive_index]
                # Create vector for each fold
                dice_scores = []
                for fold in np.unique(results['Fold']):
                    fold_result = results.loc[results['Fold'] == fold, 'Dice'].values
                    dice_scores.append(fold_result)

                confint = self.pooled_interval(dice_scores)
                # confint, theta, theta_hat = self.BCa_interval(arch, study)
                print(f"\tConfig {self.studies_description[study]}, confint = [{confint[0]:.4f}, {confint[1]:.4f}]")

    def pooled_interval(self, list_fold_scores):
        counts = np.array([len(scores) for scores in list_fold_scores])
        means = np.array([np.mean(scores) for scores in list_fold_scores])
        stds = np.array([np.std(scores) for scores in list_fold_scores])

        pooled_m = pooled_mean(means, counts)
        pooled_s = pooled_std(stds, counts)
        interval_width = 1.96*pooled_s/np.sqrt(np.sum(counts))
        return [pooled_m - interval_width, pooled_m + interval_width]

    def mannwhitneyutest_segmentation(self, study_number_nnunet, study_number_agunet):
        # Should only be run for the test-set
        best_model_nnunet = self.studies_lists[0][study_number_nnunet]
        best_model_agunet = self.studies_lists[1][study_number_agunet]

        results_nnunet = self.load_optimal_results(arch=0, study=best_model_nnunet, subdir='Test')
        results_agunet = self.load_optimal_results(arch=1, study=best_model_agunet, subdir='Test')

        if not np.array_equal(results_agunet.loc[:, 'Patient'], results_nnunet.loc[:, 'Patient']):
            results_agunet = results_agunet.set_index('Patient')
            results_agunet = results_agunet.reindex(results_nnunet['Patient'])
            results_agunet = results_agunet.reset_index()

        positive_index = (results_agunet['#GT'].values >= 1)
        mannwhit_res = mannwhitneyu(results_agunet.loc[positive_index, 'Dice'].values,
                                results_nnunet.loc[positive_index, 'Dice'].values)
        print(f"Wilcoxon test for comparing models nnU-Net {self.studies_description[study_number_nnunet]} and " \
              f"AGU-Net {self.studies_description[study_number_agunet]}")
        print(mannwhit_res)

    def tukey(self, arch):
        print(f"\nRunning Tukey-HSD for comparing inputs for architecture {self.arch_names[arch]}")
        all_results = []

        for study, study_name in enumerate(self.studies_lists[arch]):
            results = self.load_optimal_results(arch=arch, study=study_name, subdir='Test')
            all_results.append(results)

        positive_index = (all_results[0]['#GT'].values >= 1)
        all_dice_scores_pos = [res.loc[positive_index, 'Dice'].values for res in all_results]

        m_comp = tukey_hsd(all_dice_scores_pos, self.studies_description, all_dice_scores_pos[0].shape, alpha=0.05)
        latex_table = m_comp._results_table.as_latex_tabular()
        latex_table_filepath = Path(self.output_dir, f'tukey_results_{self.arch_names[arch]}.txt')
        with open(latex_table_filepath, 'w+') as pfile:
            pfile.write(latex_table)

        print(m_comp)

    def classification_confints_bootstrapping_testset(self):
        means = []
        confints = []
        for arch, arch_name in enumerate(self.arch_names):
            print(f"\nConfidence intervals for architecture {arch_name}:")
            means.append([])
            confints.append([])
            for study, study_name in enumerate(self.studies_lists[arch]):
                confint, theta, theta_hat = self.BCa_interval(arch, study)
                means[arch].append(theta_hat)
                confints[arch].append(confint)
                print(f"\tConfig {self.studies_description[study]}, theta_hat = {theta_hat:.4f},  confint = [{confint[0]:.4f}, {confint[1]:.4f}]")
        self._create_latex_table_confints(means, confints, 'classification_confints_bootstrap_test.txt')

    def BCa_interval(self, arch=0, study=0):
        model = self.studies_lists[arch][study]
        results = self.load_optimal_results(arch=arch, study=model, subdir='Test')
        pred, gt = create_prediction_vector(results)
        confint, theta, theta_hat = BCa_interval_macro_metric(gt.copy(), pred.copy(), balanced_accuracy, B=10000, q=0.975)
        return confint, theta, theta_hat # this is what you care about

    def confint_classification_cv(self):
        print("\nConf ints for classification on cross-validation set")
        means = []
        confints = []
        for arch, arch_name in enumerate(self.arch_names):
            means.append([])
            confints.append([])
            print(f"Confidence intervals for architecture {arch_name}:")
            for study, study_name in enumerate(self.studies_lists[arch]):
                metrics_filepath = Path(self.input_study_dirs[arch], study_name, 'Validation',
                                        'folds_metrics_average_volume_cutoff.csv')
                results = pd.read_csv(str(metrics_filepath))
                baccs = results['Balanced accuracy']
                counts = results['# samples']

                mean = pooled_mean(baccs, counts)
                std = np.std(baccs)
                interval_width = 1.96*std/np.sqrt(len(baccs))
                confint = [mean - interval_width, mean + interval_width]

                means[arch].append(mean)
                confints[arch].append(confint)

                print(f"\tConfig {self.studies_description[study]}, mean bacc = [{mean:.4f}], confint = [{confint[0]:.4f}, {confint[1]:.4f}]")
        self._create_latex_table_confints(means, confints, 'classification_confints_cv.txt')

    def _create_latex_table_confints(self, means, confints, output_fname):
        latex_table_filepath = Path(self.output_dir, output_fname)
        with open(latex_table_filepath, "w+") as pfile:
            pfile.write("\\begin{center} \n\\begin{tabular}{ccS[table-format = 1.1]\n")
            pfile.write("\t@{\\quad[\\,}S[table-format = -1.3]@{,\\,}S[table-format = -1.3]@{\\,]} \n\t}\n")
            pfile.write("\tArch & Config  & \\multicolumn{1}{c@{\\quad\\space}}{mean} & \\multicolumn{2}{c}{CI} \\\ \n \t\\hline \n")
            for arch, arch_name in enumerate(self.arch_names):
                for study, config_name in enumerate(self.studies_description):
                    pfile.write(f"\t{arch_name} & {config_name} & {means[arch][study]:.3f} & {confints[arch][study][0]:.3f} & {confints[arch][study][1]:.3f} \\\ \n")
                pfile.write("\t\\hline\n")
            pfile.write("\\end{tabular}\n\\end{center}")

    def equivalence_tests_interrater_study(self):
        results = pd.read_csv(self.interrater_study_filepath)
        results.replace('inf', 0, inplace=True)
        results.replace('', 0, inplace=True)
        results.replace(' ', 0, inplace=True)

        # Results for positive cases
        results = results.loc[results['residual_tumor'] == 1]

        annotators = ['nov1', 'nov2', 'nov3', 'nov4', 'exp1', 'exp2', 'exp3', 'exp4']
        avg_annotators = ['nov-avg', 'exp-avg', 'all-avg']
        grouped_annotators = [[a for a in annotators if 'nov' in a],
                              [a for a in annotators if 'exp' in a],
                              annotators.copy()]
        models = ['nnU-Net_B', 'AGU-Net_B']
        references = ['ground_truth_segmentation', 'strict-consensus-all-annotators']
        ref_short = {'ground_truth_segmentation': 'Ground truth',
                     'strict-consensus-all-annotators': 'Consensus'}
        results = results[results['annotator/model'].isin(annotators + models)]
        results.sort_values(by=['pid'], inplace=True)
        metric = "Jaccard"
        latex_table_rows = []
        for ref in references:
            latex_table_rows = []
            ref_results = results.loc[results['reference'] == ref]
            for model in models:
                model_res = ref_results.loc[ref_results['annotator/model'] == model]
                print(f"Mann-Whitney U tests for model {model}")
                model_tabname = model.replace('_', ' ')
                for annotator in annotators:
                    annot_res = ref_results.loc[ref_results['annotator/model'] == annotator]
                    test_result = mannwhitneyu(model_res[metric].values, annot_res[metric].values)
                    # print(np.all(np.array_equal(model_res['pid'].values, annot_res['pid'].values)))
                    print(f"\t Compared against annotator {annotator}: ", test_result)
                    latex_table_rows.append(f"{model_tabname} & {annotator} & " +
                                            f"${np.mean(model_res[metric].values):.3f} \\pm {np.std(model_res[metric].values):.3f}$ & " +
                                            f"${np.mean(annot_res[metric].values):.3f} \\pm {np.std(annot_res[metric].values):.3f}$ & " +
                                            f"{test_result.statistic:.1f} & {test_result.pvalue:.3f} \\\ ")
                # Compare models against averages over groups
                for group_name, group in zip(avg_annotators, grouped_annotators):
                    group_results = ref_results.loc[ref_results['annotator/model'].isin(group)]
                    results_grouped_mean = group_results.groupby('pid').mean()[metric]
                    test_result = mannwhitneyu(model_res[metric].values, results_grouped_mean.values)
                    latex_table_rows.append(f"{model_tabname} & {group_name} & " +
                                            f"${np.mean(model_res[metric].values):.3f} \\pm {np.std(model_res[metric].values):.3f}$ & " +
                                            f"${np.mean(results_grouped_mean.values):.3f} \\pm {np.std(results_grouped_mean.values):.3f}$ & " +
                                            f"{test_result.statistic:.1f} & {test_result.pvalue:.3f} \\\ ")
                latex_table_rows.append('\\hline')
            self._create_latex_table_interrater_tests(latex_table_rows, f'interrater_tests_ref_{ref_short[ref]}.txt')
        print("OK")
    def _create_latex_table_interrater_tests(self, latex_table_rows, output_fname):
        latex_table_filepath = Path(self.output_dir, output_fname)
        with open(latex_table_filepath, "w+") as pfile:
            pfile.write("\\begin{tabular}{cccccc}\n")
            pfile.write("\\toprule\n")
            pfile.write("\t\\textbf{Model - config} & \\textbf{Annotator} & \\textbf{Model Mean $\pm$ Std} " + \
                        "& \\textbf{Annotator Mean $\pm$ Std}  \\textbf{Statistic} & \\textbf{p-value} \\\ \n")
            pfile.write("\t\\midrule\n")
            for row in latex_table_rows:
                pfile.write("\t" + row +"\n")
            pfile.write("\\bottomrule\n")
            pfile.write("\\end{tabular}")
        return None

    def mcnemar_test(self):
        best_model_nnunet = self.studies_lists[0][1]
        best_model_agunet = self.studies_lists[1][1]
        # best_model_agunet2 = self.studies_lists[1][2]

        results_nnunet = self.load_optimal_results(arch=0, study=best_model_nnunet, subdir='Test')
        results_agunet = self.load_optimal_results(arch=1, study=best_model_agunet, subdir='Test')
        # results_nnunet = self.load_optimal_results(arch=1, study=best_model_agunet2, subdir='Test')

        if not np.array_equal(results_agunet.loc[:, 'Patient'], results_nnunet.loc[:, 'Patient']):
            results_agunet = results_agunet.set_index('Patient')
            results_agunet = results_agunet.reindex(results_nnunet['Patient'])
            results_agunet = results_agunet.reset_index()

        pred1, gt1 = create_prediction_vector(results_nnunet)
        pred2, gt2 = create_prediction_vector(results_agunet)

        pv = bootstrap_t_test(pred1, pred2, gt1, nboot=10000, direction="greater")
        print(pv)

        exit()

        ctable = self.__construct_contingency_table_mcnemar(results_nnunet, results_agunet)
        test_result = mcnemar(ctable)
        print("OK")

    def __construct_contingency_table_mcnemar(self, res_model_1, res_model_2):
        hit_model_1 = self.__construct_hit_vector_classificaiton(res_model_1)
        hit_model_2 = self.__construct_hit_vector_classificaiton(res_model_2)
        ctable = np.zeros((2, 2))

        for i in range(2):
           for j in range(2):
               ctable[i, j] = len(np.where(np.logical_and(hit_model_1 == i, hit_model_2 == j))[0])

        return np.transpose(ctable)

    def __construct_hit_vector_classificaiton(self, results):
        correct_classification = np.zeros(results.shape[0])
        index_tp = results.loc[(results['#GT'] > 0) & (results['#Det'] > 0)].index
        index_tn = results.loc[(results['#GT'] == 0) & (results['#Det'] == 0)].index
        correct_classification[index_tp] = 1
        correct_classification[index_tn] = 1
        return correct_classification

    def run_subdir(self, subdir):

        output_dir = Path(self.output_dir, subdir)
        output_dir.mkdir(exist_ok=True)

        self.read_results_subdir_all_exp(subdir)

        # for i in range(len(self.input_study_dirs)):
        #     self.read_results_arch(i, subdir=subdir)

        print("OK")
        self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'HD95', 'Absolute volume error'],
                                        suffix='diceP-diceTP-HD95-VolErr', subdirs=[subdir])
        # self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Patient-wise recall postop',
        #                                          'Patient-wise precision postop',  'Specificity', 'Patient-wise F1 postop'],
        #                                 suffix='diceP-diceTP-Rec-Prec-Spec-F1', subdirs=[subdir])
        # self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Patient-wise recall postop',
        #                                          'Patient-wise precision postop', 'Specificity',
        #                                          'Patient-wise F1 postop', 'Balanced accuracy'],
        #                                 suffix='diceP-diceTP-Rec-Prec-Spec-F1-bAcc', subdirs=[subdir])
        # self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Object-wise recall',
        #                                          'Object-wise precision', 'Object-wise F1'],
        #                                 suffix='seg_scores_DiceP_Dice_TP_obj-wise_Rec_Prec_F1', subdirs=[subdir])
        self.write_cutoff_results_latex(metrics=['Patient-wise recall postop', 'Patient-wise precision postop', 'Specificity',
                                                 'Patient-wise F1 postop', 'Balanced accuracy'],
                                        suffix='classif_scores-Rec-Prec-Spec-F1-bAcc', subdirs=[subdir])
        self.plot_all_metrics(subdir=subdir)

    def read_results_subdir_all_exp(self, subdir=''):
        self.folds_metrics_average_list = []
        self.overall_metrics_average_list = []
        self.results_list = []
        self.results_cutoff_list = []

        for arch_index in range(len(self.input_study_dirs)):
            self.folds_metrics_average_list.append([])
            self.overall_metrics_average_list.append([])
            self.results_list.append([])
            self.results_cutoff_list.append([])

            for study in self.studies_lists[arch_index]:
                folds_metrics_filepath = Path(self.input_study_dirs[arch_index], study, subdir,
                                              f'folds_metrics_average_volume_cutoff.csv')
                metrics = pd.read_csv(str(folds_metrics_filepath))
                self.folds_metrics_average_list[arch_index].append(metrics)

                overall_metrics_filepath = Path(self.input_study_dirs[arch_index], study, subdir,
                                              f'overall_metrics_average_volume_cutoff.csv')
                metrics = pd.read_csv(str(overall_metrics_filepath))
                self.overall_metrics_average_list[arch_index].append(metrics)

                results_filepath = Path(self.input_study_dirs[arch_index], study, subdir, 'all_dice_scores_clean.csv')
                metrics = pd.read_csv(str(results_filepath))
                self.results_list[arch_index].append(metrics)

                results_cutoff_filepath = Path(self.input_study_dirs[arch_index], study, subdir, 'all_dice_scores_volume_cutoff.csv')
                metrics = pd.read_csv(str(results_cutoff_filepath))
                self.results_cutoff_list[arch_index].append(metrics)

    def load_optimal_results(self, arch, study, subdir):
        metrics_filepath = Path(self.input_study_dirs[arch], study, subdir, 'all_dice_scores_volume_cutoff.csv')
        results = pd.read_csv(str(metrics_filepath))
        _, optimal_threshold = self.__retrieve_optimum_values(arch, study)
        results.replace('inf', np.nan, inplace=True)
        results.replace('', np.nan, inplace=True)
        results.replace(' ', np.nan, inplace=True)
        thresh_index = (np.round(results['Threshold'], 1) == optimal_threshold)
        optimal_results = results.loc[thresh_index]
        return optimal_results

    def read_dice_scores(self, arch_index, study_index, subdir=''):
        self.all_dice_scores_list.append([])
        for study in self.studies_lists[arch_index]:
            all_dice_scores_filepath = Path(self.input_study_dirs[arch_index].parent,
                                            self.studies_lists[arch_index][study_index],
                                            subdir, f'all_dice_scores_clean.csv')
            metrics = pd.read_csv(str(all_dice_scores_filepath))
            self.all_dice_scores_list[arch_index].append(metrics)

    def write_cutoff_results_latex(self, metrics=['Dice', 'Dice-TP', 'Dice-P', 'Patient-wise recall postop',
                                                 'Patient-wise precision postop', 'Patient-wise F1 postop'],
                                    suffix='', subdirs=[]):
        latex_table_fname = 'key_metrics_after_cutoff_latex.txt' if suffix == '' else f'results_{suffix}_latex.txt'
        latex_table_fpath = Path(self.output_dir, subdirs[0], latex_table_fname) if len(subdirs) == 1 else \
            Path(self.output_dir, latex_table_fname)
        pfile = open(latex_table_fpath, 'w+')
        n_arch = len(self.studies_lists)

        for i in range(len(self.studies_lists[0])):
            pfile.write(f"\\hline \n")
            for k, subdir in enumerate(subdirs):
                for j in range(n_arch):
                    # output_string = self.arch_names[j] + " & " + self.studies_description[i] + " & " + self.subdir_labels[k]
                    output_string = ""
                    output_string += "\\multirow{4}{*}{" + self.studies_description[i] + "}" if j == 0 and k == 0 else ""
                    output_string += " & \\multirow{2}{*}{" + self.subdir_labels[k] + "}" if j == 0 else " & "
                    output_string += " & " + self.arch_names[j]

                    fname = Path(self.input_study_dirs[j], self.studies_lists[j][i], subdir, f'overall_metrics_average_volume_cutoff.csv')
                    results = pd.read_csv(fname)

                    for m in metrics:
                        if m in ['HD95', 'Absolute volume error', 'Absolute volume error_median']:
                            scaling_factor = 1
                        else:
                            scaling_factor = 100

                        if (m + '_std' in results.columns) and (results.loc[0, m + '_std'] > 0):
                            output_string += f" & {results.loc[0, m + '_mean'] * scaling_factor:.2f}$\pm${results.loc[0, m + '_std'] * scaling_factor:.2f}"
                        elif m == 'Absolute volume error_median':
                            output_string += f" & {results.loc[0, m] * scaling_factor:.2f}"
                        else:
                            output_string += f" & {results.loc[0, m + '_mean'] * scaling_factor:.2f}"

                    pfile.write(output_string + "\\tabularnewline \n")
        pfile.close()

    def __retrieve_optimum_values(self, arch, study):
        study_filename = os.path.join(self.input_study_dirs[arch], study, 'Validation', 'optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this.')
        return reload_optimal_validation_parameters(study_filename=study_filename)
    def plot_all_metrics(self, subdir=''):
        dice_metrics = ['Dice', 'Dice-P', 'Dice-TP', 'Dice-N']
        dice_xticks = ['Global Dice', 'Dice - Positives (True pos + false neg)', 'Dice - True positives', 'Dice - Negatives (True neg + false pos)']
        self.plot_metrics(dice_metrics,
                          'dice_metrics.png', f'Dice scores for different subgroups after classification - {subdir}', figsize=(17, 10),
                          subdir=subdir, legend_loc=1, xticks=dice_xticks)

        classification_metrics = ['Patient-wise recall postop', 'Specificity', 'Patient-wise precision postop',
                                  'Patient-wise F1 postop', 'Balanced accuracy', 'Accuracy']
        classif_xticks = ['Sensitivity / Recall', 'Specificity', 'Precision',
                          'F1', 'Balanced acc (Mean(sens., spec.))', 'Accuracy']
        self.plot_metrics(classification_metrics,
                          'res_tumor_classification.png',
                          f'Patient-wise metrics for classification: residual tumor / gross total resection - {subdir}',
                          figsize=(18, 10),
                          subdir=subdir, xticks=classif_xticks, legend_loc=4)

        # specific_classification_metrics = ['Patient-wise recall postop', 'Patient-wise precision postop',
        #                                    'Patient-wise F1 postop', 'Object-wise recall', 'Object-wise precision',
        #                                    'Object-wise F1']
        # self.plot_metrics(specific_classification_metrics,
        #                   'specific_classification_metrics.png',
        #                   f'Patient-wise and object-wise classification metrics - {subdir}',
        #                   figsize=(16, 10),
        #                   subdir=subdir)
        #
        # metrics_cutoff = ['Patient-wise recall postop', 'Patient-wise precision postop',
        #                   'Patient-wise F1 postop', 'Accuracy', 'Balanced accuracy',
        #                   'Positive rate', 'Specificity', 'Dice-P', 'Dice-TP']
        # self.plot_metrics(metrics_cutoff,
        #                   'metrics_volume_cutoff.png',
        #                   f'Classification metrics and DSC after volume cutoff - {subdir}',
        #                   figsize=(18, 10), cutoff=True, subdir=subdir)
        # self.plot_metrics(metrics_cutoff,
        #                   'metrics_before_volume_cutoff.png',
        #                   f'Classification metrics and DSC before volume cutoff - {subdir}',
        #                   figsize=(18, 10), cutoff=False, subdir=subdir)
        #
        # key_metrics = ['Patient-wise recall postop', 'Patient-wise precision postop',
        #                'Patient-wise F1 postop', 'Dice-P', 'Dice-TP']
        # self.plot_metrics(key_metrics,
        #                   'key_metrics_volume_cutoff.png',
        #                   f'Metrics for post-operative segmentation performance',
        #                   figsize=(16, 10), cutoff=True, subdir=subdir)

    def plot_metrics(self, metric_names, file_name, plot_title, figsize=(14, 10), subdir='', xticks=None,
                     legend_loc=1):
        index = metric_names
        std_err = []
        scores = []
        n_arch = len(self.studies_lists)

        for i in range(len(self.studies_lists[0])):
            for j in range(n_arch):
                scores.append([self.overall_metrics_average_list[j][i][ind + '_mean'][0] for ind in index])
                std_err.append([self.overall_metrics_average_list[j][i][ind + '_std'][0] for ind in index])

        fig, ax = plt.subplots(figsize=figsize)
        rects = []
        ind = np.arange(len(index))
        width = 0.8 / len(self.studies_lists[0] * n_arch)

        legends = []
        for i in range(len(self.studies_lists[0])):
            for j in range(n_arch):
                rect = ax.bar(ind + width * (i*2+j), scores[i*2+j], width, color=self.colors[j][i], yerr=std_err[i*2+j])
                rects.append(rect)
                legends.append(f"{self.arch_names[j]}, {self.studies_description[i]}")

        ax.set_ylabel('Scores', fontsize=18)
        ax.set_title(plot_title, fontsize=20)
        ax.set_xticks(ind + (len(self.studies_lists[0])*n_arch - 1) * width / 2)
        ax.set_xticklabels(index if xticks is None else xticks, fontsize=18, rotation=15)
        ax.set_ylim([0, 1])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)

        for i, rect in enumerate(rects):
            for j, r in enumerate(rect):
                height = r.get_height()
                text = round(scores[i][j], 2)
                ax.text(r.get_x() + r.get_width() / 2, height + 0.01,
                        text, ha='center', va='bottom', fontsize=16)

        ax.legend((r[0] for r in rects), (l for l in legends), loc=legend_loc, fontsize=16)
        plot_output_path = Path(self.output_dir, subdir, file_name)
        print(f"Saving figure at {plot_output_path}")
        plt.tight_layout()
        plt.savefig(plot_output_path)

    def create_interrater_consensus_segmentations(self):
        interrater_data_path = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/interrater_data')
        interrater_glioblastoma_data_path = Path(interrater_data_path, 'Glioblastoma')

        patient_folders = [d for d in interrater_glioblastoma_data_path.iterdir() if d.is_dir()]
        annotator_groups = ['nov', 'exp']
        annotators = [[f"{annot}{i}" for i in range(1, 5)] for annot in annotator_groups]

        for ind, patient_folder in enumerate(patient_folders):
            loaded_annotations = []

            # Load T1c image to check shape
            annotation_folder = Path(patient_folder, 'ses-postop', 'anat')
            image_filepath = [f for f in annotation_folder.iterdir() if 'NativeT1c' in f.name and 'dseg' not in f.name][
                0]
            image_ni = nib.load(image_filepath)
            all_annotations = [f for f in annotation_folder.iterdir() if
                               'label' in f.name and 'TMRenh_dseg' in f.name and
                               'NativeT1c' in f.name]

            all_loaded_annotations = []
            for j, group in enumerate(annotator_groups):
                group_loaded_annotations = []
                for annotator in annotators[j]:
                    annotation_filepath = [f for f in all_annotations if annotator in f.name]
                    if len(annotation_filepath) == 0:
                        # print(f"No annotations for patient {pid} annot {annotator}, creating empty mask")
                        annotation_ni = nib.Nifti1Image(np.zeros(image_ni.shape), affine=image_ni.affine)
                    else:
                        annotation_ni = nib.load(annotation_filepath[0])
                        if len(annotation_ni.shape) == 4:
                            annotation_ni = nib.four_to_three(annotation_ni)[0]
                    group_loaded_annotations.append(annotation_ni.get_data())
                    all_loaded_annotations.append(annotation_ni.get_data())

                consensus_group = np.mean(np.array(group_loaded_annotations), axis=0)
                consensus_group[consensus_group > 0.5] = 1
                consensus_group[consensus_group <= 0.5] = 0
                output_filepath = Path(annotation_folder, image_filepath.name.split('.')[0] + f'_label-strict-consensus-{group}TMRenh_dseg.nii.gz')
                nib.save(nib.Nifti1Image(consensus_group, affine=image_ni.affine), output_filepath)

                # print(consensus_group.shape, np.min(consensus_group), np.max(consensus_group))

            consensus_all = np.mean(np.array(all_loaded_annotations), axis=0)
            consensus_all[consensus_all > 0.5] = 1
            consensus_all[consensus_all <= 0.5] = 0
            output_filepath = Path(annotation_folder,
                                   image_filepath.name.split('.')[0] + f'_label-strict-consensus-all-annotatorsTMRenh_dseg.nii.gz')
            nib.save(nib.Nifti1Image(consensus_all, affine=image_ni.affine), output_filepath)
            # print(consensus_all.shape, np.min(consensus_all), np.max(consensus_all))

    def interrater_study(self):
        interrater_data_path = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/interrater_data')
        interrater_glioblastoma_data_path = Path(interrater_data_path, 'Glioblastoma')

        patient_folders = [d for d in interrater_glioblastoma_data_path.iterdir() if d.is_dir()]
        patient_op_ids = [int(d.name.split('-')[1][4:]) for d in patient_folders]

        patient_id_mapping_filepath = Path(
            '/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/patient_id_mapping.csv')
        id_df = pd.read_csv(patient_id_mapping_filepath)
        vumc_df = id_df[(id_df['Hospital'] == 'VUmc')]

        annotator_groups = ['nov', 'exp']
        all_annotators = sum([[f"{annot}{i}" for i in range(1, 5)] for annot in annotator_groups], [])
        consensus_annotators = ['strict-consensus-nov', 'strict-consensus-exp', 'strict-consensus-all-annotators']
        # consensus_annotators = ['consensus-nov', 'consensus-exp', 'consensus-all-annotators']

        all_annotators += consensus_annotators
        # all_annotators = consensus_annotators
        metrics = ['Dice', 'Jaccard', 'volume_seg', 'residual_tumor_prediction']
        columns = ['reference', 'pid', 'opid', 'volume', 'residual_tumor', 'annotator/model'] + metrics

        self.interrater_study_filepath = Path(self.output_dir, 'interrater_study.csv')
        if not self.interrater_study_filepath.exists():
            self.interrater_results_df = pd.DataFrame(columns=columns)
        else:
            self.interrater_results_df = pd.read_csv(self.interrater_study_filepath)

        # indx = [8, 10]
        # patient_op_ids = [patient_op_ids[i] for i in indx]
        # patient_folders = [patient_folders[i] for i in indx]
        for ind, (op_id, patient_folder) in enumerate(zip(patient_op_ids, patient_folders)):
            patient = vumc_df[vumc_df['OP.ID'] == op_id]
            annotation_folder = Path(patient_folder, 'ses-postop', 'anat')
            pid = patient['DB_ID'].values[0]
            db_index = patient['DB_index'].values[0]

            # Load reference segmentations
            references = self.__load_interrater_references(pid, db_index, annotation_folder)

            print(f"Compute scores for patient pid {pid} / opid {op_id}")
            for ref_name, ref_ni in references.items():
                print(f"Reference: {ref_name}")

                # Get segmentations
                segmentations = self.__load_evaluator_segmentations(pid, db_index, patient, ref_ni, annotation_folder,
                                                                    all_annotators)
                segmentations['ground_truth_segmentation'] = [deepcopy(references['ground_truth_segmentation']), 0.5]
                # If any of them does not exist - compute reference volume + res tumor
                ref_volume, ref_residual_tumor = compute_volume_residual_tumor(ref_ni,
                                                                               threshold_segmentation(ref_ni, 0.5))
                ref_basic_info = [ref_name, pid, op_id, ref_volume, ref_residual_tumor]

                for eval_name, (seg_ni, thresh) in segmentations.items():
                    # Check for entries in results
                    eval_res = self.interrater_results_df.loc[(self.interrater_results_df['reference'] == ref_name) &
                                                              (self.interrater_results_df['pid'] == pid) &
                                                              (self.interrater_results_df[
                                                                   'annotator/model'] == eval_name)]
                    if len(eval_res) != 0 and not np.isnan(np.sum(eval_res.values[6:])):
                        continue

                    if not seg_ni.shape == ref_ni.shape:
                        print(
                            f"Mismatch in shape between {eval_name} segmentation and GT for patient {pid} / opid {op_id}, skip")
                        continue
                    results = dice_computation(ref_ni, seg_ni, thresh)
                    results_df = pd.DataFrame([ref_basic_info + [eval_name] + results], columns=columns)

                    self.interrater_results_df = self.interrater_results_df.append(results_df, ignore_index=True)
                    self.interrater_results_df.to_csv(self.interrater_study_filepath, index=False)

    def __load_interrater_references(self, pid, db_index, annotation_folder):
        references = {}
        image_filepath = [f for f in annotation_folder.iterdir() if 'NativeT1c' in f.name and 'dseg' not in f.name][
            0]
        image_ni = nib.load(image_filepath)
        print(image_ni.shape)

        # Load ground truth segmentation
        gt_data_path = Path(self.data_root, str(db_index), str(pid), 'segmentations')
        gt_files = [f for f in gt_data_path.iterdir() if ('T1' in f.name) and
                    ('post' in f.name) and ('label_tumor' in f.name) and
                    (not 'T1Woc' in f.name)]
        if len(gt_files) > 1:
            print(f"Found several possible GT annotations for patient {pid}, skip")
            return references

        gt_filename = gt_files[0]
        gt_ni = nib.load(gt_filename)
        if len(gt_ni.shape) == 4:
            gt_ni = nib.four_to_three(gt_ni)[0]

        if not image_ni.shape == gt_ni.shape:
            print(f"Mismatch in shape between T1c image and GT for patient {pid}, skip")
            print(f"GT shape = {gt_ni.shape}, im shape = {image_ni.shape}")
        else:
            references['ground_truth_segmentation'] = gt_ni

        # Load consensus segmentation(s)
        # consensus_annotation_files = [f for f in annotation_folder.iterdir() if 'consensus-all-annotators' in f.name]
        consensus_annotation_files = [f for f in annotation_folder.iterdir() if 'strict-consensus' in f.name]
        # consensus_annotation_files = [f for f in annotation_folder.iterdir() if 'consensus' in f.name]
        for f in consensus_annotation_files:
            consensus_identifier = re.search('label-(.*)TMRenh_dseg', f.name).group(1)
            consensus_ni = nib.load(f)

            if len(consensus_ni.shape) == 4:
                consensus_ni = nib.four_to_three(consensus_ni)[0]

            if not image_ni.shape == consensus_ni.shape:
                print(
                    f"Mismatch in shape between T1c image and {consensus_identifier} segmentation for patient {pid}, skip")
            else:
                references[consensus_identifier] = consensus_ni

        return references

    def __load_evaluator_segmentations(self, pid, db_index, patient, ref_ni, annotation_folder, annotators):
        segmentations_thresholds = {}

        # Load all annotations
        all_annotations = [f for f in annotation_folder.iterdir() if 'label' in f.name and 'TMRenh_dseg' in f.name and
                           'NativeT1c' in f.name]
        for annotator in annotators:
            if not 'strict' in annotator:
                annotation_filepath = [f for f in all_annotations if annotator in f.name and not 'strict' in f.name]
            else:
                annotation_filepath = [f for f in all_annotations if annotator in f.name]

            if len(annotation_filepath) == 0:
                # print(f"No annotations for patient {pid} annot {annotator}, creating empty mask")
                annotation_ni = nib.Nifti1Image(np.zeros(ref_ni.shape), affine=ref_ni.affine)
            else:
                annotation_ni = nib.load(annotation_filepath[0])
                if len(annotation_ni.shape) == 4:
                    annotation_ni = nib.four_to_three(annotation_ni)[0]

            segmentations_thresholds[annotator] = [annotation_ni, 0.5]

        # Load all segmentations from models
        for i, arch in enumerate(self.arch_names):
            # for j, study in enumerate(self.studies_lists[i]):
            # j = 4
            for j, study in enumerate(self.studies_lists[i]):
                # Find prediction file
                prediction_dir = Path(self.input_study_dirs[i], study, 'test_predictions', '0')
                self.arch_names = ['nnU-Net', 'AGU-Net']
                if arch == 'AGU-Net':
                    patient_folder = Path(prediction_dir, f'{db_index}_{pid}')
                    prediction_file = [f for f in patient_folder.iterdir() if f.is_file() and 'pred_tumor' in f.name][0]
                else:
                    # nnunet_index = patient['nnU-Net_ID'].values[0]
                    # patient_folder = Path(prediction_dir, '0', f'{nnunet_index}')
                    patient_folder = Path(prediction_dir, f'HGG_{pid}')
                    prediction_file = [f for f in patient_folder.iterdir() if f.is_file() and 'pred_tumor' in f.name][0]

                pred_ni = nib.load(prediction_file)
                # Load optimal threshold and threshold predictions
                optimal_dice_study = Path(self.input_study_dirs[i], study, 'Validation',
                                          'optimal_dice_study.csv')
                optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=optimal_dice_study)
                segmentations_thresholds[f"{arch}_{self.studies_description[j]}"] = [pred_ni, optimal_threshold]

        return segmentations_thresholds

    def interrater_study_summary(self):
        results = pd.read_csv(self.interrater_study_filepath)
        results.replace('inf', 0, inplace=True)
        results.replace('', 0, inplace=True)
        results.replace(' ', 0, inplace=True)

        # Compute metrics
        eval_metrics = ['Dice', 'Dice-P', 'Jaccard', 'Jaccard-P']
        average_columns = ['reference', 'evaluator', 'Patient-wise recall', 'Patient-wise precision',
                           'Patient-wise specificity',
                           'Patient-wise F1', 'Accuracy', 'Balanced accuracy', 'Positive rate', 'Negative rate']
        for m in eval_metrics:
            average_columns.extend([m + '_mean', m + '_std'])

        unique_references = np.unique(results['reference'])
        # unique_evaluators = np.unique(['_'.join(colname.split('_')[:-1]) for colname in results.columns if ('Dice' in colname or 'Jaccard' in colname)])
        unique_evaluators = np.unique(results['annotator/model'])
        print(len(unique_evaluators), unique_evaluators)

        metrics_per_ref_evaluator = []
        for ref in unique_references:
            ref_results = results.loc[results['reference'] == ref]

            for evaluator in unique_evaluators:
                evaluator_average = []
                eval_results = ref_results.loc[results['annotator/model'] == evaluator]
                true_pos = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_prediction'] == 1) & (eval_results['residual_tumor'] == 1)])
                false_pos = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_prediction'] == 1) & (eval_results['residual_tumor'] == 0)])
                true_neg = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_prediction'] == 0) & (eval_results['residual_tumor'] == 0)])
                false_neg = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_prediction'] == 0) & (eval_results['residual_tumor'] == 1)])
                # print(f"{evaluator}: {sum([true_pos, false_pos, true_neg, false_neg])}")

                recall = 1 if (true_pos + false_neg) == 0 else true_pos / (true_pos + false_neg)
                precision = 1 if (true_pos + false_pos) == 0 else true_pos / (true_pos + false_pos)
                specificity = 1 if (true_neg + false_pos) == 0 else true_neg / (true_neg + false_pos)
                f1 = 2 * ((recall * precision) / (recall + precision))
                accuracy = (true_pos + true_neg) / sum([true_pos, false_pos, true_neg, false_neg])
                balanced_acc = (recall + specificity) / 2
                pos_rate = (true_pos + false_neg) / len(eval_results)
                neg_rate = (true_neg + false_pos) / len(eval_results)
                evaluator_average.extend([recall, precision, specificity, f1, accuracy, balanced_acc,
                                          pos_rate, neg_rate])

                for m in eval_metrics:
                    if '-P' in m:
                        positives = eval_results.loc[(eval_results['residual_tumor'] == 1)]
                        # avg = positives[f'{evaluator}_{m.split("-")[0]}'].astype('float32').mean()
                        # std = positives[f'{evaluator}_{m.split("-")[0]}'].astype('float32').std()
                        avg = positives[m.split('-')[0]].astype('float32').mean()
                        std = positives[m.split('-')[0]].astype('float32').std()
                        evaluator_average.extend([avg, std])
                    else:
                        # avg = ref_results[f'{evaluator}_{m.split("-")[0]}'].astype('float32').mean()
                        # std = ref_results[f'{evaluator}_{m.split("-")[0]}'].astype('float32').std()
                        avg = ref_results[m].astype('float32').mean()
                        std = ref_results[m].astype('float32').std()
                        evaluator_average.extend([avg, std])

                metrics_per_ref_evaluator.append([ref, evaluator] + evaluator_average)

        metrics_df = pd.DataFrame(metrics_per_ref_evaluator, columns=average_columns)
        output_filepath = Path(self.output_dir, 'interrater_study_average_metrics.csv')
        metrics_df.to_csv(output_filepath, index=False)

    def visualize_interrater_results(self, references=[], evaluators=[]):
        # Results to plot
        evaluators = ['nov1', 'nov2', 'nov3', 'nov4', 'exp1', 'exp2', 'exp3', 'exp4',
                      'nnU-Net_D', 'AGU-Net_E']
        references = ['ground_truth_segmentation', 'strict-consensus-all-annotators']

        # Plot parameters
        palettes = [sns.color_palette("light:seagreen", n_colors=6),
                    sns.color_palette("light:b", n_colors=6),
                    sns.color_palette("light:m", n_colors=7),
                    sns.color_palette("light:#fdae6b", n_colors=6)]

        palette = {f'nov{i + 1}': palettes[0][i+1] for i in range(4)}
        palette.update({f'exp{i+1}': palettes[1][i+1] for i in range(4)})
        palette.update({evaluators[-1]: palettes[2][2], evaluators[-2]: palettes[2][3]})
        palette.update({'ground_truth_segmentation': palettes[3][2],  'strict-consensus-all-annotators': palettes[3][2]})
        figsize = (18, 8)
        tick_fontsize = 16
        label_fontsize = tick_fontsize + 1
        title_fontsize = label_fontsize + 1
        all_refs_one_fig = True
        plt.rcParams["font.family"] = "serif"

        # Load and process results
        study_filepath = Path(self.output_dir, 'interrater_study.csv')
        results = pd.read_csv(study_filepath)
        results.replace('inf', 0, inplace=True)
        results.replace('', 0, inplace=True)
        results.replace(' ', 0, inplace=True)

        if len(references) == 0:
            references = np.unique(results['reference'])
        if len(evaluators) == 0:
            evaluators = np.unique(results['annotator/model'])

        # results = results[results['annotator/model'].isin(evaluators)]

        reference_names = {'strict-consensus-all-annotators': "Consensus agreement annotation",
                           'ground_truth_segmentation': "Ground truth annotation"}
        ref_short = {'ground_truth_segmentation': 'Ground truth',
                     'strict-consensus-all-annotators': 'Consensus'}
        if all_refs_one_fig:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            b_list = []
            for i, ref in enumerate(references):
                evaluators_w_ref = evaluators + [ev for ev in references if ev != ref]
                eval_results = results.loc[results['annotator/model'].isin(evaluators_w_ref)]
                sort_dict = {annot: i for i, annot in enumerate(evaluators_w_ref)}
                eval_results = eval_results.sort_values(by=['annotator/model'], key=lambda x: x.map(sort_dict))
                ref_results = eval_results.loc[eval_results['reference'] == ref]
                ref_results_positive = ref_results.loc[ref_results['residual_tumor'] == 1]

                b_list.append(sns.boxplot(ax=axes[i], data=ref_results_positive, x='annotator/model', y='Jaccard',
                                palette=palette))

                xticks = evaluators_w_ref.copy()
                xticks[-1] = ref_short[xticks[-1]]
                b_list[i].set_xticklabels(xticks, rotation=45, fontsize=tick_fontsize)
                b_list[i].set(ylim=(0, 1))
                b_list[i].set_yticklabels(np.round(b_list[i].get_yticks(), 1), fontsize=tick_fontsize)
                b_list[i].set_xlabel("Annotator / Model", fontsize=label_fontsize)
                b_list[i].set_ylabel("Jaccard score", fontsize=label_fontsize)

                b_list[i].set_title(f"Reference:  {reference_names[ref]}", fontsize=title_fontsize)

            plt.tight_layout()
            plt.savefig(Path(self.output_dir, f'jaccard_scores_all_refs.png'), dpi=300)

        else:
            for i, ref in enumerate(references):
                ref_results = results.loc[results['reference'] == ref]
                ref_results_positive = ref_results.loc[ref_results['residual_tumor'] == 1]

                plt.figure(figsize=figsize)
                b = sns.boxplot(data=ref_results_positive, x='annotator/model', y='Jaccard',
                                palette=palette)

                b.set_xticklabels(b.get_xticklabels(), rotation=45, fontsize=tick_fontsize)
                b.set(ylim=(0, 1))
                b.set_yticklabels(np.round(b.get_yticks(), 1), fontsize=tick_fontsize)
                b.set_xlabel("Annotator / Model", fontsize=label_fontsize)
                b.set_ylabel("Jaccard score", fontsize=label_fontsize)

                plt.title(f"Reference:  {reference_names[ref]}", fontsize=title_fontsize)
                plt.tight_layout()
                plt.savefig(Path(self.output_dir, f'jaccard_scores_ref_{ref}.png'))

def dice_computation(reference_ni, detection_ni, t=0.5):
    reference = threshold_segmentation(reference_ni, t)
    detection = threshold_segmentation(detection_ni, t)

    dice = compute_dice(reference, detection)
    jaccard = dice / (2-dice)
    volume_seg_ml, res_tumor = compute_volume_residual_tumor(detection_ni, detection)
    return [dice, jaccard, volume_seg_ml, res_tumor]

def threshold_segmentation(segmentation_ni, t):
    segmentation = deepcopy(segmentation_ni.get_data())
    segmentation[segmentation <= t] = 0
    segmentation[segmentation > t] = 1
    return segmentation.astype('uint8')


def compute_volume_residual_tumor(segmentation_ni, segmentation):
    voxel_size = np.prod(segmentation_ni.header.get_zooms()[0:3])
    volume_seg_ml = compute_tumor_volume(segmentation, voxel_size)
    res_tumor = 1 if volume_seg_ml > 0.175 else 0
    return volume_seg_ml, res_tumor

def tukey_hsd (lst, ind, n, alpha=0.05):
    data_arr = np.hstack(lst)
    ind_arr = np.repeat(ind, n)
    return pairwise_tukeyhsd(data_arr, ind_arr, alpha=alpha)

def sample(data):
    sample = [random.choice(data) for _ in np.arange(len(data))]
    return sample

def create_prediction_vector(results):
    pred = np.zeros(results.shape[0])
    pred[results.loc[(results['#Det'] > 0)].index] = 1

    gt = np.zeros(results.shape[0])
    gt[results.loc[(results['#GT'] > 0)].index] = 1
    return pred, gt

def test_statistic(pred1, pred2, gt):
    # bacc1 = sklearn.metrics.balanced_accuracy_score(gt, pred1)
    # bacc2 = sklearn.metrics.balanced_accuracy_score(gt, pred2)
    bacc1 = balanced_accuracy(gt, pred1)
    bacc2 = balanced_accuracy(gt, pred2)
    return bacc1 - bacc2

def balanced_accuracy(gt, pred):
    tp = len(np.where(np.logical_and(gt==1, pred==1))[0])
    fn = len(np.where(np.logical_and(gt==1, pred==0))[0])
    fp = len(np.where(np.logical_and(gt==0, pred==1))[0])
    tn = len(np.where(np.logical_and(gt==0, pred==0))[0])
    sensitivity = tp / (tp + fn)
    specificity = 1 if (tn + fp) == 0 else tn / (tn + fp)
    return (sensitivity + specificity) / 2

def bootstrap_t_test(pred1, pred2, gt, nboot = 1000, direction = "less"):
    mu_pred1 = balanced_accuracy(gt, pred1)
    mu_pred2 = balanced_accuracy(gt, pred2)
    tobs = mu_pred2 - mu_pred1

    print(mu_pred1, mu_pred2, tobs)

    N = len(pred1)
    #tboot = np.zeros(nboot)
    tboot = []
    for i in np.arange(nboot):
        current_random_indices = np.random.choice(np.arange(N), N, replace=True)
        curr_pred1 = pred1[current_random_indices]
        curr_pred2 = pred2[current_random_indices]
        curr_gt = gt[current_random_indices]

        boot_mu_pred1 = balanced_accuracy(curr_gt, curr_pred1)
        boot_mu_pred2 = balanced_accuracy(curr_gt, curr_pred2)
        boot_tobs = boot_mu_pred2 - boot_mu_pred1

        #print(boot_mu_pred1, boot_mu_pred2, boot_tobs, np.mean([boot_mu_pred1, boot_mu_pred2]))

        tboot.append(boot_mu_pred2 - np.mean([boot_mu_pred1, boot_mu_pred2]))

        #tboot.append(test_statistic(curr_pred1, curr_pred2, curr_gt) - )

        #sboot = sample(Z)
        #sboot = pd.DataFrame(np.array(sboot), columns=['treat', 'vals', 'gt'])
        #tboot[i] = test_statistic(sboot['vals'][sboot['treat'] == 1], np.mean(sboot['vals'][sboot['treat'] == 0]),
                                  #)
        #tboot[i] = np.mean(sboot['vals'][sboot['treat'] == 1]) - np.mean(sboot['vals'][sboot['treat'] == 0]) - tstat

    tboot = np.asarray(tboot)
    print(tboot)
    print(tobs)
    if direction == "greater":
        pvalue = np.sum(tboot >= tobs) / nboot
    elif direction == "less":
        pvalue = np.sum(tboot <= tobs) / nboot
    elif direction == "both":
        pvalue = np.sum(np.abs(tboot) >= tobs) / nboot
    else:
        print('Enter a valid arg for direction')
    return pvalue
    # print('The p-value is %f' % (pvalue))


def BCa_interval_macro_metric(pred, gt, func, B=1000, q=0.975):
    theta_hat = func(pred, gt)

    N = len(pred)
    order = np.array(range(N))
    order_boot = np.random.choice(order, size=(B, N), replace=True)
    pred_boot = pred[order_boot]
    gt_boot = gt[order_boot]

    # bootstrap
    theta_hat_boot = []
    for i in range(pred_boot.shape[0]):
        theta_hat_boot.append(func(pred_boot[i], gt_boot[i]))
    theta_hat_boot = np.asarray(theta_hat_boot)
    #theta_hat_boot = np.array([func(pred_boot[i]) for i in range(X_boot.shape[0])])

    # 1) find jackknife estimates
    tmp = np.transpose(np.reshape(np.repeat(order, repeats=len(order)), (len(order), len(order))))  # make NxN matrix
    tmp_mat = tmp[~np.eye(tmp.shape[0], dtype=bool)].reshape(tmp.shape[0], -1)
    #X_tmp_mat = X[tmp_mat]
    pred_tmp_mat = pred[tmp_mat]
    gt_tmp_mat = gt[tmp_mat]

    #jk_theta = np.array([func(X_tmp_mat[i]) for i in range(X_tmp_mat.shape[0])])
    jk_theta = np.array([func(pred_tmp_mat[i], gt_tmp_mat[i]) for i in range(pred_tmp_mat.shape[0])])
    phi_jk = np.mean(jk_theta) - jk_theta  # jackknife estimates

    # 2) Find a
    a = 1 / 6 * np.sum(phi_jk ** 3) / np.sum(phi_jk ** 2) ** (3 / 2)

    # 3) Find b
    b = norm.ppf(np.sum(theta_hat_boot < theta_hat) / B)  # inverse standard normal

    # 4) Find gamma values -> limits
    def gamma1_func(a, b, q):
        return norm.cdf(b + (b + norm.ppf(1 - q)) / (1 - a * (b + norm.ppf(1 - q))))

    def gamma2_func(a, b, q):
        return norm.cdf(b + (b + norm.ppf(q)) / (1 - a * (b + norm.ppf(q))))

    # 5) get confidence interval of BCa
    CI_BCa = np.percentile(theta_hat_boot, [100 * gamma1_func(a, b, q), 100 * gamma2_func(a, b, q)])

    return CI_BCa, theta_hat_boot, theta_hat
