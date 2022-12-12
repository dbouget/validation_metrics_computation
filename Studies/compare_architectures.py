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
from Validation.kfold_model_validation import compute_dice, compute_dice_uncertain, compute_tumor_volume
from PIL import Image
from pathlib import Path
import seaborn as sns
import re
from .hgg_postop_segmentation import threshold_volume_and_compute_classification_metrics


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
        self.input_study_dirs = [Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/Alexandros_experiments/compare_nnUNet_501-505'),
                                 Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/compare_run2_exp1-5')]

        #self.studies = ['Ams_Trd_T1c', 'Ams_Trd_T1c_T1w', 'Ams_Trd_T1c_T1w_FLAIR', 'All_T1c_T1w_preop']
        self.studies_lists = [['501', '502', '503', '504', '505'], #'504',
            ['run2_exp1_T1c', 'run2_exp2_T1c_T1w', 'run2_exp3_T1c_T1w_flair', 'run2_exp4_T1c_T1w_preop', 'run2_exp5_T1c_T1w_flair_preop']] #
        self.studies_description = ['T1ce', 'T1ce+T1w', 'T1ce+T1w+FLAIR', 'T1ce+T1w+Preop T1ce', 'T1ce+T1w+FLAIR+Preop T1ce'] #

        palette_names = ['cubehelix', 'cubehelix']
        palette_names = ['flare', 'crest']
        palette_names = ["light:salmon","light:b"]
        self.palettes = [sns.color_palette(palette_names[0], n_colors=6), sns.color_palette(palette_names[1], n_colors=7, desat=1)]
        self.colors = [[pal[i+1] for i in range(len(self.studies_lists[0]))] for pal in self.palettes]

        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            self.extra_patient_parameters.loc[:, 'Patient'] = self.extra_patient_parameters.Patient.astype(int).astype(
                str)

    def run(self):
        # self._run_subdir('Validation')
        # self._run_subdir('Test')
        # self.create_interrater_consensus_segmentations()
        # self.interrater_study()
        # self.interrater_study_summary()
        self.visualize_interrater_results()
        return

    def _run_subdir(self, subdir):

        output_dir = Path(self.output_dir, subdir)
        output_dir.mkdir(exist_ok=True)

        self.folds_metrics_average_list = []
        self.overall_metrics_average_list = []

        for i in range(len(self.input_study_dirs)):
            self.read_results_arch(i, subdir=subdir)

        print("OK")
        self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Patient-wise recall postop',
                                                 'Patient-wise precision postop',  'Specificity', 'Patient-wise F1 postop'],
                                        suffix='diceP-diceTP-Rec-Prec-Spec-F1', subdir=subdir)
        self.write_cutoff_results_latex(metrics=['Dice-P', 'Dice-TP', 'Patient-wise recall postop',
                                                 'Patient-wise precision postop', 'Specificity',
                                                 'Patient-wise F1 postop', 'Balanced accuracy'],
                                        suffix='diceP-diceTP-Rec-Prec-Spec-F1-bAcc', subdir=subdir)
        self.plot_all_metrics(subdir=subdir)

    def read_results_arch(self, arch_index, subdir=''):
        self.folds_metrics_average_list.append([])
        self.overall_metrics_average_list.append([])

        for study in self.studies_lists[arch_index]:
            folds_metrics_filepath = Path(self.input_study_dirs[arch_index], subdir,
                                          f'folds_metrics_average_{study}_minimal_overlap_cutoff.csv')
            metrics = pd.read_csv(str(folds_metrics_filepath))
            self.folds_metrics_average_list[arch_index].append(metrics)

            overall_metrics_filepath = Path(self.input_study_dirs[arch_index], subdir,
                                          f'overall_metrics_average_{study}_minimal_overlap_cutoff.csv')
            metrics = pd.read_csv(str(overall_metrics_filepath))
            self.overall_metrics_average_list[arch_index].append(metrics)

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
        n_arch = len(self.studies_lists)

        for i in range(len(self.studies_lists[0])):
            pfile.write(f"\\midrule \n")
            for j in range(n_arch):
                output_string = self.arch_names[j] + " / " + self.studies_description[i]
                fname = Path(self.input_study_dirs[j].parent, self.studies_lists[j][i], subdir, f'overall_metrics_average_volume_cutoff.csv')
                results = pd.read_csv(fname)

                for m in metrics:
                    if results.loc[0, m + '_std'] > 0:
                        output_string += f" & {results.loc[0, m + '_mean'] * 100:.2f}$\pm${results.loc[0, m + '_std'] * 100:.2f}"
                    else:
                        output_string += f" & {results.loc[0, m + '_mean'] * 100:.2f}"

                pfile.write(output_string + "\\\ \n")


        pfile.close()

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

        patient_id_mapping_filepath = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/patient_id_mapping.csv')
        id_df = pd.read_csv(patient_id_mapping_filepath)
        vumc_df = id_df[(id_df['Hospital'] == 'VUmc')]

        annotator_groups = ['nov', 'exp']
        all_annotators = sum([[f"{annot}{i}" for i in range(1, 5)] for annot in annotator_groups], [])
        consensus_annotators = ['strict-consensus-nov', 'strict-consensus-exp', 'strict-consensus-all-annotators']
                                #'consensus-nov', 'consensus-exp', 'consensus-all-annotators']
        # consensus_annotators = ['consensus-all-annotators']

        all_annotators += consensus_annotators
        # all_annotators = consensus_annotators
        metrics = ['Dice', 'Jaccard', 'volume_seg', 'residual_tumor_prediction']

        annot_metrics = [[f"{annotator}_{metric}" for metric in metrics] for annotator in all_annotators]
        group_metrics = [[f"{group}_{metric}" for metric in metrics] for group in annotator_groups]
        all_annotators_metrics = [f"all_annotators_{metric}" for metric in metrics]
        architecture_metrics = [[f"{arch}_{metric}" for metric in metrics] for arch in self.arch_names]
        # columns = ['reference', 'pid', 'opid', 'volume', 'res_tumor'] + sum(annot_metrics, []) + \
        #     sum(group_metrics, []) + all_annotators_metrics + sum(architecture_metrics, [])
        columns = ['reference', 'pid', 'opid', 'volume', 'residual_tumor', 'annotator/model'] + metrics

        all_results = []
        # indx = [3, 8] #, 10]
        # patient_op_ids = [patient_op_ids[i] for i in indx]
        # patient_folders = [patient_folders[i] for i in indx]
        for ind, (op_id, patient_folder) in enumerate(zip(patient_op_ids, patient_folders)):
            patient = vumc_df[vumc_df['OP.ID'] == op_id]
            annotation_folder = Path(patient_folder, 'ses-postop', 'anat')
            pid = patient['DB ID'].values[0]
            db_index = patient['DB index'].values[0]

            # Load reference segmentations
            references, reference_volumes_res_tumor = self.__load_interrater_references(pid, db_index, annotation_folder)

            print(f"Compute scores for patient pid {pid} / opid {op_id}")
            for ref_name, (ref_ni, ref) in references.items():
                print(f"Reference: {ref_name}")
                # Store reference data
                ref_basic_info = [ref_name, pid, op_id] + reference_volumes_res_tumor[ref_name]
                #all_results.append([ref_name, pid, op_id] + reference_volumes_res_tumor[ref_name])

                all_annotations = [f for f in annotation_folder.iterdir() if 'label' in f.name and 'TMRenh_dseg' in f.name and
                                   'NativeT1c' in f.name]
                for annotator in all_annotators:
                    if not 'strict' in annotator:
                        annotation_filepath = [f for f in all_annotations if annotator in f.name and not 'strict' in f.name]
                    else:
                        annotation_filepath = [f for f in all_annotations if annotator in f.name]

                    if len(annotation_filepath) == 0:
                        #print(f"No annotations for patient {pid} annot {annotator}, creating empty mask")
                        annotation_ni = nib.Nifti1Image(np.zeros(ref.shape), affine=ref_ni.affine)
                    else:
                        annotation_ni = nib.load(annotation_filepath[0])
                        if len(annotation_ni.shape) == 4:
                            annotation_ni = nib.four_to_three(annotation_ni)[0]

                        if not annotation_ni.shape == ref_ni.shape:
                            print(f"Mismatch in shape between {annotator} annotation and GT for patient {pid} / opid {op_id}, skip")
                            continue

                    results = dice_computation(ref, annotation_ni)
                    # all_results[-1].extend(results)
                    all_results.append(ref_basic_info + [annotator] + results)

                # res_inter = pd.DataFrame([all_results[-1]], columns=columns[:len(all_results[-1])])
                # #Compute average scores per group
                # for group in annotator_groups:
                #     group_res = []
                #     for m in metrics:
                #         cols = [c for c in res_inter.columns if group in c and m in c]
                #         avg = np.mean(res_inter[cols].values)
                #         if m == 'res_tumor':
                #             avg = 1 if avg >= 0.5 else 0
                #         group_res.append(avg)
                #
                #     all_results[-1].extend(group_res)
                # #Compute average scores for all annotators
                # all_annot_res = []
                # for m in metrics:
                #     cols = [c for c in res_inter.columns if ('exp' in c or 'nov' in c) and m in c]
                #     avg = np.mean(res_inter[cols].values)
                #     if m == 'res_tumor':
                #         avg = 1 if avg >= 0.5 else 0
                #     all_annot_res.append(avg)
                # all_results[-1].extend(all_annot_res)

                # Compute scores for models
                for i, arch in enumerate(self.arch_names):
                    #for j, study in enumerate(self.studies_lists[i]):
                    j = 4
                    study = self.studies_lists[i][j]

                    # Find prediction file
                    prediction_dir = Path(self.input_study_dirs[i].parent, study, 'test_predictions', '0')
                    self.arch_names = ['nnU-Net', 'AGU-Net']
                    if arch == 'AGU-Net':
                        patient_folder = Path(prediction_dir, f'{db_index}_{pid}')
                        prediction_file = [f for f in patient_folder.iterdir() if f.is_file() and 'pred_tumor' in f.name][0]
                    else:
                        nnunet_index = patient['nnU-Net ID'].values[0]
                        patient_folder = Path(prediction_dir, f'index0_{nnunet_index}')
                        prediction_file = [f for f in patient_folder.iterdir() if f.is_file() and 'predictions' in f.name][0]

                    pred_ni = nib.load(prediction_file)

                    if pred_ni.shape != ref_ni.shape:
                        print(f"Error, mismatch in shape between prediction and ground truth for pid {pid} / opid {op_id}, arch {arch}")
                        print(f"Pred shape: {pred_ni.shape}, GT shape {ref_ni.shape}")
                        #all_results[-1].extend([None, None, None, None])
                        results = [None for _ in metrics]
                    else:
                        # Load optimal threshold and threshold predictions
                        optimal_dice_study = Path(self.input_study_dirs[i].parent, study, 'Validation', 'optimal_dice_study.csv')
                        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(
                            study_filename=optimal_dice_study)

                        results = dice_computation(ref, pred_ni, t=optimal_threshold)

                    all_results.append(ref_basic_info + [arch] + results)

        output_filepath = Path(self.output_dir, 'interrater_study.csv')
        results_df = pd.DataFrame(all_results, columns=columns)
        results_df.to_csv(output_filepath, index=False)

    def __load_interrater_references(self, pid, db_index, annotation_folder):
        references = {}
        reference_volume_res_tumor = {}
        image_filepath = [f for f in annotation_folder.iterdir() if 'NativeT1c' in f.name and 'dseg' not in f.name][
            0]
        image_ni = nib.load(image_filepath)

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
        else:
            gt = gt_ni.get_data()
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt = gt.astype('uint8')
            references['ground_truth_segmentation'] = [gt_ni, deepcopy(gt)]

            # Compute tumor volume and check for residual tumor
            voxel_size = np.prod(gt_ni.header.get_zooms()[0:3])
            volume_seg_ml = compute_tumor_volume(gt, voxel_size)
            res_tumor = 1 if volume_seg_ml > 0.175 else 0
            reference_volume_res_tumor['ground_truth_segmentation'] = [volume_seg_ml, res_tumor]

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
                print(f"Mismatch in shape between T1c image and {consensus_identifier} segmentation for patient {pid}, skip")
            else:
                consensus = consensus_ni.get_data()
                consensus[consensus > 0.5] = 1
                consensus[consensus <= 0.5] = 0
                consensus = consensus.astype('uint8')
                references[consensus_identifier] = [consensus_ni, deepcopy(consensus)]

                # Compute tumor volume and check for residual tumor
                voxel_size = np.prod(consensus_ni.header.get_zooms()[0:3])
                volume_seg_ml = compute_tumor_volume(consensus, voxel_size)
                res_tumor = 1 if volume_seg_ml > 0.175 else 0
                reference_volume_res_tumor[consensus_identifier] = [volume_seg_ml, res_tumor]

        return references, reference_volume_res_tumor

    def interrater_study_summary(self):
        study_filepath = Path(self.output_dir, 'interrater_study.csv')
        results = pd.read_csv(study_filepath)
        results.replace('inf', 0, inplace=True)
        results.replace('', 0, inplace=True)
        results.replace(' ', 0, inplace=True)

        annotator_groups = ['nov', 'exp']
        all_annotators = sum([[f"{annot}{i}" for i in range(1, 5)] for annot in annotator_groups], [])

        metrics = ['Dice', 'Jaccard', 'volume_seg', 'res_tumor']
        annot_metrics = [[f"{annotator}_{metric}" for metric in metrics] for annotator in all_annotators]
        group_metrics = [[f"{group}_{metric}" for metric in metrics] for group in annotator_groups]
        all_annotators_metrics = [f"all_annotators_{metric}" for metric in metrics]
        architecture_metrics = [[f"{arch}_{metric}" for metric in metrics] for arch in self.arch_names]
        # columns = ['reference', 'pid', 'opid', 'volume', 'res_tumor'] + sum(annot_metrics, []) + \
        #           sum(group_metrics, []) + all_annotators_metrics + sum(architecture_metrics, [])

        # Compute metrics
        eval_metrics = ['Dice', 'Dice-P', 'Jaccard', 'Jaccard-P']
        average_columns = ['reference', 'evaluator', 'Patient-wise recall', 'Patient-wise precision', 'Patient-wise specificity',
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
                # true_pos = len(ref_results.loc[(ref_results[f'{evaluator}_res_tumor'] == 1) & (ref_results['res_tumor'] == 1)])
                # false_pos = len(
                #     ref_results.loc[(ref_results[f'{evaluator}_res_tumor'] == 1) & (ref_results['res_tumor'] == 0)])
                # true_neg = len(
                #     ref_results.loc[(ref_results[f'{evaluator}_res_tumor'] == 0) & (ref_results['res_tumor'] == 0)])
                # false_neg = len(
                #     ref_results.loc[(ref_results[f'{evaluator}_res_tumor'] == 0) & (ref_results['res_tumor'] == 1)])
                true_pos = len(
                    eval_results.loc[(eval_results[f'residual_tumor_prediction'] == 1) & (eval_results['residual_tumor'] == 1)])
                false_pos = len(
                    eval_results.loc[(eval_results[f'residual_tumor_prediction'] == 1) & (eval_results['residual_tumor'] == 0)])
                true_neg = len(
                    eval_results.loc[(eval_results[f'residual_tumor_prediction'] == 0) & (eval_results['residual_tumor'] == 0)])
                false_neg = len(
                    eval_results.loc[(eval_results[f'residual_tumor_prediction'] == 0) & (eval_results['residual_tumor'] == 1)])
                print(sum([true_pos, false_pos, true_neg, false_neg]))

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
        metrics_df.to_csv(output_filepath, index = False)

    def visualize_interrater_results(self):
        study_filepath = Path(self.output_dir, 'interrater_study.csv')
        results = pd.read_csv(study_filepath)
        results.replace('inf', 0, inplace=True)
        results.replace('', 0, inplace=True)
        results.replace(' ', 0, inplace=True)

        annotator_groups = ['nov', 'exp']
        all_annotators = sum([[f"{annot}{i}" for i in range(1, 5)] for annot in annotator_groups], [])

        unique_references = np.unique(results['reference'])
        unique_evaluators = np.unique(['_'.join(colname.split('_')[:-1]) for colname in results.columns if
                                       ('Dice' in colname or 'Jaccard' in colname)])
        print(len(unique_evaluators), unique_evaluators)

        for ref in unique_references:
            ref_results = results.loc[results['reference'] == ref]
            ref_results_positive = ref_results.loc[ref_results['residual_tumor'] == 1]

            # ref_results_annot = ref_results[all_annotators]

            plt.figure(figsize=(15, 8))
            b = sns.boxplot(data=ref_results_positive, x='annotator/model', y='Jaccard')
            b.set_xticklabels(b.get_xticklabels(), rotation=45)
            b.set(ylim=(0, 1))
            plt.title(f"Reference = {ref}")
            plt.tight_layout()
            plt.savefig(Path(self.output_dir, f'jaccard_scores_ref_{ref}.png'))
            # plt.show(block=False)

def dice_computation(reference, detection_ni, t=0.5):
    detection = deepcopy(detection_ni.get_data())
    detection[detection <= t] = 0
    detection[detection > t] = 1
    detection = detection.astype('uint8')

    ref = deepcopy(reference)
    ref[ref <= t] = 0
    ref[ref > t] = 1
    ref = ref.astype('uint8')

    dice = compute_dice(ref, detection)
    jaccard = dice / (2-dice)
    voxel_size = np.prod(detection_ni.header.get_zooms()[0:3])
    volume_seg_ml = compute_tumor_volume(detection, voxel_size)
    res_tumor = 1 if volume_seg_ml > 0.175 else 0

    return [dice, jaccard, volume_seg_ml, res_tumor]