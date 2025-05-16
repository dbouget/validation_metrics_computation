import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import pandas as pd


def test_validation_pipeline_metrics_space(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard reporting unit test.\n")

    logging.info("Preparing configuration file.\n")
    output_folder = ""
    try:
        output_folder = os.path.join(test_dir, 'Test1', 'Results')
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        val_config = configparser.ConfigParser()
        val_config.add_section('Default')
        val_config.set('Default', 'task', 'validation')
        val_config.set('Default', 'data_root', os.path.join(test_dir, 'Test1','Inputs'))
        val_config.set('Default', 'number_processes', "1")
        val_config.add_section('Validation')
        val_config.set('Validation', 'input_folder', os.path.join(test_dir, 'Test1', 'Predictions'))
        val_config.set('Validation', 'output_folder', output_folder)
        val_config.set('Validation', 'gt_files_suffix', 'label_tumor.nii.gz')
        val_config.set('Validation', 'prediction_files_suffix', 'pred_tumor.nii.gz')
        val_config.set('Validation', 'use_index_naming_convention', 'false')
        val_config.set('Validation', 'nb_folds', '1')
        val_config.set('Validation', 'split_way', 'three-way')
        val_config.set('Validation', 'detection_overlap_thresholds', '0.01')
        val_config.set('Validation', 'metrics_space', 'patientwise, pixelwise, objectwise')
        val_config.set('Validation', 'class_names', 'tumor')
        val_config.set('Validation', 'extra_metrics', '')
        val_config.set('Validation', 'tiny_objects_removal_threshold', '25')
        val_config.set('Validation', 'true_positive_volume_thresholds', '0.1')
        val_config.set('Validation', 'use_brats_data', 'false')
        config_filename = os.path.join(output_folder, 'config.ini')
        with open(config_filename, 'w') as outfile:
            val_config.write(outfile)

        logging.info("Running k-fold cross-validation unit test.\n")
        from raidionicsval.compute import compute
        compute(config_filename)

        logging.info("Collecting and comparing results.\n")
        scores_filename = os.path.join(output_folder, 'Validation', 'tumor_dice_scores.csv')
        assert os.path.exists(scores_filename), "Scores file does not exist on disk."
        overall_metrics_filename = os.path.join(output_folder, 'Validation', 'tumor_overall_metrics_average_TP.csv')
        assert os.path.exists(overall_metrics_filename), "File with overall metrics averaged for TP patients does not exist on disk."
        gt_metrics_filename = os.path.join(test_dir, 'Test1', 'verif', "tumor_overall_metrics_average_TP_metrics_space.csv")
        results_df = pd.read_csv(overall_metrics_filename)
        gt_df = pd.read_csv(gt_metrics_filename)
        assert round(gt_df['Patient-wise recall (Mean)'][0], 2) == round(results_df['Patient-wise recall (Mean)'][0], 2), "Patient-wise recall (Mean) values do not match"
        assert round(gt_df['OW Recall (Mean)'][0], 2) == round(results_df['OW Recall (Mean)'][0],2), "OW Recall (Mean) values do not match"
    except Exception as e:
        logging.error(f"Error during k-fold cross-validation unit test with: {e} \n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during k-fold cross-validation unit test with.\n")

    logging.info("k-fold cross-validation unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

def test_validation_pipeline_extra_metrics(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard reporting unit test.\n")

    logging.info("Preparing configuration file.\n")
    output_folder = ""
    try:
        output_folder = os.path.join(test_dir, 'Test1', 'Results')
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        val_config = configparser.ConfigParser()
        val_config.add_section('Default')
        val_config.set('Default', 'task', 'validation')
        val_config.set('Default', 'data_root', os.path.join(test_dir, 'Test1','Inputs'))
        val_config.set('Default', 'number_processes', "1")
        val_config.add_section('Validation')
        val_config.set('Validation', 'input_folder', os.path.join(test_dir, 'Test1', 'Predictions'))
        val_config.set('Validation', 'output_folder', output_folder)
        val_config.set('Validation', 'gt_files_suffix', 'label_tumor.nii.gz')
        val_config.set('Validation', 'prediction_files_suffix', 'pred_tumor.nii.gz')
        val_config.set('Validation', 'use_index_naming_convention', 'false')
        val_config.set('Validation', 'nb_folds', '1')
        val_config.set('Validation', 'split_way', 'three-way')
        val_config.set('Validation', 'detection_overlap_thresholds', '0.01')
        val_config.set('Validation', 'metrics_space', 'patientwise, pixelwise, objectwise')
        val_config.set('Validation', 'class_names', 'tumor')
        val_config.set('Validation', 'extra_metrics', 'HD95, MI')
        val_config.set('Validation', 'tiny_objects_removal_threshold', '25')
        val_config.set('Validation', 'true_positive_volume_thresholds', '0.1')
        val_config.set('Validation', 'use_brats_data', 'false')
        config_filename = os.path.join(output_folder, 'config.ini')
        with open(config_filename, 'w') as outfile:
            val_config.write(outfile)

        logging.info("Running k-fold cross-validation unit test.\n")
        from raidionicsval.compute import compute
        compute(config_filename)

        logging.info("Collecting and comparing results.\n")
        scores_filename = os.path.join(output_folder, 'Validation', 'tumor_dice_scores.csv')
        assert os.path.exists(scores_filename), "Scores file does not exist on disk."
        overall_metrics_filename = os.path.join(output_folder, 'Validation', 'tumor_overall_metrics_average_TP.csv')
        assert os.path.exists(overall_metrics_filename), "File with overall metrics averaged for TP patients does not exist on disk."
        gt_metrics_filename = os.path.join(test_dir, 'Test1', 'verif', "tumor_overall_metrics_average_TP_extra_metrics.csv")
        results_df = pd.read_csv(overall_metrics_filename)
        gt_df = pd.read_csv(gt_metrics_filename)
        assert round(gt_df['OW HD95 (Mean)'][0], 2) == round(results_df['OW HD95 (Mean)'][0], 2), "OW HD95 (Mean) values do not match"
        assert round(gt_df['OW MI (Std)'][0], 2) == round(results_df['OW MI (Std)'][0],2), "OW MI (Std) values do not match"
    except Exception as e:
        logging.error(f"Error during test with: {e} \n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError(f"{e}.\n")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)