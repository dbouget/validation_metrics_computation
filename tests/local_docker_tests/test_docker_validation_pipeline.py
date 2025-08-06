import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile
import pandas as pd


def test_validation_docker(test_dir):
    """
    Testing the CLI within a Docker container for the validation unit test, running on CPU.
    The latest Docker image is being hosted at: dbouget/raidionics-val:v1.1.1-py39-cpu

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running validation unit test in Docker container.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, 'Test1', 'Results')
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        val_config = configparser.ConfigParser()
        val_config.add_section('Default')
        val_config.set('Default', 'task', "validation")
        val_config.set('Default', 'data_root', '/workspace/resources/Test1/Inputs')
        val_config.set('Default', 'number_processes', '1')
        val_config.add_section('Validation')
        val_config.set('Validation', 'input_folder', '/workspace/resources/Test1/Predictions')
        val_config.set('Validation', 'output_folder', '/workspace/resources/Test1/Results')
        val_config.set('Validation', 'gt_files_suffix', 'label_tumor.nii.gz')
        val_config.set('Validation', 'prediction_files_suffix', 'pred_tumor.nii.gz')
        val_config.set('Validation', 'use_index_naming_convention', 'false')
        val_config.set('Validation', 'nb_folds', '1')
        val_config.set('Validation', 'split_way', 'three-way')
        val_config.set('Validation', 'detection_overlap_thresholds', '0.')
        val_config.set('Validation', 'metrics_space', 'pixelwise, objectwise')
        val_config.set('Validation', 'class_names', 'tumor')
        val_config.set('Validation', 'extra_metrics', 'IOU')
        val_config.set('Validation', 'tiny_objects_removal_threshold', '25')
        val_config.set('Validation', 'true_positive_volume_thresholds', '0.1')
        val_config.set('Validation', 'use_brats_data', 'false')
        val_config_filename = os.path.join(test_dir, 'test_val_config.ini')
        with open(val_config_filename, 'w') as outfile:
            val_config.write(outfile)

        logging.info("Running validation unit test in Docker container.\n")
        try:
            import platform
            cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(test_dir),
                          '--network=host', '--ipc=host', '--user', str(os.geteuid()),
                          'dbouget/raidionics-val:v1.1.1-py39-cpu',
                          '-c', '/workspace/resources/test_val_config.ini', '-v', 'debug']
            logging.info("Executing the following Docker call: {}".format(cmd_docker))
            if platform.system() == 'Windows':
                subprocess.check_call(cmd_docker, shell=True)
            else:
                subprocess.check_call(cmd_docker)
        except Exception as e:
            logging.error("Error during validation test in Docker container with: \n {}.\n".format(traceback.format_exc()))
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            raise ValueError("Error during validation test in Docker container.\n")

        logging.info("Collecting and comparing results.\n")
        scores_filename = os.path.join(output_folder, 'Validation', 'tumor_dice_scores.csv')
        assert os.path.exists(scores_filename), "Scores file does not exist on disk."
        overall_metrics_filename = os.path.join(output_folder, 'Validation', 'tumor_overall_metrics_average_TP.csv')
        assert os.path.exists(overall_metrics_filename), "File with overall metrics averaged for TP patients does not exist on disk."
        gt_metrics_filename = os.path.join(test_dir, 'Test1', 'verif', "tumor_overall_metrics_average_TP.csv")
        results_df = pd.read_csv(overall_metrics_filename)
        gt_df = pd.read_csv(gt_metrics_filename)
        assert round(gt_df['PiW Dice (Mean)'][0], 2) == round(results_df['PiW Dice (Mean)'][0], 2), "PiW Dice (Mean) values do not match"
    except Exception as e:
        logging.error("Error during validation in Docker container with: \n {}.\n".format(traceback.format_exc()))
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        raise ValueError("Error during validation in Docker container with.\n")

    logging.info("Validation unit test in Docker container succeeded.\n")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
