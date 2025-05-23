import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile



def inference_test_docker(test_dir):
    """
    Testing the CLI within a Docker container for the validation unit test, running on CPU.
    The latest Docker image is being hosted at: dbouget/raidionics-val:v1.0-py38-cpu

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running validation unit test in Docker container.\n")

    logging.info("Preparing configuration file.\n")
    try:
        val_config = configparser.ConfigParser()
        val_config.add_section('Default')
        val_config.set('Default', 'task', "validation")
        val_config.set('Default', 'data_root', '/workspace/resources/Input_dataset')
        val_config.set('Default', 'number_processes', '1')
        val_config.add_section('Validation')
        val_config.set('Validation', 'input_folder', '/workspace/resources/StudyResults')
        val_config.set('Validation', 'output_folder', '/workspace/resources/StudyResults')
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
                          'dbouget/raidionics-val:v1.0-py38-cpu',
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
        scores_segmentation_filename = os.path.join(test_dir, 'StudyResults', 'Validation', 'all_dice_scores.csv')
        if not os.path.exists(scores_segmentation_filename):
            logging.error("Validation in Docker container failed, no dice scores were generated.\n")
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            raise ValueError("Validation in Docker container failed, no dice scores were generated.\n")
        logging.info("Validation unit test in Docker container succeeded.\n")
    except Exception as e:
        logging.error("Error during validation in Docker container with: \n {}.\n".format(traceback.format_exc()))
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        raise ValueError("Error during validation in Docker container with.\n")

    logging.info("Validation unit test in Docker container succeeded.\n")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


inference_test_docker()
