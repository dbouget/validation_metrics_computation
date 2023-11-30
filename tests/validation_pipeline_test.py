import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile

try:
    import requests
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'requests'])
    import requests


def validation_pipeline_test():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard reporting unit test.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    try:
        resources_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsValLib_UnitTest1.zip'

        archive_dl_dest = os.path.join(test_dir, 'resources.zip')
        headers = {}
        response = requests.get(resources_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(test_dir)

        if not os.path.exists(os.path.join(test_dir, 'Input_dataset')):
            raise ValueError('Resources download or extraction failed, content not available on disk.')
    except Exception as e:
        logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during resources download.\n")

    logging.info("Preparing configuration file.\n")
    try:
        val_config = configparser.ConfigParser()
        val_config.add_section('Default')
        val_config.set('Default', 'task', 'validation')
        val_config.set('Default', 'data_root', os.path.join(test_dir, 'Input_dataset'))
        val_config.set('Default', 'number_processes', "1")
        val_config.add_section('Validation')
        val_config.set('Validation', 'input_folder', os.path.join(test_dir, 'StudyResults'))
        val_config.set('Validation', 'output_folder', os.path.join(test_dir, 'StudyResults'))
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
        config_filename = os.path.join(test_dir, 'config.ini')
        with open(config_filename, 'w') as outfile:
            val_config.write(outfile)

        logging.info("Running k-fold cross-validation unit test.\n")
        from raidionicsval.compute import compute
        compute(config_filename)

        logging.info("Collecting and comparing results.\n")
        scores_filename = os.path.join(test_dir, 'StudyResults', 'Validation', 'all_dice_scores.csv')
        if not os.path.exists(scores_filename):
            logging.error("k-fold cross-validation unit test failed, no scores were generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("k-fold cross-validation unit test failed, no scores were generated.\n")

        logging.info("k-fold cross-validation CLI unit test started.\n")
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsval',
                                       '{config}'.format(config=config_filename),
                                       '--verbose', 'debug'], shell=True)
            elif platform.system() == 'Darwin' and platform.processor() == 'arm':
                subprocess.check_call(['python3', '-m', 'raidionicsval',
                                       '{config}'.format(config=config_filename),
                                       '--verbose', 'debug'])
            else:
                subprocess.check_call(['raidionicsval',
                                       '{config}'.format(config=config_filename),
                                       '--verbose', 'debug'])
        except Exception as e:
            logging.error("Error during k-fold cross-validation CLI unit test with: \n {}.\n".format(traceback.format_exc()))
            shutil.rmtree(test_dir)
            raise ValueError("Error during k-fold cross-validation CLI unit test.\n")

        logging.info("Collecting and comparing results.\n")
        scores_filename = os.path.join(test_dir, 'StudyResults', 'Validation', 'all_dice_scores.csv')
        if not os.path.exists(scores_filename):
            logging.error("k-fold cross-validation CLI unit test failed, no scores were generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("k-fold cross-validation CLI unit test failed, no scores were generated.\n")
        logging.info("k-fold cross-validation CLI unit test succeeded.\n")
    except Exception as e:
        logging.error("Error during k-fold cross-validation unit test with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during k-fold cross-validation unit test with.\n")

    logging.info("k-fold cross-validation unit test succeeded.\n")
    shutil.rmtree(test_dir)


validation_pipeline_test()
