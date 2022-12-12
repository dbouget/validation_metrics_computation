import os
import pickle
import pandas as pd
import numpy as np


def get_fold_from_file(filename, fold_number):
    """
    Return the content of the validation and test folds for the current fold number by parsing the specified file.
    :param filename: lists of samples used for training and specifying the folds' content.
    :param fold_number: fold from which the sets' content should be collected.
    :return: two lists of strings containing the patients' names featured in the validation and test folds.
    """
    val_set = None
    test_set = None
    if filename is None or not os.path.exists(filename):
        raise AttributeError(f'The provided filename containing the folds distribution is invalid: {filename}.\n')

    if filename.split('.')[-1] == 'txt':
        with open(filename) as f:
            for i in range(fold_number):
                val_line = f.readline()
                test_line = f.readline()

            val_line = f.readline().strip()
            test_line = f.readline().strip()

            val_set = val_line.split(' ')
            val_set = [x for x in val_set]
            test_set = test_line.split(' ')
            test_set = [x for x in test_set]
    elif filename.split('.')[-1] == 'pkl':
        folds_file = open(filename, 'rb')
        folds = pickle.load(folds_file)

        val_set = folds[int(fold_number)]['val']
        test_set = folds[int(fold_number)]['val']
        if 'test' in folds[int(fold_number)].keys():
            test_set = folds[int(fold_number)]['test']

    return val_set, test_set


def reload_optimal_validation_parameters(study_filename):
    """
    Load the optimal probability and Dice thresholds identified during the validation process, from the
    optimal_dice_study.csv file located inside the validation folder.
    :param study_filename: filename to the optimal_dice_study.csv file located inside the validation folder.
    :return: two floats, the best probability threshold and best Dice threshold.
    """
    study_df = pd.read_csv(study_filename)
    optimums = study_df.iloc[-1]

    return optimums['Detection threshold'], optimums['Dice threshold']
