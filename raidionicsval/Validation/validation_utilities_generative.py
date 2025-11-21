import pandas as pd
import os
import numpy as np
from copy import deepcopy

from ..Utils.resources import SharedResources

def compute_fold_average(folder, data=None, metrics=[], suffix='', condition='All'):
    """

    :param folder: Main study folder where the results will be dumped (assuming inside a Validation sub-folder)
    :param best_threshold:
    :param best_overlap:
    :param metric_names:
    :return:
    """
    results = None
    if data is None:
        results_filename = os.path.join(folder, 'all_scores.csv')
        results = pd.read_csv(results_filename, na_values="NaN")
    else:
        results = deepcopy(data)

    results.replace('inf', np.nan, inplace=True)
    results.replace(float('inf'), np.nan, inplace=True)
    results.replace('', np.nan, inplace=True)
    results.replace(' ', np.nan, inplace=True)
    unique_folds = np.unique(results['Fold'])
    nb_folds = len(unique_folds)
    metrics_per_fold = []
    avg_metrics_per_fold = []
    fold_average_columns = ['Fold', '# samples']

    for f in unique_folds:
        # Selecting only the one line per patient showing slice -1, i.e., full volume metrics.
        patient_wise_metrics = results.loc[results['Fold'] == f].filter(like="PW ").dropna().reset_index(drop=True)
        patientaverage_slicewise_metrics_df = compute_patientaverage_slicewise_metrics(results, f, condition)
        fold_averages = pd.concat([patientaverage_slicewise_metrics_df, patient_wise_metrics], axis=1)
        metrics_per_fold.append(fold_averages)
        avg_metrics_per_fold.append([f, len(patient_wise_metrics)] + [item for pair in zip(fold_averages[fold_averages.columns[1:]].mean(), fold_averages[fold_averages.columns[1:]].std()) for item in pair])
        if len(fold_average_columns) == 2:
            tmp_columns = [[f"{x} - Mean", f"{x} - Std"] for x in fold_averages.columns[1:]]
            new_columns = [element for sublist in tmp_columns for element in sublist]
            fold_average_columns.extend(new_columns)

    fold_averages_df = pd.DataFrame(avg_metrics_per_fold, columns=fold_average_columns)
    study_filename = os.path.join(folder, 'folds_metrics_average.csv') if suffix == '' else os.path.join(folder,
                                                                                                 'folds_metrics_average_' + suffix + '.csv')

    fold_averages_df.to_csv(study_filename, index=False)

    # @TODO. Has to check if this works when more than one fold!
    metrics_per_fold_df = pd.concat(metrics_per_fold, keys=[','.join(f"{x}" for x in list(unique_folds))], names=["Fold"])

    ####### Averaging the results from the different folds ###########
    # @TODO. Has to be done in the same way as the segmentation one, taking into account the sample size in each fold


def compute_patientaverage_slicewise_metrics(results, fold_number, condition):
    fold_results = results.loc[results['Fold'] == fold_number]
    unique_patients = np.unique(results['Patient'].values)

    patient_averages = []
    for p in unique_patients:
        patient_results = fold_results.loc[fold_results['Patient'] == p].filter(like="SW ")
        patient_results_clean = patient_results.dropna()
        averages = patient_results_clean.mean()
        stds = patient_results_clean.std()
        patient_averages.append([p] + [item for pair in zip(averages, stds) for item in pair])

    new_columns = [[f"{x} - Mean", f"{x} - Std"] for x in fold_results.filter(like="SW ")]
    patient_averages_df = pd.DataFrame(patient_averages, columns=['Patient'] + [element for sublist in new_columns for element in sublist])

    return patient_averages_df
