import os

import numpy as np
from typing import List
import pandas as pd

from ..Utils.resources import SharedResources


class PatientMetrics:
    _unique_id = ""  # Internal unique identifier for the patient
    _objective = "segmentation"  #
    _patient_id = ""  # Unique identifier for the patient (might be multiple times the same patient in different folds)
    _fold_number = None  # Fold integer to which the patient belongs to
    _ground_truth_filepaths = None
    _prediction_filepaths = None
    _patientwise_metrics = None
    _pixelwise_metrics = None
    _objectwise_metrics = None
    _classification_metrics = None
    _extra_metrics = None
    _class_names = None
    _class_metrics = None

    def __init__(self, id: str, patient_id: str, fold_number: int, class_names: List[str],
                 objective: str = "segmentation") -> None:
        """

        """
        self.__reset()
        self._unique_id = id
        self._objective = objective
        self._patient_id = patient_id
        self._fold_number = fold_number
        self._class_names = class_names
        self._class_metrics = {}
        for c in class_names:
            self._class_metrics[c] = ClassMetrics(c, self._patient_id, fold_number=self._fold_number)

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = ""
        self._objective = "segmentation"
        self._patient_id = ""
        self._fold_number = None
        self._prediction_filepaths = None
        self._ground_truth_filepaths = None
        self._patientwise_metrics = None
        self._pixelwise_metrics = None
        self._objectwise_metrics = None
        self._classification_metrics = None
        self._extra_metrics = None
        self._class_metrics = None
        self._class_names = None

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def objective(self) -> str:
        return self._objective

    @objective.setter
    def objective(self, objective: str) -> None:
        self._objective = objective

    @property
    def patient_id(self) -> str:
        return self._patient_id

    @patient_id.setter
    def patient_id(self, patient_id: str) -> None:
        self._patient_id = patient_id

    @property
    def extra_metrics(self) -> str:
        return self._extra_metrics

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def ground_truth_filepaths(self) -> str:
        return self._ground_truth_filepaths

    @ground_truth_filepaths.setter
    def ground_truth_filepaths(self, ground_truth_filepaths) -> None:
        self._ground_truth_filepaths = ground_truth_filepaths

    @property
    def prediction_filepaths(self) -> str:
        return self._prediction_filepaths

    @prediction_filepaths.setter
    def prediction_filepaths(self, prediction_filepaths) -> None:
        self._prediction_filepaths = prediction_filepaths

    def init_from_file(self, study_folder: str):
        if self.objective == "segmentation":
            self.__init_from_file_segmentation(study_folder=study_folder)
        else:
            self.__init_from_file_classification(study_folder=study_folder)

    def __init_from_file_segmentation(self, study_folder: str):
        all_scores_filename = os.path.join(study_folder, 'all_dice_scores.csv')

        for c in list(self._class_metrics.keys()):
            class_scores_filename = os.path.join(study_folder, c + '_dice_scores.csv')
            self._class_metrics[c].init_from_file(class_scores_filename)

        if not os.path.exists(all_scores_filename):
            return
        scores_df = pd.read_csv(all_scores_filename)
        scores_df['Patient'] = scores_df.Patient.astype(str)
        if len(scores_df.loc[(scores_df["Patient"] == self._patient_id) & (scores_df["Fold"] == self._fold_number)]) == 0:
            return

        patient_class_scores = scores_df.loc[(scores_df["Patient"] == self._patient_id) & (scores_df["Fold"] == self._fold_number)]
        self._patientwise_metrics = []
        self._pixelwise_metrics = []
        self._objectwise_metrics = []
        self._extra_metrics = []
        for thr in list(patient_class_scores["Threshold"].values):
            thr_results = patient_class_scores.loc[patient_class_scores["Threshold"] == thr].values[0]
            thr_val = thr_results[2]
            pixelwise_values = list(thr_results[3:7])
            patientwise_values = list(thr_results[7:10])
            objectwise_values = list(thr_results[10:SharedResources.getInstance().upper_default_metrics_index])
            extra_values = list(thr_results[SharedResources.getInstance().upper_default_metrics_index:])
            [extra_values.append(None) for x in range(len(list(thr_results[SharedResources.getInstance().upper_default_metrics_index:])), 2 * len(SharedResources.getInstance().validation_metric_names))]
            extra_values_description = list(scores_df.columns[SharedResources.getInstance().upper_default_metrics_index:])
            extra_metric_names = []
            for m in SharedResources.getInstance().validation_metric_names:
                extra_metric_names.extend([f'PiW {m}', f'OW {m}'])
            [extra_values_description.append(x) for x in extra_metric_names]
            self._pixelwise_metrics.append([thr_val] + pixelwise_values)
            self._patientwise_metrics.append([thr_val] + patientwise_values)
            self._objectwise_metrics.append([thr_val] + objectwise_values)
            extra_values_cat = [[x, y] for x, y in zip(extra_values_description, extra_values)]
            if len(extra_values_cat) != 0:
                self._extra_metrics.append([thr_val] + extra_values_cat)
        if len(self._extra_metrics) == 0:
            self._extra_metrics = None

    def __init_from_file_classification(self, study_folder: str):
        all_scores_filename = os.path.join(study_folder, 'all_scores.csv')
        if not os.path.exists(all_scores_filename):
            return

        scores_df = pd.read_csv(all_scores_filename)
        scores_df['Patient'] = scores_df.Patient.astype(str)
        if len(scores_df.loc[(scores_df["Patient"] == self._patient_id) & (scores_df["Fold"] == self._fold_number)]) == 0:
            return

        patient_scores = scores_df.loc[(scores_df["Patient"] == self._patient_id) & (scores_df["Fold"] == self._fold_number)]
        self._classification_metrics = []
        classification_values = list(patient_scores[2:7])
        self._classification_metrics.append(classification_values)

    def is_complete(self):
        """
        @TODO. Will require much deeper checks to see if any value is missing and a recompute triggered
        :return:
        """
        status = False
        if self.objective == "segmentation":
            complete_pixelwise_metrics = True not in [-1. in x for x in
                                                      self._pixelwise_metrics] if self._pixelwise_metrics is not None else False
            complete_objectwise_metrics = True not in [-1. in x for x in
                                                      self._objectwise_metrics] if self._objectwise_metrics is not None else False
            if 'objectwise' not in SharedResources.getInstance().validation_metric_spaces:
                status = complete_pixelwise_metrics
            else:
                status = complete_pixelwise_metrics & complete_objectwise_metrics

            for c in list(self._class_metrics.keys()):
                status = status & self._class_metrics[c].is_complete()
        else:
            status = self._classification_metrics is not None
        return status

    def set_patient_filenames(self, filenames: dict) -> None:
        self._ground_truth_filepaths = []
        self._prediction_filepaths = []

        if self.objective == "segmentation":
            for c in list(filenames.keys()):
                self._ground_truth_filepaths.append(filenames[c][0])
                self._prediction_filepaths.append(filenames[c][1])
        else:
            self._ground_truth_filepaths.append(filenames[0])
            self._prediction_filepaths.append(filenames[1])

    def get_class_filenames(self, class_index: int) -> List[str]:
        return [self._ground_truth_filepaths[class_index], self.prediction_filepaths[class_index]]

    def set_class_regular_metrics(self, class_name: str, results: list):
        self._class_metrics[class_name].set_results(results)

    def get_class_metrics(self, class_name: str):
        return self._class_metrics[class_name].get_all_metrics()

    def get_class_extra_metrics(self, class_name: str):
        return self._class_metrics[class_name].get_extra_metrics()

    def get_class_extra_metrics_without_header(self, class_name: str):
        return self._class_metrics[class_name].get_extra_metrics_without_header()

    def get_optimal_class_metrics(self, class_index: int, optimal_threshold: float):
        class_name = self._class_names[class_index]
        for i in range(len(self._class_metrics[class_name].get_all_metrics())):
            if self._class_metrics[class_name].get_all_metrics()[i][0] == optimal_threshold:
                return self._class_metrics[class_name].get_all_metrics()[i]
        return None

    def get_optimal_class_extra_metrics(self, class_index: int, optimal_threshold: float):
        class_name = self._class_names[class_index]
        for i in range(len(self._class_metrics[class_name].get_all_metrics())):
            if self._class_metrics[class_name].get_all_metrics()[i][0] == optimal_threshold:
                if self._class_metrics[class_name].get_extra_metrics() is not None:
                    return self._class_metrics[class_name].get_extra_metrics()[i]
        return None

    def set_optimal_class_extra_metrics(self, class_index: int, optimal_threshold: float, metrics_values: List):
        class_name = self._class_names[class_index]
        self._class_metrics[class_name].set_extra_metrics(optimal_threshold, metrics_values)

    def setup_extra_metrics(self, metric_names):
        """
        Adjust the size of the extra metrics, if new metrics have been requested to be computed in the config file.
        N-B: For already computed metrics, even if removed from the list in the config file, a removal from the
        container will not be performed and results will be kept.


        """
        thr_list = self._class_metrics[self.class_names[0]].get_probability_thresholds_list()
        complete_metric_names = []
        for m in metric_names:
            complete_metric_names.extend([f'PiW {m}', f'OW {m}'])
        if self._extra_metrics is None:
            self._extra_metrics = []
            for thr in thr_list:
                curr_thr = [thr]
                for m in complete_metric_names:
                    curr_thr.append([m, float('nan')])
                self._extra_metrics.append(curr_thr)
        else:
            existing_metrics = [x[0] for x in self._extra_metrics[0][1:]]
            matching_metrics_states = all(element in existing_metrics for element in complete_metric_names)
            if not matching_metrics_states:
                for m in complete_metric_names:
                    if m not in existing_metrics:
                        for th in range(len(self._extra_metrics)):
                            self._extra_metrics[th].append([m, float('nan')])

        # Performs the same operation on the extra metrics for each class
        for cl in self._class_names:
            self._class_metrics[cl].setup_extra_metrics(complete_metric_names)


class ClassMetrics:
    _unique_id = ""  # Internal unique identifier for the class
    _patient_id = None
    _fold_number = None
    _patientwise_metrics = None
    _pixelwise_metrics = None
    _objectwise_metrics = None
    _extra_metrics = None

    def __init__(self, id: str, patient_id: str, fold_number: int) -> None:
        """

        """
        self.__reset()
        self._unique_id = id
        self._patient_id = patient_id
        self._fold_number = fold_number

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = ""
        self._patient_id = None
        self._fold_number = None
        self._patientwise_metrics = None
        self._pixelwise_metrics = None
        self._objectwise_metrics = None
        self._extra_metrics = None

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def pixelwise_metrics(self) -> str:
        return self._pixelwise_metrics

    def set_results(self, results):
        """
        Updates the internal values only for the "regular" metrics (i.e. excluding the extra metrics)
        :param results:
        :return:
        """
        self._patientwise_metrics = []
        self._pixelwise_metrics = []
        self._objectwise_metrics = []
        for index in range(len(results)):
            thr_results = results[index][0]
            thr_val = thr_results[2]
            pixelwise_values = thr_results[3:7]
            patientwise_values = thr_results[7:10]
            objectwise_values = thr_results[10:SharedResources.getInstance().upper_default_metrics_index]
            extra_values = thr_results[SharedResources.getInstance().upper_default_metrics_index:]
            self._pixelwise_metrics.append([thr_val] + pixelwise_values)
            self._patientwise_metrics.append([thr_val] + patientwise_values)
            self._objectwise_metrics.append([thr_val] + objectwise_values)

    def init_from_file(self, scores_filename: str) -> None:
        if not os.path.exists(scores_filename):
            return

        scores_df = pd.read_csv(scores_filename)
        scores_df['Patient'] = scores_df.Patient.astype(str)
        if len(scores_df.loc[(scores_df["Patient"] == self._patient_id) & (scores_df["Fold"] == self._fold_number)]) == 0:
            return

        patient_class_scores = scores_df.loc[(scores_df["Patient"] == self._patient_id) & (scores_df["Fold"] == self._fold_number)]
        self._patientwise_metrics = []
        self._pixelwise_metrics = []
        self._objectwise_metrics = []
        self._extra_metrics = []
        for thr in list(np.unique(patient_class_scores["Threshold"].values)):
            thr_results = patient_class_scores.loc[patient_class_scores["Threshold"] == thr].values[0]
            thr_val = thr_results[2]
            pixelwise_values = list(thr_results[3:7])
            patientwise_values = list(thr_results[7:10])
            objectwise_values = list(thr_results[10:SharedResources.getInstance().upper_default_metrics_index])
            extra_values = list(thr_results[SharedResources.getInstance().upper_default_metrics_index:])
            [extra_values.append(float('nan')) for x in range(len(list(thr_results[SharedResources.getInstance().upper_default_metrics_index:])), 2 * len(SharedResources.getInstance().validation_metric_names))]
            extra_values_description = list(scores_df.columns[SharedResources.getInstance().upper_default_metrics_index:])
            extra_metric_names = []
            for m in SharedResources.getInstance().validation_metric_names:
                extra_metric_names.extend([f'PiW {m}', f'OW {m}'])
            [extra_values_description.append(x) for x in extra_metric_names if x not in extra_values_description]
            self._pixelwise_metrics.append([thr_val] + pixelwise_values)
            self._patientwise_metrics.append([thr_val] + patientwise_values)
            self._objectwise_metrics.append([thr_val] + objectwise_values)
            extra_values_cat = [[x, y] for x, y in zip(extra_values_description, extra_values)]
            if len(extra_values_cat) != 0:
                self._extra_metrics.append([thr_val] + extra_values_cat)
        if len(self._extra_metrics) == 0:
            self._extra_metrics = None

    def is_complete(self):
        """

        :return:
        """
        status = False
        complete_pixelwise_metrics = True not in [-1. in x for x in
                                                  self._pixelwise_metrics] if self._pixelwise_metrics is not None else False
        complete_objectwise_metrics = True not in [-1. in x for x in
                                                  self._objectwise_metrics] if self._objectwise_metrics is not None else False
        if 'objectwise' not in SharedResources.getInstance().validation_metric_spaces:
            status = complete_pixelwise_metrics
        else:
            status = complete_pixelwise_metrics & complete_objectwise_metrics
        return status

    def get_all_metrics(self):
        return [[self._pixelwise_metrics[x][0]] + self._pixelwise_metrics[x][1:] + self._patientwise_metrics[x][1:] + self._objectwise_metrics[x][1:] for x in range(len(self._pixelwise_metrics))]

    def get_extra_metrics(self):
        return self._extra_metrics

    def set_extra_metrics(self, optimal_threshold: float, metrics_values: List):
        for i in range(len(self._extra_metrics)):
            if self._extra_metrics[i][0] == optimal_threshold:
                self._extra_metrics[i][1:] = metrics_values
                return True
        return False

    def get_extra_metrics_without_header(self):
        if self._extra_metrics:
            return [x[1][1::2] for x in self._extra_metrics]
        else:
            #return [[None]] * len(self._pixelwise_metrics)
            # return [[None] * len(SharedResources.getInstance().validation_metric_names)] * len(self._pixelwise_metrics)
            # Twice the length because pixelwise and objectwise versions.
            return [[None] * 2 * len(SharedResources.getInstance().validation_metric_names)] * len(self._pixelwise_metrics)

    def get_probability_thresholds_list(self) -> List[float]:
        res = [x[0] for x in self.pixelwise_metrics]
        return res

    def setup_extra_metrics(self, metric_names):
        """

        """
        thr_list = self.get_probability_thresholds_list()
        if self._extra_metrics is None:
            self._extra_metrics = []
            for thr in thr_list:
                curr_thr = [thr]
                for m in metric_names:
                    curr_thr.append([m, float('nan')])
                self._extra_metrics.append(curr_thr)
        else:
            existing_metrics = [x[0] for x in self._extra_metrics[0][1:]]
            matching_metrics_states = all(element in existing_metrics for element in metric_names)
            if not matching_metrics_states:
                for m in metric_names:
                    if m not in existing_metrics:
                        for th in range(len(self._extra_metrics)):
                            self._extra_metrics[th].append([m, float('nan')])
