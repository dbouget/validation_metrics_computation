import os
import configparser
import datetime
import dateutil
import shutil
import traceback
from os.path import expanduser
import nibabel as nib
import SimpleITK as sitk
from copy import deepcopy

import numpy as np
import json
import logging
from typing import Union, Any, Tuple, List

import pandas as pd


class PatientMetrics:
    _unique_id = ""  # Internal unique identifier for the patient
    _ground_truth_filepaths = None
    _prediction_filepaths = None
    _patientwise_metrics = None
    _pixelwise_metrics = None
    _objectwise_metrics = None
    _extra_metrics = None
    _class_metrics = None

    def __init__(self, id: str, class_names: List[str]) -> None:
        """

        """
        self.__reset()
        self._unique_id = id
        self._class_metrics = {}
        for c in class_names:
            self._class_metrics[c] = ClassMetrics(c, self._unique_id)

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = ""
        self._prediction_filepaths = None
        self._ground_truth_filepaths = None
        self._patientwise_metrics = None
        self._pixelwise_metrics = None
        self._objectwise_metrics = None
        self._extra_metrics = None
        self._class_metrics = None

    @property
    def unique_id(self) -> str:
        return self._unique_id

    def init_from_file(self, study_folder: str):
        all_scores_filename = os.path.join(study_folder, 'all_dice_scores.csv')

        for c in list(self._class_metrics.keys()):
            class_scores_filename = os.path.join(study_folder, c + '_dice_scores.csv')
            self._class_metrics[c].init_from_file(class_scores_filename)

        if not os.path.exists(all_scores_filename):
            return
        scores_df = pd.read_csv(all_scores_filename)
        scores_df['Patient'] = scores_df.Patient.astype(str)
        if len(scores_df.loc[scores_df["Patient"] == self._unique_id]) == 0:
            return

        patient_class_scores = scores_df.loc[scores_df["Patient"] == self._unique_id]
        self._patientwise_metrics = []
        self._pixelwise_metrics = []
        self._objectwise_metrics = []
        for thr in list(patient_class_scores["Threshold"].values):
            thr_results = patient_class_scores.loc[patient_class_scores["Threshold"] == thr].values[0]
            thr_val = thr_results[2]
            pixelwise_values = list(thr_results[3:7])
            patientwise_values = list(thr_results[7:14])
            objectwise_values = list(thr_results[14:])
            self._pixelwise_metrics.append([thr_val] + pixelwise_values)
            self._patientwise_metrics.append([thr_val] + patientwise_values)
            self._objectwise_metrics.append([thr_val] + objectwise_values)

    def is_complete(self):
        """
        @TODO. Will require much deeper checks to see if any value is missing and a recompute triggered
        :return:
        """
        if self._pixelwise_metrics is None:
            return False
        else:
            return True

    def set_patient_filenames(self, filenames: dict) -> None:
        self._ground_truth_filepaths = []
        self._prediction_filepaths = []

        for c in list(filenames.keys()):
            self._ground_truth_filepaths.append(filenames[c][0])
            self._prediction_filepaths.append(filenames[c][1])

    def get_class_filenames(self, class_index: int) -> List[str]:
        return [self._ground_truth_filepaths[class_index], self._prediction_filepaths[class_index]]

    def set_class_metrics(self, class_name: str, results: list):
        self._class_metrics[class_name].set_results(results)

    def get_class_metrics(self, class_name: str):
        return self._class_metrics[class_name].get_all_metrics()


class ClassMetrics:
    _unique_id = ""  # Internal unique identifier for the class
    _patient_id = None
    _patientwise_metrics = None
    _pixelwise_metrics = None
    _objectwise_metrics = None
    _extra_metrics = None

    def __init__(self, id: str, patient_id: str) -> None:
        """

        """
        self.__reset()
        self._unique_id = id
        self._patient_id = patient_id

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = ""
        self._patient_id = None
        self._patientwise_metrics = None
        self._pixelwise_metrics = None
        self._objectwise_metrics = None
        self._extra_metrics = None

    @property
    def unique_id(self) -> str:
        return self._unique_id

    def set_results(self, results):
        self._patientwise_metrics = []
        self._pixelwise_metrics = []
        self._objectwise_metrics = []
        for index in range(len(results)):
            thr_results = results[index][0]
            thr_val = thr_results[2]
            pixelwise_values = thr_results[3:7]
            patientwise_values = thr_results[7:14]
            objectwise_values = thr_results[14:]
            self._pixelwise_metrics.append([thr_val] + pixelwise_values)
            self._patientwise_metrics.append([thr_val] + patientwise_values)
            self._objectwise_metrics.append([thr_val] + objectwise_values)

    def init_from_file(self, scores_filename: str) -> None:
        if not os.path.exists(scores_filename):
            return
        scores_df = pd.read_csv(scores_filename)
        scores_df['Patient'] = scores_df.Patient.astype(str)
        if len(scores_df.loc[scores_df["Patient"] == self._patient_id]) == 0:
            return

        patient_class_scores = scores_df.loc[scores_df["Patient"] == self._patient_id]
        self._patientwise_metrics = []
        self._pixelwise_metrics = []
        self._objectwise_metrics = []
        for thr in list(patient_class_scores["Threshold"].values):
            thr_results = patient_class_scores.loc[patient_class_scores["Threshold"] == thr].values[0]
            thr_val = thr_results[2]
            pixelwise_values = list(thr_results[3:7])
            patientwise_values = list(thr_results[7:14])
            objectwise_values = list(thr_results[14:])
            self._pixelwise_metrics.append([thr_val] + pixelwise_values)
            self._patientwise_metrics.append([thr_val] + patientwise_values)
            self._objectwise_metrics.append([thr_val] + objectwise_values)

    def get_all_metrics(self):
        return [[self._pixelwise_metrics[x][0]] + self._pixelwise_metrics[x][1:] + self._patientwise_metrics[x][1:] + self._objectwise_metrics[x][1:] for x in range(len(self._pixelwise_metrics))]
