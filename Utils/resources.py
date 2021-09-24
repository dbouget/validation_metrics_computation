import os
import sys
import logging
logger = logging.getLogger(__name__)


class SharedResources:
    """
    Singleton class to have access from anywhere in the code at the resources/parameters.
    """
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if SharedResources.__instance == None:
            SharedResources()
        return SharedResources.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if SharedResources.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            SharedResources.__instance = self

    def set_environment(self, config):
        self.config = config

        self.home_path = ''
        if os.name == 'posix':  # Linux system
            self.home_path = os.path.expanduser("~")

        self.__parse_default_parameters()
        self.__parse_validation_parameters()
        self.__parse_studies_parameters()

    def __parse_default_parameters(self):
        """
        Parse the user-selected configuration parameters linked to the overall behaviour.
        :param: data_root: main folder entry-point containing the raw data (assuming a specific folder structure).
        :param: task: identifier for the task to perform, for now validation or study
        :return:
        """
        self.data_root = ""
        self.task = None

        if self.config.has_option('Default', 'data_root'):
            if self.config['Default']['data_root'].split('#')[0].strip() != '':
                self.data_root = self.config['Default']['data_root'].split('#')[0].strip()

        if self.config.has_option('Default', 'task'):
            if self.config['Default']['task'].split('#')[0].strip() != '':
                self.task = self.config['Default']['task'].split('#')[0].strip()

    def __parse_studies_parameters(self):
        """
        Parse the user-selected configuration parameters linked to the study process (plotting and visualization).
        :param: studies_input_folder: main directory containing the validation results.
        :param: (optional) studies_output_folder: destination directory where the study results will be saved.
        If empty, the study results will be saved in the studies_input_folder location.
        :param: studies_task: identifier for the study script to run. Each identified should link to a python file in
        the /Studies sub-directory.
        :param: studies_study_name: folder name for the specific study.
        :param: studies_extra_parameters_filename: resources file containing patient-specific information, for example
        the tumor volume, data origin, etc... for in-depth results analysis.
        :return:
        """
        self.studies_input_folder = ''
        self.studies_output_folder = ''
        self.studies_task = ''
        self.studies_study_name = None
        self.studies_extra_parameters_filename = ''

        if self.config.has_option('Studies', 'input_folder'):
            if self.config['Studies']['input_folder'].split('#')[0].strip() != '':
                self.studies_input_folder = self.config['Studies']['input_folder'].split('#')[0].strip()

        if self.config.has_option('Studies', 'output_folder'):
            if self.config['Studies']['output_folder'].split('#')[0].strip() != '':
                self.studies_output_folder = self.config['Studies']['output_folder'].split('#')[0].strip()

        if self.config.has_option('Studies', 'task'):
            if self.config['Studies']['task'].split('#')[0].strip() != '':
                self.studies_task = self.config['Studies']['task'].split('#')[0].strip()

        if self.config.has_option('Studies', 'study_name'):
            if self.config['Studies']['study_name'].split('#')[0].strip() != '':
                self.studies_study_name = self.config['Studies']['study_name'].split('#')[0].strip()

        if self.config.has_option('Studies', 'extra_parameters_filename'):
            if self.config['Studies']['extra_parameters_filename'].split('#')[0].strip() != '':
                self.studies_extra_parameters_filename = self.config['Studies']['extra_parameters_filename'].split('#')[0].strip()

    def __parse_validation_parameters(self):
        """
        Parse the user-selected configuration parameters linked to the validation process.
        :param: validation_input_folder: main directory containing a network's predictions.
        :param: (optional) validation_output_folder: destination directory where the validation results will be saved.
        If empty, the validation results will be saved in the validation_input_folder location.
        :param: validation_nb_folds: number of folds for the k-fold cross-validation.
        :param: validation_split_way: specification regarding the training approach. If only a train and validation set
        were used then the keyword two-way must be used. Otherwise, if a train/validation/test set distribution was used
        then the keyword three-way must be used.
        :param: validation_metric_names: list of metric names which should be computed in addition to the default ones.
        The exhaustive list of supported metrics can be found in extra_metrics_computation.py
        :param: validation_detection_overlap_thresholds: list of Dice score thresholds to use for considering true
        positive detections at the patient level.
        e.g., 0.25 means that a Dice of at least 25% must be reached to consider the network's prediction as a true
        positive.
        :param: validation_prediction_files_suffix: suffix to append to the input sample name (from the list in
        cross_validation_folds.txt) in order to generate the network's prediction filename, including its extension.
        :return:
        """
        self.validation_input_folder = ''
        self.validation_output_folder = ''
        self.validation_nb_folds = 5
        self.validation_split_way = 'two-way'
        self.validation_metric_names = []
        self.validation_detection_overlap_thresholds = []
        self.validation_prediction_files_suffix = ''

        if self.config.has_option('Validation', 'input_folder'):
            if self.config['Validation']['input_folder'].split('#')[0].strip() != '':
                self.validation_input_folder = self.config['Validation']['input_folder'].split('#')[0].strip()

        if self.config.has_option('Validation', 'output_folder'):
            if self.config['Validation']['output_folder'].split('#')[0].strip() != '':
                self.validation_output_folder = self.config['Validation']['output_folder'].split('#')[0].strip()

        if self.config.has_option('Validation', 'nb_folds'):
            if self.config['Validation']['nb_folds'].split('#')[0].strip() != '':
                self.validation_nb_folds = int(self.config['Validation']['nb_folds'].split('#')[0].strip())

        if self.config.has_option('Validation', 'split_way'):
            if self.config['Validation']['split_way'].split('#')[0].strip() != '':
                self.validation_split_way = self.config['Validation']['split_way'].split('#')[0].strip()

        if self.config.has_option('Validation', 'extra_metrics'):
            if self.config['Validation']['extra_metrics'].split('#')[0].strip() != '':
                self.validation_metric_names = self.config['Validation']['extra_metrics'].split('#')[0].strip().split(',')

        if self.config.has_option('Validation', 'detection_overlap_thresholds'):
            if self.config['Validation']['detection_overlap_thresholds'].split('#')[0].strip() != '':
                self.validation_detection_overlap_thresholds = [float(x) for x in self.config['Validation']['detection_overlap_thresholds'].split('#')[0].strip().split(',')]
        if len(self.validation_detection_overlap_thresholds) == 0:
            self.validation_detection_overlap_thresholds = [0.]

        if self.config.has_option('Validation', 'prediction_files_suffix'):
            if self.config['Validation']['prediction_files_suffix'].split('#')[0].strip() != '':
                self.validation_prediction_files_suffix = self.config['Validation']['prediction_files_suffix'].split('#')[0].strip()