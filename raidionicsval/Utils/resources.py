import os
import sys
import logging
import configparser
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
            self.__setup()

    def __setup(self):
        """
        Definition all of attributes accessible through this singleton.
        """
        self.config_filename = None
        self.config = None
        self.home_path = ''

        self.upper_default_metrics_index = 23  # Single place for holding this attribute, "safer" approach, column index after # GT and # Det

        self.data_root = ""
        self.task = None
        self.number_processes = 8
        self.overall_objective = "segmentation"

        self.studies_input_folder = ''
        self.studies_output_folder = ''
        self.studies_task = ''
        self.studies_extra_parameters_filename = ''
        self.studies_class_names = []
        self.studies_selections_dense = []
        self.studies_selections_categorical = []

        self.validation_input_folder = ''
        self.validation_output_folder = ''
        self.validation_gt_files_suffix = []
        self.validation_prediction_files_suffix = []
        self.validation_use_index_naming_convention = True
        self.validation_nb_folds = 5
        self.validation_split_way = 'two-way'
        self.validation_metric_spaces = []
        self.validation_metric_names = []
        self.validation_detection_overlap_thresholds = []
        self.validation_tiny_objects_removal_threshold = 50
        self.validation_class_names = []
        self.validation_true_positive_volume_thresholds = []
        self.validation_use_brats_data = []

        self.standalone_gt_filename = None
        self.standalone_detection_filename = None
        self.standalone_class_names = []
        self.standalone_metrics_spaces = []
        self.standalone_extra_metric_names = []
        self.standalone_detection_overlap_thresholds = []
        self.standalone_tiny_objects_removal_threshold = 50
        self.standalone_true_positive_volume_thresholds = []

    def set_environment(self, config_filename):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_filename):
            pass

        self.config_filename = config_filename
        self.config.read(self.config_filename)

        if os.name == 'posix':  # Linux system
            self.home_path = os.path.expanduser("~")

        self.__parse_default_parameters()
        self.__parse_validation_parameters()
        self.__parse_studies_parameters()
        self.__parse_standalone_parameters()

    def __parse_default_parameters(self):
        """
        Parse the user-selected configuration parameters linked to the overall behaviour.
        :param: data_root: (str) main folder entry-point containing the raw data (assuming a specific folder structure).
        :param: task: (str) identifier for the task to perform, for now validation or study
        :param: number_processes: (int) number of parallel processes to use to perform the different task
        :return:
        """
        if self.config.has_option('Default', 'data_root'):
            if self.config['Default']['data_root'].split('#')[0].strip() != '':
                self.data_root = self.config['Default']['data_root'].split('#')[0].strip()

        if self.config.has_option('Default', 'task'):
            if self.config['Default']['task'].split('#')[0].strip() != '':
                self.task = self.config['Default']['task'].split('#')[0].strip()

        if self.config.has_option('Default', 'number_processes'):
            if self.config['Default']['number_processes'].split('#')[0].strip() != '':
                self.number_processes = int(self.config['Default']['number_processes'].split('#')[0].strip())

        if self.config.has_option('Default', 'objective'):
            if self.config['Default']['objective'].split('#')[0].strip() != '':
                self.overall_objective = self.config['Default']['objective'].split('#')[0].strip()
        if self.overall_objective not in ["segmentation", "classification"]:
            raise ValueError("Provided ['Default']['objective'] should be inside [segmentation, classification]."
                             "\n Please provide a correct value!")

    def __parse_studies_parameters(self):
        """
        Parse the user-selected configuration parameters linked to the study process (plotting and visualization).
        :param: studies_input_folder: main directory containing the validation results.
        :param: studies_output_folder: destination directory where the study results will be saved.
        If empty, the study results will be saved in the studies_input_folder location.
        :param: studies_task: identifier for the study script to run. Each identified should link to a python file in
        the /Studies sub-directory.
        :param: studies_extra_parameters_filename: resources file containing patient-specific information, for example
        the tumor volume, data origin, etc... for in-depth results analysis.
        :return:
        """
        if self.config.has_option('Studies', 'input_folder'):
            if self.config['Studies']['input_folder'].split('#')[0].strip() != '':
                self.studies_input_folder = self.config['Studies']['input_folder'].split('#')[0].strip()

        if self.config.has_option('Studies', 'output_folder'):
            if self.config['Studies']['output_folder'].split('#')[0].strip() != '':
                self.studies_output_folder = self.config['Studies']['output_folder'].split('#')[0].strip()

        if self.config.has_option('Studies', 'task'):
            if self.config['Studies']['task'].split('#')[0].strip() != '':
                self.studies_task = self.config['Studies']['task'].split('#')[0].strip()

        if self.config.has_option('Studies', 'extra_parameters_filename'):
            if self.config['Studies']['extra_parameters_filename'].split('#')[0].strip() != '':
                self.studies_extra_parameters_filename = self.config['Studies']['extra_parameters_filename'].split('#')[0].strip()

        if self.config.has_option('Studies', 'class_names'):
            if self.config['Studies']['class_names'].split('#')[0].strip() != '':
                self.studies_class_names = [x.strip() for x in
                                            self.config['Studies']['class_names'].split('#')[0].strip().split(',')]

        if self.config.has_option('Studies', 'selections_dense'):
            if self.config['Studies']['selections_dense'].split('#')[0].strip() != '':
                self.studies_selections_dense = [x.strip() for x in self.config['Studies']['selections_dense'].split('#')[0].strip().split('\\')]

        if self.config.has_option('Studies', 'selections_categorical'):
            if self.config['Studies']['selections_categorical'].split('#')[0].strip() != '':
                self.studies_selections_categorical = [x.strip() for x in self.config['Studies']['selections_categorical'].split('#')[0].strip().split('\\')]

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
        if self.config.has_option('Validation', 'input_folder'):
            if self.config['Validation']['input_folder'].split('#')[0].strip() != '':
                self.validation_input_folder = self.config['Validation']['input_folder'].split('#')[0].strip()

        if self.config.has_option('Validation', 'output_folder'):
            if self.config['Validation']['output_folder'].split('#')[0].strip() != '':
                self.validation_output_folder = self.config['Validation']['output_folder'].split('#')[0].strip()

        if self.config.has_option('Validation', 'gt_files_suffix'):
            if self.config['Validation']['gt_files_suffix'].split('#')[0].strip() != '':
                self.validation_gt_files_suffix = [x.strip() for x in self.config['Validation']['gt_files_suffix'].split('#')[0].strip().split(',')]

        if self.config.has_option('Validation', 'prediction_files_suffix'):
            if self.config['Validation']['prediction_files_suffix'].split('#')[0].strip() != '':
                self.validation_prediction_files_suffix = [x.strip() for x in self.config['Validation']['prediction_files_suffix'].split('#')[0].strip().split(',')]

        if self.config.has_option('Validation', 'use_index_naming_convention'):
            if self.config['Validation']['use_index_naming_convention'].split('#')[0].strip() != '':
                self.validation_use_index_naming_convention = True \
                    if self.config['Validation']['use_index_naming_convention'].split('#')[0].strip().lower() == 'true'\
                    else False

        if self.config.has_option('Validation', 'nb_folds'):
            if self.config['Validation']['nb_folds'].split('#')[0].strip() != '':
                self.validation_nb_folds = int(self.config['Validation']['nb_folds'].split('#')[0].strip())

        if self.config.has_option('Validation', 'split_way'):
            if self.config['Validation']['split_way'].split('#')[0].strip() != '':
                self.validation_split_way = self.config['Validation']['split_way'].split('#')[0].strip()

        if self.config.has_option('Validation', 'metrics_space'):
            if self.config['Validation']['metrics_space'].split('#')[0].strip() != '':
                self.validation_metric_spaces = [x.strip() for x in self.config['Validation']['metrics_space'].split('#')[0].strip().split(',')]

        if self.config.has_option('Validation', 'extra_metrics'):
            if self.config['Validation']['extra_metrics'].split('#')[0].strip() != '':
                self.validation_metric_names = [x.strip() for x in self.config['Validation']['extra_metrics'].split('#')[0].strip().split(',')]

        if self.config.has_option('Validation', 'detection_overlap_thresholds'):
            if self.config['Validation']['detection_overlap_thresholds'].split('#')[0].strip() != '':
                self.validation_detection_overlap_thresholds = [float(x) for x in self.config['Validation']['detection_overlap_thresholds'].split('#')[0].strip().split(',')]
        if len(self.validation_detection_overlap_thresholds) == 0:
            self.validation_detection_overlap_thresholds = [0.]

        if self.config.has_option('Validation', 'class_names'):
            if self.config['Validation']['class_names'].split('#')[0].strip() != '':
                self.validation_class_names = [x.strip() for x in self.config['Validation']['class_names'].split('#')[0].strip().split(',')]

        if self.config.has_option('Validation', 'tiny_objects_removal_threshold'):
            if self.config['Validation']['tiny_objects_removal_threshold'].split('#')[0].strip() != '':
                self.validation_tiny_objects_removal_threshold = int(self.config['Validation']['tiny_objects_removal_threshold'].split('#')[0].strip())

        if self.config.has_option('Validation', 'true_positive_volume_thresholds'):
            if self.config['Validation']['true_positive_volume_thresholds'].split('#')[0].strip() != '':
                self.validation_true_positive_volume_thresholds = [float(x.strip()) for x in self.config['Validation']['true_positive_volume_thresholds'].split('#')[0].strip().split(',')]

        if self.config.has_option('Validation', 'use_brats_data'):
            if self.config['Validation']['use_brats_data'].split('#')[0].strip() != '':
                self.validation_use_brats_data = True if self.config['Validation']['use_brats_data'].split('#')[0].strip().lower() == 'true' else False

    def __parse_standalone_parameters(self):
        """

        """
        if self.config.has_option('Standalone', 'groundtruth_filename'):
            if self.config['Standalone']['groundtruth_filename'].split('#')[0].strip() != '':
                self.standalone_gt_filename = self.config['Standalone']['groundtruth_filename'].split('#')[0].strip()

        if self.config.has_option('Standalone', 'prediction_filename'):
            if self.config['Standalone']['prediction_filename'].split('#')[0].strip() != '':
                self.standalone_detection_filename = self.config['Standalone']['prediction_filename'].split('#')[0].strip()

        if self.config.has_option('Standalone', 'metrics_space'):
            if self.config['Standalone']['metrics_space'].split('#')[0].strip() != '':
                self.standalone_metrics_spaces = [x.strip() for x in self.config['Standalone']['metrics_space'].split('#')[0].strip().split(',')]

        if self.config.has_option('Standalone', 'extra_metrics'):
            if self.config['Standalone']['extra_metrics'].split('#')[0].strip() != '':
                self.standalone_extra_metric_names = [x.strip() for x in self.config['Standalone']['extra_metrics'].split('#')[0].strip().split(',')]

        if self.config.has_option('Standalone', 'detection_overlap_thresholds'):
            if self.config['Standalone']['detection_overlap_thresholds'].split('#')[0].strip() != '':
                self.standalone_detection_overlap_thresholds = [float(x) for x in (self.config['Standalone']['detection_overlap_thresholds'].split('#')[0].strip().split(','))]

        if self.config.has_option('Standalone', 'class_names'):
            if self.config['Standalone']['class_names'].split('#')[0].strip() != '':
                self.standalone_class_names = [x.strip() for x in self.config['Standalone']['class_names'].split('#')[0].strip().split(',')]

        if self.config.has_option('Standalone', 'tiny_objects_removal_threshold'):
            if self.config['Standalone']['tiny_objects_removal_threshold'].split('#')[0].strip() != '':
                self.standalone_tiny_objects_removal_threshold = int(self.config['Standalone']['tiny_objects_removal_threshold'].split('#')[0].strip())

        if self.config.has_option('Standalone', 'true_positive_volume_thresholds'):
            if self.config['Standalone']['true_positive_volume_thresholds'].split('#')[0].strip() != '':
                self.standalone_true_positive_volume_thresholds = [float(x.strip()) for x in self.config['Standalone']['true_positive_volume_thresholds'].split('#')[0].strip().split(',')]