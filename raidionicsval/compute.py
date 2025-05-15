import traceback
import logging
from .Studies.study_connector import StudyConnector
from .Validation.kfold_model_validation import ModelValidation
from .Validation.kfold_model_validation_classification import ClassificationModelValidation
from .Computation.standalone_computation import StandaloneComputation
from .Utils.resources import SharedResources


def compute(config_filename: str, logging_filename: str = None) -> None:
    """

    :param config_filename: Filepath to the *.ini with the user-specific runtime parameters
    :param logging_filename: Filepath to an external file used for logging events (e.g., the Raidionics .log)
    :return:
    """
    try:
        SharedResources.getInstance().set_environment(config_filename=config_filename)
        if logging_filename:
            logger = logging.getLogger()
            handler = logging.FileHandler(filename=logging_filename, mode='a', encoding='utf-8')
            handler.setFormatter(logging.Formatter(fmt="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s",
                                                   datefmt='%d/%m/%Y %H.%M'))
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
    except Exception as e:
        logging.error(f'Compute could not proceed. Issue arose during environment setup.'
                      f' Collected: {e}\n{traceback.format_exc()}')

    task = SharedResources.getInstance().task
    objective = SharedResources.getInstance().overall_objective
    try:
        if task == 'validation' and objective == "segmentation":
            processor = ModelValidation()
            processor.run()
        elif task == 'validation' and objective == "classification":
            processor = ClassificationModelValidation()
            processor.run()
        elif task == 'study':
            runner = StudyConnector()
            runner.run()
        elif task == 'standalone':
            runner = StandaloneComputation()
            runner.run()
    except Exception as e:
        logging.error(f'Compute could not proceed. Issue arose during environment setup.'
                      f' Collected: {e}\n{traceback.format_exc()}')
