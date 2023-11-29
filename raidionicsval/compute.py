import traceback
import logging
from .Studies.study_connector import StudyConnector
from .Validation.kfold_model_validation import ModelValidation
from .Utils.resources import SharedResources


def compute(config_filename: str, logging_filename: str = None) -> None:
    """

    :param config_filename:
    :param logging_filename:
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
        print('Compute could not proceed. Issue arose during environment setup. Collected: \n')
        print('{}'.format(traceback.format_exc()))

    task = SharedResources.getInstance().task
    try:
        if task == 'validation':
            processor = ModelValidation()
            processor.run()
        elif task == 'study':
            runner = StudyConnector()
            runner.run()
    except Exception as e:
        print('Compute could not proceed. Issue arose during task {}. Collected: \n'.format(task))
        print('{}'.format(traceback.format_exc()))
