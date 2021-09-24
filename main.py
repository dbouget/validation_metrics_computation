import getopt
import traceback
import sys
import configparser
from Validation.kfold_model_validation import *
from Studies.study_connector import StudyConnector


def main(argv):
    config = configparser.ConfigParser()
    config_file = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config_file=None"])
    except getopt.GetoptError:
        print('main.py -c <config_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -c <config_file>')
            sys.exit()
        elif opt in ("-c", "--config_file"):
            config_file = arg
    config.read(config_file)

    shared = SharedResources.getInstance()
    shared.set_environment(config=config)

    task = config['Default']['task'].split('#')[0].strip()

    try:
        if task == 'validation':
            processor = ModelValidation()
            processor.run()
        elif task == 'study':
            runner = StudyConnector()
            runner.run()
    except Exception as e:
        print('Process could not proceed. Caught error: {}'.format(e.args[0]))
        print('{}'.format(traceback.format_exc()))


if __name__ == "__main__":
    main(sys.argv[1:])

