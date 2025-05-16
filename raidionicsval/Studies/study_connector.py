from ..Utils.resources import SharedResources
from ..Studies.SegmentationStudy import SegmentationStudy
from ..Studies.ClassificationStudy import ClassificationStudy


class StudyConnector:
    """
    Instantiate the proper study class corresponding to the user choice from the configuration file.
    """
    def __init__(self):
        self.perform_study = SharedResources.getInstance().studies_task

    def run(self):
        if self.perform_study == 'segmentation':
            processor = SegmentationStudy()
            processor.run()
        elif self.perform_study == 'classification':
            processor = ClassificationStudy()
            processor.run()
