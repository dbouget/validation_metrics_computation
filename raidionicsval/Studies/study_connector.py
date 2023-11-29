from ..Utils.resources import SharedResources
from ..Studies.SegmentationStudy import SegmentationStudy


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
