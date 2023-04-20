from Utils.resources import SharedResources
from Studies.hgg_preop_segmentation import *
# from Studies.hgg_postop_segmentation import *


class StudyConnector:
    """
    Instantiate the proper class corresponding to the manually-input study name.
    """
    def __init__(self):
        self.perform_study = SharedResources.getInstance().studies_task

    def run(self):
        if self.perform_study == 'hgg_preop_seg':
            processor = HGGPreopSegmentationStudy()
            processor.run()
        # elif self.perform_study == 'hgg_postop_seg':
        #     processor = HGGPostOpSegmentationStudy()
        #     processor.run()
