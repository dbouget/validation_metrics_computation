import os
import pandas as pd
from ..Studies.AbstractStudy import AbstractStudy
from ..Utils.resources import SharedResources
from ..Plotting.classification_plot import confusion_matrix_plot


class ClassificationStudy(AbstractStudy):

    def __init__(self):
        super().__init__()

    def run(self):
        """

        :return:

        Examples

        """
        results_filename = os.path.join(self.input_folder, 'Validation', 'all_scores.csv')
        results_df = pd.read_csv(results_filename)
        confusion_matrix_plot(gt=results_df["GT"].values, pred=results_df["Prediction"].values,
                              classes=self.class_names, output_dir=self.output_folder)

        for c in self.class_names:
            pass
