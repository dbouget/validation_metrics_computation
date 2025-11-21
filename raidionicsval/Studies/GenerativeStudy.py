import os
import pandas as pd
import numpy as np
from ..Studies.AbstractStudy import AbstractStudy
from ..Validation.validation_utilities_generative import compute_fold_average
from ..Plotting.generative_plot import plot_fold_average
from ..Utils.latex_converter_generative import export_fold_averages_latex
from ..Utils.resources import SharedResources


class GenerativeStudy(AbstractStudy):

    def __init__(self):
        super().__init__()

    def run(self):
        """

        :return:

        Examples

        """
        results_filename = os.path.join(self.input_folder, 'Validation', 'all_scores.csv')
        results_df = pd.read_csv(results_filename, na_values="NaN")
        results_df.replace('inf', np.nan, inplace=True)

        compute_fold_average(folder=self.output_folder, data=results_df)
        export_fold_averages_latex(os.path.join(self.output_folder, 'folds_metrics_average.csv'))
        plot_fold_average(folder=self.output_folder)
