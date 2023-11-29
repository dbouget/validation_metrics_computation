from ..Studies.AbstractStudy import AbstractStudy


class SegmentationStudy(AbstractStudy):

    def __init__(self):
        super().__init__()

    def run(self):
        for c in self.class_names:
            super().compute_and_plot_overall(c, category='All')
            super().compute_and_plot_overall(c, category='True Positive')
            # The 'GT volume (ml)' column is a default column computed during the validation phase
            self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice', metric2='GT volume (ml)',
                                                                metric2_cutoffs=[1.], category='All')
            # self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='HD95', metric2='GT volume (ml)',
            #                                                     metric2_cutoffs=[0.], category='All')
            # Other information, such as 'SpacZ', must be provided as part of the self.extra_patient_parameters
            # self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice', metric2='SpacZ',
            #                                                     metric2_cutoffs=[2.], category='All')

            # compute_overall_metrics_correlation(self.input_folder, best_threshold=self.optimal_threshold)  # Not tested yet
