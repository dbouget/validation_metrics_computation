import logging
from ..Studies.AbstractStudy import AbstractStudy
from ..Utils.resources import SharedResources


class SegmentationStudy(AbstractStudy):

    def __init__(self):
        super().__init__()

    def run(self):
        """

        :return:

        Examples
            # The 'GT volume (ml)' column is a default column computed during the validation phase, here showing
            # the results for two categories: objects to detect larger than 4 ml and objects smaller than 4 ml.
            # Would correspond in the config file to: selections_dense=PiW Dice,GT volume (ml),4,All
            self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice',
                                                                metric2='GT volume (ml)', metric2_cutoffs=[4.],
                                                                category='All')
            # Showing the relationship between Hausdorff and the object to detect volumes
            # Would correspond in the config file to: selections_dense=HD95,GT volume (ml),0,All
            self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='HD95', metric2='GT volume (ml)',
                                                                metric2_cutoffs=[0.], category='All')

            # Other information, such as the image spacing along one axis (e.g., 'SpacZ'), can be used to further analyze the results
            # Such parameters must be provided as part of the self.studies_extra_parameters_filename
            # Would correspond in the config file to: selections_dense=PiW Dice,SpacZ,2,All
            self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice', metric2='SpacZ',
                                                                metric2_cutoffs=[2.], category='All')
        """

        if len(self.class_names):
            logging.warning("No class names were provided, the study will not run as intended!")

        for c in self.class_names:
            super().compute_and_plot_overall(c, category='All')
            super().compute_and_plot_overall(c, category='Positive')
            super().compute_and_plot_overall(c, category='TP')

            # Plotting the results based on the selection of dense parameters
            for s in SharedResources.getInstance().studies_selections_dense:
                parsing = s.split(',')
                metric1 = parsing[0]
                metric2 = parsing[1]
                if parsing[2] != '':
                    metric2_cutoff = [float(x) for x in parsing[2].split('-')]
                else:
                    metric2_cutoff = None
                category = parsing[3]
                self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1=metric1, metric2=metric2,
                                                                    metric2_cutoffs=metric2_cutoff, category=category)

            # Plotting the results based on the selection of categorical parameters
            for s in SharedResources.getInstance().studies_selections_categorical:
                parsing = s.split(',')
                metric1 = parsing[0].strip()
                metric2 = parsing[1].strip()
                if parsing[2].strip() != '':
                    metric2_cutoff = [x for x in parsing[2].split('-')]
                else:
                    metric2_cutoff = None
                category = parsing[3].strip()
                self.compute_and_plot_categorical_metric_over_metric_categories(class_name=c, metric1=metric1,
                                                                                metric2=metric2,
                                                                                metric2_cutoffs=metric2_cutoff,
                                                                                category=category)

            # Correlation matrix between all metrics
            super().compute_and_plot_metrics_correlation_matrix(class_name=c, category='All')
            super().compute_and_plot_metrics_correlation_matrix(class_name=c, category='Positive')
            super().compute_and_plot_metrics_correlation_matrix(class_name=c, category='TP')

            # Cascading results based on a combination of the selected dense/categorical parameters
            self.compute_and_plot_metric_over_metric_cascading_categories(class_name=c, category='All')
            self.compute_and_plot_metric_over_metric_cascading_categories(class_name=c, category='Positive')
            self.compute_and_plot_metric_over_metric_cascading_categories(class_name=c, category='TP')
