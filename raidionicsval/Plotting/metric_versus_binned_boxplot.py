import sys
import os
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def compute_binned_metric_over_metric_boxplot(folder, data, metric1, metric2, criterion1=0., postfix="",
                                              number_bins=10):
    metric1_name = metric1.replace(' ', '')
    metric2_name = metric2.replace(' ', '')
    dump_filename = os.path.join(folder, metric1_name + '_over_' + metric2_name + postfix + '.txt')
    scatterplot_fn = os.path.join(folder, "scatter_plots")
    boxplot_fn = os.path.join(folder, "boxplots")
    os.makedirs(scatterplot_fn, exist_ok=True)
    os.makedirs(boxplot_fn, exist_ok=True)
    tmp_stoud = sys.stdout
    dump_file = open(dump_filename, "w")
    sys.stdout = dump_file

    try:
        nb_bins = number_bins
        binned, edges = pd.cut(data[metric2], bins=nb_bins, retbins=True, precision=2)
        # To make sure all values have two digits after the comma, for symmetrical plots.
        binned = binned.apply(lambda x: pd.Interval(left=round(x.left, 2), right=round(x.right, 2)))
        normal_bins_metric1 = []
        normal_bins_selected_metric1 = []
        normal_bins_recalls = []
        total_selected = 0
        total_to_find = 0
        print('-------------- NORMAL BINS --------------------------------\n')
        for b in range(nb_bins):
            selection = data.loc[(data[metric2] > edges[b]) & (data[metric2] <= edges[b + 1])]
            average_metric1 = np.mean(selection[metric1].values)
            nb_found = np.count_nonzero(selection.loc[selection[metric1] > criterion1][metric1].values)
            nb_missed = selection[metric1].shape[0] - nb_found
            normal_bins_metric1.append(average_metric1)
            total_selected = total_selected + nb_found
            total_to_find = total_to_find + selection.shape[0]
            if nb_missed != 0:
                found_avg_metric1 = np.mean(selection.loc[selection[metric1] > criterion1][metric1].values)
                normal_bins_selected_metric1.append(found_avg_metric1)
                normal_bins_recalls.append(nb_found/selection.shape[0])
                print('Average ' + metric1 + ' of {:.2f} ({:.2f} found) for bin [{:.2f}, {:.2f}] for {}/{} tumors'.format(average_metric1,
                                                                                                               found_avg_metric1,
                                                                                                               round(edges[b], 2),
                                                                                                               round(edges[b + 1], 2),
                                                                                                               nb_found,
                                                                                                               selection.shape[
                                                                                                                   0]))
            else:
                normal_bins_selected_metric1.append(average_metric1)
                normal_bins_recalls.append(1.0)
                print('Average ' + metric1 + ' of {:.2f} for bin [{:.2f}, {:.2f}] for {}/{} tumors'.format(average_metric1, round(edges[b], 2),
                                                                                                round(edges[b + 1], 2),
                                                                                                nb_found,
                                                                                                selection.shape[0]))
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"

        fig1, ax1 = plt.subplots()
        ax1.set(xlabel=metric2 + ' bins (equal-sized intervals)', ylabel='Probability')#, title='Dice over tumor volume.')
        ax1.scatter(list(range(nb_bins)), normal_bins_metric1, marker='o', label=metric1 + ' (all tumors)')
        ax1.scatter(list(range(nb_bins)), normal_bins_selected_metric1, marker='o', label=metric1 + ' (selected tumors)')
        ax1.scatter(list(range(nb_bins)), normal_bins_recalls, marker='^', label='Recall')
        ax1.grid(linestyle='--')
        ax1.set_xlim(-0.05, nb_bins)
        ax1.set_ylim(0., 1.05)
        # ax.title()
        ax1.legend(loc='lower right')
        # plt.show()
        fig1.savefig(os.path.join(scatterplot_fn, metric1_name + '_scatter_over_' + metric2_name + '_normal_bins' +
                                  postfix + '.png'), dpi=300, bbox_inches="tight")
        plt.close(fig1)
        plt.clf()

        sns_plot = sns.boxplot(x=data[metric1], y=binned)
        sns_plot.figure.savefig(os.path.join(boxplot_fn, 'boxplot_' + metric1_name + '_over_' + metric2_name +
                                             '_normal_bins' + postfix + '.png'), dpi=300, bbox_inches="tight")
        plt.close(sns_plot.figure)

        binned, edges = pd.qcut(data[metric2], q=nb_bins, retbins=True, precision=2, duplicates='drop')
        nb_bins = len(edges) - 1
        binned = binned.apply(lambda x: pd.Interval(left=round(x.left, 2), right=round(x.right, 2)))
        equal_bins_metric1 = []
        equal_bins_selected_metric1 = []
        equal_bins_recall = []
        print('\n\n-------------- EQUAL BINS --------------------------------\n')
        for b in range(nb_bins):
            selection = data.loc[(data[metric2] > edges[b]) & (data[metric2] <= edges[b + 1])]
            average_metric1 = np.mean(selection[metric1].values)
            nb_found = np.count_nonzero(selection.loc[selection[metric1] > criterion1][metric1].values)
            nb_missed = selection[metric1].shape[0] - nb_found
            equal_bins_metric1.append(average_metric1)
            if nb_missed != 0:
                found_avg_metric1 = np.mean(selection.loc[selection[metric1] > criterion1][metric1].values)
                equal_bins_selected_metric1.append(found_avg_metric1)
                equal_bins_recall.append(nb_found/selection.shape[0])
                print('Average ' + metric1 + ' of {:.2f} ({:.2f} found) for bin [{:.2f}, {:.2f}] for {}/{} tumors'.format(average_metric1,
                                                                                                               found_avg_metric1,
                                                                                                               round(edges[b], 2),
                                                                                                               round(edges[b + 1], 2),
                                                                                                               nb_found,
                                                                                                               selection.shape[
                                                                                                                   0]))
            else:
                equal_bins_selected_metric1.append(average_metric1)
                equal_bins_recall.append(1.0)
                print('Average ' + metric1 + ' of {:.2f} for bin [{:.2f}, {:.2f}] for {}/{} tumors'.format(average_metric1, round(edges[b], 2),
                                                                                                round(edges[b + 1], 2),
                                                                                                nb_found,
                                                                                                selection.shape[0]))
        fig2, ax2 = plt.subplots()
        ax2.set(xlabel=metric2 + ' bins (equal-populated intervals)', ylabel='Probability')#, title='Dice over tumor volume.')
        ax2.scatter(list(range(nb_bins)), equal_bins_metric1, marker='o', label=metric1 + ' (all tumors)')
        ax2.scatter(list(range(nb_bins)), equal_bins_selected_metric1, marker='o', label=metric1 + ' (selected tumors)')
        ax2.scatter(list(range(nb_bins)), equal_bins_recall, marker='^', label='Recall')
        ax2.grid(linestyle='--')
        ax2.set_xlim(-0.05, nb_bins)
        ax2.set_ylim(0., 1.05)
        ax2.legend(loc='lower right')
        # plt.show()
        fig2.savefig(os.path.join(scatterplot_fn, metric1_name + '_scatter_over_' + metric2_name + '_equal_bins' +
                                  postfix + '.png'), dpi=300, bbox_inches="tight")
        plt.close(fig2)
        plt.clf()

        sns_plot = sns.boxplot(x=data[metric1], y=binned)
        sns_plot.figure.savefig(os.path.join(boxplot_fn, 'boxplot_' + metric1_name + '_over_' + metric2_name +
                                             '_equal_bins' + postfix + '.png'), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close(sns_plot.figure)

        print('\n\nTotal selected: {}, total to find: {}. Recall: {:.4f}'.format(total_selected, total_to_find,
                                                                                 total_selected / total_to_find))
        sys.stdout = tmp_stoud
        dump_file.close()
    except Exception as e:
        dump_file.close()
        sys.stdout = tmp_stoud
        print('{}'.format(traceback.format_exc()))
