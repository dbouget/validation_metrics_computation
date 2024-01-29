import os
import numpy as np
import matplotlib.pyplot as plt
import traceback
from statsmodels.graphics.agreement import mean_diff_plot


def compute_agreement_plot(folder, array1, array2, postfix=""):
    """

    :param folder:
    :param array1:
    :param array2:
    :param postfix:
    :return:
    """
    folder = os.path.join(folder, 'Blant-Altman')
    os.makedirs(folder, exist_ok=True)

    try:
        fig, ax = plt.subplots()
        ax.set(xlabel='Ground-truth', ylabel='Predictions',
               title='Blant-Altman plot')

        mean_diff_plot(array1, array2, ax=ax)
        ax.grid(linestyle='--')
        fig.savefig(os.path.join(folder, 'blant_altman_plot' + postfix + '.png'), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        plt.clf()
    except Exception as e:
        print('Agreement plot computation could not proceed. Collected: \n')
        print('{}'.format(traceback.format_exc()))
