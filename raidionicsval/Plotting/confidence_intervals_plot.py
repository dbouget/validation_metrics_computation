import os
import numpy as np
import matplotlib.pyplot as plt
import traceback
from arch.bootstrap import IIDBootstrap


def compute_dice_confidence_intervals(folder, dices, best_overlap, postfix=""):
    folder = os.path.join(folder, 'Dice_CIs')
    os.makedirs(folder, exist_ok=True)

    try:
        best_dices_per_patient = np.array(dices)
        bs_pls = IIDBootstrap(best_dices_per_patient)
        ci_pls = bs_pls.conf_int(np.mean, 10000, method='bca')
        print("Computed confidence intervals for: {}\n".format(postfix))
        print(ci_pls)

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
        fig, ax = plt.subplots()
        ax.set(xlabel='Pixel-wise Dice score', ylabel='Number of patients',
               title='Pixel-wise Dice confidence interval: [{:.2f}, {:.2f}]'.format(ci_pls[0, 0], ci_pls[1, 0]))
        ax.hist(best_dices_per_patient, bins=20)
        ax.axvline(x=np.mean(best_dices_per_patient), color='r', linewidth=2)
        ax.axvline(x=ci_pls[0, 0], color='g', linewidth=2)
        ax.axvline(x=ci_pls[1, 0], color='g', linewidth=2)
        ax.set_xlim(0., 1.)
        ax.grid(linestyle='--')
        # ax.legend()
        # plt.show()
        fig.savefig(os.path.join(folder, 'dice_confidence_intervals_all' + postfix + '.png'), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        plt.clf()

        best_dices_per_patient = np.ma.masked_array(best_dices_per_patient, [x < best_overlap for x in best_dices_per_patient])
        bs_pls = IIDBootstrap(best_dices_per_patient)
        ci_pls = bs_pls.conf_int(np.mean, 10000, method='bca')
        print(ci_pls)

        fig2, ax2 = plt.subplots()
        ax2.set(xlabel='Pixel-wise Dice score', ylabel='Number of patients',
                title='Pixel-wise Dice confidence interval: [{:.2f}, {:.2f}]'.format(ci_pls[0, 0], ci_pls[1, 0]))
        ax2.hist(best_dices_per_patient, bins=20)
        ax2.axvline(x=np.mean(best_dices_per_patient), color='r', linewidth=2)
        ax2.axvline(x=ci_pls[0, 0], color='g', linewidth=2)
        ax2.axvline(x=ci_pls[1, 0], color='g', linewidth=2)
        ax2.set_xlim(0., 1.)
        ax2.grid(linestyle='--')
        #ax.title()
        #ax2.legend()
        # plt.show()
        fig2.savefig(os.path.join(folder, 'dice_confidence_intervals_found' + postfix + '.png'),
                     dpi=300, bbox_inches="tight")
        plt.close(fig2)
        plt.clf()
    except Exception as e:
        print('{}'.format(traceback.format_exc()))
