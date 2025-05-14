import numpy as np
import traceback
from copy import deepcopy


def compute_dice(volume1, volume2):
    dice = 0.
    if np.sum(volume1[volume2 == 1]) != 0:
        dice = (np.sum(volume1[volume2 == 1]) * 2.0) / (np.sum(volume1) + np.sum(volume2))
    return dice


def compute_dice_uncertain(volume1, volume2, epsilon=0.1):
    dice = (np.sum(volume1[volume2 == 1]) * 2.0 + epsilon) / (np.sum(volume1) + np.sum(volume2) + epsilon)
    return dice


def pixelwise_computation(gt, detection):
    dice = compute_dice(gt, detection)

    tp_array = np.zeros(detection.shape)
    fp_array = np.zeros(detection.shape)
    tn_array = np.zeros(detection.shape)
    fn_array = np.zeros(detection.shape)

    tp_array[(gt == 1) & (detection == 1)] = 1
    fp_array[(gt == 0) & (detection == 1)] = 1
    tn_array[(gt == 0) & (detection == 0)] = 1
    fn_array[(gt == 1) & (detection == 0)] = 1

    recall = np.sum(tp_array) / (np.sum(tp_array) + np.sum(fn_array) + 1e-6)
    precision = np.sum(tp_array) / (np.sum(tp_array) + np.sum(fp_array) + 1e-6)
    f1 = 2 * np.sum(tp_array) / ((2 * np.sum(tp_array)) + np.sum(fp_array) + np.sum(fn_array) + 1e-6)

    return [dice, recall, precision, f1]
