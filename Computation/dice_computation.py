import numpy as np
import traceback
from copy import deepcopy

from Utils.resources import SharedResources
from Validation.instance_segmentation_validation import InstanceSegmentationValidation


def compute_dice(volume1, volume2):
    dice = 0.
    if np.sum(volume1[volume2 == 1]) != 0:
        dice = (np.sum(volume1[volume2 == 1]) * 2.0) / (np.sum(volume1) + np.sum(volume2))
    return dice


def compute_dice_uncertain(volume1, volume2, epsilon=0.1):
    dice = (np.sum(volume1[volume2 == 1]) * 2.0 + epsilon) / (np.sum(volume1) + np.sum(volume2) + epsilon)
    return dice


def separate_dice_computation(args):
    """
    Dice computation method linked to the multiprocessing strategy. Effectively where the call to compute is made.
    :param args: list of arguments split from the lists given to the multiprocessing.Pool call.
    :return: list with the computed results for the current patient, at the given probability threshold.
    """
    t = np.round(args[0], 2)
    fold_number = args[1]
    gt = args[2]
    detection_ni = args[3]
    patient_id = args[4]
    volumes_extra = args[5]
    results = []

    detection = deepcopy(detection_ni.get_data())
    detection[detection < t] = 0
    detection[detection >= t] = 1
    detection = detection.astype('uint8')

    pixelwise_results = [-1., -1., -1., -1.]
    if "pixelwise" in SharedResources.getInstance().validation_metric_spaces:
        pixelwise_results = __pixelwise_computation(gt, detection)

    patientwise_results = [-1., -1., -1., -1.]
    det_volume = np.round(np.count_nonzero(detection) * np.prod(detection_ni.header.get_zooms()) * 1e-3, 4)
    # if "patientwise" in SharedResources.getInstance().validation_metric_spaces:
    #     patientwise_results = __patientwise_computation(gt, detection)

    obj_val = InstanceSegmentationValidation(gt_image=gt, detection_image=detection)
    if "objectwise" in SharedResources.getInstance().validation_metric_spaces:
        try:
            # obj_val.set_trace_parameters(self.output_folder, fold_number, patient, t)
            obj_val.spacing = detection_ni.header.get_zooms()
            obj_val.run()
        except Exception as e:
            print('Issue computing instance segmentation parameters for patient {}'.format(patient_id))
            print(traceback.format_exc())
    instance_results = obj_val.instance_detection_results

    results.append([fold_number, patient_id, t] + pixelwise_results + patientwise_results + volumes_extra +
                   [det_volume] + instance_results + [len(obj_val.gt_candidates), len(obj_val.detection_candidates)])

    return results


def __pixelwise_computation(gt, detection):
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
