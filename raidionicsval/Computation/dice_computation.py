import numpy as np
import traceback
from copy import deepcopy

from raidionicsval.Utils.resources import SharedResources
from raidionicsval.Validation.instance_segmentation_validation import InstanceSegmentationValidation


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
    detection_raw = args[3]
    patient_id = args[4]
    volumes_extra = args[5]
    results = []

    detection = np.zeros(detection_raw.shape, dtype='uint8')
    detection[detection_raw >= t] = 1

    # # Cleaning the too small objects that might be noise in the detection
    # if np.count_nonzero(detection) > 0:
    #     detection_labels = measurements.label(detection)[0]
    #     # print('Found {} objects.'.format(np.max(self.detection_labels)))
    #     refined_image = deepcopy(detection)
    #     for c in range(1, np.max(detection_labels) + 1):
    #         if np.count_nonzero(detection_labels == c) < SharedResources.getInstance().validation_tiny_objects_removal_threshold:
    #             refined_image[refined_image == c] = 0
    #     refined_image[refined_image != 0] = 1
    #     detection = refined_image

    pixelwise_results = [-1., -1., -1., -1.]
    if "pixelwise" in SharedResources.getInstance().validation_metric_spaces:
        pixelwise_results = __pixelwise_computation(gt, detection)

    det_volume = np.round(np.count_nonzero(detection) * np.prod(volumes_extra[2]) * 1e-3, 4)

    obj_val = InstanceSegmentationValidation(gt_image=gt, detection_image=detection)
    if "objectwise" in SharedResources.getInstance().validation_metric_spaces:
        try:
            # obj_val.set_trace_parameters(self.output_folder, fold_number, patient, t)
            obj_val.spacing = volumes_extra[2]
            obj_val.run()
        except Exception as e:
            print('Issue computing instance segmentation parameters for patient {}'.format(patient_id))
            print(traceback.format_exc())
    instance_results = obj_val.instance_detection_results

    results.append([fold_number, patient_id, t] + pixelwise_results + volumes_extra[:-1] +
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
