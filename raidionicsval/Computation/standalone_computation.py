import logging
import os.path
import math
import numpy as np

from ..Computation.dice_computation import compute_dice, pixelwise_computation
from ..Validation.extra_metrics_computation import compute_specific_metric_value
from ..Validation.instance_segmentation_validation import *
from ..Utils.resources import SharedResources
from ..Utils.io_converters import open_image_file



class StandaloneComputation:
    """

    """
    def __init__(self):
        gt_filename = SharedResources.getInstance().standalone_gt_filename
        det_filename = SharedResources.getInstance().standalone_detection_filename

        if gt_filename is None or not os.path.exists(gt_filename):
            raise ValueError("Provided ground truth filename does not exist on disk.")
        if det_filename is None or not os.path.exists(det_filename):
            raise ValueError("Provided detection filename does not exist on disk.")
        self.gt_array, self.gt_ext, self.gt_input_specifics = open_image_file(gt_filename)
        self.det_array, self.det_ext, self.det_input_specifics = open_image_file(det_filename)
        if self.gt_array.shape != self.det_array.shape:
            raise ValueError("Provided ground truth and detection arrays do not have matching dimensions.")
        self.class_names = SharedResources.getInstance().standalone_class_names
        self.metrics_space = SharedResources.getInstance().standalone_metrics_spaces
        self.metric_names = SharedResources.getInstance().standalone_extra_metric_names
        self.detection_overlap_thresholds = SharedResources.getInstance().standalone_detection_overlap_thresholds
        if len(SharedResources.getInstance().standalone_detection_overlap_thresholds) == 1 and len(self.class_names) > 1:
            self.detection_overlap_thresholds = SharedResources.getInstance().standalone_detection_overlap_thresholds * len(self.class_names)
        self.tiny_objects_removal_threshold = SharedResources.getInstance().standalone_tiny_objects_removal_threshold
        self.positive_volume_thresholds = SharedResources.getInstance().standalone_true_positive_volume_thresholds
        if len(SharedResources.getInstance().standalone_true_positive_volume_thresholds) == 1 and len(self.class_names) > 1:
            self.positive_volume_thresholds = SharedResources.getInstance().standalone_true_positive_volume_thresholds * len(
                self.class_names)

        self.results = {}

    def run(self):
        for i, c in enumerate(self.class_names):
            class_results = {}
            gt_class_array = np.zeros(self.gt_array.shape, dtype="uint8")
            gt_class_array[self.gt_array == (i + 1)] = 1
            det_class_array = np.zeros(self.det_array.shape, dtype="uint8")
            det_class_array[self.det_array == (i + 1)] = 1

            if "patientwise" in self.metrics_space:
                pw_res = self.compute_patientwise(gt=gt_class_array, det=det_class_array,
                                         positive_thr=self.positive_volume_thresholds[i],
                                         detection_thr=self.detection_overlap_thresholds[i])
                class_results["PatientWise"] = pw_res
            if "pixelwise" in self.metrics_space:
                pw_res = self.compute_pixelwise(gt=gt_class_array, det=det_class_array)
                class_results["PixelWise"] = pw_res
            if "objectwise" in self.metrics_space:
                ow_res = self.compute_objectwise(gt=gt_class_array, det=det_class_array,
                                                 object_threshold=self.tiny_objects_removal_threshold)
                class_results["ObjectWise"] = ow_res
            self.results[c] = class_results

        for c in self.class_names:
            print(self.results[c])

    def compute_patientwise(self, gt: np.ndarray, det: np.ndarray, positive_thr: float, detection_thr: float) -> dict:
        res = {"TP": False, "TN": False, "FP": False, "FN": False}
        gt_volume = np.count_nonzero(gt) * np.prod(self.gt_input_specifics[1]) * 1e-3
        det_volume = np.count_nonzero(det) * np.prod(self.det_input_specifics[1]) * 1e-3

        overlap = compute_dice(gt, det)

        if gt_volume >= positive_thr and det_volume >= positive_thr and overlap >= detection_thr:
            res["TP"] = True
        elif gt_volume < positive_thr and det_volume < positive_thr:
            res["TN"] = True
        elif gt_volume >= positive_thr and (det_volume < positive_thr or overlap < detection_thr):
            res["FN"] = True
        elif gt_volume < positive_thr <= det_volume:
            res["FP"] = True
        return res

    def compute_pixelwise(self, gt: np.ndarray, det: np.ndarray) -> dict:
        pw_res = {"Dice": None, "Recall": None, "Precision": None, "F1-score": None}
        res = pixelwise_computation(gt=gt, detection=det)
        pw_res["Dice"] = res[0]
        pw_res["Recall"] = res[1]
        pw_res["Precision"] = res[2]
        pw_res["F1-score"] = res[3]

        if len(self.metric_names) != 0:
            tp_array = np.zeros(det.shape)
            fp_array = np.zeros(det.shape)
            tn_array = np.zeros(det.shape)
            fn_array = np.zeros(det.shape)

            tp_array[(gt == 1) & (det == 1)] = 1
            fp_array[(gt == 0) & (det == 1)] = 1
            tn_array[(gt == 0) & (det == 0)] = 1
            fn_array[(gt == 1) & (det == 0)] = 1
            for metric in self.metric_names:
                try:
                    metric_value = compute_specific_metric_value(metric=metric, gt=gt, detection=det,
                                                                 tp=np.sum(tp_array), tn=np.sum(tn_array),
                                                                 fp=np.sum(fp_array), fn=np.sum(fn_array),
                                                                 gt_spacing=self.gt_input_specifics[1],
                                                                 det_spacing=self.det_input_specifics[1])
                    pw_res[metric] = metric_value
                except Exception as e:
                    logging.warning(f"Computing {metric} resulted in an error with {e}")
                    continue

        return pw_res

    def compute_objectwise(self, gt: np.ndarray, det: np.ndarray, object_threshold: int) -> dict:
        res = {}

        obj_val = InstanceSegmentationValidation(gt_image=gt, detection_image=det,
                                                 tiny_objects_removal_threshold=object_threshold)

        obj_val.spacing = self.gt_input_specifics[1]
        obj_val.run()

        # Computing all metrics in an object-wise fashion. How to int
        instance_results = []
        for g, go in enumerate(obj_val.gt_candidates):
            gt_label = g + 1
            if gt_label in np.asarray(obj_val.matching_results)[:, 0]:
                indices = np.where(np.asarray(obj_val.matching_results)[:, 0] == gt_label)[0]
                if len(indices) > 1:
                    # Should not happen anymore
                    print(f"Warning - Entering a use-case which should not be possible!")
                    pass
                det_label = np.asarray(obj_val.matching_results)[indices[0]][1]
                instance_gt_array = np.zeros(gt.shape, dtype="uint8")
                instance_det_array = np.zeros(det.shape, dtype="uint8")
                instance_gt_array[obj_val.gt_labels == gt_label] = 1
                instance_det_array[obj_val.detection_labels == det_label] = 1
            else:
                instance_gt_array = np.zeros(gt.shape, dtype="uint8")
                instance_det_array = np.zeros(det.shape, dtype="uint8")
            instance_metrics = self.compute_pixelwise(gt=instance_gt_array, det=instance_det_array)
            instance_results.append(instance_metrics)

        res["OW Recall"] = obj_val.instance_detection_results[1]
        res["OW Precision"] = obj_val.instance_detection_results[2]
        res["OW F1-score"] = obj_val.instance_detection_results[3]
        if len(instance_results) != 0:
            for k in instance_results[0].keys():
                all_values = []
                tp_values = []
                for i in range(len(instance_results)):
                    if instance_results[i][k] != 0 and instance_results[i][k] != math.inf:
                        tp_values.append(instance_results[i][k])
                    all_values.append(instance_results[i][k])
                all_mean = np.mean(all_values)
                all_std = np.std(all_values)
                tp_mean = np.mean(tp_values)
                tp_std = np.std(tp_values)
                res[k] = {"All": {}, "TP": {}}
                res[k]["All"]["Mean"] = all_mean
                res[k]["All"]["Std"] = all_std
                res[k]["TP"]["Mean"] = tp_mean
                res[k]["TP"]["Std"] = tp_std
        return res