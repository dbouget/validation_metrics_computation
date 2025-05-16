from __future__ import division
import traceback
import numpy as np
import os
import h5py
import nibabel as nib
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from ..Utils.resources import SharedResources
from ..Computation.dice_computation import pixelwise_computation


def box_overlap(box1, box2):
    overlap_perc = 0
    dx = min(box1[0].stop, box2[0].stop) - max(box1[0].start, box2[0].start)
    dy = min(box1[1].stop, box2[1].stop) - max(box1[1].start, box2[1].start)
    dz = min(box1[2].stop, box2[2].stop) - max(box1[2].start, box2[2].start)
    if (dx >= 0) and (dy >= 0) and (dz >= 0):
        overlap = dx * dy * dz
        tot_area = ((box1[0].stop - box1[0].start) * (box1[1].stop - box1[1].start) * (box1[2].stop - box1[2].start)) +\
                   ((box2[0].stop - box2[0].start) * (box2[1].stop - box2[1].start) * (box2[2].stop - box2[2].start))
        overlap_perc = (2 * overlap) / tot_area

    return overlap_perc


def box_overlap_leniant(box1, box2):
    overlap_perc = 0
    dx = min(box1[0].stop, box2[0].stop) - max(box1[0].start, box2[0].start)
    dy = min(box1[1].stop, box2[1].stop) - max(box1[1].start, box2[1].start)
    dz = min(box1[2].stop, box2[2].stop) - max(box1[2].start, box2[2].start)
    if (dx >= 0) and (dy >= 0) and (dz >= 0):
        overlap = dx * dy * dz
        area_box1 = (box1[0].stop - box1[0].start) * (box1[1].stop - box1[1].start) * (box1[2].stop - box1[2].start)
        area_box2 = (box2[0].stop - box2[0].start) * (box2[1].stop - box2[1].start) * (box2[2].stop - box2[2].start)
        tot_area = area_box1 + area_box2
        overlap_perc = overlap / min(area_box1, area_box2)

    return overlap_perc


class InstanceSegmentationValidation:
    """
    Perform the instance detection validation side (i.e., recall, precision).
    N.B.: most likely not fully optimal when compared to CVPR/PAMI processes.
    """
    def __init__(self, gt_image, detection_image, tiny_objects_removal_threshold: int = 50):
        self.gt_image = gt_image
        self.detection_image = detection_image

        self.dump_trace = False
        self.spacing = None
        self.gt_labels = None
        self.detection_labels = None
        self.gt_candidates = []
        self.detection_candidates = []
        self.matching_results = []
        self.instance_detection_results = [0., 1., 0.5, 0., 0., 0., 0., 1., 0., 0.5, 0.]  # Global Recall, Global Precision, Global F1, Dice (avg/std), Recall (avg/std), Precision (avg/std), F1 (avg/std)
        self.tiny_objects_removal_threshold = tiny_objects_removal_threshold

    def set_trace_parameters(self, output_folder, fold_number, patient, threshold):
        """
        For debugging purposes, should dump some more files for verification.
        :param output_folder:
        :param fold_number:
        :param patient:
        :param threshold:
        :return:
        """
        self.output_folder = output_folder
        self.fold_number = fold_number
        self.patient = patient
        self.threshold = threshold

    def run(self):
        self.__select_candidates()
        if len(self.detection_candidates) != 0 and len(self.gt_candidates) != 0:
            self.__pair_candidates()
            self.__compute_metrics()
        elif len(self.gt_candidates) == 0 and len(self.detection_candidates) == 0:
            self.instance_detection_results[0] = 1.
            self.instance_detection_results[1] = 1.
            self.instance_detection_results[2] = 1.
            self.instance_detection_results[5] = 1.
            self.instance_detection_results[7] = 1.
            self.instance_detection_results[9] = 1.
        elif len(self.gt_candidates) != 0 and len(self.detection_candidates) == 0:
            self.instance_detection_results[0] = 0.
            self.instance_detection_results[1] = 1.
            self.instance_detection_results[2] = 0.5
            self.instance_detection_results[5] = 0.
            self.instance_detection_results[7] = 1.
            self.instance_detection_results[9] = 0.5
        elif len(self.gt_candidates) == 0 and len(self.detection_candidates) != 0:
            self.instance_detection_results[0] = 1.
            self.instance_detection_results[1] = 0.
            self.instance_detection_results[2] = 0.5
            self.instance_detection_results[5] = 1.
            self.instance_detection_results[7] = 0.
            self.instance_detection_results[9] = 0.5

        return 1

    def run_study(self):
        self.__select_candidates()
        self.__pair_candidates(study_state=True)

        return self.matching_results

    def __select_candidates(self):
        """
        Perform a connected components analysis to identify the stand-alone objects in both the ground truth and
        binarized prediction volumes. Objects with a number of voxels below the limit set in self.tiny_objects_removal_threshold
        are discarded, in both instances. Safe way to handle potential noise in the ground truth, especially if a
        third-party software (e.g. 3DSlicer) was used.
        """
        from scipy.ndimage import measurements
        from skimage.measure import regionprops

        # Cleaning the too small objects that might be noise in the ground truth
        self.gt_labels = measurements.label(self.gt_image)[0]
        refined_image = deepcopy(self.gt_labels)
        for c in range(1, np.max(self.gt_labels)+1):
            if np.count_nonzero(self.gt_labels == c) < self.tiny_objects_removal_threshold:
                refined_image[refined_image == c] = 0
        refined_image[refined_image != 0] = 1
        self.gt_labels = measurements.label(refined_image)[0]
        self.gt_candidates = regionprops(self.gt_labels)

        # Cleaning the too small objects that might be noise in the detection
        if np.count_nonzero(self.detection_image) > 0:
            self.detection_labels = measurements.label(self.detection_image)[0]
            # print('Found {} objects.'.format(np.max(self.detection_labels)))
            refined_image = deepcopy(self.detection_labels)
            for c in range(1, np.max(self.detection_labels) + 1):
                if np.count_nonzero(self.detection_labels == c) < self.tiny_objects_removal_threshold:
                    refined_image[refined_image == c] = 0
            refined_image[refined_image != 0] = 1
            self.detection_labels = measurements.label(refined_image)[0]
            # print('Found {} objects after cleaning up.'.format(np.max(self.detection_labels)))
            self.detection_candidates = regionprops(self.detection_labels)

        if self.dump_trace:
            dump_gt_filename = os.path.join(self.output_folder, str(self.fold_number), '1_' +
                                            self.patient.split('_')[0], 'gt_labels.nii.gz')
            nib.save(nib.Nifti1Image(self.gt_labels,
                                     affine=[[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0]]),
                     dump_gt_filename)

            dump_detection_filename = os.path.join(self.output_folder, str(self.fold_number),
                                                   '1_' + self.patient.split('_')[0],
                                                   'detection_labels_' + str(int(self.threshold*100)) + '.nii.gz')
            nib.save(nib.Nifti1Image(self.detection_labels,
                                     affine=[[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0]]),
                     dump_detection_filename)

    def __pair_candidates(self, study_state=False):
        """
        Identify matching objects between the ground truth and detection candidates generated in self.__select_candidates.

        If multiple detection candidates overlap with the same ground truth candidate, only the one with the highest
        Dice is kept as the correct pair and the second detection candidate should be considered as a false positive.
        Using the Hungarian algorithm for it, using the Dice overlap as criteria
        """
        dice_matrix = np.zeros(shape=(len(self.gt_candidates), len(self.detection_candidates)))
        for g, go in enumerate(self.gt_candidates):
            gt_object = go.slice
            gt_label_value = g + 1
            for d, do in enumerate(self.detection_candidates):
                det_label_value = d + 1
                det_object = do.slice
                is_overlap = box_overlap(gt_object, det_object)
                if is_overlap > 0:
                    roi = [min(gt_object[0].start, det_object[0].start), max(gt_object[0].stop, det_object[0].stop),
                           min(gt_object[1].start, det_object[1].start), max(gt_object[1].stop, det_object[1].stop),
                           min(gt_object[2].start, det_object[2].start), max(gt_object[2].stop, det_object[2].stop)]

                    sub_gt_object = deepcopy(self.gt_labels[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]])
                    sub_det_object = deepcopy(self.detection_labels[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]])

                    sub_gt_object[sub_gt_object != gt_label_value] = 0
                    sub_gt_object[sub_gt_object == gt_label_value] = 1

                    sub_det_object[sub_det_object != det_label_value] = 0
                    sub_det_object[sub_det_object == det_label_value] = 1

                    dice_overlap = np.sum(sub_det_object[sub_gt_object == 1]) * 2.0 / (np.sum(sub_gt_object) + np.sum(sub_det_object))
                    # if dice_overlap > 0.0:
                    #     if study_state:
                    #         self.matching_results.append([gt_label_value, det_label_value, dice_overlap, sub_gt_object, sub_det_object])
                    #     else:
                    #         self.matching_results.append([gt_label_value, det_label_value, dice_overlap])
                    dice_matrix[g, d] = dice_overlap
                else:
                    dice_matrix[g, d] = 0.
        cost_matrix = -dice_matrix
        gt_inds, pred_inds = linear_sum_assignment(cost_matrix)
        matches = [(i, j) for i, j in zip(gt_inds, pred_inds) if dice_matrix[i, j] > 0.1]
        for m in matches:
            self.matching_results.append([m[0] + 1, m[1] + 1, dice_matrix[m[0], m[1]]])
        if self.dump_trace:
            output_file = os.path.join(self.output_folder, str(self.fold_number), '1_' + self.patient.split('_')[0],
                                       'matching_' + str(int(self.threshold * 100)) + '.hd5')
            f = h5py.File(output_file, "w")
            data_group = f.create_group('matching')
            dset = data_group.create_dataset('0', np.asarray(self.matching_results).shape, dtype=np.float32,
                                             compression="gzip")
            dset[:] = np.asarray(self.matching_results).astype('float32')
            f.close()

    def __compute_metrics(self):
        """
        :return:
        """
        recall = 0.0
        precision = 1.0
        f1_score = 0.0
        averaged_objectwise_results = [0] * 8
        try:
            if len(self.matching_results) != 0:
                array_matching = np.asarray(self.matching_results)
                recall = len(np.unique(array_matching[:, 0])) / len(self.gt_candidates)
                precision = len(np.unique(array_matching[:, 1])) / len(self.detection_candidates)
                f1_score = 2. * ((precision * recall) / (precision + recall))
                if len(self.gt_candidates) == 0 and len(self.detection_candidates) > 0:
                    precision = 0

                instance_results = []
                for g, go in enumerate(self.gt_candidates):
                    gt_label = g + 1
                    if gt_label in array_matching[:, 0]:
                        indices = np.where(array_matching[:, 0] == gt_label)[0]
                        if len(indices) > 1:
                            # Should not happen anymore
                            print(f"Warning - Entering a use-case which should not be possible!")
                            pass
                        else:
                            det_label = self.matching_results[indices[0]][1]
                        instance_gt_array = np.zeros(self.gt_image.shape, dtype="uint8")
                        instance_det_array = np.zeros(self.detection_image.shape, dtype="uint8")
                        instance_gt_array[self.gt_labels == gt_label] = 1
                        instance_det_array[self.detection_labels == det_label] = 1
                        inst_metric = pixelwise_computation(instance_gt_array, instance_det_array)
                        instance_results.append(inst_metric)
                averaged_objectwise_results = [x for pair in zip(list(np.mean(np.asarray(instance_results), axis=0)), list(np.std(np.asarray(instance_results), axis=0))) for x in pair]
            global_objectwise_results = [recall, precision, f1_score]
            self.instance_detection_results = global_objectwise_results + averaged_objectwise_results
        except Exception as e:
            print(f'Issue computing the objectwise metrics with {e}\n {traceback.format_exc()}')

    def __compute_metrics_old(self):
        """
        @TODO. Might need to improve the object-wise metrics computation, here again with TP/TN/FP/FN
        :return:
        """
        average_dice = 0.0
        largest_component_dice = 0.0
        recall = 0.0
        precision = 1.0
        f1_score = 0.0
        try:
            if len(self.matching_results) != 0:
                array_matching = np.asarray(self.matching_results)
                average_dice = np.mean(array_matching, axis=0)[2]
                recall = len(np.unique(array_matching[:, 0])) / len(self.gt_candidates)
                precision = len(np.unique(array_matching[:, 1])) / len(self.detection_candidates)
                f1_score = 2. * ((precision * recall) / (precision + recall))
                index_larger_component = [x.area for x in self.gt_candidates].index(np.max([x.area for x in self.gt_candidates]))
                matching_larger_component = [x[0] for x in self.matching_results].index(index_larger_component + 1) if (index_larger_component + 1) in [x[0] for x in self.matching_results] else -1
                if matching_larger_component != -1:
                    largest_component_dice = self.matching_results[matching_larger_component][2]
                if len(self.gt_candidates) == 0 and len(self.detection_candidates) > 0:
                    precision = 0
        except Exception as e:
            print(f'Issue computing the objectwise metrics with {e}\n {traceback.format_exc()}')
        self.instance_detection_results = [average_dice, recall, precision, f1_score, largest_component_dice]
