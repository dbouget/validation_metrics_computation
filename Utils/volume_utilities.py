import numpy as np


def compute_tumor_volume(labels_array, voxel_size):
    """
    Compute volume of segmented tumor
    :param label_image: image containing the labels
    :retrn: volume_ml: volume in ml
    """
    volume_pixels = np.count_nonzero(labels_array != 0)  # Might be more than one label, but not considering it yet
    volume_mmcube = voxel_size * volume_pixels
    volume_ml = volume_mmcube * 1e-3

    return volume_ml