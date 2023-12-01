import logging
import traceback
import numpy as np
from typing import Tuple
from scipy.stats import pearsonr
from scipy.ndimage import _ni_support, label, find_objects, distance_transform_edt
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

"""
The MedPy library has not been updated since 2019 and no recent package has been produced as of yet
(expecting release end of 2023).
Sampling the code for some of the metrics to continue supporting here (https://github.com/loli/medpy)
"""


def compute_volume_correlation(results, references):
    results = np.atleast_2d(np.array(results).astype(np.bool_))
    references = np.atleast_2d(np.array(references).astype(np.bool_))

    results_volumes = [np.count_nonzero(r) for r in results]
    references_volumes = [np.count_nonzero(r) for r in references]

    return pearsonr(results_volumes, references_volumes)


def compute_ravd(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return (vol1 - vol2) / float(vol2)


def compute_hd95(reference, result, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def compute_assd(volume1, volume2, voxel_spacing=(1, 1, 1)):
    """
    Compute the average symmetric surface distance between two 3D volumes.

    Parameters:
    - volume1: First 3D volume (binary mask).
    - volume2: Second 3D volume (binary mask).
    - voxel_spacing: Tuple representing the spacing between voxels in each dimension (default is (1, 1, 1)).

    Returns:
    - Average symmetric surface distance.
    """
    # Ensure volumes are binary masks
    volume1 = np.asarray(volume1).astype(bool)
    volume2 = np.asarray(volume2).astype(bool)

    # Compute the distance transform for both volumes
    dist_transform1 = distance_transform_edt(volume1, sampling=voxel_spacing)
    dist_transform2 = distance_transform_edt(volume2, sampling=voxel_spacing)

    # Compute the surface distances
    surface_distances_1_to_2 = dist_transform1[volume2]
    surface_distances_2_to_1 = dist_transform2[volume1]

    # Combine distances from both volumes
    all_surface_distances = np.concatenate([surface_distances_1_to_2, surface_distances_2_to_1])

    # Compute the average symmetric surface distance
    average_symmetric_surface_distance = np.mean(all_surface_distances)

    return average_symmetric_surface_distance


def compute_object_assd(volume1, volume2, voxel_spacing=(1, 1, 1)):
    """
    Compute Object-wise Average Symmetric Surface Distance (oASD) between two 3D volumes.

    Parameters:
    - volume1: Binary mask of the first volume (numpy array).
    - volume2: Binary mask of the second volume (numpy array).
    - voxel_spacing: Tuple representing the voxel spacing in each dimension (e.g., (1.0, 1.0, 1.0)).

    Returns:
    - oASD: Object-wise Average Symmetric Surface Distance.
    """

    def surface_distances(mask1, mask2, voxel_spacing):
        """
        Compute surface distances between two binary masks.
        """
        distances = distance_transform_edt(mask1, sampling=voxel_spacing) * mask2
        surface_distances = distances[mask2 == 1]
        return surface_distances

    def compute_ASD(mask1, mask2, voxel_spacing):
        """
        Compute Average Symmetric Surface Distance (ASD) between two binary masks.
        """
        distances1 = surface_distances(mask1, mask2, voxel_spacing)
        distances2 = surface_distances(mask2, mask1, voxel_spacing)
        distances = np.concatenate([distances1, distances2])
        ASD = np.mean(distances)
        return ASD

    # Ensure input volumes have the same shape
    if volume1.shape != volume2.shape:
        raise ValueError("Input volumes must have the same shape.")

    # Convert volumes to binary masks
    mask1 = np.asarray(volume1, dtype=bool)
    mask2 = np.asarray(volume2, dtype=bool)

    # Compute ASD for each object in the volumes
    unique_objects = np.unique(np.concatenate([mask1, mask2]))
    oASD_values = []

    for obj_label in unique_objects:
        if obj_label == 0:  # Skip background
            continue

        obj1 = (mask1 == obj_label).astype(int)
        obj2 = (mask2 == obj_label).astype(int)

        ASD_obj = compute_ASD(obj1, obj2, voxel_spacing)
        oASD_values.append(ASD_obj)

    # Compute oASD as the average of individual ASD values
    oASD = np.mean(oASD_values)

    return oASD


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds
