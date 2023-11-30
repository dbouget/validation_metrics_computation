import logging
import traceback
import numpy as np
from typing import Tuple
from scipy.stats import pearsonr
from scipy.ndimage import _ni_support, label, find_objects
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


def compute_assd(result, reference, voxelspacing=None, connectivity=1):
    assd = np.mean((__surface_distances(result, reference, voxelspacing, connectivity),
                    __surface_distances(reference, result, voxelspacing, connectivity)))
    return assd


def compute_object_assd(result, reference, voxelspacing=None, connectivity=1):
    assd = np.mean((__obj_surface_distances(result, reference, voxelspacing, connectivity),
                    __obj_surface_distances(reference, result, voxelspacing, connectivity)))
    return assd


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


def __obj_surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel between all corresponding binary
    objects in result and reference. Correspondence is defined as unique and at least one voxel overlap.
    """
    sds = list()
    labelmap1, labelmap2, _a, _b, mapping = __distinct_binary_object_correspondences(result, reference, connectivity)
    slicers1 = find_objects(labelmap1)
    slicers2 = find_objects(labelmap2)
    for lid2, lid1 in list(mapping.items()):
        window = __combine_windows(slicers1[lid1 - 1], slicers2[lid2 - 1])
        object1 = labelmap1[window] == lid1
        object2 = labelmap2[window] == lid2
        sds.extend(__surface_distances(object1, object2, voxelspacing, connectivity))
    return sds


def __distinct_binary_object_correspondences(reference, result, connectivity=1):
    """
    Determines all distinct (where connectivity is defined by the connectivity parameter
    passed to scipy's `generate_binary_structure`) binary objects in both of the input
    parameters and returns a 1to1 mapping from the labelled objects in reference to the
    corresponding (whereas a one-voxel overlap suffices for correspondence) objects in
    result.

    All stems from the problem, that the relationship is non-surjective many-to-many.

    @return (labelmap1, labelmap2, n_lables1, n_labels2, labelmapping2to1)
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # label distinct binary objects
    labelmap1, n_obj_result = label(result, footprint)
    labelmap2, n_obj_reference = label(reference, footprint)

    # find all overlaps from labelmap2 to labelmap1; collect one-to-one relationships and store all one-two-many for later processing
    slicers = find_objects(labelmap2)  # get windows of labelled objects
    mapping = dict()  # mappings from labels in labelmap2 to corresponding object labels in labelmap1
    used_labels = set()  # set to collect all already used labels from labelmap2
    one_to_many = list()  # list to collect all one-to-many mappings
    for l1id, slicer in enumerate(slicers):  # iterate over object in labelmap2 and their windows
        l1id += 1  # labelled objects have ids sarting from 1
        bobj = (l1id) == labelmap2[slicer]  # find binary object corresponding to the label1 id in the segmentation
        l2ids = np.unique(labelmap1[slicer][
                                 bobj])  # extract all unique object identifiers at the corresponding positions in the reference (i.e. the mapping)
        l2ids = l2ids[0 != l2ids]  # remove background identifiers (=0)
        if 1 == len(
                l2ids):  # one-to-one mapping: if target label not already used, add to final list of object-to-object mappings and mark target label as used
            l2id = l2ids[0]
            if not l2id in used_labels:
                mapping[l1id] = l2id
                used_labels.add(l2id)
        elif 1 < len(l2ids):  # one-to-many mapping: store relationship for later processing
            one_to_many.append((l1id, set(l2ids)))

    # process one-to-many mappings, always choosing the one with the least labelmap2 correspondences first
    while True:
        one_to_many = [(l1id, l2ids - used_labels) for l1id, l2ids in
                       one_to_many]  # remove already used ids from all sets
        one_to_many = [x for x in one_to_many if x[1]]  # remove empty sets
        one_to_many = sorted(one_to_many, key=lambda x: len(x[1]))  # sort by set length
        if 0 == len(one_to_many):
            break
        l2id = one_to_many[0][1].pop()  # select an arbitrary target label id from the shortest set
        mapping[one_to_many[0][0]] = l2id  # add to one-to-one mappings
        used_labels.add(l2id)  # mark target label as used
        one_to_many = one_to_many[1:]  # delete the processed set from all sets

    return labelmap1, labelmap2, n_obj_result, n_obj_reference, mapping


def __combine_windows(w1, w2):
    """
    Joins two windows (defined by tuple of slices) such that their maximum
    combined extend is covered by the new returned window.
    """
    res = []
    for s1, s2 in zip(w1, w2):
        res.append(slice(min(s1.start, s2.start), max(s1.stop, s2.stop)))
    return tuple(res)
