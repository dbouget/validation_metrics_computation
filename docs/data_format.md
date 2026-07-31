# Data format

This page describes the folder structures and naming conventions expected by the validation and metrics computation pipeline.

## Table of contents

- [Overview](#overview)
- [1. Original data folder structure](#1-original-data-folder-structure)
- [2. Inference results folder structure](#2-inference-results-folder-structure)
- [3. Cross-validation folds file](#3-cross-validation-folds-file)

---

## Overview

Two main structure types are supported: with or without an index-based naming convention. In the examples below, the indices (`index0`, `index1`, ...) represent the data's origin (e.g., different institutions or cohorts) — the folder names themselves are arbitrary strings and can be renamed freely.

Metrics and validation can be computed for **multiple segmentation classes simultaneously**, provided you supply unique, name-matching sets of ground truth and prediction files for each.

---

## 1. Original data folder structure

### Index-based naming convention

```
└── path/to/data/root/
    └── index0/
        ├── Pat001/
        │   ├── volumes/
        │   │   └── Pat001_MRI.nii.gz
        │   └── segmentations/
        │       ├── Pat001_MRI_label_tumor.nii.gz
        │       └── Pat001_MRI_label_other.nii.gz
        ├── Pat025/
        └── Pat050/
    └── index1/
        ├── Pat100/
        └── Pat150/
```

### Without an index-based naming convention

```
└── path/to/data/root/
    └── Pat001/
    │   ├── Pat001_MRI.nii.gz
    │   ├── Pat001_MRI_label_tumor.nii.gz
    │   └── Pat001_MRI_label_other.nii.gz
    └── Pat010/
    │   ├── Pat010_MRI.nii.gz
    │   ├── Pat010_MRI_label_tumor.nii.gz
    │   └── Pat010_MRI_label_other.nii.gz
    [...]
    └── Pat100/
        ├── Pat100_MRI.nii.gz
        ├── Pat100_MRI_label_tumor.nii.gz
        └── Pat100_MRI_label_other.nii.gz
```

---

## 2. Inference results folder structure

Prediction results must be stored inside a `predictions/` sub-folder. The outer-most sub-folders inside `predictions/` are the **fold numbers**. This whole structure lives inside what becomes your "validation study" folder.

### Index-based naming convention

```
└── path/to/validation/study/
    └── predictions/
        ├── 0/
        │   ├── index0_Pat001/
        │   │   ├── Pat001_MRI-pred_tumor.nii.gz
        │   │   └── Pat001_MRI-pred_other.nii.gz
        │   ├── index0_Pat002/
        │   │   ├── Pat002_MRI-pred_tumor.nii.gz
        │   │   └── Pat002_MRI-pred_other.nii.gz
        └── 1/
            ├── index1_Pat100/
            │   ├── Pat100_MRI-pred_tumor.nii.gz
            │   └── Pat100_MRI-pred_other.nii.gz
            └── index1_Pat150/
                ├── Pat150_MRI-pred_tumor.nii.gz
                └── Pat150_MRI-pred_other.nii.gz
```

### Without an index-based naming convention

```
└── path/to/validation/study/
    └── predictions/
        ├── 0/
        │   ├── Pat001/
        │   │   ├── Pat001_MRI-pred_tumor.nii.gz
        │   │   └── Pat001_MRI-pred_other.nii.gz
        │   ├── Pat002/
        │   │   ├── Pat002_MRI-pred_tumor.nii.gz
        │   │   └── Pat002_MRI-pred_other.nii.gz
        └── 1/
            ├── Pat100/
            │   ├── Pat100_MRI-pred_tumor.nii.gz
            │   └── Pat100_MRI-pred_other.nii.gz
            └── Pat150/
                ├── Pat150_MRI-pred_tumor.nii.gz
                └── Pat150_MRI-pred_other.nii.gz
```

---

## 3. Cross-validation folds file

The file listing patient distribution across folds must be named **`cross_validation_folds.txt`** and placed in the validation study folder, alongside the `predictions/` sub-folder.

It should list the contents of the validation and test sets iteratively, one pair per line.

### Index-based naming convention

```
index0_Pat1000_MRI_sample index1_Pat1250_MRI_sample
index0_Pat001_MRI_sample index1_Pat025_MRI_sample
index0_Pat001_MRI_sample index1_Pat025_MRI_sample
index0_Pat100_MRI_sample index1_Pat150_MRI_sample
```

### Without an index-based naming convention

```
Pat001_MRI Pat002_MRI
Pat100_MRI Pat150_MRI
Pat100_MRI Pat150_MRI
Pat200_MRI Pat250_MRI
```
