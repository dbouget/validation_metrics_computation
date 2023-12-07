# Segmentation validation and metrics computation backend for Raidionics related publications

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11|3.12-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/dbouget/7560fe410db03e384a45ddc77bbe9a57/compute_validation_example.ipynb)

The code corresponds to the Raidionics backend for running the k-fold cross-validation and metrics computation.
The module can either be used as a Python library, as CLI, or as Docker container.

## [Installation](https://github.com/dbouget/validation_metrics_computation#installation)

```
pip install git+https://github.com/dbouget/validation_metrics_computation.git
```

## [Continuous integration](https://github.com/dbouget/validation_metrics_computation#continuous-integration)

<div style="display: flex;">
  <div style="flex: 1; margin-right: 20px;">

| Operating System | Status                                                                                                             |
|------------------|--------------------------------------------------------------------------------------------------------------------|
| **Windows**      | [![Build macOS](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_windows.yml/badge.svg)](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_windows.yml) |
| **Ubuntu**       | [![Build macOS](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_ubuntu.yml/badge.svg)](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_ubuntu.yml) |
| **macOS**        | [![Build macOS](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_macos.yml/badge.svg)](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_macos.yml) |
| **macOS ARM**    | [![Build macOS](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_macos_arm.yml/badge.svg)](https://github.com/dbouget/validation_metrics_computation/actions/workflows/build_macos_arm.yml) |
  </div>
</div>

## [Getting started](https://github.com/dbouget/validation_metrics_computation#getting-started)

### [Notebooks](https://github.com/dbouget/validation_metrics_computation#notebooks)

Below are two Jupyter Notebooks which include simple examples on how to get started.

<div style="display: flex;">
  <div style="flex: 1; margin-right: 20px;">

| Notebook       | Colab                                                | GitHub                                                                                                                   |
|----------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Validation** | <a href="https://colab.research.google.com/gist/dbouget/7560fe410db03e384a45ddc77bbe9a57/compute_validation_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [![View on GitHub](https://img.shields.io/badge/View%20on%20GitHub-blue?logo=github)](https://github.com/dbouget/validation_metrics_computation/blob/master/notebooks/compute_validation_example.ipynb) |
| **Study**      | <a href="https://colab.research.google.com/gist/dbouget/8a0e093284688e993244930bd36fd367/compute_study_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>   | [![View on GitHub](https://img.shields.io/badge/View%20on%20GitHub-blue?logo=github)](https://github.com/dbouget/validation_metrics_computation/blob/master/notebooks/compute_study_example.ipynb)      |

  </div>
</div>

### [Usage](https://github.com/dbouget/validation_metrics_computation#usage)

In the following, a description of how the data should be organized on disk is provided, and a test dataset can
be downloaded [here](https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsValLib_UnitTest1.zip).

<details>
<summary>

### [1. Folder and data structures and naming conventions](https://github.com/dbouget/validation_metrics_computation#1-folder-and-data-structures-and-naming-conventions)
</summary>

Two main structure types are supported, without or without following an index-based naming convention.
Assuming in the following example that the data indexes are based on their origin, but
anything should work. The folders named _index0_ and _index1_ could be renamed to any sets of strings.

The metrics and overall validation can be computed for multiple segmentation classes at the same time, granted that
unique and name-matching sets of files (i.e., ground truth and prediction files) are provided.

#### [1.1 Original data folder structure](https://github.com/dbouget/validation_metrics_computation#11-original-data-folder-structure)
The main data directory containing the original 3D volumes and corresponding manual annotations is expected
to resemble the following structure using an index-based naming convention:

    └── path/to/data/root/
        └── index0/
            ├── Pat001/
            │   ├── volumes/
            │   │   └── Pat001_MRI.nii.gz
            │   └── segmentations/
            │   │   ├── Pat001_MRI_label_tumor.nii.gz
            │   │   └── Pat001_MRI_label_other.nii.gz
            ├── Pat025/
            └── Pat050/
        └── index1/
            ├── Pat100/
            └── Pat150/

The main data directory containing the original 3D volumes and corresponding manual annotations is expected
to resemble the following structure when **not** using an index-based naming convention:

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
        │   ├── Pat100_MRI.nii.gz
        │   ├── Pat100_MRI_label_tumor.nii.gz
        │   └── Pat100_MRI_label_other.nii.gz

#### [1.2 Inference results folder structure](https://github.com/dbouget/validation_metrics_computation#12-inference-results-folder-structure)
Predictions results are expected to be stored inside a _predictions/_ sub-folder, the outer-most sub-folder 
naming convention inside the folder are the fold numbers.
The inference results should be grouped inside what will become the validation folder, resembling the following
structure when using an index-based naming convention.

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
            │   ├── index1_Pat100/
            │   │   ├── Pat100_MRI-pred_tumor.nii.gz
            │   │   └── Pat100_MRI-pred_other.nii.gz  
            │   └── index1_Pat150/ 
            │   │   ├── Pat150_MRI-pred_tumor.nii.gz
            │   │   └── Pat150_MRI-pred_other.nii.gz  

The inference results should be grouped inside what will become the validation folder, resembling the following
structure when **not** using an index-based naming convention.

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
            │   ├── Pat100/
            │   │   ├── Pat100_MRI-pred_tumor.nii.gz
            │   │   └── Pat100_MRI-pred_other.nii.gz  
            │   └── Pat150/ 
            │   │   ├── Pat150_MRI-pred_tumor.nii.gz
            │   │   └── Pat150_MRI-pred_other.nii.gz  

#### [1.3 Folds file](https://github.com/dbouget/validation_metrics_computation#13-folds-file)
The file with patients' distribution within each fold used for training should list
the content of the validation and test sets iteratively.  
The file should be called __cross\_validation\_folds.txt__ and placed in the validation
study folder side-by-side with the _predictions_ sub-folder.  

An example of its content is given below when using an index-based naming convention:
```
  index0_Pat1000_MRI_sample index1_Pat1250_MRI_sample\n    
  index0_Pat001_MRI_sample index1_Pat025_MRI_sample\n  
  index0_Pat001_MRI_sample index1_Pat025_MRI_sample\n    
  index0_Pat100_MRI_sample index1_Pat150_MRI_sample\n  
```

An example of its content is given below when **not** using an index-based naming convention:
```
  Pat001_MRI Pat002_MRI\n    
  Pat100_MRI Pat150_MRI\n  
  Pat100_MRI Pat150_MRI\n    
  Pat200_MRI Pat250_MRI\n  
```

</details>

<details>
<summary>

### [2. Installation](https://github.com/dbouget/validation_metrics_computation#2-installation)
</summary>
Create a virtual environment using at least Python 3.8, and install all dependencies from
the requirements.txt file.

```
  cd /path/to/validation_metrics_computation  
  virtualenv -p python3 venv  
  source venv/bin/activate  
  TMPDIR=$PWD/venv pip install --cache-dir=$PWD/venv -r requirements.txt (--no-deps)
```

Then the final step is to do the following in a terminal.
```
  cd /path/to/validation_metrics_computation  
  cp blank_main_config.ini main_config.ini 
```

You can now edit your __main\_config.ini__ file for running the different processes.  
An additional explanation of all parameters specified in the configuration file can be
found in _/Utils/resources.py_. 

</details>

<details>
<summary>

### [3. Process](https://github.com/dbouget/validation_metrics_computation#3-process)
</summary>
To run, you need to supply the configuration file as parameter.

```
  python main.py -c main_config.ini (-v debug)
```

After filling in the configuration file, you should run first the 
__validation__ task and then the __study__ task.  
N.B. If no study fits your need, you can create a new study file in _/Studies/_.

</details>

## [How to cite](https://github.com/dbouget/validation_metrics_computation#how-to-cite)

If you are using Raidionics in your research, please cite the following references.

For segmentation validation and metrics computation:
```
@article{bouget2022preoptumorseg,
    title={Preoperative Brain Tumor Imaging: Models and Software for Segmentation and Standardized Reporting},
    author={Bouget, David and Pedersen, André and Jakola, Asgeir S. and Kavouridis, Vasileios and Emblem, Kyrre E. and Eijgelaar, Roelant S. and Kommers, Ivar and Ardon, Hilko and Barkhof, Frederik and Bello, Lorenzo and Berger, Mitchel S. and Conti Nibali, Marco and Furtner, Julia and Hervey-Jumper, Shawn and Idema, Albert J. S. and Kiesel, Barbara and Kloet, Alfred and Mandonnet, Emmanuel and Müller, Domenique M. J. and Robe, Pierre A. and Rossi, Marco and Sciortino, Tommaso and Van den Brink, Wimar A. and Wagemakers, Michiel and Widhalm, Georg and Witte, Marnix G. and Zwinderman, Aeilko H. and De Witt Hamer, Philip C. and Solheim, Ole and Reinertsen, Ingerid},
    journal={Frontiers in Neurology},
    volume={13},
    year={2022},
    url={https://www.frontiersin.org/articles/10.3389/fneur.2022.932219},
    doi={10.3389/fneur.2022.932219},
    issn={1664-2295}
}
```

The final software including updated performance metrics for preoperative tumors and introducing postoperative tumor segmentation:
```
@article{bouget2023raidionics,
    author = {Bouget, David and Alsinan, Demah and Gaitan, Valeria and Holden Helland, Ragnhild and Pedersen, André and Solheim, Ole and Reinertsen, Ingerid},
    year = {2023},
    month = {09},
    pages = {},
    title = {Raidionics: an open software for pre-and postoperative central nervous system tumor segmentation and standardized reporting},
    volume = {13},
    journal = {Scientific Reports},
    doi = {10.1038/s41598-023-42048-7},
}
```
