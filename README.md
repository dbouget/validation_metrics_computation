# Segmentation validation and metrics computation backend for Raidionics related publications

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11|3.12-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)
[![codecov](https://codecov.io/gh/dbouget/validation_metrics_computation/branch/master/graph/badge.svg?token=ZSPQVR7RKX)](https://codecov.io/gh/dbouget/validation_metrics_computation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/dbouget/491b0d34e3df00e730cd7fe7a8989202/compute_validation_example.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/raidionicsval.svg)](https://pypi.org/project/raidionicsval/)

The code corresponds to the Raidionics backend for running the k-fold cross-validation and metrics computation.
The module can either be used as a Python library, as CLI, or as Docker container. It supports both 2D and 3D inputs,
the only hard requirement is the expected folder structure to use as input.  
:warning: For using custom structures, modifying the code [here](https://github.com/dbouget/validation_metrics_computation/blob/master/raidionicsval/Validation/kfold_model_validation.py#L155) is a good place to start.

## [Installation](https://github.com/dbouget/validation_metrics_computation#installation)

```
pip install git+https://github.com/dbouget/validation_metrics_computation.git
```

## [Getting started](https://github.com/dbouget/validation_metrics_computation#getting-started)

### [Notebooks](https://github.com/dbouget/validation_metrics_computation#notebooks)

Below are two Jupyter Notebooks which include simple examples on how to get started.

<div style="display: flex;">
  <div style="flex: 1; margin-right: 20px;">

| Notebook       | Colab                                                | GitHub                                                                                                                   |
|----------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Validation** | <a href="https://colab.research.google.com/gist/dbouget/491b0d34e3df00e730cd7fe7a8989202/compute_validation_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [![View on GitHub](https://img.shields.io/badge/View%20on%20GitHub-blue?logo=github)](https://github.com/dbouget/validation_metrics_computation/blob/master/notebooks/compute_validation_example.ipynb) |
| **Study**      | <a href="https://colab.research.google.com/gist/dbouget/ccf77f31ac4ef58bb61d0808eaa9f454/compute_study_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>   | [![View on GitHub](https://img.shields.io/badge/View%20on%20GitHub-blue?logo=github)](https://github.com/dbouget/validation_metrics_computation/blob/master/notebooks/compute_study_example.ipynb)      |

  </div>
</div>

### [Usage](https://github.com/dbouget/validation_metrics_computation#usage)

In the following, a description of how the data should be organized on disk is provided, and a test dataset can
be downloaded [here](https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsValLib_UnitTest1-v1.1.zip).

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
Create a virtual environment using at least Python 3.8, and install the library.

```
  cd /path/to/validation_metrics_computation  
  virtualenv -p python3 venv  
  source venv/bin/activate  
  TMPDIR=$PWD/venv pip install --cache-dir=$PWD/venv -e .
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

After filling in the configuration file specifying all runtime parameters,
according to the pattern from [**blank_main_config.ini**](https://github.com/dbouget/validation_metrics_computation/blob/master/blank_main_config.ini),
you should run first the __validation__ task and then the __study__ task.  


#### [CLI](https://github.com/dbouget/validation_metrics_computation#cli)
```
raidionicsval -c CONFIG (-v debug)
```

CONFIG should point to a configuration file (*.ini).

#### [Python module](https://github.com/dbouget/validation_metrics_computation#python-module)
```
from raidionicsval import compute
compute(config_filename="/path/to/main_config.ini")
```

"/path/to/main_config.ini" should point to a valid configuration file.

#### [Docker](https://github.com/dbouget/validation_metrics_computation#docker)
When calling Docker images, the --user flag must be properly used in order for the folders and files created inside
the container to inherit the proper read/write permissions. The user ID is retrieved on-the-fly in the following
examples, but it can be given in a more hard-coded fashion if known by the user.

```
docker pull dbouget/raidionics-val:v1.0-py38-cpu
```

For opening the Docker image and interacting with it, run:  
```
docker run --entrypoint /bin/bash -v /home/<username>/<resources_path>:/workspace/resources -t -i --network=host --ipc=host --user $(id -u) dbouget/raidionics-val:v1.0-py38-cpu
```

The `/home/<username>/<resources_path>` before the column sign has to be changed to match a directory on your local 
machine containing the data to expose to the docker image. Namely, it must contain folder(s) with data to use as input
for the validation studies, and it will contain the destination folder where the results will be saved.

For launching the Docker image as a CLI, run:  
```
docker run -v /home/<username>/<resources_path>:/workspace/resources -t -i --network=host --ipc=host --user $(id -u) dbouget/raidionics-val:v1.0-py38-cpu -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

The `<path>/<to>/main_config.ini` must point to a valid configuration file on your machine, as a relative path to the `/home/<username>/<resources_path>` described above.
For example, if the file is located on my machine under `/home/myuser/Data/Validation/main_config.ini`, 
and that `/home/myuser/Data` is the mounted resources partition mounted on the Docker image, the new relative path will be `Validation/main_config.ini`.  
The `<verbose>` level can be selected from [debug, info, warning, error].

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
