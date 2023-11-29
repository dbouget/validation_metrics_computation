# Segmentation validation and metrics computation backend for Raidionics related publications

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11|(3.12)-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://zenodo.org/badge/DOI/10.3389/fneur.2022.932219.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)

The code corresponds to the Raidionics backend for running the k-fold cross-validation and metrics computation.
The module can either be used as a Python library, as CLI, or as Docker container.

## [Installation](https://github.com/dbouget/validation_metrics_computation/installation)

```
pip install git+https://github.com/dbouget/validation_metrics_computation.git
```

## [Continuous integration](https://github.com/dbouget/validation_metrics_computation/continuous-integration)

| Operating System | Status                                                                                                             |
|------------------|--------------------------------------------------------------------------------------------------------------------|
| **Windows**      | ![CI](https://github.com/dbouget/validation_metrics_computation/workflows/Build%20Windows/badge.svg?branch=master) |
| **Ubuntu**       | ![CI](https://github.com/dbouget/validation_metrics_computation/workflows/Build%20Ubuntu/badge.svg?branch=master)             |
| **macOS**        | ![CI](https://github.com/dbouget/validation_metrics_computation/workflows/Build%20macOS/badge.svg?branch=master)              |
| **macOS ARM**    | ![CI](https://github.com/dbouget/validation_metrics_computation/workflows/Build%20macOS%20ARM/badge.svg?branch=master)        |

## [How to cite](https://github.com/dbouget/validation_metrics_computation#how-to-cite)
If you are using Raidionics in your research, please cite the following references.

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

## [How to use](https://github.com/dbouget/validation_metrics_computation#how-to-use)
### 1. Folder and data structures and naming conventions
Assuming in the following example that the data are stored based on their origin, but
anything should work. The folders named _index0_ and _index1_ could be renamed to
_Amsterdam_ and _StOlavs_ for instance.

#### 1.1 Original data folder structure
The main data directory containing the original MRI images and corresponding
manual annotations is expected to resemble:

    └── path/to/data/root/
        └── index0/
            ├── AMS0/
            │   ├── volumes/
            │   │   └── AMS0_T1.nii.gz
            │   └── segmentations/
            │       └── AMS0_T1_label_tumor.nii.gz
            ├── AMS25/
            └── AMS50/
        └── index1/
            ├── STO25/
            └── STO50/

#### 1.2 Inference results folder structure
The inference results should be grouped inside what will become the validation folder,
resembling the following structure (here for Study1). The outer-most sub-folder
naming convention inside _predictions_ are the fold numbers.

    └── path/to/validation/study/
        └── predictions/
            ├── 0/
            │   ├── index0_AMS0/
            │   |   └── AMS0_T1-predictions.nii.gz  
            │   ├── index1_STO25/ 
            │   |   └── STO25_T1-predictions.nii.gz  
            └── 0/
                ├── index0_AMS50/
                │   └── AMS50_T1-predictions.nii.gz  
                └── index1_STO50/ 
                    └── STO50_T1-predictions.nii.gz  

#### 1.3 Folds file
The file with patients' distribution within each fold used for training should list
the content of the validation and test sets iteratively.  
The file should be called __cross\_validation\_folds.txt__ and placed in the validation
study folder side-by-side with the _predictions_ sub-folder.  
An example of its content is given below:
  > index0_AMS1000_T1_sample index1_STO250_T1_sample\n    
  > index0_AMS0_T1_sample index1_STO25_T1_sample\n  
  > index0_AMS0_T1_sample index1_STO25_T1_sample\n    
  > index0_AMS25_T1_sample index1_STO50_T1_sample\n  

### 2. Installation
Create a virtual environment using at least Python 3.8, and install all dependencies from
the requirements.txt file.
  > cd /path/to/validation_metrics_computation  
  > virtualenv -p python3 venv  
  > source venv/bin/activate  
  > TMPDIR=$PWD/venv pip install --cache-dir=$PWD/venv -r requirements.txt (--no-deps)

Then the final step is to do the following in a terminal.
  > cd /path/to/validation_metrics_computation  
  > cp blank_main_config.ini main_config.ini 

You can now edit your __main\_config.ini__ file for running the different processes.  
An additional explanation of all parameters specified in the configuration file can be
found in _/Utils/resources.py_. 

### 3. Process
To run, you need to supply the configuration file as parameter.
  > python main.py -c main_config.ini

After filling in the configuration file, you should run first the 
__validation__ task and then the __study__ task.  
N.B. If no study fits your need, you can create a new study file in _/Studies/_.
