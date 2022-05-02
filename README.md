# Stand-alone project for segmentation validation and metrics computation
Designed for brain tumor segmentation purposes. This software was released together with the article [Preoperative brain tumor imaging: models and software for segmentation and standardized reporting](https://doi.org/10.48550/arxiv.2204.14199).

# 1. Folder and data structures and naming conventions
Assuming in the following example that the data are stored based on their origin, but
anything should work. The folders named _index0_ and _index1_ could be renamed to
_Amsterdam_ and _StOlavs_ for instance.

## 1.1 Original data folder structure
The main data directory containing the original MRI images and corresponding
manual annotations is expected to resemble:

    ├── path/to/data/root/
        ├── index0/
        │   ├── AMS0/
        |   |   ├── volumes/
        |   |   |   ├── AMS0_T1.nii.gz
        |   |   ├── segmentations/
        |   |   |   ├── AMS0_T1_label_tumor.nii.gz
        │   ├── AMS25/
        │   ├── AMS50/
        ├── index1/
        │   ├── STO25/
        │   ├── STO50/

## 1.2 Inference results folder structure
The inference results should be grouped inside what will become the validation folder,
resembling the following structure (here for Study1). The outer-most sub-folder
naming convention inside _predictions_ are the fold numbers.

    ├── path/to/validation/study/
        ├── predictions/
        │   ├── 0/
        |   |   ├── index0_AMS0/
        |   |   |   ├── AMS0_T1-predictions.nii.gz  
        |   |   ├── index1_STO25/ 
        |   |   |   ├── STO25_T1-predictions.nii.gz  
        │   ├── 0/
        |   |   ├── index0_AMS50/
        |   |   |   ├── AMS50_T1-predictions.nii.gz  
        |   |   ├── index1_STO50/ 
        |   |   |   ├── STO50_T1-predictions.nii.gz  

## 1.3 Folds file
The file with patients' distribution within each fold used for training should list
the content of the validation and test sets iteratively.  
The file should be called __cross\_validation\_folds.txt__ and placed in the validation
study folder side-by-side with the _predictions_ sub-folder.  
An example of its content is given below:
  > index0_AMS1000_T1_sample index1_STO250_T1_sample\n    
  > index0_AMS0_T1_sample index1_STO25_T1_sample\n  
  > index0_AMS0_T1_sample index1_STO25_T1_sample\n    
  > index0_AMS25_T1_sample index1_STO50_T1_sample\n  

# 2. Installation
Create a virtual environment using at least Python 3.7.0, and install all dependencies from
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

# 3. Process
To run, you need to supply the configuration file as parameter.
  > python main.py -c main_config.ini

After filling in the configuration file, you should run first the 
__validation__ task and then the __study__ task.  
N.B. If no study fits your need, you can create a new study file in _/Studies/_.

### How to cite
Please, consider citing our paper, if you find the work useful:
```
@misc{https://doi.org/10.48550/arxiv.2204.14199,
title = {Preoperative brain tumor imaging: models and software for segmentation and standardized reporting},
author = {Bouget, D. and Pedersen, A. and Jakola, A. S. and Kavouridis, V. and Emblem, K. E. and Eijgelaar, R. S. and Kommers, I. and Ardon, H. and Barkhof, F. and Bello, L. and Berger, M. S. and Nibali, M. C. and Furtner, J. and Hervey-Jumper, S. and Idema, A. J. S. and Kiesel, B. and Kloet, A. and Mandonnet, E. and Müller, D. M. J. and Robe, P. A. and Rossi, M. and Sciortino, T. and Brink, W. Van den and Wagemakers, M. and Widhalm, G. and Witte, M. G. and Zwinderman, A. H. and Hamer, P. C. De Witt and Solheim, O. and Reinertsen, I.},
doi = {10.48550/ARXIV.2204.14199},
url = {https://arxiv.org/abs/2204.14199},
keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.6; J.3},
publisher = {arXiv},
year = {2022},
copyright = {Creative Commons Attribution 4.0 International}}
```
