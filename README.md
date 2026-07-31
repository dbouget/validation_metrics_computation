<div align="center">

# Raidionics Validation & Metrics Computation

**Backend for k-fold cross-validation and metrics computation over 2D-3D medical data.**

Part of the [Raidionics](https://github.com/raidionics) ecosystem.

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12%7C3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/raidionicsval.svg)](https://pypi.org/project/raidionicsval/)
[![codecov](https://img.shields.io/codecov/c/github/dbouget/validation_metrics_computation)](https://codecov.io/gh/dbouget/validation_metrics_computation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/dbouget/491b0d34e3df00e730cd7fe7a8989202/compute_validation_example.ipynb)
[![Paper](https://img.shields.io/badge/DOI-10.3389%2Ffneur.2022.932219-blue.svg)](https://www.frontiersin.org/articles/10.3389/fneur.2022.932219/full)

</div>

---

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Notebooks](#notebooks)
- [Usage](#usage)
  - [CLI](#cli)
  - [Python module](#python-module)
  - [Docker](#docker)
- [Data format](#data-format)
- [How to cite](#how-to-cite)
- [License](#license)

---

## Overview

This library computes cross-validation performance and segmentation metrics for the Raidionics ecosystem. It supports both **2D and 3D** inputs and can run for multiple segmentation classes simultaneously.

It can be used in three ways:

| Mode | Best for |
|---|---|
| **Python module** | Integrating metrics computation into your own pipeline |
| **CLI** | Quick, scriptable runs from a config file |
| **Docker** | Reproducible environments, no local Python setup needed |

> ⚠️ The only hard requirement is that your data follows the [expected folder structure](docs/data_format.md). For custom structures, [`kfold_model_validation.py`](raidionicsval/Validation/kfold_model_validation.py#L155) is the place to start adapting the code.

---

## Installation

```bash
pip install raidionicsval
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/dbouget/validation_metrics_computation.git
```

---

## Quick start

```bash
cd /path/to/validation_metrics_computation
cp blank_main_config.ini main_config.ini
```

Edit `main_config.ini` with your paths and parameters (see [`Utils/resources.py`](raidionicsval/Utils/resources.py) for a full description of every field), then run the **validation** task first, followed by the **study** task:

```bash
raidionicsval -c main_config.ini
```

---

## Notebooks

Two Jupyter notebooks demonstrate the library end-to-end:

| Notebook | Colab | GitHub |
|---|---|---|
| **Validation** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/dbouget/491b0d34e3df00e730cd7fe7a8989202/compute_validation_example.ipynb) | [View](notebooks/compute_validation_example.ipynb) |
| **Study** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/dbouget/ccf77f31ac4ef58bb61d0808eaa9f454/compute_study_example.ipynb) | [View](notebooks/compute_study_example.ipynb) |

A sample dataset for testing is available [here](https://github.com/raidionics/Raidionics-models/releases/download/v1.3.0-rc/Samples-RaidionicsValLib_UnitTest1-v1.1.zip).

---

## Usage

### CLI

```bash
raidionicsval -c CONFIG (-v debug)
```

`CONFIG` must point to a valid `.ini` configuration file.

### Python module

```python
from raidionicsval import compute

compute(config_filename="/path/to/main_config.ini")
```

### Docker

```bash
docker pull dbouget/raidionics-val:v1.1.1-py39-cpu

docker run \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-val:v1.1.1-py39-cpu \
  -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

For the interactive shell variant, permission notes, and path-mapping details, see the full **[Docker guide](docs/docker.md)**.

---

## Data format

Full details on expected folder layouts, naming conventions, and the cross-validation folds file are documented in **[docs/data_format.md](docs/data_format.md)**, covering:

- Original data folder structure (index-based and non-index-based)
- Inference results folder structure
- The `cross_validation_folds.txt` file format

---

## How to cite

If you use Raidionics in your research, please cite the software and associated papers. Citation metadata is provided in [`CITATION.cff`](CITATION.cff) — click **"Cite this repository"** in the sidebar for ready-to-use APA/BibTeX formats, covering both the validation/metrics methodology (Frontiers in Neurology, 2022) and the main software release (Scientific Reports, 2023).

---

## License

Distributed under the [BSD-2-Clause License](LICENSE.md).