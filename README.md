# Building Damage Change Detection using Graph Convolution Networks

This repository contains code for the work [Towards Cross-Disaster Building Damage Assessment with Graph Convolutional Networks](https://arxiv.org/abs/2201.10395) (currently in preprint).

# Setup

Install Anaconda or Miniconda.

Run:
```
conda env create -f environment.yaml
```

# Use

It is preferred to create the following directory tree in the same directory of the code files to avoid having to modify path variables inside the script files.

```
.
├── weights
├── results
├── datasets
│   ├── xbd
```

Download the xBD dataset from https://xview2.org/ and unzip the content into the `xbd` subdirectory. Run the `bldgs_xbd.py` script to extract building crops from the xBD dataset.

The `exp_settings.json` file contains the experimental configurations such as model hyperparameters and data subsets and can be modified accordingly.

# Citation
If you use this work, please cite:

```
@misc{ismail2022crossdisaster,
      title={Towards Cross-Disaster Building Damage Assessment with Graph Convolutional Networks}, 
      author={Ali Ismail and Mariette Awad},
      year={2022},
      eprint={2201.10395},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
