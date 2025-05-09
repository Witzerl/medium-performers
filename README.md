# MICCAI-LH-BraTS2025-MET Challenge - Brain Metastases Segmentation

This repository contains code and utilities for preprocessing, loading, and training a segmentation model on the
MICCAI-LH-BraTS2025-MET Challenge dataset.

---

## 📁 Project Structure

<pre>
. 
├── ml/ 
│   ├── dataset.py 
│   └── trainer.py  
├── utils/ 
│   ├── preprocessing.py    # Data preprocessing
│   └── loading_utils.py    # File loading, parsing, and helper functions 
├── visualization/ 
│   └── visualization.py    # Plotting utilities (modalities + overlays)  
├── intro.ipynb             # Getting started: visualizations, sample inspection 
├── training.ipynb          # Full pipeline: data loading, training loop 
├── train.py                # Training pipeline
└── requirements.txt        # Dependencies 
</pre>

---

## 📚 Notebooks

- **`intro.ipynb`**:
    - Demonstrates how to inspect and visualize the raw NIfTI files.
    - Includes how to apply basic transformations using `TorchIO`.

- **`training.ipynb`**:
    - Prepares the training and validation datasets.
    - Trains a U-Net using a `Trainer` class and patch-based sampling.

---

## ⚙️ Setup Environment

Recommended setup using uv (much faster than conda and more reproducible).

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:
   ```shell
   uv venv --python 3.13
   ```
2. Activate the virtual environment (Windows):
   ```shell
   .venv\Scripts\activate
   ```
   or (Linux):
   ```shell
   source .venv/bin/activate
   ```
3. Install the requirements:
   ```shell
   uv pip install -r requirements.txt
   ```
4. Install monai separately because the numpy dependency is messed up:
   ```bash
   uv pip install --no-deps monai>=1.4.0
   ``` 

---

## 🧠 Model

MONAI implementation of a 3D U-Net, trained with a combination of:

- **CrossEntropy Loss** (for voxel-wise classification)
- **Dice Loss** (for overlap-based regularization) (keine ahnung was da am gscheitesten is)

The patch-based training is handled using TorchIO's `GridSampler` and `Queue`.

---
