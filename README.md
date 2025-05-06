# MICCAI-LH-BraTS2025-MET Challenge - Brain Metastases Segmentation

This repository contains code and utilities for preprocessing, loading, and training a segmentation model on the MICCAI-LH-BraTS2025-MET Challenge dataset.

---

<pre> ## 📁 Project Structure ``` . ├── ml/ │ ├── dataset.py │ ├── trainer.py  │ ├── utils/ │ └── loading_utils.py # File loading, parsing, and helper functions │ ├── visualization/ │ └── visualization.py # Plotting utilities (modalities + overlays) │ ├── training_helper/ # Subset of samples for debugging or testing ├── intro.ipynb # Getting started: visualizations, sample inspection ├── training.ipynb # Full pipeline: data loading, training loop ├── requirements.txt # Dependencies ``` </pre>

---

## 📚 Notebooks

- **`intro.ipynb`**:  
  - Demonstrates how to inspect and visualize the raw NIfTI files.  
  - Includes how to apply basic transformations using `TorchIO`.

- **`training.ipynb`**:  
  - Prepares the training and validation datasets.
  - Trains a U-Net using a `Trainer` class and patch-based sampling.

---

## 🧠 Model

MONAI implementation of a 3D U-Net, trained with a combination of:
- **CrossEntropy Loss** (for voxel-wise classification)
- **Dice Loss** (for overlap-based regularization) (keine ahnung was da am gscheitesten is)

The patch-based training is handled using TorchIO's `GridSampler` and `Queue`.

---
