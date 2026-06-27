# Federated Brain Tumour Segmentation
### Communication-Efficient Federated Learning with Adaptive Client Selection

---

## Overview

This project implements a communication-efficient federated learning framework for 3D brain tumour segmentation using the BraTS 2020 dataset.

**Key contributions:**
- Federated learning with FedAvg across 5 simulated hospitals
- Hybrid client selection strategy (Round-Robin + Random) for faster convergence
- 3D Simple U-Net and Attention U-Net architectures
- Evaluation on Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET)

---

## Architecture

```
BraTS 2020 Dataset (369 cases)
         |
    Preprocessing
    (crop 128³, normalise, stack T1ce/T2/FLAIR)
         |
    _____|_____
   |           |
Centralised  Federated
 Training     Setup
 (100 epochs) (5 hospitals: 120,80,50,30,15)
   |              |
Validation     FedAvg (50 rounds)
Dice/Jaccard      |
             Hybrid Selection
             (2 RR + 1 Random)
                  |
           Global Evaluation
```

---

## Results

| Model | Mean Dice | Mean Jaccard | WT | TC | ET |
|---|---|---|---|---|---|
| Centralised Simple U-Net | 0.723 | 0.607 | 0.804 | 0.722 | 0.553 |
| Centralised Attention U-Net | 0.781 | 0.658 | 0.835 | 0.762 | 0.603 |
| Federated Simple U-Net | 0.673 | 0.554 | 0.765 | 0.682 | 0.503 |
| Federated Attention U-Net | 0.746 | 0.622 | 0.814 | 0.734 | 0.573 |

**Hybrid strategy converges in ~42 rounds vs 50 rounds for random selection (~16% reduction in communication overhead).**

---

## Setup

```bash
pip install torch torchvision nibabel scikit-learn matplotlib
```

---

## Usage

### 1. Preprocess BraTS 2020
```bash
python src/preprocessing.py \
    --data_root  /path/to/BraTS2020_TrainingData \
    --output_root ./preprocessed
```

### 2. Centralised Training
```bash
# Simple U-Net
python src/train_centralised.py \
    --data_dir ./preprocessed \
    --model    simple_unet

# Attention U-Net
python src/train_centralised.py \
    --data_dir ./preprocessed \
    --model    attention_unet
```

### 3. Federated Training
```bash
# Random client selection
python src/federated_learning.py \
    --data_dir ./preprocessed \
    --strategy random

# Hybrid client selection (proposed)
python src/federated_learning.py \
    --data_dir ./preprocessed \
    --strategy hybrid
```

---

## Project Structure

```
├── src/
│   ├── preprocessing.py      # Data loading, cropping, normalisation
│   ├── models.py             # 3D Simple U-Net and Attention U-Net
│   ├── metrics.py            # Dice loss, Dice/Jaccard evaluation
│   ├── train_centralised.py  # Centralised training pipeline
│   └── federated_learning.py # Federated training with client selection
├── results/                  # Saved checkpoints and convergence plots
└── README.md
```

---

## Dataset

[BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) — 369 multi-institutional glioma cases with 4 MRI modalities (T1, T1ce, T2, FLAIR) and expert segmentation masks.

---

## Technologies

`Python` `PyTorch` `Federated Learning` `3D U-Net` `Medical Image Segmentation` `NiBabel` `NumPy` `Matplotlib`
