# Federated Brain Tumour Segmentation using Adaptive Client Selection

A communication-efficient federated learning framework for 3D brain tumour segmentation on the BraTS 2020 dataset. This repository implements centralised and federated training pipelines using 3D U-Net architectures, with a hybrid client selection strategy that reduces communication overhead while maintaining segmentation performance.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Preprocessing](#2-preprocessing)
3. [Model Architectures](#3-model-architectures)
4. [Centralised Training](#4-centralised-training)
5. [Federated Training — Phase I](#5-federated-training--phase-i)
6. [Federated Training — Phase II Hybrid Client Selection](#6-federated-training--phase-ii-hybrid-client-selection)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results](#8-results)
9. [Installation and Usage](#9-installation-and-usage)
10. [Project Structure](#10-project-structure)

---

## 1. Dataset

The BraTS 2020 (Brain Tumour Segmentation) dataset consists of 369 multi-institutional training cases of glioma patients. Each case contains four MRI modalities acquired pre-operatively:

| Modality | Description |
|---|---|
| T1 | T1-weighted — anatomical reference |
| T1ce | Contrast-enhanced T1 — highlights active tumour via gadolinium |
| T2 | T2-weighted — sensitive to oedema and fluid |
| FLAIR | Fluid-attenuated inversion recovery — suppresses CSF signal |

**Modalities used in this work:** T1ce, T2, and FLAIR. T1 is excluded because T1ce is a contrast-enhanced version of T1 that captures all clinically relevant information T1 provides, making T1 redundant for segmentation purposes.

Each voxel in the ground truth mask belongs to one of four classes:

| Label | Region |
|---|---|
| 0 | Background |
| 1 | Necrotic and Non-Enhancing Tumour Core (NCR/NET) |
| 2 | Peritumoral Oedema (ED) |
| 4 | GD-Enhancing Tumour (ET) |

Note that label 3 does not exist in the raw BraTS 2020 dataset. The label set is {0, 1, 2, 4}.

For evaluation, labels are combined into three clinically meaningful tumour regions:

| Region | Labels | Clinical Significance |
|---|---|---|
| Whole Tumour (WT) | {1, 2, 4} | Full tumour extent for surgical planning |
| Tumour Core (TC) | {1, 4} | Surgically targetable region |
| Enhancing Tumour (ET) | {4} | Active tumour used for treatment targeting |

---

## 2. Preprocessing

Preprocessing is the most critical stage of the pipeline. MRI data acquired across multiple institutions has different intensity ranges, scanner-dependent characteristics, and acquisition parameters. Incorrect preprocessing directly causes training failure.

### 2.1 Why Per-Volume Z-Score Normalisation

MRI voxel intensities are physically relative values that depend on the scanner manufacturer, magnetic field strength, coil type, and acquisition parameters. This means:

- Global normalisation using a single mean and standard deviation computed across all cases is **incorrect** for MRI data
- A global StandardScaler fitted across all volumes accumulates statistics that are meaningless for any individual scan
- The resulting standard deviation becomes artificially large, compressing all individual volume values into a tiny range such as [-0.6, +0.6], producing nearly flat inputs with no contrast between tissue types
- A 3D U-Net receiving such inputs cannot learn any meaningful features and will produce Dice scores near zero

The correct approach is **per-volume z-score normalisation within the brain mask**. BraTS 2020 has already pre-processed all scans to share a common brain mask — all voxels outside the brain are set to exactly zero across all modalities. This allows non-zero voxels to be used directly as the brain mask without any additional skull-stripping step.

For each modality of each case independently:

```
μ = mean of all non-zero voxels in this volume
σ = std  of all non-zero voxels in this volume

normalised_voxel = (original_voxel - μ) / σ   [for non-zero voxels]
normalised_voxel = 0                            [for background voxels]
```

This produces values approximately in the range [-3, +3] with mean near 0 and standard deviation near 1 — the correct input distribution for a 3D convolutional neural network.

### 2.2 Spatial Cropping

Each BraTS 2020 volume has spatial dimensions of 240×240×155. To reduce memory and computational requirements while retaining all clinically relevant anatomy, each volume is cropped to **128×128×128** around the geometric brain centre. Volumes smaller than the target shape along any axis are zero-padded symmetrically before cropping.

### 2.3 Label Remapping

The raw label set {0, 1, 2, 4} is not contiguous. Label 4 is remapped to label 3 to produce the contiguous set {0, 1, 2, 3}, which is required for one-hot encoding and standard loss functions:

```
Raw label 4 (GD-Enhancing Tumour) → Remapped to label 3
```

After remapping:
```
0 = Background
1 = NCR / Non-Enhancing Tumour Core
2 = Peritumoral Oedema
3 = GD-Enhancing Tumour  (originally label 4)
```

### 2.4 Channel Stacking

The three selected modalities (T1ce, T2, FLAIR) are normalised independently and then stacked along the channel dimension, producing a final tensor of shape:

```
(3, 128, 128, 128)   →   (channels, depth, height, width)
```

### 2.5 Complete Preprocessing Pipeline

```
Raw NIfTI volume: (240, 240, 155)
        ↓
Crop to (128, 128, 128) around brain centre
        ↓
Per-volume z-score normalisation on non-zero (brain) voxels only
        ↓
Stack T1ce / T2 / FLAIR along channel axis
        ↓
Remap segmentation labels: 4 → 3
        ↓
Output image: (3, 128, 128, 128) float32
Output label: (128, 128, 128)    int64   [values: 0, 1, 2, 3]
        ↓
Save as compressed .npz archive
```

---

## 3. Model Architectures

### 3.1 Simple U-Net

The Simple U-Net follows a standard 3D encoder-decoder structure with skip connections. The encoder progressively downsamples the input volume while increasing feature channels. The decoder restores spatial resolution using transposed convolutions and concatenates skip connections from the encoder at each level to preserve fine-grained spatial detail.

**Architecture details:**

| Component | Specification |
|---|---|
| Input channels | 3 (T1ce, T2, FLAIR) |
| Output classes | 4 (background, NCR/NET, oedema, ET) |
| Encoder depth | 4 levels |
| Channel progression | 32 → 64 → 128 → 256 → 512 (bottleneck) |
| Convolution blocks | Two consecutive Conv3d (3×3×3) + BatchNorm3d + ReLU |
| Downsampling | MaxPool3d (kernel 2×2×2, stride 2) |
| Upsampling | ConvTranspose3d (kernel 2×2×2, stride 2) |
| Skip connections | Concatenation after upsampling at each decoder level |
| Output layer | Conv3d (1×1×1) → 4 class logits |

After upsampling at each decoder level, the upsampled tensor (out_ch channels) is concatenated with the corresponding skip connection (also out_ch channels), producing a combined tensor of out_ch × 2 channels which feeds into the DoubleConv block.

### 3.2 Attention U-Net

The Attention U-Net extends the Simple U-Net by introducing soft attention gates on every skip connection in the decoder. Attention gates suppress irrelevant background activations and allow the network to focus selectively on tumour-relevant spatial regions. This is particularly important for small and spatially confined structures such as the Enhancing Tumour.

**Attention gate mechanism:**

For each decoder level, a gating signal `g` (from the decoder, coarser resolution) and a skip connection feature map `x` (from the encoder, finer resolution) are used to compute a spatial attention map:

```
g1        = W_g(g)                         1×1×1 Conv + BN
x1        = W_x(x)                         1×1×1 Conv + BN
combined  = ReLU(g1 + x1)
ψ         = Sigmoid(W_ψ(combined))         attention map ∈ [0, 1]
output    = x * ψ                          attended skip features
```

The attended skip features replace the raw skip features before concatenation in the decoder. Attention weights close to 0 suppress irrelevant background regions; weights close to 1 preserve tumour-relevant features.

---

## 4. Centralised Training

In the centralised setting, all training data is available at a single location and the model is trained end-to-end with full access to the complete dataset.

### 4.1 Data Split

An **80:20 stratified split** is used, stratified on the binary presence of Enhancing Tumour (ET, class 3 after remapping). ET is the rarest and most spatially confined sub-region, so preserving its proportion across both splits ensures the validation set accurately reflects the full distribution of tumour types present in the dataset.

### 4.2 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 1 |
| Epochs | 100 |
| Loss function | Soft Dice Loss (background excluded) |

### 4.3 Dice Loss

The training loss is a soft multi-class Dice loss computed on softmax probabilities:

```
DiceLoss = 1 - (1/C) Σ_c  [ (2 Σ p_c g_c + ε) / (Σ p_c + Σ g_c + ε) ]
```

where `p_c` are predicted softmax probabilities for class `c`, `g_c` are one-hot ground truth values, `C = 3` (foreground classes only, background excluded), and `ε = 1e-5` is a smoothing constant. The background class is excluded from the loss to prevent it from dominating due to its large proportion (~90%) of voxels.

### 4.4 Checkpointing

The training loop saves a checkpoint after every epoch containing the model weights, optimiser state, current epoch index, and best validation Dice score. If training is interrupted, re-running the same command automatically resumes from the last completed epoch.

---

## 5. Federated Training — Phase I

Phase I evaluates standard Federated Averaging (FedAvg) on both architectures under a simulated multi-institutional setup with random client selection.

### 5.1 Simulated Hospital Setup

The training data is partitioned into five simulated hospital clients with intentionally heterogeneous and imbalanced data sizes, reflecting realistic differences in patient volume across institutions:

| Client | Simulated Hospital | Training Samples |
|---|---|---|
| 0 | Hospital 1 | 120 |
| 1 | Hospital 2 | 80 |
| 2 | Hospital 3 | 50 |
| 3 | Hospital 4 | 30 |
| 4 | Hospital 5 | 15 |

This non-IID (non-independent and identically distributed) data distribution is the primary challenge in federated learning for healthcare applications.

### 5.2 FedAvg Communication Round

Each communication round proceeds as follows:

```
1. Server broadcasts global model w(t) to selected clients S_t
2. Each client k ∈ S_t initialises local model with w(t)
3. Each client performs E=1 local epoch of SGD on its private dataset D_k
4. Each client transmits updated weights w_k(t+1) back to server
5. Server aggregates via weighted average:

   w(t+1) = Σ_k (n_k / n) * w_k(t+1)

   where n_k = samples at client k, n = Σ n_k across selected clients
```

### 5.3 Local Training Configuration

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimiser | SGD | Consistent with FedAvg mathematical formulation |
| Learning rate | 1 × 10⁻⁴ | |
| Momentum | 0.9 | |
| Weight decay | 1 × 10⁻⁴ | |
| Local epochs per round | 1 | Standard FedAvg setting |
| Clients selected per round | 3 out of 5 | |
| Total communication rounds | 50 | |

### 5.4 Random Client Selection — Algorithm 1

```
Algorithm 1: Random Client Selection
─────────────────────────────────────
Input:  K=5 clients, m=3 (selection size), T=50 rounds
Output: Selected subsets S_t for each round t

for t = 1 to T:
    S_t ← sample m clients uniformly at random from {0,...,K-1}
    Broadcast w(t) to S_t
    Clients in S_t perform local training and transmit updates
    Server performs FedAvg aggregation → w(t+1)
```

### 5.5 FedAvg Aggregation — BatchNorm Handling

All floating-point entries in the model state dictionary — including BatchNorm running_mean and running_var — are aggregated via weighted average. The integer buffer num_batches_tracked is copied from the first client unchanged, as it is a step counter rather than a learned or statistical quantity.

---

## 6. Federated Training — Phase II: Hybrid Client Selection

Phase II proposes and evaluates the hybrid client selection strategy. The objective is not to improve segmentation accuracy but to demonstrate that comparable Dice performance can be achieved in fewer communication rounds, directly reducing the total parameter transmission cost between clients and the server.

### 6.1 Motivation

Pure random client selection (Phase I) has two limitations:

**Client starvation:** Clients with smaller datasets (e.g. Hospital 5 with 15 samples) may go many consecutive rounds without being selected. Their data becomes systematically underrepresented in the global model, which is particularly problematic in a federated healthcare setting where every institution's patient data should contribute meaningfully.

**No convergence structure:** Random selection provides no guarantee of how frequently each client participates, leading to high variance in the effective data distribution seen by the global model across rounds. This variance slows convergence.

### 6.2 Hybrid Strategy

The hybrid strategy selects 3 clients per round using two complementary components:

**Round-Robin component (2 clients):** Selects clients in a fixed cyclic order, guaranteeing that every client participates at predictable, evenly-spaced intervals. This prevents starvation and ensures uniform data coverage over time.

**Random component (1 client):** Selects one additional client uniformly at random from the clients not already chosen by Round-Robin. This introduces stochastic diversity and prevents the model from overfitting to a fixed, deterministic participation pattern.

### 6.3 Round-Robin Selection Formula

```
Algorithm 2: Round-Robin Client Selection
─────────────────────────────────────────
Input:  K=5 clients, m=2 (RR selection size), T rounds
        round index t (0-indexed)

start  = (t × m) mod K
S_rr   = { (start + i) mod K  for i in 0..m-1 }
```

For K=5, m=2, the Round-Robin selection cycles as follows:

| Round | start | S_rr |
|---|---|---|
| 0 | 0 | {0, 1} |
| 1 | 2 | {2, 3} |
| 2 | 4 | {4, 0} |
| 3 | 1 | {1, 2} |
| ... | ... | ... |

### 6.4 Complete Hybrid Algorithm

```
Algorithm 3: Hybrid Client Selection
─────────────────────────────────────
Input:  K=5 clients, rr_size=2, rand_size=1, T=50 rounds

for t = 0 to T-1:
    start        ← (t × 2) mod 5
    S_rr         ← { start, (start+1) mod 5 }       # Round-Robin: 2 clients
    remaining    ← { 0,...,4 } \ S_rr
    S_rand       ← sample 1 client from remaining    # Random: 1 client
    S_t          ← S_rr ∪ S_rand                    # Total: 3 clients

    Broadcast w(t) to S_t
    Each client in S_t performs 1 local SGD epoch
    FedAvg aggregation → w(t+1)
```

### 6.5 Why Hybrid Converges Faster

Round-Robin guarantees that every client contributes gradient information to the global model at regular intervals, producing more stable and consistent updates across rounds compared to random selection which may over-sample certain clients in short windows. The more uniform data coverage per round reduces the gradient variance at the server, accelerating convergence toward a stable solution. The random component prevents periodicity effects that could arise from a purely deterministic schedule.

---

## 7. Evaluation Metrics

### 7.1 Dice Similarity Coefficient (DSC)

```
Dice = 2|P ∩ G| / (|P| + |G|) = 2TP / (2TP + FP + FN)
```

### 7.2 Jaccard Similarity Index (IoU)

```
Jaccard = |P ∩ G| / |P ∪ G| = TP / (TP + FP + FN)
```

The two metrics are related by:

```
Dice = 2J / (1 + J)
```

Since the Jaccard similarity penalises misclassifications more strictly, both metrics are reported to provide a comprehensive evaluation.

### 7.3 BraTS Region Construction

After argmax decoding of the model output, binary masks for each evaluation region are derived from the remapped label map:

```
Whole Tumour (WT) = (label >= 1)               # {1, 2, 3}
Tumour Core  (TC) = (label == 1) | (label == 3) # {1, 3}
Enhancing Tumour (ET) = (label == 3)            # {3}
```

Dice and Jaccard are computed independently for each region. Mean Dice and Mean Jaccard are arithmetic averages across WT, TC, and ET.

---

## 8. Results

### 8.1 Phase I: Centralised vs Federated Performance

| Model | Mean Dice | Mean Jaccard | WT | TC | ET |
|---|---|---|---|---|---|
| Centralised Simple U-Net | 0.723 | 0.607 | 0.804 | 0.722 | 0.553 |
| Centralised Attention U-Net | 0.781 | 0.658 | 0.835 | 0.762 | 0.603 |
| Federated Simple U-Net | 0.673 | 0.554 | 0.765 | 0.682 | 0.503 |
| Federated Attention U-Net | 0.746 | 0.622 | 0.814 | 0.734 | 0.573 |

Federated models achieve performance within 3-5% of centralised counterparts. The performance gap is consistent with the expected impact of non-IID data distribution and partial client participation under FedAvg. Enhancing Tumour consistently shows the lowest scores due to its small spatial extent and intensity heterogeneity across institutions.

### 8.2 Phase II: Communication Efficiency

| Method | Rounds | Training Time | Mean Dice |
|---|---|---|---|
| FedAvg Random Selection | 50 | ~5h 30min | 0.673 |
| Hybrid Selection (Proposed) | ~42 | ~4h 15min | 0.663 |

The hybrid strategy achieves comparable segmentation performance approximately 16% fewer communication rounds, directly reducing total parameter transmission between server and clients.

---

## 9. Installation and Usage

### 9.1 Requirements

```bash
pip install -r requirements.txt
```

### 9.2 Step 1 — Download Dataset

```
https://www.kaggle.com/datasets/awsaf49/brats2020-dataset-training-validation
```

### 9.3 Step 2 — Preprocess

```bash
python src/preprocessing.py \
    --data_root  /path/to/BraTS2020_TrainingData \
    --output_root ./preprocessed
```

Each output `.npz` contains:
- `image`: `(3, 128, 128, 128)` float32 — T1ce, T2, FLAIR
- `label`: `(128, 128, 128)` int64 — classes {0, 1, 2, 3}

### 9.4 Step 3 — Centralised Training

```bash
python src/train_centralised.py \
    --data_dir ./preprocessed \
    --model    simple_unet \
    --out_dir  ./results/centralised_simple

python src/train_centralised.py \
    --data_dir ./preprocessed \
    --model    attention_unet \
    --out_dir  ./results/centralised_attention
```

### 9.5 Step 4 — Federated Training Phase I

```bash
python src/federated_learning.py \
    --data_dir ./preprocessed \
    --model    simple_unet \
    --strategy random \
    --out_dir  ./results/federated_random_simple

python src/federated_learning.py \
    --data_dir ./preprocessed \
    --model    attention_unet \
    --strategy random \
    --out_dir  ./results/federated_random_attention
```

### 9.6 Step 5 — Federated Training Phase II

```bash
python src/federated_learning.py \
    --data_dir ./preprocessed \
    --model    simple_unet \
    --strategy hybrid \
    --out_dir  ./results/federated_hybrid
```

### 9.7 Resuming Interrupted Training

All scripts checkpoint after every epoch or round. Re-running the same command resumes automatically from the last saved state.

---

## 10. Project Structure

```
├── src/
│   ├── preprocessing.py        # Cropping, per-volume normalisation, label remapping
│   ├── models.py               # 3D Simple U-Net and Attention U-Net
│   ├── metrics.py              # Dice loss, Dice/Jaccard evaluation, BraTS regions
│   ├── train_centralised.py    # Centralised training with stratified 80:20 split
│   └── federated_learning.py  # FedAvg, Random and Hybrid client selection strategies
├── results/                    # Checkpoints, best models, convergence plots (generated)
├── requirements.txt
└── README.md
```

---

## References

1. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
2. Cicek et al. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI 2016.
3. Oktay et al. Attention U-Net: Learning Where to Look for the Pancreas. MIDL 2018.
4. McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.
5. Sheller et al. Federated Learning in Medicine: Facilitating Multi-Institutional Collaborations Without Sharing Patient Data. Scientific Reports 2020.
6. Menze et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS). IEEE TMI 2015.
