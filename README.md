# MBG: Mamba Based Graph

**A Graph-Guided Mamba Foundation Model for EEG Decoding**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuraldecoding/mbg/blob/main/MBG_Comprehensive_Guide.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Key Novelties](#key-novelties)
- [Comparison with Prior Works](#comparison-with-prior-works)
- [Downstream Tasks](#downstream-tasks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Motivation

Electroencephalography (EEG) signals are inherently multi-channel recordings captured from electrodes placed on the scalp according to standardized systems (e.g., the International 10-20 system). These electrodes have well-defined spatial relationships governed by both physical distance on the scalp surface and underlying functional brain connectivity patterns.

Existing EEG foundation models have addressed the spatial and temporal dimensions of EEG through different strategies:

- **CBraMod** (ICLR 2025) employs a Criss-Cross Transformer architecture that applies separate spatial attention across channels and temporal attention across time segments. While this captures both dimensions, it treats spatial relationships as fully connected attention without exploiting the known electrode topology.

- **EEGMamba** (Neural Networks 2025) leverages the Mamba2 State Space Model as its backbone, flattening channels and time segments into a single long sequence. While computationally efficient with linear complexity, this approach discards the inherent graph structure of EEG electrode placements.

**The Gap:** Neither approach explicitly models the graph topology of EEG electrodes. Brain signals exhibit structured spatial dependencies -- neighboring electrodes capture correlated signals, and functionally connected brain regions (even if physically distant) share information. This graph structure is a powerful inductive bias that remains unexploited by current foundation models.

**MBG** bridges this gap by introducing a Graph-Guided Mamba architecture that:
1. Explicitly models electrode relationships as a graph with both fixed anatomical and learned functional edges.
2. Combines Graph Neural Network message passing for spatial reasoning with Mamba SSM for efficient temporal modeling.
3. Uses graph-aware scan ordering to make even the temporal model topology-aware.

---

## Architecture Overview

```
Input: (B, C, T, P)
B = Batch Size, C = Channels, T = Time Segments, P = Points per Patch

                    +---------------------------+
                    |      Patch Embedding       |
                    | (Temporal Conv + Spectral  |
                    |  FFT + Positional Conv2d)  |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |                           |
                    |   Graph-Mamba Encoder     |
                    |   (N stacked GEM Blocks)  |
                    |                           |
                    |  +---------------------+  |
                    |  | Graph Attention Layer|  |  <-- Multi-head GAT across channels
                    |  | (Spatial: C dim)     |  |      with adaptive adjacency
                    |  +----------+----------+  |
                    |             |              |
                    |             v              |
                    |  +---------------------+  |
                    |  | Mamba SSM Scan       |  |  <-- Bidirectional Mamba per channel
                    |  | (Temporal: T dim)    |  |      for temporal dynamics
                    |  +----------+----------+  |
                    |             |              |
                    |             v              |
                    |  +---------------------+  |
                    |  | Cross-Domain Fusion  |  |  <-- Learned gating mechanism
                    |  | Gate                 |  |      to integrate spatial &
                    |  +----------+----------+  |      temporal features
                    |             |              |
                    +-------------+--------------+
                                  |
                                  v
                    +---------------------------+
                    |     Output Projection      |
                    | (Linear: D -> out_dim)     |
                    +---------------------------+
```

### Detailed Data Flow

```
Input EEG Signal: (B, C, T, P)
         |
         v
[Patch Embedding]
    - Temporal Conv2d: (B, 1, C*T, P) -> conv with kernel (1,49), stride (1,25)
    - Spectral FFT: rfft per patch -> linear projection of magnitude
    - Positional Conv2d: depthwise conv2d (7x7) for position encoding
    - Output: (B, C, T, D)
         |
         v
[GEM Block x N]
    |
    |--- [1] Graph Attention Layer (pre-norm) -----+
    |         Reshape to (B*T, C, D)               |
    |         Compute adaptive adjacency:          |
    |           A = sigmoid(alpha)*A_prior +       |
    |               (1-sigmoid(alpha))*A_learned   |
    |         Multi-head GAT with adjacency mask   |
    |         Output: (B, C, T, D) z_spatial       |
    |                                              |
    |--- [2] Mamba SSM Scan (pre-norm) ------------+
    |         Reshape to (B*C, T, D)               |
    |         Bidirectional Mamba (MixerModel)     |
    |         Output: (B, C, T, D) z_temporal      |
    |                                              |
    |--- [3] Cross-Domain Fusion Gate -------------+
              gate = sigmoid(W_g * [z_spatial; z_temporal])
              fused = gate * z_spatial + (1-gate) * z_temporal
    |--- [4] Residual + FFN -----------------------+
              x = x + fused
              x = x + FFN(LayerNorm(x))
         |
         v
Final Representation: (B, C, T, D)
         |
         v
[Output Projection: Linear(D, out_dim)]
         |
         v
Output: (B, C, T, out_dim)
```

---

## Key Novelties

### 1. Adaptive Graph Construction

MBG constructs the channel adjacency matrix by combining two complementary sources:

- **Prior Graph (A_prior):** Derived from physical electrode distances using a Gaussian kernel on 10-20 system coordinates. Stored as a fixed buffer per layer.
- **Learned Graph (A_learned):** Attention-based dynamic adjacency computed per sample via query-key projections.

```
A_effective = sigmoid(alpha) * A_prior + (1 - sigmoid(alpha)) * A_learned

where alpha is a learnable scalar parameter per layer (initialized to 0.5)
```

### 2. Graph-Enhanced Mamba (GEM) Block

Each GEM block performs a three-stage computation:

1. **Graph Attention (Spatial):** Multi-head GAT across channels at each time step, masked by the adaptive adjacency matrix.
2. **Mamba SSM (Temporal):** Bidirectional Mamba processes temporal sequences per channel independently via MixerModel.
3. **Gating Fusion:** A learned linear gate integrates spatial and temporal outputs adaptively.

### 3. Multi-Scale Graph Pooling for Pretraining

Hierarchical graph pooling for multi-resolution brain representations:
- Level 0: Individual electrodes
- Level 1: Brain sub-regions (frontal, central, parietal, temporal, occipital clusters)
- Level 2: Brain hemispheres + midline
- Level 3: Global brain state

### 4. Bidirectional Mamba with Graph-Aware Scan Order

Graph-aware scan ordering for topology-preserving sequence processing:
- **BFS Order:** Breadth-first traversal from a seed electrode creates spatially coherent scan sequences.
- **DFS Order:** Depth-first traversal follows strongest connection paths.
- Both orderings are deterministic and weighted by adjacency strength.

---

## Comparison with Prior Works

| Aspect | CBraMod (ICLR 2025) | EEGMamba (Neural Networks 2025) | **MBG (Ours)** |
|--------|---------------------|--------------------------------|----------------|
| **Backbone** | Criss-Cross Transformer | Mamba2 SSM | Graph Attention + Mamba SSM |
| **Spatial Modeling** | Fully connected attention | Implicit via flattened sequence | Explicit GAT with electrode topology |
| **Temporal Modeling** | Attention across time (O(T^2)) | Mamba over flattened sequence (O(C*T)) | Mamba per channel (O(T) per channel) |
| **Graph Structure** | Not used | Not used | Adaptive (prior + learned adjacency) |
| **Spatial Inductive Bias** | None | None | Strong (Gaussian kernel on 10-20 distances) |
| **Feature Processing** | Split features for spatial/temporal | Full features in single sequence | Full features in both, then gated fusion |
| **Scan Order** | N/A (attention-based) | Fixed linear | Graph-aware BFS/DFS ordering |

---

## Downstream Tasks

MBG supports evaluation on diverse EEG benchmarks:

| Dataset | Task | Domain | Channels | Subjects |
|---------|------|--------|----------|----------|
| BCI Competition IV 2a | Motor Imagery Classification | BCI | 22 | 9 |
| TUAB | Abnormal Detection | Clinical | 16-21 | 2,993 |
| TUEV | Event Detection | Clinical | 16-21 | 313 |
| ISRUC | Sleep Staging | Sleep | 10 | 100 |
| SEED-V | Emotion Recognition (5-class) | Affective | 62 | 16 |
| CHB-MIT | Seizure Detection | Clinical | 23 | 22 |
| FACED | Emotion Recognition | Affective | 30 | 123 |
| SHU | Motor Imagery | BCI | 14-22 | 25 |
| Speech | Speech Imagery | BCI | 64 | 15 |
| Stress | Stress Detection | Affective | 14 | 28 |
| PhysioNet | Motor Imagery | BCI | 64 | 109 |
| SEED-VIG | Vigilance Estimation | Cognitive | 17 | 23 |
| Mumtaz | Depression Detection | Clinical | 19 | 63 |
| MODMA | Depression Detection | Clinical | 128 | 53 |

---

## Installation

### Requirements

```
torch>=2.0.0
mamba-ssm>=1.2.0   # requires CUDA
numpy>=1.24.0
scipy>=1.10.0
einops>=0.7.0
scikit-learn>=1.3.0
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/neuraldecoding/mbg.git
cd mbg

# Create conda environment
conda create -n mbg python=3.10
conda activate mbg

# Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm (requires matching CUDA toolkit)
pip install mamba-ssm einops

# Install remaining dependencies
pip install -r requirements.txt
```

### Google Colab

Use the provided notebook `MBG_Comprehensive_Guide.ipynb` which handles installation automatically. See the [Quick Start](#quick-start) section below.

> **Note:** `mamba-ssm` and `causal-conv1d` require CUDA compilation. If installation fails due to CUDA version mismatch, install pre-built wheels matching your PyTorch/CUDA version. See [mamba-ssm GitHub](https://github.com/state-spaces/mamba) for compatible versions.

---

## Quick Start

### Basic Usage

```python
import torch
from models.mbg import MBG

# Initialize model
model = MBG(
    in_dim=200,          # Points per patch (e.g., 1 second at 200Hz)
    out_dim=200,         # Output dimension per patch
    d_model=200,         # Hidden dimension
    dim_feedforward=800, # FFN intermediate dimension
    seq_len=30,          # Maximum number of time segments
    n_layer=12,          # Number of GEM blocks
    nhead=8,             # Number of attention heads
    num_channels=22      # Number of EEG channels
)

# Input: (batch, channels, time_segments, points_per_patch)
x = torch.randn(8, 22, 4, 200)
output = model(x)  # Output: (8, 22, 4, 200)
```

### Graph Utilities

```python
from utils.electrode_positions import get_electrode_positions, compute_distance_matrix
from utils.graph_utils import gaussian_adjacency, bfs_ordering, dfs_ordering

# Get electrode positions (supports 19, 22, 64 channels)
positions = get_electrode_positions(22)
dist_matrix = compute_distance_matrix(positions)
adj_matrix = gaussian_adjacency(dist_matrix, sigma=1.0)

# Graph-aware scan orderings
bfs_order = bfs_ordering(adj_matrix, seed_idx=9)  # From Cz
dfs_order = dfs_ordering(adj_matrix, seed_idx=9)
```

### Task-Specific Model (TUAB Example)

```python
from models.model_for_tuab import Model

# Requires a param object with configuration
# See models/model_for_tuab.py for the full interface
```

### Training Utilities

```python
from utils.training_utils import CheckpointManager, EarlyStopping, MetricLogger
from utils.kfold_cv import KFoldCrossValidator, compute_metrics, aggregate_fold_results

# Checkpoint management (designed for Google Colab + Drive)
checkpoint_manager = CheckpointManager(
    checkpoint_dir='/content/drive/MyDrive/MBG_Project/checkpoints',
    save_every_n_epochs=10,
    keep_top_k=3,
    metric_name='balanced_accuracy',
    metric_mode='max',
)

# Subject-independent K-Fold cross validation
cv = KFoldCrossValidator(n_folds=5, split_type='group', random_seed=42)
splits = cv.get_splits(labels, groups=subject_ids)
```

---

## Project Structure

```
mbg/
├── models/
│   ├── __init__.py               # Exports MBG class
│   ├── mbg.py                    # Main MBG model (PatchEmbedding + GEM Blocks + proj_out)
│   ├── gem_block.py              # GEM Block (Graph Attention + Mamba + Fusion Gate + FFN)
│   ├── graph_attention.py        # Multi-head GAT with adaptive adjacency masking
│   ├── graph_construction.py     # AdaptiveAdjacency module & get_prior_adjacency()
│   └── model_for_tuab.py         # Task-specific head for TUAB binary classification
├── modules/
│   ├── __init__.py               # Exports MambaConfig, MixerModel
│   ├── config_mamba.py           # Mamba SSM configuration dataclass
│   └── mixer_seq_simple.py       # MixerModel (Mamba backbone wrapper)
├── utils/
│   ├── __init__.py
│   ├── electrode_positions.py    # 10-20 system coordinates (19/22/64 channels)
│   ├── graph_utils.py            # BFS/DFS orderings, gaussian_adjacency()
│   ├── training_utils.py         # CheckpointManager, EarlyStopping, MetricLogger
│   └── kfold_cv.py               # KFoldCrossValidator, compute_metrics, statistical_test
├── MBG_Comprehensive_Guide.ipynb # Comprehensive notebook (Colab-ready)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{awangga2026mbg,
  title={MBG: A Graph-Guided Mamba Foundation Model for EEG Decoding},
  author={Awangga, Rolly Maulana and Suyanto and Purnama, Bedy},
  journal={},
  year={2026}
}
```

---

## Acknowledgements

This work builds upon:

- **CBraMod:** Yi, Z., et al. "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding." *ICLR 2025*.
- **EEGMamba:** Yi, Z., et al. "EEGMamba: Bidirectional Mamba with Cross-Domain Transfer Learning for EEG Foundation Model." *Neural Networks 2025*.

We thank the developers of [Mamba](https://github.com/state-spaces/mamba) and [MNE-Python](https://mne.tools/) for their excellent open-source tools.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
