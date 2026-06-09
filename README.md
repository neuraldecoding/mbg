# MBG: Mamba Based Graph

**A Graph-Guided Mamba Foundation Model for EEG Decoding**

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
- [Pretraining](#pretraining)
- [Finetuning](#finetuning)
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
                    |       Projection)          |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |                           |
                    |   Graph-Mamba Encoder     |
                    |   (N stacked GEM Blocks)  |
                    |                           |
                    |  +---------------------+  |
                    |  | Graph Attention Layer|  |  <-- GNN across channels per
                    |  | (Spatial: C dim)     |  |      time step using adjacency
                    |  +----------+----------+  |      from electrode topology
                    |             |              |
                    |             v              |
                    |  +---------------------+  |
                    |  | Mamba SSM Scan       |  |  <-- Mamba per channel for
                    |  | (Temporal: T dim)    |  |      temporal dynamics
                    |  +----------+----------+  |
                    |             |              |
                    |             v              |
                    |  +---------------------+  |
                    |  | Cross-Domain Fusion  |  |  <-- Gating mechanism to
                    |  | Gate                 |  |      integrate spatial &
                    |  +----------+----------+  |      temporal features
                    |             |              |
                    +-------------+--------------+
                                  |
                                  v
                    +---------------------------+
                    |     Projection Output      |
                    | (Task-Specific Head)       |
                    +---------------------------+
```

### Detailed Data Flow

```
Input EEG Signal: (B, C, T, P)
         |
         v
[Patch Embedding]
         |  Temporal Conv1D per channel per segment
         |  Spectral Projection (FFT-based features)
         v
Token Sequence: (B, C, T, D)    where D = embedding dimension
         |
         v
[GEM Block x N]
    |
    |--- [1] Graph Attention Layer ----------------+
    |         Input: (B, C, T, D)                  |
    |         Reshape to (B*T, C, D)               |
    |         Apply GAT with adjacency A           |
    |         A = A_prior + A_learned              |
    |         Output: (B, C, T, D) spatially       |
    |                 enriched                      |
    |                                              |
    |--- [2] Mamba SSM Scan -----------------------+
    |         Input: (B, C, T, D)                  |
    |         Reshape to (B*C, T, D)               |
    |         Bidirectional Mamba2 with            |
    |         graph-aware scan order               |
    |         Output: (B, C, T, D) temporally      |
    |                 enriched                      |
    |                                              |
    |--- [3] Cross-Domain Fusion Gate -------------+
              Z_spatial, Z_temporal                 
              gate = sigmoid(W_g * [Z_s; Z_t])     
              Output = gate * Z_s + (1-gate) * Z_t 
         |
         v
Final Representation: (B, C, T, D)
         |
         v
[Projection Output / Task Head]
```

---

## Key Novelties

### 1. Adaptive Graph Construction

MBG constructs the channel adjacency matrix by combining two complementary sources of structural information:

- **Prior Graph (A_prior):** Derived from the physical electrode distances in the International 10-20 system. Electrodes closer together on the scalp receive stronger prior connections. This encodes known anatomical proximity.

- **Learned Graph (A_learned):** An attention-based dynamic adjacency matrix that is computed per sample and per layer. This captures task-dependent functional connectivity that varies with cognitive state, stimulus, and individual differences.

```
A_effective = alpha * A_prior + (1 - alpha) * A_learned

where alpha is a learnable parameter per layer
```

This dual-graph approach captures both stable anatomical structure and dynamic functional relationships that change across samples and cognitive states.

### 2. Graph-Enhanced Mamba (GEM) Block

Each GEM block performs a three-stage computation:

1. **Graph Message Passing (Spatial):** Full-feature graph attention across the channel dimension at each time step. Unlike CBraMod, which splits features into two halves for spatial and temporal processing, GEM operates on the complete feature representation.

2. **Mamba Scan (Temporal):** Mamba2 SSM processes the temporal sequence per channel. After spatial enrichment from the graph layer, temporal modeling benefits from already-contextualized channel representations.

3. **Gating Fusion:** A learned gating mechanism integrates spatial and temporal features adaptively, allowing the model to weight the contribution of each domain per feature dimension.

This design ensures that spatial context informs temporal modeling (graph before Mamba) while maintaining both representations through gated fusion rather than simple addition or concatenation.

### 3. Multi-Scale Graph Pooling for Pretraining

During pretraining, MBG employs hierarchical graph pooling to create multi-resolution brain representations:

```
Level 0: Individual electrodes (e.g., 64 nodes)
Level 1: Brain sub-regions (e.g., 16 clusters: frontal-left, frontal-right, 
         temporal-left, temporal-right, central, parietal, occipital, ...)
Level 2: Brain hemispheres + midline (3 super-nodes)
Level 3: Global brain state (1 node)
```

This multi-scale hierarchy provides:
- Fine-grained electrode-level reconstruction targets
- Region-level contrastive objectives
- Global representation for downstream classification

### 4. Bidirectional Mamba with Graph-Aware Scan Order

Standard Mamba processes sequences in a fixed linear order. For EEG, a naive channel ordering (e.g., Fp1, Fp2, F3, F4, ...) does not reflect spatial proximity. MBG introduces **graph-aware scan ordering**:

- **BFS Order:** Breadth-first traversal from a seed electrode (e.g., Cz) creates a scan sequence where spatially adjacent electrodes are processed consecutively.
- **DFS Order:** Depth-first traversal captures hierarchical spatial paths from central to peripheral electrodes.
- **Bidirectional:** Forward and backward scans in both orderings provide comprehensive spatial coverage.

This makes the Mamba temporal scan implicitly aware of the spatial electrode topology, even before the explicit graph attention layer processes the signal.

---

## Comparison with Prior Works

| Aspect | CBraMod (ICLR 2025) | EEGMamba (Neural Networks 2025) | **MBG (Ours)** |
|--------|---------------------|--------------------------------|----------------|
| **Backbone** | Criss-Cross Transformer | Mamba2 SSM | Graph Attention + Mamba2 SSM |
| **Spatial Modeling** | Attention across channels (fully connected) | Implicit via flattened sequence | Explicit GNN with electrode topology |
| **Temporal Modeling** | Attention across time segments | Mamba2 over flattened sequence | Mamba2 per channel (after graph enrichment) |
| **Graph Structure** | Not used | Not used | Adaptive (prior + learned adjacency) |
| **Complexity** | O(C^2 + T^2) per block | O((C*T)) linear | O(C^2 + C*T) per block |
| **Spatial Inductive Bias** | None (learned from data) | None (positional embedding only) | Strong (electrode topology graph) |
| **Feature Processing** | Split features for spatial/temporal | Full features in single sequence | Full features in both, then gated fusion |
| **Scan Order** | N/A (attention-based) | Fixed linear (channel-then-time) | Graph-aware BFS/DFS ordering |
| **Multi-Scale** | Single resolution | Single resolution | Hierarchical graph pooling |
| **Pretraining Data** | TUEG | TUEG | TUEG |
| **Parameters** | ~5M | ~3M | ~6M (estimated) |

### Key Advantages of MBG

1. **Topology-Aware:** Explicitly leverages the known physical layout of EEG electrodes as a structural prior, reducing the amount of spatial structure that must be learned from data alone.

2. **Adaptive Connectivity:** The learned adjacency matrix discovers functional connections that go beyond physical proximity (e.g., inter-hemispheric coherence between homologous regions).

3. **Efficient Temporal Processing:** Retains Mamba's linear-time complexity for temporal modeling while adding spatial graph structure that is typically O(C^2) -- manageable given typical EEG channel counts (16-256).

4. **Complementary Fusion:** Rather than forcing a single mechanism to handle both spatial and temporal reasoning, MBG uses specialized modules for each and fuses them adaptively.

---

## Downstream Tasks

MBG supports evaluation on the following downstream benchmarks covering diverse EEG applications:

| Dataset | Task | Domain | Channels | Subjects |
|---------|------|--------|----------|----------|
| BCI Competition IV 2a | Motor Imagery Classification | BCI | 22 | 9 |
| TUAB | Abnormal Detection | Clinical | 21 | 2,993 |
| TUEV | Event Detection | Clinical | 21 | 313 |
| ISRUC | Sleep Staging | Sleep | 10 | 100 |
| SEED-V | Emotion Recognition (5-class) | Affective | 62 | 16 |
| CHB-MIT | Seizure Detection | Clinical | 23 | 22 |
| FACED | Emotion Recognition | Affective | 30 | 123 |
| SHU | Motor Imagery | BCI | 14 | 25 |
| Speech | Speech Imagery | BCI | 64 | 15 |
| Stress | Stress Detection | Affective | 14 | 28 |
| PhysioNet | Motor Imagery | BCI | 64 | 109 |
| SEED-VIG | Vigilance Estimation | Cognitive | 17 | 23 |
| Mumtaz | Depression Detection | Clinical | 19 | 63 |
| MODMA | Depression Detection | Clinical | 128 | 53 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/neuraldecoding/mbg.git
cd mbg

# Create conda environment
conda create -n mbg python=3.10
conda activate mbg

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
mamba-ssm>=1.2.0
torch-geometric>=2.4.0
numpy>=1.24.0
scipy>=1.10.0
einops>=0.7.0
tensorboard>=2.14.0
scikit-learn>=1.3.0
mne>=1.5.0
```

> **Note:** Installation instructions will be updated once the codebase is finalized.

---

## Pretraining

MBG is pretrained on the Temple University EEG Corpus (TUEG) using a masked patch prediction objective combined with multi-scale graph contrastive learning.

```bash
# Pretraining (placeholder)
python pretrain_main.py \
    --data_path /path/to/tueg \
    --output_dir ./pretrained_weights \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-4 \
    --num_layers 8 \
    --embed_dim 256 \
    --num_heads 8
```

> **Note:** Pretraining scripts and configurations will be released upon paper acceptance.

---

## Finetuning

After pretraining, MBG can be finetuned on any downstream task:

```bash
# Finetuning example (placeholder)
python finetune_main.py \
    --task tuab \
    --pretrained_path ./pretrained_weights/mbg_pretrained.pth \
    --data_path /path/to/tuab \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5
```

> **Note:** Finetuning scripts and task-specific configurations will be released upon paper acceptance.

---

## Project Structure

```
mbg/
├── models/
│   ├── mbg.py                    # Main MBG model
│   ├── graph_attention.py        # Graph Attention Network layers
│   ├── mamba_block.py            # Mamba2 SSM blocks
│   ├── gem_block.py              # Graph-Enhanced Mamba block
│   ├── graph_construction.py     # Adaptive adjacency matrix
│   ├── graph_pooling.py          # Multi-scale graph pooling
│   └── model_for_*.py            # Task-specific heads
├── datasets/
│   └── *_dataset.py              # Dataset loaders
├── preprocessing/
│   └── preprocessing_*.py        # Data preprocessing scripts
├── utils/
│   ├── graph_utils.py            # Graph construction utilities
│   ├── electrode_positions.py    # 10-20 system coordinates
│   └── util.py                   # General utilities
├── pretrain_main.py              # Pretraining entry point
├── finetune_main.py              # Finetuning entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

> **Note:** Project structure will be populated as development progresses.

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{mbg2025,
  title={MBG: A Graph-Guided Mamba Foundation Model for EEG Decoding},
  author={},
  journal={},
  year={2025}
}
```

---

## Acknowledgements

This work builds upon the following prior works from our research group:

- **CBraMod:** Yi, Z., et al. "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding." *ICLR 2025*.
- **EEGMamba:** Yi, Z., et al. "EEGMamba: Bidirectional Mamba with Cross-Domain Transfer Learning for EEG Foundation Model." *Neural Networks 2025*.

We thank the developers of [Mamba](https://github.com/state-spaces/mamba), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), and [MNE-Python](https://mne.tools/) for their excellent open-source tools.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
