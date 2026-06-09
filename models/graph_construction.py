"""
Adaptive Adjacency Matrix construction for Graph-Enhanced Mamba.
Combines a fixed prior (from electrode distances) with a learned dynamic adjacency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.electrode_positions import get_electrode_positions, compute_distance_matrix
from utils.graph_utils import gaussian_adjacency


def get_prior_adjacency(num_channels, sigma=1.0):
    """
    Build the fixed prior adjacency matrix from electrode physical distances.

    Args:
        num_channels: Number of EEG channels.
        sigma: Gaussian kernel standard deviation.

    Returns:
        numpy.ndarray: NxN prior adjacency matrix.
    """
    positions = get_electrode_positions(num_channels)
    dist_matrix = compute_distance_matrix(positions)
    adj = gaussian_adjacency(dist_matrix, sigma=sigma)
    return adj


class AdaptiveAdjacency(nn.Module):
    """
    Adaptive adjacency matrix: A = alpha * A_prior + (1 - alpha) * A_learned.

    A_prior: Fixed adjacency from electrode physical distances (Gaussian kernel).
    A_learned: Dynamic attention-based adjacency computed per sample.
    alpha: Learnable parameter per layer (initialized to 0.5).
    """

    def __init__(self, num_channels, d_model, sigma=1.0):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model

        # Prior adjacency from electrode distances
        prior_adj = get_prior_adjacency(num_channels, sigma=sigma)
        self.register_buffer('A_prior', torch.from_numpy(prior_adj).float())

        # Learnable blending parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Projections for computing dynamic adjacency
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** 0.5

    def forward(self, x):
        """
        Compute adaptive adjacency matrix.

        Args:
            x: (B*T, C, D) - node features for each graph instance.

        Returns:
            A: (B*T, C, C) - adaptive adjacency matrices.

        Note:
            The full (B*T, C, C) attention matrix is recomputed on every forward pass.
            At large batch sizes (e.g. B=64, T=30, C=22) this produces ~130 MB of
            intermediates per layer. Consider caching or chunked computation for
            training with large batches if memory becomes a bottleneck.
        """
        # Compute dynamic adjacency via attention
        Q = self.W_q(x)  # (B*T, C, D)
        K = self.W_k(x)  # (B*T, C, D)
        A_learned = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B*T, C, C)
        A_learned = F.softmax(A_learned, dim=-1)

        # Blend prior and learned adjacency
        alpha = torch.sigmoid(self.alpha)
        A = alpha * self.A_prior.unsqueeze(0) + (1 - alpha) * A_learned

        return A
