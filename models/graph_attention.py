"""
Multi-head Graph Attention Layer (pure PyTorch, no PyG dependency).
Implements standard GAT mechanism with adjacency masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .graph_construction import AdaptiveAdjacency


class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention Network layer with adaptive adjacency masking.

    GAT mechanism per head:
        e_ij = LeakyReLU(a^T [W*h_i || W*h_j])
        alpha_ij = softmax_j(e_ij) * A_ij  (masked by adjacency)
        h_i' = sum_j(alpha_ij * W*h_j)

    Input: (B, C, T, D)
    Output: (B, C, T, D)
    """

    def __init__(self, d_model, nhead, dropout=0.1, num_channels=22):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Adaptive adjacency matrix
        self.adjacency = AdaptiveAdjacency(num_channels, d_model)

        # Per-head linear projections
        self.W = nn.Linear(d_model, d_model, bias=False)

        # Attention parameters: one vector per head for [Wh_i || Wh_j] concatenation
        self.attn_src = nn.Parameter(torch.zeros(1, nhead, 1, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(1, nhead, 1, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src.view(nhead, -1))
        nn.init.xavier_uniform_(self.attn_dst.view(nhead, -1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, D)

        Returns:
            out: (B, C, T, D)
        """
        B, C, T, D = x.shape

        # Reshape to (B*T, C, D) for graph operations
        x_graph = rearrange(x, 'b c t d -> (b t) c d')

        # Compute adaptive adjacency: (B*T, C, C)
        A = self.adjacency(x_graph)

        # Linear projection
        h = self.W(x_graph)  # (B*T, C, D)

        # Reshape for multi-head: (B*T, C, nhead, head_dim) -> (B*T, nhead, C, head_dim)
        h_heads = rearrange(h, 'bt c (nh hd) -> bt nh c hd', nh=self.nhead)

        # Compute attention scores: e_ij = LeakyReLU(a_src^T * h_i + a_dst^T * h_j)
        # Score from source nodes
        e_src = (h_heads * self.attn_src).sum(dim=-1, keepdim=True)  # (B*T, nhead, C, 1)
        # Score from destination nodes
        e_dst = (h_heads * self.attn_dst).sum(dim=-1, keepdim=True)  # (B*T, nhead, C, 1)

        # Pairwise attention: (B*T, nhead, C, C)
        e = e_src + e_dst.transpose(-2, -1)
        e = self.leaky_relu(e)

        # Mask with adjacency: set non-connected edges to -inf
        # Expand adjacency for heads: (B*T, 1, C, C)
        A_mask = A.unsqueeze(1)
        # Apply mask: where adjacency is near zero, set attention to -inf
        e = e.masked_fill(A_mask < 1e-6, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(e, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle isolated nodes
        attn = self.dropout(attn)

        # Aggregate: (B*T, nhead, C, head_dim)
        out = torch.matmul(attn, h_heads)

        # Concatenate heads: (B*T, C, D)
        out = rearrange(out, 'bt nh c hd -> bt c (nh hd)')
        out = self.out_proj(out)

        # Reshape back to (B, C, T, D)
        out = rearrange(out, '(b t) c d -> b c t d', b=B, t=T)

        return out
