"""
Graph-Enhanced Mamba (GEM) Block.
Combines spatial graph attention, temporal Mamba processing, and cross-domain fusion gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .graph_attention import GraphAttentionLayer
from modules.mixer_seq_simple import MixerModel
from modules.config_mamba import MambaConfig


class GEMBlock(nn.Module):
    """
    Graph-Enhanced Mamba Block.

    Architecture:
        1. Graph Attention (spatial across channels)
        2. Mamba SSM scan (temporal per channel, bidirectional)
        3. Cross-Domain Fusion Gate: gate = sigmoid(W_g * [Z_spatial; Z_temporal])
           output = gate * Z_spatial + (1 - gate) * Z_temporal
        4. Residual connection + FFN

    Input/Output: (B, C, T, D)
    """

    def __init__(self, d_model, nhead, dim_feedforward, n_layer_mamba=1,
                 num_channels=22, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Spatial: Graph Attention
        self.graph_attention = GraphAttentionLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            num_channels=num_channels,
        )

        # Temporal: Mamba per channel (bidirectional via MixerModel)
        config = MambaConfig()
        config.d_model = d_model
        config.n_layer = n_layer_mamba
        config.fused_add_norm = False
        config.rms_norm = False
        config.residual_in_fp32 = False

        self.mamba_temporal = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=None,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
        )

        # Cross-Domain Fusion Gate
        self.fusion_gate = nn.Linear(2 * d_model, d_model)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T, D)

        Returns:
            x: (B, C, T, D)
        """
        B, C, T, D = x.shape

        # Pre-norm
        z = self.norm1(x)

        # Spatial: Graph Attention across channels
        z_spatial = self.graph_attention(z)  # (B, C, T, D)

        # Temporal: Mamba scan per channel
        z_norm = self.norm2(x)
        z_temporal_in = rearrange(z_norm, 'b c t d -> (b c) t d')
        z_temporal_out = self.mamba_temporal(z_temporal_in)  # (B*C, T, D)
        # Compensate for the odd-layer bidirectional flip in MixerModel (n_layer_mamba=1
        # flips once, leaving the output time-reversed relative to the input).
        z_temporal_out = z_temporal_out.flip(1)
        z_temporal = rearrange(z_temporal_out, '(b c) t d -> b c t d', b=B, c=C)

        # Cross-Domain Fusion Gate
        gate_input = torch.cat([z_spatial, z_temporal], dim=-1)  # (B, C, T, 2D)
        gate = torch.sigmoid(self.fusion_gate(gate_input))  # (B, C, T, D)
        fused = gate * z_spatial + (1 - gate) * z_temporal  # (B, C, T, D)

        # Residual connection
        x = x + fused

        # FFN with residual
        x = x + self.ffn(self.norm3(x))

        return x
