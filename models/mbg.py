"""
MBG: Mamba-Based Graph model for EEG signal processing.
Combines Graph Attention (spatial) with Mamba SSM (temporal) via a fusion gate.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.gem_block import GEMBlock


class MBG(nn.Module):
    """
    MBG (Mamba-Based Graph) foundation model for EEG.

    Architecture:
        PatchEmbedding -> N x GEM Blocks -> proj_out

    Args:
        in_dim: Input patch dimension (points per patch).
        out_dim: Output dimension per patch.
        d_model: Hidden dimension.
        dim_feedforward: FFN intermediate dimension.
        seq_len: Maximum sequence length (number of time segments).
        n_layer: Number of GEM blocks.
        nhead: Number of attention heads.
        num_channels: Number of EEG channels (for graph construction).
    """

    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                 seq_len=30, n_layer=12, nhead=8, num_channels=22):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.num_channels = num_channels

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        # Encoder: stack of GEM blocks
        self.encoder = nn.ModuleList([
            GEMBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                n_layer_mamba=1,
                num_channels=num_channels,
                dropout=0.1,
            )
            for _ in range(n_layer)
        ])

        # Output projection
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

        self.apply(_weights_init)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T, P) - batch, channels, time_segments, points_per_patch
            mask: Optional mask for patch embedding.

        Returns:
            out: (B, C, T, out_dim)
        """
        bz, ch_num, seq_len, patch_size = x.shape

        # Patch embedding: (B, C, T, P) -> (B, C, T, D)
        hidden_states = self.patch_embedding(x, mask=mask)

        # GEM blocks: (B, C, T, D) -> (B, C, T, D)
        for gem_block in self.encoder:
            hidden_states = gem_block(hidden_states)

        # Output projection: (B, C, T, D) -> (B, C, T, out_dim)
        out = self.proj_out(hidden_states)

        return out


class PatchEmbedding(nn.Module):
    """
    Patch embedding with temporal Conv2d + spectral FFT projection + positional Conv2d encoding.
    Same pattern as EEGMamba's PatchEmbedding.
    """

    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        # Positional encoding via depthwise Conv2d
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model,
                      kernel_size=(7, 7), stride=(1, 1), padding=(3, 3),
                      groups=d_model, bias=False),
        )

        # Mask token encoding
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        # Temporal projection: Conv2d on raw signal
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25,
                      kernel_size=(1, 49), stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )

        # Spectral projection: FFT magnitude -> linear projection
        self.spectral_proj = nn.Sequential(
            nn.Linear(in_dim // 2 + 1, d_model, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T, P) where P = in_dim (points per patch)
            mask: Optional mask tensor

        Returns:
            patch_emb: (B, C, T, D)
        """
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # Temporal embedding via Conv2d
        # Reshape: (B, C, T, P) -> (B, P, C, T) -> (B, C*T, P) with unsqueeze for Conv2d
        mask_x_t = rearrange(mask_x, 'b c l d -> b d c l')
        time_x = rearrange(mask_x_t, 'b d c l -> b (c l) d').unsqueeze(1)  # (B, 1, C*T, P)

        time_emb = self.proj_in(time_x)  # (B, 25, C*T, 8) approx with kernel/stride
        # Reshape to (B, C, T, d_model): permute channels to features
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        # Spectral embedding via FFT
        freq_x = rearrange(mask_x_t, 'b d c l -> b c l d')
        spectral = torch.fft.rfft(freq_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral)  # (B, C, T, P//2+1)
        spectral_emb = self.spectral_proj(spectral)  # (B, C, T, D)

        # Combine temporal and spectral embeddings
        patch_emb = time_emb + spectral_emb

        # Positional encoding via Conv2d
        # (B, C, T, D) -> (B, D, C, T)
        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)  # (B, C, T, D)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = MBG(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30,
        n_layer=4, nhead=8, num_channels=22
    )
    x = torch.randn(2, 22, 4, 200)
    y = model(x)
    print(f'Input shape:  {x.shape}')
    print(f'Output shape: {y.shape}')
    assert y.shape == (2, 22, 4, 200), f'Expected (2, 22, 4, 200), got {y.shape}'
    print('Forward pass test PASSED')
