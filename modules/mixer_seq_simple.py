# Mamba MixerModel with fallback for CPU environments without mamba-ssm.
# Based on: https://github.com/state-spaces/mamba

import math
from functools import partial
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.config_mamba import MambaConfig

# Try to import mamba_ssm; fall back gracefully if unavailable (e.g., no CUDA)
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.modules.mha import MHA
    from mamba_ssm.modules.mlp import GatedMLP
    from mamba_ssm.modules.block import Block
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class SimpleMambaFallback(nn.Module):
    """
    Fallback mixer when mamba-ssm is not available.
    Mimics the same interface: forward(hidden_states) -> hidden_states
    Uses a causal Conv1d with kernel_size=4 to provide temporal receptive field,
    followed by GELU activation and a linear projection.
    """
    def __init__(self, d_model, layer_idx=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        kernel_size = 4
        # Causal Conv1d: pad on the left only so output depends only on past/current
        self.causal_pad = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=1, bias=True)
        self.activation = nn.GELU()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, inference_params=None, **kwargs):
        # hidden_states: (B, T, D)
        x = hidden_states.transpose(1, 2)  # (B, D, T)
        # Causal padding: pad left side only
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)  # (B, D, T)
        x = x.transpose(1, 2)  # (B, T, D)
        x = self.activation(x)
        x = self.proj(x)
        return x


class FallbackBlock(nn.Module):
    """
    Fallback block when mamba-ssm Block is not available.
    Implements: residual + LayerNorm + Mixer pattern.
    """
    def __init__(self, d_model, mixer_cls, norm_cls=None, **kwargs):
        super().__init__()
        self.mixer = mixer_cls(d_model)
        self.norm = norm_cls(d_model) if norm_cls is not None else nn.LayerNorm(d_model)

    def forward(self, hidden_states, residual=None, inference_params=None, **kwargs):
        # Pre-norm architecture
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(
    d_model,
    d_intermediate=0,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if MAMBA_AVAILABLE:
        if layer_idx not in attn_layer_idx:
            ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
            ssm_layer = ssm_cfg.pop("layer", "Mamba1")
            if ssm_layer not in ["Mamba1", "Mamba2"]:
                raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
            mixer_cls = partial(
                Mamba2 if ssm_layer == "Mamba2" else Mamba,
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs
            )
        else:
            mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        if d_intermediate == 0:
            mlp_cls = nn.Identity
        else:
            mlp_cls = partial(
                GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
            )
        block = Block(
            d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
    else:
        # Fallback: use SimpleMambaFallback
        mixer_cls = partial(SimpleMambaFallback, layer_idx=layer_idx)
        norm_cls = partial(nn.LayerNorm, eps=norm_epsilon)
        block = FallbackBlock(d_model, mixer_cls, norm_cls=norm_cls)

    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int = 0,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            # Bidirectional flip: reverse the sequence after each layer
            hidden_states = hidden_states.flip(1)
            residual = residual.flip(1)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
