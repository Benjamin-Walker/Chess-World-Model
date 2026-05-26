from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm as OfficialRMSNorm
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn

from .factory import canonicalise_mamba_variant
from .transformer import RMSNorm


def _init_weights(
    module: nn.Module,
    *,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
) -> None:
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if not rescale_prenorm_residual:
        return

    for name, parameter in module.named_parameters():
        if name not in {"out_proj.weight", "fc2.weight"}:
            continue
        nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        with torch.no_grad():
            parameter /= math.sqrt(n_residuals_per_layer * n_layer)


class ChessMambaStateModel(nn.Module):
    _HEAD_SIZES = {
        "pieces": 64 * 13,
        "side": 2,
        "castle": 4 * 2,
        "ep_file": 9,
        "ep_rank": 3,
        "halfmove": 2 * 256,
        "fullmove": 2 * 256,
    }

    def __init__(
        self,
        move_vocab: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        *,
        variant: str = "mamba-3",
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        d_intermediate: int | None = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = False,
        use_fast_path: bool = True,
        use_mem_eff_path: bool = True,
        chunk_size: int | None = None,
        norm_eps: float = 1e-5,
        rope_fraction: float = 0.5,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.move_vocab = int(move_vocab)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.variant = canonicalise_mamba_variant(variant)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.headdim = int(headdim)
        self.ngroups = int(ngroups)
        self.d_intermediate = (
            4 * self.d_model if d_intermediate is None else int(d_intermediate)
        )
        self.dropout_p = float(dropout)
        self.rms_norm = bool(rms_norm)
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.fused_add_norm = bool(fused_add_norm)
        self.use_fast_path = bool(use_fast_path)
        self.use_mem_eff_path = bool(use_mem_eff_path)
        self.norm_eps = float(norm_eps)
        self.rope_fraction = float(rope_fraction)
        self.is_outproj_norm = bool(is_outproj_norm)
        self.is_mimo = bool(is_mimo)
        self.mimo_rank = int(mimo_rank)

        if chunk_size is None:
            if self.variant == "mamba-2":
                chunk_size = 256
            elif self.variant == "mamba-3":
                chunk_size = max(1, 64 // self.mimo_rank) if self.is_mimo else 64
        self.chunk_size = None if chunk_size is None else int(chunk_size)

        if self.rms_norm:
            norm_impl = OfficialRMSNorm if self.fused_add_norm else RMSNorm
            norm_cls = partial(norm_impl, eps=self.norm_eps)
        else:
            norm_cls = partial(nn.LayerNorm, eps=self.norm_eps)

        self.embedding = nn.Embedding(self.move_vocab, self.d_model)
        self.embedding_dropout = nn.Dropout(self.dropout_p)
        self.layers = nn.ModuleList()

        if self.variant == "mamba-1":
            from mamba_ssm.modules.mamba_simple import Mamba as mixer_impl
        elif self.variant == "mamba-2":
            from mamba_ssm.modules.mamba2 import Mamba2 as mixer_impl
        else:
            try:
                from mamba_ssm.modules.mamba3 import Mamba3 as mixer_impl
            except ImportError as exc:
                raise ImportError(
                    "Mamba-3 requires a source-built `mamba-ssm` install that exposes "
                    "`mamba_ssm.modules.mamba3`. If `uv sync --group mamba` leaves a "
                    "build without `mamba3.py`, reinstall the pinned source build "
                    "described in `README.md`."
                ) from exc

        for layer_idx in range(self.num_layers):
            if self.variant == "mamba-1":
                mixer_cls = partial(
                    mixer_impl,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    use_fast_path=self.use_fast_path,
                    layer_idx=layer_idx,
                )
            elif self.variant == "mamba-2":
                mixer_cls = partial(
                    mixer_impl,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    chunk_size=self.chunk_size,
                    use_mem_eff_path=self.use_mem_eff_path,
                    layer_idx=layer_idx,
                )
            else:
                mixer_cls = partial(
                    mixer_impl,
                    d_state=self.d_state,
                    expand=self.expand,
                    headdim=self.headdim,
                    ngroups=self.ngroups,
                    chunk_size=self.chunk_size,
                    rope_fraction=self.rope_fraction,
                    is_outproj_norm=self.is_outproj_norm,
                    is_mimo=self.is_mimo,
                    mimo_rank=self.mimo_rank,
                    dropout=self.dropout_p,
                    layer_idx=layer_idx,
                    n_layer=self.num_layers,
                )

            mlp_cls = nn.Identity
            if self.d_intermediate > 0:
                mlp_cls = partial(
                    GatedMLP,
                    hidden_features=self.d_intermediate,
                    out_features=self.d_model,
                )

            layer = Block(
                self.d_model,
                mixer_cls,
                mlp_cls,
                norm_cls=norm_cls,
                fused_add_norm=self.fused_add_norm,
                residual_in_fp32=self.residual_in_fp32,
            )
            layer.layer_idx = layer_idx
            self.layers.append(layer)

        self.final_norm = norm_cls(self.d_model)
        self.output_head = nn.Linear(self.d_model, sum(self._HEAD_SIZES.values()))

        self.apply(
            partial(
                _init_weights,
                n_layer=self.num_layers,
                n_residuals_per_layer=1 if self.d_intermediate == 0 else 2,
            )
        )

    def forward(self, moves: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, seq_len = moves.shape
        if self.variant == "mamba-3" and self.is_mimo and self.chunk_size is not None:
            pad_tokens = (-seq_len) % self.chunk_size
            if pad_tokens:
                moves = F.pad(moves, (0, pad_tokens), value=0)

        hidden_states = self.embedding_dropout(self.embedding(moves))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if self.fused_add_norm:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.final_norm.weight,
                getattr(self.final_norm, "bias", None),
                eps=self.final_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=self.rms_norm,
            )
        else:
            residual = (
                hidden_states + residual if residual is not None else hidden_states
            )
            hidden_states = self.final_norm(
                residual.to(dtype=self.final_norm.weight.dtype)
            )

        logits = self.output_head(hidden_states)
        if logits.shape[1] != seq_len:
            logits = logits[:, :seq_len, :]

        pieces, side, castle, ep_file, ep_rank, halfmove, fullmove = torch.split(
            logits,
            list(self._HEAD_SIZES.values()),
            dim=-1,
        )
        return {
            "pieces": pieces.view(bsz, seq_len, 64, 13),
            "side": side,
            "castle": castle.view(bsz, seq_len, 4, 2),
            "ep_file": ep_file,
            "ep_rank": ep_rank,
            "halfmove": halfmove.view(bsz, seq_len, 2, 256),
            "fullmove": fullmove.view(bsz, seq_len, 2, 256),
        }
