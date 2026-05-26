from __future__ import annotations

import argparse

import torch

from .slice import ChessSLiCEStateModel
from .transformer import ChessCausalTransformer

_MAMBA_VARIANT_ALIASES = {
    "mamba1": "mamba-1",
    "mamba2": "mamba-2",
    "mamba3": "mamba-3",
}


def canonicalise_mamba_variant(variant: str) -> str:
    return _MAMBA_VARIANT_ALIASES.get(str(variant), str(variant))


def resolve_gated_deltanet_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "ff_mult": int(getattr(args, "ff_mult", 4)),
        "head_dim": int(getattr(args, "gated_deltanet_head_dim", 48)),
        "expand_v": float(getattr(args, "gated_deltanet_expand_v", 2.0)),
        "use_gate": bool(getattr(args, "gated_deltanet_use_gate", True)),
        "use_short_conv": bool(getattr(args, "gated_deltanet_use_short_conv", True)),
        "allow_neg_eigval": bool(
            getattr(args, "gated_deltanet_allow_neg_eigval", True)
        ),
        "conv_size": int(getattr(args, "gated_deltanet_conv_size", 4)),
        "attn_mode": str(getattr(args, "gated_deltanet_attn_mode", "chunk")),
        "norm_eps": float(getattr(args, "gated_deltanet_norm_eps", 1e-6)),
    }


def resolve_mamba_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    variant = canonicalise_mamba_variant(getattr(args, "mamba_variant", "mamba-3"))
    d_state = getattr(args, "mamba_d_state", None)
    if d_state is None:
        d_state = 64 if variant == "mamba-1" else 128
    d_intermediate = getattr(args, "mamba_d_intermediate", None)
    if d_intermediate is None:
        d_intermediate = 4 * int(args.d_model)
    is_mimo = bool(getattr(args, "mamba3_is_mimo", False))
    mimo_rank = int(getattr(args, "mamba3_mimo_rank", 4))

    chunk_size = getattr(args, "mamba_chunk_size", None)
    if chunk_size is None:
        if variant == "mamba-2":
            chunk_size = 256
        elif variant == "mamba-3":
            chunk_size = max(1, 64 // mimo_rank) if is_mimo else 64

    return {
        "variant": variant,
        "d_state": int(d_state),
        "d_conv": int(getattr(args, "mamba_d_conv", 4)),
        "expand": int(getattr(args, "mamba_expand", 2)),
        "headdim": int(getattr(args, "mamba_headdim", 64)),
        "ngroups": int(getattr(args, "mamba_ngroups", 1)),
        "d_intermediate": int(d_intermediate),
        "rms_norm": bool(getattr(args, "mamba_rms_norm", True)),
        "residual_in_fp32": bool(getattr(args, "mamba_residual_in_fp32", True)),
        "fused_add_norm": bool(getattr(args, "mamba_fused_add_norm", False)),
        "use_fast_path": bool(getattr(args, "mamba_use_fast_path", True)),
        "use_mem_eff_path": bool(getattr(args, "mamba_use_mem_eff_path", True)),
        "chunk_size": None if chunk_size is None else int(chunk_size),
        "norm_eps": float(getattr(args, "mamba_norm_eps", 1e-5)),
        "rope_fraction": float(getattr(args, "mamba3_rope_fraction", 0.5)),
        "is_outproj_norm": bool(getattr(args, "mamba3_is_outproj_norm", False)),
        "is_mimo": is_mimo,
        "mimo_rank": mimo_rank,
    }


def build_world_model(
    args: argparse.Namespace,
    *,
    move_vocab: int,
    device: torch.device,
    pad_move_id: int | None = None,
) -> torch.nn.Module:
    if args.arch == "transformer":
        return ChessCausalTransformer(
            move_vocab=move_vocab,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            ff_mult=args.ff_mult,
        ).to(device)

    if args.arch == "slice":
        return ChessSLiCEStateModel(
            move_vocab=move_vocab,
            d_model=args.d_model,
            num_layers=args.n_layers,
            dropout=args.dropout,
            use_parallel=args.slice_use_parallel,
            chunk_size=args.slice_chunk_size,
            block_size=args.slice_block_size,
            diagonal_dense=args.slice_diagonal_dense,
            init_std=args.slice_init_std,
            scale=args.slice_scale,
            input_dependent_init=args.slice_input_dependent_init,
            ff_mult=args.ff_mult,
            path_mode=args.slice_path_mode,
            norm_type=args.slice_norm_type,
            ff_style=args.slice_ff_style,
            ff_activation=args.slice_ff_activation,
            dropout_position=args.slice_dropout_position,
            norm_eps=args.slice_norm_eps,
            final_norm=args.slice_final_norm,
        ).to(device)

    if args.arch == "gated_deltanet":
        from .gated_deltanet import ChessGatedDeltaNetStateModel

        return ChessGatedDeltaNetStateModel(
            move_vocab=move_vocab,
            d_model=args.d_model,
            num_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            pad_move_id=pad_move_id,
            **resolve_gated_deltanet_model_kwargs(args),
        ).to(device)

    if args.arch == "mamba":
        from .mamba import ChessMambaStateModel

        return ChessMambaStateModel(
            move_vocab=move_vocab,
            d_model=args.d_model,
            num_layers=args.n_layers,
            dropout=args.dropout,
            **resolve_mamba_model_kwargs(args),
        ).to(device)

    raise ValueError(f"Unsupported arch='{args.arch}'.")
