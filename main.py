import argparse
import json
import os
from typing import Optional

from data.loading import (
    find_shards,
    load_spec_from_first_line,
)
from models.factory import canonicalise_mamba_variant
from training import CHESS_DIR, train


# ----------------------------
# CLI
# ----------------------------

DEFAULT_EXPERIMENT_CONFIG = {
    "data_path": os.path.join(CHESS_DIR, "data", "processed"),
    "batch_size": 16,
    "num_workers": 0,
    "max_steps": None,
    "max_train_steps": None,
    "d_model": 384,
    "n_layers": 6,
    "n_heads": 6,
    "dropout": 0.1,
    "ff_mult": 4,
    "lr": 3e-4,
    "beta2": 0.95,
    "lr_scheduler": "none",
    "lr_warmup_steps": 0,
    "lr_decay_steps": 0,
    "lr_min_ratio": 0.1,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "epochs": 1,
    "seed": 0,
    "ckpt_path": os.path.join(CHESS_DIR, "checkpoints"),
    "resume": None,
    "save_every": 20000,
    "log_every": 500,
    "overfit_iters": 300,
    "val_fraction": 0.005,
    "split_mod": 10000,
    "val_every": 20000,
    "val_batches": 10,
    "val_batch_size": None,
    "val_num_workers": 0,
    "bin_size": 20,
    "arch": "transformer",
    "slice_use_parallel": True,
    "slice_chunk_size": 256,
    "slice_block_size": 4,
    "slice_diagonal_dense": False,
    "slice_init_std": 0.01,
    "slice_scale": 1.0 / 200,
    "slice_input_dependent_init": False,
    "slice_path_mode": "values",
    "slice_norm_type": "rmsnorm",
    "slice_ff_style": "mlp",
    "slice_ff_activation": "gelu",
    "slice_dropout_position": "residual",
    "slice_norm_eps": 1e-6,
    "slice_final_norm": False,
    "gated_deltanet_head_dim": 48,
    "gated_deltanet_expand_v": 2.0,
    "gated_deltanet_use_gate": True,
    "gated_deltanet_use_short_conv": True,
    "gated_deltanet_allow_neg_eigval": True,
    "gated_deltanet_conv_size": 4,
    "gated_deltanet_attn_mode": "chunk",
    "gated_deltanet_norm_eps": 1e-6,
    "mamba_variant": "mamba-3",
    "mamba_d_state": None,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "mamba_headdim": 64,
    "mamba_ngroups": 1,
    "mamba_d_intermediate": None,
    "mamba_rms_norm": True,
    "mamba_residual_in_fp32": True,
    "mamba_fused_add_norm": False,
    "mamba_use_fast_path": True,
    "mamba_use_mem_eff_path": True,
    "mamba_chunk_size": None,
    "mamba_norm_eps": 1e-5,
    "mamba3_rope_fraction": 0.5,
    "mamba3_is_outproj_norm": False,
    "mamba3_is_mimo": False,
    "mamba3_mimo_rank": 4,
    "nan_guard": True,
    "nan_restore": True,
    "nan_snapshot_every": 500,
    "nan_max_consecutive_bad_batches": 20,
    "nan_check_params_after_step": True,
    "wandb_project": "world-modelling",
    "wandb_entity": "",
    "wandb_run_name": "",
    "wandb_mode": "online",
}


def resolve_experiment_config_path(config_name_or_path: str) -> str:
    if os.path.isfile(config_name_or_path):
        return config_name_or_path
    if not config_name_or_path.endswith(".json"):
        config_name_or_path = f"{config_name_or_path}.json"
    cfg_path = os.path.join(
        os.path.dirname(__file__), "experiment_configs", config_name_or_path
    )
    if os.path.isfile(cfg_path):
        return cfg_path
    raise FileNotFoundError(
        "Could not find experiment config "
        f"'{config_name_or_path}' as a file path or in "
        "experiment_configs/."
    )


def load_experiment_config(config_name_or_path: Optional[str]) -> dict[str, object]:
    if not config_name_or_path:
        return dict(DEFAULT_EXPERIMENT_CONFIG)
    cfg_path = resolve_experiment_config_path(config_name_or_path)
    with open(cfg_path, encoding="utf-8") as f:
        loaded = json.load(f)
    cfg = dict(DEFAULT_EXPERIMENT_CONFIG)
    cfg.update(loaded)
    return cfg


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--experiment_config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_experiment_config(pre_args.experiment_config)

    p = argparse.ArgumentParser("Chess state-tracking trainer (moves -> states)")
    p.set_defaults(**cfg)

    p.add_argument(
        "--experiment_config",
        type=str,
        default=pre_args.experiment_config,
        help=(
            "JSON config file path, or name under experiment_configs (without .json)."
        ),
    )

    p.add_argument(
        "--data_path",
        type=str,
        help=(
            "Shard file, directory, or prefix for chess_*.jsonl "
            "(default: data/processed)."
        ),
    )
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument(
        "--max_steps",
        type=int,
        help="Optional truncation of sequence length (includes START step).",
    )
    p.add_argument(
        "--max_train_steps",
        type=int,
        help="Optional hard cap on optimiser steps across all epochs.",
    )

    p.add_argument("--d_model", type=int)
    p.add_argument("--n_layers", type=int)
    p.add_argument("--n_heads", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--ff_mult", type=int)

    p.add_argument("--lr", type=float)
    p.add_argument(
        "--beta2",
        type=float,
        help="AdamW beta2 coefficient.",
    )
    p.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["none", "cosine", "linear"],
        help="Per-step LR scheduler. 'none' keeps LR constant.",
    )
    p.add_argument(
        "--lr_warmup_steps",
        type=int,
        help="Linear warmup steps at the start of training.",
    )
    p.add_argument(
        "--lr_decay_steps",
        type=int,
        help=(
            "Total scheduler horizon in optimiser steps. "
            "Required when --lr_scheduler is not 'none'."
        ),
    )
    p.add_argument(
        "--lr_min_ratio",
        type=float,
        help=(
            "Minimum LR ratio reached at the end of decay "
            "(e.g. 0.1 => 10%% of base LR)."
        ),
    )
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--grad_clip", type=float)

    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)

    p.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available."
    )

    p.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint directory (recommended) or a .pt file path.",
    )
    p.add_argument(
        "--resume", type=str, help="Path to a checkpoint .pt to resume from."
    )
    p.add_argument("--save_every", type=int)
    p.add_argument("--log_every", type=int)
    p.add_argument(
        "--no_shuffle_shards",
        action="store_true",
        help="Disable per-epoch shuffling of shard order.",
    )

    p.add_argument(
        "--overfit_one_batch",
        action="store_true",
        help="Debug mode: repeatedly train on a single batch to sanity-check learning.",
    )
    p.add_argument("--overfit_iters", type=int)

    # Validation split + metrics
    p.add_argument(
        "--val_fraction",
        type=float,
        help="Fraction of games reserved for validation, by stable hash of example_id.",
    )
    p.add_argument(
        "--split_mod", type=int, help="Hash bucket modulus for train/val split."
    )
    p.add_argument(
        "--val_every",
        type=int,
        help="Run validation every N training steps. Set 0 to disable.",
    )
    p.add_argument(
        "--val_batches",
        type=int,
        help=(
            "Number of validation batches per run. "
            "Set 0 to iterate all val in the shard."
        ),
    )
    p.add_argument(
        "--val_batch_size",
        type=int,
        help="Optional separate batch size for validation.",
    )
    p.add_argument(
        "--val_num_workers", type=int, help="Num workers for validation loader."
    )
    p.add_argument(
        "--bin_size",
        type=int,
        help="Bin size for metrics by timestep: 0-20, 20-40, etc.",
    )

    p.add_argument(
        "--arch",
        type=str,
        choices=["transformer", "slice", "gated_deltanet", "mamba"],
        help=(
            "Backbone architecture: causal Transformer, SLiCELayer-based SLiCE, "
            "FLA Gated DeltaNet, or Mamba."
        ),
    )

    # SLiCE options (used when --arch slice)
    p.add_argument(
        "--slice_use_parallel",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable associative-scan parallel execution in SLiCE.",
    )
    p.add_argument(
        "--slice_chunk_size", type=int, help="Chunk size for SLiCE parallel mode."
    )
    p.add_argument("--slice_block_size", type=int)
    p.add_argument("--slice_diagonal_dense", action="store_true")
    p.add_argument("--slice_init_std", type=float)
    p.add_argument("--slice_scale", type=float)
    p.add_argument("--slice_input_dependent_init", action="store_true")
    p.add_argument(
        "--slice_path_mode",
        type=str,
        choices=["values", "increments"],
        help=(
            "Driving-path semantics for each SLiCELayer. "
            "'values' lets SLiCE difference the sequence internally; "
            "'increments' expects pre-differenced inputs."
        ),
    )
    p.add_argument(
        "--slice_norm_type",
        type=str,
        choices=["rmsnorm", "layernorm"],
        help="Normalisation used inside each SLiCELayer wrapper.",
    )
    p.add_argument(
        "--slice_ff_style",
        type=str,
        choices=["mlp", "single"],
        help="Feedforward branch shape inside each SLiCELayer.",
    )
    p.add_argument(
        "--slice_ff_activation",
        type=str,
        choices=["gelu", "glu", "tanh"],
        help="Feedforward activation inside each SLiCELayer.",
    )
    p.add_argument(
        "--slice_dropout_position",
        type=str,
        choices=["residual", "output"],
        help=(
            "Whether SLiCELayer dropout is applied on residual "
            "branches or on the layer output."
        ),
    )
    p.add_argument(
        "--slice_norm_eps",
        type=float,
        help="Epsilon used by SLiCELayer normalisation modules.",
    )
    p.add_argument(
        "--slice_final_norm",
        action=argparse.BooleanOptionalAction,
        help=("Apply a final RMSNorm before the output head in models/slice.py."),
    )

    # FLA Gated DeltaNet options (used when --arch gated_deltanet)
    p.add_argument(
        "--gated_deltanet_head_dim",
        type=int,
        help="Per-head key dimension for FLA Gated DeltaNet.",
    )
    p.add_argument(
        "--gated_deltanet_expand_v",
        type=float,
        help="Value expansion ratio for FLA Gated DeltaNet.",
    )
    p.add_argument(
        "--gated_deltanet_use_gate",
        action=argparse.BooleanOptionalAction,
        help="Enable the output gate in FLA Gated DeltaNet.",
    )
    p.add_argument(
        "--gated_deltanet_use_short_conv",
        action=argparse.BooleanOptionalAction,
        help="Enable short convolutions in FLA Gated DeltaNet.",
    )
    p.add_argument(
        "--gated_deltanet_allow_neg_eigval",
        action=argparse.BooleanOptionalAction,
        help="Enable the negative-eigenvalue beta scaling path.",
    )
    p.add_argument(
        "--gated_deltanet_conv_size",
        type=int,
        help="Short-convolution kernel size for FLA Gated DeltaNet.",
    )
    p.add_argument(
        "--gated_deltanet_attn_mode",
        type=str,
        choices=["chunk", "fused_recurrent"],
        help="Execution mode for FLA Gated DeltaNet.",
    )
    p.add_argument(
        "--gated_deltanet_norm_eps",
        type=float,
        help="RMSNorm epsilon for FLA Gated DeltaNet.",
    )

    # Mamba options (used when --arch mamba)
    p.add_argument(
        "--mamba_variant",
        type=str,
        choices=["mamba-1", "mamba-2", "mamba-3", "mamba1", "mamba2", "mamba3"],
        help="mamba_ssm block variant. Defaults to mamba-3.",
    )
    p.add_argument(
        "--mamba_d_state",
        type=int,
        help="SSM state size. Defaults to 64 for mamba-1 and 128 for mamba-2/3.",
    )
    p.add_argument(
        "--mamba_d_conv",
        type=int,
        help="Local convolution width for mamba-1/2 blocks.",
    )
    p.add_argument(
        "--mamba_expand",
        type=int,
        help="Block expansion factor used inside the Mamba mixer.",
    )
    p.add_argument(
        "--mamba_headdim",
        type=int,
        help="Head dimension for mamba-2/3 blocks.",
    )
    p.add_argument(
        "--mamba_ngroups",
        type=int,
        help="Number of state groups for mamba-2/3 blocks.",
    )
    p.add_argument(
        "--mamba_d_intermediate",
        type=int,
        help=(
            "Optional GatedMLP hidden size inside the Mamba block. "
            "Defaults to 4 * d_model when unset; use 0 to disable it."
        ),
    )
    p.add_argument(
        "--mamba_rms_norm",
        action=argparse.BooleanOptionalAction,
        help="Use RMSNorm instead of LayerNorm inside the Mamba block.",
    )
    p.add_argument(
        "--mamba_residual_in_fp32",
        action=argparse.BooleanOptionalAction,
        help="Keep residual branches in fp32 inside the Mamba block.",
    )
    p.add_argument(
        "--mamba_fused_add_norm",
        action=argparse.BooleanOptionalAction,
        help="Enable fused add+norm from mamba_ssm when available.",
    )
    p.add_argument(
        "--mamba_use_fast_path",
        action=argparse.BooleanOptionalAction,
        help="Use the fast fused path for mamba-1 when available.",
    )
    p.add_argument(
        "--mamba_use_mem_eff_path",
        action=argparse.BooleanOptionalAction,
        help="Use the memory-efficient fused path for mamba-2 when available.",
    )
    p.add_argument(
        "--mamba_chunk_size",
        type=int,
        help=(
            "Chunk size for mamba-2/3 fused scan kernels. "
            "Defaults to 256 for mamba-2 and 64 for mamba-3."
        ),
    )
    p.add_argument(
        "--mamba_norm_eps",
        type=float,
        help="Epsilon for the Mamba block normalisation layers.",
    )
    p.add_argument(
        "--mamba3_rope_fraction",
        type=float,
        help="RoPE fraction for the Mamba-3 block.",
    )
    p.add_argument(
        "--mamba3_is_outproj_norm",
        action=argparse.BooleanOptionalAction,
        help="Enable output-projection norm in the Mamba-3 block.",
    )
    p.add_argument(
        "--mamba3_is_mimo",
        action=argparse.BooleanOptionalAction,
        help="Enable MIMO mode in the Mamba-3 block.",
    )
    p.add_argument(
        "--mamba3_mimo_rank",
        type=int,
        help="MIMO rank for the Mamba-3 block.",
    )
    p.add_argument(
        "--nan_guard",
        action=argparse.BooleanOptionalAction,
        help="Skip batches with non-finite loss/gradients to avoid divergence.",
    )
    p.add_argument(
        "--nan_restore",
        action=argparse.BooleanOptionalAction,
        help=(
            "Restore last known-good model weights when non-finite values are detected."
        ),
    )
    p.add_argument(
        "--nan_snapshot_every",
        type=int,
        help=(
            "Update in-memory rollback snapshot every N successful "
            "steps (<=0 disables updates)."
        ),
    )
    p.add_argument(
        "--nan_max_consecutive_bad_batches",
        type=int,
        help="Abort if this many bad batches are seen in a row.",
    )
    p.add_argument(
        "--nan_check_params_after_step",
        action=argparse.BooleanOptionalAction,
        help=(
            "Validate model parameters after optimiser step and rollback if non-finite."
        ),
    )

    # W&B tracking
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases tracking."
    )
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity", type=str)
    p.add_argument("--wandb_run_name", type=str)
    p.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"])

    args = p.parse_args()
    args.mamba_variant = canonicalise_mamba_variant(args.mamba_variant)
    return args


def main():
    args = parse_args()
    shards = find_shards(args.data_path)
    spec = load_spec_from_first_line(shards[0])

    for k in ["move_vocab", "start_move_id", "pad_move_id", "piece_symbols"]:
        if k not in spec:
            raise KeyError(f"spec missing key: {k}")

    train(args, spec, shards)


if __name__ == "__main__":
    main()
