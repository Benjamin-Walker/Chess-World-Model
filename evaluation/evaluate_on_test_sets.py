import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from data.loading import (
    ChessJsonlIterableDataset,
    find_shards,
    load_spec_from_first_line,
    make_collate_fn,
)
from main import DEFAULT_EXPERIMENT_CONFIG
from evaluation.metrics import print_val_report, run_validation
from models.factory import (
    build_world_model,
    resolve_mamba_model_kwargs,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DATASETS = [
    (
        "lichess_april_2025_10k",
        str(
            PROJECT_ROOT
            / "data"
            / "test"
            / "lichess_april_2025_10k"
            / "real_game_test.jsonl"
        ),
    ),
    (
        "random_uniform_10k",
        str(
            PROJECT_ROOT
            / "data"
            / "test"
            / "random_uniform_10k"
            / "random_game_test.jsonl"
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Evaluate one or more checkpoints on the standard chess test sets."
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint .pt path or checkpoint directory containing latest.pt.",
    )
    p.add_argument(
        "--checkpoints_file",
        type=str,
        default=None,
        help="Optional text file with one checkpoint path per line.",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Optional repeated NAME=PATH test-set override. "
            "Defaults to the two standard test sets."
        ),
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "test_eval"),
        help="Directory to write per-model JSON results and a summary TSV.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional evaluation batch size override.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional evaluation DataLoader worker count override.",
    )
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Optional cap on batches per dataset. Use 0 to evaluate all batches.",
    )
    p.add_argument(
        "--bin_size",
        type=int,
        default=None,
        help="Optional timestep bin size override for the reported metrics.",
    )
    p.add_argument(
        "--slice_chunk_size",
        type=int,
        default=None,
        help="Optional SLiCE chunk-size override at evaluation time.",
    )
    p.add_argument(
        "--mamba_chunk_size",
        type=int,
        default=None,
        help="Optional Mamba chunk-size override at evaluation time.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU evaluation even if CUDA is available.",
    )
    return p.parse_args()


def parse_datasets(dataset_args: list[str]) -> list[tuple[str, str]]:
    if not dataset_args:
        return list(DEFAULT_TEST_DATASETS)

    datasets: list[tuple[str, str]] = []
    for item in dataset_args:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset value '{item}'. Expected NAME=PATH.")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Invalid --dataset value '{item}': empty dataset name.")
        if not path:
            raise ValueError(f"Invalid --dataset value '{item}': empty dataset path.")
        datasets.append((name, path))
    return datasets


def read_checkpoints(args: argparse.Namespace) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add_path(path: str):
        resolved = resolve_checkpoint_path(path)
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)

    for ckpt in args.checkpoint:
        add_path(ckpt)

    if args.checkpoints_file:
        with open(args.checkpoints_file, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                add_path(stripped)

    if not out:
        raise ValueError("Provide at least one --checkpoint or --checkpoints_file.")
    return out


def resolve_checkpoint_path(path: str) -> str:
    expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    if os.path.isdir(expanded):
        latest = os.path.join(expanded, "latest.pt")
        if os.path.isfile(latest):
            return latest
        raise FileNotFoundError(
            f"Checkpoint directory does not contain latest.pt: {expanded}"
        )
    if os.path.isfile(expanded):
        return expanded
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def checkpoint_label(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    if path.name == "latest.pt" and path.parent.name:
        return path.parent.name
    if path.suffix == ".pt":
        return path.stem
    return path.name


def checkpoint_args_namespace(payload: dict[str, Any]) -> argparse.Namespace:
    cfg = dict(DEFAULT_EXPERIMENT_CONFIG)
    cfg.update(payload.get("args", {}))
    return argparse.Namespace(**cfg)


def build_model(
    model_args: argparse.Namespace, spec: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    return build_world_model(
        model_args,
        move_vocab=int(spec["move_vocab"]),
        device=device,
        pad_move_id=int(spec["pad_move_id"]),
    )


def make_eval_loader(
    *,
    model_args: argparse.Namespace,
    spec: dict[str, Any],
    data_path: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    device: torch.device,
) -> DataLoader:
    shards = find_shards(data_path)
    shard_spec = load_spec_from_first_line(shards[0])
    for key in ("move_vocab", "pad_move_id"):
        if int(spec[key]) != int(shard_spec[key]):
            raise ValueError(
                f"Dataset spec mismatch for {data_path}: "
                f"checkpoint {key}={spec[key]!r}, dataset {key}={shard_spec[key]!r}"
            )

    collate_fn = make_collate_fn(
        pad_move_id=int(spec["pad_move_id"]),
        max_steps=model_args.max_steps,
    )
    dataset = ChessJsonlIterableDataset(
        shards=shards,
        seed=model_args.seed,
        shuffle_shards=False,
        max_steps=model_args.max_steps,
        split="all",
        val_fraction=0.0,
        split_mod=model_args.split_mod,
    )
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else (model_args.val_batch_size or model_args.batch_size)
    )
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else model_args.val_num_workers
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_summary_tsv(path: str, rows: list[dict[str, Any]]) -> None:
    header = [
        "model",
        "arch",
        "d_model",
        "n_layers",
        "step",
        "dataset",
        "loss",
        "partial_exact",
        "full_exact",
        "game_exact",
        "game_num",
        "game_denom",
        "checkpoint",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(key, "")) for key in header) + "\n")


def evaluate_checkpoint(
    ckpt_path: str,
    datasets: list[tuple[str, str]],
    cli_args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint_args_namespace(payload)
    if cli_args.slice_chunk_size is not None and model_args.arch == "slice":
        model_args.slice_chunk_size = cli_args.slice_chunk_size
    if cli_args.mamba_chunk_size is not None and model_args.arch == "mamba":
        model_args.mamba_chunk_size = cli_args.mamba_chunk_size

    spec = payload.get("spec")
    if spec is None:
        first_dataset_path = datasets[0][1]
        first_shard = find_shards(first_dataset_path)[0]
        spec = load_spec_from_first_line(first_shard)

    model = build_model(model_args, spec, device)
    model.load_state_dict(payload["model"], strict=True)

    label = checkpoint_label(ckpt_path)
    step = int(payload.get("step", 0))
    epoch = int(payload.get("epoch", 0))

    result = {
        "checkpoint": ckpt_path,
        "model": {
            "label": label,
            "arch": model_args.arch,
            "d_model": int(model_args.d_model),
            "n_layers": int(model_args.n_layers),
            "run_name": model_args.wandb_run_name,
        },
        "step": step,
        "epoch": epoch,
        "datasets": {},
    }
    summary_rows: list[dict[str, Any]] = []

    print(
        f"[eval] model={label} arch={model_args.arch} "
        f"d_model={model_args.d_model} n_layers={model_args.n_layers} "
        f"step={step}"
    )
    if model_args.arch == "slice":
        print(f"[eval] slice_chunk_size={model_args.slice_chunk_size}")
    if model_args.arch == "gated_deltanet":
        print(
            f"[eval] gated_deltanet_head_dim={model_args.gated_deltanet_head_dim} "
            f"allow_neg_eigval={model_args.gated_deltanet_allow_neg_eigval}"
        )
    if model_args.arch == "mamba":
        mamba_kwargs = resolve_mamba_model_kwargs(model_args)
        print(
            f"[eval] mamba_variant={mamba_kwargs['variant']} "
            f"mamba_chunk_size={mamba_kwargs['chunk_size']}"
        )

    for dataset_name, dataset_path in datasets:
        loader = make_eval_loader(
            model_args=model_args,
            spec=spec,
            data_path=dataset_path,
            batch_size_override=cli_args.batch_size,
            num_workers_override=cli_args.num_workers,
            device=device,
        )
        print(f"[eval:{dataset_name}] data={dataset_path}")
        metrics = run_validation(
            model=model,
            val_loader=loader,
            device=device,
            max_batches=cli_args.max_batches,
            bin_size=(
                cli_args.bin_size
                if cli_args.bin_size is not None
                else model_args.bin_size
            ),
        )
        print_val_report(step, metrics)
        result["datasets"][dataset_name] = {
            key: float(value) for key, value in metrics.items()
        }
        summary_rows.append(
            {
                "model": label,
                "arch": model_args.arch,
                "d_model": int(model_args.d_model),
                "n_layers": int(model_args.n_layers),
                "step": step,
                "dataset": dataset_name,
                "loss": float(metrics.get("val/loss", 0.0)),
                "partial_exact": float(metrics.get("val/overall/partial_exact", 0.0)),
                "full_exact": float(metrics.get("val/overall/full_exact", 0.0)),
                "game_exact": float(metrics.get("val/game_exact", 0.0)),
                "game_num": int(metrics.get("val/game_exact_num", 0.0)),
                "game_denom": int(metrics.get("val/game_exact_denom", 0.0)),
                "checkpoint": ckpt_path,
            }
        )

    return result, summary_rows


def main() -> None:
    args = parse_args()
    checkpoints = read_checkpoints(args)
    datasets = parse_datasets(args.dataset)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"[info] device={device.type}")
    print("[info] datasets=" + ", ".join(f"{name}:{path}" for name, path in datasets))
    print(f"[info] checkpoints={len(checkpoints)}")

    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    os.makedirs(output_root, exist_ok=True)

    all_summary_rows: list[dict[str, Any]] = []
    for ckpt_path in checkpoints:
        result, summary_rows = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            datasets=datasets,
            cli_args=args,
            device=device,
        )
        label = result["model"]["label"]
        result_path = os.path.join(output_root, label, "test_eval_results.json")
        write_json(result_path, result)
        print(f"[save] wrote {result_path}")
        all_summary_rows.extend(summary_rows)

    summary_path = os.path.join(output_root, "summary.tsv")
    write_summary_tsv(summary_path, all_summary_rows)
    print(f"[save] wrote {summary_path}")

    print("model\tdataset\tpartial_exact\tfull_exact\tgame_exact\tgame_num/game_denom")
    for row in all_summary_rows:
        print(
            f"{row['model']}\t{row['dataset']}\t"
            f"{row['partial_exact']:.6f}\t{row['full_exact']:.6f}\t"
            f"{row['game_exact']:.6f}\t{row['game_num']}/{row['game_denom']}"
        )


if __name__ == "__main__":
    main()
