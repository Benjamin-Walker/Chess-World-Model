import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext, suppress
from typing import Optional

import torch
import torch.nn as nn
from data.loading import ChessJsonlIterableDataset, make_collate_fn
from evaluation.metrics import compute_state_loss, print_val_report, run_validation
from models.factory import (
    build_world_model,
    resolve_gated_deltanet_model_kwargs,
    resolve_mamba_model_kwargs,
)
from torch.utils.data import DataLoader

CHESS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import wandb
except Exception:
    wandb = None


def format_rate(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.2f}K"
    return f"{x:.0f}"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_model_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model_state_from_cpu(
    model: nn.Module, state: dict[str, torch.Tensor], device: torch.device
):
    model.load_state_dict(
        {k: v.to(device=device) for k, v in state.items()}, strict=True
    )


def model_has_nonfinite_params(model: nn.Module) -> bool:
    for p in model.parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all():
            return True
    return False


def build_optimiser(
    args: argparse.Namespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    use_explicit_no_decay_split = getattr(args, "arch", None) == "gated_deltanet"

    for _, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if use_explicit_no_decay_split and getattr(param, "_no_weight_decay", False):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, object]]
    if no_decay_params:
        param_groups = [
            {"params": decay_params, "weight_decay": float(args.weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    else:
        param_groups = [
            {"params": decay_params, "weight_decay": float(args.weight_decay)}
        ]

    return torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, args.beta2),
    )


def should_use_mamba3_autocast(args: argparse.Namespace, device: torch.device) -> bool:
    if device.type != "cuda" or getattr(args, "arch", None) != "mamba":
        return False
    return resolve_mamba_model_kwargs(args)["variant"] == "mamba-3"


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


# ----------------------------
# Checkpointing
# ----------------------------


def save_checkpoint(
    ckpt_dir_or_file: str,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    step: int,
    epoch: int,
    spec: dict,
    args: argparse.Namespace,
):
    if ckpt_dir_or_file.endswith(".pt"):
        os.makedirs(os.path.dirname(ckpt_dir_or_file) or ".", exist_ok=True)
    else:
        os.makedirs(ckpt_dir_or_file, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "spec": spec,
        "args": vars(args),
        "torch_version": torch.__version__,
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()

    if ckpt_dir_or_file.endswith(".pt"):
        torch.save(payload, ckpt_dir_or_file)
        return

    step_path = os.path.join(ckpt_dir_or_file, f"ckpt_step_{step:09d}.pt")
    latest_path = os.path.join(ckpt_dir_or_file, "latest.pt")
    torch.save(payload, step_path)
    torch.save(payload, latest_path)


def run_summary_path(ckpt_dir_or_file: str) -> str:
    if ckpt_dir_or_file.endswith(".pt"):
        return f"{ckpt_dir_or_file}.summary.json"
    return os.path.join(ckpt_dir_or_file, "run_summary.json")


def save_run_summary(
    ckpt_dir_or_file: str,
    *,
    status: str,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    val_metrics: Optional[dict[str, float]] = None,
    error: Optional[str] = None,
):
    if ckpt_dir_or_file.endswith(".pt"):
        os.makedirs(os.path.dirname(ckpt_dir_or_file) or ".", exist_ok=True)
    else:
        os.makedirs(ckpt_dir_or_file, exist_ok=True)

    payload = {
        "status": status,
        "step": int(step),
        "epoch": int(epoch),
        "ckpt_path": args.ckpt_path,
        "run_name": args.wandb_run_name,
        "arch": args.arch,
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "beta2": float(args.beta2),
    }
    if val_metrics is not None:
        payload["val_metrics"] = {
            key: float(value) for key, value in val_metrics.items()
        }
    if error is not None:
        payload["error"] = error

    summary_path = run_summary_path(ckpt_dir_or_file)
    tmp_path = f"{summary_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, summary_path)


# ----------------------------
# Training
# ----------------------------


def build_lr_scheduler(
    args: argparse.Namespace, optim: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler_name = args.lr_scheduler.lower()
    if scheduler_name == "none":
        return None

    warmup_steps = int(args.lr_warmup_steps)
    decay_steps = int(args.lr_decay_steps)
    min_ratio = float(args.lr_min_ratio)

    if decay_steps <= 0:
        raise ValueError("lr_decay_steps must be > 0 when lr_scheduler is enabled.")
    if warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be >= 0.")
    if warmup_steps >= decay_steps:
        raise ValueError("lr_warmup_steps must be < lr_decay_steps.")
    if not (0.0 < min_ratio <= 1.0):
        raise ValueError("lr_min_ratio must be in (0, 1].")

    def _warmup_factor(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        return 1.0

    if scheduler_name == "cosine":

        def lr_lambda(step: int) -> float:
            warmup_factor = _warmup_factor(step)
            if step < warmup_steps:
                return warmup_factor
            clipped_step = min(step, decay_steps)
            progress = (clipped_step - warmup_steps) / float(decay_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

    elif scheduler_name == "linear":

        def lr_lambda(step: int) -> float:
            warmup_factor = _warmup_factor(step)
            if step < warmup_steps:
                return warmup_factor
            clipped_step = min(step, decay_steps)
            progress = (clipped_step - warmup_steps) / float(decay_steps - warmup_steps)
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)

    else:
        raise ValueError(
            f"Unsupported lr_scheduler='{args.lr_scheduler}'. "
            "Expected one of: none, cosine, linear."
        )

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


def train(args: argparse.Namespace, spec: dict, shards: list[str]):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("max_train_steps must be > 0 when set.")
    if not (0.0 < args.beta2 < 1.0):
        raise ValueError("beta2 must be in (0, 1).")

    move_vocab = int(spec["move_vocab"])
    pad_move_id = int(spec["pad_move_id"])

    collate_fn = make_collate_fn(pad_move_id=pad_move_id, max_steps=args.max_steps)

    train_ds = ChessJsonlIterableDataset(
        shards=shards,
        seed=args.seed,
        shuffle_shards=not args.no_shuffle_shards,
        max_steps=args.max_steps,
        split="train",
        val_fraction=args.val_fraction,
        split_mod=args.split_mod,
    )

    val_ds = ChessJsonlIterableDataset(
        shards=shards,
        seed=args.seed,
        shuffle_shards=False,
        max_steps=args.max_steps,
        split="val",
        val_fraction=args.val_fraction,
        split_mod=args.split_mod,
    )

    def make_train_loader(epoch: int) -> DataLoader:
        train_ds.set_epoch(epoch)
        return DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
        )

    def make_val_loader() -> DataLoader:
        return DataLoader(
            val_ds,
            batch_size=(args.val_batch_size or args.batch_size),
            num_workers=args.val_num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.val_num_workers > 0),
            collate_fn=collate_fn,
        )

    model = build_world_model(
        args,
        move_vocab=move_vocab,
        device=device,
        pad_move_id=pad_move_id,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params}")

    optim = build_optimiser(args, model)
    lr_scheduler = build_lr_scheduler(args, optim)
    use_mamba3_autocast = should_use_mamba3_autocast(args, device)

    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        if lr_scheduler is not None and "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        print(
            f"[resume] loaded {args.resume} (epoch={start_epoch}, step={global_step})"
        )

    nan_skipped_batches = 0
    nan_rollbacks = 0
    consecutive_bad_batches = 0
    last_good_model_state: Optional[dict[str, torch.Tensor]] = None
    if args.nan_guard and args.nan_restore:
        last_good_model_state = clone_model_state_to_cpu(model)

    # Overfit one batch mode
    if args.overfit_one_batch:
        loader = make_train_loader(start_epoch)
        batch = next(iter(loader))
        batch_moves = batch.moves.to(device, non_blocking=True)
        batch_states = batch.states.to(device, non_blocking=True)
        batch_pad = batch.pad_mask.to(device, non_blocking=True)

        print("[debug] overfitting one batch")
        model.train()
        t0 = time.time()
        last_tokens = 0

        for it in range(args.overfit_iters):
            optim.zero_grad(set_to_none=True)

            with autocast_context(use_mamba3_autocast):
                out = model(batch_moves)
                loss, metrics, valid_tokens = compute_state_loss(
                    out, batch_states, batch_pad
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            last_tokens += valid_tokens
            if (it + 1) % args.log_every == 0 or it == 0:
                dt = max(time.time() - t0, 1e-6)
                tok_s = last_tokens / dt
                print(
                    f"[debug it={it + 1:5d}] loss={metrics['loss']:.4f} "
                    f"grad_norm={float(grad_norm):.2f} state_tok/s={format_rate(tok_s)}"
                )
                t0 = time.time()
                last_tokens = 0

        if args.ckpt_path:
            save_checkpoint(
                args.ckpt_path,
                model,
                optim,
                lr_scheduler,
                global_step,
                start_epoch,
                spec,
                args,
            )
            print(f"[debug] saved checkpoint to {args.ckpt_path}")
        return

    # Regular training
    print(
        f"[info] device={device.type} | shards={len(shards)} | move_vocab={move_vocab}"
    )
    if args.arch == "transformer":
        print(
            f"[info] model: d_model={args.d_model} "
            f"n_layers={args.n_layers} n_heads={args.n_heads} "
            f"dropout={args.dropout} ff_mult={args.ff_mult}"
        )
    elif args.arch == "slice":
        print(
            f"[info] model: d_model={args.d_model} n_layers={args.n_layers} "
            f"dropout={args.dropout} ff_mult={args.ff_mult} "
            f"slice_block_size={args.slice_block_size}"
        )
    elif args.arch == "gated_deltanet":
        gated_deltanet_kwargs = resolve_gated_deltanet_model_kwargs(args)
        print(
            f"[info] model: d_model={args.d_model} n_layers={args.n_layers} "
            f"n_heads={args.n_heads} ff_mult={args.ff_mult} "
            f"head_dim={gated_deltanet_kwargs['head_dim']} "
            f"expand_v={gated_deltanet_kwargs['expand_v']} "
            f"allow_neg_eigval={gated_deltanet_kwargs['allow_neg_eigval']}"
        )
    else:
        mamba_kwargs = resolve_mamba_model_kwargs(args)
        print(
            f"[info] model: variant={mamba_kwargs['variant']} "
            f"d_model={args.d_model} n_layers={args.n_layers} "
            f"d_state={mamba_kwargs['d_state']} expand={mamba_kwargs['expand']} "
            f"chunk_size={mamba_kwargs['chunk_size']}"
        )
        if use_mamba3_autocast:
            print("[info] precision: cuda bf16 autocast enabled for mamba-3")
    print(
        f"[info] val_fraction={args.val_fraction} "
        f"split_mod={args.split_mod} bin_size={args.bin_size}"
    )
    print(
        "[info] optimiser: "
        f"lr={args.lr} weight_decay={args.weight_decay} beta2={args.beta2}"
    )
    if lr_scheduler is None:
        print("[info] lr_scheduler=none")
    else:
        print(
            f"[info] lr_scheduler={args.lr_scheduler} "
            f"warmup_steps={args.lr_warmup_steps} "
            f"decay_steps={args.lr_decay_steps} "
            f"min_ratio={args.lr_min_ratio}"
        )
    if args.max_train_steps is None:
        print("[info] max_train_steps=None (epoch-bounded training)")
    else:
        print(f"[info] max_train_steps={args.max_train_steps}")

    val_loader = make_val_loader()
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("W&B logging requested but `wandb` is not installed.")
        os.makedirs(os.path.join(CHESS_DIR, "wandb"), exist_ok=True)
        wandb_config = {
            **vars(args),
            "move_vocab": move_vocab,
            "data_shards": len(shards),
        }
        if args.arch in {"slice", "mamba"}:
            wandb_config.pop("n_heads", None)

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            dir=CHESS_DIR,
            config=wandb_config,
        )

    # Throughput timer
    last_log_t = time.time()
    last_log_tokens = 0
    last_val_step = -1
    latest_val_metrics: Optional[dict[str, float]] = None

    if args.ckpt_path:
        save_run_summary(
            args.ckpt_path,
            status="running",
            step=global_step,
            epoch=start_epoch,
            args=args,
        )

    reached_max_train_steps = False
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            current_epoch = epoch
            train_loader = make_train_loader(epoch)
            model.train()

            for batch in train_loader:
                if (
                    args.max_train_steps is not None
                    and global_step >= args.max_train_steps
                ):
                    reached_max_train_steps = True
                    break
                global_step += 1

                moves = batch.moves.to(device, non_blocking=True)
                states = batch.states.to(device, non_blocking=True)
                pad_mask = batch.pad_mask.to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)

                with autocast_context(use_mamba3_autocast):
                    out = model(moves)
                    loss, metrics, valid_tokens = compute_state_loss(
                        out, states, pad_mask
                    )

                if args.nan_guard and not torch.isfinite(loss.detach()).item():
                    nan_skipped_batches += 1
                    consecutive_bad_batches += 1
                    optim.zero_grad(set_to_none=True)
                    print(
                        f"[warn step={global_step:09d}] non-finite loss; skipped batch "
                        f"(bad_streak={consecutive_bad_batches}, "
                        f"skipped={nan_skipped_batches}, "
                        f"rollbacks={nan_rollbacks})"
                    )
                    if consecutive_bad_batches >= args.nan_max_consecutive_bad_batches:
                        raise RuntimeError(
                            "Too many consecutive non-finite batches. "
                            "Stopping to avoid an infinite bad-batch loop."
                        )
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if args.nan_guard and (not math.isfinite(float(grad_norm))):
                    nan_skipped_batches += 1
                    consecutive_bad_batches += 1
                    optim.zero_grad(set_to_none=True)
                    print(
                        f"[warn step={global_step:09d}] "
                        f"non-finite grad_norm; skipped batch "
                        f"(bad_streak={consecutive_bad_batches}, "
                        f"skipped={nan_skipped_batches}, "
                        f"rollbacks={nan_rollbacks})"
                    )
                    if consecutive_bad_batches >= args.nan_max_consecutive_bad_batches:
                        raise RuntimeError(
                            "Too many consecutive non-finite batches. "
                            "Stopping to avoid an infinite bad-batch loop."
                        )
                    continue
                optim.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if (
                    args.nan_guard
                    and args.nan_check_params_after_step
                    and model_has_nonfinite_params(model)
                ):
                    nan_skipped_batches += 1
                    consecutive_bad_batches += 1
                    if args.nan_restore and last_good_model_state is not None:
                        restore_model_state_from_cpu(
                            model, last_good_model_state, device=device
                        )
                        nan_rollbacks += 1
                    optim.zero_grad(set_to_none=True)
                    print(
                        f"[warn step={global_step:09d}] "
                        f"non-finite model params after step; restored "
                        f"(bad_streak={consecutive_bad_batches}, "
                        f"skipped={nan_skipped_batches}, "
                        f"rollbacks={nan_rollbacks})"
                    )
                    if consecutive_bad_batches >= args.nan_max_consecutive_bad_batches:
                        raise RuntimeError(
                            "Too many consecutive non-finite batches. "
                            "Stopping to avoid an infinite bad-batch loop."
                        )
                    continue

                consecutive_bad_batches = 0
                if (
                    args.nan_guard
                    and args.nan_restore
                    and (args.nan_snapshot_every > 0)
                    and (global_step % args.nan_snapshot_every == 0)
                ):
                    last_good_model_state = clone_model_state_to_cpu(model)

                last_log_tokens += valid_tokens

                if global_step % args.log_every == 0:
                    now = time.time()
                    dt = max(now - last_log_t, 1e-6)
                    tok_s = last_log_tokens / dt
                    lr = optim.param_groups[0]["lr"]
                    print(
                        f"[epoch={epoch:03d} step={global_step:09d}] "
                        f"loss={metrics['loss']:.4f} lr={lr:.2e} "
                        f"grad_norm={float(grad_norm):.2f} "
                        f"state_tok/s={format_rate(tok_s)}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": metrics["loss"],
                                "train/lr": lr,
                                "train/grad_norm": float(grad_norm),
                                "train/state_tok_per_s": tok_s,
                                "train/epoch": epoch,
                                "train/nan_skipped_batches": nan_skipped_batches,
                                "train/nan_rollbacks": nan_rollbacks,
                            },
                            step=global_step,
                        )
                    last_log_t = now
                    last_log_tokens = 0

                if (
                    args.ckpt_path
                    and (args.save_every > 0)
                    and (global_step % args.save_every == 0)
                ):
                    save_checkpoint(
                        args.ckpt_path,
                        model,
                        optim,
                        lr_scheduler,
                        global_step,
                        epoch,
                        spec,
                        args,
                    )
                    print(f"[ckpt] saved at step {global_step} -> {args.ckpt_path}")

                if args.val_every > 0 and (global_step % args.val_every == 0):
                    tval0 = time.time()
                    val_metrics = run_validation(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        max_batches=args.val_batches,
                        bin_size=args.bin_size,
                        amp_bfloat16=use_mamba3_autocast,
                    )
                    tval = time.time() - tval0
                    print(f"[val] done in {tval:.1f}s")
                    print_val_report(global_step, val_metrics)
                    latest_val_metrics = val_metrics
                    if wandb_run is not None:
                        wandb_run.log(val_metrics, step=global_step)
                    if args.ckpt_path:
                        save_run_summary(
                            args.ckpt_path,
                            status="running",
                            step=global_step,
                            epoch=epoch,
                            args=args,
                            val_metrics=latest_val_metrics,
                        )
                    last_val_step = global_step
                    model.train()

                    # Reset throughput stats so validation time does not
                    # tank state_tok/s.
                    last_log_t = time.time()
                    last_log_tokens = 0

            if args.val_every > 0 and global_step > 0 and global_step != last_val_step:
                tval0 = time.time()
                val_metrics = run_validation(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    max_batches=args.val_batches,
                    bin_size=args.bin_size,
                    amp_bfloat16=use_mamba3_autocast,
                )
                tval = time.time() - tval0
                print(f"[val] end-epoch done in {tval:.1f}s")
                print_val_report(global_step, val_metrics)
                latest_val_metrics = val_metrics
                if wandb_run is not None:
                    wandb_run.log(val_metrics, step=global_step)
                if args.ckpt_path:
                    save_run_summary(
                        args.ckpt_path,
                        status="running",
                        step=global_step,
                        epoch=epoch,
                        args=args,
                        val_metrics=latest_val_metrics,
                    )
                last_val_step = global_step
                model.train()

            if args.ckpt_path:
                save_checkpoint(
                    args.ckpt_path,
                    model,
                    optim,
                    lr_scheduler,
                    global_step,
                    epoch + 1,
                    spec,
                    args,
                )
                print(f"[ckpt] end-epoch save -> {args.ckpt_path}")
                run_status = (
                    "finished"
                    if reached_max_train_steps
                    or (epoch + 1) >= (start_epoch + args.epochs)
                    else "running"
                )
                save_run_summary(
                    args.ckpt_path,
                    status=run_status,
                    step=global_step,
                    epoch=epoch + 1,
                    args=args,
                    val_metrics=latest_val_metrics,
                )
            if reached_max_train_steps:
                print(
                    f"[info] reached max_train_steps={args.max_train_steps}; stopping."
                )
                break

    except KeyboardInterrupt:
        with suppress(BrokenPipeError):
            print("\n[interrupt] caught KeyboardInterrupt")
        if args.ckpt_path:
            save_checkpoint(
                args.ckpt_path,
                model,
                optim,
                lr_scheduler,
                global_step,
                epoch,
                spec,
                args,
            )
            with suppress(BrokenPipeError):
                print(f"[ckpt] saved interrupt checkpoint -> {args.ckpt_path}")
            save_run_summary(
                args.ckpt_path,
                status="interrupted",
                step=global_step,
                epoch=current_epoch,
                args=args,
                val_metrics=latest_val_metrics,
            )
    except Exception as exc:
        if args.ckpt_path:
            save_run_summary(
                args.ckpt_path,
                status="failed",
                step=global_step,
                epoch=current_epoch,
                args=args,
                val_metrics=latest_val_metrics,
                error=repr(exc),
            )
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()
