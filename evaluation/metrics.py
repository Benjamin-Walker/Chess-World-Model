from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----------------------------
# Loss
# ----------------------------


def masked_ce_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [..., C]
    targets: [...]
    valid_mask: same shape as targets (bool)
    Returns: (loss_sum, denom_sum)
    """
    C = logits.shape[-1]
    logits_f = logits.reshape(-1, C)
    targets_f = targets.reshape(-1)
    mask_f = valid_mask.reshape(-1).to(dtype=logits.dtype)

    ce = F.cross_entropy(logits_f, targets_f, reduction="none")
    loss_sum = (ce * mask_f).sum()
    denom_sum = mask_f.sum()
    return loss_sum, denom_sum


def compute_state_loss(
    out: dict[str, torch.Tensor],
    states: torch.Tensor,  # [B,T,75]
    pad_mask: torch.Tensor,  # [B,T] True for pad
) -> tuple[torch.Tensor, dict[str, float], int]:
    B, T, S = states.shape
    assert S == 75

    valid_steps = ~pad_mask  # [B,T] bool

    tgt_pieces = states[:, :, 0:64]
    tgt_side = states[:, :, 64]
    tgt_castle = states[:, :, 65:69]
    tgt_ep_file = states[:, :, 69]
    tgt_ep_rank = states[:, :, 70]
    tgt_halfmove = states[:, :, 71:73]
    tgt_fullmove = states[:, :, 73:75]

    m_pieces = valid_steps.unsqueeze(-1).expand(B, T, 64)
    m_castle = valid_steps.unsqueeze(-1).expand(B, T, 4)
    m_half = valid_steps.unsqueeze(-1).expand(B, T, 2)
    m_full = valid_steps.unsqueeze(-1).expand(B, T, 2)

    loss_sum = torch.zeros((), device=states.device, dtype=out["side"].dtype)
    denom_sum = torch.zeros((), device=states.device, dtype=out["side"].dtype)

    ls, ds = masked_ce_sum(out["pieces"], tgt_pieces, m_pieces)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    ls, ds = masked_ce_sum(out["side"], tgt_side, valid_steps)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    ls, ds = masked_ce_sum(out["castle"], tgt_castle, m_castle)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    ls, ds = masked_ce_sum(out["ep_file"], tgt_ep_file, valid_steps)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    ls, ds = masked_ce_sum(out["ep_rank"], tgt_ep_rank, valid_steps)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    ls, ds = masked_ce_sum(out["halfmove"], tgt_halfmove, m_half)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    ls, ds = masked_ce_sum(out["fullmove"], tgt_fullmove, m_full)
    loss_sum = loss_sum + ls
    denom_sum = denom_sum + ds

    denom_sum = denom_sum.clamp_min(1.0)
    loss = loss_sum / denom_sum

    valid_state_tokens = int((~pad_mask).sum().item()) * 75
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "loss_sum": float(loss_sum.detach().cpu().item()),
        "denom": float(denom_sum.detach().cpu().item()),
    }
    return loss, metrics, valid_state_tokens


# ----------------------------
# Validation metrics: full_exact + partial_exact (binned) + game_exact
# ----------------------------


@torch.no_grad()
def binned_exact_sums(
    out: dict[str, torch.Tensor],
    states: torch.Tensor,  # [B,T,75]
    pad_mask: torch.Tensor,  # [B,T]
    bin_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      (
          full_num_per_bin,
          partial_sum_per_bin,
          denom_per_bin,
          game_exact_num,
          game_exact_denom,
      )
    All returned as float64 CPU tensors.
    denom counts valid (non-pad) steps in each time bin.
    """
    B, T, _ = states.shape
    device = states.device
    valid = ~pad_mask  # [B,T]

    # Targets
    tgt_pieces = states[:, :, 0:64]
    tgt_side = states[:, :, 64]
    tgt_castle = states[:, :, 65:69]
    tgt_ep_file = states[:, :, 69]
    tgt_ep_rank = states[:, :, 70]
    tgt_half = states[:, :, 71:73]
    tgt_full = states[:, :, 73:75]

    # Predictions
    pred_pieces = out["pieces"].argmax(-1)
    pred_side = out["side"].argmax(-1)
    pred_castle = out["castle"].argmax(-1)
    pred_ep_file = out["ep_file"].argmax(-1)
    pred_ep_rank = out["ep_rank"].argmax(-1)
    pred_half = out["halfmove"].argmax(-1)
    pred_full = out["fullmove"].argmax(-1)

    correct = torch.zeros((B, T), device=device, dtype=torch.int32)
    correct += (pred_pieces == tgt_pieces).sum(dim=-1).to(torch.int32)
    correct += (pred_side == tgt_side).to(torch.int32)
    correct += (pred_castle == tgt_castle).sum(dim=-1).to(torch.int32)
    correct += (pred_ep_file == tgt_ep_file).to(torch.int32)
    correct += (pred_ep_rank == tgt_ep_rank).to(torch.int32)
    correct += (pred_half == tgt_half).sum(dim=-1).to(torch.int32)
    correct += (pred_full == tgt_full).sum(dim=-1).to(torch.int32)

    full = correct == 75  # [B,T]
    partial = correct.float() / 75.0

    # Whole-game exact: every valid step must be full-exact.
    # padded steps are treated as OK.
    game_ok = ((~valid) | full).all(dim=1)  # [B]
    game_exact_num = game_ok.float().sum()
    game_exact_denom = torch.tensor(float(B), device=device)

    n_bins = (T + bin_size - 1) // bin_size
    full_num = torch.zeros((n_bins,), device=device, dtype=torch.float32)
    partial_sum = torch.zeros((n_bins,), device=device, dtype=torch.float32)
    denom = torch.zeros((n_bins,), device=device, dtype=torch.float32)

    # Bin indices for each timestep
    t_idx = torch.arange(T, device=device, dtype=torch.int64)
    bin_idx = (t_idx // bin_size).view(1, T).expand(B, T)

    mask = valid
    if mask.any():
        idx = bin_idx[mask].reshape(-1)
        ones = torch.ones_like(idx, dtype=torch.float32, device=device)
        denom.scatter_add_(0, idx, ones)
        full_num.scatter_add_(0, idx, full[mask].float().reshape(-1))
        partial_sum.scatter_add_(0, idx, partial[mask].float().reshape(-1))

    return (
        full_num.double().cpu(),
        partial_sum.double().cpu(),
        denom.double().cpu(),
        game_exact_num.double().cpu(),
        game_exact_denom.double().cpu(),
    )


class ValMetricAccumulator:
    def __init__(self, bin_size: int):
        self.bin_size = int(bin_size)
        self.full_num = torch.zeros((0,), dtype=torch.float64)
        self.partial_sum = torch.zeros((0,), dtype=torch.float64)
        self.denom = torch.zeros((0,), dtype=torch.float64)

        self.loss_sum = 0.0
        self.loss_denom = 0.0

        self.game_exact_num = 0.0
        self.game_exact_denom = 0.0

    def _ensure_bins(self, n_bins: int):
        if self.full_num.numel() >= n_bins:
            return
        new_len = n_bins

        def pad_to(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((new_len,), dtype=torch.float64)
            if x.numel() > 0:
                out[: x.numel()] = x
            return out

        self.full_num = pad_to(self.full_num)
        self.partial_sum = pad_to(self.partial_sum)
        self.denom = pad_to(self.denom)

    def add_batch(
        self,
        out: dict[str, torch.Tensor],
        states: torch.Tensor,
        pad_mask: torch.Tensor,
        loss_metrics: dict[str, float],
    ):
        full_num, partial_sum, denom, gnum, gden = binned_exact_sums(
            out, states, pad_mask, self.bin_size
        )
        self._ensure_bins(full_num.numel())
        self.full_num[: full_num.numel()] += full_num
        self.partial_sum[: partial_sum.numel()] += partial_sum
        self.denom[: denom.numel()] += denom

        self.loss_sum += float(loss_metrics["loss_sum"])
        self.loss_denom += float(loss_metrics["denom"])

        self.game_exact_num += float(gnum.item())
        self.game_exact_denom += float(gden.item())

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.loss_denom > 0:
            out["val/loss"] = self.loss_sum / self.loss_denom

        denom_total = float(self.denom.sum().item())
        if denom_total <= 0:
            out["val/overall/full_exact"] = 0.0
            out["val/overall/partial_exact"] = 0.0
        else:
            out["val/overall/full_exact"] = (
                float(self.full_num.sum().item()) / denom_total
            )
            out["val/overall/partial_exact"] = (
                float(self.partial_sum.sum().item()) / denom_total
            )

        if self.game_exact_denom > 0:
            out["val/game_exact"] = self.game_exact_num / self.game_exact_denom
            out["val/game_exact_num"] = self.game_exact_num
            out["val/game_exact_denom"] = self.game_exact_denom

        for b in range(self.denom.numel()):
            lo = b * self.bin_size
            hi = (b + 1) * self.bin_size
            d = float(self.denom[b].item())
            out[f"val/bin{lo:03d}-{hi:03d}/n"] = d
            if d <= 0:
                fe = 0.0
                pe = 0.0
            else:
                fe = float(self.full_num[b].item()) / d
                pe = float(self.partial_sum[b].item()) / d
            out[f"val/bin{lo:03d}-{hi:03d}/full_exact"] = fe
            out[f"val/bin{lo:03d}-{hi:03d}/partial_exact"] = pe

        return out


def print_val_report(step: int, metrics: dict[str, float]):
    loss = metrics.get("val/loss", 0.0)
    fe = metrics.get("val/overall/full_exact", 0.0)
    pe = metrics.get("val/overall/partial_exact", 0.0)
    print(
        f"[val step={step:09d}] loss={loss:.4f} full_exact={fe:.4f} "
        f"partial_exact={pe:.4f} avg_wrong={(1.0 - pe) * 75.0:.2f}"
    )

    gx = metrics.get("val/game_exact", 0.0)
    gnum = int(metrics.get("val/game_exact_num", 0.0))
    gden = int(metrics.get("val/game_exact_denom", 0.0))
    print(f"  game_exact={gx:.6f} ({gnum}/{gden})")

    # Collect bin labels like "000-020" from keys "val/bin000-020/full_exact"
    bins = []
    for k in metrics:
        if k.startswith("val/bin") and k.endswith("/full_exact"):
            base = k.rsplit("/", 1)[0]  # "val/bin000-020"
            label = base.split("val/bin", 1)[1]  # "000-020"
            bins.append(label)
    bins = sorted(set(bins))
    if not bins:
        return

    print("  bin        full_exact  partial_exact  avg_wrong   n_steps")
    for b in bins:
        fe_b = metrics.get(f"val/bin{b}/full_exact", 0.0)
        pe_b = metrics.get(f"val/bin{b}/partial_exact", 0.0)
        n_b = int(metrics.get(f"val/bin{b}/n", 0.0))
        wrong = (1.0 - pe_b) * 75.0

        lo_s, hi_s = b.split("-")
        lo = int(lo_s)
        hi = int(hi_s)
        print(
            f"  {lo:>3d}-{hi:<3d}     {fe_b:>8.4f}     "
            f"{pe_b:>10.4f}   {wrong:>8.2f}   {n_b:>7d}"
        )


@torch.no_grad()
def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int,
    bin_size: int,
    amp_bfloat16: bool = False,
) -> dict[str, float]:
    model.eval()
    acc = ValMetricAccumulator(bin_size=bin_size)

    for i, batch in enumerate(val_loader):
        if max_batches is not None and max_batches > 0 and i >= max_batches:
            break

        moves = batch.moves.to(device, non_blocking=True)
        states = batch.states.to(device, non_blocking=True)
        pad_mask = batch.pad_mask.to(device, non_blocking=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp_bfloat16 and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            out = model(moves)
            _, loss_metrics, _ = compute_state_loss(out, states, pad_mask)

        acc.add_batch(out, states, pad_mask, loss_metrics)

    return acc.summary()
