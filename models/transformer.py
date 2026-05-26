from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}.")
        self.head_dim = int(head_dim)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def _build_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self._cached_cos = cos.to(dtype=dtype)
        self._cached_sin = sin.to(dtype=dtype)
        self._cached_seq_len = int(seq_len)

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rebuild = (
            self._cached_cos is None
            or self._cached_sin is None
            or self._cached_seq_len < seq_len
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        )
        if rebuild:
            self._build_cache(seq_len=seq_len, device=device, dtype=dtype)
        assert self._cached_cos is not None and self._cached_sin is not None
        return (
            self._cached_cos[:, :, :seq_len, :],
            self._cached_sin[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.stack((-x2, x1), dim=-1)
    return y.flatten(start_dim=-2)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = int(ff_mult * d_model)
        self.in_proj = nn.Linear(d_model, 2 * hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.out_proj(x)
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.attn_dropout = float(dropout)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,H,T,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(T, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(B, T, self.d_model)
        y = self.out_proj(y)
        return self.resid_dropout(y)


class ModernDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ChessCausalTransformer(nn.Module):
    """
    Input: moves [B,T]
    Output: grouped logits aligned per step t, using only moves <= t.

    Heads grouped by slot vocab size:
      - pieces: 64 slots, 13 classes each
      - side: 1 slot, 2 classes
      - castling: 4 slots, 2 classes each
      - ep_file: 1 slot, 9 classes
      - ep_rank: 1 slot, 3 classes
      - halfmove bytes: 2 slots, 256 classes each
      - fullmove bytes: 2 slots, 256 classes each
    """

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
        n_layers: int,
        n_heads: int,
        dropout: float,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.move_vocab = int(move_vocab)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_mult = int(ff_mult)
        self.tok_emb = nn.Embedding(self.move_vocab, self.d_model)
        self.emb_dropout = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            [
                ModernDecoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    ff_mult=self.ff_mult,
                    dropout=float(dropout),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_norm = RMSNorm(self.d_model)

        total_logits = sum(self._HEAD_SIZES.values())
        self.output_head = nn.Linear(self.d_model, total_logits)

    def forward(self, moves: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T = moves.shape
        x = self.emb_dropout(self.tok_emb(moves))
        for block in self.blocks:
            x = block(x)
        h = self.final_norm(x)

        logits = self.output_head(h)
        pieces, side, castle, ep_file, ep_rank, halfmove, fullmove = torch.split(
            logits,
            [
                self._HEAD_SIZES["pieces"],
                self._HEAD_SIZES["side"],
                self._HEAD_SIZES["castle"],
                self._HEAD_SIZES["ep_file"],
                self._HEAD_SIZES["ep_rank"],
                self._HEAD_SIZES["halfmove"],
                self._HEAD_SIZES["fullmove"],
            ],
            dim=-1,
        )

        return {
            "pieces": pieces.view(B, T, 64, 13),
            "side": side,
            "castle": castle.view(B, T, 4, 2),
            "ep_file": ep_file,
            "ep_rank": ep_rank,
            "halfmove": halfmove.view(B, T, 2, 256),
            "fullmove": fullmove.view(B, T, 2, 256),
        }
