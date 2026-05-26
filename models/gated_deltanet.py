from __future__ import annotations

import torch
import torch.nn as nn

try:
    from fla.models.gated_deltanet.configuration_gated_deltanet import (
        GatedDeltaNetConfig,
    )
    from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetModel

    _FLA_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    GatedDeltaNetConfig = None
    GatedDeltaNetModel = None
    _FLA_IMPORT_ERROR = exc


class ChessGatedDeltaNetStateModel(nn.Module):
    """World-model head built on top of an FLA GatedDeltaNet backbone."""

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
        n_heads: int,
        dropout: float,
        *,
        ff_mult: int = 4,
        pad_move_id: int | None = None,
        head_dim: int = 48,
        expand_v: float = 2.0,
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        attn_mode: str = "chunk",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        del dropout  # FLA GatedDeltaNetModel does not expose dropout knobs.

        if GatedDeltaNetConfig is None or GatedDeltaNetModel is None:
            raise ImportError(
                "FLA GatedDeltaNet support requires `flash-linear-attention`. "
                "Install the optional dependency group with `uv sync --group fla`."
            ) from _FLA_IMPORT_ERROR

        self.move_vocab = int(move_vocab)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.n_heads = int(n_heads)
        self.ff_mult = int(ff_mult)
        self.pad_move_id = None if pad_move_id is None else int(pad_move_id)
        self.head_dim = int(head_dim)
        self.expand_v = float(expand_v)
        self.use_gate = bool(use_gate)
        self.use_short_conv = bool(use_short_conv)
        self.allow_neg_eigval = bool(allow_neg_eigval)
        self.conv_size = int(conv_size)
        self.attn_mode = str(attn_mode)
        self.norm_eps = float(norm_eps)

        self.backbone_config = GatedDeltaNetConfig(
            attn_mode=self.attn_mode,
            hidden_size=self.d_model,
            expand_v=self.expand_v,
            use_gate=self.use_gate,
            use_short_conv=self.use_short_conv,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            head_dim=self.head_dim,
            num_heads=self.n_heads,
            num_v_heads=self.n_heads,
            hidden_ratio=self.ff_mult,
            num_hidden_layers=self.num_layers,
            norm_eps=self.norm_eps,
            use_cache=False,
            vocab_size=self.move_vocab,
            pad_token_id=self.pad_move_id,
        )
        self.backbone = GatedDeltaNetModel(self.backbone_config)
        self.output_head = nn.Linear(self.d_model, sum(self._HEAD_SIZES.values()))
        self._reset_output_head()

    def _reset_output_head(self) -> None:
        std = float(getattr(self.backbone_config, "initializer_range", 0.02))
        nn.init.normal_(self.output_head.weight, mean=0.0, std=std)
        if self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)

    def forward(self, moves: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, seq_len = moves.shape
        attention_mask = None
        if self.pad_move_id is not None:
            attention_mask = moves.ne(self.pad_move_id).to(dtype=torch.long)

        outputs = self.backbone(
            input_ids=moves,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        logits = self.output_head(hidden_states)
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
