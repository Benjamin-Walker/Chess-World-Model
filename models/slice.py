import torch
import torch.nn as nn

from slices import SLiCELayer

from .transformer import RMSNorm

_PATH_MODES = {"values", "increments"}


class ChessSLiCEStateModel(nn.Module):
    """Thin world-model head on top of stacked prenorm SLiCELayer blocks."""

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
        use_parallel: bool = True,
        chunk_size: int = 256,
        block_size: int = 4,
        diagonal_dense: bool = False,
        init_std: float = 0.01,
        scale: float = 1.0,
        input_dependent_init: bool = False,
        ff_mult: int = 4,
        final_norm: bool = False,
        path_mode: str = "values",
        norm_type: str = "rmsnorm",
        ff_style: str = "mlp",
        ff_activation: str = "gelu",
        dropout_position: str = "residual",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.move_vocab = int(move_vocab)
        self.d_model = int(d_model)

        if num_layers is None:
            raise ValueError("num_layers must be provided.")
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.path_mode = str(path_mode)
        if self.path_mode not in _PATH_MODES:
            raise ValueError(
                f"path_mode must be one of {_PATH_MODES}, got '{self.path_mode}'."
            )
        self.use_final_norm = bool(final_norm)
        self.norm_type = str(norm_type)
        self.ff_style = str(ff_style)
        self.ff_activation = str(ff_activation)
        self.dropout_position = str(dropout_position)
        self.norm_eps = float(norm_eps)

        self.embedding = nn.Embedding(self.move_vocab, self.d_model)
        self.embedding_dropout = nn.Dropout(float(dropout))
        self.layers = nn.ModuleList(
            [
                SLiCELayer(
                    input_dim=self.d_model,
                    block_size=int(block_size),
                    diagonal_dense=bool(diagonal_dense),
                    init_std=float(init_std),
                    scale=float(scale),
                    input_dependent_init=bool(input_dependent_init),
                    use_parallel=bool(use_parallel),
                    chunk_size=int(chunk_size),
                    dropout_rate=float(dropout),
                    path_mode=self.path_mode,
                    norm_type=self.norm_type,
                    prenorm=True,
                    second_norm=True,
                    ff_style=self.ff_style,
                    ff_activation=self.ff_activation,
                    ff_mult=int(ff_mult),
                    dropout_position=self.dropout_position,
                    norm_eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = (
            RMSNorm(self.d_model) if self.use_final_norm else nn.Identity()
        )
        total_logits = sum(self._HEAD_SIZES.values())
        self.output_head = nn.Linear(self.d_model, total_logits)

    def forward(self, moves: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, seq_len = moves.shape

        h = self.embedding_dropout(self.embedding(moves))
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)

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
            "pieces": pieces.view(bsz, seq_len, 64, 13),
            "side": side,
            "castle": castle.view(bsz, seq_len, 4, 2),
            "ep_file": ep_file,
            "ep_rank": ep_rank,
            "halfmove": halfmove.view(bsz, seq_len, 2, 256),
            "fullmove": fullmove.view(bsz, seq_len, 2, 256),
        }
