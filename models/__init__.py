from .slice import ChessSLiCEStateModel
from .transformer import ChessCausalTransformer

__all__ = [
    "ChessCausalTransformer",
    "ChessGatedDeltaNetStateModel",
    "ChessMambaStateModel",
    "ChessSLiCEStateModel",
]


def __getattr__(name: str):
    if name == "ChessGatedDeltaNetStateModel":
        from .gated_deltanet import ChessGatedDeltaNetStateModel

        return ChessGatedDeltaNetStateModel
    if name == "ChessMambaStateModel":
        from .mamba import ChessMambaStateModel

        return ChessMambaStateModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
