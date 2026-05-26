"""
Build an online seq->seq chess dataset

moves:   [steps] packed move IDs with a leading START token
states:  [steps, 75] fixed-size state tokens aligned with moves

Alignment:
  moves[0]  = START
  states[0] = s0 (initial position)
  moves[t]  = m_t
  states[t] = s_t (position after applying m_t)

Time accounting:
  steps = len(moves) = T + 1
  plies = steps - 1  = T   (real half-moves in the game)

Output:
  Sharded JSONL: up to --shard-size games per .jsonl file (default 1,000,000)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TextIO

import chess
import chess.pgn
from data.example_ids import load_private_id_key, make_example_id


def make_shard_path(out_prefix: str, shard_idx: int) -> str:
    """
    out_prefix may be:
      - /path/to/name.jsonl
      - /path/to/name
      - /path/to/dir/   (directory)

    We will write: <prefix>_<shard_idx:06d>.jsonl
    """
    # If out_prefix looks like a directory, use a default base name.
    if out_prefix.endswith(os.sep) or (
        os.path.exists(out_prefix) and os.path.isdir(out_prefix)
    ):
        out_dir = out_prefix.rstrip(os.sep)
        base = "chess"
        prefix = os.path.join(out_dir, base)
    else:
        root, ext = os.path.splitext(out_prefix)
        if ext.lower() == ".jsonl":
            prefix = root
        else:
            prefix = out_prefix

    return f"{prefix}_{shard_idx:06d}.jsonl"


# ----------------------------
# Token specs
# ----------------------------

# State[0:64]: piece id on a8..h1
# piece ids: 0=".", 1..6 = "PNBRQK", 7..12 = "pnbrqk"
PIECE_SYMBOLS = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
PIECE_TO_ID = {s: i for i, s in enumerate(PIECE_SYMBOLS)}

# State layout:
#   [0:64] pieces (a8..h1), each in 0..12
#   [64]   side (0=w, 1=b)
#   [65:69] castling bits K,Q,k,q each in {0,1}
#   [69] ep_file in {0:"-", 1:"a", ..., 8:"h"}
#   [70] ep_rank in {0:"-", 1:"3", 2:"6"}
# State[71:73]: halfmove clock uint16 split into (hi, lo) bytes, each 0..255
# State[73:75]: fullmove number uint16 split into (hi, lo) bytes, each 0..255
STATE_LEN = 75
FILES = "abcdefgh"

# Square order for 64 squares: a8..h8, a7..h7, ..., a1..h1
SQUARES_A8_TO_H1 = [
    chess.square(file, rank)
    for rank in reversed(range(8))  # 7..0
    for file in range(8)  # 0..7
]


# ----------------------------
# Move packing
# ----------------------------

# promo_id: 0 none, 1 q, 2 r, 3 b, 4 n
PROMO_TO_ID = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

MOVE_VOCAB = 64 * 64 * 5  # 20480 real moves
START_MOVE = MOVE_VOCAB
PAD_MOVE = MOVE_VOCAB + 1
MOVE_VOCAB_WITH_SPECIALS = MOVE_VOCAB + 2
SPEC = {
    "move_vocab": MOVE_VOCAB_WITH_SPECIALS,
    "start_move_id": START_MOVE,
    "pad_move_id": PAD_MOVE,
    "state_len": STATE_LEN,
    "piece_symbols": PIECE_SYMBOLS,
    "square_order": "a8..h8,a7..h7,...,a1..h1",
    "state_layout": {
        "pieces_0_63": "piece_id in 0..12 over PIECE_SYMBOLS",
        "side_64": "0=w,1=b",
        "castle_65_68": "K,Q,k,q bits in {0,1}",
        "ep_file_69": "0='-',1..8='a'..'h'",
        "ep_rank_70": "0='-',1='3',2='6'",
        "halfmove_71_72": "uint16 hi,lo (0..65535)",
        "fullmove_73_74": "uint16 hi,lo (0..65535)",
    },
}


def pack_move(mv: chess.Move) -> int:
    promo_id = PROMO_TO_ID.get(mv.promotion, 0)
    return (mv.from_square * 64 + mv.to_square) * 5 + promo_id


def move_id_to_uci(move_id: int) -> str:
    if move_id == START_MOVE:
        return "<START>"
    if move_id == PAD_MOVE:
        return "<PAD>"
    p = move_id % 5
    x = move_id // 5
    t = x % 64
    f = x // 64
    u = chess.square_name(f) + chess.square_name(t)
    if p != 0:
        u += {1: "q", 2: "r", 3: "b", 4: "n"}[p]
    return u


# ----------------------------
# State encoding / decoding
# ----------------------------


def encode_state(board: chess.Board) -> list[int]:
    s: list[int] = [0] * STATE_LEN

    for i, sq in enumerate(SQUARES_A8_TO_H1):
        piece = board.piece_at(sq)
        if piece is None:
            s[i] = PIECE_TO_ID["."]
        else:
            sym = piece.symbol()  # lowercase
            if piece.color == chess.WHITE:
                sym = sym.upper()
            s[i] = PIECE_TO_ID[sym]

    s[64] = 0 if board.turn == chess.WHITE else 1

    s[65] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    s[66] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    s[67] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    s[68] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

    if board.ep_square is None:
        s[69] = 0
        s[70] = 0
    else:
        ep_file = chess.square_file(board.ep_square) + 1  # 1..8 for a..h
        ep_rank_idx = chess.square_rank(board.ep_square)  # 2 for rank 3, 5 for rank 6
        if ep_rank_idx == 2:
            s[69] = ep_file
            s[70] = 1
        elif ep_rank_idx == 5:
            s[69] = ep_file
            s[70] = 2
        else:
            s[69] = 0
            s[70] = 0

    # Halfmove clock as uint16 (hi, lo)
    hm = min(max(board.halfmove_clock, 0), 65535)
    s[71] = (hm >> 8) & 0xFF
    s[72] = hm & 0xFF

    # Fullmove number as uint16 (hi, lo)
    fm = min(max(board.fullmove_number, 0), 65535)
    s[73] = (fm >> 8) & 0xFF
    s[74] = fm & 0xFF

    return s


def decode_state_to_board(state: list[int]) -> chess.Board:
    if len(state) != STATE_LEN:
        raise ValueError(f"Expected state length {STATE_LEN}, got {len(state)}")

    try:
        board = chess.Board(None)
    except Exception:
        board = chess.Board()
        board.clear_board()

    for i, sq in enumerate(SQUARES_A8_TO_H1):
        pid = int(state[i])
        sym = PIECE_SYMBOLS[pid]
        if sym != ".":
            colour = chess.WHITE if sym.isupper() else chess.BLACK
            piece_type = chess.Piece.from_symbol(sym.lower()).piece_type
            board.set_piece_at(sq, chess.Piece(piece_type, colour))

    board.turn = chess.WHITE if int(state[64]) == 0 else chess.BLACK

    wk, wq, bk, bq = (int(state[65]), int(state[66]), int(state[67]), int(state[68]))
    rights = 0
    if wk:
        rights |= chess.BB_H1
    if wq:
        rights |= chess.BB_A1
    if bk:
        rights |= chess.BB_H8
    if bq:
        rights |= chess.BB_A8
    board.castling_rights = rights

    ep_file = int(state[69])
    ep_rank = int(state[70])
    if ep_file == 0 or ep_rank == 0:
        board.ep_square = None
    else:
        rank = 3 if ep_rank == 1 else 6
        board.ep_square = chess.parse_square(FILES[ep_file - 1] + str(rank))

    board.halfmove_clock = (int(state[71]) << 8) + int(state[72])
    board.fullmove_number = (int(state[73]) << 8) + int(state[74])

    return board


# ----------------------------
# Extraction
# ----------------------------


def extract_game(game: chess.pgn.Game) -> tuple[list[int], list[list[int]]]:
    board = game.board()

    moves: list[int] = [START_MOVE]
    states: list[list[int]] = [encode_state(board)]  # s0

    for mv in game.mainline_moves():
        moves.append(pack_move(mv))
        board.push(mv)
        states.append(encode_state(board))  # s_t after applying move t

    return moves, states


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Input PGN file (.pgn).")
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output prefix or directory for sharded JSONL. "
            "Use '-' for stdout (no sharding)."
        ),
    )

    ap.add_argument(
        "--min-fullmoves",
        type=int,
        default=10,
        help="Keep only games with at least this many full moves (default: 10).",
    )

    ap.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Stop after writing this many kept games (0 means no limit).",
    )
    ap.add_argument(
        "--max-parsed",
        type=int,
        default=0,
        help="Stop after parsing this many games total (0 means no limit).",
    )

    ap.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
        help=(
            "Max games per output JSONL file (default: 1,000,000). "
            "Ignored if --out is '-'."
        ),
    )

    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--preview-steps", type=int, default=3)
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    min_plies_required = 2 * args.min_fullmoves

    out_is_stdout = args.out == "-"
    if not out_is_stdout:
        # Ensure output directory exists
        # If args.out is a directory, make it; if it is a file prefix, make its parent.
        if args.out.endswith(os.sep) or (
            os.path.exists(args.out) and os.path.isdir(args.out)
        ):
            os.makedirs(args.out, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    parsed = 0
    kept = 0
    skipped_short = 0
    skipped_err = 0
    id_key = load_private_id_key()

    shard_idx = 0
    shard_written = 0
    f_out = None

    def open_new_shard() -> TextIO:
        nonlocal shard_idx, shard_written
        path = make_shard_path(args.out, shard_idx)
        shard_idx += 1
        shard_written = 0
        print(f"[shard] writing {path}", file=sys.stderr)
        return open(path, "w", encoding="utf-8")

    try:
        if out_is_stdout:
            f_out = sys.stdout
        else:
            f_out = open_new_shard()

        with open(args.pgn, encoding="utf-8", newline="") as f_in:
            while True:
                if args.max_parsed and parsed >= args.max_parsed:
                    break

                game = chess.pgn.read_game(f_in)
                if game is None:
                    break

                parsed += 1
                headers = dict(game.headers)
                source_url = headers["Site"].strip()

                try:
                    moves, states = extract_game(game)
                except Exception:
                    skipped_err += 1
                    continue

                steps = len(moves)  # includes START
                plies = steps - 1  # real plies (half-moves)
                if plies < min_plies_required:
                    skipped_short += 1
                    continue

                # Optional sanity check
                if args.sanity_check:
                    board = game.board()

                    # Check s0
                    decoded0 = decode_state_to_board(states[0])
                    if decoded0.fen() != board.fen():
                        skipped_err += 1
                        print(
                            f"[sanity] mismatch at game {source_url} step 0",
                            file=sys.stderr,
                        )
                        print(f"  true: {board.fen()}", file=sys.stderr)
                        print(f"  dec : {decoded0.fen()}", file=sys.stderr)
                        continue

                    ok = True
                    for t, mv in enumerate(game.mainline_moves(), start=1):
                        board.push(mv)
                        decoded = decode_state_to_board(states[t])
                        if decoded.fen() != board.fen():
                            ok = False
                            print(
                                f"[sanity] mismatch at game {source_url} ply {t}",
                                file=sys.stderr,
                            )
                            print(f"  true: {board.fen()}", file=sys.stderr)
                            print(f"  dec : {decoded.fen()}", file=sys.stderr)
                            break
                    if not ok:
                        skipped_err += 1
                        continue

                ex = {
                    "example_id": make_example_id(source_url, key=id_key),
                    "moves": moves,  # [steps] ints, includes START
                    "states": states,  # [steps][75]
                    "steps": steps,
                    "plies": plies,
                    "fullmoves": (plies + 1) // 2,
                    "result": headers.get("Result", ""),
                    "spec": SPEC,
                }

                # Shard rotation
                if (
                    (not out_is_stdout)
                    and (args.shard_size > 0)
                    and (shard_written >= args.shard_size)
                ):
                    f_out.close()
                    f_out = open_new_shard()

                f_out.write(json.dumps(ex) + "\n")
                kept += 1
                shard_written += 1

                # Preview
                if args.preview and kept <= args.preview:
                    k = min(args.preview_steps, steps)
                    print(
                        (
                            f"\n[preview] {source_url} "
                            f"plies={plies} fullmoves={(plies + 1) // 2}"
                        ),
                        file=sys.stderr,
                    )
                    for i in range(k):
                        uci = move_id_to_uci(moves[i])
                        fen = decode_state_to_board(states[i]).fen()
                        print(f"  step {i:02d}: move={uci}  fen={fen}", file=sys.stderr)
                    last_fen = decode_state_to_board(states[-1]).fen()
                    print(f"  last: fen={last_fen}", file=sys.stderr)

                if args.max_games and kept >= args.max_games:
                    break

                if args.progress_every and (parsed % args.progress_every == 0):
                    progress_msg = (
                        f"parsed={parsed} kept={kept} "
                        f"skipped_short={skipped_short} skipped_err={skipped_err}"
                    )
                    print(
                        progress_msg,
                        file=sys.stderr,
                    )

        done_msg = (
            f"done: parsed={parsed} kept={kept} "
            f"skipped_short={skipped_short} skipped_err={skipped_err}"
        )
        print(
            done_msg,
            file=sys.stderr,
        )

    finally:
        if (f_out is not None) and (not out_is_stdout):
            f_out.close()


if __name__ == "__main__":
    main()
