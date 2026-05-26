from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import TextIO

import chess
from data.process import (
    SPEC,
    START_MOVE,
    encode_state,
    make_shard_path,
    pack_move,
)


def generate_random_game_example(
    rng: random.Random,
    game_idx: int,
    *,
    claim_draw: bool = True,
) -> dict:
    board = chess.Board()

    moves: list[int] = [START_MOVE]
    states: list[list[int]] = [encode_state(board)]

    while not board.is_game_over(claim_draw=claim_draw):
        legal_moves = list(board.legal_moves)
        move = legal_moves[rng.randrange(len(legal_moves))]
        moves.append(pack_move(move))
        board.push(move)
        states.append(encode_state(board))

    plies = len(moves) - 1
    example_id = f"cwm_random_{game_idx:05d}"

    return {
        "example_id": example_id,
        "moves": moves,
        "states": states,
        "steps": len(moves),
        "plies": plies,
        "fullmoves": (plies + 1) // 2,
        "result": board.result(claim_draw=claim_draw),
        "spec": SPEC,
    }


def open_output_shard(out_prefix: str, shard_idx: int) -> TextIO:
    path = make_shard_path(out_prefix, shard_idx)
    print(f"[shard] writing {path}", file=sys.stderr)
    return open(path, "w", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output prefix or directory for sharded JSONL. "
            "Use '-' for stdout (no sharding)."
        ),
    )
    ap.add_argument(
        "--games",
        type=int,
        default=10_000,
        help="Number of kept games to write (default: 10000).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for uniform legal-move sampling (default: 0).",
    )
    ap.add_argument(
        "--min-fullmoves",
        type=int,
        default=10,
        help="Keep only games with at least this many full moves (default: 10).",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Max games per output JSONL file (default: 10000).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress every N parsed games (default: 1000).",
    )
    args = ap.parse_args()

    min_plies_required = 2 * args.min_fullmoves
    out_is_stdout = args.out == "-"
    if not out_is_stdout:
        if args.out.endswith(os.sep) or (
            os.path.exists(args.out) and os.path.isdir(args.out)
        ):
            os.makedirs(args.out, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rng = random.Random(args.seed)
    parsed = 0
    kept = 0
    skipped_short = 0
    shard_idx = 0
    shard_written = 0
    f_out = sys.stdout if out_is_stdout else open_output_shard(args.out, shard_idx)
    if not out_is_stdout:
        shard_idx += 1

    try:
        while kept < args.games:
            example = generate_random_game_example(rng, parsed)
            parsed += 1
            if example["plies"] < min_plies_required:
                skipped_short += 1
                continue

            if (
                (not out_is_stdout)
                and (args.shard_size > 0)
                and (shard_written >= args.shard_size)
            ):
                f_out.close()
                f_out = open_output_shard(args.out, shard_idx)
                shard_idx += 1
                shard_written = 0

            f_out.write(json.dumps(example) + "\n")
            kept += 1
            shard_written += 1

            if args.progress_every and (parsed % args.progress_every == 0):
                print(
                    f"parsed={parsed} kept={kept} skipped_short={skipped_short}",
                    file=sys.stderr,
                )

        print(
            f"done: parsed={parsed} kept={kept} skipped_short={skipped_short}",
            file=sys.stderr,
        )
    finally:
        if (not out_is_stdout) and (f_out is not None):
            f_out.close()


if __name__ == "__main__":
    main()
