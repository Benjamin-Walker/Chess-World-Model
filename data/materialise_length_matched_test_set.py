from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import BinaryIO, TypedDict

import chess
import numpy as np
import orjson

from data.process import START_MOVE, encode_state, move_id_to_uci


EXPECTED_GAMES = 10_000
EXPECTED_PLIES = 689_713
EXPECTED_PREFIXES = 699_713
EXPECTED_MIN_PLIES = 20
EXPECTED_MAX_PLIES = 400
EXPECTED_MATCHING_SEED = 20_260_724


class AssignmentRow(TypedDict):
    matched_game_index: int
    human_game_index: int
    human_example_id: str
    human_length_plies: int
    random_game_index: int
    random_example_id: str
    random_original_length_plies: int
    random_truncated_length_plies: int
    matching_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Materialise the released length-matched random test split."
    )
    parser.add_argument("--human", required=True, type=Path)
    parser.add_argument("--random", required=True, type=Path)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    return parser.parse_args()


def file_digests(path: Path) -> dict[str, object]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return {
        "name": path.name,
        "bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def array_sha256(values: list[int]) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def read_assignment(path: Path) -> list[AssignmentRow]:
    required = {
        "matched_game_index",
        "human_game_index",
        "human_example_id",
        "human_length_plies",
        "random_game_index",
        "random_example_id",
        "random_original_length_plies",
        "random_truncated_length_plies",
        "matching_seed",
    }
    rows: list[AssignmentRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Assignment is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                AssignmentRow(
                    matched_game_index=int(row["matched_game_index"]),
                    human_game_index=int(row["human_game_index"]),
                    human_example_id=row["human_example_id"],
                    human_length_plies=int(row["human_length_plies"]),
                    random_game_index=int(row["random_game_index"]),
                    random_example_id=row["random_example_id"],
                    random_original_length_plies=int(
                        row["random_original_length_plies"]
                    ),
                    random_truncated_length_plies=int(
                        row["random_truncated_length_plies"]
                    ),
                    matching_seed=int(row["matching_seed"]),
                )
            )

    if len(rows) != EXPECTED_GAMES:
        raise ValueError(f"Expected {EXPECTED_GAMES} assignments, found {len(rows)}")
    if [row["matched_game_index"] for row in rows] != list(range(EXPECTED_GAMES)):
        raise ValueError("matched_game_index must be contiguous and ordered")
    if [row["human_game_index"] for row in rows] != list(range(EXPECTED_GAMES)):
        raise ValueError("human_game_index must preserve human test-set order")
    random_indices = [row["random_game_index"] for row in rows]
    if sorted(random_indices) != list(range(EXPECTED_GAMES)):
        raise ValueError("Assignment must use every random source trajectory once")
    if {row["matching_seed"] for row in rows} != {EXPECTED_MATCHING_SEED}:
        raise ValueError("Unexpected matching seed")
    if any(
        not row[id_field]
        for row in rows
        for id_field in ("human_example_id", "random_example_id")
    ):
        raise ValueError("Assignment example IDs must not be empty")
    return rows


def read_jsonl_index(path: Path) -> tuple[list[int], list[str], list[int], bytes]:
    offsets: list[int] = []
    example_ids: list[str] = []
    lengths: list[int] = []
    expected_spec: bytes | None = None
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            record = orjson.loads(line)
            offsets.append(offset)
            example_ids.append(record["example_id"])
            plies = int(record["plies"])
            if int(record["steps"]) != plies + 1:
                raise ValueError(f"steps/plies mismatch at row {len(offsets) - 1}")
            if len(record["moves"]) != plies + 1 or len(record["states"]) != plies + 1:
                raise ValueError(f"array-length mismatch at row {len(offsets) - 1}")
            if record["moves"][0] != START_MOVE:
                raise ValueError(f"missing START token at row {len(offsets) - 1}")
            spec = orjson.dumps(record["spec"], option=orjson.OPT_SORT_KEYS)
            if expected_spec is None:
                expected_spec = spec
            elif spec != expected_spec:
                raise ValueError(f"inconsistent state spec at row {len(offsets) - 1}")
            lengths.append(plies)

    if len(offsets) != EXPECTED_GAMES:
        raise ValueError(
            f"Expected {EXPECTED_GAMES} rows in {path}, found {len(offsets)}"
        )
    if len(set(example_ids)) != EXPECTED_GAMES:
        raise ValueError(f"example_id values are not unique in {path}")
    if expected_spec is None:
        raise ValueError(f"Empty JSONL file: {path}")
    return offsets, example_ids, lengths, expected_spec


def load_record(handle: BinaryIO, offset: int) -> dict:
    handle.seek(offset)
    line = handle.readline()
    if not line:
        raise EOFError(f"Could not read JSONL record at byte offset {offset}")
    return orjson.loads(line)


def validate_replay(record: dict, target_plies: int, matched_index: int) -> int:
    board = chess.Board()
    moves = record["moves"][: target_plies + 1]
    states = record["states"][: target_plies + 1]
    if states[0] != encode_state(board):
        raise ValueError(f"Initial state mismatch for matched row {matched_index}")
    for ply, token in enumerate(moves[1:], start=1):
        move = chess.Move.from_uci(move_id_to_uci(int(token)))
        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal move for matched row {matched_index}, ply {ply}: {move.uci()}"
            )
        board.push(move)
        if states[ply] != encode_state(board):
            raise ValueError(
                f"State alignment mismatch for matched row {matched_index}, ply {ply}"
            )
    return target_plies


def write_public_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "matched_game_index",
        "human_game_index",
        "human_example_id",
        "human_length_plies",
        "random_game_index",
        "random_example_id",
        "random_original_length_plies",
        "random_truncated_length_plies",
        "matching_seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    assignment = read_assignment(args.assignment)
    _, human_ids, human_lengths, human_spec = read_jsonl_index(args.human)
    random_offsets, random_ids, random_lengths, random_spec = read_jsonl_index(
        args.random
    )
    if human_spec != random_spec:
        raise ValueError("Human and random state specifications differ")

    for matched_index, row in enumerate(assignment):
        human_index = row["human_game_index"]
        if row["human_example_id"] != human_ids[human_index]:
            raise ValueError(
                f"Human example ID mismatch for matched row {matched_index}"
            )

    target_lengths = [row["human_length_plies"] for row in assignment]
    if target_lengths != human_lengths:
        raise ValueError("Assignment target lengths do not match the human test set")
    if sum(target_lengths) != EXPECTED_PLIES:
        raise ValueError("Unexpected matched-split ply count")
    if min(target_lengths) != EXPECTED_MIN_PLIES:
        raise ValueError("Unexpected minimum matched trajectory length")
    if max(target_lengths) != EXPECTED_MAX_PLIES:
        raise ValueError("Unexpected maximum matched trajectory length")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    public_rows: list[dict[str, object]] = []
    legal_moves_replayed = 0
    with (
        args.random.open("rb") as random_handle,
        output_temporary.open("wb") as output_handle,
    ):
        for matched_index, row in enumerate(assignment):
            random_index = row["random_game_index"]
            target = row["human_length_plies"]
            source = load_record(random_handle, random_offsets[random_index])
            source_plies = random_lengths[random_index]
            if (
                row["random_example_id"] != random_ids[random_index]
                or row["random_example_id"] != source["example_id"]
            ):
                raise ValueError(
                    f"Random example ID mismatch for matched row {matched_index}"
                )
            if row["random_original_length_plies"] != source_plies:
                raise ValueError(
                    f"Source length mismatch for matched row {matched_index}"
                )
            if row["random_truncated_length_plies"] != target:
                raise ValueError(
                    f"Truncation length mismatch for matched row {matched_index}"
                )
            if source_plies < target:
                raise ValueError(f"Source too short for matched row {matched_index}")

            legal_moves_replayed += validate_replay(source, target, matched_index)
            released = {
                "example_id": source["example_id"],
                "moves": source["moves"][: target + 1],
                "states": source["states"][: target + 1],
                "steps": target + 1,
                "plies": target,
                "fullmoves": (target + 1) // 2,
                "result": "*",
                "spec": source["spec"],
            }
            output_handle.write(orjson.dumps(released))
            output_handle.write(b"\n")
            public_rows.append(
                {
                    "matched_game_index": matched_index,
                    "human_game_index": row["human_game_index"],
                    "human_example_id": row["human_example_id"],
                    "human_length_plies": target,
                    "random_game_index": random_index,
                    "random_example_id": row["random_example_id"],
                    "random_original_length_plies": source_plies,
                    "random_truncated_length_plies": target,
                    "matching_seed": row["matching_seed"],
                }
            )
    output_temporary.replace(args.output)
    write_public_manifest(args.manifest, public_rows)

    if legal_moves_replayed != EXPECTED_PLIES:
        raise ValueError("Unexpected legal replay count")
    if sum(target + 1 for target in target_lengths) != EXPECTED_PREFIXES:
        raise ValueError("Unexpected evaluated-prefix count")

    source_indices = [row["random_game_index"] for row in assignment]
    validation = {
        "construction": (
            "without replacement; human targets processed longest first; shortest "
            "feasible random trajectory selected with seeded tie ordering; each source "
            "trajectory truncated to its paired human length"
        ),
        "matching_seed": EXPECTED_MATCHING_SEED,
        "candidate_pool": "original 10,000-game random-uniform test split",
        "number_human_games": len(human_lengths),
        "number_matched_random_games": len(public_rows),
        "number_unique_random_games": len(set(source_indices)),
        "all_original_random_trajectories_used_once": sorted(source_indices)
        == list(range(EXPECTED_GAMES)),
        "human_and_matched_length_arrays_identical": target_lengths == human_lengths,
        "all_sources_at_least_target_length": all(
            random_lengths[row["random_game_index"]] >= row["human_length_plies"]
            for row in assignment
        ),
        "sum_game_lengths_plies": sum(target_lengths),
        "prefixes_including_initial_state": sum(
            target + 1 for target in target_lengths
        ),
        "minimum_length_plies": min(target_lengths),
        "maximum_length_plies": max(target_lengths),
        "all_moves_legal_and_states_aligned": True,
        "legal_moves_replayed": legal_moves_replayed,
        "initial_state_included": True,
        "truncated_result_marker": "*",
        "human_length_array_sha256": array_sha256(human_lengths),
        "matched_length_array_sha256": array_sha256(target_lengths),
        "random_source_index_array_sha256": array_sha256(source_indices),
        "authoritative_assignment": file_digests(args.assignment),
        "human_source": file_digests(args.human),
        "random_source": file_digests(args.random),
        "matched_jsonl": file_digests(args.output),
        "public_manifest": file_digests(args.manifest),
    }
    args.validation.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.validation.with_suffix(args.validation.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.validation)


if __name__ == "__main__":
    main()
