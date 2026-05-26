from __future__ import annotations

import glob
import hashlib
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import orjson
import torch
from torch.utils.data import IterableDataset, get_worker_info


def stable_id_bucket(id_str: str, mod: int = 10000) -> int:
    h = hashlib.md5(id_str.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "little", signed=False)
    return x % mod


def find_shards(data_path: str) -> list[str]:
    """
    Accepts:
      - a single JSONL shard file
      - a directory containing shards (recursively searches for chess_*.jsonl)
      - a prefix (glob prefix + *.jsonl)
    """
    if os.path.isfile(data_path):
        return [data_path]

    if os.path.isdir(data_path):
        pattern = os.path.join(os.path.abspath(data_path), "**", "chess_*.jsonl")
        shards = sorted(set(glob.glob(pattern, recursive=True)))
        if not shards:
            raise FileNotFoundError(f"No shards found under directory: {data_path}")
        return shards

    shards = sorted(set(glob.glob(data_path + "*.jsonl")))
    if not shards:
        raise FileNotFoundError(f"No shards found for prefix: {data_path}*.jsonl")
    return shards


def load_spec_from_first_line(shard_path: str) -> dict:
    with open(shard_path, "rb") as f:
        line = f.readline()
        if not line:
            raise RuntimeError(f"Empty shard: {shard_path}")
        obj = orjson.loads(line)
    if "spec" not in obj:
        raise KeyError("Expected key 'spec' in JSON line.")
    return obj["spec"]


class ChessJsonlIterableDataset(IterableDataset):
    """
    Streams JSONL shards line by line.
    Supports:
      - shard shuffling per epoch
      - truncation by max_steps
      - deterministic train/val split by stable hash of example_id
    """

    def __init__(
        self,
        shards: list[str],
        seed: int = 0,
        shuffle_shards: bool = True,
        max_steps: Optional[int] = None,
        split: str = "train",  # "train" or "val" or "all"
        val_fraction: float = 0.01,
        split_mod: int = 10000,
    ):
        super().__init__()
        self.shards = list(shards)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.max_steps = max_steps
        self.split = split
        self.val_fraction = float(val_fraction)
        self.split_mod = int(split_mod)
        self._epoch = 0

        if self.split not in ("train", "val", "all"):
            raise ValueError("split must be one of: train, val, all")
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0, 1).")

        self.val_k = int(round(self.val_fraction * self.split_mod))

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _keep_example(self, ex_id: str) -> bool:
        if self.split == "all":
            return True
        b = stable_id_bucket(ex_id, mod=self.split_mod)
        is_val = b < self.val_k
        return is_val if self.split == "val" else (not is_val)

    def _iter_examples_from_shard(self, shard_path: str) -> Iterable[dict]:
        with open(shard_path, "rb") as f:
            for line in f:
                if not line:
                    continue
                obj = orjson.loads(line)
                ex_id = obj["example_id"]
                if not self._keep_example(ex_id):
                    continue

                moves = obj["moves"]
                states = obj["states"]
                steps = int(obj["steps"])
                if self.max_steps is not None and steps > self.max_steps:
                    steps = self.max_steps
                    moves = moves[:steps]
                    states = states[:steps]

                yield {
                    "example_id": ex_id,
                    "moves": moves,
                    "states": states,
                    "steps": steps,
                }

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers

        shards = list(self.shards)
        if self.shuffle_shards:
            rng = random.Random(self.seed + 1009 * self._epoch)
            rng.shuffle(shards)

        # shard-level split across workers
        shards = shards[worker_id::num_workers]

        for sp in shards:
            yield from self._iter_examples_from_shard(sp)


@dataclass
class Batch:
    moves: torch.Tensor  # [B, T]
    states: torch.Tensor  # [B, T, 75]
    pad_mask: torch.Tensor  # [B, T] True where padded
    lengths: torch.Tensor  # [B]


def make_collate_fn(pad_move_id: int, max_steps: Optional[int]):
    def collate(batch: list[dict]) -> Batch:
        if len(batch) == 0:
            raise RuntimeError("Empty batch received.")

        lengths = [int(ex["steps"]) for ex in batch]
        if max_steps is not None:
            lengths = [min(L, max_steps) for L in lengths]
        T = max(lengths)

        B = len(batch)
        moves = torch.full((B, T), int(pad_move_id), dtype=torch.long)
        states = torch.zeros((B, T, 75), dtype=torch.long)
        pad_mask = torch.ones((B, T), dtype=torch.bool)

        for i, ex in enumerate(batch):
            L = lengths[i]
            mv = ex["moves"][:L]
            st = ex["states"][:L]
            moves[i, :L] = torch.as_tensor(mv, dtype=torch.long)
            states[i, :L] = torch.as_tensor(st, dtype=torch.long)
            pad_mask[i, :L] = False

        return Batch(
            moves=moves,
            states=states,
            pad_mask=pad_mask,
            lengths=torch.tensor(lengths, dtype=torch.long),
        )

    return collate
