# Chess Experiments

`main.py` trains state-tracking models from move sequences and supports
JSON experiment presets for reproducible runs.

## Setup

This project uses Python 3.12 and `uv` for environment management. Install the
base dependencies from the repository root with:

```bash
uv sync
```

The base environment supports the Transformer and SLiCE configs. Gated DeltaNet
and Mamba use optional dependency groups described below.

## Data

Processed chess data is not included in this code release. To train on human
games, download PGN exports from the Lichess open database and convert them to
the JSONL format expected by `main.py`.

Lichess publishes monthly standard rated games at:

```text
https://database.lichess.org/
```

The standard game files follow this URL pattern:

```text
https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst
```

For example:

```bash
mkdir -p data/raw data/processed

curl -L \
  -o data/raw/lichess_db_standard_rated_2025-04.pgn.zst \
  https://database.lichess.org/standard/lichess_db_standard_rated_2025-04.pgn.zst
```

The repository processor currently reads plain PGN files, so decompress the
archive before processing:

```bash
zstd -d \
  -o data/raw/lichess_db_standard_rated_2025-04.pgn \
  data/raw/lichess_db_standard_rated_2025-04.pgn.zst
```

Generate processed training shards:

```bash
export CHESS_WORLD_MODEL_ID_KEY="$(openssl rand -hex 32)"

uv run python -m data.process \
  --pgn data/raw/lichess_db_standard_rated_2025-04.pgn \
  --out data/processed \
  --min-fullmoves 10 \
  --shard-size 1000000
```

`CHESS_WORLD_MODEL_ID_KEY` is used only to derive stable opaque example IDs
from source game URLs. Keep the same value if you want deterministic IDs and
train/validation splits across repeated preprocessing runs. Do not commit it.

For a quick local smoke test, limit the conversion:

```bash
uv run python -m data.process \
  --pgn data/raw/lichess_db_standard_rated_2025-04.pgn \
  --out data/processed_smoke \
  --max-games 1000 \
  --min-fullmoves 10
```

Then run a small training job:

```bash
uv run python -m main \
  --data_path data/processed_smoke \
  --experiment_config transformer_d128_defaults_simple \
  --epochs 1 \
  --val_every 0 \
  --save_every 0
```

To generate a random legal-move test set:

```bash
uv run python -m data.generate_random_uniform_test_set \
  --out data/test/random_uniform_10k \
  --games 10000 \
  --seed 0
```

Notes:

- Lichess PGN exports are large. A full recent monthly standard-rated file is
  tens of GB compressed and much larger after decompression.
- Lichess also provides `.torrent` downloads on the database page, which are
  often more reliable for full monthly files.
- Processed JSONL files, decompressed PGNs, checkpoints, logs, and test outputs
  are intentionally ignored by git.

## Experiment configs

Presets live in `experiment_configs/`.

Usage:

```bash
uv run python -m main \
  --data_path data/processed \
  --experiment_config mamba_d384_defaults_simple
```

`--experiment_config` accepts:
- preset name in `experiment_configs` (with or without `.json`)
- direct path to a JSON file

CLI flags always override values from the selected config.

Available presets:
- `transformer_d128_defaults_simple`
- `transformer_d256_defaults_simple`
- `transformer_d384_defaults_simple`
- `transformer_d512_defaults_simple`
- `slice_d128_defaults_simple`
- `slice_d256_defaults_simple`
- `slice_d384_defaults_simple`
- `slice_d512_defaults_simple`
- `gated_deltanet_d128_defaults_simple`
- `gated_deltanet_d256_defaults_simple`
- `gated_deltanet_d384_defaults_simple`
- `gated_deltanet_d512_defaults_simple`
- `mamba_d128_defaults_simple`
- `mamba_d256_defaults_simple`
- `mamba_d384_defaults_simple`
- `mamba_d512_defaults_simple`

The transformer presets only pin model size.

The slice presets pin the same size schedule plus these non-default SLiCE
values:
- `slice_block_size = 8`
- `slice_init_std = 0.1`
- `slice_scale = 0.1`

The Gated DeltaNet presets use the same size schedule with `arch =
"gated_deltanet"`, `gated_deltanet_head_dim = 48`, and
`gated_deltanet_allow_neg_eigval = true` on top of the
`flash-linear-attention` dependency.

Gated DeltaNet runs use the optional FLA dependency group:

```bash
uv sync --group fla
```

The mamba presets pin the same size schedule with `arch = "mamba"` and
`mamba_variant = "mamba-3"` by default. You can switch block versions with:

```bash
uv run python -m main \
  --experiment_config mamba_d384_defaults_simple \
  --mamba_variant mamba-2
```

Mamba runs use the optional Mamba dependency group. Sync it from this
directory:

```bash
MAMBA_FORCE_BUILD=TRUE \
uv sync --group mamba --refresh-package mamba-ssm --reinstall-package mamba-ssm
```

This installs the pinned `mamba-ssm` source build and its `causal-conv1d`
extra with build isolation disabled. The repo pins `mamba-ssm` to a Git
revision because the tested PyPI release does not expose
`mamba_ssm.modules.mamba3`. The install command also forces a source build,
since cached or prebuilt wheels can still omit `mamba3.py`.

Verify the install before launching `mamba-3` runs:

```bash
uv run python -c "from mamba_ssm.modules.mamba3 import Mamba3; print(Mamba3.__name__)"
```

If that import still fails after `uv sync --group mamba`, the local environment
has usually kept a cached `mamba-ssm` build without `mamba3.py`. Recover with a
direct no-cache source install:

```bash
MAMBA_FORCE_BUILD=TRUE \
uv pip install --python .venv/bin/python \
  --no-build-isolation \
  --no-deps \
  --force-reinstall \
  --no-cache-dir \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba@b267be48e9e71a3a37310ade04b058625409da2d"
```

## Learning-rate scheduler

Training supports optional per-step scheduling:
- `lr_scheduler`: `none` (default), `cosine`, `linear`
- `lr_warmup_steps`: linear warmup steps
- `lr_decay_steps`: scheduler horizon in optimiser steps (required for non-`none`)
- `lr_min_ratio`: final LR ratio relative to base `lr`

Example:

```bash
uv run python -m main \
  --experiment_config transformer_d256_defaults_simple \
  --batch_size 128 \
  --lr 3e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 5000 \
  --lr_decay_steps 620000 \
  --lr_min_ratio 0.1
```

## Evaluating Checkpoints

Use `evaluation.evaluate_on_test_sets` to evaluate one or more checkpoints on
processed JSONL test sets. The script accepts either checkpoint files or
checkpoint directories containing `latest.pt`.

```bash
uv run python -m evaluation.evaluate_on_test_sets \
  --checkpoint checkpoints/example/latest.pt \
  --dataset lichess=data/test/lichess/real_game_test.jsonl \
  --dataset random=data/test/random_uniform_10k/random_game_test.jsonl
```

Results are written under `test_eval/` by default, with per-checkpoint JSON
files and a `summary.tsv`. Pass `--cpu` to force CPU evaluation, or
`--output_root` to choose another output directory.
