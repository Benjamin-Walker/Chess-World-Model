from __future__ import annotations

import hashlib
import hmac
import os

DEFAULT_ID_KEY_ENV = "CHESS_WORLD_MODEL_ID_KEY"


def load_private_id_key(env_var: str = DEFAULT_ID_KEY_ENV) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(f"Set {env_var} to generate stable private example IDs.")
    return key


def make_example_id(
    source_id: str,
    *,
    key: str | None = None,
    env_var: str = DEFAULT_ID_KEY_ENV,
    prefix: str = "cwm_",
) -> str:
    if not source_id:
        raise ValueError("source_id must be non-empty")
    if key is None:
        key = load_private_id_key(env_var=env_var)
    digest = hmac.new(
        key.encode("utf-8"),
        source_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{prefix}{digest[:24]}"
