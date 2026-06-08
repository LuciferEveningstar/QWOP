"""Gymnasium environments wrapping QWOP.

Currently a thin factory around ``smanolloff/qwop-gym`` (env id ``QWOP-v1``).
Importing :mod:`qwop_gym` registers the environment with Gymnasium as a side
effect, so callers only need to invoke :func:`make_env`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

# Importing qwop_gym registers "QWOP-v1" with Gymnasium. Done at module level
# so any caller of make_env() sees the registration; the import is cheap.
import qwop_gym  # noqa: F401

DEFAULT_ENV_ID = "QWOP-v1"


def make_env(config: dict[str, Any] | None = None) -> gym.Env:
    """Build a QWOP Gymnasium environment from a config dict.

    Parameters
    ----------
    config:
        Optional mapping with two recognised keys:

        - ``id``: Gymnasium env id, defaults to ``"QWOP-v1"``.
        - ``kwargs``: passed straight to :func:`gymnasium.make`.

        Anything else is ignored so YAML configs can carry extra metadata.
    """
    config = config or {}
    env_id = config.get("id", DEFAULT_ENV_ID)
    kwargs = config.get("kwargs") or {}
    return gym.make(env_id, **kwargs)
