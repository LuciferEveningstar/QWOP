"""Gymnasium environments wrapping QWOP.

Currently a thin factory around ``smanolloff/qwop-gym`` (env id ``QWOP-v1``).
Importing :mod:`qwop_gym` registers the environment with Gymnasium as a side
effect, so callers only need to invoke :func:`make_env`.

The qwop-gym ``QWOP-v1`` env requires per-machine paths (``browser``,
``driver``, …) at construction time. ``qwop-gym bootstrap`` writes those into
``./config/env.yml`` — :func:`make_env` reads that file as a default and lets
the YAML training config override individual keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

# Importing qwop_gym registers "QWOP-v1" with Gymnasium. Done at module level
# so any caller of make_env() sees the registration; the import is cheap.
import qwop_gym  # noqa: F401
import yaml
from stable_baselines3.common.monitor import Monitor

DEFAULT_ENV_ID = "QWOP-v1"
DEFAULT_QWOP_GYM_CONFIG = Path("config/env.yml")


def _load_qwop_gym_kwargs(path: Path = DEFAULT_QWOP_GYM_CONFIG) -> dict[str, Any]:
    """Read the qwop-gym env config (browser/driver paths, render_mode, …).

    Returns an empty dict if the file doesn't exist — useful for unit tests
    that pass dummy env ids and don't need the real Chrome stack.
    """
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data


def make_env(config: dict[str, Any] | None = None) -> gym.Env:
    """Build a QWOP Gymnasium environment from a config dict.

    The returned env is wrapped in :class:`stable_baselines3.common.monitor.Monitor`
    so SB3 logs ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean`` automatically.

    Parameters
    ----------
    config:
        Optional mapping with two recognised keys:

        - ``id``: Gymnasium env id, defaults to ``"QWOP-v1"``.
        - ``kwargs``: passed straight to :func:`gymnasium.make`. For the real
          ``QWOP-v1`` env, defaults from ``config/env.yml`` (created by
          ``qwop-gym bootstrap``) are merged in first, so individual keys can
          be overridden per training run.
        - ``reward_shaping``: optional mapping to enable
          :class:`qwop_rl.envs._reward.UprightRewardWrapper` (keys ``enabled``,
          ``upright_weight``, ``height_threshold``, ``gate_on_forward``).
          Absent or ``enabled: false`` → no shaping, unchanged behavior.

        Anything else is ignored so YAML configs can carry extra metadata.
    """
    config = config or {}
    env_id = config.get("id", DEFAULT_ENV_ID)
    user_kwargs = config.get("kwargs") or {}

    # For the real qwop-gym env we want the bootstrap defaults (browser path,
    # driver path, reward weights, …) without making every YAML training
    # config repeat them. Tests that pass non-QWOP ids skip this step.
    if env_id == DEFAULT_ENV_ID:
        merged: dict[str, Any] = _load_qwop_gym_kwargs()
        merged.update(user_kwargs)
        kwargs = merged
    else:
        kwargs = dict(user_kwargs)

    env = gym.make(env_id, **kwargs)

    # Optionales Reward-Shaping VOR Monitor einziehen, damit der Bonus in
    # rollout/ep_rew_mean auftaucht. Fehlt der Block oder enabled=false, bleibt
    # das Verhalten unverändert (rückwärtskompatibel — keine bestehende Config
    # muss angepasst werden).
    shaping_cfg = config.get("reward_shaping") or {}
    shaping_enabled = bool(shaping_cfg.get("enabled"))
    if shaping_enabled:
        from qwop_rl.envs._reward import UprightRewardWrapper

        env = UprightRewardWrapper(
            env,
            upright_weight=float(shaping_cfg.get("upright_weight", 0.5)),
            height_threshold=float(shaping_cfg.get("height_threshold", 0.0)),
            gate_on_forward=bool(shaping_cfg.get("gate_on_forward", True)),
            penalty_below=bool(shaping_cfg.get("penalty_below", False)),
        )

    # Faire Vergleichsmetriken pro Episode nach W&B/TensorBoard durchreichen
    # (Monitor loggt info_keywords in ep_info → SB3 → sync_tensorboard → W&B).
    # NUR für die echte QWOP-Env, deren _build_info diese Keys garantiert
    # liefert; torso_height nur wenn der Shaping-Wrapper aktiv ist. Für Test-
    # Dummy-Envs leer lassen, sonst crasht Monitor mit KeyError.
    info_keywords: tuple[str, ...] = ()
    if env_id == DEFAULT_ENV_ID:
        info_keywords = ("distance", "avgspeed", "is_success")
        if shaping_enabled:
            info_keywords += ("torso_height",)

    return Monitor(env, info_keywords=info_keywords)
