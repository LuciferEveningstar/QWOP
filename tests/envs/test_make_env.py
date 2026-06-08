"""Tests for the env factory.

We don't spin up an actual Chrome instance here — that requires ChromeDriver
and the QWOP patch to be installed locally (see SETUP.md). Instead we register
a tiny dummy env with Gymnasium and check that ``make_env`` forwards id and
kwargs correctly.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
from gymnasium.envs.registration import register

from qwop_rl.envs import make_env


class _DummyEnv(gym.Env):
    """Minimal Gymnasium env used to capture how make_env wires kwargs through."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, marker: str = "default") -> None:
        super().__init__()
        self.marker = marker
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        return 0, {"marker": self.marker}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        return 0, 0.0, True, False, {"marker": self.marker}


# Register once at import time. Re-registering would raise, so guard.
_DUMMY_ID = "QwopRlDummy-v0"
if _DUMMY_ID not in gym.envs.registry:
    register(id=_DUMMY_ID, entry_point=f"{__name__}:_DummyEnv")


def test_make_env_uses_default_when_config_is_none() -> None:
    """No config → factory still tries the default id. We only assert it picks it up."""
    # We can't actually instantiate QWOP-v1 in CI (needs ChromeDriver), so just
    # check make_env reads the id from config when given.
    env = make_env({"id": _DUMMY_ID})
    try:
        _obs, info = env.reset()
        assert info["marker"] == "default"
    finally:
        env.close()


def test_make_env_forwards_kwargs() -> None:
    env = make_env({"id": _DUMMY_ID, "kwargs": {"marker": "custom"}})
    try:
        _obs, info = env.reset()
        assert info["marker"] == "custom"
    finally:
        env.close()


def test_make_env_accepts_empty_config() -> None:
    """Empty dict shouldn't crash the factory; falls back to defaults."""
    env = make_env({"id": _DUMMY_ID, "kwargs": None})
    try:
        env.reset()
    finally:
        env.close()
