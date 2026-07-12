"""Tests für UprightRewardWrapper.

Ohne Chrome: ein Stub-Env mit Box(60)-Obs liefert steuerbare Observation +
Reward, sodass die Shaping-Mathematik exakt prüfbar ist. Ein Integrationstest
prüft zusätzlich die Verdrahtung in make_env (env.reward_shaping).
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register

from qwop_rl.envs import make_env
from qwop_rl.envs._reward import UprightRewardWrapper


def _obs(torso_height: float, torso_vel_x: float) -> np.ndarray:
    """Baut einen 60-Float-Obs-Vektor mit gesetzter Torso-Höhe (idx 1) + vel_x (idx 3)."""
    arr = np.zeros(60, dtype=np.float32)
    arr[1] = torso_height
    arr[3] = torso_vel_x
    return arr


class _StubEnv(gym.Env):
    """Env, das einen vorab gesetzten Obs-Vektor + Reward zurückgibt (steuerbar)."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, obs: np.ndarray | None = None, reward: float = 1.0) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(60,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self._obs = obs if obs is not None else np.zeros(60, dtype=np.float32)
        self._reward = reward

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return self._obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        return self._obs, self._reward, True, False, {}


def test_upright_and_forward_adds_bonus() -> None:
    # torso hoch (0.4), vorwärts (vel_x=0.5). Gate = max(0, 0.5-(-0.5)) = 1.0.
    # shaping = 0.5 * max(0, 0.4-0.0) * 1.0 = 0.2
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.4, 0.5), reward=1.0), upright_weight=0.5, height_threshold=0.0
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert reward == pytest.approx(1.2)
    assert info["reward_shaping"] == pytest.approx(0.2)
    assert info["torso_height"] == pytest.approx(0.4)


def test_crawling_below_threshold_no_bonus() -> None:
    # torso niedrig (-0.3) unter threshold 0.0 → height_bonus 0 → kein Shaping.
    env = UprightRewardWrapper(
        _StubEnv(_obs(-0.3, 0.5), reward=1.0), upright_weight=0.5, height_threshold=0.0
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert reward == pytest.approx(1.0)
    assert info["reward_shaping"] == pytest.approx(0.0)


def test_upright_but_standing_still_gated_to_zero() -> None:
    # ANTI-REWARD-HACKING: torso hoch (0.4), aber Stillstand (vel_x=-0.5 = raw 0).
    # Gate = max(0, -0.5-(-0.5)) = 0 → kein Bonus fürs Rumstehen.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.4, -0.5), reward=1.0), upright_weight=0.5, height_threshold=0.0
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert reward == pytest.approx(1.0)
    assert info["reward_shaping"] == pytest.approx(0.0)


def test_backward_motion_no_bonus() -> None:
    # Rückwärts (vel_x=-0.8 < -0.5) → Gate 0.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.4, -0.8), reward=1.0), upright_weight=0.5, height_threshold=0.0
    )
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.0)


def test_gate_disabled_ignores_velocity() -> None:
    # gate_on_forward=False → Bonus geschwindigkeitsunabhängig = 0.5 * 0.4 = 0.2.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.4, -0.5), reward=1.0),
        upright_weight=0.5,
        height_threshold=0.0,
        gate_on_forward=False,
    )
    _obs_out, reward, _t, _tr, _info = env.step(0)
    assert reward == pytest.approx(1.2)


def test_threshold_boundary_gives_zero() -> None:
    # torso genau auf threshold → height_bonus max(0, 0)=0 → kein Shaping.
    # abs-Toleranz wegen float32-Rundung (0.3f32 - 0.3f64 ≈ 6e-9).
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.3, 0.5), reward=1.0), upright_weight=0.5, height_threshold=0.3
    )
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.0, abs=1e-6)


# --- Integration über make_env ---

_SHAPE_ID = "QwopRlShapeDummy-v0"
if _SHAPE_ID not in gym.envs.registry:
    register(id=_SHAPE_ID, entry_point=f"{__name__}:_StubEnv")


def test_make_env_shaping_disabled_by_default() -> None:
    # Kein reward_shaping-Block → Wrapper nicht aktiv → Reward unverändert.
    env = make_env({"id": _SHAPE_ID, "kwargs": {"obs": _obs(0.4, 0.5), "reward": 1.0}})
    try:
        env.reset()
        _o, reward, _t, _tr, _i = env.step(0)
        assert reward == pytest.approx(1.0)
    finally:
        env.close()


def test_make_env_shaping_enabled_adds_bonus() -> None:
    env = make_env(
        {
            "id": _SHAPE_ID,
            "kwargs": {"obs": _obs(0.4, 0.5), "reward": 1.0},
            "reward_shaping": {"enabled": True, "upright_weight": 0.5, "height_threshold": 0.0},
        }
    )
    try:
        env.reset()
        _o, reward, _t, _tr, info = env.step(0)
        assert reward == pytest.approx(1.2)
        assert info["reward_shaping"] == pytest.approx(0.2)
    finally:
        env.close()
