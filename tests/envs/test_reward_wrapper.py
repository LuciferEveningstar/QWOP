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


def test_penalty_below_punishes_crawling() -> None:
    # penalty_below=True: torso 0.30 unter threshold 0.40 → delta -0.10.
    # Malus = weight 10 * -0.10 = -1.0 → reward 1.0 - 1.0 = 0.0.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.30, 0.5), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        penalty_below=True,
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(-1.0, abs=1e-5)
    assert reward == pytest.approx(0.0, abs=1e-5)


def test_penalty_below_not_gated_by_standing() -> None:
    # Malus greift AUCH beim Stehen (vel_x=-0.5): Robben-Strafe ist nicht gegated,
    # sonst könnte der Agent ihr durch Rumstehen entkommen.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.30, -0.5), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        penalty_below=True,
    )
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(-1.0, abs=1e-5)


def test_penalty_below_false_ignores_crawling() -> None:
    # Ohne penalty_below: unter Schwelle kein Effekt (altes Verhalten).
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.30, 0.5), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        penalty_below=False,
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.0)
    assert reward == pytest.approx(1.0)


def test_penalty_below_still_rewards_upright() -> None:
    # penalty_below ändert den Bonus-Fall nicht: aufrecht+vorwärts → positiver Bonus.
    # torso 0.6 über threshold 0.40 → delta 0.2, gate 1.0 → 10 * 0.2 = 2.0.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.60, 0.5), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        penalty_below=True,
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(2.0)
    assert reward == pytest.approx(3.0)


def test_speed_bonus_rewards_forward_velocity() -> None:
    # obs[3]=-0.125 → raw vel_x = -0.125*40+20 = 15 m/s. speed_weight=0.1 → +1.5.
    # torso 0.30 unter threshold 0.40, ohne penalty → Torso-Shaping 0, nur Speed.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.30, -0.125), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        speed_weight=0.1,
    )
    _obs_out, reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(1.5)
    assert reward == pytest.approx(2.5)


def test_speed_bonus_zero_when_standing() -> None:
    # Stehen: obs[3]=-0.5 → raw = 0 m/s → max(0,0) → kein Speed-Bonus.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.30, -0.5), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        speed_weight=0.1,
    )
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.0)


def test_speed_bonus_adds_to_torso_shaping() -> None:
    # Aufrecht (torso 0.60, delta 0.20, gate: raw 15→norm-0.125, gate=0.375)
    # Torso-Bonus = 10 * 0.20 * 0.375 = 0.75; Speed = 0.1*15 = 1.5; Summe 2.25.
    env = UprightRewardWrapper(
        _StubEnv(_obs(0.60, -0.125), reward=1.0),
        upright_weight=10.0,
        height_threshold=0.40,
        penalty_below=True,
        speed_weight=0.1,
    )
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.75 + 1.5)


def test_gait_bonus_rewards_thigh_angle_diff() -> None:
    # Thigh-Winkel: leftThigh idx 32, rightThigh idx 57. Diff |0.8-(-0.4)|=1.2.
    # obs[3]=-0.125 → gate = -0.125-(-0.5) = 0.375. gait_weight 0.5.
    # gait-Bonus = 0.5 * 1.2 * 0.375 = 0.225. Torso: torso 0.30 unter 0.45,
    # ohne penalty → 0. Nur gait.
    obs = _obs(0.30, -0.125)
    obs[32] = 0.8
    obs[57] = -0.4
    env = UprightRewardWrapper(
        _StubEnv(obs, reward=1.0),
        upright_weight=10.0,
        height_threshold=0.45,
        gait_weight=0.5,
    )
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.5 * 1.2 * 0.375)


def test_gait_bonus_zero_when_standing() -> None:
    # Beine gespreizt (diff 1.2), aber Stehen (obs[3]=-0.5 → gate 0) → kein Bonus.
    obs = _obs(0.30, -0.5)
    obs[32] = 0.8
    obs[57] = -0.4
    env = UprightRewardWrapper(_StubEnv(obs, reward=1.0), gait_weight=0.5, height_threshold=0.45)
    _obs_out, _reward, _t, _tr, info = env.step(0)
    assert info["reward_shaping"] == pytest.approx(0.0)


class _DistEnv(gym.Env):
    """Stub, der distance/time in info liefert und mehrere Steps laeuft."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(60,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self._steps = 0

    def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(60, dtype=np.float32), {"distance": 0.0, "time": 0.0}

    def step(self, action):  # type: ignore[no-untyped-def]
        self._steps += 1
        # pro Step: +2m in +0.5s → v=4 m/s
        info = {"distance": 2.0 * self._steps, "time": 0.5 * self._steps, "is_success": False}
        return np.zeros(60, dtype=np.float32), 99.0, False, False, info


def test_replace_base_decouples_speed_from_distance() -> None:
    # replace_base: eigener Reward = distance_weight*Δdist - time_penalty*Δzeit.
    # Erster Step nach reset hat keine Vergangenheit (Δ=0) → 2. Step prüfen.
    # Δdist=2, Δzeit=0.5. distance_weight=0.01 → 0.02; time_penalty=1.0 → -0.5.
    # → -0.48. Der originale reward (99.0) wird IGNORIERT (entkoppelt).
    env = UprightRewardWrapper(
        _DistEnv(), replace_base=True, distance_weight=0.01, time_penalty=1.0
    )
    env.reset()
    env.step(0)  # erster Step: Δ=0 (Referenz)
    _obs_out, reward, _t, _tr, _info = env.step(0)
    assert reward == pytest.approx(0.01 * 2.0 - 1.0 * 0.5)


def test_replace_base_off_keeps_original() -> None:
    # Ohne replace_base bleibt der Original-Reward (99.0) erhalten.
    env = UprightRewardWrapper(_DistEnv(), replace_base=False)
    env.reset()
    _obs_out, reward, _t, _tr, _info = env.step(0)
    assert reward == pytest.approx(99.0)


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
