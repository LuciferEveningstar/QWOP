"""Test für ExtraRolloutMetricsCallback.

Prüft, dass die Extra-Episode-Metriken (distance/avgspeed/torso_height) aus dem
ep_info_buffer als rollout/<key> geloggt werden — ohne SB3-Training/Chrome.
"""

from __future__ import annotations

from collections import deque

import pytest

from qwop_rl.utils.callbacks import ExtraRolloutMetricsCallback


class _FakeLogger:
    def __init__(self) -> None:
        self.recorded: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.recorded[key] = value


class _FakeModel:
    def __init__(self, buffer: deque) -> None:
        self.logger = _FakeLogger()
        self.ep_info_buffer = buffer


def test_logs_extra_keys_as_rollout() -> None:
    model = _FakeModel(
        deque(
            [
                {"r": 250.0, "l": 3000, "distance": 100.0, "avgspeed": 9.7, "torso_height": 0.42},
                {"r": 248.0, "l": 3100, "distance": 101.0, "avgspeed": 9.5, "torso_height": 0.40},
            ]
        )
    )
    cb = ExtraRolloutMetricsCallback()
    cb.model = model  # BaseCallback.logger delegiert an model.logger
    cb._on_rollout_end()

    rec = model.logger.recorded
    assert rec["rollout/distance"] == pytest.approx(100.5)
    assert rec["rollout/avgspeed"] == pytest.approx(9.6)
    assert rec["rollout/torso_height"] == pytest.approx(0.41)


def test_skips_missing_keys() -> None:
    # Kein torso_height (Shaping aus) → Key wird übersprungen, kein Fehler.
    model = _FakeModel(deque([{"r": 1.0, "l": 10, "distance": 5.0, "avgspeed": 1.0}]))
    cb = ExtraRolloutMetricsCallback()
    cb.model = model
    cb._on_rollout_end()
    assert "rollout/distance" in model.logger.recorded
    assert "rollout/torso_height" not in model.logger.recorded


def test_empty_buffer_no_crash() -> None:
    model = _FakeModel(deque())
    cb = ExtraRolloutMetricsCallback()
    cb.model = model
    cb._on_rollout_end()  # darf nicht crashen
    assert model.logger.recorded == {}
