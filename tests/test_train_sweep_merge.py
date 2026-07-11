"""Tests für apply_sweep_overrides aus scripts/train.py.

Rein funktional — kein Chrome, kein SB3, kein W&B. Prüft nur, dass die flachen
gepunkteten Sweep-Parameter korrekt in die verschachtelte YAML-Config gemerged
werden.
"""

from __future__ import annotations

import copy

from train import apply_sweep_overrides


def _base_config() -> dict:
    return {
        "env": {"id": "QWOP-v1", "kwargs": {}},
        "ppo": {"learning_rate": 3.0e-4, "ent_coef": 0.0, "n_steps": 2048},
        "training": {"total_timesteps": 100_000},
    }


def test_overrides_ppo_keys() -> None:
    cfg = _base_config()
    apply_sweep_overrides(cfg, {"ppo.learning_rate": 1.0e-3, "ppo.ent_coef": 0.05})
    assert cfg["ppo"]["learning_rate"] == 1.0e-3
    assert cfg["ppo"]["ent_coef"] == 0.05
    # nicht angefasste Keys bleiben
    assert cfg["ppo"]["n_steps"] == 2048


def test_overrides_nested_env_kwargs() -> None:
    cfg = _base_config()
    apply_sweep_overrides(cfg, {"env.kwargs.failure_cost": 0})
    assert cfg["env"]["kwargs"]["failure_cost"] == 0
    # id bleibt erhalten
    assert cfg["env"]["id"] == "QWOP-v1"


def test_creates_missing_kwargs_dict() -> None:
    # env ohne kwargs-Key → muss angelegt werden
    cfg = {"env": {"id": "QWOP-v1"}}
    apply_sweep_overrides(cfg, {"env.kwargs.failure_cost": 10})
    assert cfg["env"]["kwargs"]["failure_cost"] == 10


def test_noop_without_sweep_keys() -> None:
    # Nicht-Sweep-Lauf: flat enthält nur die zurückgespiegelte Config, keine
    # gepunkteten Whitelist-Keys → Config bleibt unverändert.
    cfg = _base_config()
    before = copy.deepcopy(cfg)
    apply_sweep_overrides(cfg, {"env": cfg["env"], "ppo": cfg["ppo"]})
    assert cfg == before


def test_ignores_non_whitelisted_dotted_keys() -> None:
    cfg = _base_config()
    apply_sweep_overrides(cfg, {"ppo.clip_range": 0.9, "some.random.key": 1})
    # clip_range steht NICHT in der Whitelist → darf nicht durchgereicht werden
    assert "clip_range" not in cfg["ppo"]


def test_returns_same_mutated_config() -> None:
    cfg = _base_config()
    result = apply_sweep_overrides(cfg, {"ppo.ent_coef": 0.01})
    assert result is cfg
