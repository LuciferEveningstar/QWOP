"""Reward-Shaping-Wrapper gegen Robben (aufrechte Haltung belohnen).

qwop-gyms Basis-Reward belohnt allein Vorwärtsgeschwindigkeit; Robben am Boden
ist damit das degenerierte Optimum (kein Sturzrisiko). :class:`UprightRewardWrapper`
addiert pro Step einen Bonus für aufrechte Rumpfhaltung — aber multipliziert mit
der Vorwärtsgeschwindigkeit, damit reines Stehenbleiben nichts einbringt.

Der Wrapper liest die (normalisierte) Observation, die ``step()`` ohnehin liefert
(12 Körperteile x 5 Floats ``[pos_x, pos_y, angle, vel_x, vel_y]``):

- Rumpfhöhe   = ``obs[1]``  (torso ist Index 0 → 0*5 + 1)
- Rumpf-vel_x = ``obs[3]``  (0*5 + 3)

Normalisierung (aus qwop_env): ``pos_y`` = ``raw/10`` (Range [-1,1]); ``vel_x`` =
``(raw-20)/40`` → **Stillstand (raw vel_x=0) entspricht -0.5**, NICHT 0. Das
Vorwärts-Gate nutzt daher -0.5 als Nullpunkt.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# Indizes in der normalisierten 60-Float-Obs (torso = Körperteil 0).
_TORSO_POS_Y_IDX = 1
_TORSO_VEL_X_IDX = 3

# Rohgeschwindigkeit 0 → normalisiert (0 - 20) / 40 = -0.5. Nullpunkt fürs
# Vorwärts-Gate: stehen/rückwärts ergeben Gate 0, echtes Vorwärts > 0.
_VEL_X_ZERO_NORM = -0.5


class UprightRewardWrapper(gym.Wrapper):
    """Belohnt aufrechte Haltung, aber nur während Vorwärtsbewegung.

    Parameters
    ----------
    env:
        Das zu umhüllende QWOP-Env (liefert normalisierte Obs).
    upright_weight:
        Gewicht des Aufrecht-Bonus. Größenordnung an den Basis-Reward
        (``v * 0.01`` pro Step) angepasst halten, damit der Bonus nudged statt
        die ±-Terminal-Rewards (Erfolg/Sturz) zu übertönen.
    height_threshold:
        Normalisierte Rumpfhöhe, ab der Bonus greift. In normalisiertem Raum
        (``raw/10``) — muss pro Maschine kalibriert werden (siehe Modul-Doku).
    gate_on_forward:
        Wenn True (Default), wird der Bonus mit der Vorwärtsgeschwindigkeit
        multipliziert → Stehenbleiben bringt nichts (Anti-Reward-Hacking).
        False nur für Ablations-Experimente (riskant: erlaubt Steh-Hack).
    penalty_below:
        Wenn True, wird Robben (Rumpfhöhe UNTER der Schwelle) aktiv bestraft
        (negatives Shaping), statt nur ignoriert. Der Malus ist NICHT gegated,
        damit Stehenbleiben ihm nicht entkommt. Macht Robben unprofitabel statt
        nur „weniger gut". Default False = altes Bonus-only-Verhalten.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        upright_weight: float = 0.5,
        height_threshold: float = 0.0,
        gate_on_forward: bool = True,
        penalty_below: bool = False,
    ) -> None:
        super().__init__(env)
        self.upright_weight = float(upright_weight)
        self.height_threshold = float(height_threshold)
        self.gate_on_forward = bool(gate_on_forward)
        self.penalty_below = bool(penalty_below)

    def _shaping_term(self, obs: Any) -> tuple[float, float]:
        """Berechnet (shaping, torso_height) aus der Observation.

        Zwei Modi:
        - ``penalty_below=False`` (Default): nur Bonus für aufrechte Haltung
          über der Schwelle, gegated mit Vorwärtsgeschwindigkeit. Nie negativ.
        - ``penalty_below=True``: beidseitig — Bonus über der Schwelle (gegated),
          MALUS unter der Schwelle (Robben). Der Malus ist bewusst NICHT gegated,
          sonst könnte der Agent der Strafe durch Stehenbleiben (Gate=0) entkommen
          → er würde nur Robben gegen Rumstehen tauschen.

        Defensiv: liefert (0.0, nan) wenn obs kein numerischer Vektor der
        erwarteten Länge ist (schützt Dummy-/Discrete-Envs in Tests).
        """
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] <= _TORSO_VEL_X_IDX:
            return 0.0, float("nan")

        torso_height = float(arr[_TORSO_POS_Y_IDX])
        delta = torso_height - self.height_threshold

        if self.gate_on_forward:
            norm_vel_x = float(arr[_TORSO_VEL_X_IDX])
            forward_gate = max(0.0, norm_vel_x - _VEL_X_ZERO_NORM)
        else:
            forward_gate = 1.0

        if delta >= 0.0:
            # Aufrecht: Bonus, gegated (nur bei Vorwärtsbewegung).
            shaping = self.upright_weight * delta * forward_gate
        elif self.penalty_below:
            # Robben (unter Schwelle): Malus, NICHT gegated (Stehen entkommt nicht).
            shaping = self.upright_weight * delta
        else:
            # Bonus-only-Modus: unter Schwelle kein Effekt.
            shaping = 0.0

        return shaping, torso_height

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping, torso_height = self._shaping_term(obs)
        # Für Logging/Kalibrierung sichtbar machen (W&B / eval.py).
        info["reward_shaping"] = shaping
        info["torso_height"] = torso_height
        return obs, float(reward) + shaping, terminated, truncated, info
