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

# Oberschenkel-Winkel für den Gang-Symmetrie-Bonus. OBS_PARTS-Reihenfolge (aus
# extensions.js): leftThigh = Teil 6, rightThigh = Teil 11; angle = Teil*5 + 2.
_LEFT_THIGH_ANGLE_IDX = 6 * 5 + 2  # = 32
_RIGHT_THIGH_ANGLE_IDX = 11 * 5 + 2  # = 57

# Rohgeschwindigkeit 0 → normalisiert (0 - 20) / 40 = -0.5. Nullpunkt fürs
# Vorwärts-Gate: stehen/rückwärts ergeben Gate 0, echtes Vorwärts > 0.
_VEL_X_ZERO_NORM = -0.5

# vel_x-Normalisierung (qwop_env): Normalizable(-20, 60) → center=20, maxdev=40.
# Rückrechnung norm → echte m/s: raw = norm * 40 + 20.
_VEL_X_SCALE = 40.0
_VEL_X_CENTER = 20.0


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
    speed_weight:
        Gewicht eines zusätzlichen, KONTINUIERLICHEN Speed-Bonus: belohnt echte
        Vorwärtsgeschwindigkeit (m/s, aus obs[3] zurückgerechnet) linear pro Step.
        Nur der positive Vorwärtsanteil zählt (rückwärts/stehen = 0). Zielt darauf,
        schnelles Laufen attraktiver zu machen als sicheres langsames Kriechen.
        Default 0.0 = kein Speed-Bonus (unverändert).
    gait_weight:
        Gewicht eines Gang-Symmetrie-Bonus: belohnt die Winkel-DIFFERENZ der
        beiden Oberschenkel |angle(left)-angle(right)| — hoch = Schrittstellung
        (ein Bein vor, eins zurück), niedrig = Beine parallel (Stehen/Fallen).
        Gegated mit Vorwärtsgeschwindigkeit, damit breitbeiniges Stehen nichts
        bringt. Soll echtes abwechselndes Laufen gegen den asymmetrischen
        Fall-Fang-Gang pushen. Default 0.0 = aus.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        upright_weight: float = 0.5,
        height_threshold: float = 0.0,
        gate_on_forward: bool = True,
        penalty_below: bool = False,
        speed_weight: float = 0.0,
        gait_weight: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.upright_weight = float(upright_weight)
        self.height_threshold = float(height_threshold)
        self.gate_on_forward = bool(gate_on_forward)
        self.penalty_below = bool(penalty_below)
        self.speed_weight = float(speed_weight)
        self.gait_weight = float(gait_weight)

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

        # Kontinuierlicher Speed-Bonus: echte Vorwärts-m/s aus obs[3]
        # zurückrechnen (raw = norm*40 + 20), nur positiver Anteil zählt.
        # Belohnt schnelles Laufen linear — soll Kriechen unattraktiver machen.
        if self.speed_weight:
            raw_vel_x = float(arr[_TORSO_VEL_X_IDX]) * _VEL_X_SCALE + _VEL_X_CENTER
            shaping += self.speed_weight * max(0.0, raw_vel_x)

        # Gang-Symmetrie-Bonus: Winkeldifferenz der Oberschenkel (Schrittstellung),
        # gegated mit Vorwärtsgeschwindigkeit (breitbeiniges Stehen bringt nichts).
        # Soll abwechselndes Laufen gegen den asymmetrischen Fall-Fang-Gang pushen.
        if self.gait_weight and arr.shape[0] > _RIGHT_THIGH_ANGLE_IDX:
            thigh_diff = abs(float(arr[_LEFT_THIGH_ANGLE_IDX]) - float(arr[_RIGHT_THIGH_ANGLE_IDX]))
            shaping += self.gait_weight * thigh_diff * forward_gate

        return shaping, torso_height

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaping, torso_height = self._shaping_term(obs)
        # Für Logging/Kalibrierung sichtbar machen (W&B / eval.py).
        info["reward_shaping"] = shaping
        info["torso_height"] = torso_height
        return obs, float(reward) + shaping, terminated, truncated, info
