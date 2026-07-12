"""SB3-Callbacks für QWOP-Training.

SB3s PPO loggt aus dem ``ep_info_buffer`` nur ``ep_rew_mean``/``ep_len_mean``/
``success_rate`` nach ``rollout/``. Zusätzliche ``info_keywords`` (die Monitor in
die Episode-Infos packt — bei uns ``distance``, ``avgspeed``, ``torso_height``)
werden NICHT automatisch geloggt. :class:`ExtraRolloutMetricsCallback` schließt
diese Lücke: sie mittelt die Extra-Keys über den ``ep_info_buffer`` und schreibt
sie als ``rollout/<key>`` in den Logger → landen via ``sync_tensorboard`` in W&B.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

# Zusätzliche Episode-Metriken, die aus dem ep_info_buffer nach rollout/ geloggt
# werden (sofern in den Episode-Infos vorhanden — Monitor.info_keywords).
_EXTRA_KEYS = ("distance", "avgspeed", "is_success", "torso_height")


class ExtraRolloutMetricsCallback(BaseCallback):
    """Loggt Monitor-info_keywords als rollout/<key> (Mittel über ep_info_buffer).

    Läuft bei jedem Rollout-Ende, analog zu SB3s eigenem rollout/-Logging.
    Fehlt ein Key in den Infos (z.B. torso_height ohne aktives Shaping), wird er
    übersprungen — kein Fehler.
    """

    def _on_rollout_end(self) -> None:
        buffer = self.model.ep_info_buffer
        if not buffer:
            return
        for key in _EXTRA_KEYS:
            values = [ep[key] for ep in buffer if key in ep]
            if values:
                self.logger.record(f"rollout/{key}", safe_mean(values))

    def _on_step(self) -> bool:
        # Pflicht-Override von BaseCallback; hier nichts pro Step zu tun.
        return True
