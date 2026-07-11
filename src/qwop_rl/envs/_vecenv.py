"""Nicht-daemonische SubprocVecEnv-Variante für qwop-gym.

Problem: SB3s ``SubprocVecEnv`` startet jeden Worker-Prozess mit ``daemon=True``
(hartkodiert). qwop-gym spawnt aber INNERHALB jedes Envs einen eigenen
WSServer-Prozess (``qwop_env.py``: ``self.proc.start()``) — und daemonische
Prozesse dürfen laut Python keine Kindprozesse haben:

    AssertionError: daemonic processes are not allowed to have children

Lösung: eine minimale Subclass, die die Worker mit ``daemon=False`` startet.
Ansonsten identisch zu SB3s Implementierung (``_worker``/``CloudpickleWrapper``
werden direkt wiederverwendet).

Trade-off von ``daemon=False``: crasht der Hauptprozess hart, könnten die Worker
weiterlaufen. In der Praxis räumt ``close()`` sie ab; als Fallback
``pkill -f chromedriver`` (siehe CLAUDE.md, Stolperstein 8).
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env.subproc_vec_env import (
    CloudpickleWrapper,
    SubprocVecEnv,
    _worker,
)


class NonDaemonicSubprocVecEnv(SubprocVecEnv):
    """Wie ``SubprocVecEnv``, aber Worker laufen als ``daemon=False``.

    Nötig, weil qwop-gym-Envs selbst Kindprozesse (WSServer) starten.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: str | None = None,
    ) -> None:
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # spawn ist auf macOS die sichere Default-Methode.
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns, strict=True
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # Einziger Unterschied zu SB3: daemon=False, damit der Worker den
            # qwop-gym-WSServer-Kindprozess starten darf.
            process = ctx.Process(target=_worker, args=args, daemon=False)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        # VecEnv-Basis initialisieren (überspringt SubprocVecEnv.__init__).
        super(SubprocVecEnv, self).__init__(n_envs, observation_space, action_space)
