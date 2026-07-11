"""Headless-Diagnose für qwop-gym (Spike, kein Produktions-Code).

Klärt die eine offene Frage vor der Trainings-Parallelisierung: Läuft qwop-gym
mit unsichtbarem Chrome (``--headless=new``)? Wenn ja, greift die macOS-
Vordergrund-Drosselung (Stolperstein 12) nicht und echte ``SubprocVecEnv``-
Parallelität wird nutzbar.

qwop-gym bietet KEINEN Headless-Kwarg — die ChromeOptions sind in
``WSServer._launch_browser`` hardcodet. Wir monkey-patchen die Methode auf
Modulebene und ergänzen die Headless-Flags.

WICHTIG — ``spawn`` vs. ``fork``:
    qwop-gym startet den WSServer als eigenen Prozess (``multiprocessing``).
    macOS-Default ist ``spawn`` — dabei würde der Child das Modul neu
    importieren und unseren Patch NICHT sehen. Für diese reine Diagnose setzen
    wir daher ``fork``: der Child erbt das bereits gepatchte Klassenobjekt, der
    Patch ist garantiert da.

    ``fork`` ist NUR für diesen Spike ok. Der Produktions-Pfad (train.py mit
    SubprocVecEnv) darf kein fork nutzen (SB3 vermeidet es) — dort muss der
    Patch in einem importierbaren Library-Modul liegen, damit jeder gespawnte
    Worker ihn beim Import re-appliziert. Siehe Plan, Schritt 2.

Ausführen (im eigenen Terminal, venv aktiv):
    python scripts/spike_headless.py

Grün:  "obs changed = True", steps/s dreistellig+, distance variiert, KEIN
       Browserfenster erscheint, Exit 0.
Rot:   30-s-Timeout / Selenium-Exception (headless lädt file:// nicht) ODER
       "obs changed = False" / einstellige steps/s (Canvas/WebGL-Kontext
       initialisiert headless nicht) → Headless nicht nutzbar, Fallback n_envs=1.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time
import uuid

import numpy as np

# fork ZUERST setzen — vor jedem qwop-gym-Import, der einen Prozess spawnen
# könnte. Nur für diesen Diagnose-Spike (siehe Modul-Docstring).
multiprocessing.set_start_method("fork", force=True)

from qwop_gym.envs.v1.util.wsserver import WSServer  # noqa: E402
from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.service import Service as ChromeService  # noqa: E402


async def _launch_browser_headless(self: WSServer) -> None:
    """Treue Kopie von ``WSServer._launch_browser`` + Headless-Flags.

    Basiert auf qwop_gym/envs/v1/util/wsserver.py (Zeilen 164-196). Ergänzt
    ``--headless=new`` sowie ``--disable-gpu`` / ``--use-gl=swiftshader`` als
    Software-Render-Fallback für den WebGL/Canvas-Kontext unter ``file://``.
    """
    self.logger.info("Launching web browser (HEADLESS spike)...")

    options = webdriver.ChromeOptions()
    options.add_argument("allow-file-access-from-files")
    options.add_argument("allow-cross-origin-auth-prompt")
    options.add_argument(f"user-agent=Chrome-{uuid.uuid4()}")
    options.add_argument("disable-infobars")
    options.add_argument("disable-extensions")
    options.add_argument("disable-popup-blocking")
    options.add_argument("disable-notifications")

    # --- Headless-Ergänzung (der eigentliche Zweck des Patches) ---
    options.add_argument("--headless=new")

    # WebGL headless: QWOP.min.js braucht einen WebGL-Kontext (getContext
    # "webgl"/"experimental-webgl"). Ohne GPU muss Chrome per SwiftShader
    # software-rendern. Chrome ≥ ~137 hat SwiftShader-WebGL standardmäßig
    # deprecatet → ohne --enable-unsafe-swiftshader bleibt der Kontext leer,
    # das Spiel-JS läuft nicht durch und der WS-Client verbindet nie
    # (Symptom: 30-s-Server-Timeout + Reconnect-Schleife).
    #
    # Über SPIKE_GL umschaltbar, um mehrere Varianten ohne Code-Edit zu testen:
    #   SPIKE_GL=swiftshader (Default) | angle | egl | none
    gl_mode = os.environ.get("SPIKE_GL", "swiftshader")
    options.add_argument("--ignore-gpu-blocklist")
    options.add_argument("--enable-unsafe-swiftshader")
    if gl_mode == "swiftshader":
        options.add_argument("--use-gl=swiftshader")
        options.add_argument("--use-angle=swiftshader")
    elif gl_mode == "angle":
        options.add_argument("--use-gl=angle")
        options.add_argument("--use-angle=swiftshader")
    elif gl_mode == "egl":
        options.add_argument("--use-gl=egl")
    # gl_mode == "none": keine expliziten GL-Flags, nur die zwei oben.
    print(f"[spike] GL-Modus: {gl_mode}")

    options.add_argument("window-size=660,585")

    service = ChromeService(executable_path=self.driver)
    options.binary_location = self.browser
    options.add_argument("--incognito")

    self._driver = webdriver.Chrome(service=service, options=options)
    self._window = self._driver.window_handles[0]
    self._driver.get(self.build_url())
    self._initialized = True


# Patch auf Klassenebene — greift für alle WSServer-Instanzen dieses Prozesses.
WSServer._launch_browser = _launch_browser_headless  # type: ignore[method-assign]


def main() -> int:
    # Import erst hier, damit der Patch oben sicher vor der Env-Erstellung steht.
    from qwop_rl.envs import make_env

    # Watchdog: verbindet der WS-Client nicht (headless-WebGL scheitert), hängt
    # make_env()/reset() sonst in einer Endlos-Reconnect-Schleife. Nach 45 s hart
    # abbrechen mit klarer Diagnose (der qwop-gym-Server timeout't intern bei 30 s).
    def _timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(
            "Env-Setup > 45 s — headless-WebGL hat vermutlich nicht initialisiert "
            "(WS-Client verbindet nie). ROT."
        )

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(45)

    n_steps = 500
    print("[spike] Erstelle Env headless (kein Fenster sollte erscheinen)...")
    try:
        env = make_env(
            {
                "id": "QWOP-v1",
                "kwargs": {"render_mode": "browser", "game_in_browser": False},
            }
        )
        obs0, _info = env.reset()
        signal.alarm(0)  # Setup geschafft — Watchdog aus.
    except TimeoutError as exc:
        print(f"\n[spike] ROT: {exc}")
        print(
            "[spike] Tipp: andere GL-Variante testen — "
            "SPIKE_GL=angle python scripts/spike_headless.py   (oder =egl / =none)"
        )
        return 1

    try:
        obs0 = np.asarray(obs0, dtype=np.float64)
        seen: set[bytes] = {obs0.tobytes()}
        changed = False
        # qwop-gyms info-Dict liefert "distance" (zurückgelegte Strecke) —
        # steigt/variiert die Distanz, läuft die Physik wirklich (siehe
        # qwop_env.py:_build_info).
        first_dist: float | None = None
        last_dist: float | None = None

        print(f"[spike] Laufe {n_steps} Random-Steps ...")
        start = time.perf_counter()
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, info = env.step(action)
            obs_arr = np.asarray(obs, dtype=np.float64)
            seen.add(obs_arr.tobytes())
            if not changed and np.any(obs_arr != obs0):
                changed = True
            dist = info.get("distance") if isinstance(info, dict) else None
            if dist is not None:
                if first_dist is None:
                    first_dist = float(dist)
                last_dist = float(dist)
            if terminated or truncated:
                env.reset()
        elapsed = time.perf_counter() - start

        steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")
        print("\n===== SPIKE-ERGEBNIS =====")
        print(f"  Steps:          {n_steps}")
        print(f"  Wall time:      {elapsed:.2f} s")
        print(f"  Steps/sec:      {steps_per_sec:.0f}")
        print(f"  obs changed:    {changed}")
        print(f"  unique obs:     {len(seen)} / {n_steps + 1}")
        print(f"  distance first: {first_dist}")
        print(f"  distance last:  {last_dist}")
        print("==========================")

        # Grob-Diagnose zur schnellen Einordnung.
        if changed and steps_per_sec >= 100:
            print(
                "[spike] GRÜN: Physik läuft headless, Durchsatz ok → "
                "Parallelisierung (n_envs>1) realistisch."
            )
        elif not changed:
            print(
                "[spike] ROT: obs ändert sich nicht → Canvas/WebGL headless "
                "nicht initialisiert. Fallback n_envs=1."
            )
        else:
            print(
                "[spike] GELB: Durchsatz niedrig → headless läuft, aber langsam. "
                "Parallel-Gewinn prüfen."
            )
    finally:
        env.close()
        print("[spike] Env geschlossen.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
