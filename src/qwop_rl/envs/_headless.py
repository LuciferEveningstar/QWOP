"""Headless-Chrome-Patch für qwop-gym (spawn-sicher, für Multi-Env-Training).

qwop-gym startet Chrome immer sichtbar (``WSServer._launch_browser`` hardcodet
die ChromeOptions, kein Headless-Kwarg). Für Parallelität mit ``SubprocVecEnv``
müssen aber alle Env-Browser headless laufen — sichtbare Fenster würden von
macOS gedrosselt (Stolperstein 12), sobald sie überdeckt sind.

Verifiziert (scripts/spike_headless.py, GRÜN @ ~1050 steps/s headless): mit
``--headless=new`` + SwiftShader-WebGL läuft die QWOP-Physik headless durch, weil
sie WebSocket-Command-getrieben ist (nicht requestAnimationFrame-abhängig).
Chrome ≥137 hat SwiftShader-WebGL deprecatet → ``--enable-unsafe-swiftshader``
ist zwingend, sonst initialisiert der WebGL-Kontext nicht und der WS-Client
verbindet nie (30-s-Server-Timeout).

**Spawn-Sicherheit:** :func:`apply_headless_patch` muss in JEDEM Prozess laufen,
der ein qwop-gym-Env baut. Unter der macOS-``spawn``-Startmethode re-importiert
jeder ``SubprocVecEnv``-Worker die Env-Factory — die ruft diese Funktion beim
Bauen auf, wodurch der Patch pro Worker re-appliziert wird. Ein Patch, der nur
im ``__main__`` des Trainings-Skripts sitzt, würde in den Workern fehlen.
Deshalb lebt er hier, in einem importierbaren Library-Modul.
"""

from __future__ import annotations

import os
import uuid

_PATCH_MARKER = "_qwop_rl_headless_patched"


def _build_headless_launch(gl_mode: str):  # type: ignore[no-untyped-def]
    """Fabriziert die gepatchte, async ``_launch_browser``-Methode.

    Treue Kopie von ``qwop_gym/envs/v1/util/wsserver.py`` (Zeilen 164-196), plus
    die Headless-/WebGL-Flags. ``gl_mode`` steuert die Software-Render-Variante
    (Default ``swiftshader`` — im Spike als funktionierend verifiziert).
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService

    async def _launch_browser_headless(self) -> None:  # type: ignore[no-untyped-def]
        self.logger.info("Launching web browser (HEADLESS)...")

        options = webdriver.ChromeOptions()
        options.add_argument("allow-file-access-from-files")
        options.add_argument("allow-cross-origin-auth-prompt")
        options.add_argument(f"user-agent=Chrome-{uuid.uuid4()}")
        options.add_argument("disable-infobars")
        options.add_argument("disable-extensions")
        options.add_argument("disable-popup-blocking")
        options.add_argument("disable-notifications")

        # --- Headless + WebGL-Software-Rendering ---
        options.add_argument("--headless=new")
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

        options.add_argument("window-size=660,585")

        service = ChromeService(executable_path=self.driver)
        options.binary_location = self.browser
        options.add_argument("--incognito")

        self._driver = webdriver.Chrome(service=service, options=options)
        self._window = self._driver.window_handles[0]
        self._driver.get(self.build_url())
        self._initialized = True

    return _launch_browser_headless


def apply_headless_patch() -> None:
    """Patch ``WSServer._launch_browser`` auf die Headless-Variante (idempotent).

    Mehrfachaufruf ist unschädlich — ein Marker-Attribut verhindert doppeltes
    Patchen. Der GL-Modus kommt aus ``QWOP_HEADLESS_GL`` (Default ``swiftshader``).
    """
    from qwop_gym.envs.v1.util.wsserver import WSServer

    if getattr(WSServer, _PATCH_MARKER, False):
        return

    gl_mode = os.environ.get("QWOP_HEADLESS_GL", "swiftshader")
    WSServer._launch_browser = _build_headless_launch(gl_mode)  # type: ignore[method-assign]
    setattr(WSServer, _PATCH_MARKER, True)
