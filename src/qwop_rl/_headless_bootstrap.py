"""Startup-Hook: appliziert den Headless-Patch bei ``QWOP_HEADLESS=1``.

Dieses Modul wird über eine ``.pth``-Import-Zeile in site-packages beim Start
JEDES Python-Interpreters dieser venv geladen (siehe
``scripts/install_headless_hook.py``). Das ist der einzige zuverlässige Weg, den
Headless-Patch auch in den WSServer-**Enkelprozessen** zu setzen, die qwop-gym
per ``spawn`` startet — diese re-importieren ``WSServer`` frisch und würden einen
nur im Worker gesetzten Monkey-Patch nicht sehen (→ sichtbare Chrome-Fenster,
Stolperstein 16).

Der Import läuft vor jedem User-Code. Damit fremde venv-Prozesse (und sichtbare
Einzel-Env-Läufe) NICHT betroffen sind, ist alles hinter ``QWOP_HEADLESS=1``
gegated — ohne die Var ist dieses Modul ein Quasi-Nullkosten-No-op.
"""

from __future__ import annotations

import os

if os.environ.get("QWOP_HEADLESS") == "1":
    try:
        from qwop_rl.envs._headless import apply_headless_patch

        apply_headless_patch()
    except Exception:
        # Interpreter-Start fremder Prozesse NIE brechen — schlägt der Import
        # fehl (z.B. qwop_gym nicht installiert), still ignorieren.
        pass
