"""Installiert den Headless-Startup-Hook in die aktuelle venv.

Schreibt eine ``.pth``-Datei nach site-packages, deren Import-Zeile beim Start
jedes Interpreters dieser venv ``qwop_rl._headless_bootstrap`` lädt. Nur so
erreicht der Headless-Patch auch die WSServer-Enkelprozesse (siehe dortigen
Docstring und CLAUDE.md Stolperstein 16).

Einmal nach ``pip install -e .`` ausführen:
    python scripts/install_headless_hook.py

Maschinenlokal: die ``.pth`` liegt in der (gitignoreten) venv und überlebt keinen
venv-Rebuild — nach Neuaufsetzen erneut ausführen. Idempotent: mehrfaches
Ausführen ist unschädlich.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

PTH_NAME = "qwop_rl_headless.pth"
PTH_LINE = "import qwop_rl._headless_bootstrap\n"


def _site_packages_dir() -> Path:
    """Bestimmt das site-packages-Verzeichnis der aktiven venv."""
    dirs = site.getsitepackages()
    if not dirs:
        # Fallback für ungewöhnliche venv-Layouts.
        dirs = [site.getusersitepackages()]
    return Path(dirs[0])


def main() -> int:
    target = _site_packages_dir() / PTH_NAME

    if target.exists() and target.read_text() == PTH_LINE:
        print(f"[hook] Bereits installiert: {target}")
        return 0

    try:
        target.write_text(PTH_LINE)
    except OSError as exc:
        print(f"[hook] FEHLER beim Schreiben von {target}: {exc}", file=sys.stderr)
        return 1

    print(f"[hook] Installiert: {target}")
    print("[hook] Wirkt bei gesetztem QWOP_HEADLESS=1 (train.py setzt das ab n_envs>1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
