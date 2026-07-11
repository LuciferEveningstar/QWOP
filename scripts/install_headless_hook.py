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
import sysconfig
from pathlib import Path

PTH_NAME = "qwop_rl_headless.pth"
PTH_LINE = "import qwop_rl._headless_bootstrap\n"


def _site_packages_dir() -> Path:
    """Bestimmt das site-packages-Verzeichnis, aus dem Python .pth-Dateien lädt.

    Nutzt ``sysconfig.get_path("purelib")`` — das liefert plattformübergreifend
    zuverlässig das korrekte Verzeichnis (venv-``Lib\\site-packages`` auf Windows,
    ``lib/pythonX.Y/site-packages`` auf macOS/Linux). ``site.getsitepackages()[0]``
    ist NICHT zuverlässig: auf Windows-venvs ist dessen erster Eintrag das
    venv-Root statt site-packages → die .pth landet dort und wird nie gelesen.
    """
    purelib = sysconfig.get_path("purelib")
    if purelib:
        return Path(purelib)
    # Fallback für ungewöhnliche Layouts.
    dirs = site.getsitepackages() or [site.getusersitepackages()]
    return Path(dirs[0])


def _cleanup_misplaced(correct: Path) -> None:
    """Entfernt eine evtl. am falschen Ort liegende .pth (z.B. venv-Root auf Windows).

    Frühere Installer-Versionen nutzten ``site.getsitepackages()[0]``, das auf
    Windows-venvs auf das venv-Root zeigte statt auf site-packages. Solche
    verwaisten .pth-Dateien werden hier aufgeräumt, damit es keine Verwirrung gibt.
    """
    candidates = set()
    for d in site.getsitepackages() or []:
        candidates.add(Path(d) / PTH_NAME)
    candidates.add(Path(site.getusersitepackages()) / PTH_NAME)
    for cand in candidates:
        if cand != correct and cand.exists():
            try:
                cand.unlink()
                print(f"[hook] Alte fehlplatzierte .pth entfernt: {cand}")
            except OSError:
                pass


def main() -> int:
    target = _site_packages_dir() / PTH_NAME
    _cleanup_misplaced(target)

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
