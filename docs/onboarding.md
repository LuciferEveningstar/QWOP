# Onboarding — QWOP-RL

Hi 👋 willkommen im Team! Diese Checkliste bringt dich von Null zu "lokal lauffähig".

## 1. Voraussetzungen

- **Python 3.11+** (`python3.11 --version`)
- **Git** (`git --version`)
- **GitHub-Account** mit Zugriff auf `LuciferEveningstar/QWOP`
- Optional: **GitHub CLI** `gh` für komfortable PRs

## 2. Repo holen & Setup

```bash
git clone https://github.com/LuciferEveningstar/QWOP.git
cd QWOP
python3.11 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## 3. Sanity-Check

```bash
pytest                                    # sollte grün sein
ruff check .                              # sollte clean sein
python -c "import qwop_rl; print(qwop_rl.__version__)"
```

## 4. Lies dich ein

1. [`README.md`](../README.md) — Projektüberblick
2. [`CLAUDE.md`](../CLAUDE.md) — Projekt-Konventionen
3. [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) — Workflow & Commit-Regeln
4. [`docs/architecture.md`](architecture.md) — Big Picture & offene Fragen
5. `docs/adr/` — bisherige Entscheidungen

## 5. Erste Schritte

Such dir ein Issue mit Label `good-first-issue`, oder frag im Team-Channel nach. Branch anlegen, los geht's.

## 6. Tools, die wir empfehlen

- **VS Code** mit Extensions: Python, Pylance, Ruff, GitLens
- **TensorBoard** zum Beobachten von Trainings: `tensorboard --logdir logs/`
- **Claude Code** als KI-Assistent (Konfig in `.claude/settings.json`)

## 7. Hilfe

- Fragen → Team-Chat
- Bugs / Vorschläge → GitHub Issue
- Architektur-Diskussionen → ADR-Vorschlag in `docs/adr/`
