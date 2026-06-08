# Onboarding — QWOP-RL

Hi 👋 willkommen im Team! Diese Checkliste bringt dich von Null zu "lokal lauffähig".

## 1. Voraussetzungen

- **Python 3.11** (NICHT 3.12 — siehe `CLAUDE.md`. Mac: `brew install python@3.11`)
- **Git** (`git --version`)
- **GitHub-Account** mit Zugriff auf `LuciferEveningstar/QWOP`
- **W&B-Account** (kostenlos, [wandb.ai/signup](https://wandb.ai/signup))
- Optional: **GitHub CLI** `gh` für komfortable PRs

## 2. Repo holen & Setup

**Vollständige Anleitung:** [`SETUP.md`](../SETUP.md) (mit allen Stolpersteinen für Windows + macOS).

Kurzform für erfahrene Entwickler:

```bash
git clone https://github.com/LuciferEveningstar/QWOP.git
cd QWOP
python3.11 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
pip install qwop-gym
pre-commit install

# W&B-Verknüpfung
cp .env.example .env                     # WANDB_API_KEY eintragen
```

Plus ChromeDriver passend zur Chrome-Version (siehe SETUP.md Schritt 5) und `qwop-gym patch` für deterministisches Spiel.

## 3. Sanity-Check

```bash
pytest                                    # sollte grün sein
ruff check .                              # sollte clean sein
python -c "import qwop_rl; print(qwop_rl.__version__)"
python -c "import wandb; wandb.login(); print('W&B OK')"
```

## 4. Lies dich ein

1. [`README.md`](../README.md) — Projektüberblick
2. [`docs/concepts.md`](concepts.md) — RL-Grundbegriffe (Step, Observation, Action, …)
3. [`CLAUDE.md`](../CLAUDE.md) — Projekt-Konventionen + bekannte Stolpersteine
4. [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) — Workflow & Commit-Regeln
5. [`docs/wandb-setup.md`](wandb-setup.md) — Experiment-Tracking
6. [`docs/architecture.md`](architecture.md) — Big Picture & offene Fragen
7. [`docs/adr/`](adr/) — bisherige Architektur-Entscheidungen

## 5. Erste Schritte

Such dir ein Issue mit Label `good-first-issue`, oder frag im Team-Channel nach. Branch anlegen, los geht's.

## 6. Tools, die wir empfehlen

- **VS Code** mit Extensions: Python, Pylance, Ruff, GitLens
- **W&B** zum Beobachten von Trainings ([wandb.ai](https://wandb.ai))
- **TensorBoard** als lokale Alternative: `tensorboard --logdir logs/`
- **Claude Code** als KI-Assistent (Konfig in `.claude/settings.json`)

## 7. Hilfe

- Fragen → Team-Chat
- Bugs / Vorschläge → GitHub Issue
- Architektur-Diskussionen → ADR-Vorschlag in `docs/adr/`
