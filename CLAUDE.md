# CLAUDE.md — Projekt-Instruktionen für Claude Code

Dieses Dokument gibt Claude Code (und allen Teammitgliedern, die ihn nutzen) Kontext zum Projekt.

## Projekt

**QWOP-RL** — ein Reinforcement-Learning-Agent, der lernen soll, das Browserspiel QWOP zu spielen.
Studienprojekt "Neue Konzepte 2" (DHBW).

## Sprache

- **Code, Variablennamen, Code-Kommentare:** Englisch
- **Doku, Commit-Messages, PR-Beschreibungen, Chat:** Deutsch (Team-Sprache)

## Tech-Stack

- Python 3.11+
- Stable-Baselines3 (RL)
- Gymnasium (Env-Interface)
- PyTorch (Backend)
- pytest / ruff / mypy / pre-commit

## Architektur — wichtig

Die Anbindung an QWOP (Browser via Selenium/Playwright vs. Python-Reimplementation vs. nur Gym-Wrapper) ist **noch nicht entschieden**. Siehe [`docs/architecture.md`](docs/architecture.md). Bevor Code im `envs/`-Ordner geschrieben wird, muss diese Entscheidung in einem ADR (`docs/adr/`) festgehalten werden.

## Projektstruktur

```
src/qwop_rl/
├── envs/      # Gym-Environments (QWOP-Anbindung)
├── agents/    # RL-Agent-Wrapper, Trainings-Logik
└── utils/     # Hilfsfunktionen (Logging, Config-Loader, …)

configs/       # YAML — pro Trainingslauf eine Datei
scripts/       # CLI-Einstiegspunkte (train.py, eval.py, …)
tests/         # pytest, spiegelt src/-Struktur
docs/          # Architektur + ADRs + Onboarding
```

## Konventionen

### Code

- **Style:** Ruff-formatiert (`ruff format`), Lint clean (`ruff check`).
- **Typing:** Type-Hints in allen öffentlichen Funktionen. `mypy` muss durchlaufen.
- **Imports:** Absolute Imports (`from qwop_rl.envs import ...`), keine relativen.
- **Logging:** `logging`-Modul, kein `print` in Library-Code.
- **Configs:** Hyperparameter gehören in YAML unter `configs/`, nicht hardcoded.

### Tests

- pytest. Eine Test-Datei pro Modul (`tests/envs/test_qwop_env.py` für `src/qwop_rl/envs/qwop_env.py`).
- Bei jedem Bugfix einen Regressions-Test ergänzen.

### Git

- **Branches:** `feat/<name>`, `fix/<name>`, `docs/<name>`, `refactor/<name>`, `chore/<name>`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) auf Deutsch:
  ```
  feat(envs): QWOP-Browser-Wrapper hinzugefügt
  fix(agents): Reward-Normalisierung korrigiert
  docs(adr): ADR-0001 Browser-Anbindung
  ```
- **PRs:** Mindestens ein Review vor Merge. Squash-Merge auf `main`.
- Niemals direkt auf `main` pushen.

## Workflows für Claude

- **Vor größeren Änderungen** zuerst Plan vorschlagen / EnterPlanMode nutzen.
- **Architektur-Entscheidungen** als ADR in `docs/adr/` festhalten (siehe `docs/adr/0000-template.md`).
- **Keine echten Trainings starten**, ohne dass das vorher abgestimmt wurde — GPU-Zeit ist begrenzt.
- **Modelle / Logs / Daten** nicht committen (siehe `.gitignore`).
- **Secrets** (W&B-Keys etc.) nur in `.env`, niemals ins Repo.

## Häufige Befehle

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pre-commit install

# Entwicklung
ruff check . && ruff format .
mypy
pytest
pytest --cov=qwop_rl

# Training (sobald Env existiert)
python scripts/train.py --config configs/ppo_default.yaml
tensorboard --logdir logs/
```

## Bekannte Stolpersteine

- _(Hier ergänzen, sobald welche auftauchen — z.B. "Selenium-Driver muss zur Chrome-Version passen".)_
