# QWOP-RL

Reinforcement-Learning-Agent, der das Browserspiel [QWOP](http://www.foddy.net/Athletics.html) eigenständig spielen lernt.

> **Status:** 🚧 Setup-Phase — Architektur (qwop-gym als Basis vs. Eigenbau) wird in [`docs/architecture.md`](docs/architecture.md) festgelegt.

## Zielsetzung

Ein RL-Agent (Stable-Baselines3) soll lernen, einen QWOP-Läufer möglichst weit zu bewegen — als Studienprojekt im Rahmen "Neue Konzepte 2" (DHBW).

## Tech-Stack

| Bereich              | Wahl                                            |
| -------------------- | ----------------------------------------------- |
| Sprache              | **Python 3.11** (NICHT 3.12 — siehe `CLAUDE.md`)|
| RL-Framework         | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) |
| Env-Standard         | [Gymnasium](https://gymnasium.farama.org/)      |
| Experiment-Tracking  | [Weights & Biases](https://wandb.ai) (siehe [`docs/wandb-setup.md`](docs/wandb-setup.md)) |
| Game-Interface       | _TBD_ (siehe [`docs/architecture.md`](docs/architecture.md)) |
| Dependency-Management| `pip` + `requirements.txt`                      |

## Quickstart

> Vollständige Anleitung mit Troubleshooting: **[`SETUP.md`](SETUP.md)** — *empfohlen für den ersten Setup*

```bash
# 1. Repo klonen
git clone https://github.com/LuciferEveningstar/QWOP.git
cd QWOP

# 2. Virtuelles Environment (Python 3.11!)
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1

# 3. Dependencies
pip install -r requirements-dev.txt
pip install -e .
pip install qwop-gym

# 4. Pre-Commit Hooks
pre-commit install

# 5. W&B-Account verknüpfen (siehe docs/wandb-setup.md)
cp .env.example .env                # dann WANDB_API_KEY eintragen

# 6. Tests
pytest
```

Der Quickstart überspringt **ChromeDriver-Setup**, **QWOP-Patch** und **W&B-Verifikation** — diese Schritte stehen detailliert in [`SETUP.md`](SETUP.md).

## Projektstruktur

```
QWOP/
├── src/qwop_rl/        # Python-Package (Env, Agents, Utils)
│   ├── envs/           # Gym-Environments (QWOP-Anbindung)
│   ├── agents/         # RL-Agents / Trainings-Logik
│   └── utils/          # Hilfsfunktionen
├── configs/            # YAML-Configs für Trainingsläufe
├── scripts/            # Trainings-/Eval-Skripte (CLI-Einstiegspunkte)
├── notebooks/          # Jupyter-Notebooks für Exploration
├── tests/              # pytest-Suites
├── docs/               # Architektur, Concepts, ADRs, Onboarding
├── models/             # Trainierte Checkpoints (gitignored, siehe W&B)
├── logs/               # Tensorboard-Logs (gitignored)
└── .claude/            # Claude-Code-Konfiguration für das Team
```

## Wichtige Doku

| Datei                                                     | Inhalt                                                          |
| --------------------------------------------------------- | --------------------------------------------------------------- |
| [`SETUP.md`](SETUP.md)                                    | **Komplette Setup-Anleitung** (Win + macOS, mit Troubleshooting)|
| [`CLAUDE.md`](CLAUDE.md)                                  | Projekt-Konventionen + bekannte Stolpersteine                   |
| [`docs/concepts.md`](docs/concepts.md)                    | **RL-Grundbegriffe** (Step, Observation, Action, Reward, …)     |
| [`docs/architecture.md`](docs/architecture.md)            | Anbindungs-Optionen, Recherche-Stand, Smoke-Test-Ergebnisse     |
| [`docs/wandb-setup.md`](docs/wandb-setup.md)              | Experiment-Tracking: Account, API-Key, Konventionen             |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)            | Branches, Commits, PR-Workflow                                  |
| [`docs/onboarding.md`](docs/onboarding.md)                | Checkliste für neue Teammitglieder                              |
| [`docs/adr/`](docs/adr/)                                  | Architektur-Entscheidungen (ADRs)                               |

## Entwicklungs-Workflow

Wir arbeiten mit **Feature-Branches + Pull Requests** auf `main`. Details siehe [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

```bash
git switch -c feat/<kurzer-name>
# ... arbeiten ...
git push -u origin feat/<kurzer-name>
gh pr create
```

## Modelle und Trainings-Ergebnisse

Modelle, Lernkurven und Metriken werden **nicht** im Git-Repo gespeichert, sondern in **Weights & Biases**:

- **Workspace:** [wandb.ai/qwop-rl/qwop-rl-dhbw](https://wandb.ai/qwop-rl/qwop-rl-dhbw)
- Setup-Anleitung: [`docs/wandb-setup.md`](docs/wandb-setup.md)

## Team

- Patryk Lellek
- _(Teammitglieder bitte ergänzen)_

## Lizenz

MIT — siehe [`LICENSE`](LICENSE).
