# QWOP-RL

Reinforcement-Learning-Agent, der das Browserspiel [QWOP](http://www.foddy.net/Athletics.html) eigenständig spielen lernt.

> **Status:** 🚧 Setup-Phase — Architektur (Browser-Anbindung vs. Python-Reimplementation vs. Gym-Wrapper) wird noch festgelegt.

## Zielsetzung

Ein RL-Agent (Stable-Baselines3) soll lernen, einen QWOP-Läufer möglichst weit zu bewegen — als Studienprojekt im Rahmen "Neue Konzepte 2" (DHBW).

## Tech-Stack

| Bereich              | Wahl                                            |
| -------------------- | ----------------------------------------------- |
| Sprache              | Python 3.11+                                    |
| RL-Framework         | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) |
| Env-Standard         | [Gymnasium](https://gymnasium.farama.org/)      |
| Logging              | TensorBoard (+ optional Weights & Biases)       |
| Game-Interface       | _TBD_ (siehe `docs/architecture.md`)            |
| Dependency-Management| `pip` + `requirements.txt` (alternativ `uv`)    |

## Quickstart

```bash
# 1. Repo klonen
git clone https://github.com/LuciferEveningstar/QWOP.git
cd QWOP

# 2. Virtuelles Environment
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
pip install -e .                    # Projekt im Editable-Mode

# 4. Pre-Commit Hooks
pre-commit install

# 5. Tests
pytest
```

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
├── docs/               # Architektur, Decisions, Onboarding
├── models/             # Trainierte Checkpoints (gitignored)
├── logs/               # Tensorboard-Logs (gitignored)
└── .claude/            # Claude-Code-Konfiguration für das Team
```

## Entwicklungs-Workflow

Wir arbeiten mit **Feature-Branches + Pull Requests** auf `main`. Details siehe [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

```bash
git switch -c feat/<kurzer-name>
# ... arbeiten ...
git push -u origin feat/<kurzer-name>
gh pr create
```

## Team

- Patryk Lellek
- _(Teammitglieder bitte ergänzen)_

## Lizenz

MIT — siehe [`LICENSE`](LICENSE).
