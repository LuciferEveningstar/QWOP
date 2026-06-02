# Contributing — QWOP-RL

Willkommen! Diese Datei beschreibt unseren Workflow.

## Setup (einmalig)

```bash
git clone https://github.com/LuciferEveningstar/QWOP.git
cd QWOP
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## Branch-Strategie

`main` ist immer grün und deploybar (so weit wir das im Lernkontext halten können).
Es wird nicht direkt auf `main` gepusht.

| Prefix      | Wofür                                  |
| ----------- | -------------------------------------- |
| `feat/`     | Neue Funktionalität                    |
| `fix/`      | Bugfix                                 |
| `refactor/` | Umbau ohne Verhaltensänderung          |
| `docs/`     | Doku-Änderungen                        |
| `chore/`    | Build, CI, Dependencies, Konfiguration |
| `exp/`      | Experimente (werden evtl. nicht gemerged) |

Beispiel:
```bash
git switch -c feat/ppo-baseline
```

## Commit-Messages

Wir nutzen [Conventional Commits](https://www.conventionalcommits.org/) auf Deutsch:

```
<typ>(<scope>): <kurze Beschreibung>

<optionaler längerer Body>

<optionaler Footer, z.B. "Closes #12">
```

Beispiele:
```
feat(envs): QWOP-Browser-Env hinzugefügt
fix(agents): Beobachtungs-Normalisierung war off-by-one
docs(adr): ADR-0001 für Browser-Anbindung
chore(deps): stable-baselines3 auf 2.4.0 angehoben
```

## Pull-Request-Workflow

1. Aktuellen `main` ziehen: `git switch main && git pull`
2. Branch erstellen: `git switch -c feat/<name>`
3. Arbeiten, committen (gerne kleine Commits).
4. Vor dem Push lokal: `ruff check . && ruff format . && mypy && pytest`
5. Pushen: `git push -u origin feat/<name>`
6. PR via GitHub anlegen (Template wird automatisch geladen).
7. **Mindestens ein Review** durch ein Teammitglied.
8. CI muss grün sein.
9. **Squash-Merge** auf `main`.

## Code-Qualität

- **Ruff:** Format und Lint — wird durch pre-commit erzwungen.
- **MyPy:** Type-Checks; neuer Code soll typed sein.
- **Pytest:** Tests für jede neue Funktionalität. Coverage-Ziel: >70%.
- **Reviews:** Konstruktiv und auf Deutsch. Keine Nitpicks ohne Begründung.

## Architektur-Entscheidungen (ADRs)

Größere Entscheidungen (Framework-Wahl, Game-Anbindung, Reward-Design …) werden als ADR in `docs/adr/` dokumentiert. Template: [`docs/adr/0000-template.md`](adr/0000-template.md).

## Daten und Modelle

- Trainierte Modelle gehören **nicht** ins Repo (siehe `.gitignore`).
- Für geteilte Checkpoints nutzen wir _(TBD: SharePoint / Drive / W&B Artifacts)_.
- TensorBoard-Logs werden nur lokal gehalten; relevante Plots ggf. ins PR posten.
