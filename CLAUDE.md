# CLAUDE.md — Projekt-Instruktionen für Claude Code

Dieses Dokument gibt Claude Code (und allen Teammitgliedern, die ihn nutzen) Kontext zum Projekt.

## Projekt

**QWOP-RL** — ein Reinforcement-Learning-Agent, der lernen soll, das Browserspiel QWOP zu spielen.
Studienprojekt "Neue Konzepte 2" (DHBW).

## Sprache

- **Code, Variablennamen, Code-Kommentare:** Englisch
- **Doku, Commit-Messages, PR-Beschreibungen, Chat:** Deutsch (Team-Sprache)

## Tech-Stack

- Python **3.11** (3.12 funktioniert nicht zuverlässig mit `qwop-gym 1.0.1`, siehe „Bekannte Stolpersteine")
- Stable-Baselines3 (RL)
- Gymnasium (Env-Interface)
- PyTorch (Backend)
- pytest / ruff / mypy / pre-commit

## Architektur — kurz

Die QWOP-Anbindung läuft über [`smanolloff/qwop-gym`](https://github.com/smanolloff/qwop-gym) (Browser/Chrome via ChromeDriver, optimiert auf 1900+ Steps/s). qwop-gym ist als Dependency in `requirements.txt` gesetzt und in `src/qwop_rl/envs/__init__.py` über `make_env()` eingebunden, die `config/env.yml` einliest und Defaults mit User-Kwargs merged.

Details, Optionen-Vergleich und Recherche-Stand stehen in [`docs/architecture.md`](docs/architecture.md). Größere Architektur-Wechsel (z.B. Migration auf eigenen Box2D-Port) gehören als ADR in `docs/adr/` — für die initiale qwop-gym-Wahl reicht `architecture.md`.

## Lokales Setup — was zuerst checken

Auf einer Maschine, auf der schon mal trainiert wurde, sind die meisten Setup-Schritte schon erledigt. Bevor du jemanden durch SETUP.md schickst, prüf den Repo-State:

1. `ls <repo>/bin/chromedriver` — ChromeDriver vorhanden (Binary oder Symlink)?
2. `ls <repo>/config/env.yml` — qwop-gym-Bootstrap schon gelaufen?
3. `ls <repo>/.venv/lib/python*/site-packages/qwop_gym/envs/v1/game/QWOP.min.js` — gepatcht?
4. `<repo>/bin/chromedriver --version` matched zur Chrome-Major (`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version`)?

Erst wenn _eine_ dieser vier Sachen fehlt, das Onboarding starten — und dann gezielt nur den fehlenden Schritt aus `SETUP.md`, nicht den ganzen Block.

`bin/`, `config/` und `.venv/` sind alle gitignored — `git status` sagt _nichts_ über sie aus. Im Zweifel reinschauen.

## So startest du Trainings (Run-Anleitung)

> **Für den User:** Die Befehle in dieser Sektion **nicht aus Claude heraus** laufen lassen — Claude überdeckt zwangsläufig den Browser, macOS drosselt ihn dann auf 0 fps (Stolperstein 12). Selbst in einem Terminal starten und das kleine Browserfenster (660×585, Position 650,130) **vorne lassen**.
>
> **Für Claude:** Wenn ein User trainieren will, dieses Rezept verlinken/zeigen, NICHT selbst ausführen.

### Vorbedingungen

- Setup steht (siehe „Lokales Setup" oben — Pre-Check 1–4 alle ✓)
- venv aktiviert (`source .venv/bin/activate`)
- `.env` mit gültigem `WANDB_API_KEY` (Connection-Test: `python -c "from dotenv import load_dotenv; load_dotenv(); import wandb; wandb.login(verify=True); print('OK')"`)

### Variante A: Reine qwop-gym-CLI (am schnellsten zum ersten Run)

```bash
# Benchmark — nur Env-Geschwindigkeit messen, kein Lernen
qwop-gym benchmark
# erwartet auf M-Series: ~1900 Steps/s in ~5s

# Mini-PPO-Training — qwop-gym's eingebauter Trainer, Default 100k Steps
qwop-gym train_ppo
# Modell landet in data/PPO-<run-id>/model.zip, loggt NICHT zu W&B (nur lokal)

# Mit W&B-Logging
qwop-gym -c config/wandb/ppo.yml train_ppo
```

### Variante B: Unser scripts/train.py (für eigene Experimente / Konfigs)

```bash
# Pipeline-Smoke (10k Steps, ~30s, prüft End-to-End: Env → SB3 → W&B → Modell-Save)
python scripts/train.py --config configs/ppo_smoke.yaml --tags smoke

# Echtes Training gegen configs/ppo_default.yaml (1M Steps, ~2-3h auf M-Series)
python scripts/train.py --config configs/ppo_default.yaml

# Mit eigenem Namen + Tags (siehe wandb-setup.md für Konventionen)
python scripts/train.py \
  --config configs/ppo_default.yaml \
  --run-name pl-2026-06-11-baseline \
  --tags baseline experiment

# Quick-Debug ohne W&B-Roundtrip
python scripts/train.py --config configs/ppo_default.yaml --no-wandb
```

### Beim ersten Mal auf einer Maschine: Smoke vor langem Lauf

`configs/ppo_smoke.yaml` ist genau dafür da: 10k Steps, 1 Env, ~30s. Bestätigt:
- W&B-Login klappt und der Run landet im richtigen Workspace
- Modell-Save-Pfad funktioniert
- Browser bleibt stabil im Vordergrund

Erst wenn der grün ist, lange Läufe (1M+ Steps, mehrere Stunden) starten.

### Während das Training läuft

- **Browser nicht überdecken** — bei Bedarf zweiten Monitor nutzen oder das Fenster auf einen freien Screenbereich ziehen
- **Bildschirm nicht sperren / schlafen lassen** — drosselt ebenfalls (`caffeinate -d &` hilft, mit `pkill caffeinate` wieder ausschalten)
- **W&B live mitlesen** auf [wandb.ai/qwop-rl/qwop-rl-dhbw](https://wandb.ai/qwop-rl/qwop-rl-dhbw) oder lokal `tensorboard --logdir logs/`
- **Trainings-Browser nie schließen** — nur via Skript-Ende oder `pkill` (siehe Stolperstein 8)

### Nach dem Training

- Modell liegt in `models/<run-name>/final.zip` (für `scripts/train.py`) bzw. `data/PPO-<id>/model.zip` (für `qwop-gym train_ppo`)
- Wenn der Run einen Best-of-Group setzt: in W&B-UI `tag:best` setzen + Artifact als „Production"/„Latest" aliasen (siehe `docs/wandb-setup.md`)
- Verwaiste Chrome-Prozesse putzen, falls was hängt:
  ```bash
  pkill -f "user-agent=Chrome-"
  pkill -f chromedriver
  ```

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
  docs(architecture): Anbindungs-Optionen ergänzt
  ```
- **PRs:** Mindestens ein Review vor Merge. Squash-Merge auf `main`.
- Niemals direkt auf `main` pushen.

### Experiment-Tracking (Weights & Biases)

Modelle und Trainings-Metriken **gehören nicht ins Git-Repo**, sondern in W&B. Setup-Anleitung: [`docs/wandb-setup.md`](docs/wandb-setup.md).

- **Workspace:** [wandb.ai/qwop-rl/qwop-rl-dhbw](https://wandb.ai/qwop-rl/qwop-rl-dhbw) (Entity `qwop-rl`, Projekt `qwop-rl-dhbw`)
- **Jeder Trainingslauf loggt zu W&B**. Lokale-Only-Läufe nur fürs Debuggen mit `--no-wandb` oder `WANDB_MODE=disabled`.
- **Run-Naming:** `<initialen>-<datum>-<beschreibung>`, z.B. `pl-2026-06-04-failure50`.
- **Tags vergeben:** `baseline`, `experiment`, `final`, `broken`, `wip` — direkt in der W&B-UI.
- **API-Keys:** in `.env` im **Repo-Root** (`QWOP/.env`, gitignored). Vorlage in `.env.example`. Niemals committen.
- **Beste Modelle:** in W&B mit Tag `best` markieren + Artifact-Alias setzen, damit Team-Mitglieder sie laden können.
- **Configs für interessante Läufe** committen (`configs/ppo_<variante>.yaml`). Wegwerf-Experimente bleiben lokal.

## Workflows für Claude

- **Vor größeren Änderungen** zuerst Plan vorschlagen / EnterPlanMode nutzen.
- **Architektur-Entscheidungen** nur dann als ADR in `docs/adr/` festhalten, wenn echte Trade-offs unklar bleiben (Template: `docs/adr/0000-template.md`). Was ein Smoke-Test oder Commit-Body sauber dokumentiert, braucht kein ADR.
- **Keine echten Trainings starten**, ohne dass das vorher abgestimmt wurde — GPU-Zeit ist begrenzt.
- **Modelle / Logs / Daten** nicht committen (siehe `.gitignore`).
- **Secrets** (W&B-Keys etc.) nur in `.env`, niemals ins Repo.

### Anti-Choke: vor jedem „Setup-Vorschlag" prüfen, was schon da ist

Bevor du **irgendwelche** Setup-Schritte vorschlägst:

1. **Erst die vier Pre-Checks** aus „Lokales Setup" oben durchgehen.
2. **Aktiv fragen, was schon mal lief**, statt aus der Doku einen Onboarding-Pfad zu rekonstruieren — Doku beschreibt _was möglich ist_, nicht _was auf dieser Maschine schon da ist_.
3. **In den Repo-State schauen, nicht nur in `git status`**: `bin/`, `config/`, `.venv/` sind alle gitignored.
4. **Beispiel-Snippets in dieser Datei sind Doku, kein Setup-Status.** Wenn ein Befehl gezeigt wird, heißt das nicht, dass er auf dieser Maschine schon gelaufen ist — und nicht, dass er nicht gelaufen ist. Nachschauen.

### Antitest: was Claude NICHT tun soll

- **Keine `qwop-gym benchmark`/`train_ppo`-Aufrufe und keine `python scripts/train.py`-Aufrufe direkt aus Claude starten** — Browser-Vordergrund-Drosselung (Stolperstein 12) lässt das hängen. User soll selbst starten, in eigenem Terminal. Code-Validierung (`make_env() + close()` ohne Steps) ist okay, das spannt keinen vordergrund-pflichtigen Browser auf.
- **Keine echten Trainings** ohne Absprache.
- **Keine ADRs für offensichtliche Entscheidungen.** ADRs lohnen sich, wenn echte Trade-offs unklar bleiben — nicht für „wir nehmen die Library, die im Smoke-Test funktioniert hat".

## Häufige Befehle

```bash
# Setup (frischer Clone — Details in SETUP.md)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .   # zieht auch qwop-gym mit
pre-commit install
cp .env.example .env   # WANDB_API_KEY eintragen

# Sanity-Check (soll alles grün laufen)
ruff check . && ruff format --check . && mypy && pytest

# Entwicklung
ruff check . && ruff format .
mypy
pytest
pytest --cov=qwop_rl

# Training
python scripts/train.py --config configs/ppo_smoke.yaml --tags smoke      # 10k Steps, ~30s
python scripts/train.py --config configs/ppo_default.yaml                 # 1M Steps, ~2-3h
python scripts/train.py --config configs/ppo_default.yaml --no-wandb      # Quick-Debug ohne Tracking

# Lokale Lernkurven
tensorboard --logdir logs/
# (oder live im Browser auf wandb.ai)
```

## Bekannte Stolpersteine

Diese Liste ist die Sammelstelle für alles, worüber wir oder andere QWOP-RL-Projekte gestolpert sind. Bei neuen Bugs hier ergänzen.

1. **ChromeDriver muss exakt zur Chrome-Major-Version passen.**
   - Homebrew (`brew install --cask chromedriver`) liefert immer die neueste Version → Mismatch zu Chrome wahrscheinlich. Brew-Cask ist außerdem deprecated (passt macOS-Gatekeeper nicht).
   - Lösung: passende Version manuell von [Chrome-for-Testing](https://googlechromelabs.github.io/chrome-for-testing/) holen, in `bin/chromedriver` legen, in `config/env.yml` referenzieren.
   - Vollständige Snippets pro Plattform: `SETUP.md` Schritt 5.

2. **`qwop-gym bootstrap` ist interaktiv.** Per Pipe füttern:
   ```bash
   printf "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n/path/to/chromedriver\n" | qwop-gym bootstrap
   ```

3. **QWOP.min.js muss gepatcht werden** (sonst nicht-deterministisch):
   ```bash
   curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch
   ```

4. **`pip install qwop-gym` zieht SB3/tqdm/rich nicht mit.** Unsere `requirements.txt` deckt das ab — wer nur qwop-gym standalone installiert, muss SB3 etc. selber nachziehen.

5. **Direktes Env-Skripten braucht `if __name__ == "__main__":`** — sonst Endlos-Spawn-Schleife auf macOS (Default-`spawn`-Methode reimportiert das Skript).

6. **Python-Version: 3.11 nutzen, NICHT 3.12.** Mit Python 3.12 hängt `qwop-gym train_ppo` (mit aktueller SB3 ≥ 2.7) bei Step 1 — vermutlich Inkompatibilität SB3 2.8 × qwop-gym 1.0.1 (Okt 2023) × Python 3.12. Mit **Python 3.11 läuft es out-of-the-box**. Brew: `brew install python@3.11`.

7. **macOS-Quarantäne entfernen** vor dem ersten Driver-Aufruf, sonst hängt `chromedriver --version` ohne Output:
   ```bash
   xattr -d com.apple.quarantine /pfad/zu/chromedriver
   ```

8. **Cleanup nach Test:** Wenn der Trainings-Prozess hängt, bleiben Chrome-Test-Instanzen offen (kleines Fenster auf Position 650,130, leicht zu übersehen). Aufräumen:
   ```bash
   pkill -f "user-agent=Chrome-"
   pkill -f chromedriver
   ```

9. **`qwop-gym` läuft erst mit ChromeDriver-Setup.** Auf Maschinen ohne Chrome (z.B. CI für reine Lint/Test-Jobs) reicht `pytest` über die jetzigen Tests — die spannen kein Chrome auf.

10. **`gym.make("QWOP-v1")` braucht Konstruktor-Argumente, nicht nur `config/env.yml`.** Die qwop-gym-CLI liest `config/env.yml` automatisch — `gym.make()` aber **nicht**. Sonst:
    ```
    ValueError: please specify a valid path to a chrome-based browser
    executable via the `browser` constructor argument
    ```
    Unsere `qwop_rl.envs.make_env()` lädt `config/env.yml` selbst und merged Defaults in die Kwargs — User-Configs in `configs/*.yaml` können einzelne Keys überschreiben.

11. **`bin/` und `config/` gehören NICHT ins Repo.** Sind in `.gitignore`. Beide sind maschinenlokal — `bin/chromedriver` ist plattformspezifisch, `config/env.yml` enthält absolute Pfade. **Verwechslungs-Falle:** `config/` (singular, von qwop-gym) ist gitignored, `configs/` (plural, unsere Trainings-YAMLs) bleibt im Repo.

12. **Browser-Vordergrund-Drosselung — Trainings aus Claude heraus funktionieren nicht.** macOS drosselt Chrome-Fenster, sobald sie überdeckt sind. `qwop-gym benchmark`/`train_ppo` und `python scripts/train.py` hängen dann scheinbar — kein Output, Python-Prozess hat 0 % CPU. Symptom: `ps -p <pid> -o time` zeigt seit Minuten dieselbe CPU-Zeit. Lösung: das kleine 660×585-Browserfenster muss vorne sein. **Konsequenz für Claude:** Trainings/Benchmarks NICHT direkt aus dem Tool aufrufen — User soll selbst starten. Code-Validierung (`make_env() + close()`) braucht den Browser-Vordergrund nicht.

13. **macOS-Datenschutz: `~/Documents` blockiert Chromes file://-Zugriff — auch mit Full Disk Access.** Wenn das Repo unter `~/Documents/...` liegt, öffnet Chrome die `QWOP.html` im venv per `file://` und wird von macOS blockiert:
    ```
    Access to the file was denied
    file:///Users/.../QWOP/.venv/lib/python3.11/site-packages/qwop_gym/envs/v1/game/QWOP.html?...
    ERR_ACCESS_DENIED
    ```
    Symptom: Chrome-Fenster geht auf, zeigt das Sad-Face, qwop-gym hängt bei `Loading configuration from config/benchmark.yml`.

    **Wichtig:** Das **„Full Disk Access"-Toggle in System Settings reicht NICHT**. macOS hat für iCloud-managed Documents eine **separate** TCC-Schutzschicht. Diagnose: `ls -la@e ~/Documents | head` — wenn `com.apple.file-provider-domain-id` als xattr steht, ist dein Documents-Ordner iCloud-managed und FDA hilft nicht.

    **Robuster Fix:** Repo aus `~/Documents/` rausziehen (z.B. `~/dev/QWOP/`). venv muss neu gebaut werden (Pfade absolut), `config/env.yml` muss neue Driver-Pfade kriegen. Versuch zuerst trotzdem: Privacy & Security → **„Files and Folders"** (NICHT „Full Disk Access") → Chrome → Documents-Ordner aktivieren — manchmal reicht das, oft nicht.

    Auf Windows ist das Pendant der OneDrive-Pfad — siehe SETUP.md, Schritt 2.

### Allgemein

- Selenium-Driver muss zur Chrome-Version passen (siehe Stolperstein 1).
- Browser darf laut qwop-gym-README nicht in den Hintergrund wechseln (OS drosselt) — heißt für längere Trainings: dedizierte Maschine, Bildschirm nicht sperren.

### `.env` und W&B

- **`.env` muss im Repo-Root (`QWOP/.env`) liegen**, nicht im übergeordneten Ordner. `python-dotenv` (von `scripts/train.py` genutzt) sucht relativ zum Skript-Aufrufpfad.
- **Variable heißt `WANDB_API_KEY`**, nicht `WANDB_KEY`. Die `.env.example` ist die Referenz für korrekte Variablennamen.
- **Eine Variable pro Zeile.** Beim Eintippen aufpassen, dass jede Zeile mit Newline endet — sonst wird der API-Key mit dem nächsten Variablennamen verklebt und W&B meldet „API key invalid: has 7 chars" o.ä. Diagnose-Trick:
  ```bash
  awk -F= '/^[A-Z]/ {print $1, "(" length($2), "Zeichen)"}' .env
  ```
  Erwartete Längen: `WANDB_API_KEY` ~40 (alt) oder ~80 (neu, mit `wandb_v1_`-Prefix), `WANDB_ENTITY` ~7 (`qwop-rl`), `WANDB_PROJECT` ~12 (`qwop-rl-dhbw`).
- **Connection-Test:**
  ```bash
  python -c "from dotenv import load_dotenv; load_dotenv(); import wandb; wandb.login(verify=True); print('OK')"
  ```
- **Wenn ein API-Key versehentlich geleakt wurde** (Logs, Tool-Output, fremder Bildschirm), **sofort rotieren** unter [wandb.ai/settings](https://wandb.ai/settings) → API keys → Reset.
