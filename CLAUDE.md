# CLAUDE.md — Projekt-Instruktionen für Claude Code

Dieses Dokument gibt Claude Code (und allen Teammitgliedern, die ihn nutzen) Kontext zum Projekt.

## Projekt

**QWOP-RL** — ein Reinforcement-Learning-Agent, der lernen soll, das Browserspiel QWOP zu spielen.
Studienprojekt "Neue Konzepte 2" (DHBW).

## Sprache

- **Code, Variablennamen, Code-Kommentare:** Englisch
- **Doku, Commit-Messages, PR-Beschreibungen, Chat:** Deutsch (Team-Sprache)

## Tech-Stack

- Python **3.11** (3.12 funktioniert nicht zuverlässig mit `qwop-gym 1.0.1`, siehe "Bekannte Stolpersteine")
- Stable-Baselines3 (RL)
- Gymnasium (Env-Interface)
- PyTorch (Backend)
- pytest / ruff / mypy / pre-commit

## Lokales Setup — auf welcher Maschine bin ich?

> **Für Claude:** Lies diese Sektion **bevor** du jemanden durch das ChromeDriver-/Bootstrap-/Patch-Setup schickst. Auf Maintainer-Maschinen ist das oft schon erledigt — frag/check zuerst, bevor du den vollen Onboarding-Pfad vorschlägst.

### Maintainer-Maschine (Patryks Mac, M-Series)

Auf dieser Maschine **gibt es bereits ein funktionsfähiges qwop-gym-Setup** vom Smoke-Test (2026-06-02):

```
~/qwop-gym-test/
├── .venv/                  # alte Test-venv (Python 3.11)
├── bin/chromedriver        # ChromeDriver 148.0.7778.178 (passend zu Chrome 148)
├── config/env.yml          # bereits konfiguriert (Browser-/Driver-Pfade)
├── benchmark.log           # 1920 Steps/s
├── env_smoketest.py
├── env_smoketest.log       # 1291 Steps/s
├── run_train.py
└── train.log               # 10k PPO-Steps in 22.92 s
```

Wenn du auf dieser Maschine arbeitest und ChromeDriver brauchst — **nicht neu ziehen**, einfach symlinken:

```bash
cd /pfad/zu/QWOP
mkdir -p bin
ln -sf ~/qwop-gym-test/bin/chromedriver bin/chromedriver
./bin/chromedriver --version   # ChromeDriver 148.0.7778.178
```

Vorher prüfen, ob deine Chrome-Version noch matched:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version
# Match → Driver wiederverwenden, Mismatch → neu ziehen wie in SETUP.md Schritt 5
```

### Andere Maschinen (Team-Mitglieder, frische Clones)

Voller Setup-Pfad: `SETUP.md` Schritt 5–6. ChromeDriver passend zur Chrome-Major holen → `qwop-gym bootstrap` → `qwop-gym patch`.

### Was Claude zuerst checken sollte

Bevor du einen User durch das ChromeDriver-Setup schickst:

1. `ls ~/qwop-gym-test/bin/chromedriver 2>/dev/null` — existiert ein Driver vom Smoke-Test?
2. `ls -la <repo>/bin/chromedriver 2>/dev/null` — schon im aktuellen Repo (Symlink)?
3. `ls <repo>/config/env.yml 2>/dev/null` — bootstrap schon gelaufen?
4. `ls <repo>/.venv/lib/python*/site-packages/qwop_gym/envs/v1/game/QWOP.min.js 2>/dev/null` — gepatcht?

Erst wenn _eine_ dieser vier Sachen fehlt, das Onboarding starten — und dann gezielt nur den fehlenden Schritt, nicht den ganzen Block aus SETUP.md.

## Architektur — wichtig

Die QWOP-Anbindung läuft über [`smanolloff/qwop-gym`](https://github.com/smanolloff/qwop-gym) (Browser/Chrome via ChromeDriver, optimiert auf 1900+ Steps/s). Smoke-Test 2026-06-02 hat das validiert; Details + Optionen-Vergleich stehen in [`docs/architecture.md`](docs/architecture.md). Größere Architektur-Wechsel (z.B. Migration auf eigenen Box2D-Port) gehören als ADR in `docs/adr/`.

### Recherche-Stand (2026-06-02)

Wir haben **mehrere existierende QWOP-RL-Projekte** evaluiert. Das vielversprechendste ist:

**[`smanolloff/qwop-gym`](https://github.com/smanolloff/qwop-gym)** (Apache-2.0, zuletzt aktualisiert Februar 2025)
- Genau unser Stack: Gymnasium + Stable-Baselines3 + PyTorch
- Browser-Anbindung (Chrome via ChromeDriver) — aber **stark optimiert**: WebSocket-Brücke, gepatchtes QWOP für Determinismus, kein WebGL beim Training, **>1900 Steps/s gemessen**
- Joint-State-Observation: kompakte 60-Byte-Vektoren
- 15 diskrete Aktionen (alle QWOP-Tastenkombinationen)
- Pretrained Models in einem W&B Public Project verfügbar
- Mehrere Algorithmen vorimplementiert: PPO, DQN, QRDQN, plus BC/GAIL/AIRL für Imitation Learning
- `pip install qwop-gym` als PyPI-Package

Andere geprüfte Projekte (alle weniger geeignet):
- `drakesvoboda/RL-QWOP` — alter Stack (gym+SB v2 von 2021), keine Lizenz, undokumentiert
- `Kirkados/QWOP` — Box2D-Reimplementation, aber TF 1.15 (uralt), keine Lizenz
- 6 weitere (siehe Recherche-Notes) — alle entweder JS-only, tot, oder schlechter dokumentiert

**Empfehlungs-Tendenz:** qwop-gym als Basis nehmen, eigener wissenschaftlicher Beitrag in Reward-Engineering / Algorithmen-Vergleich / Hyperparameter-Studie / Imitation-Learning. Endgültige Entscheidung folgt in **ADR-0001** nach Smoke-Test-Abschluss.

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
- **Architektur-Entscheidungen** als ADR in `docs/adr/` festhalten (siehe `docs/adr/0000-template.md`).
- **Keine echten Trainings starten**, ohne dass das vorher abgestimmt wurde — GPU-Zeit ist begrenzt.
- **Modelle / Logs / Daten** nicht committen (siehe `.gitignore`).
- **Secrets** (W&B-Keys etc.) nur in `.env`, niemals ins Repo.

### Anti-Choke: vor jedem „Setup-Vorschlag" prüfen, was schon da ist

Wir haben am 2026-06-08 einen Choke-Moment produziert: Claude hat den User durch das volle ChromeDriver-Setup geschickt, obwohl ein lauffähiges Setup unter `~/qwop-gym-test/` schon seit dem Smoke-Test (2026-06-02) lag. Der User hatte berechtigt zurückgefragt: _„Hatten wir das nicht schon getestet?"_

Damit das nicht wieder passiert — bevor du **irgendwelche** Setup-Schritte vorschlägst:

1. **Lies zuerst die Sektion „Lokales Setup — auf welcher Maschine bin ich?"** oben in dieser Datei und führe die vier Pre-Checks aus.
2. **Frag aktiv, was schon mal lief**, statt aus der Doku einen Onboarding-Pfad zu rekonstruieren — Doku beschreibt _was möglich ist_, nicht _was auf dieser Maschine schon da ist_.
3. **Schau in den Repo-State, nicht nur in `git status`**: `bin/`, `config/`, `.venv/`, `~/qwop-gym-test/` sind alle gitignored — `git status` sagt _nichts_ über sie aus.
4. **Beispiel-Snippets in dieser Datei sind Dokumentation, kein Setup-Status.** Wenn CLAUDE.md zeigt _„Beispiel: ChromeDriver nach `~/qwop-gym-test/bin/` ziehen"_, heißt das nicht, dass das auf der aktuellen Maschine getan ist — und nicht, dass es nicht getan ist. Heißt: nachschauen.

### Antitest: was Claude NICHT tun soll

- **Keine `qwop-gym benchmark`/`train_ppo`-Aufrufe direkt aus Claude starten** — Browser-Vordergrund-Drosselung (Stolperstein 12) lässt das hängen. User soll selbst starten, in eigenem Terminal.
- **Keine echten Trainings** ohne Absprache.
- **Keine ADRs für offensichtliche Entscheidungen.** ADRs lohnen sich, wenn echte Trade-offs unklar bleiben — nicht für „wir nehmen die Library, die im Smoke-Test funktioniert hat". Letzteres gehört in `docs/architecture.md` oder einen Commit-Body.

## Häufige Befehle

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .   # zieht auch qwop-gym mit
pre-commit install
cp .env.example .env   # WANDB_API_KEY eintragen

# Sanity-Check (frischer Clone — soll alles grün laufen)
ruff check . && ruff format --check . && mypy && pytest

# Entwicklung
ruff check . && ruff format .
mypy
pytest
pytest --cov=qwop_rl

# Training (sobald Env existiert)
python scripts/train.py --config configs/ppo_default.yaml
python scripts/train.py --config configs/ppo_default.yaml --run-name pl-baseline --tags experiment
python scripts/train.py --config configs/ppo_default.yaml --no-wandb   # Quick-Debug ohne Tracking

# Lokale Lernkurven
tensorboard --logdir logs/
# (oder live im Browser auf wandb.ai)
```

## Bekannte Stolpersteine

### Aus dem qwop-gym-Smoke-Test (2026-06-02)

Falls ihr `smanolloff/qwop-gym` lokal aufsetzt — folgendes vorab beachten:

1. **ChromeDriver muss exakt zur Chrome-Major-Version passen.**
   - Homebrew (`brew install --cask chromedriver`) liefert immer die neueste Version → Mismatch zu Chrome wahrscheinlich.
   - Lösung: passende Version manuell von [Chrome-for-Testing](https://googlechromelabs.github.io/chrome-for-testing/) holen.
   - Beispiel-Snippet (mac-arm64, Chrome 148):
     ```bash
     curl -sSL -o /tmp/cd.zip "https://storage.googleapis.com/chrome-for-testing-public/148.0.7778.178/mac-arm64/chromedriver-mac-arm64.zip"
     unzip -d /tmp/cd /tmp/cd.zip
     mkdir -p ~/qwop-gym-test/bin
     cp /tmp/cd/chromedriver-mac-arm64/chromedriver ~/qwop-gym-test/bin/chromedriver
     chmod +x ~/qwop-gym-test/bin/chromedriver
     xattr -d com.apple.quarantine ~/qwop-gym-test/bin/chromedriver
     ```
   - Driver-Pfad in `config/env.yml` setzen.
   - Hinweis: Brew-Cask `chromedriver` ist seit kurzem deprecated (passt macOS-Gatekeeper nicht).

2. **`bootstrap` ist interaktiv.** Es fragt nach Browser- und Driver-Pfad. Per Pipe füttern:
   ```bash
   printf "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\n/path/to/chromedriver\n" | qwop-gym bootstrap
   ```

3. **Spiel-Quelle herunterladen:** `curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch` (Patch macht das Spiel deterministisch).

4. **`pip install qwop-gym` installiert SB3/tqdm/rich NICHT mit.** Für Training zusätzlich nachziehen:
   ```bash
   pip install stable-baselines3 sb3-contrib tensorboard tqdm rich
   ```

5. **Direktes Env-Skripten braucht `if __name__ == "__main__":`** — sonst Endlos-Spawn-Schleife auf macOS (Default-`spawn`-Methode reimportiert das Skript). Beispiel:
   ```python
   import gymnasium as gym, qwop_gym, yaml
   def main():
       with open("config/env.yml") as f: kw = yaml.safe_load(f)
       env = gym.make("QWOP-v1", **kw)
       env.reset(seed=42)
       # ...
       env.close()
   if __name__ == "__main__":
       main()
   ```

6. **Python-Version: 3.11 nutzen, NICHT 3.12.** Mit Python 3.12 hängt `qwop-gym train_ppo` (mit aktueller SB3 ≥ 2.7) bei Step 1 — vermutlich Inkompatibilität SB3 2.8 × qwop-gym 1.0.1 (Okt 2023) × Python 3.12. Mit **Python 3.11 läuft es out-of-the-box** (10k PPO-Steps in ~23s gemessen). Brew: `brew install python@3.11`.

7. **macOS-Quarantäne entfernen** vor dem ersten Driver-Aufruf, sonst hängt `chromedriver --version` ohne Output:
   ```bash
   xattr -d com.apple.quarantine /pfad/zu/chromedriver
   ```

8. **Cleanup nach Test:** Wenn der Trainings-Prozess hängt, bleiben Chrome-Test-Instanzen offen (kleines Fenster auf Position 650,130, leicht zu übersehen). Aufräumen:
   ```bash
   pkill -f "user-agent=Chrome-"
   pkill -f chromedriver
   ```

9. **`qwop-gym` läuft erst mit ChromeDriver-Setup.** Die Library ist seit ADR-Klärung in `requirements.txt`, aber der Browser-/Driver-Pfad muss lokal gemacht werden (siehe `SETUP.md` Schritt 5–6). Auf Maschinen ohne Chrome (z.B. CI für reine Lint/Test-Jobs) reicht `pytest` über die jetzigen Tests — die spannen kein Chrome auf.

10. **`gym.make("QWOP-v1")` braucht Konstruktor-Argumente, nicht nur `config/env.yml`.** Die qwop-gym-CLI (`qwop-gym train_ppo` etc.) liest `config/env.yml` automatisch — `gym.make()` aber **nicht**. Wer das Env aus eigenem Python aufruft, muss die Pfade selber durchreichen, sonst kommt:
    ```
    ValueError: please specify a valid path to a chrome-based browser
    executable via the `browser` constructor argument
    ```
    Unsere `qwop_rl.envs.make_env()` lädt `config/env.yml` selbst und merged die Defaults in die Kwargs — User-Configs in `configs/*.yaml` können einzelne Keys überschreiben.

11. **`bin/` und `config/` gehören NICHT ins Repo.** Sind seit dem qwop-gym-Integration-PR in `.gitignore`. Beide sind maschinenlokal:
    - `bin/chromedriver` ist ein Plattform-spezifisches Binary (oder ein Symlink dahin)
    - `config/env.yml` enthält absolute Pfade zu `/Applications/Google Chrome.app/...` und `/Users/<user>/...`

    Achtung Verwechslungs-Falle: **`config/`** (singular, von qwop-gym) ist gitignored, **`configs/`** (plural, unsere Trainings-YAMLs) bleibt im Repo.

12. **Browser-Vordergrund-Drosselung — Trainings aus Claude heraus funktionieren nicht.** macOS drosselt Chrome-Fenster, sobald sie überdeckt sind (Background-Throttling). `qwop-gym benchmark`/`train_ppo` hängen dann scheinbar — kein Output, Python-Prozess hat 0 % CPU, Frames pro Sekunde fallen auf ~0. Symptom: `ps -p <pid> -o time` zeigt seit Minuten dieselbe CPU-Zeit. Lösung: das kleine 660×585-Browserfenster (Position 650,130) muss vorne sein.

    **Konsequenz für Claude:** echte qwop-gym-Trainings/Benchmarks NICHT direkt aus dem Tool aufrufen — das Terminal überdeckt zwangsläufig den Browser. Stattdessen dem User ein konkretes Kommando in die Hand geben („führe in einem eigenen Terminal aus, klick danach das Browserfenster vorne") und ihn das selbst starten lassen. Code-Validierung (Env spawnt sauber, schließt sauber, observation/action-Spaces stimmen) geht aber problemlos — ein einzelner `make_env() + close()` braucht den Browser-Vordergrund nicht.

### Allgemein

- Selenium-Driver muss zur Chrome-Version passen (siehe oben).
- Browser darf laut qwop-gym-README nicht in den Hintergrund wechseln (OS drosselt) — heißt für längere Trainings: dedizierte Maschine, Bildschirm nicht sperren.

### `.env` und W&B

- **`.env` muss im Repo-Root (`QWOP/.env`) liegen**, nicht im übergeordneten Ordner. `python-dotenv` (von `scripts/train.py` genutzt) sucht relativ zum Skript-Aufrufpfad.
- **Variable heißt `WANDB_API_KEY`**, nicht `WANDB_KEY`. Die `.env.example` ist die Referenz für korrekte Variablennamen.
- **Eine Variable pro Zeile.** Beim Eintippen aufpassen, dass jede Zeile mit Newline endet — sonst wird der API-Key mit dem nächsten Variablennamen verklebt und W&B meldet "API key invalid: has 7 chars" o.ä. Diagnose-Trick:
  ```bash
  awk -F= '/^[A-Z]/ {print $1, "(" length($2), "Zeichen)"}' .env
  ```
  Erwartete Längen: `WANDB_API_KEY` ~40 (alt) oder ~80 (neu, mit `wandb_v1_`-Prefix), `WANDB_ENTITY` ~7 (`qwop-rl`), `WANDB_PROJECT` ~12 (`qwop-rl-dhbw`).
- **Connection-Test:**
  ```bash
  python -c "from dotenv import load_dotenv; load_dotenv(); import wandb; wandb.login(verify=True); print('OK')"
  ```
- **Wenn ein API-Key versehentlich geleakt wurde** (Logs, Tool-Output, fremder Bildschirm), **sofort rotieren** unter [wandb.ai/settings](https://wandb.ai/settings) → API keys → Reset.
