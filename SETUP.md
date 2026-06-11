# SETUP — QWOP-RL lokal aufsetzen

Schritt-für-Schritt-Anleitung, vom frisch geklonten Repo bis zum ersten Trainingslauf, der live in W&B sichtbar ist.

**Plattformen:** Windows (PowerShell) und macOS. Linux ist sehr ähnlich zu macOS.

> **Vorab:** Wir nutzen zwei Stacks parallel.
> - Das **Repo selbst** (`QWOP/`) hält Code, Configs, Doku.
> - **`qwop-gym`** ist die fertige Library, die das Spiel anbietet — die wird als Dependency installiert.
>
> Kapitel 1–6 reichen, um *irgendwas* auf den Bildschirm zu kriegen. Ab Kapitel 7 wird's interessant.

---

## Inhalt

1. [Voraussetzungen](#1-voraussetzungen)
2. [Repo klonen](#2-repo-klonen)
3. [Python-Umgebung](#3-python-umgebung)
4. [Dependencies installieren](#4-dependencies-installieren)
5. [Browser + ChromeDriver](#5-browser--chromedriver)
6. [QWOP patchen](#6-qwop-patchen)
7. [W&B verknüpfen](#7-wb-verknüpfen)
8. [Erster Sanity-Check](#8-erster-sanity-check)
9. [Erstes Mini-Training](#9-erstes-mini-training)
10. [Häufige Probleme](#10-häufige-probleme)
11. [Was als nächstes?](#11-was-als-nächstes)

---

## 1. Voraussetzungen

| Tool | Wofür | Installation |
|---|---|---|
| **Python 3.11** (genau!) | Hauptsprache | siehe unten |
| **Git** | Repo holen | siehe unten |
| **Google Chrome** | qwop-gym braucht einen Chrome-Browser | [chrome.com](https://www.google.com/chrome/) |
| **GitHub-Account** | mit Zugriff auf `LuciferEveningstar/QWOP` | [github.com](https://github.com) |
| **W&B-Account** | Trainings-Tracking | [wandb.ai/signup](https://wandb.ai/signup) |

> **⚠️ Python 3.11, nicht 3.12 oder 3.13!**
> qwop-gym 1.0.1 hängt mit Python 3.12 im Trainings-Modus (Multiprocessing-Inkompatibilität). Auf 3.11 läuft alles. Siehe [`CLAUDE.md`](../CLAUDE.md) → "Bekannte Stolpersteine".

### Python 3.11 installieren

**Windows:**
```powershell
# Variante A: über winget (Windows Package Manager, vorinstalliert auf Win 10+)
winget install Python.Python.3.11

# Variante B: Installer von python.org
# https://www.python.org/downloads/release/python-3119/
# Wichtig: Beim Installieren "Add python.exe to PATH" anhaken
```

Verifizieren:
```powershell
py -3.11 --version
# Expected: Python 3.11.x
```

> **Tipp Windows:** `py -3.11` ist der saubere Weg, eine spezifische Version zu nutzen, ohne dass deine Standard-Python-Version dazwischenfunkt. Wenn du nur Python 3.11 installiert hast, geht auch einfach `python`.

**macOS:**
```bash
# Variante A: über Homebrew
brew install python@3.11

# Variante B: Installer von python.org
# https://www.python.org/downloads/release/python-3119/
```

Verifizieren:
```bash
python3.11 --version
# Expected: Python 3.11.x
```

### Git installieren

**Windows:**
```powershell
winget install Git.Git
# Oder: https://git-scm.com/download/win
```

**macOS:**
```bash
# Wenn Xcode-CLI-Tools fehlen, fragt macOS automatisch beim ersten git-Aufruf
git --version
# Falls nicht: brew install git
```

### Google Chrome

Einfach von [chrome.com](https://www.google.com/chrome/) installieren. **Wichtig:** Notiere dir die installierte Version (Chrome → Menü → Hilfe → Über Google Chrome). Die brauchen wir später beim ChromeDriver.

---

## 2. Repo klonen

```bash
# In deinen Workspace-Ordner wechseln
cd <wo-auch-immer-deine-projekte-liegen>

git clone https://github.com/LuciferEveningstar/QWOP.git
cd QWOP
```

**Windows-Hinweis:** Lege das Repo nicht in einen tief verschachtelten OneDrive-Pfad — Pfadlängen-Probleme mit Python-Tooling sind ein häufiger Stolperstein. Etwas wie `C:\dev\QWOP` oder `C:\Users\<du>\Projects\QWOP` ist sicher.

---

## 3. Python-Umgebung

Wir bauen ein **virtuelles Environment** (venv) im Repo. Das ist ein isolierter Python-Bereich nur für dieses Projekt — alle Dependencies landen darin und stören keine anderen Projekte.

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Falls PowerShell meckert "Ausführen von Skripten ist auf diesem System deaktiviert":
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# (einmalig im Admin-PowerShell, dann nochmal Activate.ps1)

python --version
# Expected: Python 3.11.x
```

**macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate

python --version
# Expected: Python 3.11.x
```

> **Wichtig:** Du musst `.venv` **bei jedem neuen Terminal aktivieren**, sonst nimmt Python wieder die System-Version. Erkennbar am `(.venv)`-Prefix in der Prompt.

---

## 4. Dependencies installieren

```bash
# pip aktualisieren
python -m pip install --upgrade pip

# Projekt-Dependencies (inklusive dev-Tools)
pip install -r requirements-dev.txt
pip install -e .

# qwop-gym (das eigentliche Spiel-Env)
pip install qwop-gym
```

Verifizieren:
```bash
python -c "import qwop_rl; print('qwop_rl', qwop_rl.__version__)"
python -c "import wandb; print('wandb', wandb.__version__)"
python -c "import qwop_gym; print('qwop_gym OK')"
python -c "import stable_baselines3; print('SB3', stable_baselines3.__version__)"
```

Alle vier sollten sauber durchlaufen.

---

## 5. Browser + ChromeDriver

qwop-gym steuert Chrome über **ChromeDriver** — eine Brücke zwischen Python (Selenium) und Chrome. **ChromeDriver muss zur installierten Chrome-Version passen** (Major-Version zumindest).

### 5a. Chrome-Version finden

- Chrome öffnen → Menü → "Hilfe" → "Über Google Chrome"
- Zeile lesen, z.B. *"Version 148.0.7778.216 (Offizieller Build) (arm64)"*
- Dich interessiert die **erste Zahl**: `148`

### 5b. ChromeDriver passend dazu holen

Wir holen ihn von der offiziellen Google-Seite [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/).

**Windows (PowerShell):**
```powershell
# Setze hier deine Chrome-Major-Version ein (z.B. 148)
$ChromeMajor = "148"

# Such-URL (zeigt die letzte verfügbare Patch-Version dieser Major)
# Schau auf https://googlechromelabs.github.io/chrome-for-testing/ → "Stable" Tabelle
# Beispiel-Download für Version 148.0.7778.178 win64:
$Url = "https://storage.googleapis.com/chrome-for-testing-public/148.0.7778.178/win64/chromedriver-win64.zip"
$ZipPath = "$env:TEMP\chromedriver.zip"
$ExtractPath = "$PWD\bin"

mkdir bin -Force | Out-Null
Invoke-WebRequest -Uri $Url -OutFile $ZipPath
Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
Move-Item "$ExtractPath\chromedriver-win64\chromedriver.exe" "$ExtractPath\chromedriver.exe" -Force
Remove-Item "$ExtractPath\chromedriver-win64" -Recurse

# Test
.\bin\chromedriver.exe --version
# Expected: ChromeDriver 148.0.7778.178 (...)
```

**macOS (arm64 / M-Series):**
```bash
# Major-Version aus Schritt 5a einsetzen
CHROME_MAJOR="148"

# Beispiel-Download für 148.0.7778.178 mac-arm64
URL="https://storage.googleapis.com/chrome-for-testing-public/148.0.7778.178/mac-arm64/chromedriver-mac-arm64.zip"

mkdir -p bin
curl -sSL -o /tmp/chromedriver.zip "$URL"
unzip -q /tmp/chromedriver.zip -d /tmp/cd
mv /tmp/cd/chromedriver-mac-arm64/chromedriver bin/chromedriver
chmod +x bin/chromedriver
xattr -d com.apple.quarantine bin/chromedriver 2>/dev/null || true
rm -rf /tmp/cd /tmp/chromedriver.zip

./bin/chromedriver --version
# Expected: ChromeDriver 148.0.7778.178 (...)
```

**macOS (Intel):**
Wie oben, aber URL `mac-x64` statt `mac-arm64`.

> **Tipp:** Wenn deine Chrome-Major neuer als 148 ist (was bei dir zu Hause vermutlich der Fall ist), schau auf [googlechromelabs.github.io/chrome-for-testing](https://googlechromelabs.github.io/chrome-for-testing/) in der "Stable"-Tabelle nach der passenden URL. Die ist immer aufgebaut wie:
> `chrome-for-testing-public/<volle-version>/<plattform>/chromedriver-<plattform>.zip`

### 5c. qwop-gym konfigurieren

qwop-gym hat einen interaktiven Bootstrap, der Browser- und Driver-Pfade konfiguriert. Wir müssen ihm die Pfade reinpipen:

**Windows (PowerShell):**
```powershell
$ChromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$DriverPath = "$PWD\bin\chromedriver.exe"

# Falls Chrome woanders installiert ist:
# Get-Command chrome.exe -ErrorAction SilentlyContinue
# oder: Test-Path "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

"$ChromePath`n$DriverPath" | qwop-gym bootstrap
```

**macOS:**
```bash
ChromePath="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DriverPath="$PWD/bin/chromedriver"

printf "%s\n%s\n" "$ChromePath" "$DriverPath" | qwop-gym bootstrap
```

Bootstrap legt einen `config/`-Ordner an mit YAML-Configs. Die `config/env.yml` enthält jetzt deine Browser- und Driver-Pfade.

> **Hinweis:** Der `config/`-Ordner wird **nicht** ins Repo committet (er gehört zu deinem persönlichen Setup). Schau bitte auch in `.gitignore` nach, falls noch nicht ergänzt.

---

## 6. QWOP patchen

qwop-gym muss das original `QWOP.min.js` einmalig patchen, damit das Spiel deterministisch läuft (gleicher Seed → gleiches Verhalten). Eine Internetverbindung ist nötig.

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://www.foddy.net/QWOP.min.js" -OutFile "QWOP.min.js"
qwop-gym patch QWOP.min.js
Remove-Item QWOP.min.js
```

**macOS:**
```bash
curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch
```

Erwartete Ausgabe: `Patch applied successfully`.

---

## 7. W&B verknüpfen

W&B ist unser zentrales Tracking-System für alle Trainingsläufe. Setup ist einmalig.

### 7a. Account und API-Key

1. Account erstellen auf [wandb.ai/signup](https://wandb.ai/signup) (kostenlos).
2. Dem Team-Workspace `qwop-rl` beitreten (Patryk lädt dich ein).
3. API-Key kopieren von [wandb.ai/authorize](https://wandb.ai/authorize) — ist ein langer Hex-String.

### 7b. `.env` anlegen

Im **Repo-Root** (also im `QWOP/`-Ordner):

```bash
# beide Plattformen
cp .env.example .env
# Windows ohne cp:  copy .env.example .env
```

Dann `.env` in einem Editor öffnen und die Werte ausfüllen:

```ini
WANDB_API_KEY=<dein-key-aus-wandb.ai/authorize>
WANDB_ENTITY=qwop-rl
WANDB_PROJECT=qwop-rl-dhbw
```

> **Stolpersteine — bitte sorgfältig:**
> - **Liegt die `.env` wirklich in `QWOP/.env`?** Nicht in `Documents/...` eine Ebene drüber. `python-dotenv` findet sie sonst nicht.
> - **Variable heißt `WANDB_API_KEY`**, nicht `WANDB_KEY`.
> - **Eine Variable pro Zeile**, jede mit Newline am Ende.
> - **Keine Anführungszeichen** um die Werte.

### 7c. Verifizieren

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import wandb; wandb.login(verify=True); print('W&B OK')"
```

Wenn `W&B OK` erscheint: läuft. Falls Fehler kommen: siehe [`docs/wandb-setup.md`](wandb-setup.md) → FAQ.

---

## 8. Erster Sanity-Check

Wir starten **nicht** das große Training, sondern nur den Benchmark — der prüft schnell, ob der ganze Stack zusammenspielt. Das dauert ein paar Sekunden und macht 10.000 zufällige Steps.

```bash
qwop-gym benchmark
```

Erwartete Ausgabe (Zahlen variieren je nach Hardware):

```
0%...10%...20%...30%...40%...50%...60%...70%...80%...90%...
1920.64 steps/s (10000 steps in 5.21 seconds)
```

**Was du erwarten kannst:**

| System | Erwartete Steps/s |
|---|---|
| MacBook M2 | ~1900 |
| 7800X3D + RTX5080 | ~3000–4000 |
| Älterer Laptop | ~800–1500 |

Wenn das funktioniert: alle Komponenten reden miteinander. Glückwunsch.

---

## 9. Erstes Mini-Training

Jetzt der erste richtige RL-Lauf — bewusst klein gehalten, um die ganze Pipeline (Env → SB3 → W&B → Modell-Save) end-to-end zu prüfen, bevor jemand stundenlang trainiert. Wir nutzen die mitgelieferte Smoke-Config:

```bash
python scripts/train.py --config configs/ppo_smoke.yaml --tags smoke
```

10.000 Steps, 1 Env, ~30 s. Während es läuft öffnet sich ein Chrome-Fenster — **darf nicht in den Hintergrund** (sonst drosselt das OS).

Erwartete Ausgabe am Ende:

```
[train] Model saved to models/<run-name>/final.zip
wandb: 🚀 View run <name> at: https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/<id>
```

Im W&B-Browser solltest du sehen:
- Run mit Tag `smoke`
- Lernkurven (Reward, Loss, fps)
- Hyperparameter unter „Config"
- Modell als Artifact unter „Files"

Wenn das geht, hast du die komplette Pipeline verifiziert. Glückwunsch.

### Alternative: qwop-gym-CLI (ohne unseren `train.py`)

`qwop-gym` bringt einen eigenen Trainer mit, der unabhängig von unserem `scripts/train.py` läuft — gut zum Vergleich oder wenn `train.py` aus irgendeinem Grund streikt:

```bash
qwop-gym train_ppo                                # 100k Steps, lokal, kein W&B
qwop-gym -c config/wandb/ppo.yml train_ppo        # mit W&B
```

Modelle landen in `data/PPO-<run-id>/model.zip`.

---

## 10. Häufige Probleme

### "ChromeDriver only supports Chrome version X"
Versions-Mismatch. Lade ChromeDriver für deine Chrome-Major-Version neu (siehe Schritt 5).

### "Chrome for Testing" startet, aber Spiel reagiert nicht
Auf Windows kann der Defender ChromeDriver kurz blockieren. Alternativ: chromedriver.exe einmal manuell starten, "Trotzdem ausführen" klicken.

### `qwop-gym train_ppo` hängt bei Step 1, Python 3.12 installiert
Python 3.12 funktioniert nicht zuverlässig mit qwop-gym 1.0.1. **Python 3.11 verwenden.** Siehe Voraussetzungen.

### W&B sagt "API key invalid: 7 characters"
Deine `.env` hat zwei Variablen ohne Newline dazwischen verklebt. Diagnose:
```bash
# macOS / Linux
awk -F= '/^[A-Z]/ {print $1, "(" length($2), "Zeichen)"}' .env
# Windows PowerShell
Get-Content .env | ForEach-Object { if ($_ -match "^([A-Z_]+)=(.*)$") { "$($Matches[1]) ($($Matches[2].Length) Zeichen)" } }
```
Erwartete Längen: API-Key ~40 oder ~80 Zeichen, ENTITY ~7, PROJECT ~12.

### `python-dotenv` findet `.env` nicht
Liegt sie in `QWOP/.env` oder eine Ebene drüber? Muss im **Repo-Root**.

### PowerShell verweigert Aktivierung des venv
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Bestätigen mit J
```

### macOS-Quarantäne blockiert chromedriver
```bash
xattr -d com.apple.quarantine bin/chromedriver
```

### Trainings-Browser geht in den Hintergrund, Training wird langsam
qwop-gym braucht den Browser im Vordergrund. Bildschirm wach halten, Fenster nicht minimieren. Auf längere Trainings (mehrere Stunden) am besten dedizierten Bildschirm/PC nutzen.

---

## 11. Was als nächstes?

Wenn alles aus den Schritten 1–9 funktioniert hat, hast du die Setup-Phase abgeschlossen. Weiter geht's mit:

- **[`docs/concepts.md`](concepts.md)** — RL-Begriffe verstehen (Step, Observation, Action, Reward, Episode)
- **[`docs/architecture.md`](architecture.md)** — was wir mit qwop-gym machen wollen, Forschungs-Ideen
- **[`docs/wandb-setup.md`](wandb-setup.md)** — Konventionen für eure W&B-Runs (Naming, Tags, Modell-Sharing)
- **[`docs/CONTRIBUTING.md`](CONTRIBUTING.md)** — Branch- und PR-Workflow

**Erster echter Trainingslauf:** Eigene Config in `configs/ppo_<variante>.yaml` anlegen (oder `configs/ppo_default.yaml` als Vorlage kopieren), `total_timesteps` auf z.B. `5_000_000` setzen, dann `python scripts/train.py --config configs/ppo_<variante>.yaml --run-name <initialen>-<datum>-<beschreibung> --tags experiment`. Auf einem 7800X3D ~2–3 h. Über Mittag oder über Nacht starten.

Bei Problemen, die nicht in Kapitel 10 stehen, frag im Team-Channel oder hänge eine Issue an [GitHub](https://github.com/LuciferEveningstar/QWOP/issues) ran.
