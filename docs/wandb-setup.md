# Experiment-Tracking mit Weights & Biases

Wir nutzen [Weights & Biases (W&B)](https://wandb.ai), um Trainingsläufe, Modelle und Lernkurven zentral zu sammeln und im Team zu teilen.

## Warum nicht Git?

Git ist gut für Code und Configs. Nicht für:

- **Modelle** — binär, viele Versionen, schnell groß. GitHub-Limit: 100 MB pro Datei.
- **Logs** — TensorBoard-Events sind hunderte MB pro Lauf.
- **Vergleiche** — Git kann nicht "zeig mir, wie sich Reward über die Zeit zwischen Lauf A und B unterscheidet".

W&B löst genau das: Modelle landen in einer Cloud-Registry, Metriken werden live geloggt und visualisiert, Läufe sind miteinander vergleichbar.

## Was W&B für uns macht

- **Experiment-Tracking** — Reward, Loss und alle SB3-Metriken pro Step, automatisch
- **Live-Charts** im Browser, schon während das Training läuft
- **Model Registry** — Checkpoints als versionierte "Artifacts"
- **Hyperparameter** automatisch mitgespeichert
- **Vergleiche** — mehrere Läufe übereinanderlegen, sortieren, filtern, taggen
- **Team-Workspace** — alle sehen alle Läufe, kein manuelles Teilen nötig
- **Reproducibility** — jeder Lauf trägt vollständige Config + Code-Commit

## Setup (einmalig pro Person)

### 1. Account erstellen

[wandb.ai/signup](https://wandb.ai/signup) — kostenlos. Akademisch / persönlich / Open Source ist im Free-Tier abgedeckt (100 GB Storage).

### 2. Team-Workspace beitreten

Patryk legt das Team-Projekt `qwop-rl-dhbw` an und lädt euch über die W&B-UI ein.
(Alternativ kann jede:r das Projekt als persönliches Projekt nutzen — aber dann sehen sich die Läufe nicht gegenseitig.)

### 3. API-Key holen

Auf [wandb.ai/authorize](https://wandb.ai/authorize) findest du deinen persönlichen API-Key. **Nicht weitergeben, nicht ins Repo committen.**

### 4. Lokal verknüpfen

Im Repo-Root:

```bash
cp .env.example .env
# .env in einem Editor öffnen, WANDB_API_KEY=<dein-key> eintragen
```

`.env` ist gitignored — jede:r hat seine/ihre eigene.

Alternativ einmalig global:

```bash
wandb login
# Key einfügen, fertig
```

### 5. Verifizieren

```bash
source .venv/bin/activate
python -c "import wandb; wandb.login(); print('OK')"
```

## Alltag: ein Trainingslauf

Wenn alles steht:

```bash
# Mit Default-Config
python scripts/train.py --config configs/ppo_default.yaml

# Mit eigener Config (z.B. Reward-Variation)
python scripts/train.py --config configs/ppo_high_failure.yaml --run-name pl-failure50
```

Das Skript:

1. Initialisiert W&B (Projekt: `qwop-rl-dhbw`, Run-Name nach Konvention)
2. Loggt automatisch alle SB3-Metriken (Reward, Episode-Länge, Loss-Werte)
3. Speichert das fertige Modell als W&B-Artifact
4. Verknüpft alles mit dem aktuellen Git-Commit-Hash

Im Browser unter [wandb.ai/qwop-rl-dhbw](https://wandb.ai/) (URL anpassen, sobald Workspace existiert) seht ihr die Lernkurve **live**.

## Konventionen

### Run-Naming

```
<initialen>-<datum>-<kurz-beschreibung>
```

Beispiele:

- `pl-2026-06-04-baseline`
- `pl-2026-06-05-failure50`
- `mm-2026-06-06-discrete9`

So sieht man auf einen Blick, **wer**, **wann**, **was** trainiert hat.

### Tags

Vergebt Tags in der W&B-UI, um Läufe zu kategorisieren:

| Tag         | Bedeutung                                         |
| ----------- | ------------------------------------------------- |
| `baseline`  | Referenz-Lauf, gegen den verglichen wird          |
| `experiment`| Eigene Variation (Reward, Hyperparam, …)          |
| `final`     | Lauf, der in die Studienarbeit / Demo geht        |
| `broken`    | Lief schief, ignorieren                           |
| `wip`       | Work in progress, noch am Tunen                   |

### Wann committet man eine Config?

- **Sofort:** wenn die Config "interessant genug für andere zum Probieren" ist (z.B. neue Reward-Variante, neuer Algorithmus).
- **Nicht:** für schnelle Wegwerf-Experimente. Die laufen lokal, ihre W&B-Runs reichen.

### Modelle als "Release" markieren

Wenn ein Lauf einen neuen **Best-of-Group** liefert:

1. In W&B den Run mit `tag:best` markieren
2. Das Modell-Artifact in der UI als "Production" oder "Latest" aliased

Andere können dann mit `wandb.use_artifact("qwop-rl-dhbw/qwop-ppo-model:best")` direkt das beste Modell laden.

## FAQ

**F: Was, wenn ich offline trainieren will?**
A: `WANDB_MODE=offline python scripts/train.py …` — loggt lokal, beim nächsten `wandb sync` wird hochgeladen.

**F: Was, wenn ich versehentlich meinen API-Key committed habe?**
A: Sofort in der W&B-UI rotieren ([Settings → API keys](https://wandb.ai/settings)). Den alten Key invalidieren.

**F: Geht das auch ohne W&B?**
A: Ja, dann hat man nur lokale TensorBoard-Logs. Aber: **kein Teilen** und **keine Vergleiche** zwischen Teammitgliedern. Wir machen es, weil es für Studienarbeiten den größten Mehrwert bringt.

**F: Wie viel Storage werden wir brauchen?**
A: Pro Training ~100 KB (Modell) + ~20 MB (Logs). Bei 100 Trainings: ~2 GB. **Free-Tier hat 100 GB**, also weit weg von der Grenze.

**F: Kann ich auch lokale Backups machen?**
A: Ja — `models/` ist gitignored, da kannst du zusätzlich ablegen. Aber W&B ist die "Single Source of Truth".

## Weiterführend

- [W&B Docs — Quickstart](https://docs.wandb.ai/quickstart)
- [W&B + Stable-Baselines3 Integration](https://docs.wandb.ai/guides/integrations/stable-baselines-3)
- [smanolloff/qwop-gym W&B-Beispiele](https://github.com/smanolloff/qwop-gym/tree/main/config/wandb) — falls wir uns daran orientieren wollen
