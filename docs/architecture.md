# Architektur — QWOP-RL

> Dieses Dokument ist ein **Living Doc**. Architektur-Entscheidungen werden zusätzlich in `docs/adr/` festgeschrieben.

## Big Picture

```
┌────────────────────┐    actions      ┌────────────────────┐
│                    │ ──────────────▶ │                    │
│   RL-Agent (SB3)   │                 │   QWOP-Env (Gym)   │
│   PPO / SAC / …    │ ◀────────────── │   (Anbindung TBD)  │
│                    │ obs, reward     │                    │
└────────────────────┘                 └─────────┬──────────┘
        │                                        │
        │ TensorBoard / W&B                      │
        ▼                                        ▼
   logs/, models/                         QWOP (Browser/Reimpl)
```

## Komponenten

### 1. QWOP-Env (`src/qwop_rl/envs/`)
Gymnasium-kompatibles Environment. **Anbindungs-Frage offen** — siehe unten.
Liefert:
- **Action Space:** `MultiBinary(4)` für die Tasten Q, W, O, P (oder `Discrete(16)` für Kombis).
- **Observation Space:** _TBD_, abhängig von Anbindung (Pixel? Gelenk-State? Hybrid?).
- **Reward:** zurückgelegte Distanz pro Step (+ ggf. Strafen für Sturz / Zeit).

### 2. Agent (`src/qwop_rl/agents/`)
Dünner Wrapper um Stable-Baselines3-Algorithmen. Lädt Hyperparameter aus YAML, kümmert sich um Checkpoints, Eval-Runs, Video-Aufzeichnung.

### 3. Training-Skripte (`scripts/`)
- `train.py` — startet einen Trainingslauf basierend auf einer Config.
- `eval.py` — lädt Checkpoint und evaluiert (mit Video).
- `play.py` — manuelles Spielen (Sanity-Check fürs Env).

### 4. Configs (`configs/`)
YAML pro Lauf: Algorithmus, Hyperparameter, Env-Parameter, Logging-Pfade, Seed.

## Offene Architektur-Entscheidungen

### A. QWOP-Anbindung — **Hauptfrage**

| Option                                            | Pro                                                                                  | Contra                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **(A1) Browser via Selenium/Playwright**          | Originalspiel; keine Physik-Reimpl nötig; visuell überzeugendes Demo                 | Langsam (1× Realtime); fragil; Pixel-Obs erfordert CNN → mehr Compute   |
| **(A2) Python-Port von QWOP** (Box2D / pymunk)    | Schnell (Headless, parallelisierbar, > 100× Realtime); deterministisch; reproduzierbar | Aufwändig nachzubauen; ist nicht mehr "echtes" QWOP                     |
| **(A3) Bestehender Open-Source-Port als Env**     | Spart Reimpl; oft schon Gym-kompatibel                                               | Lizenz/Status prüfen; Code-Qualität variabel                            |
| **(A4) Hybrid: Browser für Eval, Reimpl fürs Training** | Beste aus beiden Welten                                                          | Doppelte Wartung; Sim-to-Real-Gap zwischen den beiden Envs              |

### Recherche zu A3 — Stand 2026-06-02

Insgesamt 9 existierende Projekte gefunden, davon einer eindeutiger Spitzenkandidat:

| Repo | Stack | Letzter Commit | Lizenz | Bewertung |
| --- | --- | --- | --- | --- |
| [`smanolloff/qwop-gym`](https://github.com/smanolloff/qwop-gym) | gymnasium + SB3 + PyTorch | 02/2025 | **Apache-2.0** | ⭐ **Top-Kandidat** |
| [`Kirkados/QWOP`](https://github.com/Kirkados/QWOP) | TF 1.15 + Box2D | 04/2021 | keine | Stack veraltet |
| [`drakesvoboda/RL-QWOP`](https://github.com/drakesvoboda/RL-QWOP) | gym + SB v2 | 02/2021 | keine | unterdurchschnittlich |
| 6 weitere | — | — | — | nicht relevant |

#### `smanolloff/qwop-gym` — Detail

- **Architektur:** Browser-Anbindung (Chrome/ChromeDriver), aber stark optimiert via WebSocket-Brücke und gepatchtem QWOP-Code
- **Performance:** **1900+ Steps/s** auf macOS M-Series (gemessen, README verspricht 2000+)
- **Observation:** Joint-State, kompakte 60-Byte-Vektoren (kein Pixel-Lernen nötig)
- **Action Space:** 15 diskrete Tastenkombinationen (Discrete(16) inkl. "kein Tasten")
- **Algorithmen vorimplementiert:** PPO, DQN, QRDQN, BC, GAIL, AIRL
- **Pretrained Models:** ja, in einem W&B Public Project
- **Determinismus:** ja (gepatchtes QWOP-Spiel)
- **CLI:** fertig, `pip install qwop-gym` als PyPI-Package
- **Dokumentation:** README + `doc/env.md` + `doc/game.md`

#### Smoke-Test-Ergebnis (2026-06-02)

Auf einer Mac-arm64-Testmaschine durchgeführt:

| Schritt | Ergebnis |
| --- | --- |
| `pip install qwop-gym` | ✅ funktioniert |
| `qwop-gym bootstrap` (interaktiv) | ✅ funktioniert (mit Pipe-Input) |
| `qwop-gym patch` (Spiel patchen) | ✅ funktioniert |
| `qwop-gym benchmark` (10k random steps) | ✅ **1920 Steps/s** |
| Direktes Env-Skript (200 Steps) | ✅ **1291 Steps/s** (mit `if __name__ == "__main__":`) |
| `qwop-gym train_ppo` (Mini-Training, Python 3.11) | ✅ **10.000 PPO-Steps in 22,9 s**, Modell gespeichert |
| `qwop-gym train_ppo` (Mini-Training, Python 3.12) | ❌ hängt bei Step 1 — Inkompatibilität SB3 2.8 × qwop-gym 1.0.1 × Python 3.12 |

**Konsequenz:** Mit Python 3.11 ist die komplette Pipeline lauffähig. Das Tech-Stack-`Python 3.11+` aus `CLAUDE.md` muss daher genauer auf **Python 3.11** festgelegt werden, nicht 3.12+.

### Empfehlungs-Tendenz

`qwop-gym` als Basis. Der wissenschaftliche Beitrag des Studienprojekts würde dann darauf aufsetzen — z.B.:
- Algorithmen-Vergleich (PPO vs. DQN vs. QRDQN, ggf. eigener Algorithmus)
- Reward-Engineering-Studie
- Imitation Learning vs. RL from scratch
- Hyperparameter-Studie mit W&B-Sweeps
- Curriculum Learning
- Observation-Ablation (welche der 60 Bytes sind nötig?)

Endgültige Entscheidung folgt in **ADR-0001** nach abgeschlossenem Smoke-Test.

### B. Observation-Design

- **Pixel** (Bildschirm-Crop) — generisch, aber sample-ineffizient.
- **Joint-State** (Winkel + Geschwindigkeiten der Gelenke) — viel effizienter, setzt aber (A2)/(A3) voraus.
- **Hybrid** — Joint-State plus optional ein Velocity-Feature.

### C. Reward-Shaping

- Naiv: `Δx` pro Step.
- Mit Penalties: Sturz (Kopf am Boden), übermäßige Hüft-Auslenkung, Zeitstrafe.
- Bonus für hohe Geschwindigkeit / aufrechte Haltung.

### D. Algorithmus

Start mit **PPO** (robust, gut dokumentiert, sample-efficient genug). SAC/TD3 falls wir auf kontinuierliche Action-Spaces (Druckdauer pro Taste) umstellen.

## Nächste Schritte

1. **ADR-0001** für Anbindung (A1–A4) erstellen und entscheiden.
2. Minimaler Env-Skeleton (`QwopEnv`) als Stub committen — `reset`/`step` werfen `NotImplementedError`, aber Action-/Observation-Spaces sind definiert.
3. Sanity-Check: zufälliger Agent läuft 100 Steps ohne Crash.
4. Erstes PPO-Training mit Default-Hyperparametern.
