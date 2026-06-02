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

**Empfehlung für die Diskussion:** (A2) oder (A3) fürs eigentliche RL-Training (sonst dauert Training Wochen statt Stunden), und (A1) optional fürs Abschluss-Demo. Endgültige Wahl in **ADR-0001** festhalten.

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
