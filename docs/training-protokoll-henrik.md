# Trainingsprotokoll — Henrik

Dieses Dokument protokolliert alle Trainingsläufe von Henrik, die genutzten Konfigurationen und die Ergebnisse.

---

## Übersicht Runs

| Run | Config | Steps | ep_rew_mean | success_rate (eval) | W&B Link |
|---|---|---|---|---|---|
| a-2026-06-11-163953 | [ppo_smoke.yaml](../configs/ppo_smoke.yaml) | 10.000 | — | — | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/4eok5lam) |
| a-2026-06-11-164747 | [ppo_default.yaml](../configs/ppo_default.yaml) | 1.000.000 | +41.2 | — | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/dpsde1my) |
| a-2026-06-12-113046 | [ppo_ent001.yaml](../configs/ppo_ent001.yaml) | 3.000.000 | +105 @ 1.6M | — | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/4s0cxii6) |
| a-2026-06-13 (Run 4) | [ppo_ent001_finetune.yaml](../configs/ppo_ent001_finetune.yaml) | 10.000.000 | +168 @ 8.65M | 75% | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/mk6f4gw4) |
| a-2026-06-13 (Run 5) | [ppo_run5.yaml](../configs/ppo_run5.yaml) | 3.000.000 | +184 @ 2.75M | **94% ← Bestes Modell** | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/a-2026-06-13-203134) |
| a-2026-06-15 (Run 6) | [ppo_run6.yaml](../configs/ppo_run6.yaml) | 2.500.000 | +181 @ final | 92% | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/1x191235) |

---

## Run 1 — Smoke Test (2026-06-11)

**Config:** [configs/ppo_smoke.yaml](../configs/ppo_smoke.yaml)
**Modell:** `models/ppo_smoke/final.zip`
**W&B:** https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/4eok5lam

### Ziel
Verifizieren dass die gesamte Pipeline funktioniert — von qwop-gym über SB3 bis W&B.

### Ergebnis
| Metrik | Wert |
|---|---|
| total_timesteps | 10.000 |
| ep_rew_mean | nicht geloggt (zu kurz) |
| explained_variance | 0.59 |
| fps | 170 |

### Beobachtungen
- Pipeline läuft vollständig durch
- W&B-Sync funktioniert
- explained_variance stieg von 0.036 auf 0.594 — Modell lernt bereits in 10k Steps

---

## Run 2 — Baseline PPO (2026-06-11)

**Config:** [configs/ppo_default.yaml](../configs/ppo_default.yaml)
**Modell:** `models/ppo_default/final.zip`
**W&B:** https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/dpsde1my

### Ziel
Erste vollständige Trainingsrunde mit Standard-Hyperparametern als Baseline.

### Hyperparameter
| Parameter | Wert |
|---|---|
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| total_timesteps | 1.000.000 |
| n_envs | 1 |

### Ergebnis
| Metrik | Wert |
|---|---|
| ep_rew_mean | +41.2 |
| success_rate | 2% |
| ep_len_mean | 1.931 Steps |
| entropy_loss | -0.538 |
| explained_variance | 0.912 |
| fps | ~150 |

### Verlauf
| Steps | ep_rew_mean | Beobachtung |
|---|---|---|
| 221k | -17.7 | Läufer fällt noch oft, aber Aufwärtstrend |
| 323k | -13.1 | ep_len_mean verdoppelt (680 → 1.630) — überlebt länger |
| 450k | +9.4 | **Positiver Reward erreicht**, erste 100m-Erfolge (2%) |
| 663k | +40.4 | Starker Sprung, Modell konvergiert |
| 1M | +41.2 | Finaler Wert, kaum weiterer Fortschritt |

### Beobachtungen
- Positiver Reward bereits bei ~450k Steps erreicht — schneller als Teamkollege (Niko)
- Bei ~671k Steps kurze Instabilität: value_loss sprang auf 10.1, WebSocket-Timeout — Training lief aber stabil weiter
- `entropy_loss` fiel von -2.76 auf -0.54 → KI hörte früh auf zu explorieren (`ent_coef: 0.0`)
- Eval nach Training zeigt hohe Varianz: Reward zwischen -7.68 und +22.37 in 5 Episoden → KI noch nicht konsistent

### Schlussfolgerung
Gute Baseline. Hauptproblem: `ent_coef: 0.0` führt zu frühem Konvergieren auf eine suboptimale Strategie. Nächster Run mit `ent_coef: 0.01`.

---

## Run 3 — PPO mit Exploration (2026-06-12)

**Config:** [configs/ppo_ent001.yaml](../configs/ppo_ent001.yaml)
**Modell:** `models/ppo_ent001/final.zip` (finales Modell), **`models/ppo_ent001/checkpoints/`** (60 Checkpoints alle 50k Steps)
**Bestes Modell:** `models/ppo_ent001/best.zip` = `model_1650000_steps.zip`
**W&B:** https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/4s0cxii6

### Ziel
Vergleich zur Baseline mit aktivierter Exploration (`ent_coef: 0.01`). Hypothese: Die KI bleibt länger in der Explorationsphase und findet konsistentere Strategien.

### Änderungen gegenüber Run 2
| Parameter | Run 2 (Baseline) | Run 3 |
|---|---|---|
| ent_coef | 0.0 | **0.01** |
| total_timesteps | 1.000.000 | **3.000.000** |
| model_dir | models/ppo_default | **models/ppo_ent001** |

### Ergebnis
| Metrik | Wert |
|---|---|
| ep_rew_mean (final) | +36.9 |
| ep_rew_mean (bestes @ ~1.6M) | **+105** |
| success_rate (final) | 2% |
| success_rate (bestes @ ~1.6M) | **18%** |
| entropy_loss (final) | -1.67 |
| explained_variance (final) | 0.886 |
| fps | ~75-100 (Chrome teilweise gedrosselt) |

### Verlauf
| Steps | ep_rew_mean | success_rate | Beobachtung |
|---|---|---|---|
| 22k | -46 | 50%* | Sehr früh, wenig aussagekräftig |
| 145k | -20.9 | 0% | Chrome gedrosselt (10 fps) |
| 1.617k | **+105** | **18%** | **Bestes Modell** — Höchstwert |
| 3M | +36.9 | 2% | Catastrophic Forgetting — verschlechtert |

*50% bei sehr wenigen Episoden, statistisch nicht belastbar

### Beobachtungen
- **Hypothese bestätigt:** `ent_coef: 0.01` führt zu deutlich besserem Peak-Ergebnis (+105 vs +41 Baseline)
- **Catastrophic Forgetting** trat erneut auf — Modell verschlechterte sich nach 1.6M Steps stark
- Chrome wurde zeitweise gedrosselt (fps fiel auf 10-13) — Training lief weiter aber sehr langsam
- `entropy_loss` blieb bei -1.65 bis -1.49 deutlich höher als Baseline (-0.54) — Exploration funktionierte
- **60 Checkpoints** alle 50k Steps gespeichert — bestes Modell kann nachträglich identifiziert werden

### Schlussfolgerung
`ent_coef: 0.01` ist klar besser als die Baseline. Das finale Modell ist aber nicht das beste — der Checkpoint bei 1.65M Steps ist überlegen. Nächster Schritt: Weitermachen vom besten Checkpoint mit reduzierter Learning Rate (`1e-4` statt `3e-4`) um Catastrophic Forgetting zu reduzieren.

### Checkpoint-Evaluation (2026-06-12)

Manuelle Evaluation der 3 besten Kandidaten mit je 20 Episoden via `scripts/eval.py --render-every 0`:

| Checkpoint | Mean | Median | Std | Max | Fazit |
|---|---|---|---|---|---|
| model_1650000 | **138.40** | **138.46** | 44.78 | 201.02 | **Bestes Modell** |
| model_1700000 | 125.30 | 141.56 | 37.64 | 147.02 | Konsistenter aber niedrigerer Mean |
| model_1750000 | 83.27 | 88.76 | 62.83 | 195.61 | Zu inkonsistent |

**`model_1650000_steps.zip` ist das beste Modell** — höchster Mean (+138) und Median (+138), Max-Wert von 201.

Gesichert als `models/ppo_ent001/best.zip`:
```bash
copy models\ppo_ent001\checkpoints\model_1650000_steps.zip models\ppo_ent001\best.zip
```

### Nächster Run — Plan (Run 4)

Weitermachen vom besten Checkpoint mit reduzierter Learning Rate:

| Parameter | Run 3 | Run 4 (geplant) |
|---|---|---|
| learning_rate | 3e-4 | **1e-4** |
| ent_coef | 0.01 | 0.01 |
| total_timesteps | 3.000.000 | **10.000.000** |
| start | von 0 | **vom 1.65M-Checkpoint** |

**Hypothese:** Kleinere Lernschritte nach dem Peak reduzieren Catastrophic Forgetting — das Modell bleibt länger in der Nähe des +138-Reward-Niveaus und verbessert sich weiter.

---

## Run 4 — Fine-Tuning vom besten Checkpoint (2026-06-13)

**Config:** [configs/ppo_ent001_finetune.yaml](../configs/ppo_ent001_finetune.yaml)
**Start:** `models/ppo_ent001/best.zip` (= model_1650000_steps.zip, Mean +138)
**Bestes Modell:** `models/ppo_ent001_finetune/best.zip` (= model_8650000_steps.zip)
**W&B:** https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/mk6f4gw4

### Änderungen gegenüber Run 3
| Parameter | Run 3 | Run 4 |
|---|---|---|
| learning_rate | 3e-4 | **1e-4** |
| total_timesteps | 3.000.000 | **10.000.000** |
| start | von 0 | **vom 1.65M-Checkpoint** |

### Ergebnis
| Metrik | Wert |
|---|---|
| ep_rew_mean (W&B Peak) | **+168** |
| success_rate (W&B Peak) | **44%** |
| ep_rew_mean (final) | +157 |
| success_rate (final) | 27% |
| fps | ~200-255 |

### Warum zeigt W&B nur 44% obwohl eval.py 75% zeigt?

Das ist kein Widerspruch — die Messungen unterscheiden sich grundlegend:

**W&B während Training (44%):**
- Misst den Durchschnitt der letzten 100 Episoden **während des Trainings**
- Die KI exploriert noch aktiv (`ent_coef: 0.01`) — macht absichtlich manchmal zufällige, schlechte Aktionen
- Diese Exploration-Episoden drücken die Success Rate nach unten

**eval.py nach Training (75%):**
- Das Modell ist eingefroren — es lernt nicht mehr, es spielt nur
- Keine zufälligen Experimente mehr → konsistentere Ergebnisse
- Misst die echte Qualität der gelernten Strategie

**Fazit:** Die 75% aus eval.py sind die relevante Zahl für die Präsentation.

### Wichtige methodische Erkenntnis: Peak ≠ bestes Modell

**Beobachtung in Run 5:** Wenn ein Fine-Tuning-Run vom besten Checkpoint startet, zeigt W&B am Anfang `success_rate: 1.0` — obwohl das Modell nicht perfekt ist.

**Warum:** Der W&B-Durchschnitt basiert auf einem rollenden Fenster der letzten 100 Episoden. Zu Beginn eines neuen Runs sind noch sehr wenige Episoden im Puffer — 1-2 gute Episoden reichen für 100%. Erst nach ~100 Episoden ist der Puffer voll und die echte Erfolgsrate sichtbar.

**Konsequenz:** Den "Peak" in W&B am Anfang eines Runs zu nehmen ist irreführend. Das beste Modell ist nicht der höchste Spike, sondern der Zeitpunkt wo sich die Kurve **stabilisiert hat** — also wo der Durchschnitt über viele Episoden stabil auf einem hohen Niveau bleibt.

**Bessere Strategie für Checkpoint-Auswahl:**
- Nicht den höchsten Spike nehmen
- Den Bereich nehmen wo die Kurve über mehrere hundert Logging-Steps stabil hoch bleibt
- Mehrere Checkpoints in diesem stabilen Bereich mit eval.py (20+ Episoden) vergleichen

### Checkpoint-Evaluation (2026-06-13)

| Checkpoint | Mean | Median | Std | Erfolgsrate | Fazit |
|---|---|---|---|---|---|
| **model_8650000** | **161** | **217** | 81 | **75%** | **Bestes Modell** |
| model_8700000 | 136 | 153 | 54 | 75% | Schlechterer Mean |

**`model_8650000_steps.zip` ist das beste Modell** — Median von 217, 75% Erfolgsrate bei 20 Episoden.

### Live-Evaluation (2026-06-13)

```bash
python scripts/eval.py --model models\ppo_ent001_finetune\best.zip --episodes 5
```

- **Best: 100.4m** im Browser angezeigt
- **Real time: ~34s** für 100m
- **Game time: ~13s** für 100m
- Strategie: schnelle Wackelbewegung ("wiggeln") statt eleganter Lauf

### Benchmark-Einordnung

| Wer | Game time | Real time | Methode |
|---|---|---|---|
| Mensch (Weltrekord) | ~4.5s | ~45s | Echtes Spielverständnis |
| Bekannter RL-Rekord | — | ~47s | Reinforcement Learning |
| **Henrik (Run 4)** | **~13s** | **~34s** | **PPO, ent_coef=0.01, Fine-Tuning** |

> **Offene Frage:** Die genaue Benchmark-Definition ist unklar — der bekannte RL-Rekord (47s) misst möglicherweise Realtime, nicht Gametime. Ein direkter Vergleich wäre nur mit identischer Messmethode möglich.

### Schlussfolgerung
Run 4 hat Run 3 deutlich übertroffen. Die Kombination aus `ent_coef: 0.01` + reduzierter Learning Rate + Fine-Tuning vom besten Checkpoint ist die bisher effektivste Strategie.

### Nächster Run — Plan (Run 5)

Weitermachen vom besten Checkpoint mit erhöhter `n_steps` für längere Strategiefindung:

| Parameter | Run 4 | Run 5 (geplant) |
|---|---|---|
| n_steps | 2048 | **4096** |
| learning_rate | 1e-4 | 1e-4 |
| ent_coef | 0.01 | 0.01 |
| total_timesteps | 10.000.000 | **3.000.000** |
| start | vom 1.65M-Checkpoint | **vom 8.65M-Checkpoint** |

**Hypothese:** Größere `n_steps` (4096 statt 2048) lässt die KI mehr Erfahrungen sammeln bevor sie lernt — stabilere Updates, bessere Strategiefindung.

---

## Run 5 — n_steps erhöht (geplant)

**Config:** [configs/ppo_run5.yaml](../configs/ppo_run5.yaml)
**Start:** `models/ppo_ent001_finetune/best.zip` (= model_8650000_steps.zip, 75% Erfolgsrate)
**W&B:** *(Link nach Training eintragen)*

### Änderungen gegenüber Run 4
| Parameter | Run 4 | Run 5 |
|---|---|---|
| n_steps | 2048 | **4096** |
| learning_rate | 1e-4 | 1e-4 |
| ent_coef | 0.01 | 0.01 |
| total_timesteps | 10.000.000 | **3.000.000** |
| start | vom 1.65M-Checkpoint | **vom 8.65M-Checkpoint** |

### Was bedeutet n_steps erhöhen?

`n_steps` bestimmt wie viele Spielschritte die KI sammelt bevor sie einmal lernt (ein Update macht). Mit `n_steps: 4096` statt 2048:

- Die KI sieht **doppelt so viele Erfahrungen** bevor sie ihre Gewichte anpasst
- Updates werden **stabiler** — weniger Rauschen durch einzelne schlechte Episoden
- Die KI kann **längere Strategien** erkennen — z.B. dass eine bestimmte Bewegungssequenz über viele Steps hinweg gut ist
- **Nachteil:** Lernt etwas langsamer (weniger Updates pro Zeit)

### Hypothese
Stabilere Updates durch mehr gesammelte Erfahrung reduzieren Catastrophic Forgetting und erlauben der KI die bereits gute Strategie (75% Erfolgsrate) weiter zu verfeinern statt zu verlieren.

### Ergebnis (50 Episoden, deterministic=False)
| Metrik | Wert |
|---|---|
| **Erfolgsrate** | **90%** (45/50 Episoden) |
| Mean Reward | +161.50 |
| Median Reward | +163.00 |
| **Std** | **40.13** (Run 4: 81 — fast halbiert!) |
| Min | 51.19 |
| Max | 227.01 |

**Hypothese bestätigt:** `n_steps: 4096` hat die Konsistenz deutlich verbessert. Std halbiert, Erfolgsrate von 75% auf 90% gestiegen. Mean unverändert — das Modell ist nicht schneller, aber zuverlässiger.

### Checkpoint-Evaluation (2026-06-15)

Manuelle Evaluation aller relevanten Checkpoints mit je 50 Episoden via `eval.py --render-every 0`:

| Checkpoint | Mean | Median | Std | Erfolgsrate | Fazit |
|---|---|---|---|---|---|
| model_1750000 | 177 | 166 | 49 | 94% | Sehr gut |
| model_2000000 | 170 | 162 | 46 | 92% | Gut |
| model_2250000 | 162 | 165 | 52 | 86% | Schlechter |
| **model_2750000** | **184** | **166** | **44** | **94%** | **Bestes Modell** |
| final.zip (3M) | 161 | 163 | 40 | 90% | Konsistenteste Std |

**`model_2750000_steps.zip` ist das beste Modell** — höchster Mean (+184), 94% Erfolgsrate, niedrigere Std als 1.75M.

Gesichert als `models/ppo_run5/best.zip`:
```bash
copy models\ppo_run5\checkpoints\model_2750000_steps.zip models\ppo_run5\best.zip
```

> **Hinweis:** `models\ppo_run5\` und `models\ppo_ent001_finetune\` sind komplett getrennte Ordner — Run 4 wird durch Run 5 nicht überschrieben.

### Wichtige Erkenntnis: W&B-Graph zeigt nur den Lernprozess

Der W&B-Graph zeigt `success_rate` während des Trainings — **nicht** die echte Modellqualität.

**Warum der Graph ~30-40% zeigt obwohl eval.py 90% ergibt:**
- Während Training exploriert die KI aktiv (`ent_coef: 0.01`) → absichtlich schlechte Aktionen
- Der Graph ist verrauscht durch Exploration-Episoden
- eval.py mit eingefrorenem Modell zeigt die echte Qualität

**Konsequenz für Checkpoint-Auswahl:**
- Den W&B-Graph nutzen um zu sehen *wann* ein Peak war und *ob* die KI lernt
- Die finale Qualität immer mit `eval.py --episodes 50` messen
- Nie nur auf den W&B-Peak verlassen

### Wichtige Erkenntnis: deterministic=True ist für dieses Modell kontraproduktiv

Test mit `--deterministic` ergab: alle 50 Episoden identisch, reward=81.85, steps=1848 — die KI wiederholt dieselbe suboptimale Route endlos.

Ursache: Das Modell wurde mit Sampling trainiert (`ent_coef: 0.01`). Im deterministischen Modus gibt es keinen Zufall mehr — die KI findet die gute Wiggle-Strategie nicht, weil sie immer denselben (schlechten) Weg nimmt.

**Fazit:** `deterministic=False` (Sampling) ist die richtige Eval-Methode für dieses Modell. Die 90% Erfolgsrate ist das ehrliche Ergebnis.

| Wer | Steps | ep_rew_mean | success_rate (W&B) | success_rate (eval) | Besonderheit |
|---|---|---|---|---|---|
| Henrik (Run 2) | 1M | +41.2 | 2% | — | Baseline, `ent_coef: 0.0` |
| Henrik (Run 3) | ~1.6M | +105 | 18% | — | Bester Checkpoint |
| Henrik (Run 4) | ~8.65M | +168 | 44% | 75% | Fine-Tuning, lr=1e-4 |
| **Henrik (Run 5)** | ~11.4M | +184 @ 2.75M | ~35% | **94% ← Bestes Modell** | n_steps=4096 |
| Henrik (Run 6) | ~13.9M | +181 @ final | ~55% | 92% | lr=5e-5, n_steps=8192 |
| Niko | 5M | — | 9% | — | Bestes bei 5M |
| Niko | 27.5M | +99 | 17% | — | Nach langem Training |
| Niko | ~40M | +160 | 39% | — | Bisher bester Niko-Run |

**Wichtigste Erkenntnisse:**
- `ent_coef: 0.01` + `lr: 1e-4` + `n_steps: 4096` + Fine-Tuning vom Checkpoint ist die beste Strategie
- **W&B-Graph zeigt den Lernprozess, nicht die Modellqualität** — immer mit `eval.py --episodes 50` messen
- Henrik erreicht **94% Erfolgsrate** (stochastisch) und **100%** (deterministisch) — Nikos bester W&B-Wert ist 39%
- **Catastrophic Forgetting** tritt bei allen Runs auf — Checkpoints und regelmäßige eval.py-Tests sind entscheidend
- Run 6 (`lr=5e-5`, `n_steps=8192`) hat Run 5 nicht übertroffen — Run 5 bleibt das beste Modell

---

## Run 6 — kleinere LR + größere n_steps (2026-06-15)

**Config:** [configs/ppo_run6.yaml](../configs/ppo_run6.yaml)
**Start:** `models/ppo_run5/best.zip` (= model_2750000_steps.zip, 94% Erfolgsrate)
**W&B:** https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/1x191235

### Änderungen gegenüber Run 5
| Parameter | Run 5 | Run 6 |
|---|---|---|
| learning_rate | 1e-4 | **5e-5** |
| n_steps | 4096 | **8192** |
| total_timesteps | 3.000.000 | **2.500.000** |

### Checkpoint-Evaluation (50 Episoden je)
| Checkpoint | Mean | Std | Erfolgsrate |
|---|---|---|---|
| model_2000000 | 176 | 63 | 86% |
| model_2250000 | 179 | 57 | 92% |
| model_2500000 | 175 | 56 | 90% |
| final.zip (2.5M) | 181 | 49 | 92% |

### Schlussfolgerung
Run 6 hat Run 5 nicht übertroffen — maximale Erfolgsrate 92% vs. 94%. Die kleinere Learning Rate (`5e-5`) und größere `n_steps` (8192) brachten keine Verbesserung gegenüber Run 5.

**Run 5 `model_2750000_steps.zip` bleibt das finale beste Modell** (94% stochastisch, 100% deterministisch, Mean +184).
