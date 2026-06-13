# Trainingsprotokoll — Henrik

Dieses Dokument protokolliert alle Trainingsläufe von Henrik, die genutzten Konfigurationen und die Ergebnisse.

---

## Übersicht Runs

| Run | Config | Steps | ep_rew_mean | success_rate | W&B Link |
|---|---|---|---|---|---|
| a-2026-06-11-163953 | [ppo_smoke.yaml](../configs/ppo_smoke.yaml) | 10.000 | — | — | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/4eok5lam) |
| a-2026-06-11-164747 | [ppo_default.yaml](../configs/ppo_default.yaml) | 1.000.000 | +41.2 | 2% | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/dpsde1my) |
| a-2026-06-12-113046 | [ppo_ent001.yaml](../configs/ppo_ent001.yaml) | 3.000.000 | +36.9 (final) / **+105 @ 1.6M** | 2% (final) / **18% @ 1.6M** | [Link](https://wandb.ai/qwop-rl/qwop-rl-dhbw/runs/4s0cxii6) |

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

## Vergleich mit Teamkollegen

| Wer | Steps | ep_rew_mean | success_rate (W&B) | success_rate (eval) | Besonderheit |
|---|---|---|---|---|---|
| Henrik (Run 2) | 1M | +41.2 | 2% | — | Baseline, `ent_coef: 0.0` |
| Henrik (Run 3) | ~1.6M | +105 | 18% | — | Bester Checkpoint |
| Henrik (Run 4) | ~8.65M | +168 | **44%** | **75%** | **Bestes Modell, Fine-Tuning** |
| Niko | 5M | — | 9% | — | Bestes bei 5M |
| Niko | 27.5M | +99 | 17% | — | Nach langem Training |
| Niko | ~40M | +160 | 39% | — | Bisher bester Niko-Run |

**Wichtigste Erkenntnisse:**
- `ent_coef: 0.01` + niedrige Learning Rate + Fine-Tuning vom Checkpoint ist die effektivste Strategie
- Henrik erreicht **75% Erfolgsrate** (eval) bei ~8.65M Steps — Niko hat bei 40M Steps 39% (W&B, nicht direkt vergleichbar)
- **Catastrophic Forgetting** tritt bei beiden auf — Checkpoints sind entscheidend
- W&B Success Rate während Training ≠ eval.py Success Rate nach Training (Exploration vs. reines Spielen)
