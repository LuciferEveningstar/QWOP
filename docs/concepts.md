# RL-Grundbegriffe — was steckt hinter QWOP-RL

Dieses Dokument erklärt die Konzepte, die im Rest des Projekts vorausgesetzt werden.
Wenn du noch nie mit Reinforcement Learning gearbeitet hast, fang hier an.

## Inhalt

1. [Was ist ein "Step"?](#1-was-ist-ein-step)
2. [Was ist eine "Observation"?](#2-was-ist-eine-observation)
3. [Was ist eine "Action"?](#3-was-ist-eine-action)
4. [Was ist ein "Reward"?](#4-was-ist-ein-reward)
5. [Was ist eine "Episode"?](#5-was-ist-eine-episode)
6. [Was ist "Training"?](#6-was-ist-training)
7. [Was haben wir konkret gemessen?](#7-was-haben-wir-konkret-gemessen)
8. [Wo siehst du das alles in den Configs?](#8-wo-siehst-du-das-alles-in-den-configs)
9. [Take-aways](#9-take-aways)

---

## 1. Was ist ein "Step"?

### Das Grundprinzip von Reinforcement Learning

Stell dir vor, du sitzt vor QWOP und drückst Tasten. In jeder kleinen Zeit-Scheibe passiert das Gleiche:

1. Du **siehst** den Bildschirm                ← Observation
2. Du **drückst** eine Taste oder Kombi        ← Action
3. Das Spiel **sagt dir**, ob's gut war        ← Reward
4. Der Bildschirm **zeigt** was Neues          ← nächste Observation

Diese vier Phasen zusammen — Sehen, Handeln, Belohnung, neuer Zustand — nennt man einen **Step**.

Genau so läuft es auch beim RL-Agenten. Nur dass statt "du" eben das neuronale Netz da sitzt:

```
Agent ──── Action ─────▶ Environment
   ▲                          │
   │                          │
   └── (Observation, Reward) ─┘
```

Ein Step ist **nicht** ein Spielschritt im Sinne von "ein Trainingslauf von QWOP", sondern eine **einzelne Zeit-Scheibe** innerhalb des Spiels. Eine Episode (= ein QWOP-Lauf vom Start bis zum Sturz oder bis 100m) besteht aus vielen Hundert oder Tausend Steps.

### Wie schnell muss das gehen?

Hier kommt der Knackpunkt vom RL: Der Agent lernt **durch Wiederholung**. Am Anfang macht er nur Quatsch, fällt nach 2 Steps um. Damit er zu was Brauchbarem kommt, muss er **Millionen Steps** machen.

Konkrete Zahlen:

- Ein guter QWOP-Agent braucht ca. **10–50 Millionen Steps** Training
- Bei **30 Steps/s** (= echtes Spielen in Echtzeit) → ~4 bis ~20 Tage Training
- Bei **2.000 Steps/s** → ~1,5 Stunden bis ~7 Stunden

**Deshalb** ist die Steps/Sekunde-Zahl so wichtig. Sie entscheidet, ob das Projekt machbar ist oder nicht.

---

## 2. Was ist eine "Observation"?

In `qwop-gym` ist eine Observation ein **Vektor mit 60 Zahlen**, die den Spielzustand beschreiben.

Was diese 60 Zahlen genau enthalten, ist im qwop-gym-Code festgelegt — vermutlich:

- Position des Torsos (x, y)
- Winkel und Winkelgeschwindigkeit jedes Gelenks (Hüften, Knie, Ellbogen)
- Ob der Kopf den Boden berührt
- Aktuelle Spielzeit, gelaufene Distanz
- usw.

**Wichtig:** Das sind **keine Pixel**. Das ist der entscheidende Unterschied, warum qwop-gym so schnell ist. Der Agent muss nicht aus Bildern erst rausfinden, "was ist überhaupt ein Bein?" — er bekommt die Zahlen direkt geliefert. Das macht Training **viel sample-effizienter**.

In Code:

```python
obs = [0.34, -1.2, 0.05, 0.78, ..., 0.0]   # 60 Zahlen
```

Im Smoke-Test haben wir das beobachtet:

```
-> reset OK in 0.64s, obs.shape=(60,)
```

`obs.shape=(60,)` heißt: ein Array mit 60 Werten.

---

## 3. Was ist eine "Action"?

In QWOP gibt es 4 Tasten: **Q, W, O, P**. Pro Step kann der Agent jede Taste drücken oder loslassen — das gibt **2⁴ = 16 mögliche Kombinationen** (inklusive "keine Taste drücken"):

```
0:  ----      8:   --O-
1:  Q---      9:   Q-O-
2:  -W--      10:  -WO-
3:  QW--      11:  QWO-
4:  ---P      12:  --OP
5:  Q--P      13:  Q-OP
6:  -W-P      14:  -WOP
7:  QW-P      15:  QWOP
```

In `qwop-gym` heißt das **Action Space = Discrete(16)** — der Agent wählt pro Step eine Zahl von 0 bis 15.

In `config/env.yml` (im qwop-gym-Test-Setup) gibt es ein Setting dafür:

```yaml
reduced_action_set: false
```

Auf `true` gestellt, reduziert qwop-gym die 16 Aktionen auf 9 "sinnvolle" (manche Kombinationen wie `QWOP` gleichzeitig sind Quatsch). Kleinerer Action Space = einfacher zu lernen, aber weniger Flexibilität.

---

## 4. Was ist ein "Reward"?

Der Reward ist die **Belohnung pro Step** — eine einzelne Zahl, die dem Agenten sagt: "das war gut" (positiv) oder "das war schlecht" (negativ). Daraus lernt er.

In `qwop-gym` ist der Reward standardmäßig:

```
reward = (zurückgelegte Distanz seit letztem Step) − (Zeit-Strafe)
```

Wenn der Läufer am Ende einer Episode **stürzt**, bekommt er eine Strafe drauf:

```yaml
failure_cost: 10        # → reward -= 10 bei Sturz
success_reward: 50      # → reward += 50 wenn 100m geschafft
time_cost_mult: 10      # → wieviel Zeit-Strafe pro Step
```

Diese drei Werte sind die Knöpfe für **Reward-Engineering** — einer der Bereiche, wo eigene Experimente in der Studienarbeit Sinn ergeben: *"Was passiert, wenn wir `failure_cost` auf 50 hochdrehen?"*

---

## 5. Was ist eine "Episode"?

Eine **Episode** ist ein kompletter QWOP-Durchgang vom Start bis zum Ende:

```
Reset → Step → Step → Step → … → Sturz oder 100m → Episode-Ende
```

In der Trainings-Config:

```yaml
max_episode_steps: 5000
```

heißt: nach spätestens 5000 Steps wird die Episode abgebrochen, auch wenn der Läufer noch nicht gestürzt ist und auch keine 100m erreicht hat. Sicherheitsnetz, damit nichts ewig läuft.

Im Smoke-Test:

```
-> 1291.2 steps/s (200 steps in 0.15s, 2 resets)
```

Das `2 resets` heißt: in 200 Steps ist der Läufer **zweimal gestürzt** und das Env wurde zweimal neu gestartet. Macht Sinn — bei zufälligen Aktionen fällt der Läufer praktisch sofort um.

---

## 6. Was ist "Training"?

Jetzt kommen die einzelnen Bausteine zusammen.

**Training** = die Schleife, in der der Agent durch sein eigenes Verhalten besser wird:

1. Sammle Erfahrung — N Steps mit aktueller Strategie
2. Schau dir an, welche Aktionen welchen Reward gebracht haben
3. Aktualisiere das neuronale Netz so, dass gute Aktionen wahrscheinlicher werden
4. Wiederhole

**PPO** (Proximal Policy Optimization), der Algorithmus den wir benutzt haben, macht das so:

- Sammle `n_steps` Erfahrung (Default: 64)
- Mach damit `n_epochs` (Default: 10) kleine Update-Runden in `batch_size`-Häppchen (Default: 32)
- Wiederhole, bis `total_timesteps` erreicht sind

Im Mini-Training waren das `total_timesteps: 10_000` — das sind die "10k Steps" aus dem Bericht.

---

## 7. Was haben wir konkret gemessen?

Drei verschiedene Geschwindigkeits-Tests:

### a) Benchmark — "wie schnell kann das Env theoretisch Steps machen?"

```
qwop-gym benchmark
→ 1920.64 steps/s (10000 steps in 5.21 seconds)
```

10.000 zufällige Aktionen, kein Lernen, nur "rohe" Env-Geschwindigkeit. **1920 Steps/s** = Browser/WebSocket/Env-Pipeline.

### b) Direktes Env-Skript — "läuft das Env auch außerhalb der CLI?"

```
env_smoketest.py
→ 1291.2 steps/s (200 steps in 0.15s, 2 resets)
```

Etwas langsamer (1291 vs. 1920), weil Resets nach Sturz Zeit kosten — bei nur 200 Steps fallen die zwei Resets stärker ins Gewicht. Bei 10.000 Steps würde sich das ausmitteln.

### c) Echtes Training — "wie schnell mit Lernen?"

```
qwop-gym train_ppo (10.000 timesteps)
→ duration: 22.92 Sekunden
→ ~436 Steps/s effektiv
```

Hier kommt das **Lernen** dazu — alle 64 Steps wird das neuronale Netz aktualisiert (10 Epochen × 32er-Batches). Das ist Rechenzeit, die zum reinen Step-Sammeln dazukommt. Daher ~436 Steps/s statt 1920.

**Hochgerechnet auf ein echtes Training (10 Mio Steps):**

- Bei 436 Steps/s → ca. **6,4 Stunden**
- Im Vergleich Browser-naiv (30 Steps/s) wären das **~93 Stunden ≈ 4 Tage**

Das ist der Unterschied, den qwop-gym macht.

---

## 8. Wo siehst du das alles in den Configs?

Die Datei `config/env.yml` (im qwop-gym-Test-Setup) bündelt fast alle Konzepte aus oben:

```yaml
browser:    "/Applications/Google Chrome.app/..."
            # ↑ Welcher Browser

driver:     "/Users/.../bin/chromedriver"
            # ↑ Welcher ChromeDriver — manuell auf passende Version fixiert

render_mode: "browser"
            # ↑ "browser" = Browser rendert das Spiel
            #   "rgb_array" = Render gibt RGB-Bild zurück (für Pixel-Obs)

failure_cost: 10
success_reward: 50
time_cost_mult: 10
            # ↑ Reward-Engineering-Knöpfe (siehe Abschnitt 4)

frames_per_step: 1
            # ↑ Frame-Skipping. 1 = jeden Frame sehen.
            #   Höher = Agent agiert seltener, schneller Training.

stat_in_browser: false
game_in_browser: false
            # ↑ Beim Training aus, weil Rendering Zeit kostet

reload_on_reset: false
            # ↑ Bei Episodenende: nur Game zurücksetzen statt
            #   ganze Browserseite neu laden — viel schneller.

reduced_action_set: false
            # ↑ false = 16 Aktionen, true = 9 Aktionen (siehe Abschnitt 3)
```

Das sind die Knöpfe, die ihr in der Studienarbeit drehen werdet, wenn ihr Reward-Engineering oder Action-Space-Vergleiche macht.

---

## 9. Take-aways

1. **Step ≠ ganzes Spiel.** Ein Step ist eine winzige Zeit-Scheibe (Frame oder paar Frames). Eine Episode hat hunderte/tausende Steps. Ein Training hat Millionen Steps.
2. **Steps/s ist die wichtigste Performance-Metrik.** Sie entscheidet, ob Training Stunden oder Tage dauert.
3. **Observation = was der Agent sieht** (bei uns: 60 Zahlen, kein Bild).
4. **Action = was der Agent tut** (bei uns: eine Zahl 0-15 für Tastenkombi).
5. **Reward = die Belohnung pro Step**, aus der gelernt wird (bei uns: hauptsächlich Distanz-Fortschritt).
6. **`config/env.yml`** ist die Spielwiese für Env-Verhalten — **`config/train_ppo.yml`** für Hyperparameter.

---

## Weiterführend

- [`docs/architecture.md`](architecture.md) — wie wir QWOP an den Agent anbinden
- [`docs/wandb-setup.md`](wandb-setup.md) — wie ihr Trainingsläufe loggt und teilt
- [Stable-Baselines3 — PPO Doku](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) — sehr gute Einführung von OpenAI
