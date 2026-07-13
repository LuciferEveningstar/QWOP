#!/usr/bin/env bash
# Startet das Weitertraining des gruenen Trials 162806 auf 10M (Mac-Testlauf).
# Prueft vorher, dass das Startmodell vorhanden ist, raeumt auf und haelt den
# Rechner wach.
#
# VORHER: models/gruen162806/final.zip vom PC hierher kopieren (siehe Config).
#
# Nutzung:  bash scripts/run_gruen162806.sh
set -u

MODEL="models/gruen162806/final.zip"
CONFIG="configs/ppo_gruen162806_continue.yaml"

if [ ! -f "$MODEL" ]; then
  echo "[run] FEHLER: $MODEL fehlt."
  echo "[run] Kopiere das 162806-final.zip vom PC hierher:"
  echo "[run]   PC: models/ppo_speed_023715/<162806-timestamp>/final.zip"
  echo "[run]   -> Mac: $MODEL"
  exit 1
fi

echo "[run] Startmodell gefunden: $MODEL"
echo "[run] Raeume alte Prozesse ab..."
bash scripts/cleanup.sh 2>/dev/null || true

# Rechner wach halten (Training dauert Stunden auf n_envs=2).
caffeinate -d &
CAFF_PID=$!
echo "[run] caffeinate laeuft (PID $CAFF_PID) — Rechner bleibt wach."

echo "[run] Starte Training (10M Steps)..."
python scripts/train.py --config "$CONFIG" \
  --run-name gruen162806-continue --tags experiment continue gruen

echo "[run] Training beendet. caffeinate stoppen + aufraeumen..."
kill "$CAFF_PID" 2>/dev/null || true
bash scripts/cleanup.sh 2>/dev/null || true
echo "[run] Fertig."
