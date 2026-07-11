#!/usr/bin/env bash
# QWOP-Trainings-Prozesse sauber aufräumen.
#
# Warum nötig: Wird train.py hart abgebrochen (Ctrl-C erwischt nicht alles) oder
# währenddessen chromedriver gekillt, bleiben verwaiste Prozesse zurück:
#   - der/die Python-Worker (SubprocVecEnv → "multiprocessing.spawn ... --multiprocessing-fork")
#   - deren qwop-gym-WSServer-Kinder
#   - chromedriver + die headless/sichtbaren Chrome-Instanzen (user-agent=Chrome-<uuid>)
# Ein einfaches `pkill -f chromedriver` reicht NICHT — solange train.py lebt,
# respawnt qwop-gym den Browser. Reihenfolge: erst Eltern (Python), dann Kinder.
#
# WICHTIG: trifft NUR QWOP-Trainings-Prozesse, nicht deinen Alltags-Chrome
# (der hat keinen "user-agent=Chrome-<uuid>"-Marker) und nicht VS Code.
#
# Nutzung:  bash scripts/cleanup.sh
set -u

echo "[cleanup] Stoppe Trainings-Prozesse (Eltern zuerst)..."
# 1) Eltern: Trainer + Sweep-Agent + SubprocVecEnv-Worker
pkill -9 -f "scripts/train.py"                  2>/dev/null
pkill -9 -f "wandb agent"                        2>/dev/null
pkill -9 -f "multiprocessing.spawn"              2>/dev/null
pkill -9 -f "multiprocessing.resource_tracker"   2>/dev/null

# 2) Kinder: qwop-gym-Browser + Driver
pkill -9 -f "user-agent=Chrome-"                 2>/dev/null
pkill -9 -f chromedriver                         2>/dev/null

sleep 1

# 3) Verifikation — was noch übrig ist, anzeigen
leftovers=$(ps aux | grep -iE "scripts/train.py|multiprocessing-fork|chromedriver|user-agent=Chrome-" | grep -v grep || true)
if [ -z "$leftovers" ]; then
  echo "[cleanup] Sauber — keine QWOP-Trainings-Prozesse mehr."
else
  echo "[cleanup] WARNUNG: folgende Prozesse leben noch (ggf. PID manuell killen):"
  echo "$leftovers"
fi
