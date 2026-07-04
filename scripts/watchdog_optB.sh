#!/usr/bin/env bash
#
# Watchdog fuer die Option-B-Nachtsession. Wertet den jeweils aktuellen
# `checkpoints/primal2_latest.pt` auf einem Seed-Fach der neuen Verteilung
# aus (20x20 und 40x40 statt 15x15), damit du morgens sehen kannst, wie sich
# das Modell in den Paper-Range-aehnlichen Groessen verhaelt.
#
# Rennt bis das Training aufhoert; ein Zeilenpaar pro neuem Checkpoint in
# logs/eval_optB.csv.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p logs

LOG_FILE="logs/watchdog_optB.log"
CSV_FILE="logs/eval_optB.csv"

# 4 Seeds x 128 Schritte in zwei repraesentativen Groessen; schnell genug,
# um zwischen zwei Checkpoints (alle ~13 min bei 500-Ep-Intervall) fertig zu werden.
PYTHONPATH=. nohup python -m primal2_toy.eval.watchdog \
    --seeds 7 42 123 555 \
    --agents 8 --steps 128 \
    --size 20 --density 0.3 --corridor-length 10 \
    --out "$CSV_FILE" \
    --device cpu > "$LOG_FILE" 2>&1 &

WATCH_PID=$!
echo "$WATCH_PID" > logs/watchdog_optB.pid
echo "watchdog PID=$WATCH_PID (in logs/watchdog_optB.pid)"
echo "Ergebnisse:  $CSV_FILE"
echo "Live-Log:    $LOG_FILE"
