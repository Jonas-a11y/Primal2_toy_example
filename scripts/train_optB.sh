#!/usr/bin/env bash
#
# Option-B-Nachtlauf: warm-started Retraining auf enger Paper-Range-Verteilung.
#
# Vor dem Start:
#   1. Battery-Modus auf "Wenn Netzstrom" → "Nie ausschalten".
#      System Settings → Battery → Power Adapter → Prevent automatic sleeping.
#   2. Caffeinate wird unten automatisch gestartet.
#   3. checkpoints/primal2_final.pt muss existieren (Warm-Start-Quelle).
#
# Startet das Training im Hintergrund, schreibt logs/train_stdout_optB.log,
# und liefert die PID zurück, damit du es notfalls per `kill <pid>` stoppen kannst.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
CKPT_SRC="checkpoints/primal2_final.pt"
LOG_FILE="logs/train_stdout_optB.log"
DEADLINE_HOURS="${DEADLINE_HOURS:-10}"
SEED="${SEED:-2026}"

if [[ ! -f "$CKPT_SRC" ]]; then
    echo "ERROR: Warm-Start-Checkpoint fehlt: $CKPT_SRC" >&2
    exit 1
fi

mkdir -p logs checkpoints

# Wenn `caffeinate` verfügbar ist (macOS), Deckel-Schlafen unterdrücken,
# solange das Training läuft.
CAFFEINATE_CMD=""
if command -v caffeinate >/dev/null 2>&1; then
    CAFFEINATE_CMD="caffeinate -is"
fi

echo "Startzeit:            $(date '+%F %T')"
echo "Deadline:             ${DEADLINE_HOURS} h"
echo "Ende erwartet:        $(date -v+${DEADLINE_HOURS}H '+%F %T')"
echo "Warm-Start von:       $CKPT_SRC"
echo "Log:                  $LOG_FILE"
echo "Repo:                 $REPO_ROOT"
echo

# Venv aktivieren, falls vorhanden.
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Kommando bauen; alles was Option B ausmacht ist hier.
CMD=(
    python -m primal2_toy.train.main
    --sizes 20 30 40
    --density-low 0.3 --density-high 0.5
    --corridors 5 10 15
    --n-agents 8
    --device mps
    --episodes 100000
    --deadline-hours "$DEADLINE_HOURS"
    --warmstart-weights "$CKPT_SRC"
    --il-warmup-episodes 200
    --seed "$SEED"
    --log-every 50
    --ckpt-every 500
)

echo "Kommando:"
printf '  %s\n' "${CMD[@]}"
echo

# nohup + im Hintergrund + caffeinate.
if [[ -n "$CAFFEINATE_CMD" ]]; then
    PYTHONPATH=. nohup $CAFFEINATE_CMD "${CMD[@]}" \
        > "$LOG_FILE" 2>&1 &
else
    PYTHONPATH=. nohup "${CMD[@]}" \
        > "$LOG_FILE" 2>&1 &
fi

PID=$!
echo "$PID" > logs/train_optB.pid
echo "gestartet, PID=$PID (in logs/train_optB.pid gespeichert)"
echo "Fortschritt beobachten: tail -f $LOG_FILE"
echo "Stoppen:                 kill \$(cat logs/train_optB.pid)"
