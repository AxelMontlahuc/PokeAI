#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <number_of_runs>" >&2
  exit 1
fi

N=$1
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -le 0 ]; then
  echo "Error: <number_of_runs> must be a positive integer." >&2
  exit 2
fi

PROG="./bin/savemaker"
if [ ! -x "$PROG" ]; then
  echo "Error: $PROG not found or not executable. Build it with 'make' first." >&2
  exit 3
fi

DATE=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p ./logs ./logs/$DATE

if command -v stdbuf >/dev/null 2>&1; then
  STD_BUF_PREFIX="stdbuf -oL -eL"
elif command -v script >/dev/null 2>&1; then
  STD_BUF_PREFIX=""
  USE_SCRIPT=1
else
  STD_BUF_PREFIX=""
fi

for i in $(seq 1 "$N"); do
  LOGFILE="./logs/$DATE/savemaker_${i}.log"
  if [ -n "${USE_SCRIPT:-}" ]; then
    script -q -f -c "$PROG $i" "$LOGFILE" &
  else
    $STD_BUF_PREFIX $PROG $i >"$LOGFILE" 2>&1 &
  fi
  pid=$!
  echo "Launched savemaker #$i (PID $pid) -> $LOGFILE"
  sleep 0.10
done

echo "Started $N savemaker instances. Logs: ./logs/$DATE"