#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number of workers>" >&2
    exit 1
fi

N=$1
DATE=$(date +%Y-%m-%d_%H-%M-%S)

mkdir -p ./logs
mkdir -p ./logs/$DATE

if command -v stdbuf >/dev/null 2>&1; then
    STD_BUF_PREFIX="stdbuf -oL -eL"
elif command -v script >/dev/null 2>&1; then
    STD_BUF_PREFIX=""
    USE_SCRIPT=1
else
    STD_BUF_PREFIX=""
fi

if [ -n "$USE_SCRIPT" ]; then
    script -q -f -c "./bin/learner $N" "./logs/$DATE/learner.log" &
else
    $STD_BUF_PREFIX ./bin/learner $N > "./logs/$DATE/learner.log" 2>&1 &
fi

for i in $(seq 1 $N); do
    if [ -n "$USE_SCRIPT" ]; then
        script -q -f -c "./bin/worker $i" "./logs/$DATE/worker_${i}.log" &
    else
        $STD_BUF_PREFIX ./bin/worker $i > "./logs/$DATE/worker_${i}.log" 2>&1 &
    fi
    sleep 0.10
done

echo "Started learner and $N workers. Logs are in the ./logs/$DATE directory."
echo "To stop all processes, use: pkill -f './bin/learner'; pkill -f './bin/worker'"
echo "To see running processes, use: pgrep -fl './bin/learner'; pgrep -fl './bin/worker'"