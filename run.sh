#!/bin/bash

LOGS="src/common/out"
DATA="gold_std"

# Parse the --data flag
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data) DATA="$2"; shift ;;
    esac
    shift
done

python3 "task_def.py" --data "$DATA" > "$LOGS/${DATA}_debug.out"
