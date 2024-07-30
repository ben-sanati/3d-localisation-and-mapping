#!/bin/bash

DATA="gold_std"
SETUP=false

# Parse the --data flag
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data) DATA="$2"; shift ;;
        --setup) SETUP=true ;;
    esac
    shift
done

# Run the setup Python file if --setup is true
if $SETUP; then
    python3 "src/common/data/setup.py" --data "$DATA"
fi

python3 "task_def.py" --data "$DATA"
