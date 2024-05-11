#!/bin/bash

LOGS="src/common/out"
python3 "task_def.py" &> "$LOGS/debug.out"
