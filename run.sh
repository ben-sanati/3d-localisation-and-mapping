#!/bin/bash

LOGS="src/common/out"
python3 "detection.py" > "$LOGS/debug.out"
