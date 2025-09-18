#!/bin/bash


if [ $# -ne 3 ]; then
  echo "Usage: $0 <path> <dataset> <n>"
  exit 1
fi

BASE_PATH=$1
DATASET=$2
N=$3

for file in "$BASE_PATH"/${DATASET}_batch_*.json; do
  if [ -f "$file" ]; then
    RESULT_NAME="${DATASET}_${N}"

    echo "Running: $file"

    uv run experiment \
      --test-venv test_venv/ \
      --dataset "$file" \
      --model gpt-4o \
      --config ./src/viberate/config.json \
      --experiment-result "$RESULT_NAME"
  fi
done
