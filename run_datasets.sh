#!/usr/bin/env bash
set -euo pipefail

datasets=(100_0 100_1 100_2 100_3 100_4 100_5 100_6 100_7 100_8 100_9 150_0 150_1 150_2 150_3 150_4 150_5 150_6 150_7 150_8 150_9 200_0 200_1 200_2 200_3 200_4 200_5 200_6 200_7 200_8 200_9 250_0 250_1 250_2 250_3 250_4 250_5 250_6 250_7 250_8 250_9 300_1 300_2 300_3 300_4 300_5 300_6 300_7 300_8 300_9)

# Comma-separated GPU list, e.g. GPU_IDS=0,1,2
gpu_ids_csv="${GPU_IDS:-0}"
IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
gpu_count=${#gpu_ids[@]}

# How many concurrent jobs to allow per GPU. Set env var MAX_JOBS_PER_GPU to override.
# Increased default to 5 for higher concurrency per GPU.
MAX_JOBS_PER_GPU=${MAX_JOBS_PER_GPU:-5}

# Log file for launched PIDs and GPU assignments
RUN_PIDS_LOG=${RUN_PIDS_LOG:-run_pids.log}

pick_python() {
  local candidate
  local output

  if [ -n "${PYTHON_BIN:-}" ] && { command -v "$PYTHON_BIN" >/dev/null 2>&1 || [ -x "$PYTHON_BIN" ]; }; then
    echo "$PYTHON_BIN"
    return 0
  fi

  for candidate in \
    "/mnt/c/Users/admin/Downloads/StrongBarrierCoverage-/venv/Scripts/python.exe" \
    "python3" \
    "python"; do
    if ! command -v "$candidate" >/dev/null 2>&1 && [ ! -x "$candidate" ]; then
      continue
    fi

    output=$("$candidate" -c "from gpu_utils import has_gpu; print('GPU' if has_gpu else 'CPU')" 2>/dev/null || true)
    if [ "$output" = "GPU" ]; then
      echo "$candidate"
      return 0
    fi
  done

  for candidate in \
    "/mnt/c/Users/admin/Downloads/StrongBarrierCoverage-/venv/Scripts/python.exe" \
    "python3" \
    "python"; do
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done

  echo "No Python interpreter found. Set PYTHON_BIN to a valid interpreter." >&2
  exit 1
}

PYTHON_BIN="$(pick_python)"
echo "Using Python: $PYTHON_BIN"

run_dataset() {
  local dataset="$1"
  local gpu_id="$2"

  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" moead_exp.py "$dataset" --gpu
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" nsga_exp.py "$dataset" --gpu
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" nspso_exp.py "$dataset" --gpu
}

for idx in "${!datasets[@]}"; do
  dataset="${datasets[$idx]}"
  gpu_id="${gpu_ids[$((idx % gpu_count))]}"

  # Wait while running jobs would exceed allowed concurrency (gpu_count * MAX_JOBS_PER_GPU)
  while [ "$(jobs -pr | wc -l)" -ge "$((gpu_count * MAX_JOBS_PER_GPU))" ]; do
    wait -n
  done

  # Launch dataset job in background, capture PID and log gpu assignment
  run_dataset "$dataset" "$gpu_id" &
  pid=$!
  if command -v date >/dev/null 2>&1; then
    timestamp=$(date --iso-8601=seconds 2>/dev/null || date +"%Y-%m-%dT%H:%M:%S")
  else
    timestamp="$(date +"%Y-%m-%dT%H:%M:%S")"
  fi
  echo "$timestamp dataset=$dataset gpu=$gpu_id pid=$pid" >> "$RUN_PIDS_LOG"
done

wait
