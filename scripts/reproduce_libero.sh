#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-openpi}"
LIBERO_DATA_DIR="${LIBERO_DATA_DIR:-$HOME/datasets/libero_rlds}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-2}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

TRAIN_CONFIG="${TRAIN_CONFIG:-pi0_libero_local_repro_small}"
EXP_NAME="${EXP_NAME:-libero_local_sanity}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-100000}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_libero.sh <command>

Commands:
  check-docker       Check Docker, docker compose, and GPU visibility.
  official-smoke     Run official pi05_libero eval with 2 trials/task by default.
  official-suite     Run official pi05_libero eval with NUM_TRIALS, default task suite libero_spatial.
  download-data      Download raw LIBERO RLDS data from Hugging Face.
  convert-data       Convert raw LIBERO RLDS data to local LeRobot format.
  norm-stats         Compute normalization stats for the local LeRobot LIBERO dataset.
  train-local        Run local low-memory LIBERO LoRA sanity training.
  local-smoke        Evaluate the latest local checkpoint with 2 trials/task by default.
  local-suite        Evaluate the latest local checkpoint with NUM_TRIALS.
  summarize          Summarize official/local JSONL metrics if they exist.

Environment overrides:
  CONDA_ENV=openpi
  LIBERO_DATA_DIR=$HOME/datasets/libero_rlds
  OPENPI_DATA_HOME=$HOME/.cache/openpi
  TASK_SUITE=libero_spatial|libero_object|libero_goal|libero_10|libero_90
  NUM_TRIALS=2
  MUJOCO_GL=egl|osmesa
  PYOPENGL_PLATFORM=egl|osmesa
  TRAIN_CONFIG=pi0_libero_local_repro_small
  EXP_NAME=libero_local_sanity
  TRAIN_STEPS=1000
  BATCH_SIZE=8
  NORM_MAX_FRAMES=100000
  CHECKPOINT_STEP=<step number for local eval>
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

activate_conda_env() {
  require_cmd conda
  set +u
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  set -u
  unset UV_PROJECT_ENVIRONMENT
  export OPENPI_DATA_HOME
}

check_docker() {
  cd "$ROOT_DIR"
  docker --version
  docker compose version
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
}

compose_eval() {
  local server_args="$1"
  local run_prefix="$2"
  local trials="$3"

  cd "$ROOT_DIR"
  mkdir -p "data/libero/videos_${run_prefix}" "data/libero/metrics"

  export SERVER_ARGS="$server_args"
  export CLIENT_ARGS="--args.task-suite-name ${TASK_SUITE} --args.num-trials-per-task ${trials} --args.video-out-path data/libero/videos_${run_prefix} --args.metrics-out-path data/libero/metrics/${run_prefix}_${TASK_SUITE}.jsonl"
  export MUJOCO_GL
  export PYOPENGL_PLATFORM
  export OPENPI_DATA_HOME

  docker compose -f examples/libero/compose.yml up --build --abort-on-container-exit
}

official_smoke() {
  compose_eval "--env LIBERO" "official" "$NUM_TRIALS"
}

official_suite() {
  compose_eval "--env LIBERO" "official" "$NUM_TRIALS"
}

download_data() {
  activate_conda_env
  python -m pip install -U huggingface_hub
  mkdir -p "$LIBERO_DATA_DIR"
  huggingface-cli download openvla/modified_libero_rlds \
    --repo-type dataset \
    --local-dir "$LIBERO_DATA_DIR"
}

convert_data() {
  activate_conda_env
  cd "$ROOT_DIR"
  uv sync --group rlds
  uv run --group rlds examples/libero/convert_libero_data_to_lerobot.py --data_dir "$LIBERO_DATA_DIR"
}

compute_norm_stats() {
  activate_conda_env
  cd "$ROOT_DIR"
  uv run scripts/compute_norm_stats.py \
    --config-name "$TRAIN_CONFIG" \
    --max-frames "$NORM_MAX_FRAMES"
}

train_local() {
  activate_conda_env
  cd "$ROOT_DIR"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py "$TRAIN_CONFIG" \
      --exp-name="$EXP_NAME" \
      --overwrite \
      --num-train-steps="$TRAIN_STEPS" \
      --batch-size="$BATCH_SIZE" \
      --save-interval="$SAVE_INTERVAL" \
      --log-interval="$LOG_INTERVAL" \
      --no-wandb-enabled
}

latest_checkpoint_step() {
  local ckpt_root="$ROOT_DIR/checkpoints/$TRAIN_CONFIG/$EXP_NAME"
  test -d "$ckpt_root" || die "Checkpoint directory not found: $ckpt_root"
  find "$ckpt_root" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

local_eval() {
  local trials="$1"
  local step="${CHECKPOINT_STEP:-$(latest_checkpoint_step)}"
  test -n "$step" || die "Could not find a checkpoint step for TRAIN_CONFIG=$TRAIN_CONFIG EXP_NAME=$EXP_NAME"
  compose_eval "policy:checkpoint --policy.config ${TRAIN_CONFIG} --policy.dir checkpoints/${TRAIN_CONFIG}/${EXP_NAME}/${step}" "local" "$trials"
}

local_smoke() {
  local_eval "$NUM_TRIALS"
}

local_suite() {
  local_eval "$NUM_TRIALS"
}

summarize_metrics() {
  cd "$ROOT_DIR"
  python - <<'PY'
import json
from pathlib import Path

metrics_dir = Path("data/libero/metrics")
if not metrics_dir.exists():
    print("No metrics directory found.")
    raise SystemExit(0)

for path in sorted(metrics_dir.glob("*.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        continue
    successes = sum(bool(row.get("success")) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    print(f"{path}: episodes={len(rows)} successes={successes} success_rate={successes/len(rows):.3f} errors={errors}")
PY
}

cmd="${1:-}"
case "$cmd" in
  check-docker) check_docker ;;
  official-smoke) official_smoke ;;
  official-suite) official_suite ;;
  download-data) download_data ;;
  convert-data) convert_data ;;
  norm-stats) compute_norm_stats ;;
  train-local) train_local ;;
  local-smoke) local_smoke ;;
  local-suite) local_suite ;;
  summarize) summarize_metrics ;;
  -h|--help|help|"") usage ;;
  *) usage; die "Unknown command: $cmd" ;;
esac
