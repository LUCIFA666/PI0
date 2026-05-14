#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-openpi}"
DATA_DIR="${DROID_SMALL_DATA_DIR:-$HOME/datasets/droid_small}"
EXP_NAME="${EXP_NAME:-droid_small_sanity}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_droid_small.sh <command>

Commands:
  setup        Create/activate the conda env and install openpi dependencies with uv.
  download     Download the small DROID demo subset and language annotations.
  check-data   Verify that the small DROID data has the expected files.
  convert      Convert the small raw DROID subset to a local LeRobot dataset.
  train        Run a short sanity fine-tune on pi05_droid_finetune.
  serve        Serve the latest checkpoint from the sanity run.
  client       Run the simple DROID client against a running policy server.
  export-env   Save conda/pip/git/GPU reproduction metadata.
  all-prep     Run setup, download, check-data, convert, train, export-env.

Environment overrides:
  CONDA_ENV=openpi
  DROID_SMALL_DATA_DIR=$HOME/datasets/droid_small
  EXP_NAME=droid_small_sanity
  TRAIN_STEPS=200
  BATCH_SIZE=4
  SAVE_INTERVAL=100
  LOG_INTERVAL=10
  OPENPI_DATA_HOME=$HOME/.cache/openpi
  CHECKPOINT_STEP=<step number for serve>
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
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"

  # Make uv install into the active conda env instead of creating .venv.
  export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
  export OPENPI_DATA_HOME
}

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"
}

setup_env() {
  require_cmd conda
  if ! env_exists; then
    conda create -n "$CONDA_ENV" python=3.11 -y
  fi

  activate_conda_env
  python -m pip install -U pip
  python -m pip install -U uv

  cd "$ROOT_DIR"
  git submodule update --init --recursive
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

  echo
  echo "Environment check:"
  nvidia-smi || true
  uv pip show torch
  uv pip show jax
  uv pip show transformers
}

download_data() {
  require_cmd gsutil
  mkdir -p "$DATA_DIR"
  gsutil -m cp -r \
    gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 \
    "$DATA_DIR/"
  gsutil -m cp -r \
    gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json \
    "$DATA_DIR/"
}

check_data() {
  echo "Checking data in: $DATA_DIR"
  test -d "$DATA_DIR" || die "Data directory does not exist: $DATA_DIR"
  test -f "$DATA_DIR/aggregated-annotations-030724.json" || die "Missing aggregated annotations JSON"

  local trajectory_count
  local recordings_count
  trajectory_count="$(find "$DATA_DIR" -name "trajectory.h5" | wc -l)"
  recordings_count="$(find "$DATA_DIR" -type d -path "*/recordings/MP4" | wc -l)"

  echo "trajectory.h5 files: $trajectory_count"
  echo "recordings/MP4 dirs: $recordings_count"
  find "$DATA_DIR" -name "trajectory.h5" | head
  find "$DATA_DIR" -type d -path "*/recordings/MP4" | head

  test "$trajectory_count" -gt 0 || die "No trajectory.h5 files found"
  test "$recordings_count" -gt 0 || die "No recordings/MP4 directories found"
}

convert_data() {
  activate_conda_env
  cd "$ROOT_DIR"
  uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir "$DATA_DIR"
}

train_sanity() {
  activate_conda_env
  cd "$ROOT_DIR"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    uv run scripts/train.py pi05_droid_finetune \
      --exp-name="$EXP_NAME" \
      --overwrite \
      --num-train-steps="$TRAIN_STEPS" \
      --batch-size="$BATCH_SIZE" \
      --save-interval="$SAVE_INTERVAL" \
      --log-interval="$LOG_INTERVAL" \
      --no-wandb-enabled
}

latest_checkpoint_step() {
  local ckpt_root="$ROOT_DIR/checkpoints/pi05_droid_finetune/$EXP_NAME"
  test -d "$ckpt_root" || die "Checkpoint directory not found: $ckpt_root"
  find "$ckpt_root" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1
}

serve_policy() {
  activate_conda_env
  cd "$ROOT_DIR"

  local step="${CHECKPOINT_STEP:-$(latest_checkpoint_step)}"
  test -n "$step" || die "Could not find a checkpoint step for EXP_NAME=$EXP_NAME"

  uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_finetune \
    --policy.dir="checkpoints/pi05_droid_finetune/$EXP_NAME/$step"
}

run_client() {
  activate_conda_env
  cd "$ROOT_DIR"
  uv run examples/simple_client/main.py --env DROID
}

export_env() {
  activate_conda_env
  cd "$ROOT_DIR"
  conda env export --no-builds > environment-openpi.yml
  python -m pip freeze > requirements-openpi-freeze.txt
  git rev-parse HEAD > reproduce-git-commit.txt
  nvidia-smi > reproduce-nvidia-smi.txt || true
  echo "Wrote environment-openpi.yml, requirements-openpi-freeze.txt, reproduce-git-commit.txt, reproduce-nvidia-smi.txt"
}

cmd="${1:-}"
case "$cmd" in
  setup) setup_env ;;
  download) download_data ;;
  check-data) check_data ;;
  convert) convert_data ;;
  train) train_sanity ;;
  serve) serve_policy ;;
  client) run_client ;;
  export-env) export_env ;;
  all-prep)
    setup_env
    download_data
    check_data
    convert_data
    train_sanity
    export_env
    ;;
  -h|--help|help|"") usage ;;
  *) usage; die "Unknown command: $cmd" ;;
esac
