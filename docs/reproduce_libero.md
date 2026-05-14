# LIBERO Simulation Reproduction

This guide runs LIBERO simulation evaluation on a headless GPU server. It has two
stages:

1. Evaluate the official `pi05_libero` checkpoint to see simulated task success.
2. Convert raw LIBERO data, run a small local LoRA training job, and evaluate the
   resulting checkpoint with the same simulator.

The recommended runtime is Docker with EGL headless rendering.

## Setup

From the repository root:

```bash
git submodule update --init --recursive
bash scripts/reproduce_libero.sh check-docker
```

If the Docker GPU check fails, fix Docker NVIDIA runtime before continuing.

## Official Checkpoint Evaluation

Smoke test on `libero_spatial` with 2 trials per task:

```bash
bash scripts/reproduce_libero.sh official-smoke
```

Run fuller suites:

```bash
TASK_SUITE=libero_spatial NUM_TRIALS=50 bash scripts/reproduce_libero.sh official-suite
TASK_SUITE=libero_object NUM_TRIALS=50 bash scripts/reproduce_libero.sh official-suite
TASK_SUITE=libero_goal NUM_TRIALS=50 bash scripts/reproduce_libero.sh official-suite
TASK_SUITE=libero_10 NUM_TRIALS=50 bash scripts/reproduce_libero.sh official-suite
```

Outputs:

```text
data/libero/metrics/official_<suite>.jsonl
data/libero/videos_official/*.mp4
```

If EGL fails on the headless server, retry with OSMesa:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash scripts/reproduce_libero.sh official-smoke
```

## Local Data and Training

Download raw LIBERO RLDS data:

```bash
bash scripts/reproduce_libero.sh download-data
```

Convert it to local LeRobot format:

```bash
bash scripts/reproduce_libero.sh convert-data
```

The converted dataset repo id is:

```text
your_hf_username/libero
```

Run local low-memory LoRA sanity training:

```bash
bash scripts/reproduce_libero.sh train-local
```

Defaults:

```text
TRAIN_CONFIG=pi0_libero_local_repro_small
EXP_NAME=libero_local_sanity
TRAIN_STEPS=1000
BATCH_SIZE=8
SAVE_INTERVAL=500
```

If training runs out of memory:

```bash
BATCH_SIZE=4 bash scripts/reproduce_libero.sh train-local
```

## Local Checkpoint Evaluation

Smoke test the latest local checkpoint:

```bash
bash scripts/reproduce_libero.sh local-smoke
```

Run a fuller suite:

```bash
TASK_SUITE=libero_spatial NUM_TRIALS=50 bash scripts/reproduce_libero.sh local-suite
```

Serve a specific checkpoint step:

```bash
CHECKPOINT_STEP=999 bash scripts/reproduce_libero.sh local-smoke
```

Outputs:

```text
data/libero/metrics/local_<suite>.jsonl
data/libero/videos_local/*.mp4
```

Summarize available metrics:

```bash
bash scripts/reproduce_libero.sh summarize
```
