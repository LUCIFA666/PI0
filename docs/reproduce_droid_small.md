# Small DROID Reproduction

This guide runs the small DROID reproduction path from raw data to a short
fine-tuning run and inference sanity check. It is intended for an Ubuntu 22.04
GPU server.

## One-time setup

From the repository root:

```bash
bash scripts/reproduce_droid_small.sh setup
```

This creates or reuses the `openpi` conda environment, installs `uv`, syncs the
project dependencies into the standard project `.venv`, and initializes git
submodules. The conda environment is used as the server-side launcher for `uv`;
the actual openpi dependencies live in `.venv`, matching the upstream README.

## Download and convert data

Install Google Cloud CLI / `gsutil` first, then run:

```bash
bash scripts/reproduce_droid_small.sh download
bash scripts/reproduce_droid_small.sh check-data
bash scripts/reproduce_droid_small.sh convert
```

The default raw data path is:

```text
~/datasets/droid_small
```

Override it with:

```bash
DROID_SMALL_DATA_DIR=/data/droid_small bash scripts/reproduce_droid_small.sh download
```

The conversion script writes a local LeRobot dataset with repo id:

```text
your_hf_username/my_droid_dataset
```

That matches the default `pi05_droid_finetune` training config.

## Short training run

Run a short sanity fine-tune:

```bash
bash scripts/reproduce_droid_small.sh train
```

Defaults:

```text
EXP_NAME=droid_small_sanity
TRAIN_STEPS=200
BATCH_SIZE=4
SAVE_INTERVAL=100
LOG_INTERVAL=10
```

For a smaller GPU:

```bash
BATCH_SIZE=2 bash scripts/reproduce_droid_small.sh train
```

For a longer run:

```bash
TRAIN_STEPS=1000 SAVE_INTERVAL=500 bash scripts/reproduce_droid_small.sh train
```

## Serve and test inference

Terminal 1:

```bash
bash scripts/reproduce_droid_small.sh serve
```

Terminal 2:

```bash
bash scripts/reproduce_droid_small.sh client
```

The server uses the latest numeric checkpoint under:

```text
checkpoints/pi05_droid_finetune/$EXP_NAME
```

To serve a specific step:

```bash
CHECKPOINT_STEP=199 bash scripts/reproduce_droid_small.sh serve
```

## Save reproduction metadata

```bash
bash scripts/reproduce_droid_small.sh export-env
```

This writes:

```text
environment-openpi.yml
requirements-openpi-freeze.txt
reproduce-git-commit.txt
reproduce-nvidia-smi.txt
```

## Full preparation sequence

This runs setup, download, data check, conversion, short training, and metadata
export. It does not start the long-running server/client pair.

```bash
bash scripts/reproduce_droid_small.sh all-prep
```
