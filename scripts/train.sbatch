#!/usr/bin/env bash
#
#   sbatch scripts/train.sbatch NAME [OVERRIDES...]
#
# Smoke test: sbatch scripts/train.sbatch smoke time_limit=5

# Slurm options for the MedARC cluster. Edit for yours.
#SBATCH --job-name=nanobrain
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurms/%j.out

set -euo pipefail

NAME=$1
shift
RUN_DIR="output/${NAME}"
# brainmarks-smri task group scored by the leaderboard. keep in sync with scripts/submit.py
TASK_GROUP="open"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Train until total_steps or time_limit (1 hr default).
uv run --no-sync python -m nanobrain.train \
    --overrides output_dir=output name="${NAME}" "$@"

# Evaluate the last checkpoint. Finished tasks are skipped when requeued.
uv run --no-sync python -m brainmarks_smri vit_mae_3d "${TASK_GROUP}" \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}" \
    name=eval

# Submit to labless, except smoke tests.
if [[ "${NAME}" != smoke* ]]; then
    uv run --no-sync python scripts/submit.py "${RUN_DIR}"
fi
