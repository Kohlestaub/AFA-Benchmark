#!/bin/bash
# Run the Snakemake pipeline on minerva inside one Slurm job: one GPU, at
# most a day, everything inside the Apptainer image containers/afabench.sif.
#
# Submit from the repository root and pass Snakemake arguments, for example:
#
#   sbatch extra/workflow/profiles/minerva/run_pipeline.sbatch all \
#       --config device=cuda use_wandb=false "methods=[gdfs]" "datasets=[cube]"
#
# Jobs with the same name run one after another (--dependency=singleton), so
# submitting the same command several times queues several days of work. Each
# job skips the finished steps and continues where the previous one stopped.
# Pick a GPU type with `sbatch --gres=gpu:L40s:1 ...` or `--gres=gpu:L4:1`.
# The log goes to extra/logs/slurm/, which must exist before submitting.
#SBATCH --job-name=afabench
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --dependency=singleton
#SBATCH --output=extra/logs/slurm/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
image="${AFABENCH_IMAGE:-containers/afabench.sif}"

# Training scripts with checkpoints stop and save shortly before this time
# (see afabench/training/checkpointing.py). Older Slurm versions do not set
# it, so take the end time from squeue instead.
if [[ -z "${SLURM_JOB_END_TIME:-}" ]]; then
    end_time="$(squeue --noheader --jobs "${SLURM_JOB_ID}" --format=%e || true)"
    if [[ "${end_time}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]
    then
        SLURM_JOB_END_TIME="$(date --date="${end_time}" +%s)"
        export SLURM_JOB_END_TIME
    else
        echo "Unknown job end time ('${end_time}'): training scripts only save" \
            "on their schedule and when Slurm stops the job."
    fi
fi

snakemake=(
    apptainer exec --nv "${image}" snakemake
    --profile extra/workflow/profiles/config/all
    --workflow-profile extra/workflow/profiles/minerva
)

# A job that reached its time limit leaves Snakemake's lock behind. Only one
# of these jobs runs at a time (see above), so the lock is always stale here.
"${snakemake[@]}" --unlock
"${snakemake[@]}" "$@"
