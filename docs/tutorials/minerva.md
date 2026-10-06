# Running the pipeline on minerva with Apptainer

This tutorial runs the Snakemake pipeline on the minerva Slurm cluster, where
compute nodes have no internet access. The setup is:

- **Snakemake runs on the login node** (inside `tmux`) and submits one Slurm
  job per pipeline task through the
  [Slurm executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html).
- **Every job runs inside an Apptainer image** that holds Python and all
  locked dependencies. The image contains no code: Snakemake mounts the
  repository into the container, so a code change only needs a `git pull`.
- **The image is built on your own machine** and copied to the cluster,
  because building it needs root.

The image only has to be rebuilt when `uv.lock` changes.

## 1. One-time setup on minerva

```shell
git clone <your fork or private repo> afabench
cd afabench
mkdir -p containers extra/logs/slurm

# uv, then a host-side Snakemake with the Slurm plugin. Only Snakemake itself
# runs outside the container. Versions match uv.lock.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install snakemake==9.12.0 \
    --with snakemake-executor-plugin-slurm==1.8.0 --python 3.12

# Snakemake calls the `singularity` command, which Apptainer normally provides.
singularity --version   # should print "apptainer version ..."
```

If only `apptainer` exists, add a `singularity` symlink to your `PATH`:
`mkdir -p ~/bin && ln -s "$(command -v apptainer)" ~/bin/singularity`.

The Slurm plugin starts Snakemake again inside every job, so the uv tool
installation (by default under `~/.local`) must be on a filesystem that the
compute nodes can see. A shared home directory is enough.

## 2. Build the image on your own machine

On a Linux machine with Apptainer and root access (via `sudo`), from the
repository root:

```shell
containers/build.sh
```

This writes `containers/afabench-<hash>.sif`, where `<hash>` comes from
`uv.lock`, and points the symlink `containers/afabench.sif` at it. The image
is a few GB. Copy it to minerva and point the symlink there too:

```shell
rsync -P containers/afabench-<hash>.sif minerva:afabench/containers/
ssh minerva 'cd afabench \
    && ln -sfn afabench-<hash>.sif containers/afabench.sif \
    && containers/check_image.sh'
```

`containers/check_image.sh` compares the image with the checked-out
`uv.lock`. Run it after every `git pull`; it fails when you need to rebuild.

## 3. Generate the datasets on the login node

Some datasets are downloaded on first use, which needs internet access, so
generate them on the login node (still inside the container) before
submitting anything:

```shell
snakemake --profile extra/workflow/profiles/config/all all_generate_datasets \
    --software-deployment-method apptainer --cores 4 \
    --config use_wandb=false
```

## 4. Run the pipeline stages

Start a `tmux` session first so Snakemake keeps running after you log out
(`tmux new -s afabench`, detach with `Ctrl-b d`, return with
`tmux attach -t afabench`). Then run the stages from
[Reproducing full results](reproduce_full_results.md) with the minerva
workflow profiles:

- `extra/workflow/profiles/minerva`: training, classifier and evaluation jobs
  get one GPU on the `long` partition. Use with `device=cuda`.
- `extra/workflow/profiles/minerva_cpu`: no GPUs. Use with `device=cpu`.

For example, a quick end-to-end check with one method, one dataset and one
seed:

```shell
snakemake --profile extra/workflow/profiles/config/all all \
    --workflow-profile extra/workflow/profiles/minerva \
    --config device=cuda use_wandb=false smoke_test=true \
        "methods=[gdfs]" "datasets=[cube]" "dataset_instance_indices=[0]"
```

Both profiles assume the `long` partition and a 12 hour limit for training
jobs. Check the real limits with `sinfo -o "%P %l %G"` and adjust `runtime`
in the profile if Slurm rejects the jobs. To pass extra Apptainer options on
the command line, use the `=` form: `--apptainer-args="--nv"`.

Slurm logs end up in `.snakemake/slurm_logs/`, Hydra logs in `extra/logs/`.

## 5. Time limits and checkpoints

Both profiles set `retries: 3`, so a job that fails or hits its time limit is
submitted again, up to three times.

Training scripts that support checkpoints (see
`afabench/training/checkpointing.py`) save their full training state after
10, 20, 30 and 60 minutes of training and then every hour, plus once more
shortly before the Slurm time limit. The checkpoints live next to the job's
output in a `<output>.checkpoints/` directory, which Snakemake does not
delete when a job fails. A resubmitted job continues from the latest
checkpoint.

The scripts find the time limit through `SLURM_JOB_END_TIME`, which recent
Slurm versions set inside every job. Check that minerva sets it:

```shell
srun -p long -t 2 env | grep SLURM_JOB_END_TIME
```

If it prints nothing, the scripts still save on the schedule above and when
Slurm sends `SIGTERM` at the time limit; you lose at most the training since
the last scheduled checkpoint.

## 6. Collect the results

Evaluation results and plots are written below `extra/output/`, with the
final figures in `extra/output/plot_results/`. Copy them back with, for
example, `rsync -av minerva:afabench/extra/output/plot_results/ plot_results/`.

## 7. Updating the code

```shell
git pull
containers/check_image.sh   # only rebuild and copy the image if this fails
```
