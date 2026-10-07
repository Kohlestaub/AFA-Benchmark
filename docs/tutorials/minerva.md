# Running the pipeline on minerva with Apptainer

This tutorial runs the Snakemake pipeline on the minerva Slurm cluster. On
minerva a job runs for at most a day and should use one GPU (an L40s or an
L4), compute nodes have no internet access, and nothing can be installed
system-wide. The setup is:

- **One Slurm job runs the whole pipeline.** It holds one GPU for up to a
  day. Snakemake runs inside that job and runs the pipeline steps on the
  job's node, one at a time.
- **Everything runs inside an Apptainer image** that holds Python, all locked
  dependencies and Snakemake, so nothing has to be installed on minerva. The
  image contains no code: the repository is mounted into the container, so a
  code change only needs a `git pull`.
- **The image is built on your own machine** and copied to the cluster,
  because building it needs root.
- **Long runs continue in the next job.** Submitting the same job again skips
  the finished steps, and training scripts with checkpoints continue where
  they stopped.

The image only has to be rebuilt when `uv.lock` or `containers/afabench.def`
changes.

## 1. One-time setup on minerva

```shell
git clone <your fork or private repo> afabench
cd afabench
mkdir -p containers extra/logs/slurm
apptainer --version
```

The clone has to be on a filesystem that the compute nodes can see, for
example your home directory.

## 2. Build the image on your own machine

On a Linux machine with Apptainer and root access (via `sudo`), from the
repository root:

```shell
containers/build.sh
```

This writes `containers/afabench-<hash>.sif`, where `<hash>` comes from
`uv.lock` and `containers/afabench.def`, and points the symlink
`containers/afabench.sif` at it. The image is a few GB. Copy it to minerva
and point the symlink there too:

```shell
rsync -P containers/afabench-<hash>.sif minerva:afabench/containers/
ssh minerva 'cd afabench \
    && ln -sfn afabench-<hash>.sif containers/afabench.sif \
    && containers/check_image.sh'
```

`containers/check_image.sh` compares the image with the checked-out
`uv.lock` and `containers/afabench.def`. Run it after every `git pull`; it
fails when you need to rebuild.

## 3. Generate the datasets on the login node

Some datasets (for example the UCI datasets and MNIST) are downloaded on
first use, which the compute nodes cannot do. Generate the datasets on the
login node, inside the image, before submitting anything:

```shell
apptainer exec containers/afabench.sif snakemake \
    --profile extra/workflow/profiles/config/all all_generate_datasets \
    --cores 4 --config use_wandb=false "datasets=[cube]"
```

Leave out `"datasets=[...]"` to generate every dataset. If this cannot run on
the login node, run the same command in your clone on your own machine and
copy the result with
`rsync -a extra/output/datasets/ minerva:afabench/extra/output/datasets/`.

## 4. Run the pipeline

Submit the job script from the repository root. Everything after the script
name goes to Snakemake. For example, a quick end-to-end check with one
method, one dataset and one seed:

```shell
sbatch extra/workflow/profiles/minerva/run_pipeline.sbatch all \
    --config device=cuda use_wandb=false smoke_test=true \
        "methods=[gdfs]" "datasets=[cube]" "dataset_instance_indices=[0]"
```

The job asks for one GPU of any type, 4 cores, 32 GB of memory and one day
on the `long` partition (the `#SBATCH` lines at the top of the script).
Override them before the script name, for example
`sbatch --gres=gpu:L40s:1 extra/workflow/profiles/minerva/run_pipeline.sbatch ...`
for an L40s, `--gres=gpu:L4:1` for an L4, or `--mem=64G` if a step runs out
of memory.

Follow the job with `squeue -u $USER` and
`tail -f extra/logs/slurm/afabench-<job id>.out`. The logs of the single
steps end up in `extra/logs/`.

Steps run one at a time. If they leave the GPU mostly idle, add `--cores 2`
(up to 4, the job's CPU count) to run several at once on the same GPU.

Snakemake does not rerun steps when the code or the configuration changes
(`rerun-triggers: mtime` in `extra/workflow/profiles/minerva/config.yaml`, so
that datasets generated elsewhere are reused). Rerun steps on purpose with
`--forcerun <rule>`.

For the same reason, a smoke run's results would be reused by a real run:
`smoke_test=true` writes to the same files. Before a real run, delete them
and keep the datasets:

```shell
find extra/output -mindepth 1 -maxdepth 1 ! -name datasets -exec rm -rf {} +
```

## 5. Time limits and checkpoints

A job stops after a day at the latest. Submit the same command again to
continue. Jobs with the same name run one after another
(`--dependency=singleton`), so you can also submit the command several times
at once to queue several days of work, and only one GPU is used at a time.
Each job skips the finished steps.

Training scripts that support checkpoints (see
`afabench/training/checkpointing.py`) save their full training state after
10, 20, 30 and 60 minutes of training and then every hour, and stop with a
last checkpoint 10 minutes before the job ends. The next job continues from
the latest checkpoint. Other steps that the time limit cuts off start over.
Checkpoints live next to the step's output in a `<output>.checkpoints/`
directory.

The job script takes the end time from `SLURM_JOB_END_TIME`, or from
`squeue` when Slurm does not set it.

## 6. Collect the results

Evaluation results and plots are written below `extra/output/`, with the
final figures in `extra/output/plot_results/`. Copy them back with, for
example, `rsync -av minerva:afabench/extra/output/plot_results/ plot_results/`.

## 7. Updating the code

```shell
git pull
containers/check_image.sh   # only rebuild and copy the image if this fails
```
