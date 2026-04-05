#!/bin/bash
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/exp%x_%j.out
#SBATCH --error=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/exp%x_%j.err
# NOTE: --job-name and --time are set dynamically by submit_all.sh

echo "[EXP] Starting experiment $EXP_ID — $(date)"
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"

WORKDIR=/scratch/cy2668/auto-coder-trainer
OVERLAY=$WORKDIR/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

mkdir -p $WORKDIR/outputs/tinyzero_experiments/logs

# Use heredoc so conda env is inherited by all child processes (including Ray workers)
singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE /bin/bash << EXPEOF
source /ext3/env.sh
conda activate verl-agent
export EXP_ID=$EXP_ID

# Ensure Ray workers inherit this Python environment
export PATH=\$(conda info --base)/envs/verl-agent/bin:\$PATH
export PYTHONPATH=\${PYTHONPATH:-}

# Source the experiment runner
source $WORKDIR/trainers/tinyzero/experiments/run_experiments.sh
EXPEOF
RC=$?

echo "[EXP] Experiment $EXP_ID finished with exit code $RC — $(date)"
exit $RC
