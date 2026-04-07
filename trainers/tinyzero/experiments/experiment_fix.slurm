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
# NOTE: --job-name and --time are set dynamically by submit_fixes.sh

echo "[EXP] Starting experiment $EXP_ID (fix) — $(date)"
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"

WORKDIR=/scratch/cy2668/auto-coder-trainer
OVERLAY=$WORKDIR/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

mkdir -p $WORKDIR/outputs/tinyzero_experiments/logs

# Pre-step for exp14: patch sft_dataset.py before entering container
if [ "$EXP_ID" = "14" ]; then
    echo "[EXP] Patching sft_dataset.py for exp14..."
    cp $WORKDIR/trainers/tinyzero/experiments/sft_dataset.py \
       $WORKDIR/verl_standard/verl/utils/dataset/sft_dataset.py
    echo "[EXP] sft_dataset.py patched."
fi

# Run inside singularity (same as original experiment.slurm)
# Using :ro overlay — works on most nodes (ga001-013), fails on some (ga040+)
singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE /bin/bash << EXPEOF
source /ext3/env.sh
conda activate verl-agent
export EXP_ID=$EXP_ID

# Ensure Ray workers inherit this Python environment
export PATH=\$(conda info --base)/envs/verl-agent/bin:\$PATH
export PYTHONPATH=\${PYTHONPATH:-}

# Preprocess SFT data for exp14 (inside container with correct python)
if [ "$EXP_ID" = "14" ]; then
    echo "[EXP] Preprocessing SFT data for exp14..."
    python3 $WORKDIR/trainers/tinyzero/experiments/preprocess_sft_data.py
    echo "[EXP] SFT data preprocessed."
fi

# Source the fixed experiment runner
source $WORKDIR/trainers/tinyzero/experiments/run_experiments_fix.sh
EXPEOF
RC=$?

echo "[EXP] Experiment $EXP_ID finished with exit code $RC — $(date)"
exit $RC
