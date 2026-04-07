#!/usr/bin/env bash
# =============================================================================
# TinyZero — Submit all 21 experiments serially on 1×A100
#
# Usage:
#   bash submit_all.sh              # Submit all 21 experiments
#   bash submit_all.sh 01 05 12     # Submit specific experiments only
#   bash submit_all.sh --from 08    # Submit experiments 08-21
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SETUP="$SCRIPT_DIR/setup.slurm"
SLURM_EXP="$SCRIPT_DIR/experiment.slurm"
LOG_DIR="/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs"
mkdir -p "$LOG_DIR"

# Time limits per experiment (hours) — tuned for multi-sub-run experiments
declare -A TIME_LIMITS=(
    [01]=3 [02]=2 [03]=4 [04]=3 [05]=3
    [06]=3 [07]=2 [08]=3 [09]=3 [10]=3
    [11]=2 [12]=2 [13]=4 [14]=3 [15]=2
    [16]=2 [17]=2 [18]=2 [19]=3 [20]=2
    [21]=3
)

# Experiment descriptions
declare -A EXP_NAMES=(
    [01]="PPO-vs-GRPO"
    [02]="FP8-Quantization"
    [03]="Data-Scaling-Law"
    [04]="Cross-Task-Transfer"
    [05]="Reward-Functions"
    [06]="Model-Scale"
    [07]="KL-Sensitivity"
    [08]="Rollout-N"
    [09]="Temperature"
    [10]="LoRA-vs-Full"
    [11]="Comprehensive-Scaling"
    [12]="Pass-at-K"
    [13]="1shot-RLVR"
    [14]="Distill-vs-RL"
    [15]="Self-Reflection"
    [16]="Multimodal"
    [17]="Multi-Turn"
    [18]="Capability-Preservation"
    [19]="Cost-Analysis"
    [20]="Think-Preservation"
    [21]="Reward-Thinking-Style"
)

# Parse arguments
EXPERIMENTS=()
if [ $# -eq 0 ]; then
    # All experiments
    for i in $(seq -w 1 21); do
        EXPERIMENTS+=("$i")
    done
elif [ "$1" = "--from" ]; then
    FROM=${2:-01}
    for i in $(seq -w ${FROM#0} 21); do
        EXPERIMENTS+=("$(printf '%02d' $i)")
    done
else
    EXPERIMENTS=("$@")
fi

echo "============================================================"
echo " TinyZero Experiment Pipeline"
echo " Experiments: ${EXPERIMENTS[*]}"
echo " Account: torch_pr_74_tandon_priority"
echo " Partition: a100_tandon (1×A100)"
echo "============================================================"
echo ""

# Step 1: Submit setup job (data preparation)
echo "[1/2] Submitting setup job (data preparation)..."
SETUP_JOB=$(sbatch --parsable "$SLURM_SETUP")
echo "  Setup job: $SETUP_JOB"

# Step 2: Submit experiments with dependency chain
echo "[2/2] Submitting ${#EXPERIMENTS[@]} experiments..."
PREV_JOB=$SETUP_JOB
SUBMITTED=()

for EXP_ID in "${EXPERIMENTS[@]}"; do
    # Zero-pad to 2 digits
    EXP_ID=$(printf '%02d' ${EXP_ID#0})

    TIME_H=${TIME_LIMITS[$EXP_ID]:-24}
    NAME=${EXP_NAMES[$EXP_ID]:-"Exp-$EXP_ID"}

    JOB=$(sbatch --parsable \
        --dependency=afterany:$PREV_JOB \
        --export=ALL,EXP_ID=$EXP_ID \
        --job-name="tz-${EXP_ID}-${NAME}" \
        --time="${TIME_H}:00:00" \
        "$SLURM_EXP")

    echo "  Exp $EXP_ID ($NAME): job $JOB  [after $PREV_JOB, ${TIME_H}h]"
    PREV_JOB=$JOB
    SUBMITTED+=("$EXP_ID:$JOB")
done

echo ""
echo "============================================================"
echo " All jobs submitted! Serial chain:"
echo "   Setup: $SETUP_JOB"
for ENTRY in "${SUBMITTED[@]}"; do
    IFS=':' read -r EID JID <<< "$ENTRY"
    echo "   Exp $EID (${EXP_NAMES[$EID]}): $JID"
done
echo ""
echo " Monitor: squeue -u \$USER"
echo " Logs:    $LOG_DIR/"
echo " Results: /scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/"
echo "============================================================"

# Save job manifest
MANIFEST="$LOG_DIR/job_manifest_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "# TinyZero Job Manifest — $(date)"
    echo "setup=$SETUP_JOB"
    for ENTRY in "${SUBMITTED[@]}"; do
        IFS=':' read -r EID JID <<< "$ENTRY"
        echo "exp${EID}=$JID  # ${EXP_NAMES[$EID]}"
    done
} > "$MANIFEST"
echo "Manifest saved: $MANIFEST"
