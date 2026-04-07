#!/usr/bin/env bash
# =============================================================================
# TinyZero — Submit fixed experiments only
#
# Fixed experiments: 08, 09, 10, 12, 13, 14, 18
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_EXP="$SCRIPT_DIR/experiment_fix.slurm"
LOG_DIR="/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs"
mkdir -p "$LOG_DIR"

# Time limits per experiment (hours) — tuned for fixes
declare -A TIME_LIMITS=(
    [08]=5 [09]=5 [10]=5 [12]=3 [13]=4 [14]=4 [18]=4
)

declare -A EXP_NAMES=(
    [08]="Rollout-N"
    [09]="Temperature"
    [10]="LoRA-vs-Full"
    [12]="Pass-at-K"
    [13]="1shot-RLVR"
    [14]="Distill-vs-RL"
    [18]="Capability-Preservation"
)

EXPERIMENTS=(08 09 10 12 13 14 18)

echo "============================================================"
echo " TinyZero Fixed Experiments Pipeline"
echo " Experiments: ${EXPERIMENTS[*]}"
echo "============================================================"

PREV_JOB=""
SUBMITTED=()

for EXP_ID in "${EXPERIMENTS[@]}"; do
    TIME_H=${TIME_LIMITS[$EXP_ID]:-3}
    NAME=${EXP_NAMES[$EXP_ID]:-"Exp-$EXP_ID"}

    DEP_ARG=""
    if [ -n "$PREV_JOB" ]; then
        DEP_ARG="--dependency=afterany:$PREV_JOB"
    fi

    JOB=$(sbatch --parsable \
        $DEP_ARG \
        --export=ALL,EXP_ID=$EXP_ID \
        --job-name="tz-${EXP_ID}-${NAME}" \
        --time="${TIME_H}:00:00" \
        "$SLURM_EXP")

    echo "  Exp $EXP_ID ($NAME): job $JOB  [${TIME_H}h]${DEP_ARG:+ $DEP_ARG}"
    PREV_JOB=$JOB
    SUBMITTED+=("$EXP_ID:$JOB")
done

echo ""
echo "============================================================"
echo " All ${#SUBMITTED[@]} jobs submitted!"
for ENTRY in "${SUBMITTED[@]}"; do
    IFS=':' read -r EID JID <<< "$ENTRY"
    echo "   Exp $EID (${EXP_NAMES[$EID]}): $JID"
done
echo ""
echo " Monitor: squeue -u \$USER"
echo "============================================================"

# Save manifest
MANIFEST="$LOG_DIR/job_manifest_fix_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "# TinyZero Fix Job Manifest — $(date)"
    for ENTRY in "${SUBMITTED[@]}"; do
        IFS=':' read -r EID JID <<< "$ENTRY"
        echo "exp${EID}=$JID  # ${EXP_NAMES[$EID]}"
    done
} > "$MANIFEST"
echo "Manifest saved: $MANIFEST"
