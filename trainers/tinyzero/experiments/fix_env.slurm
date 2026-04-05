#!/bin/bash
#SBATCH --job-name=tz-fix-env
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/fix_env_%j.out
#SBATCH --error=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/fix_env_%j.err

echo "[FIX] Fixing overlay environment — $(date)"

WORKDIR=/scratch/cy2668/auto-coder-trainer
OVERLAY=$WORKDIR/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

singularity exec --nv --fakeroot \
    --overlay $OVERLAY \
    $IMAGE bash << 'ENDSCRIPT'

source /ext3/env.sh
conda activate verl

echo "=== Before fix ==="
pip show huggingface-hub 2>/dev/null | grep Version
pip show transformers 2>/dev/null | grep Version
pip show datasets 2>/dev/null | grep Version
pip show verl 2>/dev/null | grep -E "Version|Location"

echo "=== Pinning transformers to 5.3.0 ==="
pip install "transformers==5.3.0" --force-reinstall --no-cache-dir 2>&1

echo "=== Patching dependency_versions_check.py ==="
python3 /scratch/cy2668/auto-coder-trainer/trainers/tinyzero/experiments/patch_transformers.py

echo "=== After fix ==="
pip show huggingface-hub 2>/dev/null | grep Version
pip show transformers 2>/dev/null | grep Version
pip show datasets 2>/dev/null | grep Version

echo "=== Verification ==="
python3 -c "import verl; print('verl OK:', verl.__version__)" 2>&1
python3 -c "import transformers; print('transformers OK:', transformers.__version__)" 2>&1
python3 -c "import datasets; print('datasets OK:', datasets.__version__)" 2>&1
python3 -c "import torch; print('torch OK:', torch.__version__)" 2>&1
python3 -c "import vllm; print('vllm OK:', vllm.__version__)" 2>&1

echo "[FIX] Done — $(date)"
ENDSCRIPT
