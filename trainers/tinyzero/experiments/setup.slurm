#!/bin/bash
#SBATCH --job-name=tz-setup
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/setup_%j.out
#SBATCH --error=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/setup_%j.err

echo "[SETUP] Starting data + model preparation — $(date)"
echo "Node: $(hostname)  Job: $SLURM_JOB_ID"

WORKDIR=/scratch/cy2668/auto-coder-trainer
OVERLAY=$WORKDIR/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

mkdir -p $WORKDIR/outputs/tinyzero_experiments/logs

singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE bash << 'SETUPEOF'

source /ext3/env.sh
conda activate verl-agent

export HF_HOME=/scratch/cy2668/hf_cache
export TRANSFORMERS_CACHE=/scratch/cy2668/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/cy2668/hf_cache/hub

echo "=== Environment check ==="
python3 --version
python3 -c "import verl; print('verl', verl.__version__)"
python3 -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import vllm; print('vllm', vllm.__version__)"
nvidia-smi -L

echo "=== Preparing datasets ==="
cd /scratch/cy2668/auto-coder-trainer
python3 trainers/tinyzero/experiments/prepare_data.py \
    --output_dir /scratch/cy2668/auto-coder-trainer/data/tinyzero

echo "=== Verifying datasets ==="
ls -lh /scratch/cy2668/auto-coder-trainer/data/tinyzero/*.parquet

echo "=== Pre-downloading FULL model weights ==="
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os

models = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
]

cache_dir = "/scratch/cy2668/hf_cache/hub"

for m in models:
    print(f"\n=== Downloading: {m} ===")
    try:
        path = snapshot_download(
            m,
            cache_dir=cache_dir,
            resume_download=True,
        )
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(path)
            for f in fns
        ) / 1e9
        print(f"  OK: {path} ({size:.1f} GB)")
    except Exception as e:
        print(f"  ERROR: {e}")
PYEOF

echo "=== Verifying models ==="
du -sh /scratch/cy2668/hf_cache/hub/models--Qwen--Qwen2.5-* 2>/dev/null

echo "[SETUP] Done — $(date)"
SETUPEOF
