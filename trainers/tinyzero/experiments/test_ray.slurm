#!/bin/bash
#SBATCH --job-name=tz-ray-test
#SBATCH --account=torch_pr_74_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/ray_test_%j.out
#SBATCH --error=/scratch/cy2668/auto-coder-trainer/outputs/tinyzero_experiments/logs/ray_test_%j.err

OVERLAY=/scratch/cy2668/auto-coder-trainer/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

singularity exec --nv \
    --overlay $OVERLAY:ro \
    $IMAGE bash << 'EOF'

source /ext3/env.sh
conda activate verl-agent
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/cy2668/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/cy2668/hf_cache/hub
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "=== System ==="
echo "SHM:" && df -h /dev/shm
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

echo "=== PyTorch GPU ==="
python3 -u -c "
import torch
print(f'cuda available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'device name: {torch.cuda.get_device_name(0)}')
    print(f'mem: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo "=== Ray init test ==="
python3 -u -c "
import os, ray
os.environ['RAY_DEDUP_LOGS'] = '0'
print('Calling ray.init(num_gpus=1)...')
ray.init(num_gpus=1, num_cpus=8)
print(f'Resources: {ray.cluster_resources()}')
print(f'GPU count: {ray.cluster_resources().get(\"GPU\", 0)}')
ray.shutdown()
print('Ray OK')
"

echo "=== Minimal verl test ==="
python3 -u -c "
import verl
print(f'verl {verl.__version__} imported OK')
from verl.trainer.main_ppo import _default_config_dict
print('main_ppo config loaded OK')
"

echo "=== DONE ==="
EOF
