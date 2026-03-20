"""RL (Reinforcement Learning) trainer — backed by veRL (volcengine/verl).

Provides GRPO and PPO training loops with composable reward functions
for training code-generation models on coding trajectories.
"""

from trainers.rl.trainer import RLTrainer
from trainers.rl.reward import (
    BaseReward,
    BinaryPassReward,
    WeightedPassReward,
    EntropyBonusReward,
    EntropyAwareReward,
    LengthPenaltyReward,
    CompositeReward,
    build_reward,
    REWARD_REGISTRY,
)
from trainers.rl.data import load_rl_prompts, setup_rollout_env

__all__ = [
    "RLTrainer",
    "BaseReward",
    "BinaryPassReward",
    "WeightedPassReward",
    "EntropyBonusReward",
    "EntropyAwareReward",
    "LengthPenaltyReward",
    "CompositeReward",
    "build_reward",
    "REWARD_REGISTRY",
    "load_rl_prompts",
    "setup_rollout_env",
]
