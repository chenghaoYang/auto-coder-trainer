"""SFT Trainer skeleton — wraps HuggingFace TRL SFTTrainer.

Expects a compiled recipe config with trainer.type == "sft" and trainer.backend == "trl".
"""

from typing import Any

from trainers.base import BaseTrainer, TrainResult, EvalResult


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer using HuggingFace TRL.

    Config expectations:
        model.base: HuggingFace model ID
        model.adapter: "full" | "lora" | "qlora"
        trainer.params: {lr, epochs, batch_size, gradient_accumulation_steps, warmup_ratio}
        dataset.sources: list of dataset specs
    """

    def prepare_data(self) -> Any:
        """Load trajectory data and format for SFT.

        TODO: Implement with datasets library.
        - Load from HF hub or local path
        - Apply filters (quality_score, length, etc.)
        - Format as chat/instruction pairs
        """
        raise NotImplementedError("SFT data preparation not yet implemented")

    def train(self) -> TrainResult:
        """Run SFT training with TRL SFTTrainer.

        TODO: Implement with trl.SFTTrainer.
        - Initialize model + tokenizer
        - Configure LoRA/QLoRA if specified
        - Set up training arguments from recipe params
        - Train and save checkpoint
        """
        raise NotImplementedError("SFT training not yet implemented")

    def evaluate(self, checkpoint_path: str, benchmark: str, seed: int = 42) -> EvalResult:
        """Evaluate SFT checkpoint on a benchmark.

        TODO: Delegate to evaluators/ module.
        """
        raise NotImplementedError("SFT evaluation not yet implemented")
