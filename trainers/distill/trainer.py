"""Distillation trainer for coding-agent trajectory and process distillation."""

from __future__ import annotations

import logging
import os
from typing import Any

from trainers.base import TrainResult
from trainers.distill.data import load_distillation_data
from trainers.rl.data import setup_rollout_env
from trainers.sft.trainer import SFTTrainer
from trainers.utils.checkpoint import save_checkpoint
from trainers.utils.seeds import set_all_seeds

logger = logging.getLogger(__name__)


class DistillTrainer(SFTTrainer):
    """Two-stage distillation trainer.

    Stage 1: positive teacher-trajectory SFT
    Stage 2: optional pairwise refinement on chosen/rejected traces
    """

    def __init__(self, config: dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self._pair_examples: list[dict[str, Any]] = []
        self._pair_train_examples: list[dict[str, Any]] = []
        self._pair_eval_examples: list[dict[str, Any]] = []

    def prepare_data(self) -> Any:
        """Load trajectory distillation data and split positive/pair sets."""
        data_cfg = self.config.get("data_config", {})
        training_params = self.config.get("training_params", {})
        distill_cfg = self.config.get("distill_config", {})

        payload = load_distillation_data(
            sources=data_cfg.get("sources", []),
            filters=data_cfg.get("filters"),
            distill_config=distill_cfg,
        )
        self._examples = payload.get("positive_examples", [])
        self._pair_examples = payload.get("pair_examples", [])

        if not self._examples:
            raise RuntimeError("No usable positive distillation examples were produced from the configured dataset")

        self._train_examples, self._eval_examples = _split_examples(
            self._examples,
            float(training_params.get("eval_split", 0.1)),
        )
        self._pair_train_examples, self._pair_eval_examples = _split_examples(
            self._pair_examples,
            float(training_params.get("eval_split", 0.1)),
        )

        env_cfg = training_params.get(
            "eval_env",
            {
                "type": "docker",
                "timeout": 60,
                "memory_limit": "4g",
                "network": False,
            },
        )
        self._eval_env = setup_rollout_env(env_cfg)
        return {
            "positive_examples": len(self._examples),
            "positive_train_examples": len(self._train_examples),
            "positive_eval_examples": len(self._eval_examples),
            "pair_examples": len(self._pair_examples),
            "pair_train_examples": len(self._pair_train_examples),
            "pair_eval_examples": len(self._pair_eval_examples),
            "eval_env": self._eval_env.get("env_type"),
            "eval_env_ready": self._eval_env.get("ready"),
        }

    def train(self) -> TrainResult:
        """Run positive trajectory SFT, then optional pairwise refinement."""
        recipe_id = self.config.get("recipe_id", "unknown")
        backend = self.config.get("backend", "trl")
        trainer_type = self.config.get("trainer_type", "distill")
        training_params = self.config.get("training_params", {})
        model_cfg = self.config.get("model_config", {})
        distill_cfg = self.config.get("distill_config", {})

        stages = distill_cfg.get("stages") or ["positive_sft"]
        if isinstance(stages, str):
            stages = [stages]

        seed = self.config.get("eval_config", {}).get("seeds", [42])[0]
        set_all_seeds(seed)
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            stack = _load_hf_stack()
            torch = stack["torch"]
            Dataset = stack["Dataset"]
            AutoModelForCausalLM = stack["AutoModelForCausalLM"]
            AutoTokenizer = stack["AutoTokenizer"]

            tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("base", ""), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else None
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg.get("base", ""),
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            adapter = model_cfg.get("adapter", "full")
            if adapter in {"lora", "qlora"}:
                model = self._apply_lora(model, adapter)

            metrics: dict[str, float] = {}
            if "positive_sft" in stages or not stages:
                metrics.update(
                    self._run_positive_stage(
                        model=model,
                        tokenizer=tokenizer,
                        dataset_cls=Dataset,
                        stack=stack,
                    )
                )

            if "pairwise_refine" in stages:
                if self._pair_train_examples:
                    metrics.update(
                        self._run_pairwise_stage(
                            model=model,
                            tokenizer=tokenizer,
                        )
                    )
                else:
                    logger.warning(
                        "pairwise_refine requested but no chosen/rejected distillation pairs were found"
                    )

            model_output = os.path.join(self.output_dir, "model")
            os.makedirs(model_output, exist_ok=True)
            model.save_pretrained(model_output)
            tokenizer.save_pretrained(model_output)
            self._checkpoint_path = save_checkpoint(
                model_path=model_output,
                recipe_id=recipe_id,
                metrics=metrics,
                checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
            )
            return TrainResult(
                recipe_id=recipe_id,
                trainer_type=trainer_type,
                backend=backend,
                status="success",
                metrics=metrics,
                checkpoint_path=self._checkpoint_path,
            )
        except Exception as exc:
            logger.error("Distillation training failed: %s", exc, exc_info=True)
            return TrainResult(
                recipe_id=recipe_id,
                trainer_type=trainer_type,
                backend=backend,
                status="failed",
                error=str(exc),
            )

    def _run_positive_stage(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dataset_cls: Any,
        stack: dict[str, Any],
    ) -> dict[str, float]:
        if not self._train_examples:
            raise RuntimeError("Positive distillation stage requires at least one training example")

        TrainingArguments = stack["TrainingArguments"]
        DataCollatorForLanguageModeling = stack["DataCollatorForLanguageModeling"]
        Trainer = stack["Trainer"]
        training_params = self.config.get("training_params", {})

        max_length = int(training_params.get("max_length", training_params.get("max_prompt_length", 4096)))
        train_dataset = dataset_cls.from_list(self._train_examples)
        eval_dataset = dataset_cls.from_list(self._eval_examples) if self._eval_examples else None

        def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
            )
            encoded["labels"] = list(encoded["input_ids"])
            return encoded

        train_dataset = train_dataset.map(_tokenize)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(_tokenize)

        args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "trainer_state", "distill_positive"),
            num_train_epochs=float(training_params.get("epochs", 1)),
            per_device_train_batch_size=int(training_params.get("batch_size", 1)),
            per_device_eval_batch_size=int(training_params.get("eval_batch_size", training_params.get("batch_size", 1))),
            gradient_accumulation_steps=int(training_params.get("gradient_accumulation_steps", 1)),
            learning_rate=float(training_params.get("lr", 2e-5)),
            warmup_ratio=float(training_params.get("warmup_ratio", 0.03)),
            logging_steps=int(training_params.get("logging_steps", 10)),
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            report_to=[],
            remove_unused_columns=False,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer_obj: Any
        try:
            from trl import SFTTrainer as HFSFTTrainer  # type: ignore[import-untyped]

            trainer_obj = HFSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                data_collator=data_collator,
                max_seq_length=max_length,
            )
            logger.info("Using TRL SFTTrainer for positive distillation stage")
        except ImportError:
            logger.warning("trl not installed — falling back to transformers.Trainer for positive stage")
            trainer_obj = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

        output = trainer_obj.train()
        metrics: dict[str, float] = {}
        if isinstance(output.metrics, dict):
            for key, value in output.metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"positive/{key}"] = float(value)
        if eval_dataset is not None:
            eval_metrics = trainer_obj.evaluate()
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"positive/{key}"] = float(value)
        return metrics

    def _run_pairwise_stage(self, *, model: Any, tokenizer: Any) -> dict[str, float]:
        training_params = self.config.get("training_params", {})
        distill_cfg = self.config.get("distill_config", {})
        refine_algorithm = distill_cfg.get("refine_algorithm", "dpo")
        if refine_algorithm == "none":
            logger.info("Pairwise refinement explicitly disabled")
            return {}
        if refine_algorithm == "redi":
            raise RuntimeError(
                "REDI refinement is intentionally routed through backend='redi' so this project can use the upstream implementation directly."
            )

        stack = _load_hf_stack()
        Dataset = stack["Dataset"]
        TrainingArguments = stack["TrainingArguments"]
        DPOTrainer = stack.get("DPOTrainer")
        DPOConfig = stack.get("DPOConfig")
        if DPOTrainer is None:
            raise RuntimeError(
                "pairwise_refine requires trl.DPOTrainer. Install a recent TRL release or switch the recipe backend to 'redi'."
            )

        beta = float(distill_cfg.get("pairwise_beta", training_params.get("pairwise_beta", 0.1)))
        max_length = int(training_params.get("max_length", training_params.get("max_prompt_length", 4096)))
        max_prompt_length = int(training_params.get("max_prompt_length", max_length // 2))
        pairwise_batch_size = int(training_params.get("pairwise_batch_size", training_params.get("batch_size", 1)))
        pairwise_epochs = int(training_params.get("pairwise_epochs", 1))
        pairwise_lr = float(training_params.get("pairwise_lr", training_params.get("lr", 2e-6)))
        grad_accum = int(
            training_params.get(
                "pairwise_gradient_accumulation_steps",
                training_params.get("gradient_accumulation_steps", 1),
            )
        )

        train_dataset = Dataset.from_list(self._pair_train_examples)
        eval_dataset = Dataset.from_list(self._pair_eval_examples) if self._pair_eval_examples else None

        args_cls = DPOConfig or TrainingArguments
        args = args_cls(
            output_dir=os.path.join(self.output_dir, "trainer_state", "distill_pairwise"),
            num_train_epochs=float(pairwise_epochs),
            per_device_train_batch_size=pairwise_batch_size,
            per_device_eval_batch_size=int(training_params.get("pairwise_eval_batch_size", pairwise_batch_size)),
            gradient_accumulation_steps=grad_accum,
            learning_rate=pairwise_lr,
            logging_steps=int(training_params.get("logging_steps", 10)),
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            report_to=[],
            remove_unused_columns=False,
        )

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "beta": beta,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
        }
        try:
            trainer_obj = DPOTrainer(ref_model=None, **trainer_kwargs)
        except TypeError:
            trainer_obj = DPOTrainer(**trainer_kwargs)

        output = trainer_obj.train()
        metrics: dict[str, float] = {}
        if isinstance(getattr(output, "metrics", None), dict):
            for key, value in output.metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"pairwise/{key}"] = float(value)
        if eval_dataset is not None:
            eval_metrics = trainer_obj.evaluate()
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"pairwise/{key}"] = float(value)

        return metrics


def _split_examples(examples: list[dict[str, Any]], eval_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        return [], []
    if len(examples) == 1:
        return list(examples), []

    eval_size = max(1, min(len(examples) // 5, int(len(examples) * eval_fraction)))
    if eval_size >= len(examples):
        eval_size = 1
    eval_examples = examples[:eval_size]
    train_examples = examples[eval_size:]
    return train_examples, eval_examples


def _load_hf_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset  # type: ignore[import-untyped]
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Distillation training requires transformers, datasets, and torch. "
            "Install them with: pip install -e '.[sft]'"
        ) from exc

    DPOTrainer = None
    DPOConfig = None
    try:
        from trl import DPOConfig as TRLDPOConfig  # type: ignore[import-untyped]
        from trl import DPOTrainer as TRLDPOTrainer  # type: ignore[import-untyped]

        DPOTrainer = TRLDPOTrainer
        DPOConfig = TRLDPOConfig
    except ImportError:
        try:
            from trl import DPOTrainer as TRLDPOTrainer  # type: ignore[import-untyped]

            DPOTrainer = TRLDPOTrainer
        except ImportError:
            DPOTrainer = None
            DPOConfig = None

    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "DPOTrainer": DPOTrainer,
        "DPOConfig": DPOConfig,
    }
