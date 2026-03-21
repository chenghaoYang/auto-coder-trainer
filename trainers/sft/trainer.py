"""SFT Trainer — wraps HuggingFace TRL/Transformers for supervised fine-tuning."""

from __future__ import annotations

import logging
import os
from typing import Any

from trainers.base import BaseTrainer, EvalResult, TrainResult
from trainers.rl.data import setup_rollout_env
from trainers.sft.data import format_for_sft, load_trajectory_data
from trainers.utils.checkpoint import save_checkpoint
from trainers.utils.seeds import set_all_seeds

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning trainer using HuggingFace TRL/Transformers."""

    def __init__(self, config: dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self._examples: list[dict[str, Any]] = []
        self._train_examples: list[dict[str, Any]] = []
        self._eval_examples: list[dict[str, Any]] = []
        self._checkpoint_path: str | None = None
        self._eval_env: dict[str, Any] | None = None

    def prepare_data(self) -> Any:
        """Load, filter, and format trajectory data for SFT."""
        data_cfg = self.config.get("data_config", {})
        training_params = self.config.get("training_params", {})

        raw_examples = load_trajectory_data(
            sources=data_cfg.get("sources", []),
            filters=data_cfg.get("filters"),
        )
        self._examples = format_for_sft(
            raw_examples,
            chat_template=training_params.get("chat_template", "chatml"),
        )
        if not self._examples:
            raise RuntimeError("No usable SFT examples were produced from the configured dataset")

        if len(self._examples) == 1:
            self._train_examples = list(self._examples)
            self._eval_examples = []
        else:
            eval_fraction = float(training_params.get("eval_split", 0.1))
            eval_size = max(1, min(len(self._examples) // 5, int(len(self._examples) * eval_fraction)))
            if eval_size >= len(self._examples):
                eval_size = 1
            self._eval_examples = self._examples[:eval_size]
            self._train_examples = self._examples[eval_size:]

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
            "num_examples": len(self._examples),
            "train_examples": len(self._train_examples),
            "eval_examples": len(self._eval_examples),
            "eval_env": self._eval_env.get("env_type"),
            "eval_env_ready": self._eval_env.get("ready"),
        }

    def train(self) -> TrainResult:
        """Run SFT training with TRL when available, else fall back to Transformers Trainer."""
        recipe_id = self.config.get("recipe_id", "unknown")
        backend = self.config.get("backend", "trl")
        trainer_type = self.config.get("trainer_type", "sft")
        training_params = self.config.get("training_params", {})
        model_cfg = self.config.get("model_config", {})

        seed = self.config.get("eval_config", {}).get("seeds", [42])[0]
        set_all_seeds(seed)
        os.makedirs(self.output_dir, exist_ok=True)

        try:
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
                    "SFT training requires transformers, datasets, and torch. "
                    "Install them with: pip install -e '.[sft]'"
                ) from exc

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

            max_length = int(training_params.get("max_length", training_params.get("max_prompt_length", 4096)))
            train_dataset = Dataset.from_list(self._train_examples)
            eval_dataset = Dataset.from_list(self._eval_examples) if self._eval_examples else None

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
                output_dir=os.path.join(self.output_dir, "trainer_state"),
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
                logger.info("Using TRL SFTTrainer backend")
            except ImportError:
                logger.warning("trl not installed — falling back to transformers.Trainer")
                trainer_obj = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                )

            train_output = trainer_obj.train()
            metrics: dict[str, float] = {}
            if isinstance(train_output.metrics, dict):
                metrics.update(
                    {
                        key: float(value)
                        for key, value in train_output.metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
            if eval_dataset is not None:
                eval_metrics = trainer_obj.evaluate()
                metrics.update(
                    {
                        key: float(value)
                        for key, value in eval_metrics.items()
                        if isinstance(value, (int, float))
                    }
                )

            model_output = os.path.join(self.output_dir, "model")
            os.makedirs(model_output, exist_ok=True)
            trainer_obj.save_model(model_output)
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
            logger.error("SFT training failed: %s", exc, exc_info=True)
            return TrainResult(
                recipe_id=recipe_id,
                trainer_type=trainer_type,
                backend=backend,
                status="failed",
                error=str(exc),
            )

    def evaluate(self, checkpoint_path: str, benchmark: str, seed: int = 42) -> EvalResult:
        """Evaluate SFT checkpoint on a benchmark."""
        recipe_id = self.config.get("recipe_id", "unknown")
        set_all_seeds(seed)

        try:
            from evaluators.runner import run_evaluation  # type: ignore[import-untyped]

            result = run_evaluation(
                checkpoint_path=checkpoint_path,
                benchmark=benchmark,
                seed=seed,
            )
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics=result.get("metrics", {}),
                seed=seed,
                details=result.get("details", {}),
            )
        except Exception as exc:
            logger.warning("Structured evaluator unavailable for %s: %s", benchmark, exc)
            return self._basic_evaluate(checkpoint_path, benchmark, seed)

    def _apply_lora(self, model: Any, adapter: str) -> Any:
        try:
            from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("LoRA/QLoRA requires peft. Install it with: pip install peft") from exc

        tp = self.config.get("training_params", {})
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(tp.get("lora_r", 16)),
            lora_alpha=int(tp.get("lora_alpha", 32)),
            lora_dropout=float(tp.get("lora_dropout", 0.05)),
            target_modules=tp.get("lora_target_modules", ["q_proj", "v_proj"]),
        )
        if adapter == "qlora":
            logger.warning("QLoRA requested without explicit quantization config; using LoRA-compatible path")
        return get_peft_model(model, lora_config)

    def _basic_evaluate(self, checkpoint_path: str, benchmark: str, seed: int) -> EvalResult:
        """Fallback evaluation using held-out examples + the rollout harness."""
        recipe_id = self.config.get("recipe_id", "unknown")
        examples = self._eval_examples or self._examples[: min(len(self._examples), 50)]
        if not examples:
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={"error": "no evaluation examples available"},
                seed=seed,
            )
        if not self._eval_env or not self._eval_env.get("ready"):
            error = self._eval_env.get("error", "eval environment not ready") if self._eval_env else "eval environment not ready"
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={"error": error},
                seed=seed,
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError:
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={"error": "torch/transformers not available for fallback evaluation"},
                seed=seed,
            )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.eval()
        execute_fn = self._eval_env["execute_fn"]

        total_passed = 0
        total_tests = 0
        for example in examples[:50]:
            prompt = example.get("prompt", "")
            if not prompt and example.get("messages"):
                prompt = "\n".join(
                    str(message.get("content", ""))
                    for message in example.get("messages", [])
                    if str(message.get("role", "")) != "assistant"
                )
            if not prompt:
                continue
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(self.config.get("training_params", {}).get("max_new_tokens", 1024)),
                    do_sample=False,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            tests = example.get("tests", [])
            test_code = tests if isinstance(tests, str) else "\n".join(str(test) for test in tests)
            result = execute_fn(response, test_code)
            total_passed += int(result.get("tests_passed", 0))
            total_tests += int(result.get("tests_total", 0))

        pass_rate = total_passed / total_tests if total_tests else 0.0
        return EvalResult(
            recipe_id=recipe_id,
            benchmark=benchmark,
            metrics={
                "pass_rate": pass_rate,
                "tests_passed": total_passed,
                "tests_total": total_tests,
            },
            seed=seed,
            details={"num_eval_examples": min(len(examples), 50)},
        )
