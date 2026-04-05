"""RL Trainer — wraps veRL for GRPO/PPO training on coding trajectories.

Default backend: veRL (https://github.com/volcengine/verl)
Expects a compiled recipe config with trainer_type in ("rl", "grpo")
and backend == "verl".
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from trainers.base import BaseTrainer, TrainResult, EvalResult
from trainers.rl.reward import build_reward, BaseReward
from trainers.rl.data import load_rl_prompts, setup_rollout_env
from trainers.utils.checkpoint import save_checkpoint
from trainers.utils.seeds import set_all_seeds

logger = logging.getLogger(__name__)


class RLTrainer(BaseTrainer):
    """Reinforcement Learning trainer using veRL.

    Config expectations (from ``TrainingConfig`` fields):
        model_config.base:  HuggingFace model ID
        model_config.adapter: "full" | "lora" | "qlora"
        training_params:   {lr, ppo_epochs, rollout_batch_size, kl_coeff,
                            entropy_coeff, gradient_accumulation_steps}
        training_params.reward: {type, components}  (optional, defaults binary_pass)
        data_config.sources: list of dataset specs
        eval_config:        {benchmarks, metrics, seeds}
    """

    def __init__(self, config: dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self._prompts: list[dict[str, Any]] = []
        self._reward_fn: BaseReward | None = None
        self._rollout_env: dict[str, Any] | None = None
        self._checkpoint_path: str | None = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def prepare_data(self) -> dict[str, Any]:
        """Load RL prompts, build the reward function, and set up the rollout env."""
        data_cfg = self.config.get("data_config", {})
        training_params = self.config.get("training_params", {})

        # 1. Load prompts
        logger.info("Loading RL prompts …")
        self._prompts = load_rl_prompts(
            sources=data_cfg.get("sources", []),
            filters=data_cfg.get("filters"),
            total_samples=data_cfg.get("total_samples"),
        )
        logger.info("Loaded %d prompts for RL training", len(self._prompts))

        # 2. Build reward function
        reward_cfg = training_params.get("reward", {"type": "binary_pass"})
        self._reward_fn = build_reward(reward_cfg)
        logger.info("Reward function: %s", type(self._reward_fn).__name__)

        # 3. Set up rollout environment
        env_cfg = training_params.get(
            "rollout_env",
            {
                "type": "docker",
                "timeout": 60,
                "memory_limit": "4g",
                "network": False,
            },
        )
        self._rollout_env = setup_rollout_env(env_cfg)
        logger.info("Rollout environment ready: %s", self._rollout_env.get("env_type"))

        return {
            "num_prompts": len(self._prompts),
            "reward_fn": type(self._reward_fn).__name__,
            "rollout_env": self._rollout_env.get("env_type"),
        }

    def train(self) -> TrainResult:
        """Run RL training with veRL (GRPO/PPO) or TRL GRPOTrainer fallback.

        Backend resolution:
            - ``verl``: uses veRL library; falls back to TRL GRPOTrainer if unavailable.
            - ``trl``: uses TRL GRPOTrainer directly.
            - other: tries TRL GRPOTrainer.
        """
        recipe_id = self.config.get("recipe_id", "unknown")
        trainer_type = self.config.get("trainer_type", "grpo")
        backend = self.config.get("backend", "verl")
        training_params = self.config.get("training_params", {})

        # Reproducibility
        seed = self.config.get("eval_config", {}).get("seeds", [42])[0]
        set_all_seeds(seed)

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            if backend == "verl":
                verl_config = self._build_verl_config()
                metrics = self._train_with_verl(verl_config)
            else:
                metrics = self._train_with_trl_grpo(training_params)

            # Save checkpoint
            model_output = os.path.join(self.output_dir, "model")
            os.makedirs(model_output, exist_ok=True)
            self._checkpoint_path = save_checkpoint(
                model_path=model_output,
                recipe_id=recipe_id,
                metrics=metrics,
                checkpoint_dir=os.path.join(self.output_dir, "checkpoints"),
            )
            logger.info("Checkpoint saved: %s", self._checkpoint_path)

            return TrainResult(
                recipe_id=recipe_id,
                trainer_type=trainer_type,
                backend=backend,
                status="success",
                metrics=metrics,
                checkpoint_path=self._checkpoint_path,
            )

        except Exception as exc:
            logger.error("RL training failed: %s", exc, exc_info=True)
            return TrainResult(
                recipe_id=recipe_id,
                trainer_type=trainer_type,
                backend=backend,
                status="failed",
                error=str(exc),
            )

    def evaluate(self, checkpoint_path: str, benchmark: str, seed: int = 42) -> EvalResult:
        """Evaluate an RL-trained checkpoint on a benchmark.

        Delegates to the evaluators/ module when available; otherwise runs a
        basic pass-rate evaluation using the rollout environment.
        """
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
        except ImportError:
            logger.warning(
                "evaluators.runner not available — running basic self-eval"
            )
            return self._basic_evaluate(checkpoint_path, benchmark, seed)

    # ------------------------------------------------------------------
    # veRL integration
    # ------------------------------------------------------------------

    def _build_verl_config(self) -> dict[str, Any]:
        """Convert recipe config to veRL-native configuration dict.

        Maps our recipe fields onto the config schema that ``verl`` expects,
        covering actor, critic, rollout, algorithm, and data settings.
        """
        model_cfg = self.config.get("model_config", {})
        tp = self.config.get("training_params", {})
        data_cfg = self.config.get("data_config", {})
        trainer_type = self.config.get("trainer_type", "grpo")

        model_name = model_cfg.get("base", "")
        is_grpo = trainer_type == "grpo"

        verl_cfg: dict[str, Any] = {
            "algorithm": "grpo" if is_grpo else "ppo",

            "actor": {
                "model_name": model_name,
                "adapter": model_cfg.get("adapter", "full"),
                "learning_rate": tp.get("lr", 1e-6),
                "gradient_accumulation_steps": tp.get("gradient_accumulation_steps", 4),
                "max_grad_norm": tp.get("max_grad_norm", 1.0),
                "warmup_ratio": tp.get("warmup_ratio", 0.03),
            },

            "rollout": {
                "batch_size": tp.get("rollout_batch_size", 256),
                "temperature": tp.get("temperature", 0.7),
                "top_p": tp.get("top_p", 0.95),
                "max_new_tokens": tp.get("max_new_tokens", 4096),
            },

            "algorithm_params": {
                "ppo_epochs": tp.get("ppo_epochs", 4),
                "kl_coeff": tp.get("kl_coeff", 0.05),
                "entropy_coeff": tp.get("entropy_coeff", 0.01),
                "clip_range": tp.get("clip_range", 0.2),
                "gamma": tp.get("gamma", 1.0),
                "lam": tp.get("lam", 0.95),
            },

            "data": {
                "sources": data_cfg.get("sources", []),
                "total_samples": data_cfg.get("total_samples"),
                "max_prompt_length": tp.get("max_prompt_length", 2048),
                "max_response_length": tp.get("max_response_length", 4096),
            },

            "output_dir": self.output_dir,
        }

        # Critic is not used in GRPO (only actor + reward), but PPO needs it
        if not is_grpo:
            verl_cfg["critic"] = {
                "model_name": model_name,
                "learning_rate": tp.get("critic_lr", tp.get("lr", 1e-6) * 1.5),
            }

        return verl_cfg

    def _train_with_verl(self, verl_config: dict[str, Any]) -> dict[str, float]:
        """Execute training using the veRL library."""
        try:
            from verl.trainer.main_ppo import main as verl_main  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "veRL not installed — falling back to TRL GRPOTrainer. "
                "For production multi-GPU training, install veRL or use backend=tinyzero."
            )
            return self._train_with_trl_grpo(self.config.get("training_params", {}))

        logger.info("Starting veRL training (algorithm=%s)", verl_config.get("algorithm"))

        # veRL expects its own config object; save ours as JSON and point to it
        config_path = os.path.join(self.output_dir, "verl_config.json")
        with open(config_path, "w") as f:
            json.dump(verl_config, f, indent=2)

        # Hook our reward function into veRL's reward interface
        reward_fn = self._reward_fn

        def verl_reward_function(batch: dict[str, Any]) -> list[float]:
            """Adapter: veRL batch → our BaseReward.compute interface."""
            rewards = []
            responses = batch.get("responses", [])
            test_results = batch.get("test_results", [])
            logprobs_batch = batch.get("logprobs", [None] * len(responses))
            for i, response in enumerate(responses):
                trajectory = {
                    "response": response,
                    "tests_passed": test_results[i].get("tests_passed", 0) if i < len(test_results) else 0,
                    "tests_total": test_results[i].get("tests_total", 0) if i < len(test_results) else 0,
                    "logprobs": logprobs_batch[i] if logprobs_batch[i] else [],
                }
                rewards.append(reward_fn.compute(trajectory))
            return rewards

        verl_main(
            config=config_path,
            reward_fn=verl_reward_function,
        )

        # Collect metrics from veRL output
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                return json.load(f)

        return {"status": "completed_no_metrics"}

    def _train_with_trl_grpo(self, training_params: dict[str, Any]) -> dict[str, float]:
        """GRPO training using TRL's GRPOTrainer (fallback when veRL is unavailable).

        Uses the mature TRL library's GRPOTrainer which handles generation,
        advantage computation, and policy optimisation internally.
        """
        try:
            from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer  # type: ignore[import-untyped]
            from datasets import Dataset  # type: ignore[import-untyped]
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "TRL GRPO fallback requires trl>=0.14, transformers, and datasets. "
                "Install them with: pip install -e '.[rl]'"
            ) from exc

        model_cfg = self.config.get("model_config", {})
        model_name = model_cfg.get("base", "")
        adapter = model_cfg.get("adapter", "full")

        lr = float(training_params.get("lr", 1e-6))
        rollout_batch_size = int(training_params.get("rollout_batch_size", 16))
        kl_coeff = float(training_params.get("kl_coeff", 0.05))
        max_new_tokens = int(training_params.get("max_new_tokens", 4096))
        group_size = int(training_params.get("group_size", 4))
        num_iterations = int(
            training_params.get("num_iterations", len(self._prompts) // rollout_batch_size or 1)
        )
        grad_accum = int(training_params.get("gradient_accumulation_steps", 4))
        ppo_epochs = int(training_params.get("ppo_epochs", 4))
        temperature = float(training_params.get("temperature", 0.7))

        logger.info("TRL GRPOTrainer: model=%s, lr=%s, group_size=%d", model_name, lr, group_size)

        # Build prompt dataset for TRL GRPOTrainer (requires a "prompt" column)
        prompt_records = []
        prompts_by_text: dict[str, dict[str, Any]] = {}
        for p in self._prompts:
            text = p.get("prompt", "")
            prompt_records.append({"prompt": text})
            prompts_by_text[text] = p
        train_dataset = Dataset.from_list(prompt_records)

        # Build reward function adapter for TRL.
        # TRL GRPOTrainer calls reward_func(completions=..., prompts=..., **kwargs)
        reward_fn = self._reward_fn
        execute_fn = self._rollout_env["execute_fn"] if self._rollout_env else None

        def trl_reward_func(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:
            """Adapter: TRL GRPOTrainer reward interface → our BaseReward."""
            rewards: list[float] = []
            for completion, prompt in zip(completions, prompts):
                prompt_data = prompts_by_text.get(prompt, {})
                tests = prompt_data.get("tests", [])
                trajectory: dict[str, Any] = {
                    "response": completion,
                    "tests_passed": 0,
                    "tests_total": 0,
                }
                if execute_fn and tests:
                    try:
                        test_code = tests if isinstance(tests, str) else "\n".join(str(t) for t in tests)
                        exec_result = execute_fn(completion, test_code)
                        trajectory["tests_passed"] = exec_result.get("tests_passed", 0)
                        trajectory["tests_total"] = exec_result.get("tests_total", 0)
                    except Exception as e:
                        logger.debug("Rollout execution failed: %s", e)
                rewards.append(reward_fn.compute(trajectory))
            return rewards

        # Configure TRL GRPOTrainer
        grpo_config = GRPOConfig(
            output_dir=os.path.join(self.output_dir, "trainer_state"),
            num_train_epochs=num_iterations,
            per_device_train_batch_size=rollout_batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            warmup_ratio=float(training_params.get("warmup_ratio", 0.03)),
            max_grad_norm=float(training_params.get("max_grad_norm", 1.0)),
            logging_steps=int(training_params.get("logging_steps", 1)),
            save_strategy="epoch",
            report_to=[],
            # GRPO-specific parameters
            max_completion_length=max_new_tokens,
            num_generations=group_size,
            temperature=temperature,
            beta=kl_coeff,
            num_iterations=ppo_epochs,
            remove_unused_columns=False,
        )

        # Apply LoRA config if requested (TRL GRPOTrainer supports peft_config)
        peft_config = None
        if adapter in ("lora", "qlora"):
            try:
                from peft import LoraConfig  # type: ignore[import-untyped]

                peft_config = LoraConfig(
                    r=int(training_params.get("lora_r", 16)),
                    lora_alpha=int(training_params.get("lora_alpha", 32)),
                    lora_dropout=float(training_params.get("lora_dropout", 0.05)),
                    target_modules=training_params.get("lora_target_modules", "all-linear"),
                    task_type="CAUSAL_LM",
                )
            except ImportError:
                logger.warning("peft not installed — training without LoRA adapter")

        # Tokenizer for processing_class
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        trainer_kwargs: dict[str, Any] = {
            "model": model_name,
            "reward_funcs": trl_reward_func,
            "args": grpo_config,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        trainer_obj = TRLGRPOTrainer(**trainer_kwargs)
        train_output = trainer_obj.train()

        # Save model
        model_output = os.path.join(self.output_dir, "model")
        os.makedirs(model_output, exist_ok=True)
        trainer_obj.save_model(model_output)
        tokenizer.save_pretrained(model_output)

        # Collect metrics
        metrics: dict[str, float] = {}
        if isinstance(getattr(train_output, "metrics", None), dict):
            metrics.update(
                {k: float(v) for k, v in train_output.metrics.items() if isinstance(v, (int, float))}
            )

        # Persist metrics
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _basic_evaluate(self, checkpoint_path: str, benchmark: str, seed: int) -> EvalResult:
        """Basic self-evaluation using the rollout environment.

        Runs a sample of prompts through the model and reports pass rate.
        """
        recipe_id = self.config.get("recipe_id", "unknown")

        if not self._rollout_env or not self._rollout_env.get("ready"):
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={},
                seed=seed,
                details={"error": "no rollout environment available"},
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()

            execute_fn = self._rollout_env["execute_fn"]
            eval_sample_size = self.config.get("training_params", {}).get("eval_sample_size", 50)
            eval_prompts = self._prompts[:eval_sample_size]

            total_passed = 0
            total_tests = 0

            for prompt_data in eval_prompts:
                prompt_text = prompt_data.get("prompt", "")
                tests = prompt_data.get("tests", [])

                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                if tests:
                    test_code = tests if isinstance(tests, str) else "\n".join(str(t) for t in tests)
                    try:
                        result = execute_fn(response, test_code)
                        total_passed += result.get("tests_passed", 0)
                        total_tests += result.get("tests_total", 0)
                    except Exception:
                        pass

            pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={"pass_rate": pass_rate, "tests_passed": total_passed, "tests_total": total_tests},
                seed=seed,
                details={"num_eval_prompts": len(eval_prompts)},
            )

        except ImportError:
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={},
                seed=seed,
                details={"error": "torch/transformers not available for evaluation"},
            )
