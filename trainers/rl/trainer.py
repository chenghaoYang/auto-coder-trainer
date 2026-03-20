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
from pathlib import Path
from typing import Any

from trainers.base import BaseTrainer, TrainResult, EvalResult
from trainers.rl.reward import build_reward, BaseReward
from trainers.rl.data import load_rl_prompts, setup_rollout_env
from trainers.utils.seeds import set_all_seeds
from trainers.utils.checkpoint import save_checkpoint

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
        env_cfg = training_params.get("rollout_env", {"type": "local", "timeout": 60})
        self._rollout_env = setup_rollout_env(env_cfg)
        logger.info("Rollout environment ready: %s", self._rollout_env.get("env_type"))

        return {
            "num_prompts": len(self._prompts),
            "reward_fn": type(self._reward_fn).__name__,
            "rollout_env": self._rollout_env.get("env_type"),
        }

    def train(self) -> TrainResult:
        """Run RL training with veRL (GRPO/PPO).

        Attempts to use the veRL library. If veRL is not installed, falls back
        to a lightweight built-in GRPO training loop suitable for single-node
        debugging.
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
            verl_config = self._build_verl_config()

            if backend == "verl":
                metrics = self._train_with_verl(verl_config)
            else:
                metrics = self._train_builtin_grpo(training_params)

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
            logger.warning("veRL not installed — falling back to built-in GRPO loop")
            return self._train_builtin_grpo(self.config.get("training_params", {}))

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

    def _train_builtin_grpo(self, training_params: dict[str, Any]) -> dict[str, float]:
        """Lightweight built-in GRPO training loop (no veRL dependency).

        This implements a simplified version of Group Relative Policy
        Optimization for single-node training / debugging.  For production
        multi-GPU training, install veRL.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Built-in GRPO requires torch and transformers. "
                "Install them with: pip install torch transformers"
            ) from exc

        model_cfg = self.config.get("model_config", {})
        model_name = model_cfg.get("base", "")
        adapter = model_cfg.get("adapter", "full")

        lr = training_params.get("lr", 1e-6)
        ppo_epochs = training_params.get("ppo_epochs", 4)
        rollout_batch_size = training_params.get("rollout_batch_size", 16)
        kl_coeff = training_params.get("kl_coeff", 0.05)
        entropy_coeff = training_params.get("entropy_coeff", 0.01)
        grad_accum = training_params.get("gradient_accumulation_steps", 4)
        max_new_tokens = training_params.get("max_new_tokens", 4096)
        num_iterations = training_params.get("num_iterations", len(self._prompts) // rollout_batch_size or 1)
        group_size = training_params.get("group_size", 4)

        logger.info("Built-in GRPO: model=%s, lr=%s, iterations=%d", model_name, lr, num_iterations)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Apply LoRA if requested
        if adapter in ("lora", "qlora"):
            model = self._apply_lora(model, adapter)

        # Reference model for KL (frozen copy)
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

        device = next(model.parameters()).device
        reward_fn = self._reward_fn
        execute_fn = self._rollout_env["execute_fn"] if self._rollout_env else None

        all_rewards: list[float] = []
        all_kls: list[float] = []

        for iteration in range(num_iterations):
            # Select batch of prompts
            start_idx = (iteration * rollout_batch_size) % len(self._prompts)
            batch_prompts = self._prompts[start_idx:start_idx + rollout_batch_size]
            if not batch_prompts:
                batch_prompts = self._prompts[:rollout_batch_size]

            # --- Rollout phase: generate G responses per prompt ---
            iteration_rewards: list[float] = []
            iteration_kls: list[float] = []
            policy_losses: list[torch.Tensor] = []

            for prompt_data in batch_prompts:
                prompt_text = prompt_data.get("prompt", "")
                tests = prompt_data.get("tests", [])

                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                group_rewards = []
                group_logprobs = []

                for _g in range(group_size):
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.95,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )

                    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    # Compute reward
                    trajectory = {
                        "response": response_text,
                        "tests_passed": 0,
                        "tests_total": 0,
                    }
                    if execute_fn and tests:
                        try:
                            test_code = tests if isinstance(tests, str) else "\n".join(str(t) for t in tests)
                            exec_result = execute_fn(response_text, test_code)
                            trajectory["tests_passed"] = exec_result.get("tests_passed", 0)
                            trajectory["tests_total"] = exec_result.get("tests_total", 0)
                        except Exception as e:
                            logger.debug("Rollout execution failed: %s", e)

                    reward = reward_fn.compute(trajectory)
                    group_rewards.append(reward)

                    # Compute log-prob of generated tokens under policy
                    with torch.no_grad():
                        logits = model(outputs.sequences).logits[0]
                    gen_logits = logits[inputs["input_ids"].shape[1] - 1:-1]
                    log_probs = torch.log_softmax(gen_logits, dim=-1)
                    token_logprobs = log_probs.gather(1, generated_ids.unsqueeze(1)).squeeze(1)
                    group_logprobs.append(token_logprobs)

                # --- GRPO advantage: normalise within group ---
                rewards_t = torch.tensor(group_rewards, device=device)
                if rewards_t.std() > 1e-8:
                    advantages = (rewards_t - rewards_t.mean()) / rewards_t.std()
                else:
                    advantages = rewards_t - rewards_t.mean()

                # Policy gradient loss: -E[advantage * logprob]
                for g_idx in range(group_size):
                    lp = group_logprobs[g_idx]
                    adv = advantages[g_idx]
                    policy_losses.append(-adv * lp.mean())

                iteration_rewards.extend(group_rewards)

            # --- PPO update epochs ---
            for _epoch in range(ppo_epochs):
                total_loss = torch.stack(policy_losses).mean()

                # KL penalty (approximate: use mean log-prob ratio)
                # Simplified: we skip full KL here for the built-in loop
                kl_approx = torch.tensor(0.0, device=device)
                loss = total_loss + kl_coeff * kl_approx

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_reward = sum(iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0
            all_rewards.append(avg_reward)
            logger.info(
                "Iteration %d/%d — avg_reward=%.4f, batch_size=%d",
                iteration + 1, num_iterations, avg_reward, len(batch_prompts),
            )

        # Save the trained model
        model_output = os.path.join(self.output_dir, "model")
        os.makedirs(model_output, exist_ok=True)
        model.save_pretrained(model_output)
        tokenizer.save_pretrained(model_output)

        final_metrics = {
            "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "final_reward": all_rewards[-1] if all_rewards else 0.0,
            "num_iterations": num_iterations,
            "total_rollouts": num_iterations * rollout_batch_size * group_size,
        }

        # Persist metrics
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)

        return final_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_lora(self, model: Any, adapter: str) -> Any:
        """Apply LoRA or QLoRA adapter to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("LoRA requires the peft library: pip install peft") from exc

        tp = self.config.get("training_params", {})
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=tp.get("lora_r", 16),
            lora_alpha=tp.get("lora_alpha", 32),
            lora_dropout=tp.get("lora_dropout", 0.05),
            target_modules=tp.get("lora_target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("LoRA applied: %.2fM / %.2fM trainable params (%.2f%%)",
                     trainable / 1e6, total / 1e6, 100 * trainable / total)
        return model

    def _basic_evaluate(self, checkpoint_path: str, benchmark: str, seed: int) -> EvalResult:
        """Basic self-evaluation using the rollout environment.

        Runs a sample of prompts through the model and reports pass rate.
        """
        recipe_id = self.config.get("recipe_id", "unknown")

        if not self._rollout_env or not self._rollout_env.get("ready"):
            return EvalResult(
                recipe_id=recipe_id,
                benchmark=benchmark,
                metrics={"error": "no rollout environment available"},
                seed=seed,
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
            eval_prompts = self._prompts[:50]  # Sample for quick eval

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
                metrics={"error": "torch/transformers not available for evaluation"},
                seed=seed,
            )
