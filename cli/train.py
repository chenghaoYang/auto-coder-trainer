"""Train command — execute a training experiment from a recipe.

Compiles a Recipe IR JSON into a training configuration, runs the
experiment, evaluates results, and submits to the experiment judge.
"""

import argparse
import json
import uuid
from pathlib import Path


def run_train(args: argparse.Namespace) -> None:
    """Execute the training pipeline.

    Pipeline:
        1. Load and validate recipe JSON
        2. Compile recipe to training config
        3. Select trainer backend (TRL for SFT, veRL for RL)
        4. Run training
        5. Evaluate on specified benchmarks
        6. Submit to experiment judge
        7. Store results in result DB
    """
    recipe_path = Path(args.recipe)
    output_dir = getattr(args, "output_dir", "outputs/")
    dry_run = getattr(args, "dry_run", False)

    # ------------------------------------------------------------------
    # 1. Load recipe
    # ------------------------------------------------------------------
    print(f"[train] Loading recipe: {recipe_path}")
    try:
        with open(recipe_path) as f:
            recipe = json.load(f)
    except FileNotFoundError:
        print(f"[train] Error: recipe file not found: {recipe_path}")
        return
    except json.JSONDecodeError as exc:
        print(f"[train] Error: invalid JSON in {recipe_path}: {exc}")
        return

    recipe_id = recipe.get("id", "unknown")
    print(f"[train] Recipe ID: {recipe_id}")

    # ------------------------------------------------------------------
    # 2. Validate recipe
    # ------------------------------------------------------------------
    try:
        from recipes.compiler import load_schema, validate_recipe

        schema = load_schema()
        errors = validate_recipe(recipe, schema)
        if errors:
            print(f"[train] Validation errors ({len(errors)}):")
            for err in errors:
                print(f"[train]   - {err}")
            print("[train] Aborting — fix the recipe and retry.")
            return
        print("[train] Recipe validation passed.")
    except Exception as exc:
        print(f"[train] Warning: could not validate recipe ({exc}). Proceeding anyway.")

    # ------------------------------------------------------------------
    # 3. Compile recipe to TrainingConfig
    # ------------------------------------------------------------------
    try:
        from recipes.compiler import compile_recipe

        config = compile_recipe(recipe)
        print(f"[train] Compiled config: {config.backend}/{config.trainer_type}")
    except Exception as exc:
        print(f"[train] Error compiling recipe: {exc}")
        return

    if dry_run:
        print("[train] Dry-run mode — skipping training.")
        print(f"[train] Config summary:")
        print(f"[train]   Trainer : {config.trainer_type} ({config.backend})")
        print(f"[train]   Model   : {config.model_config}")
        print(f"[train]   Data    : {config.data_config}")
        print(f"[train]   Eval    : {config.eval_config}")
        return

    # ------------------------------------------------------------------
    # 4. Select and instantiate trainer
    # ------------------------------------------------------------------
    trainer = None
    try:
        if config.trainer_type == "sft":
            try:
                from trainers.sft import SFTTrainer
                trainer = SFTTrainer(config.__dict__, output_dir)
                print("[train] Using SFT trainer (TRL backend).")
            except ImportError:
                print("[train] SFT trainer module not yet available.")
        elif config.trainer_type in ("rl", "grpo"):
            try:
                from trainers.rl import RLTrainer
                trainer = RLTrainer(config.__dict__, output_dir)
                print("[train] Using RL trainer (veRL backend).")
            except ImportError:
                print("[train] RL trainer module not yet available.")
        else:
            print(f"[train] Unknown trainer type: {config.trainer_type}")
    except Exception as exc:
        print(f"[train] Error initializing trainer: {exc}")

    if trainer is None:
        print("[train] No trainer available — cannot proceed with training.")
        print("[train] Implement trainers/sft/trainer.py or trainers/rl/trainer.py to enable training.")
        return

    # ------------------------------------------------------------------
    # 5. Run training + evaluation
    # ------------------------------------------------------------------
    print("[train] Starting training run ...")
    try:
        train_result, eval_results = trainer.run()
        print(f"[train] Training finished — status: {train_result.status}")
        if train_result.metrics:
            print(f"[train] Train metrics: {train_result.metrics}")
        for er in eval_results:
            print(f"[train] Eval [{er.benchmark}] seed={er.seed}: {er.metrics}")
    except Exception as exc:
        print(f"[train] Training failed: {exc}")
        train_result = None
        eval_results = []

    # ------------------------------------------------------------------
    # 6. Submit to experiment judge
    # ------------------------------------------------------------------
    verdict = None
    if train_result and train_result.status == "success":
        try:
            from judge.judge import ExperimentJudge

            judge = ExperimentJudge()
            results_dict = {
                "train": train_result.__dict__ if hasattr(train_result, "__dict__") else {},
                "eval": [er.__dict__ for er in eval_results] if eval_results else [],
            }
            verdict = judge.judge(recipe_id, results_dict)
            print(f"[train] Judge verdict: {verdict.verdict.value} — {verdict.reasoning}")
        except NotImplementedError:
            print("[train] Judge not fully implemented — skipping verdict.")
        except Exception as exc:
            print(f"[train] Judge error: {exc}")

    # ------------------------------------------------------------------
    # 7. Store results in DB
    # ------------------------------------------------------------------
    if train_result:
        try:
            from results.db import ResultDB
            from judge.dedup import compute_config_hash

            db = ResultDB()
            db.connect()

            experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
            config_hash = compute_config_hash(recipe)

            db.insert_experiment({
                "id": experiment_id,
                "recipe_id": recipe_id,
                "config_hash": config_hash,
                "status": train_result.status,
                "trainer_type": config.trainer_type,
                "backend": config.backend,
                "model_base": config.model_config.get("base", ""),
                "metrics_json": train_result.metrics or {},
                "checkpoint_path": train_result.checkpoint_path,
                "error": train_result.error,
            })

            if verdict:
                db.insert_verdict({
                    "experiment_id": experiment_id,
                    "verdict": verdict.verdict.value,
                    "reasoning": verdict.reasoning,
                    "checks_json": verdict.checks,
                    "suggestions_json": verdict.suggestions,
                })

            db.close()
            print(f"[train] Results stored — experiment_id: {experiment_id}")
        except Exception as exc:
            print(f"[train] Warning: could not store results in DB: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n[train] === Summary ===")
    print(f"[train] Recipe     : {recipe_id}")
    print(f"[train] Trainer    : {config.trainer_type} / {config.backend}")
    status = train_result.status if train_result else "not_run"
    print(f"[train] Status     : {status}")
    if verdict:
        print(f"[train] Verdict    : {verdict.verdict.value}")
    print("[train] Done.")
