"""Tests for the pipeline orchestrator and blog-style report generator."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from results.db import ResultDB
from results.report_generator import ReportGenerator
from trainers.base import EvalResult, TrainResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _baseline_recipe() -> dict:
    return {
        "id": "recipe-blog-test-001",
        "name": "Blog Report Test SFT",
        "version": "1.0",
        "source_papers": ["2410.01021"],
        "model": {
            "base": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "size": "7B",
            "adapter": "lora",
        },
        "dataset": {
            "sources": [
                {
                    "name": "swe-bench-trajectories",
                    "path": "bigcode/swe-bench-trajectories",
                    "mix_weight": 1.0,
                }
            ],
            "filters": [
                {"type": "quality_score", "params": {"min_score": 0.7}},
            ],
            "total_samples": 10000,
        },
        "trainer": {
            "type": "sft",
            "backend": "trl",
            "params": {
                "lr": 2e-5,
                "epochs": 3,
                "batch_size": 4,
            },
        },
        "eval": {
            "benchmarks": ["swe-bench-lite"],
            "metrics": ["resolve_rate", "pass@1"],
            "seeds": [42, 123, 456],
        },
        "ablation": [
            {"name": "lr_sweep", "variable": "trainer.params.lr", "values": [1e-5, 2e-5, 5e-5]},
        ],
        "budget": {
            "max_gpu_hours": 24,
            "gpu_type": "A100-80GB",
            "max_cost_usd": 50,
        },
    }


def _populate_db(db: ResultDB, recipe: dict) -> str:
    """Insert a complete experiment with eval runs, ablations, and verdicts. Returns experiment_id."""
    exp_id = "exp-blog-test-001"
    db.insert_experiment(
        {
            "id": exp_id,
            "recipe_id": recipe["id"],
            "config_hash": "abc123def456",
            "status": "success",
            "trainer_type": "sft",
            "backend": "trl",
            "model_base": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "metrics_json": {"resolve_rate": 0.82, "pass@1": 0.55, "train/loss": 0.21},
            "train_metrics_json": {"train/loss": 0.21, "train/lr": 2e-5},
            "recipe_json": recipe,
            "budget_json": recipe.get("budget", {}),
            "checkpoint_path": "/tmp/fake-checkpoint",
            "error": None,
        }
    )
    # Eval runs (3 seeds)
    for seed in (42, 123, 456):
        db.insert_eval_run(
            {
                "experiment_id": exp_id,
                "benchmark": "swe-bench-lite",
                "seed": seed,
                "metrics_json": {"resolve_rate": 0.80 + seed * 0.0001, "pass@1": 0.55},
                "details_json": {"source": "test"},
            }
        )
    # Ablation
    db.insert_ablation(
        {
            "experiment_id": exp_id,
            "variable": "trainer.params.lr",
            "value": "1e-5",
            "metrics_json": {"resolve_rate": 0.75, "pass@1": 0.50},
        }
    )
    db.insert_ablation(
        {
            "experiment_id": exp_id,
            "variable": "trainer.params.lr",
            "value": "5e-5",
            "metrics_json": {"resolve_rate": 0.78, "pass@1": 0.52},
        }
    )
    # Verdict
    db.insert_verdict(
        {
            "experiment_id": exp_id,
            "verdict": "accept",
            "reasoning": "Baseline alignment passed. All seeds present. Ablation coverage met.",
            "checks_json": {
                "baseline_aligned": True,
                "seeds_complete": True,
                "ablation_covered": True,
                "not_duplicate": True,
            },
            "suggestions_json": [
                "Consider adding more ablation points for batch_size.",
                "Try entropy-aware reward in a follow-up RL experiment.",
            ],
        }
    )
    return exp_id


# ---------------------------------------------------------------------------
# Blog report tests
# ---------------------------------------------------------------------------


class TestBlogReport:
    def test_blog_report_has_all_sections(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        output_file = tmp_path / "report.md"
        content = generator.generate_blog_report([exp_id], output_file)
        db.close()

        assert output_file.exists()
        # Check all major sections are present
        assert "# Supervised Fine-Tuning on Qwen2.5-Coder-7B-Instruct" in content
        assert "TL;DR" in content
        assert "## Introduction" in content
        assert "## Experimental Setup" in content
        assert "## Experiments" in content
        assert "## Ablation Studies" in content
        assert "## Cost & Efficiency Analysis" in content
        assert "## Practical Recommendations" in content
        assert "## Reproducibility" in content
        assert "## Conclusion" in content

    def test_blog_report_contains_metrics(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "resolve_rate" in content
        assert "pass@1" in content
        assert "0.80" in content

    def test_blog_report_contains_ablation_table(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "trainer.params.lr" in content
        assert "1e-5" in content
        assert "5e-5" in content
        assert "Best setting" in content

    def test_blog_report_contains_verdict(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "PASS" in content
        assert "Baseline alignment passed" in content

    def test_blog_report_contains_recommendations(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "entropy-aware reward" in content
        assert "batch_size" in content

    def test_blog_report_contains_reproducibility(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "act train" in content
        assert "abc123def456" in content
        assert "42" in content

    def test_blog_report_contains_cost_analysis(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "A100-80GB" in content
        assert "GPU hours" in content
        assert "per GPU-hour" in content

    def test_blog_report_multiple_experiments(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        recipe = _baseline_recipe()
        exp_id_1 = _populate_db(db, recipe)

        # Second experiment
        recipe2 = dict(recipe, id="recipe-blog-test-002", name="Blog Report Test RL")
        exp_id_2 = "exp-blog-test-002"
        db.insert_experiment(
            {
                "id": exp_id_2,
                "recipe_id": recipe2["id"],
                "config_hash": "xyz789",
                "status": "success",
                "trainer_type": "grpo",
                "backend": "verl",
                "model_base": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "metrics_json": {"resolve_rate": 0.85},
                "train_metrics_json": {},
                "recipe_json": recipe2,
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report(
            [exp_id_1, exp_id_2], tmp_path / "report.md"
        )
        db.close()

        assert "Experiment 1" in content
        assert "Experiment 2" in content
        assert "Insights from 2 Experiments" in content

    def test_blog_report_with_no_eval_runs(self, tmp_path: Path) -> None:
        db = ResultDB(tmp_path / "results.db")
        db.connect()
        exp_id = "exp-no-eval"
        db.insert_experiment(
            {
                "id": exp_id,
                "recipe_id": "recipe-empty",
                "config_hash": "hash",
                "status": "prepared",
                "trainer_type": "sft",
                "backend": "trl",
                "model_base": "test-model",
                "metrics_json": {"train/loss": 0.5},
                "train_metrics_json": {},
                "recipe_json": {"id": "recipe-empty", "budget": {}},
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )

        generator = ReportGenerator(result_db=db)
        content = generator.generate_blog_report([exp_id], tmp_path / "report.md")
        db.close()

        assert "Training metrics" in content or "train/loss" in content
        assert (tmp_path / "report.md").exists()


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_pipeline_with_existing_recipe_dry_run(self, tmp_path: Path, monkeypatch) -> None:
        """Pipeline with --recipe skips collect/compose and runs train in dry-run mode."""
        recipe = _baseline_recipe()
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2))
        db_path = tmp_path / "results.db"
        monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

        from cli.pipeline import run_pipeline

        run_pipeline(
            Namespace(
                query=None,
                atoms=None,
                recipe=str(recipe_path),
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                output_dir=str(tmp_path / "outputs"),
                report_dir=str(tmp_path / "reports"),
                report_format="blog",
                max_iterations=1,
                dry_run=True,
            )
        )

        # Should have created an experiment in DB
        db = ResultDB(db_path)
        db.connect()
        try:
            experiments = db.find_by_recipe(recipe["id"])
            assert len(experiments) >= 1
        finally:
            db.close()

    def test_pipeline_with_recipe_runs_train_and_generates_blog_report(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Pipeline runs train, gets judge verdict, and produces blog-style report."""
        import trainers.sft

        recipe = _baseline_recipe()
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2))
        db_path = tmp_path / "results.db"
        checkpoint_dir = tmp_path / "fake-checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_text("stub")
        monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

        class FakeSFTTrainer:
            def __init__(self, config, output_dir):
                self.config = config
                self.output_dir = output_dir

            def run(self):
                return (
                    TrainResult(
                        recipe_id=self.config["recipe_id"],
                        trainer_type="sft",
                        backend="trl",
                        status="success",
                        metrics={"train/loss": 0.25},
                        checkpoint_path=str(checkpoint_dir),
                    ),
                    [
                        EvalResult(
                            recipe_id=self.config["recipe_id"],
                            benchmark="swe-bench-lite",
                            seed=seed,
                            metrics={"resolve_rate": 0.80, "pass@1": 0.55},
                            details={"source": "fake"},
                        )
                        for seed in (42, 123, 456)
                    ],
                )

        monkeypatch.setattr(trainers.sft, "SFTTrainer", FakeSFTTrainer)
        # Also patch the registry so get_trainer_class() returns the fake
        import trainers.registry
        monkeypatch.setitem(trainers.registry._REGISTRY, ("sft", None), FakeSFTTrainer)
        monkeypatch.setitem(trainers.registry._REGISTRY, ("sft", "trl"), FakeSFTTrainer)

        from cli.pipeline import run_pipeline

        report_dir = tmp_path / "reports"
        run_pipeline(
            Namespace(
                query=None,
                atoms=None,
                recipe=str(recipe_path),
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                output_dir=str(tmp_path / "outputs"),
                report_dir=str(report_dir),
                report_format="blog",
                max_iterations=1,
                dry_run=False,
            )
        )

        captured = capsys.readouterr().out
        assert "[pipeline] Done." in captured

        # Blog report should exist (generated at max-iteration or terminal verdict)
        report_file = report_dir / "report.md"
        if report_file.exists():
            content = report_file.read_text()
            assert "Introduction" in content
            assert "Experimental Setup" in content

    def test_pipeline_collect_compose_train_flow(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Full pipeline from query: collect → compose → train (dry-run)."""
        import cli.collect as collect_mod

        monkeypatch.setattr(
            collect_mod,
            "_search_arxiv_papers",
            lambda query, max_papers: [
                {
                    "id": "2501.99999",
                    "title": "Test Paper for Pipeline",
                    "abstract": "SFT training with LoRA on SWE-Bench.",
                }
            ],
        )
        monkeypatch.setattr(
            collect_mod,
            "_search_github_repos",
            lambda query, max_repos: [],
        )

        db_path = tmp_path / "results.db"
        monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

        from cli.pipeline import run_pipeline

        run_pipeline(
            Namespace(
                query="sft lora swe-bench",
                atoms=None,
                recipe=None,
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                output_dir=str(tmp_path / "outputs"),
                report_dir=str(tmp_path / "reports"),
                report_format="blog",
                max_iterations=1,
                dry_run=True,
            )
        )

        captured = capsys.readouterr().out
        assert "Phase 1: COLLECT" in captured
        assert "Phase 2: COMPOSE" in captured
        assert "Phase 3: TRAIN" in captured

    def test_pipeline_with_swe_lego_recipe_waits_for_external_completion(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ) -> None:
        recipe = {
            "id": "recipe-swe-lego-pipeline-test",
            "name": "SWE-Lego pipeline handoff test",
            "model": {"base": "Qwen/Qwen3.5-9B", "adapter": "full"},
            "dataset": {
                "sources": [
                    {"name": "swe-lego-real", "path": "SWE-Lego/SWE-Lego-Real-Data"},
                ]
            },
            "trainer": {
                "type": "sft",
                "backend": "swe_lego",
                "params": {"template": "qwen3", "deepspeed": "z2_offload"},
            },
            "eval": {"benchmarks": ["swe-bench-verified"], "seeds": [42]},
        }
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2))
        db_path = tmp_path / "results.db"
        monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

        from cli.pipeline import run_pipeline

        report_dir = tmp_path / "reports"
        run_pipeline(
            Namespace(
                query=None,
                atoms=None,
                recipe=str(recipe_path),
                model="Qwen/Qwen3.5-9B",
                output_dir=str(tmp_path / "outputs"),
                report_dir=str(report_dir),
                report_format="blog",
                max_iterations=1,
                dry_run=True,
            )
        )

        captured = capsys.readouterr().out
        assert "training is running asynchronously" in captured
        assert not (report_dir / "report.md").exists()

    def test_pipeline_does_not_retrain_base_recipe_after_ablation_dispatch(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        recipe = _baseline_recipe()
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2))

        from cli import pipeline as pipeline_mod

        calls = {"train": 0, "rerun": 0, "report": 0}
        state = {"ablation_dispatched": False}

        def _fake_run_train(recipe_path_arg, output_dir, *, dry_run=False):
            calls["train"] += 1
            return "exp-base"

        def _fake_get_latest_experiment(recipe_id, *, db=None):
            if state["ablation_dispatched"]:
                return {"id": "exp-ablation", "recipe_id": recipe_id}
            return {"id": "exp-base", "recipe_id": recipe_id}

        def _fake_get_experiment_verdict(experiment_id, *, db=None):
            if experiment_id == "exp-base":
                return {
                    "experiment_id": "exp-base",
                    "verdict": "needs_ablation",
                    "reasoning": "Missing lr sweep",
                    "suggestions_json": ["Run missing ablation experiments before accepting"],
                    "research_suggestions": [],
                }
            if experiment_id == "exp-ablation":
                return {
                    "experiment_id": "exp-ablation",
                    "verdict": "accept",
                    "reasoning": "Ablation completed",
                    "suggestions_json": [],
                    "research_suggestions": [],
                }
            return None

        def _fake_run_rerun(recipe_id, *, dry_run=False):
            calls["rerun"] += 1
            state["ablation_dispatched"] = True

        def _fake_run_report(recipe_id, experiment_id, output_dir, *, fmt="blog"):
            calls["report"] += 1
            report_path = Path(output_dir) / "report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("report")
            return report_path

        monkeypatch.setattr(pipeline_mod, "_run_train", _fake_run_train)
        monkeypatch.setattr(pipeline_mod, "_get_latest_experiment", _fake_get_latest_experiment)
        monkeypatch.setattr(pipeline_mod, "_get_experiment_verdict", _fake_get_experiment_verdict)
        monkeypatch.setattr(pipeline_mod, "_run_rerun", _fake_run_rerun)
        monkeypatch.setattr(pipeline_mod, "_run_report", _fake_run_report)
        monkeypatch.setattr(pipeline_mod, "_is_waiting_on_external_execution", lambda recipe_id, experiment_id: False)

        pipeline_mod.run_pipeline(
            Namespace(
                query=None,
                atoms=None,
                recipe=str(recipe_path),
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                output_dir=str(tmp_path / "outputs"),
                report_dir=str(tmp_path / "reports"),
                report_format="blog",
                max_iterations=2,
                dry_run=False,
            )
        )

        assert calls["train"] == 1
        assert calls["rerun"] == 1
        assert calls["report"] == 1


# ---------------------------------------------------------------------------
# CLI report format integration
# ---------------------------------------------------------------------------


class TestReportCLIBlogFormat:
    def test_report_cli_blog_format(self, tmp_path: Path, monkeypatch) -> None:
        db_path = tmp_path / "results.db"
        monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

        db = ResultDB(db_path)
        db.connect()
        recipe = _baseline_recipe()
        exp_id = _populate_db(db, recipe)
        db.close()

        from cli.report import run_report

        report_dir = tmp_path / "reports"
        run_report(
            Namespace(
                experiment_id=exp_id,
                recipe_id=None,
                format="blog",
                output=str(report_dir),
            )
        )

        report_file = report_dir / "report.md"
        assert report_file.exists()
        content = report_file.read_text()
        assert "TL;DR" in content
        assert "Experimental Setup" in content
        assert "Practical Recommendations" in content
