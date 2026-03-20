from pathlib import Path

from cli.report import _generate_basic_report
from judge.judge import ExperimentJudge, Verdict
from results.db import ResultDB
from results.report_generator import ReportGenerator


def _make_db(tmp_path: Path) -> ResultDB:
    db = ResultDB(tmp_path / "results.db")
    db.connect()
    return db


def test_judge_accepts_cli_train_payload_without_db() -> None:
    judge = ExperimentJudge()
    results = {
        "train": {
            "recipe_id": "recipe-demo",
            "trainer_type": "sft",
            "backend": "trl",
            "status": "success",
            "metrics": {"loss": 0.82},
        },
        "eval": [
            {
                "benchmark": "swe-bench-lite",
                "seed": 42,
                "metrics": {"resolve_rate": 0.80, "pass@1": 0.50},
            },
            {
                "benchmark": "swe-bench-lite",
                "seed": 123,
                "metrics": {"resolve_rate": 0.82, "pass@1": 0.52},
            },
        ],
        "seeds": [42, 123],
    }

    verdict = judge.judge("recipe-demo", results)

    assert verdict.verdict is Verdict.ACCEPT
    assert verdict.checks["baseline"] is True
    assert verdict.checks["seeds"] is True
    assert verdict.checks["ablation"] is True
    assert verdict.checks["dedup"] is True


def test_judge_rejects_failed_train_payload_from_cli_shape() -> None:
    judge = ExperimentJudge()
    results = {
        "train": {
            "recipe_id": "recipe-demo",
            "trainer_type": "sft",
            "backend": "trl",
            "status": "failed",
            "error": "oom",
        },
        "eval": [],
    }

    verdict = judge.judge("recipe-demo", results)

    assert verdict.verdict is Verdict.REJECT
    assert "Training status: failed" in verdict.reasoning


def test_result_db_exposes_experiment_bundle_with_verdicts_and_ablations(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.insert_experiment(
            {
                "id": "exp-1",
                "recipe_id": "recipe-demo",
                "config_hash": "hash-1",
                "status": "success",
                "trainer_type": "sft",
                "backend": "trl",
                "model_base": "demo-model",
                "metrics_json": {"resolve_rate": 12.3, "pass@1": 0.45},
                "checkpoint_path": "outputs/demo",
                "error": None,
            }
        )
        db.insert_ablation(
            {
                "experiment_id": "exp-1",
                "variable": "trainer.params.lr",
                "value": "1e-5",
                "metrics_json": {"resolve_rate": 11.0},
            }
        )
        db.insert_verdict(
            {
                "experiment_id": "exp-1",
                "verdict": "accept",
                "reasoning": "All checks passed",
                "checks_json": {"baseline": True, "seeds": True},
                "suggestions_json": ["Keep the same recipe"],
            }
        )

        bundle = db.get_experiment_bundle("exp-1")
        detailed = db.find_by_recipe_with_details("recipe-demo")

        assert bundle["experiment"]["id"] == "exp-1"
        assert len(bundle["ablations"]) == 1
        assert len(bundle["verdicts"]) == 1
        assert bundle["verdicts"][0]["verdict"] == "accept"
        assert len(detailed) == 1
        assert detailed[0]["experiment"]["id"] == "exp-1"
    finally:
        db.close()


def test_report_generator_includes_verdicts_ablations_and_comparison(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    try:
        db.insert_experiment(
            {
                "id": "exp-1",
                "recipe_id": "recipe-demo",
                "config_hash": "hash-1",
                "status": "success",
                "trainer_type": "sft",
                "backend": "trl",
                "model_base": "demo-model",
                "metrics_json": {"resolve_rate": 12.3, "pass@1": 0.45},
                "checkpoint_path": "outputs/demo-1",
                "error": None,
            }
        )
        db.insert_experiment(
            {
                "id": "exp-2",
                "recipe_id": "recipe-demo",
                "config_hash": "hash-2",
                "status": "success",
                "trainer_type": "sft",
                "backend": "trl",
                "model_base": "demo-model",
                "metrics_json": {"resolve_rate": 10.0, "pass@1": 0.40},
                "checkpoint_path": "outputs/demo-2",
                "error": None,
            }
        )
        db.insert_ablation(
            {
                "experiment_id": "exp-1",
                "variable": "trainer.params.lr",
                "value": "1e-5",
                "metrics_json": {"resolve_rate": 11.0},
            }
        )
        db.insert_verdict(
            {
                "experiment_id": "exp-1",
                "verdict": "accept",
                "reasoning": "All checks passed",
                "checks_json": {"baseline": True, "seeds": True},
                "suggestions_json": ["Keep the same recipe"],
            }
        )

        generator = ReportGenerator(db)
        output_path = tmp_path / "report.md"
        report = generator.generate_markdown(["exp-1", "exp-2"], output_path)

        assert "## Comparison" in report
        assert "## Ablations" in report
        assert "## Verdicts" in report
        assert "accept" in report
        assert "trainer.params.lr" in report
        assert "exp-2" in report
        assert output_path.read_text() == report
    finally:
        db.close()


def test_basic_report_handles_bundle_details(tmp_path: Path) -> None:
    bundles = [
        {
            "experiment": {
                "id": "exp-1",
                "recipe_id": "recipe-demo",
                "status": "success",
                "trainer_type": "sft",
                "backend": "trl",
                "model_base": "demo-model",
                "metrics_json": {"resolve_rate": 12.3, "pass@1": 0.45},
            },
            "ablations": [
                {
                    "variable": "trainer.params.lr",
                    "value": "1e-5",
                    "metrics_json": {"resolve_rate": 11.0},
                }
            ],
            "verdicts": [
                {
                    "verdict": "accept",
                    "reasoning": "All checks passed",
                    "checks_json": {"baseline": True},
                    "suggestions_json": ["Keep the same recipe"],
                }
            ],
        }
    ]

    _generate_basic_report(bundles, "markdown", tmp_path)
    content = (tmp_path / "report.md").read_text()

    assert "Latest verdict" in content
    assert "trainer.params.lr" in content
    assert "accept" in content
