"""Tests for SLURM job tracking in DB and the sync command."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from results.db import ResultDB


@pytest.fixture
def db(tmp_path: Path):
    """Create an in-memory-like ResultDB backed by a temp file."""
    db_path = tmp_path / "test.db"
    rdb = ResultDB(db_path)
    rdb.connect()
    yield rdb
    rdb.close()


def _insert_experiment(db: ResultDB, experiment_id: str = "exp-test-001", recipe_id: str = "recipe-test-001") -> str:
    db.insert_experiment({
        "id": experiment_id,
        "recipe_id": recipe_id,
        "config_hash": "abc123",
        "status": "running",
        "trainer_type": "sft",
        "backend": "swe_lego",
        "model_base": "Qwen/Qwen2.5-Coder-7B-Instruct",
    })
    return experiment_id


class TestSlurmJobTracking:
    """Test SLURM job CRUD in ResultDB."""

    def test_insert_and_query_slurm_jobs(self, db: ResultDB) -> None:
        _insert_experiment(db)
        db.insert_slurm_job({
            "job_id": "12345",
            "experiment_id": "exp-test-001",
            "recipe_id": "recipe-test-001",
            "pipeline_id": "pipe-abc",
            "stage": "train",
            "bundle_dir": "/tmp/bundle",
            "status": "PENDING",
        })
        db.insert_slurm_job({
            "job_id": "12346",
            "experiment_id": "exp-test-001",
            "recipe_id": "recipe-test-001",
            "pipeline_id": "pipe-abc",
            "stage": "infer",
            "bundle_dir": "/tmp/bundle",
            "status": "PENDING",
        })

        jobs = db.get_slurm_jobs(recipe_id="recipe-test-001")
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == "12345"
        assert jobs[1]["job_id"] == "12346"

    def test_get_active_slurm_jobs(self, db: ResultDB) -> None:
        _insert_experiment(db)
        db.insert_slurm_job({
            "job_id": "111",
            "experiment_id": "exp-test-001",
            "recipe_id": "recipe-test-001",
            "stage": "train",
            "status": "COMPLETED",
        })
        db.insert_slurm_job({
            "job_id": "222",
            "experiment_id": "exp-test-001",
            "recipe_id": "recipe-test-001",
            "stage": "infer",
            "status": "RUNNING",
        })

        active = db.get_active_slurm_jobs()
        assert len(active) == 1
        assert active[0]["job_id"] == "222"

    def test_update_slurm_job_status(self, db: ResultDB) -> None:
        _insert_experiment(db)
        db.insert_slurm_job({
            "job_id": "333",
            "experiment_id": "exp-test-001",
            "recipe_id": "recipe-test-001",
            "stage": "train",
            "status": "PENDING",
        })

        db.update_slurm_job_status("333", "COMPLETED", elapsed="01:30:00", exit_code="0:0")

        jobs = db.get_slurm_jobs(recipe_id="recipe-test-001")
        assert len(jobs) == 1
        assert jobs[0]["status"] == "COMPLETED"
        assert jobs[0]["elapsed"] == "01:30:00"
        assert jobs[0]["exit_code"] == "0:0"

    def test_batch_insert(self, db: ResultDB) -> None:
        _insert_experiment(db)
        db.insert_slurm_jobs([
            {"job_id": "a1", "experiment_id": "exp-test-001", "recipe_id": "recipe-test-001", "stage": "train", "status": "PENDING"},
            {"job_id": "a2", "experiment_id": "exp-test-001", "recipe_id": "recipe-test-001", "stage": "infer", "status": "PENDING"},
            {"job_id": "a3", "experiment_id": "exp-test-001", "recipe_id": "recipe-test-001", "stage": "eval", "status": "PENDING"},
        ])
        jobs = db.get_slurm_jobs(experiment_id="exp-test-001")
        assert len(jobs) == 3

    def test_filter_by_pipeline_id(self, db: ResultDB) -> None:
        _insert_experiment(db)
        db.insert_slurm_job({
            "job_id": "p1", "experiment_id": "exp-test-001", "recipe_id": "recipe-test-001",
            "pipeline_id": "pipe-A", "stage": "train", "status": "PENDING",
        })
        db.insert_slurm_job({
            "job_id": "p2", "experiment_id": "exp-test-001", "recipe_id": "recipe-test-001",
            "pipeline_id": "pipe-B", "stage": "train", "status": "PENDING",
        })
        jobs = db.get_slurm_jobs(pipeline_id="pipe-A")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "p1"


class TestStatusWithSlurm:
    """Test that status rendering includes SLURM jobs."""

    def test_render_with_slurm_jobs(self) -> None:
        from cli.status import _render_status_report

        report = _render_status_report(
            recipe_id="recipe-test",
            experiments=[],
            tasks=[],
            slurm_jobs=[
                {"job_id": "999", "experiment_id": "exp-1", "stage": "train", "status": "RUNNING", "elapsed": "00:30:00", "submitted_at": "2025-01-01"},
            ],
        )
        assert "SLURM Jobs" in report
        assert "999" in report
        assert "RUNNING" in report

    def test_render_without_slurm(self) -> None:
        from cli.status import _render_status_report

        report = _render_status_report(
            recipe_id="recipe-test",
            experiments=[],
            tasks=[],
            slurm_jobs=None,
        )
        assert "SLURM Jobs" not in report


class TestSyncCommand:
    """Test sync command grouping and decision logic."""

    def test_group_by_pipeline(self) -> None:
        from cli.sync import _group_by_pipeline

        jobs = [
            {"experiment_id": "e1", "pipeline_id": "p1", "stage": "train"},
            {"experiment_id": "e1", "pipeline_id": "p1", "stage": "infer"},
            {"experiment_id": "e2", "pipeline_id": "p2", "stage": "train"},
        ]
        groups = _group_by_pipeline(jobs)
        assert len(groups) == 2
        assert len(groups["e1:p1"]) == 2
        assert len(groups["e2:p2"]) == 1

    def test_sync_no_jobs(self, tmp_path: Path, monkeypatch, capsys) -> None:
        db_path = tmp_path / "sync_test.db"
        monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

        from cli.sync import run_sync
        run_sync(Namespace(recipe_id="nonexistent", dry_run=True, report_format="blog"))

        captured = capsys.readouterr().out
        assert "No tracked SLURM jobs found" in captured

    def test_import_completed_pipeline_dispatches_tinyzero_backend(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from cli.sync import _import_completed_pipeline

        db_path = tmp_path / "sync_dispatch.db"
        rdb = ResultDB(db_path)
        rdb.connect()
        try:
            rdb.insert_experiment(
                {
                    "id": "exp-tinyzero-sync-1",
                    "recipe_id": "recipe-tinyzero-sync-1",
                    "config_hash": "hash-sync-1",
                    "status": "running",
                    "trainer_type": "sft",
                    "backend": "tinyzero",
                    "model_base": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "metrics_json": {},
                    "train_metrics_json": {},
                    "recipe_json": {"id": "recipe-tinyzero-sync-1"},
                    "budget_json": {},
                    "checkpoint_path": None,
                    "error": None,
                }
            )

            bundle_dir = tmp_path / "bundle" / "tinyzero"
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "launcher.json").write_text('{"backend":"tinyzero"}')

            called = {}

            import cli.train as train_mod

            def _fake_import(args):
                called["backend"] = getattr(args, "backend", None)
                called["bundle"] = getattr(args, "import_results", None)

            monkeypatch.setattr(train_mod, "_import_external_results", _fake_import)

            ok = _import_completed_pipeline(
                [
                    {
                        "experiment_id": "exp-tinyzero-sync-1",
                        "recipe_id": "recipe-tinyzero-sync-1",
                        "bundle_dir": str(bundle_dir),
                        "status": "COMPLETED",
                    }
                ],
                rdb,
                report_format="blog",
                dry_run=False,
            )
        finally:
            rdb.close()

        assert ok is True
        assert called["backend"] == "tinyzero"
        assert called["bundle"] == str(bundle_dir)
