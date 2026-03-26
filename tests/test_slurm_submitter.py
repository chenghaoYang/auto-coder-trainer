from pathlib import Path

import trainers.slurm.submitter as submitter
from trainers.slurm.submitter import render_sbatch


def _default_slurm_config() -> dict:
    return {
        "partition": "gpu",
        "nodes": 1,
        "gpus_per_node": 1,
        "cpus_per_task": 16,
        "mem": "256G",
        "time": "72:00:00",
    }


def test_render_sbatch_basic(tmp_path: Path) -> None:
    config = _default_slurm_config()
    content = render_sbatch(
        job_name="act-test-train",
        run_script="run.sh",
        slurm_config=config,
        log_dir=str(tmp_path / "logs"),
    )

    assert "#!/bin/bash" in content
    assert "#SBATCH --job-name=act-test-train" in content
    assert "#SBATCH --partition=gpu" in content
    assert "#SBATCH --nodes=1" in content
    assert "#SBATCH --gpus-per-node=1" in content
    assert "#SBATCH --cpus-per-task=16" in content
    assert "#SBATCH --mem=256G" in content
    assert "#SBATCH --time=72:00:00" in content
    assert "bash run.sh" in content


def test_render_sbatch_with_optional_fields(tmp_path: Path) -> None:
    config = _default_slurm_config()
    config["account"] = "myaccount"
    config["qos"] = "high"
    config["constraint"] = "h200"
    config["modules"] = ["cuda/12.8", "conda"]
    config["conda_env"] = "swe_lego"
    config["extra_sbatch"] = ["#SBATCH --exclusive"]

    content = render_sbatch(
        job_name="act-test-full",
        run_script="run.sh",
        slurm_config=config,
        log_dir=str(tmp_path),
    )

    assert "#SBATCH --account=myaccount" in content
    assert "#SBATCH --qos=high" in content
    assert "#SBATCH --constraint=h200" in content
    assert "module load cuda/12.8" in content
    assert "module load conda" in content
    assert "conda activate swe_lego" in content
    assert "#SBATCH --exclusive" in content


def test_render_sbatch_output_and_error_paths(tmp_path: Path) -> None:
    config = _default_slurm_config()
    log_dir = str(tmp_path / "logs")
    content = render_sbatch(
        job_name="act-test",
        run_script="run.sh",
        slurm_config=config,
        log_dir=log_dir,
    )

    assert f"#SBATCH --output={log_dir}" in content
    assert f"#SBATCH --error={log_dir}" in content


def test_run_swe_lego_pipeline_submits_import_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    written_scripts: list[Path] = []
    submit_calls: list[tuple[str, object, str]] = []
    job_ids = iter(["101", "102", "103", "104", "105", "106"])

    def _fake_write(content: str, output_path: Path) -> Path:
        output_path.write_text(content)
        written_scripts.append(output_path)
        return output_path

    monkeypatch.setattr(submitter, "write_sbatch_script", _fake_write)
    monkeypatch.setattr(submitter, "submit_job", lambda path: next(job_ids))

    def _fake_submit_with_dependency(path: Path, deps, dep_type: str = "afterok") -> str:
        submit_calls.append((Path(path).name, deps, dep_type))
        return next(job_ids)

    monkeypatch.setattr(submitter, "submit_with_dependency", _fake_submit_with_dependency)

    result = submitter.run_swe_lego_pipeline(bundle_dir, _default_slurm_config())

    assert "import_results" in result["job_ids"]
    assert any(path.name == "import_results.sbatch" for path in written_scripts)
    assert ("import_results.sbatch", "103", "afterok") in submit_calls


def test_run_single_script_pipeline_submits_one_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "run.sh").write_text("#!/usr/bin/env bash\necho hi\n")

    written_paths: list[Path] = []

    def _fake_write(content: str, output_path: Path) -> Path:
        output_path.write_text(content)
        written_paths.append(output_path)
        return output_path

    monkeypatch.setattr(submitter, "write_sbatch_script", _fake_write)
    monkeypatch.setattr(submitter, "submit_job", lambda _path: "777")

    result = submitter.run_single_script_pipeline(
        bundle_dir,
        _default_slurm_config(),
        backend="tinyzero",
        script_name="run.sh",
        stage="train",
    )

    assert result["job_ids"] == {"train": "777"}
    assert any(path.name == "tinyzero-train.sbatch" for path in written_paths)
