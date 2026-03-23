from pathlib import Path

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
