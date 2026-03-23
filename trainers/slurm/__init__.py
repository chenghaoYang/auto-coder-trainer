"""Generic SLURM job submission and pipeline orchestration."""

from trainers.slurm.submitter import (
    render_sbatch,
    write_sbatch_script,
    submit_job,
    submit_with_dependency,
    check_job_status,
    wait_for_job,
    cancel_job,
    run_swe_lego_pipeline,
)

__all__ = [
    "render_sbatch",
    "write_sbatch_script",
    "submit_job",
    "submit_with_dependency",
    "check_job_status",
    "wait_for_job",
    "cancel_job",
    "run_swe_lego_pipeline",
]
