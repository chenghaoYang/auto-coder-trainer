"""SLURM job submission and pipeline orchestration utilities.

Provides helpers for generating sbatch scripts, submitting jobs with
dependency chains, monitoring job status, and orchestrating the full
SWE-Lego training pipeline.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# States that indicate a job is no longer running.
_TERMINAL_STATES = frozenset({
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "CANCELLED+",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
})


# ---------------------------------------------------------------------------
# sbatch script generation
# ---------------------------------------------------------------------------

def render_sbatch(
    job_name: str,
    run_script: str,
    slurm_config: Dict[str, object],
    log_dir: Union[str, Path],
) -> str:
    """Generate the content of an sbatch script.

    Parameters
    ----------
    job_name:
        SLURM job name (e.g. ``act-recipe42-train``).
    run_script:
        Path to the shell script to execute (relative to *bundle_dir* or
        absolute).
    slurm_config:
        Dictionary with SLURM resource parameters.  Expected keys:

        * partition, nodes, gpus_per_node, cpus_per_task, mem, time
        * Optional: account, qos, constraint, modules (list[str]),
          conda_env, extra_sbatch (list[str]), bundle_dir
    log_dir:
        Directory for ``slurm-%j-*.out`` / ``slurm-%j-*.err`` files.

    Returns
    -------
    str
        Complete sbatch script content.
    """
    log_dir = Path(log_dir)

    # Derive a short stage tag from the job name for log file naming.
    stage = job_name.rsplit("-", 1)[-1] if "-" in job_name else job_name

    lines: list[str] = ["#!/bin/bash"]

    # --- Required SBATCH directives ---
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --partition={slurm_config['partition']}")
    lines.append(f"#SBATCH --nodes={slurm_config['nodes']}")
    lines.append(f"#SBATCH --gpus-per-node={slurm_config['gpus_per_node']}")
    lines.append(f"#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}")
    lines.append(f"#SBATCH --mem={slurm_config['mem']}")
    lines.append(f"#SBATCH --time={slurm_config['time']}")
    lines.append(f"#SBATCH --output={log_dir}/slurm-%j-{stage}.out")
    lines.append(f"#SBATCH --error={log_dir}/slurm-%j-{stage}.err")

    # --- Optional directives ---
    if slurm_config.get("account"):
        lines.append(f"#SBATCH --account={slurm_config['account']}")
    if slurm_config.get("qos"):
        lines.append(f"#SBATCH --qos={slurm_config['qos']}")
    if slurm_config.get("constraint"):
        lines.append(f"#SBATCH --constraint={slurm_config['constraint']}")

    # Extra free-form #SBATCH lines (e.g. --exclusive, --mail-type, …)
    for extra in slurm_config.get("extra_sbatch", []):
        lines.append(f"#SBATCH {extra}")

    lines.append("")  # blank separator

    # --- Module loading ---
    modules: list[str] = slurm_config.get("modules", [])
    if modules:
        lines.append("# Module loading")
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")

    # --- Conda activation ---
    conda_env: Optional[str] = slurm_config.get("conda_env")
    if conda_env:
        lines.append("# Conda activation")
        lines.append(f"conda activate {conda_env}")
        lines.append("")

    # --- Execute ---
    bundle_dir = slurm_config.get("bundle_dir", ".")
    lines.append("# Execute")
    lines.append(f"cd {shlex.quote(str(bundle_dir))}")
    lines.append(f"bash {shlex.quote(str(run_script))}")
    lines.append("")  # trailing newline

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def write_sbatch_script(content: str, output_path: Union[str, Path]) -> Path:
    """Write *content* to *output_path* and make it executable (0o755).

    Returns the resolved :class:`~pathlib.Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    output_path.chmod(0o755)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_job(sbatch_path: Union[str, Path]) -> str:
    """Submit a job via ``sbatch`` and return the SLURM job ID.

    Raises
    ------
    RuntimeError
        If ``sbatch`` exits non-zero or its stdout cannot be parsed.
    """
    sbatch_path = Path(sbatch_path)
    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output: {result.stdout.strip()}"
        )

    job_id = match.group(1)
    logger.info("Submitted %s -> job %s", sbatch_path.name, job_id)
    return job_id


def submit_with_dependency(
    sbatch_path: Union[str, Path],
    depend_job_ids: Union[str, List[str]],
    dep_type: str = "afterok",
) -> str:
    """Submit a job with a SLURM dependency on one or more prior jobs.

    Parameters
    ----------
    sbatch_path:
        Path to the ``.sbatch`` script.
    depend_job_ids:
        A single job-ID string or a list of job-ID strings.
    dep_type:
        Dependency type (default ``afterok``).

    Returns
    -------
    str
        The new SLURM job ID.
    """
    sbatch_path = Path(sbatch_path)
    if isinstance(depend_job_ids, str):
        depend_job_ids = [depend_job_ids]

    dep_str = f"{dep_type}:" + ":".join(depend_job_ids)
    result = subprocess.run(
        ["sbatch", f"--dependency={dep_str}", str(sbatch_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch (dependency) failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output: {result.stdout.strip()}"
        )

    job_id = match.group(1)
    logger.info(
        "Submitted %s -> job %s (deps: %s)",
        sbatch_path.name,
        job_id,
        dep_str,
    )
    return job_id


# ---------------------------------------------------------------------------
# Job monitoring
# ---------------------------------------------------------------------------

def check_job_status(job_id: str) -> Dict[str, str]:
    """Query ``sacct`` for the current state of *job_id*.

    Returns
    -------
    dict
        ``{job_id, state, exit_code, elapsed}``
    """
    result = subprocess.run(
        [
            "sacct",
            "-j", job_id,
            "--format=JobID,State,ExitCode,Elapsed",
            "--noheader",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sacct failed (rc={result.returncode}): {result.stderr.strip()}"
        )

    # sacct may return multiple lines (one per job-step).  We want the
    # *batch* line that matches the bare job ID (no ".batch" / ".extern").
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 4 and parts[0] == job_id:
            return {
                "job_id": parts[0],
                "state": parts[1],
                "exit_code": parts[2],
                "elapsed": parts[3],
            }

    # Fallback: return the first line if nothing matched exactly.
    parts = result.stdout.strip().splitlines()[0].split("|") if result.stdout.strip() else []
    if len(parts) >= 4:
        return {
            "job_id": parts[0],
            "state": parts[1],
            "exit_code": parts[2],
            "elapsed": parts[3],
        }

    raise RuntimeError(
        f"Could not parse sacct output for job {job_id}: {result.stdout.strip()}"
    )


def wait_for_job(
    job_id: str,
    poll_interval: int = 60,
    timeout: Optional[int] = None,
) -> Dict[str, str]:
    """Block until *job_id* reaches a terminal state.

    Parameters
    ----------
    job_id:
        SLURM job ID to monitor.
    poll_interval:
        Seconds between ``sacct`` polls (default 60).
    timeout:
        Maximum seconds to wait.  ``None`` means wait indefinitely.

    Returns
    -------
    dict
        Final status dictionary from :func:`check_job_status`.

    Raises
    ------
    TimeoutError
        If *timeout* is exceeded before the job finishes.
    """
    start = time.monotonic()
    while True:
        status = check_job_status(job_id)
        state = status["state"].split()[0].rstrip("+")  # e.g. "CANCELLED+"
        if state in _TERMINAL_STATES or status["state"] in _TERMINAL_STATES:
            logger.info("Job %s finished: %s", job_id, status["state"])
            return status

        if timeout is not None and (time.monotonic() - start) >= timeout:
            raise TimeoutError(
                f"Timed out waiting for job {job_id} after {timeout}s "
                f"(last state: {status['state']})"
            )

        logger.debug(
            "Job %s state=%s, sleeping %ds …", job_id, status["state"], poll_interval
        )
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------

def cancel_job(job_id: str) -> bool:
    """Cancel a SLURM job.  Returns ``True`` on success."""
    result = subprocess.run(
        ["scancel", job_id],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "scancel %s failed (rc=%d): %s",
            job_id,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    logger.info("Cancelled job %s", job_id)
    return True


# ---------------------------------------------------------------------------
# SWE-Lego pipeline orchestration
# ---------------------------------------------------------------------------

def run_swe_lego_pipeline(
    bundle_dir: Union[str, Path],
    slurm_config: Dict[str, object],
) -> Dict[str, object]:
    """Submit the full SWE-Lego 6-stage pipeline with dependencies.

    Stage dependency graph::

        train ──► infer ──► eval ──► import_results
          │
          └──► verifier_train ──────► tts

    Parameters
    ----------
    bundle_dir:
        Root of the experiment bundle (contains ``run.sh``,
        ``serve_and_infer.sh``, etc.).
    slurm_config:
        SLURM resource configuration dictionary passed through to
        :func:`render_sbatch`.  ``bundle_dir`` is injected automatically.

    Returns
    -------
    dict
        ``{pipeline_id, job_ids: {train, infer, eval, verifier_train, tts},
        bundle_dir}``
    """
    bundle_dir = Path(bundle_dir).resolve()
    slurm_dir = bundle_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    log_dir = slurm_dir

    # Inject bundle_dir into the config so render_sbatch can ``cd`` there.
    cfg = {**slurm_config, "bundle_dir": str(bundle_dir)}

    pipeline_id = uuid.uuid4().hex[:12]

    # Derive a short recipe id from the bundle directory name.
    recipe_id = bundle_dir.name

    # Define the stages: (key, script, sbatch filename)
    stages = [
        ("train", "run.sh", "train.sbatch"),
        ("infer", "serve_and_infer.sh", "infer.sbatch"),
        ("eval", "eval.sh", "eval.sbatch"),
        ("verifier_train", "verifier_train.sh", "verifier_train.sbatch"),
        ("tts", "tts.sh", "tts.sbatch"),
        ("import_results", "import_results.sh", "import_results.sbatch"),
    ]

    # 1. Render and write all sbatch scripts.
    sbatch_paths: Dict[str, Path] = {}
    for key, script, fname in stages:
        job_name = f"act-{recipe_id}-{key}"
        content = render_sbatch(job_name, script, cfg, log_dir)
        sbatch_paths[key] = write_sbatch_script(content, slurm_dir / fname)

    # 2. Submit with the dependency DAG.
    job_ids: Dict[str, str] = {}

    # Job 1: train (no deps)
    job_ids["train"] = submit_job(sbatch_paths["train"])

    # Job 2: infer (afterok:train)
    job_ids["infer"] = submit_with_dependency(
        sbatch_paths["infer"], job_ids["train"]
    )

    # Job 3: eval (afterok:infer)
    job_ids["eval"] = submit_with_dependency(
        sbatch_paths["eval"], job_ids["infer"]
    )

    # Job 4: verifier_train (afterok:train)
    job_ids["verifier_train"] = submit_with_dependency(
        sbatch_paths["verifier_train"], job_ids["train"]
    )

    # Job 5: tts (afterok:eval AND verifier_train)
    job_ids["tts"] = submit_with_dependency(
        sbatch_paths["tts"], [job_ids["eval"], job_ids["verifier_train"]]
    )

    # Job 6: import_results (afterok:eval)
    job_ids["import_results"] = submit_with_dependency(
        sbatch_paths["import_results"], job_ids["eval"]
    )

    logger.info(
        "Pipeline %s submitted: %s",
        pipeline_id,
        {k: v for k, v in job_ids.items()},
    )

    return {
        "pipeline_id": pipeline_id,
        "job_ids": job_ids,
        "bundle_dir": str(bundle_dir),
    }


def run_single_script_pipeline(
    bundle_dir: Union[str, Path],
    slurm_config: Dict[str, object],
    *,
    backend: str,
    script_name: str = "run.sh",
    stage: str = "train",
) -> Dict[str, object]:
    """Submit a single launcher script as one SLURM job.

    This is used for external backends that only need a single stage
    (for example TinyZero's generated ``run.sh``).
    """
    bundle_dir = Path(bundle_dir).resolve()
    slurm_dir = bundle_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**slurm_config, "bundle_dir": str(bundle_dir)}
    pipeline_id = uuid.uuid4().hex[:12]

    recipe_id = bundle_dir.parent.name
    job_name = f"act-{recipe_id}-{backend}-{stage}"
    script_path = bundle_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found for SLURM submission: {script_path}")

    sbatch_content = render_sbatch(
        job_name=job_name,
        run_script=script_name,
        slurm_config=cfg,
        log_dir=slurm_dir,
    )
    sbatch_path = write_sbatch_script(
        sbatch_content,
        slurm_dir / f"{backend}-{stage}.sbatch",
    )
    job_id = submit_job(sbatch_path)

    return {
        "pipeline_id": pipeline_id,
        "job_ids": {stage: job_id},
        "bundle_dir": str(bundle_dir),
    }
