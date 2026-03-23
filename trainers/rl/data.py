"""RL data loading and environment setup utilities.

Provides helpers to:
1. Load coding-task prompts for RL rollouts from HuggingFace datasets or local JSONL.
2. Apply configurable filters (issue quality, trajectory length, etc.).
3. Set up sandboxed rollout environments for code execution during training.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from trainers.utils.data_loading import load_from_path as _shared_load_from_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_rl_prompts(
    sources: list[dict[str, Any]],
    filters: list[dict[str, Any]] | None = None,
    total_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load coding task prompts for RL rollouts.

    Each source dict is expected to contain:
        - name (str): human-readable identifier
        - path (str): HuggingFace dataset ID or local file path
        - mix_weight (float): sampling weight when mixing multiple sources

    Each returned prompt is a dict with at least ``prompt`` and ``metadata`` keys.

    Args:
        sources: Dataset source specifications from the recipe.
        filters: Optional filter specs ``[{type, params}]``.
        total_samples: Cap on the total number of prompts to return.

    Returns:
        List of prompt dicts ready for rollout.
    """
    if not sources:
        raise ValueError("At least one data source is required")

    all_prompts: list[dict[str, Any]] = []

    for source in sources:
        name = source.get("name", "unnamed")
        path = source["path"]
        weight = source.get("mix_weight", 1.0)

        logger.info("Loading RL prompts from %s (path=%s, weight=%.2f)", name, path, weight)
        raw = _load_from_path(path)
        logger.info("  loaded %d raw examples from %s", len(raw), name)

        # Normalise each example into a standard prompt dict
        normalised = [_normalise_prompt(ex, source_name=name) for ex in raw]
        all_prompts.extend(normalised)

    # Apply filters
    if filters:
        all_prompts = _apply_filters(all_prompts, filters)
        logger.info("After filtering: %d prompts", len(all_prompts))

    # Cap total samples
    if total_samples is not None and len(all_prompts) > total_samples:
        all_prompts = all_prompts[:total_samples]
        logger.info("Capped to %d prompts", total_samples)

    return all_prompts


def _load_from_path(path: str) -> list[dict[str, Any]]:
    """Load raw examples from a HuggingFace dataset or local file."""
    return _shared_load_from_path(path)


def _normalise_prompt(example: dict[str, Any], source_name: str = "") -> dict[str, Any]:
    """Normalise a raw dataset example into a standard prompt dict.

    Looks for common field names used across trajectory datasets and maps them
    to a uniform schema.
    """
    # Try to find the prompt text
    prompt = (
        example.get("prompt")
        or example.get("instruction")
        or example.get("problem_statement")
        or example.get("query")
        or ""
    )

    # Try to find test specifications
    tests = (
        example.get("tests")
        or example.get("test_cases")
        or example.get("test")
        or []
    )

    return {
        "prompt": prompt,
        "tests": tests,
        "metadata": {
            "source": source_name,
            "instance_id": example.get("instance_id", example.get("id", "")),
            "repo": example.get("repo", ""),
            "quality_score": example.get("quality_score", example.get("score", 1.0)),
            "turns": example.get("turns", 0),
            "original_fields": list(example.keys()),
        },
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

_FILTER_REGISTRY: dict[str, Any] = {}


def _register_filter(name: str):
    """Decorator to register a filter function."""
    def decorator(fn):
        _FILTER_REGISTRY[name] = fn
        return fn
    return decorator


def _apply_filters(
    prompts: list[dict[str, Any]],
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a chain of filters to the prompt list."""
    for fspec in filters:
        filter_type = fspec["type"]
        params = fspec.get("params", {})
        fn = _FILTER_REGISTRY.get(filter_type)
        if fn is None:
            logger.warning("Unknown filter type %r — skipping", filter_type)
            continue
        prompts = fn(prompts, **params)
    return prompts


@_register_filter("issue_free")
def _filter_issue_free(prompts: list[dict[str, Any]], **_kwargs) -> list[dict[str, Any]]:
    """Keep only prompts that have a non-empty prompt string."""
    return [p for p in prompts if p.get("prompt")]


@_register_filter("length")
def _filter_length(
    prompts: list[dict[str, Any]],
    max_turns: int = 30,
    max_prompt_chars: int | None = None,
    **_kwargs,
) -> list[dict[str, Any]]:
    """Filter by prompt/trajectory length."""
    filtered = []
    for p in prompts:
        # max_turns applies if the prompt carries turn info
        turns = p.get("metadata", {}).get("turns", 0)
        if turns and turns > max_turns:
            continue
        if max_prompt_chars and len(p.get("prompt", "")) > max_prompt_chars:
            continue
        filtered.append(p)
    return filtered


@_register_filter("quality_score")
def _filter_quality_score(
    prompts: list[dict[str, Any]],
    min_score: float = 0.5,
    **_kwargs,
) -> list[dict[str, Any]]:
    """Keep only prompts whose metadata quality_score >= min_score."""
    return [
        p for p in prompts
        if p.get("metadata", {}).get("quality_score", 1.0) >= min_score
    ]


# ---------------------------------------------------------------------------
# Rollout environment
# ---------------------------------------------------------------------------

def setup_rollout_env(env_config: dict[str, Any]) -> dict[str, Any]:
    """Set up a sandboxed code execution environment for RL rollouts.

    Depending on the backend, this may launch Docker containers, set up
    a local venv, or configure a remote sandbox service.

    Args:
        env_config: Environment configuration, e.g.::
            {
                "type": "docker",               # or "local", "remote"
                "image": "python:3.11-slim",
                "timeout": 60,
                "memory_limit": "4g",
                "network": false
            }

    Returns:
        A dict describing the initialised environment, including:
            - env_type (str)
            - ready (bool)
            - execute_fn (callable): function(code: str) -> dict with
              ``stdout``, ``stderr``, ``exit_code``, ``tests_passed``, ``tests_total``
    """
    env_type = env_config.get("type", "docker")
    timeout = env_config.get("timeout", 60)

    if env_type == "docker":
        return _setup_docker_env(env_config, timeout)
    elif env_type == "remote":
        return _setup_remote_env(env_config, timeout)
    elif env_type == "local":
        return _setup_local_env(env_config, timeout)
    else:
        raise ValueError(
            f"Unknown rollout env type {env_type!r}. "
            "Supported types: 'docker', 'remote', 'local'."
        )


def _setup_local_env(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a local subprocess-based execution environment."""
    import atexit
    import shutil
    import subprocess
    import tempfile

    logger.warning(
        "SECURITY WARNING: Local rollout environment executes arbitrary code "
        "directly on the host via subprocess with NO sandboxing and NO network "
        "isolation. Use 'docker' env type for production workloads."
    )

    workdir = tempfile.mkdtemp(prefix="rl_rollout_")
    atexit.register(shutil.rmtree, workdir, True)

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        full_code = code
        if test_code:
            full_code += "\n\n" + test_code
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
            # Parse test results from output if available
            tests_passed, tests_total = _parse_test_output(result.stdout + result.stderr)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "exit_code": -1,
                "tests_passed": 0,
                "tests_total": 0,
            }

    return {"env_type": "local", "ready": True, "workdir": workdir, "execute_fn": execute_fn}


def _setup_docker_env(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a Docker-based sandboxed execution environment."""
    image = env_config.get("image", "python:3.11-slim")
    memory_limit = env_config.get("memory_limit", "4g")
    network_disabled = not env_config.get("network", False)
    allow_local_fallback = env_config.get("allow_local_fallback") is True

    try:
        import subprocess
        # Verify Docker is available
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        if allow_local_fallback:
            logger.warning("Docker not available, falling back to local env: %s", exc)
            return _setup_local_env(env_config, timeout)
        logger.warning(
            "Docker not available and allow_local_fallback is not explicitly True "
            "in env config. Set allow_local_fallback: true to permit unsandboxed "
            "local execution. Error: %s", exc
        )
        return {
            "env_type": "docker",
            "ready": False,
            "error": f"Docker not available: {exc}",
            "execute_fn": lambda *_args, **_kwargs: {
                "stdout": "",
                "stderr": f"Docker not available: {exc}",
                "exit_code": -1,
                "tests_passed": 0,
                "tests_total": 0,
            },
        }

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        import subprocess as sp
        full_code = code + ("\n\n" + test_code if test_code else "")
        cmd = [
            "docker", "run", "--rm",
            f"--memory={memory_limit}",
            *(["--network=none"] if network_disabled else []),
            image,
            "python", "-c", full_code,
        ]
        try:
            result = sp.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
            tests_passed, tests_total = _parse_test_output(result.stdout + result.stderr)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except sp.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Docker timeout after {timeout + 30}s",
                "exit_code": -1,
                "tests_passed": 0,
                "tests_total": 0,
            }

    return {"env_type": "docker", "ready": True, "image": image, "execute_fn": execute_fn}


def _setup_remote_env(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a remote sandboxed execution environment.

    Supported backends (via ``env_config["backend"]``):

    * ``"modal"``   – Modal (modal.com) serverless sandbox
    * ``"e2b"``     – E2B (e2b.dev) code interpreter sandbox
    * ``"k8s"``     – Ephemeral Kubernetes pod via the official client
    * ``"sandbox"`` – Generic subprocess-in-container via Docker-over-SSH

    Common env_config keys consumed by all backends:
        timeout (int), memory_limit (str), network (bool, default False).
    """
    backend = env_config.get("backend", "unspecified")
    dispatchers = {
        "modal": _setup_modal_backend,
        "e2b": _setup_e2b_backend,
        "k8s": _setup_k8s_backend,
        "sandbox": _setup_sandbox_backend,
    }
    dispatcher = dispatchers.get(backend)
    if dispatcher is None:
        raise ValueError(
            f"Unknown remote backend {backend!r}. "
            f"Supported backends: {sorted(dispatchers)}."
        )
    return dispatcher(env_config, timeout)


# -- Modal backend -----------------------------------------------------------

def _setup_modal_backend(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a Modal sandbox backend."""
    try:
        import modal  # noqa: F811
    except ImportError:
        msg = (
            "The 'modal' package is required for the Modal backend. "
            "Install it with: pip install modal"
        )
        logger.warning(msg)
        return _not_ready_env("remote/modal", msg)

    image_name = env_config.get("image", "python:3.11-slim")
    memory_limit = env_config.get("memory_limit", "4g")
    network_disabled = not env_config.get("network", False)
    app_name = env_config.get("modal_app", "rl-rollout")

    # Convert memory string like "4g" to MB int for Modal
    memory_mb = _parse_memory_mb(memory_limit)

    logger.info(
        "Modal backend: app=%s, image=%s, memory=%dMB, network=%s, timeout=%ds",
        app_name, image_name, memory_mb,
        "disabled" if network_disabled else "enabled", timeout,
    )

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        full_code = code + ("\n\n" + test_code if test_code else "")
        try:
            sb = modal.Sandbox.create(
                "python", "-c", full_code,
                image=modal.Image.from_registry(image_name),
                timeout=timeout,
                memory=memory_mb,
                **({"block_network": True} if network_disabled else {}),
                app_name=app_name,
            )
            sb.wait()
            stdout = sb.stdout.read()
            stderr = sb.stderr.read()
            exit_code = sb.returncode or 0
            tests_passed, tests_total = _parse_test_output(stdout + stderr)
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except modal.exception.SandboxTimeoutError:
            return _timeout_result(timeout, "Modal")
        except Exception as exc:
            logger.warning("Modal execution failed: %s", exc)
            return _error_result(f"Modal execution error: {exc}")

    return {"env_type": "remote/modal", "ready": True, "execute_fn": execute_fn}


# -- E2B backend -------------------------------------------------------------

def _setup_e2b_backend(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up an E2B code interpreter backend."""
    try:
        from e2b_code_interpreter import Sandbox as E2BSandbox  # noqa: F811
    except ImportError:
        msg = (
            "The 'e2b-code-interpreter' package is required for the E2B backend. "
            "Install it with: pip install e2b-code-interpreter"
        )
        logger.warning(msg)
        return _not_ready_env("remote/e2b", msg)

    api_key = env_config.get("api_key")
    template = env_config.get("e2b_template", "base")
    memory_limit = env_config.get("memory_limit", "4g")
    network_disabled = not env_config.get("network", False)

    if not api_key:
        msg = "E2B backend requires 'api_key' in env_config (E2B_API_KEY)."
        logger.warning(msg)
        return _not_ready_env("remote/e2b", msg)

    logger.info(
        "E2B backend: template=%s, timeout=%ds, network=%s",
        template, timeout, "disabled" if network_disabled else "enabled",
    )

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        full_code = code + ("\n\n" + test_code if test_code else "")
        sandbox = None
        try:
            sandbox = E2BSandbox(
                api_key=api_key,
                template=template,
                metadata={"memory": memory_limit},
            )
            execution = sandbox.run_code(full_code, timeout=timeout)

            stdout_parts = []
            stderr_parts = []
            for log in execution.logs.stdout:
                stdout_parts.append(log)
            for log in execution.logs.stderr:
                stderr_parts.append(log)
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            exit_code = 0 if execution.error is None else 1
            if execution.error is not None:
                stderr += f"\n{execution.error.name}: {execution.error.value}"

            tests_passed, tests_total = _parse_test_output(stdout + stderr)
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except TimeoutError:
            return _timeout_result(timeout, "E2B")
        except Exception as exc:
            logger.warning("E2B execution failed: %s", exc)
            return _error_result(f"E2B execution error: {exc}")
        finally:
            if sandbox is not None:
                try:
                    sandbox.kill()
                except Exception:
                    pass

    return {"env_type": "remote/e2b", "ready": True, "execute_fn": execute_fn}


# -- Kubernetes backend -------------------------------------------------------

def _setup_k8s_backend(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up an ephemeral Kubernetes pod backend."""
    try:
        from kubernetes import client as k8s_client, config as k8s_config  # noqa: F811
    except ImportError:
        msg = (
            "The 'kubernetes' package is required for the K8s backend. "
            "Install it with: pip install kubernetes"
        )
        logger.warning(msg)
        return _not_ready_env("remote/k8s", msg)

    image = env_config.get("image", "python:3.11-slim")
    memory_limit = env_config.get("memory_limit", "4g")
    namespace = env_config.get("k8s_namespace", "default")
    network_disabled = not env_config.get("network", False)
    kubeconfig = env_config.get("kubeconfig")

    # Load kube config
    try:
        if kubeconfig:
            k8s_config.load_kube_config(config_file=kubeconfig)
        else:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
    except Exception as exc:
        msg = f"Failed to load Kubernetes configuration: {exc}"
        logger.warning(msg)
        return _not_ready_env("remote/k8s", msg)

    core_v1 = k8s_client.CoreV1Api()

    logger.info(
        "K8s backend: namespace=%s, image=%s, memory=%s, network=%s, timeout=%ds",
        namespace, image, memory_limit,
        "disabled" if network_disabled else "enabled", timeout,
    )

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        import time
        import uuid

        full_code = code + ("\n\n" + test_code if test_code else "")
        pod_name = f"rl-rollout-{uuid.uuid4().hex[:12]}"

        # Build the pod spec
        container = k8s_client.V1Container(
            name="runner",
            image=image,
            command=["python", "-c", full_code],
            resources=k8s_client.V1ResourceRequirements(
                limits={"memory": memory_limit},
            ),
        )

        # Network policy: use a label so a NetworkPolicy can deny egress
        labels = {"app": "rl-rollout", "rl-network": "deny" if network_disabled else "allow"}
        pod_spec = k8s_client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            # Disable service account token auto-mount for security
            automount_service_account_token=False,
            **({"dns_policy": "None", "dns_config": k8s_client.V1PodDNSConfig(
                nameservers=["127.0.0.1"],
            )} if network_disabled else {}),
        )
        pod = k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(name=pod_name, labels=labels),
            spec=pod_spec,
        )

        try:
            core_v1.create_namespaced_pod(namespace=namespace, body=pod)

            # Wait for pod to complete
            deadline = time.monotonic() + timeout + 30  # extra grace for k8s overhead
            phase = "Pending"
            while time.monotonic() < deadline:
                pod_status = core_v1.read_namespaced_pod_status(pod_name, namespace)
                phase = pod_status.status.phase
                if phase in ("Succeeded", "Failed"):
                    break
                time.sleep(1)

            if phase not in ("Succeeded", "Failed"):
                # Timed out
                _delete_pod(core_v1, pod_name, namespace)
                return _timeout_result(timeout, "K8s")

            # Retrieve logs
            logs = core_v1.read_namespaced_pod_log(pod_name, namespace)
            exit_code = 0 if phase == "Succeeded" else 1

            # Best-effort split stdout/stderr (k8s logs merges them)
            tests_passed, tests_total = _parse_test_output(logs)
            return {
                "stdout": logs,
                "stderr": "" if phase == "Succeeded" else logs,
                "exit_code": exit_code,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except Exception as exc:
            logger.warning("K8s execution failed: %s", exc)
            return _error_result(f"K8s execution error: {exc}")
        finally:
            _delete_pod(core_v1, pod_name, namespace)

    return {"env_type": "remote/k8s", "ready": True, "execute_fn": execute_fn}


def _delete_pod(core_v1: Any, pod_name: str, namespace: str) -> None:
    """Best-effort deletion of an ephemeral pod."""
    try:
        core_v1.delete_namespaced_pod(
            pod_name, namespace,
            body={"gracePeriodSeconds": 0},
        )
    except Exception:
        pass


# -- Sandbox (generic container-over-SSH) backend ----------------------------

def _setup_sandbox_backend(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a generic subprocess-in-container sandbox backend.

    This backend shells out to ``docker`` on a (possibly remote) host via an
    optional SSH wrapper, providing a middle ground between the local Docker
    backend and fully managed cloud services.  When no ``sandbox_host`` is
    configured it behaves like the Docker backend but is routed through the
    remote env_type path.
    """
    import shutil
    import subprocess

    image = env_config.get("image", "python:3.11-slim")
    memory_limit = env_config.get("memory_limit", "4g")
    network_disabled = not env_config.get("network", False)
    sandbox_host = env_config.get("sandbox_host")  # e.g. "user@host"
    ssh_key = env_config.get("ssh_key")

    # Build the optional SSH prefix
    ssh_prefix: list[str] = []
    if sandbox_host:
        ssh_prefix = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if ssh_key:
            ssh_prefix += ["-i", ssh_key]
        ssh_prefix.append(sandbox_host)

    # Verify docker reachability
    docker_check_cmd = ssh_prefix + ["docker", "info"]
    try:
        subprocess.run(docker_check_cmd, capture_output=True, check=True, timeout=15)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        host_label = sandbox_host or "localhost"
        msg = f"Docker not reachable on {host_label}: {exc}"
        logger.warning(msg)
        return _not_ready_env("remote/sandbox", msg)

    logger.info(
        "Sandbox backend: host=%s, image=%s, memory=%s, network=%s, timeout=%ds",
        sandbox_host or "localhost", image, memory_limit,
        "disabled" if network_disabled else "enabled", timeout,
    )

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        import subprocess as sp

        full_code = code + ("\n\n" + test_code if test_code else "")
        docker_cmd = [
            "docker", "run", "--rm",
            f"--memory={memory_limit}",
            *(["--network=none"] if network_disabled else []),
            image,
            "python", "-c", full_code,
        ]
        if ssh_prefix:
            # Pass code via stdin to avoid shell metacharacter interpretation
            # when SSH concatenates remote arguments into a shell string.
            remote_docker_cmd = " ".join(
                _shell_quote(arg) for arg in docker_cmd[:-2]  # everything except "python -c CODE"
            )
            # Pipe code into python via stdin instead of -c
            remote_cmd = f"{remote_docker_cmd} python -"
            cmd = ssh_prefix + [remote_cmd]
            try:
                result = sp.run(
                    cmd, input=full_code, capture_output=True, text=True,
                    timeout=timeout + 30,
                )
                tests_passed, tests_total = _parse_test_output(result.stdout + result.stderr)
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "tests_passed": tests_passed,
                    "tests_total": tests_total,
                }
            except sp.TimeoutExpired:
                return _timeout_result(timeout, "Sandbox")
            except Exception as exc:
                return _error_result(f"Sandbox execution error: {exc}")
        else:
            cmd = docker_cmd
            try:
                result = sp.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
                tests_passed, tests_total = _parse_test_output(result.stdout + result.stderr)
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "tests_passed": tests_passed,
                    "tests_total": tests_total,
                }
            except sp.TimeoutExpired:
                return _timeout_result(timeout, "Sandbox")
            except Exception as exc:
                return _error_result(f"Sandbox execution error: {exc}")

    return {"env_type": "remote/sandbox", "ready": True, "execute_fn": execute_fn}


# -- Shared helpers for remote backends ---------------------------------------

def _shell_quote(s: str) -> str:
    """Shell-escape a string for safe use in remote SSH commands."""
    import shlex
    return shlex.quote(s)


def _not_ready_env(env_type: str, error_msg: str) -> dict[str, Any]:
    """Return a not-ready environment dict with a stub execute_fn."""
    return {
        "env_type": env_type,
        "ready": False,
        "error": error_msg,
        "execute_fn": lambda *_args, **_kwargs: {
            "stdout": "",
            "stderr": error_msg,
            "exit_code": -1,
            "tests_passed": 0,
            "tests_total": 0,
        },
    }


def _timeout_result(timeout: int, backend_label: str) -> dict[str, Any]:
    """Build a standard timeout result dict."""
    return {
        "stdout": "",
        "stderr": f"{backend_label} timeout after {timeout}s",
        "exit_code": -1,
        "tests_passed": 0,
        "tests_total": 0,
    }


def _error_result(error_msg: str) -> dict[str, Any]:
    """Build a standard error result dict."""
    return {
        "stdout": "",
        "stderr": error_msg,
        "exit_code": -1,
        "tests_passed": 0,
        "tests_total": 0,
    }


def _parse_memory_mb(memory_str: str) -> int:
    """Convert a memory string like '4g', '512m', '2048' to megabytes."""
    memory_str = memory_str.strip().lower()
    if memory_str.endswith("g"):
        return int(float(memory_str[:-1]) * 1024)
    if memory_str.endswith("m"):
        return int(float(memory_str[:-1]))
    if memory_str.endswith("k"):
        return max(1, int(float(memory_str[:-1]) / 1024))
    # Assume megabytes if no suffix
    try:
        return int(memory_str)
    except ValueError:
        logger.warning("Could not parse memory limit %r, defaulting to 4096MB", memory_str)
        return 4096


def _parse_test_output(output: str) -> tuple[int, int]:
    """Best-effort extraction of test pass/total counts from pytest-style output.

    Looks for patterns like:
        ``5 passed, 2 failed`` (pytest)
        ``Ran 7 tests`` / ``OK`` (unittest)
    """
    import re

    # pytest style: "5 passed, 2 failed, 1 error, 3 skipped"
    pytest_passed = re.search(r"(\d+)\s+passed", output)
    pytest_failed = re.search(r"(\d+)\s+failed", output)
    pytest_error = re.search(r"(\d+)\s+error", output)
    pytest_skipped = re.search(r"(\d+)\s+skipped", output)
    if pytest_passed or pytest_failed or pytest_error:
        passed = int(pytest_passed.group(1)) if pytest_passed else 0
        failed = int(pytest_failed.group(1)) if pytest_failed else 0
        errors = int(pytest_error.group(1)) if pytest_error else 0
        skipped = int(pytest_skipped.group(1)) if pytest_skipped else 0
        total = passed + failed + errors + skipped
        return passed, total

    # unittest style: "Ran N tests" + "OK" or "FAILED"
    unittest_ran = re.search(r"Ran\s+(\d+)\s+test", output)
    if unittest_ran:
        total = int(unittest_ran.group(1))
        if "OK" in output:
            return total, total
        # Try to extract failure count
        unittest_fail = re.search(r"failures=(\d+)", output)
        unittest_err = re.search(r"errors=(\d+)", output)
        fails = int(unittest_fail.group(1)) if unittest_fail else 0
        errs = int(unittest_err.group(1)) if unittest_err else 0
        return max(0, total - fails - errs), total

    return 0, 0
