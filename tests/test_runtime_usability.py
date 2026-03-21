import trainers.rl.data as rl_data
from trainers.rl.data import setup_rollout_env


def test_local_rollout_env_uses_available_python_interpreter():
    env = setup_rollout_env({"type": "local", "timeout": 1})

    result = env["execute_fn"]("print('ok')")

    assert env["ready"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "ok"


def test_rollout_env_defaults_to_docker(monkeypatch):
    seen = {}

    def fake_setup_docker_env(env_config, timeout):
        seen["env_config"] = env_config
        seen["timeout"] = timeout
        return {"env_type": "docker", "ready": True, "execute_fn": lambda *_args, **_kwargs: {}}

    monkeypatch.setattr(rl_data, "_setup_docker_env", fake_setup_docker_env)

    env = rl_data.setup_rollout_env({})

    assert env["env_type"] == "docker"
    assert seen["timeout"] == 60
