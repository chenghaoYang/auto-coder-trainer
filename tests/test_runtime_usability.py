from trainers.rl.data import setup_rollout_env


def test_local_rollout_env_uses_available_python_interpreter():
    env = setup_rollout_env({"type": "local", "timeout": 1})

    result = env["execute_fn"]("print('ok')")

    assert env["ready"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "ok"
