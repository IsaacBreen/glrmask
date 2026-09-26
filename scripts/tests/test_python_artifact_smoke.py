"""The artifact gate must test the installed package, not an ambient checkout."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "python-artifact-smoke.py"
SPEC = importlib.util.spec_from_file_location("python_artifact_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_python_search_paths_are_isolated_without_mutating_parent():
    with mock.patch.dict(smoke.os.environ, {
        "PYTHONPATH": "/unrelated/source", "PYTHONHOME": "/other/python", "KEEP": "yes"
    }, clear=True):
        result = smoke.isolated_python_environment()
        assert "PYTHONPATH" not in result and "PYTHONHOME" not in result
        assert result["KEEP"] == "yes"
        assert result["PYTHONNOUSERSITE"] == "1"
        assert result["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert smoke.os.environ["PYTHONPATH"] == "/unrelated/source"


def test_run_uses_the_isolated_environment(tmp_path):
    with mock.patch.object(smoke.subprocess, "run") as run:
        smoke.run("python", "-c", "pass", cwd=tmp_path)
    assert run.call_args.args == (("python", "-c", "pass"),)
    assert run.call_args.kwargs["cwd"] == tmp_path
    assert run.call_args.kwargs["check"] is True
    assert "PYTHONPATH" not in run.call_args.kwargs["env"]


def test_artifact_selection_rejects_ambiguous_directories(tmp_path):
    with pytest.raises(SystemExit, match="found 0"):
        smoke.find_artifact(tmp_path, "wheel")
    wheel = tmp_path / "one.whl"
    wheel.write_bytes(b"not inspected here")
    assert smoke.find_artifact(tmp_path, "wheel") == wheel.resolve()
    (tmp_path / "two.whl").write_bytes(b"not inspected here")
    with pytest.raises(SystemExit, match="found 2"):
        smoke.find_artifact(tmp_path, "wheel")


def test_full_suite_runs_after_install_and_smoke(tmp_path):
    wheel = tmp_path / "one.whl"
    wheel.write_bytes(b"selection fixture")
    with mock.patch.object(smoke.sys, "argv", [str(SCRIPT), str(tmp_path), "--kind", "wheel", "--tests"]), \
         mock.patch.object(smoke.venv.EnvBuilder, "create"), \
         mock.patch.object(smoke, "run") as run:
        smoke.main()
    calls = run.call_args_list
    assert len(calls) == 5
    assert calls[1].args[-1] == str(wheel.resolve())
    assert calls[2].args[-1].endswith("python-wheel-smoke.py")
    assert calls[3].args[-1] == "pytest"
    assert calls[4].args[1:3] == ("-m", "pytest")
    assert "--import-mode=importlib" in calls[4].args
    assert calls[4].kwargs["cwd"] == calls[2].kwargs["cwd"]
