#!/usr/bin/env python3
"""Install one built Python artifact into a fresh venv and run the public smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def find_artifact(directory: Path, kind: str) -> Path:
    pattern = "*.whl" if kind == "wheel" else "*.tar.gz"
    artifacts = sorted(directory.glob(pattern))
    if len(artifacts) != 1:
        names = ", ".join(path.name for path in artifacts) or "none"
        raise SystemExit(
            f"expected exactly one {kind} artifact in {directory}, found {len(artifacts)}: {names}"
        )
    return artifacts[0].resolve()


def isolated_python_environment() -> dict[str, str]:
    """Prevent a checkout or an editable install from shadowing the artifact."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=isolated_python_environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kind", choices=("wheel", "sdist"), required=True)
    parser.add_argument(
        "--tests", action="store_true",
        help="Also run the checked-in Python regression suite against the installed artifact.",
    )
    args = parser.parse_args()

    artifact = find_artifact(args.directory, args.kind)
    repo_root = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory(prefix="glrmask-artifact-smoke-") as tmp:
        environment = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = venv_python(environment)
        run(str(python), "-m", "pip", "install", "--upgrade", "pip")
        run(str(python), "-m", "pip", "install", str(artifact))
        run(str(python), str(repo_root / "scripts" / "python-wheel-smoke.py"), cwd=Path(tmp))
        if args.tests:
            run(str(python), "-m", "pip", "install", "pytest")
            run(
                str(python), "-m", "pytest", str(repo_root / "python" / "tests"),
                str(repo_root / "scripts" / "tests" / "test_python_artifact_smoke.py"),
                "-q", "--import-mode=importlib", cwd=Path(tmp),
            )

    print(f"clean-install smoke test passed: {artifact.name}")


if __name__ == "__main__":
    main()
