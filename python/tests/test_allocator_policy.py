from __future__ import annotations

import os
import subprocess
import sys


def _purge_delay(*, value: str | None) -> int:
    env = os.environ.copy()
    if value is None:
        env.pop("MIMALLOC_PURGE_DELAY", None)
    else:
        env["MIMALLOC_PURGE_DELAY"] = value
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import glrmask; print(glrmask._internal.mimalloc_purge_delay())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return int(result.stdout.strip())


def test_python_extension_defaults_to_no_automatic_purge() -> None:
    assert _purge_delay(value=None) == -1


def test_explicit_mimalloc_purge_delay_overrides_default() -> None:
    assert _purge_delay(value="1000") == 1000
    assert _purge_delay(value="0") == 0
