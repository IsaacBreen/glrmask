from __future__ import annotations

import os
import subprocess
import sys


def _run(code: str, *, purge_delay: str | None) -> str:
    env = os.environ.copy()
    if purge_delay is None:
        env.pop("MIMALLOC_PURGE_DELAY", None)
    else:
        env["MIMALLOC_PURGE_DELAY"] = purge_delay
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def test_python_extension_defaults_to_no_automatic_purge() -> None:
    value = _run(
        "import glrmask; print(glrmask._internal.mimalloc_purge_delay())",
        purge_delay=None,
    )
    assert value == "-1"


def test_explicit_mimalloc_purge_delay_overrides_default() -> None:
    code = "import glrmask; print(glrmask._internal.mimalloc_purge_delay())"
    assert _run(code, purge_delay="1000") == "1000"
    assert _run(code, purge_delay="0") == "0"


def test_explicit_collection_is_available() -> None:
    value = _run(
        "import glrmask; "
        "glrmask._internal.collect_allocator(); "
        "print(glrmask._internal.mimalloc_purge_delay())",
        purge_delay=None,
    )
    assert value == "-1"
