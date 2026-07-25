from __future__ import annotations

import os
import subprocess
import sys


def _run(
    code: str,
    *,
    purge_delay: str | None = None,
    purge_decommits: str | None = None,
) -> str:
    env = os.environ.copy()
    env.pop("MIMALLOC_PURGE_DELAY", None)
    env.pop("MIMALLOC_PURGE_DECOMMITS", None)
    env.pop("MIMALLOC_RESET_DECOMMITS", None)
    if purge_delay is not None:
        env["MIMALLOC_PURGE_DELAY"] = purge_delay
    if purge_decommits is not None:
        env["MIMALLOC_PURGE_DECOMMITS"] = purge_decommits
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def test_python_extension_keeps_delayed_automatic_purging() -> None:
    value = _run(
        "import glrmask; print(glrmask._internal.mimalloc_purge_delay())",
    )
    assert value == "1000"


def test_python_extension_defaults_purges_to_reset_not_decommit() -> None:
    value = _run(
        "import glrmask; print(int(glrmask._internal.mimalloc_purge_decommits()))",
    )
    assert value == "0"


def test_explicit_mimalloc_purge_settings_remain_authoritative() -> None:
    code = (
        "import glrmask; "
        "print(glrmask._internal.mimalloc_purge_delay(), "
        "int(glrmask._internal.mimalloc_purge_decommits()))"
    )
    assert _run(code, purge_delay="-1", purge_decommits="1") == "-1 1"
    assert _run(code, purge_delay="250", purge_decommits="0") == "250 0"
    assert _run(code, purge_delay="0", purge_decommits="1") == "0 1"


def test_explicit_collection_preserves_configured_policy() -> None:
    code = (
        "import glrmask; "
        "glrmask._internal.collect_allocator(); "
        "print(glrmask._internal.mimalloc_purge_delay(), "
        "int(glrmask._internal.mimalloc_purge_decommits()))"
    )
    assert _run(code, purge_delay="-1", purge_decommits="0") == "-1 0"
    assert _run(code, purge_delay="1000", purge_decommits="1") == "1000 1"
