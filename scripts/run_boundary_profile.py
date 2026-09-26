#!/usr/bin/env python3
"""Run a command with the recorded static-boundary validation profile.

The validated selectors are now the production defaults. This launcher remains
useful for reproducible benchmarking because it clears stale experiment flags
and then spells the frozen September 26 configuration out explicitly. Only the
launched child's environment is changed. Unsupported native shapes retain the
compiler's normal fallback.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, Mapping, Optional, Sequence

DEFAULT_PROFILE = Path(__file__).resolve().parent / "profiles" / "boundary-link-validated-20260926.json"
CLEARED_PREFIXES = ("GLRMASK_", "PROBE_", "PROFILE_", "PHASE_", "DYNAMIC_REFERENCE")


def load_profile(path: Path) -> Dict[str, str]:
    """Reject malformed profiles instead of silently launching a different arm."""
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict) or not values:
        raise ValueError("The profile must be a nonempty JSON object.")
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"GLRMASK_[A-Z0-9_]+", key):
            raise ValueError("Profile keys must be GLRMASK_* environment variable names.")
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("Profile values must be nonempty strings without NUL bytes.")
        if key.startswith(("GLRMASK_VALIDATE_", "GLRMASK_PROFILE_", "GLRMASK_DUMP_")):
            raise ValueError("Validation, profiling and dump flags do not belong in the timing profile.")
    return dict(values)


def child_environment(parent: Mapping[str, str], profile: Mapping[str, str],
                      threads: Optional[int] = None) -> Dict[str, str]:
    """Remove stale experiment switches, retaining unrelated environment values."""
    if threads is not None and threads < 1:
        raise ValueError("The thread count must be positive.")
    result = {key: value for key, value in parent.items()
              if not key.startswith(CLEARED_PREFIXES)}
    result.update(profile)
    if threads is not None:
        result["RAYON_NUM_THREADS"] = str(threads)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--threads", type=int, help="Set RAYON_NUM_THREADS in the child.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and selected flags without running it.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command following --; no shell expansion is performed.")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("Provide a command after --.")
    try:
        profile = load_profile(args.profile)
        env = child_environment(os.environ, profile, args.threads)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        selected = dict(profile)
        if "RAYON_NUM_THREADS" in env:
            selected["RAYON_NUM_THREADS"] = env["RAYON_NUM_THREADS"]
        print(json.dumps({"command": command, "profile": str(args.profile.resolve()),
                          "environment": selected,
                          "cleared_prefixes": list(CLEARED_PREFIXES)}, indent=2))
        return 0
    try:
        os.execvpe(command[0], command, env)
    except OSError as error:
        print("Cannot launch {!r}: {}".format(command[0], error), file=sys.stderr)
        return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
