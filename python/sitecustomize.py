"""
sitecustomize.py

Compatibility shims for environments where GrammarConstraint lacks
from_json_file_gz (and/or from_json_file). We implement these as thin wrappers
around from_json_string, reading the file (gzipped or plain) into a string.

Python auto-imports 'sitecustomize' on startup if it's importable from sys.path.
Since the benchmarks run from the 'python' directory, this file will be picked up.
"""
from __future__ import annotations

import builtins as _builtins
import gzip as _gzip
import sys as _sys
from typing import Any


def _gc_from_json_file_gz(cls: type, path: str):
    """Fallback implementation: load gzipped JSON and delegate to from_json_string."""
    with _gzip.open(path, "rt", encoding="utf-8") as f:
        s = f.read()
    return cls.from_json_string(s)


def _gc_from_json_file(cls: type, path: str):
    """Fallback implementation: load plain JSON and delegate to from_json_string."""
    with open(path, "rt", encoding="utf-8") as f:
        s = f.read()
    return cls.from_json_string(s)


def _patch_gc_if_present() -> Any:
    """
    If builtins.GrammarConstraint is present and missing expected helpers,
    add them by delegating to from_json_string.
    """
    GC = getattr(_builtins, "GrammarConstraint", None)
    if GC is not None and isinstance(GC, type):
        if hasattr(GC, "from_json_string"):
            if not hasattr(GC, "from_json_file_gz"):
                GC.from_json_file_gz = classmethod(_gc_from_json_file_gz)  # type: ignore[attr-defined]
            if not hasattr(GC, "from_json_file"):
                GC.from_json_file = classmethod(_gc_from_json_file)  # type: ignore[attr-defined]
    return GC


def _patch_in_module(mod: Any) -> None:
    """
    If a module exposes a class named GrammarConstraint, patch it similarly.
    This helps if the type isn't placed in builtins but is available via some module.
    """
    try:
        GC = getattr(mod, "GrammarConstraint", None)
    except Exception:
        GC = None
    if isinstance(GC, type) and hasattr(GC, "from_json_string"):
        if not hasattr(GC, "from_json_file_gz"):
            GC.from_json_file_gz = classmethod(_gc_from_json_file_gz)  # type: ignore[attr-defined]
        if not hasattr(GC, "from_json_file"):
            GC.from_json_file = classmethod(_gc_from_json_file)  # type: ignore[attr-defined]


# Attempt to patch immediately in case GrammarConstraint is already present.
_patch_gc_if_present()


# Wrap __import__ so that if GrammarConstraint appears later during imports,
# we patch it seamlessly without altering user code.
_orig_import = _builtins.__import__


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    try:
        # Try patching a builtins-exposed GC (if it appears later).
        _patch_gc_if_present()

        # Patch GC if exposed by the returned module.
        _patch_in_module(mod)

        # If 'from x import y' was used, also patch those attributes if modules/classes.
        if fromlist:
            for sub in fromlist:
                try:
                    attr = getattr(mod, sub, None)
                except Exception:
                    attr = None
                if attr is not None:
                    _patch_in_module(attr)
    except Exception:
        # Never disrupt import process on failure.
        pass
    return mod


if not getattr(_builtins, "_gc_import_patched", False):
    _builtins.__import__ = _patched_import
    _builtins._gc_import_patched = True
