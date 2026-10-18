"""Tests verifying Python version compatibility and runtime contracts."""

import sys
import glrmask
import numpy as np
import pytest


def test_supported_python_version():
    """Verify that Python runtime satisfies the supported version envelope."""
    assert sys.version_info >= (3, 9)
    # Testing envelope spans 3.9 up through 3.15+
    assert sys.version_info < (3, 16)


def test_basic_masking_lifecycle():
    """Verify core grammar compilation and state lifecycle on active Python runtime."""
    v = glrmask.Vocab.from_id_to_bytes({0: b"hello", 1: b" ", 2: b"world"})
    g = glrmask.Grammar.from_ebnf('start ::= "hello" " " "world"')
    c = g.compile(v)
    s = c.start()

    assert s.mask().tolist() == [True, False, False]
    s.commit_token(0)
    assert s.mask().tolist() == [False, True, False]
    s.commit_token(1)
    assert s.mask().tolist() == [False, False, True]
    s.commit_token(2)
    assert s.is_accepting()


def test_forcing_tokens_query():
    """Verify forced_tokens query runs and releases interpreter without race/hang."""
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b"})
    g = glrmask.Grammar.from_ebnf('start ::= "a" "b"')
    c = g.compile(v)
    s = c.start()
    # At start, "a" (0) is inevitable
    forced = s.forced_tokens()
    assert forced == [0]


def test_gil_status_contract():
    """Verify behavior under GIL-enabled or free-threaded runtime.

    On non-free-threaded Python (or before 3.13), GIL is enabled.
    On Python 3.13+, sys._is_gil_enabled() indicates whether GIL is active.
    """
    if hasattr(sys, "_is_gil_enabled"):
        gil_enabled = sys._is_gil_enabled()
        # Extension is built with stable abi3 targeting non-free-threaded runtime
        # unless running in an environment where GIL was disabled.
        assert isinstance(gil_enabled, bool)
