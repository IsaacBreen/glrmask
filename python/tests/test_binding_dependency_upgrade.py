"""Behavior regressions for the PyO3/rust-numpy dependency migration."""
import importlib

import glrmask
import numpy as np
import pytest


@pytest.mark.parametrize("optimization", [
    glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
    glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME,
])
def test_mask_buffer_alignment_and_validation(optimization):
    vocab = glrmask.Vocab.from_dict({b"a": 0, b"b": 33})
    state = glrmask.Grammar.from_ebnf('start ::= "a"').compile(
        vocab, optimization=optimization,
    ).start()
    words = np.full(2, -1, dtype=np.int32)
    state.fill_mask(words)
    assert words.tolist() == [1, 0]
    assert state.mask().tolist() == [True] + [False] * 33

    unaligned = np.ndarray((2,), dtype=np.int32, buffer=bytearray(9), offset=1)
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
    with pytest.raises(ValueError, match="contiguous and aligned"):
        state.fill_mask(unaligned)
    assert unaligned.tolist() == [0, 0]
    with pytest.raises(ValueError, match="contiguous and aligned"):
        state.fill_mask(np.zeros(4, dtype=np.int32)[::2])
    with pytest.raises(TypeError):
        state.fill_mask(np.zeros(2, dtype=np.int64))
    readonly = np.zeros(2, dtype=np.int32)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        state.fill_mask(readonly)
    state.fill_mask(words)  # Failed borrows must not leave the array API poisoned.
    assert words.tolist() == [1, 0]


def test_extension_reimport_keeps_public_classes_and_internal_module():
    extension = importlib.import_module("glrmask._glrmask")
    assert importlib.reload(extension) is extension
    assert extension.Grammar is glrmask.Grammar
    assert extension._internal is glrmask._internal
    assert "ParserProgram" not in extension.__all__
    vocab = glrmask.Vocab.from_dict({b"a": 0})
    with pytest.raises(ValueError, match="vocab keys must be Python bytes"):
        glrmask.Vocab.from_dict({"a": 0})
    with pytest.raises(TypeError):
        glrmask.Grammar.from_ebnf('start ::= "a"').compile(vocab, optimization="AUTO")
