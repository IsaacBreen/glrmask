"""Strict original-ID forcing, aliases, exact events and acceptance barriers."""


def _bound_snapshot(grammar, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    result = grammar.clone()
    assert result.bind(name, value) is None
    return result

import numpy as np
import pytest
import glrmask

MODES = [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
         glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME]


def assert_read_only_and_replay(state, method):
    before = state.mask().copy()
    accepted = state.is_accepting()
    result = getattr(state, method)()
    assert np.array_equal(state.mask(), before)
    assert state.is_accepting() == accepted
    cursor = state.clone()
    for token in result:
        assert not cursor.is_accepting()
        assert cursor.mask()[token]
        cursor.commit_token(token)
        assert not cursor.is_rejected()
    return result


@pytest.mark.parametrize("mode", MODES)
def test_forcing_does_not_choose_between_equivalent_tokenizations(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"ab", 3:b"ab"})
    compiled = glrmask.Grammar.from_ebnf('start ::= "ab"').compile(vocab, optimization=mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save())]:
        state = constraint.start()
        assert not hasattr(state, "forced")
        assert not hasattr(state, "forced_token_ids")
        assert assert_read_only_and_replay(state,"forced_tokens") == []
        state.commit_token(0)
        assert assert_read_only_and_replay(state,"forced_tokens") == [1]


@pytest.mark.parametrize("mode", MODES)
def test_spellingless_event_is_not_erased_from_byte_proof(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"ab", 9:b""})
    grammar = _bound_snapshot(glrmask.Grammar.from_glrm(
        'extern token MARK; start root; nt root = "ab" | MARK;'
    ), "MARK", vocab.tokens([9]))
    state = grammar.compile(vocab, optimization=mode).start()
    assert state.mask()[9]
    assert not hasattr(state, "forced")
    assert assert_read_only_and_replay(state,"forced_tokens") == []


@pytest.mark.parametrize("mode", MODES)
def test_force_stops_at_acceptance_not_optional_continuation(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"c"})
    state = glrmask.Grammar.from_ebnf('start ::= "ab" "c"?').compile(vocab,optimization=mode).start()
    assert not hasattr(state, "forced")
    assert assert_read_only_and_replay(state,"forced_tokens") == [0,1]
