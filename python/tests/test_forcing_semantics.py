"""Canonical text vs inevitable original-ID forcing, including exact events."""
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
def test_canonical_and_original_id_contracts_are_distinct(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"ab", 3:b"ab"})
    compiled = glrmask.Grammar.from_ebnf('start ::= "ab"').compile(vocab, optimization=mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save())]:
        state = constraint.start()
        assert assert_read_only_and_replay(state,"forced") == [2]
        assert assert_read_only_and_replay(state,"forced_token_ids") == []
        state.commit_token(0)
        assert assert_read_only_and_replay(state,"forced_token_ids") == [1]


@pytest.mark.parametrize("mode", MODES)
def test_spellingless_event_is_not_erased_from_byte_proof(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"ab", 9:b""})
    grammar = glrmask.Grammar.from_glrm(
        'extern token MARK; start root; nt root = "ab" | MARK;'
    ).bind("MARK", vocab.tokens([9]))
    state = grammar.compile(vocab, optimization=mode).start()
    assert state.mask()[9]
    assert assert_read_only_and_replay(state,"forced") == []
    assert assert_read_only_and_replay(state,"forced_token_ids") == []


@pytest.mark.parametrize("mode", MODES)
def test_force_stops_at_acceptance_not_optional_continuation(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"c"})
    state = glrmask.Grammar.from_ebnf('start ::= "ab" "c"?').compile(vocab,optimization=mode).start()
    assert assert_read_only_and_replay(state,"forced") == [0,1]
    assert assert_read_only_and_replay(state,"forced_token_ids") == [0,1]
