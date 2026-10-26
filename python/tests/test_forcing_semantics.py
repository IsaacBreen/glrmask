"""Canonical forcing, aliases, exact events and acceptance barriers."""


def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, value) is None
    return m

import numpy as np
import pytest
import glrmask

MODES = [
    {},
    {"mask_strategy": glrmask.MaskStrategy.PARSER_DWA},
    {"mask_strategy": glrmask.MaskStrategy.TOKEN_TRIE},
    {"mask_strategy": glrmask.TokenTrie(parser_backend=glrmask.ParserBackend.TEMPLATE_DFA)},
    {"mask_strategy": glrmask.TokenTrie(parser_backend=glrmask.ParserBackend.TEMPLATE_DFA, merge_equivalent_tokens=True)},
]


def assert_read_only_and_replay(state, method):
    before = state.mask().copy()
    accepted = state.is_accepting()
    result = getattr(state, method)()
    assert np.array_equal(state.mask(), before)
    assert state.is_accepting() == accepted
    cursor = state.copy()
    for token in result:
        assert not cursor.is_accepting()
        assert cursor.mask()[token]
        assert cursor.commit_token(token)
        assert cursor.is_accepting() or cursor.mask().any()
    return result


@pytest.mark.parametrize("mode", MODES)
def test_forcing_uses_longest_spelling_then_lowest_alias_id(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"ab", 3:b"ab"})
    compiled = glrmask.Grammar.from_lark('start: "ab"').compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert not hasattr(state, "forced")
        assert not hasattr(state, "forced_token_ids")
        assert assert_read_only_and_replay(state,"forced_tokens") == [2]
        assert state.commit_token(0)
        assert assert_read_only_and_replay(state,"forced_tokens") == [1]


@pytest.mark.parametrize("mode", MODES)
def test_incomplete_vocabulary_does_not_hide_byte_language_ambiguity(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"ab"})
    for source, expected in [('start: "ab"', [1]), ('start: "ab" | "ac"', [])]:
        compiled = glrmask.Grammar.from_lark(source).compile(vocab, **mode)
        for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab)]:
            state = constraint.start()
            assert np.flatnonzero(state.mask()).tolist() == [0, 1]
            assert assert_read_only_and_replay(state,"forced_tokens") == expected


@pytest.mark.parametrize("mode", MODES)
def test_spellingless_event_is_not_erased_from_byte_proof(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"ab", 9:b""})
    grammar = _bound_snapshot(glrmask.Grammar.from_glrm(
        'extern token MARK; start root; nt root = "ab" | MARK;'
    ), "MARK", glrmask.TokenSet([9]))
    state = grammar.compile(vocab, **mode).start()
    assert state.mask()[9]
    assert not hasattr(state, "forced")
    assert assert_read_only_and_replay(state,"forced_tokens") == []


@pytest.mark.parametrize("mode", MODES)
def test_force_stops_at_acceptance_not_optional_continuation(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0:b"a", 1:b"b", 2:b"c"})
    state = glrmask.Grammar.from_lark('start: "ab" "c"?').compile(vocab,**mode).start()
    assert not hasattr(state, "forced")
    assert assert_read_only_and_replay(state,"forced_tokens") == [0,1]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("event_bytes", [b"", b"<mark>"])
def test_shorter_token_boundary_preserves_exact_event_opportunity(mode, event_bytes):
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0:b"a", 1:b"b", 2:b"ab", 3:b"c", 9:event_bytes})
    grammar = _bound_snapshot(glrmask.Grammar.from_glrm(
        'extern token MARK; start root; nt root = "ab" | "a" MARK "c";'
    ), "MARK", glrmask.TokenSet([9]))
    compiled = grammar.compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert state.mask()[0] and state.mask()[2]
        assert assert_read_only_and_replay(state, "forced_tokens") == []
        assert state.commit_token(0)
        assert state.mask()[1] and state.mask()[9]
        assert assert_read_only_and_replay(state, "forced_tokens") == []
        assert state.commit_token(9)
        assert assert_read_only_and_replay(state, "forced_tokens") == [3]


@pytest.mark.parametrize("mode", MODES)
def test_long_token_cannot_cross_possible_root_acceptance(mode):
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0:b"a", 1:b"b", 2:b"ab", 3:b"abc", 9:b""})
    compiled = glrmask.Grammar.from_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a" | "ab"'), glrmask.TokenSet([9]))).compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert assert_read_only_and_replay(state, "forced_tokens") == []
        assert state.commit_token(0)
        # With end tokens configured, completion alone is not acceptance.
        assert not state.is_accepting() and state.mask()[1] and state.mask()[9]
        assert assert_read_only_and_replay(state, "forced_tokens") == []


@pytest.mark.parametrize("mode", MODES)
def test_dual_byte_event_id_keeps_accepting_and_nonaccepting_successors(mode):
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0:b"a", 1:b"b", 2:b"ab", 3:b"c", 9:b""})
    grammar = _bound_snapshot(glrmask.Grammar.from_glrm(
        'extern token MARK; start root; nt root = "ab" | "a" MARK "c";'
    ), "MARK", glrmask.TokenSet([1]))
    compiled = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([9]))).compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert assert_read_only_and_replay(state, "forced_tokens") == []
        assert state.commit_token(0)
        assert assert_read_only_and_replay(state, "forced_tokens") == [1]
        assert state.commit_token(1)
        assert not state.is_accepting() and state.mask()[3] and state.mask()[9]
        assert assert_read_only_and_replay(state, "forced_tokens") == []
