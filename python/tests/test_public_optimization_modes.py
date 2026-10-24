"""Normal public state operations share semantics across all optimization modes."""


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
    {"mask_strategy": glrmask.TokenTrie(merge_equivalent_tokens=True)},
]


@pytest.mark.parametrize("mode", MODES)
def test_recursive_state_clone_force_eos_and_both_load_forms(mode):
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"(", 2: b")", 7: b"()", 15: b"(", 20: b"x", 25: b""})
    grammar = glrmask.Grammar.from_ebnf('start ::= "(" start ")" start | ""')
    reference = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([100]))).compile(vocab)
    compiled = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([100]))).compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        for prefix in [b"", b"(", b"()", b"()("]:
            state, expected = constraint.start(), reference.start()
            assert state.commit_bytes(prefix)
            assert expected.commit_bytes(prefix)
            mask = state.mask()
            assert np.array_equal(mask, expected.mask())
            assert state.forced_tokens() == expected.forced_tokens()
            assert state.is_accepting() == expected.is_accepting()
            assert not state.is_accepting()
            assert not mask[25]
            snapshot = state.copy()
            if mask[100]:
                assert state.commit_token(100)
                assert state.is_accepting()
            else:
                assert not state.commit_token(100)
                assert not state.is_accepting()
            assert not state.mask().any()
            assert np.array_equal(snapshot.mask(), mask)
            assert snapshot.is_accepting() == expected.is_accepting()


@pytest.mark.parametrize("mode", MODES)
def test_typed_source_bindings_and_precompiled_links_share_masks(mode):
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"x", 1: b"a", 2: b"b", 3: b"y", 4: b"xaby", 5: b"ab", 7: b"\t"})
    parent = glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar child; nt root = "x" child "y";')
    child = glrmask.Grammar.from_lark('start: "a" "b"?\n%ignore /\\t+/')
    reference = _bound_snapshot(glrmask.Module(parent), "child", glrmask.Module(child)).compile(vocab)
    configured_child = glrmask.Module(child)
    configured = _bound_snapshot(glrmask.Module(parent), "child", configured_child)
    compiled = configured.compile(vocab, **mode)
    module = _bound_snapshot(glrmask.Module(parent), "child", configured_child)
    module.precompile(vocab, **mode)
    linked = module.compile(vocab, **mode)
    for constraint in [compiled, linked, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        for prefix in [b"", b"x", b"xa", b"xab", b"xab\t", b"xaby"]:
            actual, expected = constraint.start(), reference.start()
            assert actual.commit_bytes(prefix)
            assert expected.commit_bytes(prefix)
            assert np.array_equal(actual.mask(), expected.mask())
            assert actual.is_accepting() == expected.is_accepting()
            assert (not actual.mask().any()) == (not expected.mask().any())


@pytest.mark.parametrize("mode", MODES)
def test_exact_token_identity_survives_source_compilation(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 15: b"a"})
    grammar = _bound_snapshot(glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token MARK; nt root = MARK;"), "MARK", glrmask.TokenSet([15]))
    compiled = grammar.compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert state.mask()[15]
        assert not state.mask()[0]
        assert state.commit_token(15)
        assert state.is_accepting()


def test_compressed_public_lark_masks_commit_clone_and_reload():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"aa"})
    source = "start: r0\n" + "".join(f'r{i}: "a" r{i+1}\n' for i in range(30)) + 'r30: "b"\n'
    grammar = glrmask.Grammar.from_lark(source)
    reference = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([100]))).compile(vocab, mask_strategy=glrmask.TokenTrie(parser_backend=glrmask.ParserBackend.TEMPLATE_DFA, merge_equivalent_tokens=True))
    compiled = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([100]))).compile(vocab, mask_strategy=glrmask.MaskStrategy.FAST_COMPILE)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        for n in [0, 1, 15, 29, 30]:
            state, expected = constraint.start(), reference.start()
            assert state.commit_bytes(b"a" * n)
            assert expected.commit_bytes(b"a" * n)
            assert np.array_equal(state.mask(), expected.mask())
            assert state.forced_tokens() == expected.forced_tokens()
            snapshot = state.copy()
            if n == 30:
                assert state.commit_token(1)
                assert not state.is_accepting()
                assert state.mask()[100]
                assert state.commit_token(100)
                assert state.is_accepting() and not state.mask().any()
                assert not state.mask().any()
            assert np.array_equal(snapshot.mask(), expected.mask())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", [29, 30, 63])
def test_whole_token_is_suggested_by_compressed_static_mask(mode, n):
    source = "start: r0\n" + "".join(f'r{i}: "a" r{i+1}\n' for i in range(n)) + f'r{n}: "b"\n'
    word = b"a" * n + b"b"
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"aa", 3: word, 7: word,
                                         8: b"a" * (n-1) + b"b", 9: b"a" * (n+1) + b"b", 11: b"x"})
    grammar = glrmask.Grammar.from_lark(source)
    reference = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([100]))).compile(vocab, mask_strategy=glrmask.MaskStrategy.FAST_COMPILE)
    compiled = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([100]))).compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        for prefix in [b"", b"a", b"a" * (n-1), b"a" * n, word]:
            state, expected = constraint.start(), reference.start()
            assert state.commit_bytes(prefix)
            assert expected.commit_bytes(prefix)
            assert np.array_equal(state.mask(), expected.mask())
            assert state.is_accepting() == expected.is_accepting()
            assert state.forced_tokens() == expected.forced_tokens()
            assert np.array_equal(state.copy().mask(), expected.mask())
        state = constraint.start()
        assert state.mask()[3] and state.mask()[7]
        assert not state.mask()[8] and not state.mask()[9]
        assert state.commit_token(7)
        assert not state.is_accepting()
        assert state.commit_token(100)
        assert state.is_accepting() and not state.mask().any()
        assert not state.mask().any()
