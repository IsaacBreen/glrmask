"""Normal public state operations share semantics across all optimization modes."""
import numpy as np
import pytest
import glrmask

MODES = [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
         glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME]


@pytest.mark.parametrize("mode", MODES)
def test_recursive_state_clone_force_eos_and_both_load_forms(mode):
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"(", 2: b")", 7: b"()", 15: b"(", 20: b"x", 25: b""})
    grammar = glrmask.Grammar.from_ebnf('start ::= "(" start ")" start | ""')
    reference = grammar.compile(vocab, end_tokens=[100])
    compiled = grammar.compile(vocab, optimization=mode, end_tokens=[100])
    for constraint in [compiled, glrmask.Constraint.load(compiled.save()),
                       glrmask.Constraint.load(compiled.save(external_vocab=True), vocab)]:
        for prefix in [b"", b"(", b"()", b"()("]:
            state, expected = constraint.start(), reference.start()
            state.commit_bytes(prefix)
            expected.commit_bytes(prefix)
            mask = state.mask()
            assert np.array_equal(mask, expected.mask())
            assert state.forced() == expected.forced()
            assert state.is_accepting() == expected.is_accepting()
            assert bool(mask[100]) == state.is_accepting()
            assert not mask[25]
            snapshot = state.clone()
            state.commit_token(100)
            assert state.is_terminated() == expected.is_accepting()
            assert state.is_rejected() != expected.is_accepting()
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
    description = parent.bind("child", child)
    reference = description.compile(vocab)
    compiled = description.compile(vocab, optimization=mode)
    module = parent.compile_unlinked(vocab).bind("child", child.compile(vocab))
    linked = module.link(optimization=mode)
    for constraint in [compiled, linked, glrmask.Constraint.load(compiled.save()),
                       glrmask.Constraint.load(compiled.save(external_vocab=True), vocab)]:
        for prefix in [b"", b"x", b"xa", b"xab", b"xab\t", b"xaby"]:
            actual, expected = constraint.start(), reference.start()
            actual.commit_bytes(prefix)
            expected.commit_bytes(prefix)
            assert np.array_equal(actual.mask(), expected.mask())
            assert actual.is_accepting() == expected.is_accepting()
            assert actual.is_rejected() == expected.is_rejected()


@pytest.mark.parametrize("mode", MODES)
def test_exact_token_identity_survives_source_compilation(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 15: b"a"})
    grammar = glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token MARK; nt root = MARK;").bind("MARK", vocab.token(15))
    compiled = grammar.compile(vocab, optimization=mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save()),
                       glrmask.Constraint.load(compiled.save(external_vocab=True), vocab)]:
        state = constraint.start()
        assert state.mask()[15]
        assert not state.mask()[0]
        state.commit_token(15)
        assert state.is_accepting()
