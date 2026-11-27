"""The released namespace exposes intent and persistence, not parser machinery."""


def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, value) is None
    return m

import inspect

import numpy as np
import pytest

import glrmask

from mask_helpers import token_allowed, unpack_mask

MODES = [
    {},
    {"mask_strategy": glrmask.MaskStrategy.FAST_MASKS},
    {"mask_strategy": glrmask.MaskStrategy.FAST_COMPILE},
]


def test_parser_backend_selection_is_not_public():
    assert not hasattr(glrmask, "ParserBackend")
    assert not hasattr(glrmask, "TokenTrie")
    assert not hasattr(glrmask, "ParserProgram")
    assert not hasattr(glrmask._glrmask, "ParserProgram")
    assert hasattr(glrmask._internal, "ParserProgram")
    assert "mask_strategy" in inspect.signature(glrmask.Grammar.compile).parameters
    assert "parser_backend" not in inspect.signature(glrmask.Grammar.compile).parameters
    assert "merge_equivalent_tokens" not in inspect.signature(glrmask.Grammar.compile).parameters
    assert not hasattr(glrmask.Constraint, "parser_backend")
    vocab = glrmask.Vocab({0: b"a", 1: b"ab", 2: b"b"})
    grammar = glrmask.Grammar.from_lark('start: "a" "b"?')
    public = grammar.compile(
        vocab,
        mask_strategy=glrmask.MaskStrategy.FAST_MASKS,
    )
    internal = glrmask._internal.compile_with_backend(
        grammar, vocab,
        mask_strategy=glrmask.MaskStrategy.FAST_MASKS,
    )
    assert np.array_equal(public.start().mask(), internal.start().mask())


@pytest.mark.parametrize("mode", MODES)
def test_save_flag_preserves_default_and_external_roundtrips(mode):
    vocab = glrmask.Vocab({0: b"a", 1: b"ab", 2: b"b", 3: b""})
    constraint = _module_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a" "b"?'), glrmask.TokenSet([3]))).compile(vocab, **mode)
    assert not hasattr(constraint, "save_with_external_vocab")
    self_contained = constraint.save()
    assert self_contained == constraint.save()
    external = constraint.save()
    assert external == constraint.save()
    for restored in [glrmask.Constraint.load(self_contained, vocab),
                     glrmask.Constraint.load(external, vocab)]:
        for prefix in [b"", b"a", b"ab"]:
            expected = constraint.start()
            actual = restored.start()
            assert expected.commit_bytes(prefix)
            assert actual.commit_bytes(prefix)
            assert np.array_equal(expected.mask(), actual.mask())
            assert expected.is_accepting() == actual.is_accepting()
            if token_allowed(actual.mask(), 3):
                assert not actual.is_accepting()
                assert actual.commit_token(3)
                assert actual.is_accepting() and not actual.mask().any()
                assert not actual.mask().any()
    wrong_vocab = glrmask.Vocab({0: b"b", 1: b"ab", 2: b"a", 3: b""})
    with pytest.raises(glrmask.GLRMaskError) as exc:
        glrmask.Constraint.load(external, wrong_vocab)
    assert exc.value.kind == "vocab_mismatch"
    with pytest.raises(TypeError):
        constraint.save(True)

@pytest.mark.parametrize("mode", MODES)
def test_composed_grammar_with_automatic_boundaries(mode):
    vocab = glrmask.Vocab({0: b"x", 1: b"a", 2: b"y", 3: b"xay"})
    grammar = glrmask.Grammar.from_glrm(
        'glrm 1; start start; extern grammar child; nt start = "x" child "y";')
    child = glrmask.Module(glrmask.Grammar.from_lark('start: "a"'))
    module = glrmask.Module(grammar)
    with pytest.raises(glrmask.GLRMaskError, match="unbound"):
        module.compile(vocab, **mode)
    with pytest.raises(TypeError):
        module.compile(vocab, boundary_mask_strategy="EXACT")
    restored = glrmask.Module.load(module.save())
    restored.bind("child", child)
    requested = restored.compile(vocab, **mode)
    assert token_allowed(requested.start().mask(), 3)


@pytest.mark.parametrize("mode", MODES)
def test_acceptance_allows_continuation_and_caller_tracks_eos(mode):
    assert hasattr(glrmask.ConstraintState, "is_terminated")
    assert hasattr(glrmask.ConstraintState, "is_accepting")
    vocab = glrmask.Vocab({0: b"a", 1: b"b"})
    end_token_ids = {130}
    compiled = _module_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a" "b"?'), glrmask.TokenSet(list(end_token_ids)))).compile(vocab, **mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert hasattr(state, "is_terminated")
        assert not state.is_accepting() and not (not state.mask().any())
        assert not token_allowed(state.mask(), 130)
        assert state.commit_token(0)
        # The grammar is complete; acceptance needs the end token.
        assert not state.is_accepting() and not (not state.mask().any())
        assert token_allowed(state.mask(), 1) and token_allowed(state.mask(), 130)
        for sequence in [(130,), (1, 130)]:
            branch = state.copy()
            for token_id in sequence:
                assert token_allowed(branch.mask(), token_id)
                assert branch.commit_token(token_id)
                if token_id in end_token_ids:
                    break
            for ended in [branch, branch.copy()]:
                assert ended.is_accepting() and not ended.mask().any()
                assert not ended.mask().any()
                assert ended.forced_bytes() == b""
                assert not ended.commit_token(0)
                assert not ended.commit_bytes(b"a")
                assert not ended.is_accepting() and not ended.mask().any()
            assert not state.is_accepting() and token_allowed(state.mask(), 1) and token_allowed(state.mask(), 130)


def _module_expr(value):
    module = glrmask.Module(glrmask.Grammar.from_glrm("extern grammar VALUE start root nt root = VALUE"))
    module.bind("VALUE", value)
    return module
