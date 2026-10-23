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


def test_parser_machinery_is_only_available_in_internal_namespace():
    for name in ["ParserBackend", "ParserProgram"]:
        assert not hasattr(glrmask, name)
        assert not hasattr(glrmask._glrmask, name)
        assert hasattr(glrmask._internal, name)
    assert "parser_backend" not in inspect.signature(glrmask.Grammar.compile).parameters
    assert not hasattr(glrmask.Constraint, "parser_backend")
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b"})
    grammar = glrmask.Grammar.from_ebnf('start ::= "a" "b"?')
    with pytest.raises(TypeError, match="parser_backend"):
        grammar.compile(vocab, parser_backend=glrmask._internal.ParserBackend.TEMPLATE_DFA)
    public = grammar.compile(vocab, mask_strategy=glrmask.MaskStrategy.FAST_BUILD)
    internal = glrmask._internal.compile_with_backend(
        grammar, vocab, mask_strategy=glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
        parser_backend=glrmask._internal.ParserBackend.TEMPLATE_DFA,
    )
    assert np.array_equal(public.start().mask(), internal.start().mask())


@pytest.mark.parametrize("mode", [glrmask.MaskStrategy.AUTO,
                                  glrmask.MaskStrategy.FAST_BUILD,
                                  glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
                                  glrmask.MaskStrategy.FAST_RUNTIME])
def test_save_flag_preserves_default_and_external_roundtrips(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b", 3: b""})
    constraint = glrmask.Grammar.from_expr(glrmask.Expr.sequence(glrmask.Grammar.from_ebnf('start ::= "a" "b"?'), glrmask.TokenSet([3]))).compile(vocab, mask_strategy=mode)
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
            if actual.mask()[3]:
                assert not actual.is_accepting()
                assert actual.commit_token(3)
                assert actual.is_accepting() and not actual.mask().any()
                assert not actual.mask().any()
    wrong_vocab = glrmask.Vocab.from_id_to_bytes({0: b"b", 1: b"ab", 2: b"a", 3: b""})
    with pytest.raises(ValueError):
        glrmask.Constraint.load(external, wrong_vocab)
    with pytest.raises(TypeError):
        constraint.save(True)

@pytest.mark.parametrize("mode", [glrmask.MaskStrategy.AUTO,
                                  glrmask.MaskStrategy.FAST_BUILD,
                                  glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
                                  glrmask.MaskStrategy.FAST_RUNTIME])
def test_boundary_option_is_eager_and_does_not_change_masks(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b"})
    grammar = glrmask.Grammar.from_ebnf('start ::= "a" "b"?')
    plain = grammar.compile(vocab, mask_strategy=mode)
    assert grammar.compile(vocab, mask_strategy=mode,
        boundary_mask_strategy=glrmask.BoundaryMaskStrategy.FAST_BUILD).save() == plain.save()
    for detail in [glrmask.BoundaryMaskStrategy.FAST_BUILD, glrmask.BoundaryMaskStrategy.FAST_RUNTIME]:
        requested = grammar.compile(vocab, mask_strategy=mode, boundary_mask_strategy=detail)
        saved = requested.save()
        for prefix in [b"", b"a", b"ab"]:
            a, b = requested.start(), plain.start()
            assert a.commit_bytes(prefix)
            assert b.commit_bytes(prefix)
            assert np.array_equal(a.mask(), b.mask())
            assert a.is_accepting() == b.is_accepting()
        assert requested.save() == saved
    with pytest.raises(TypeError):
        grammar.compile(vocab, boundary_mask_strategy="EXACT")


def test_boundary_request_applies_to_compile_after_save_and_bind():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"a", 2: b"y", 3: b"xay"})
    grammar = glrmask.Grammar.from_glrm(
        'glrm 1; start start; extern grammar child; nt start = "x" child "y";')
    child = glrmask.Module(glrmask.Grammar.from_ebnf('start ::= "a"'))
    for detail in [glrmask.BoundaryMaskStrategy.FAST_BUILD, glrmask.BoundaryMaskStrategy.FAST_RUNTIME]:
        module = glrmask.Module(grammar)
        with pytest.raises(ValueError, match="unbound"):
            module.compile(vocab, boundary_mask_strategy=detail)
        restored = glrmask.Module.load(module.save())
        restored.bind("child", child)
        requested = restored.compile(vocab, boundary_mask_strategy=detail)
        assert requested.start().mask()[3]


@pytest.mark.parametrize("mode", [glrmask.MaskStrategy.AUTO,
                                  glrmask.MaskStrategy.FAST_BUILD,
                                  glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
                                  glrmask.MaskStrategy.FAST_RUNTIME])
def test_acceptance_allows_continuation_and_caller_tracks_eos(mode):
    assert hasattr(glrmask.ConstraintState, "is_terminated")
    assert hasattr(glrmask.ConstraintState, "is_accepting")
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b"})
    end_token_ids = {130}
    compiled = glrmask.Grammar.from_expr(glrmask.Expr.sequence(glrmask.Grammar.from_ebnf('start ::= "a" "b"?'), glrmask.TokenSet(list(end_token_ids)))).compile(vocab, mask_strategy=mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save(), vocab),
                       glrmask.Constraint.load(compiled.save(), vocab)]:
        state = constraint.start()
        assert hasattr(state, "is_terminated")
        assert not state.is_accepting() and not (not state.mask().any())
        assert not state.mask()[130]
        assert state.commit_token(0)
        # The grammar is complete; acceptance needs the end token.
        assert not state.is_accepting() and not (not state.mask().any())
        assert state.mask()[1] and state.mask()[130]
        for sequence in [(130,), (1, 130)]:
            branch = state.copy()
            for token_id in sequence:
                assert branch.mask()[token_id]
                assert branch.commit_token(token_id)
                if token_id in end_token_ids:
                    break
            for ended in [branch, branch.copy()]:
                assert ended.is_accepting() and not ended.mask().any()
                assert not ended.mask().any()
                assert ended.forced_tokens() == []
                assert not ended.commit_token(0)
                assert not ended.commit_bytes(b"a")
                assert not ended.is_accepting() and not ended.mask().any()
            assert not state.is_accepting() and state.mask()[1] and state.mask()[130]
