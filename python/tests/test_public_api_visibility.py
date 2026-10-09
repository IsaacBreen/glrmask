"""The released namespace exposes intent and persistence, not parser machinery."""
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
    assert "parser_backend" not in inspect.signature(glrmask.UnlinkedConstraint.link).parameters
    assert not hasattr(glrmask.Constraint, "parser_backend")
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b"})
    grammar = glrmask.Grammar.from_ebnf('start ::= "a" "b"?')
    with pytest.raises(TypeError, match="parser_backend"):
        grammar.compile(vocab, parser_backend=glrmask._internal.ParserBackend.TEMPLATE_DFA)
    module = grammar.compile_unlinked(vocab)
    with pytest.raises(TypeError, match="parser_backend"):
        module.link(parser_backend=glrmask._internal.ParserBackend.TEMPLATE_DFA)
    public = grammar.compile(vocab, optimization=glrmask.Optimization.FAST_BUILD)
    internal = glrmask._internal.compile_with_backend(
        grammar, vocab, optimization=glrmask.Optimization.BALANCED,
        parser_backend=glrmask._internal.ParserBackend.TEMPLATE_DFA,
    )
    assert np.array_equal(public.start().mask(), internal.start().mask())


@pytest.mark.parametrize("mode", [glrmask.Optimization.AUTO,
                                  glrmask.Optimization.FAST_BUILD,
                                  glrmask.Optimization.BALANCED,
                                  glrmask.Optimization.FAST_RUNTIME])
def test_save_flag_preserves_default_and_external_roundtrips(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b", 3: b""})
    constraint = glrmask.Grammar.from_ebnf('start ::= "a" "b"?').compile(
        vocab, optimization=mode, end_tokens=[3])
    assert not hasattr(constraint, "save_with_external_vocab")
    self_contained = constraint.save()
    assert self_contained == constraint.save(external_vocab=False)
    external = constraint.save(external_vocab=True)
    assert external == constraint.save(external_vocab=True)
    for restored in [glrmask.Constraint.load(self_contained),
                     glrmask.Constraint.load(external, vocab)]:
        for prefix in [b"", b"a", b"ab"]:
            expected = constraint.start()
            actual = restored.start()
            expected.commit_bytes(prefix)
            actual.commit_bytes(prefix)
            assert np.array_equal(expected.mask(), actual.mask())
            assert expected.is_accepting() == actual.is_accepting()
            if actual.is_accepting():
                actual.commit_token(3)
                assert actual.is_accepting() and not actual.is_rejected()
                assert not actual.mask().any()
    wrong_vocab = glrmask.Vocab.from_id_to_bytes({0: b"b", 1: b"ab", 2: b"a", 3: b""})
    with pytest.raises(ValueError):
        glrmask.Constraint.load(external, wrong_vocab)
    with pytest.raises(TypeError):
        constraint.save(True)

@pytest.mark.parametrize("mode", [glrmask.Optimization.AUTO,
                                  glrmask.Optimization.FAST_BUILD,
                                  glrmask.Optimization.BALANCED,
                                  glrmask.Optimization.FAST_RUNTIME])
def test_boundary_option_is_eager_and_does_not_change_masks(mode):
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"ab", 2: b"b"})
    grammar = glrmask.Grammar.from_ebnf('start ::= "a" "b"?')
    plain = grammar.compile(vocab, optimization=mode)
    assert grammar.compile(vocab, optimization=mode,
        boundary_trigger=glrmask.BoundaryTriggerDetail.NONE).save() == plain.save()
    for detail in [glrmask.BoundaryTriggerDetail.TOKENS, glrmask.BoundaryTriggerDetail.EXACT]:
        if mode == glrmask.Optimization.FAST_BUILD and detail == glrmask.BoundaryTriggerDetail.EXACT:
            with pytest.raises(ValueError, match="retained-LR Dynamic"):
                grammar.compile(vocab, optimization=mode, boundary_trigger=detail)
            continue
        requested = grammar.compile(vocab, optimization=mode, boundary_trigger=detail)
        saved = requested.save()
        assert saved != plain.save()
        for prefix in [b"", b"a", b"ab"]:
            a, b = requested.start(), plain.start()
            a.commit_bytes(prefix)
            b.commit_bytes(prefix)
            assert np.array_equal(a.mask(), b.mask())
            assert a.is_accepting() == b.is_accepting()
        assert requested.save() == saved
    with pytest.raises(TypeError):
        grammar.compile(vocab, boundary_trigger="EXACT")


def test_boundary_request_survives_module_save_bind_and_explicit_override():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"a", 2: b"y", 3: b"xay"})
    grammar = glrmask.Grammar.from_glrm(
        'glrm 1; start start; extern grammar child; nt start = "x" child "y";')
    child = glrmask.Grammar.from_ebnf('start ::= "a"').compile(vocab)
    plain = glrmask.UnlinkedConstraint.load(grammar.compile_unlinked(vocab).save()).bind("child", child).link()
    for detail in [glrmask.BoundaryTriggerDetail.TOKENS, glrmask.BoundaryTriggerDetail.EXACT]:
        module = grammar.compile_unlinked(vocab, boundary_trigger=detail)
        with pytest.raises(ValueError, match="unbound"):
            module.link()
        restored = glrmask.UnlinkedConstraint.load(module.save())
        bound = restored.bind("child", child)
        requested = bound.link()
        assert requested.save() != plain.save()
        assert bound.link(boundary_trigger=glrmask.BoundaryTriggerDetail.NONE).save() == plain.save()
        reloaded = glrmask.UnlinkedConstraint.load(bound.save()).link()
        assert reloaded.save() != bound.link(boundary_trigger=glrmask.BoundaryTriggerDetail.NONE).save()
        assert np.array_equal(reloaded.start().mask(), requested.start().mask())
        assert np.array_equal(requested.start().mask(), plain.start().mask())


@pytest.mark.parametrize("mode", [glrmask.Optimization.AUTO,
                                  glrmask.Optimization.FAST_BUILD,
                                  glrmask.Optimization.BALANCED,
                                  glrmask.Optimization.FAST_RUNTIME])
def test_acceptance_allows_continuation_and_caller_tracks_eos(mode):
    assert not hasattr(glrmask.ConstraintState, "is_terminated")
    assert hasattr(glrmask.ConstraintState, "is_accepting")
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b"})
    end_token_ids = {130}
    compiled = glrmask.Grammar.from_ebnf('start ::= "a" "b"?').compile(
        vocab, optimization=mode, end_tokens=list(end_token_ids))
    for constraint in [compiled, glrmask.Constraint.load(compiled.save()),
                       glrmask.Constraint.load(compiled.save(external_vocab=True), vocab)]:
        state = constraint.start()
        assert not hasattr(state, "is_terminated")
        assert not state.is_accepting() and not state.is_rejected()
        assert not state.mask()[130]
        state.commit_token(0)
        assert state.is_accepting() and not state.is_rejected()
        assert state.mask()[1] and state.mask()[130]
        for sequence in [(130,), (1, 130)]:
            branch = state.clone()
            for token_id in sequence:
                assert branch.mask()[token_id]
                branch.commit_token(token_id)
                if token_id in end_token_ids:
                    break
            for ended in [branch, branch.clone()]:
                assert ended.is_accepting() and not ended.is_rejected()
                assert not ended.mask().any()
                assert ended.forced() == []
                with pytest.raises(ValueError, match="already terminated"):
                    ended.commit_token(0)
                with pytest.raises(ValueError, match="already terminated"):
                    ended.commit_bytes(b"a")
                assert ended.is_accepting() and not ended.is_rejected()
            assert state.is_accepting() and state.mask()[1] and state.mask()[130]
