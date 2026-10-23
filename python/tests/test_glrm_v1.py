

def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, value) is None
    return m

import pytest
import glrmask
from glrmask._glrmask import DynamicConstraint as LegacyDynamicConstraint


def _vocab():
    return glrmask.Vocab.from_id_to_bytes({
        0: b"a",
        1: b"b",
        77: b"<END77>",
        78: b"<END78>",
    })


def _grammar():
    return '''
    glrm 1;
    start start;
    extern token END;
    nt start = "a" END;
    '''


def _final_constraint():
    vocab = _vocab()
    return (
        _bound_snapshot(glrmask.Grammar.from_glrm(_grammar()), "END", glrmask.TokenSet([77, 78]))
        .compile(vocab)
    )


def _legacy_dynamic_constraint():
    return LegacyDynamicConstraint.from_glrm_grammar(
        _grammar(), _vocab(), bindings={"END": [77, 78]}
    )


@pytest.mark.parametrize("factory", [_final_constraint, _legacy_dynamic_constraint])
def test_named_external_terminal_binding(factory):
    constraint = factory()
    state = constraint.start()
    assert state.commit_token(0)
    mask = state.mask(79)
    assert mask[77] and mask[78]
    assert state.commit_token(77)
    assert state.is_accepting()


@pytest.mark.parametrize("bad_bindings", [{}, {"OTHER": 77}, {"END": [-1]}])
def test_named_external_terminal_binding_errors(bad_bindings):
    vocab = _vocab()
    grammar = glrmask.Grammar.from_glrm(_grammar())
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        for name, value in bad_bindings.items():
            ids = value if isinstance(value, list) else [value]
            grammar = _bound_snapshot(grammar, name, glrmask.TokenSet(ids))
        grammar.compile(vocab)


def test_empty_event_binding_recognizes_no_output():
    vocab = _vocab()
    for mode in (glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_BUILD,
                 glrmask.MaskStrategy.TOKEN_CLASS_TRIE, glrmask.MaskStrategy.FAST_RUNTIME):
        grammar = _bound_snapshot(glrmask.Grammar.from_glrm(_grammar()), "END", glrmask.TokenSet([]))
        constraint = grammar.compile(vocab, mask_strategy=mode)
        state = constraint.start()
        assert not state.is_accepting()
        # The preceding literal can be a locally viable parser prefix. Empty
        # event sets promise no successful event/complete match, not a new global
        # dead-production elimination policy. Check the actual event boundary.
        prefix_mask = state.mask(79)
        assert not prefix_mask[1:].any()
        if prefix_mask[0]:
            assert state.commit_token(0)
        assert not state.is_accepting()
        assert not state.mask(79).any()
        assert not state.forced_tokens()
