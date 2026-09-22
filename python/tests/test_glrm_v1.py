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
        glrmask.Grammar.from_glrm(_grammar())
        .bind("END", vocab.tokens([77, 78]))
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
    state.commit_token(0)
    mask = state.mask(79)
    assert mask[77] and mask[78]
    state.commit_token(77)
    assert state.is_accepting()


@pytest.mark.parametrize("bad_bindings", [{}, {"END": []}, {"OTHER": 77}, {"END": [-1]}])
def test_named_external_terminal_binding_errors(bad_bindings):
    vocab = _vocab()
    grammar = glrmask.Grammar.from_glrm(_grammar())
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        for name, value in bad_bindings.items():
            ids = value if isinstance(value, list) else [value]
            grammar = grammar.bind(name, vocab.tokens(ids))
        grammar.compile(vocab)
