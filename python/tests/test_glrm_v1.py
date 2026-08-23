import pytest
import glrmask


def _vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b"})


def _grammar():
    return '''
    glrm 1;
    start start;
    extern t END;
    nt start = "a" END;
    '''


@pytest.mark.parametrize("constraint_type", [glrmask.Constraint, glrmask.DynamicConstraint])
def test_named_external_terminal_binding(constraint_type):
    constraint = constraint_type.from_glrm_grammar(
        _grammar(), _vocab(), bindings={"END": [77, 78]}
    )
    state = constraint.start()
    state.commit_token(0)
    mask = state.mask(79)
    assert mask[77] and mask[78]
    state.commit_token(77)
    assert state.is_accepting()


@pytest.mark.parametrize("bad_bindings", [{}, {"END": []}, {"OTHER": 77}, {"END": [-1]}])
def test_named_external_terminal_binding_errors(bad_bindings):
    with pytest.raises((ValueError, RuntimeError)):
        glrmask.Constraint.from_glrm_grammar(_grammar(), _vocab(), bindings=bad_bindings)
