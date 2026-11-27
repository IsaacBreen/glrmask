import glrmask
from mask_helpers import token_allowed, unpack_mask

import pytest


_BAD_IDS = (-1, 2**32, 2**40)


def _vocab():
    return glrmask.Vocab({0: b"a", 1: b"b"})


def _constraint():
    return glrmask.Grammar.from_lark('start: "a"').compile(_vocab())


@pytest.mark.parametrize("bad", _BAD_IDS)
def test_out_of_range_token_id_leaves_state_unchanged(bad):
    state = _constraint().start()
    before = state.mask().copy()
    with pytest.raises(ValueError):
        state.commit_token(bad)
    assert (state.mask() == before).all()
    assert state.is_accepting() is False
    assert state.is_terminated() is False
    assert state.commit_token(0) is True




def test_non_integer_token_id_is_a_type_error():
    state = _constraint().start()
    for bad in ("0", True, False, 1.5):
        with pytest.raises(TypeError):
            state.commit_token(bad)


def test_special_tokens_are_explicit():
    for atom in ("<[1]>", "<[1,3-8]>", "<[]>", "<[^0]>", "<[*]>", "@token(7)"):
        with pytest.raises(glrmask.GLRMaskError, match="token IDs can't appear in grammar source"):
            glrmask.Grammar.from_lark("start: " + atom)

    body = glrmask.Vocab({0: b"a", 1: b"a"})
    module = glrmask.Module(glrmask.Grammar.from_lark("start: <|begin|> <|body|>"))
    module.bind('begin', glrmask.TokenSet([99]))
    module.bind('body', glrmask.TokenSet([1]))
    state = module.compile(body).start()
    assert token_allowed(state.mask(), 99) and not token_allowed(state.mask(), 0)
    assert state.commit_token(99)
    assert token_allowed(state.mask(), 1) and not token_allowed(state.mask(), 0)
    assert state.commit_token(1)
    assert state.is_accepting()
    with pytest.raises(glrmask.GLRMaskError):
        glrmask.Grammar.from_lark("start: <|missing|>").compile(body)

def test_rejection_preserves_original_copy():
    original = glrmask.Grammar.from_lark('start: "a" "a" "a" "a" "a"').compile(_vocab()).start()
    copy = original.copy()
    assert copy.commit_token(0)
    assert copy.commit_token(0)
    assert not copy.commit_token(1)
    assert copy.is_terminated()
    assert not original.is_terminated()
    assert not original.is_accepting()


def test_index_ids_and_stepped_ranges():
    import numpy as np
    assert _constraint().start().commit_token(np.uint32(0))
    assert glrmask.TokenSet([range(1, 10, 3)]) == glrmask.TokenSet([1, 4, 7])
    assert glrmask.TokenSet([range(7, 0, -3)]) == glrmask.TokenSet([1, 4, 7])
    assert glrmask.TokenSet([range(3, 3)]) == glrmask.TokenSet([])
