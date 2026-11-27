"""Token IDs above the maximum are errors, not huge allocations that abort Python."""
import pytest

import glrmask

MAX_TOKEN_ID = 2**24 - 1
HUGE_IDS = [2**32 - 1, 4 * 10**9, MAX_TOKEN_ID + 1]


def token_extern():
    return glrmask.Grammar.from_glrm("start s extern token E nt s = E")


@pytest.mark.parametrize("token_id", HUGE_IDS)
def test_vocab_rejects_a_huge_id(token_id):
    with pytest.raises(glrmask.GLRMaskError, match=str(token_id)) as exc:
        glrmask.Vocab({0: b"a", token_id: b"b"})
    assert exc.value.kind == "invalid_input"


@pytest.mark.parametrize("token_id", HUGE_IDS)
def test_binding_a_huge_id_is_an_error(token_id):
    grammar = token_extern()
    with pytest.raises(glrmask.GLRMaskError, match="maximum token ID") as exc:
        grammar.bind("E", glrmask.TokenSet([token_id]))
    assert exc.value.kind == "invalid_input"
    module = glrmask.Module(token_extern())
    with pytest.raises(glrmask.GLRMaskError, match="maximum token ID") as exc:
        module.bind("E", glrmask.TokenSet([token_id]))
    assert exc.value.kind == "invalid_input"
    assert "E" in module.externs.unbound


@pytest.mark.parametrize("token_id", HUGE_IDS)
def test_token_sets_reject_a_huge_id(token_id):
    with pytest.raises(glrmask.GLRMaskError, match="maximum token ID") as exc:
        glrmask.TokenSet([1, token_id])
    assert exc.value.kind == "invalid_input"
    with pytest.raises(glrmask.GLRMaskError, match="maximum token ID") as exc:
        glrmask.TokenSet([range(token_id, token_id + 1)])
    assert exc.value.kind == "invalid_input"


def test_the_maximum_id_is_accepted():
    grammar = token_extern()
    grammar = glrmask.Module(grammar)
    grammar.bind("E", glrmask.TokenSet([MAX_TOKEN_ID]))
    assert glrmask.TokenSet([range(0, MAX_TOKEN_ID + 1)]) == glrmask.TokenSet(
        [range(0, MAX_TOKEN_ID + 1)]
    )


def test_mask_size_keyword_is_not_supported():
    vocab = glrmask.Vocab({0: b"a"})
    state = glrmask.Grammar.from_glrm('start s nt s = "a"').compile(vocab).start()
    with pytest.raises(TypeError):
        state.mask(size=4 * 10**9)
    assert state.mask().shape == ((state.mask_bit_len+31)//32,) == (1,)
