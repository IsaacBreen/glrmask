import glrmask
import pytest


_BAD_IDS = (-1, 2**32, 2**40)


def _vocab():
    return glrmask.Vocab.from_dict({b"a": 0, b"b": 1})


def _constraint():
    return glrmask.Grammar.from_lark('start: "a"').compile(_vocab())


@pytest.mark.parametrize("bad", _BAD_IDS)
def test_out_of_range_token_id_fails_closed(bad):
    state = _constraint().start()
    assert state.commit_token(bad) is False
    assert state.is_accepting() is False
    assert state.is_terminated() is True
    assert state.commit_token(0) is False


@pytest.mark.parametrize("bad", _BAD_IDS)
def test_out_of_range_token_id_in_list_fails_closed(bad):
    state = _constraint().start()
    assert state.commit_tokens([0, bad]) is False
    assert state.is_accepting() is False
    assert state.is_terminated() is True
    assert state.commit_tokens([0]) is False


def test_non_integer_token_id_is_a_type_error():
    state = _constraint().start()
    with pytest.raises(TypeError):
        state.commit_token("0")
    with pytest.raises(TypeError):
        state.commit_tokens([0, "1"])


def test_special_tokens_are_explicit():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"a", 3: b""})
    wildcard = glrmask.Grammar.from_lark("start: <[*]>").compile(vocab)
    mask = wildcard.start().mask()
    assert mask[0] and mask[1] and not mask[3]
    with pytest.raises(ValueError):
        glrmask.Grammar.from_lark("start: <[3]>").compile(vocab)

    body = glrmask.Vocab.from_dict({b"a": 0, b"a": 1})
    module = glrmask.Module(glrmask.Grammar.from_lark("start: <|begin|> <|body|>"))
    module.bind(begin=glrmask.TokenSet([99]), body=glrmask.TokenSet([1]))
    state = module.compile(body).start()
    assert state.mask()[99] and not state.mask()[0]
    assert state.commit_token(99)
    assert state.mask()[1] and not state.mask()[0]
    assert state.commit_token(1)
    assert state.is_accepting()
    with pytest.raises(ValueError):
        glrmask.Grammar.from_lark("start: <|missing|>").compile(body)
