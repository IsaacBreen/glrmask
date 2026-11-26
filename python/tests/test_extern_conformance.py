"""Exact event semantics must survive open preparation, caching and serialization."""
import glrmask as g
from mask_helpers import token_allowed, unpack_mask

import pytest


VOCAB = {0: b'x', 1: b'y', 2: b'z', 17: b'm', 18: b'm', 19: b'', 20: b''}
LIFECYCLES = ['prebound', 'open', 'open-loaded']
STRATEGIES = [g.MaskStrategy.FAST_MASKS, g.MaskStrategy.FAST_COMPILE]


def prepared(source, vocab, lifecycle, strategy):
    module = g.Module(g.Grammar.from_glrm(source))
    if lifecycle != 'prebound':
        module.precompile(vocab, mask_strategy=strategy)
    if lifecycle == 'open-loaded':
        module = g.Module.load(module.save())
    return module


def verify(constraint, prefix, allowed, accepting, forced, *, conservative=False):
    def at_prefix():
        state = constraint.start()
        for token in prefix:
            assert state.commit_token(token)
        return state

    state = at_prefix()
    assert state.is_accepting() == accepting
    fresh = state.mask().copy()
    assert set(unpack_mask(fresh, state.mask_bit_len).nonzero()[0]) == set(allowed)
    assert (state.mask() == fresh).all()
    assert (at_prefix().mask() == fresh).all()
    for token in VOCAB:
        assert state.copy().commit_token(token) == (token in allowed), token
    actual = state.forced_bytes()
    expected = b"".join(VOCAB[token] for token in __import__("itertools").takewhile(lambda token: token in (0, 1, 2), forced))
    if conservative:
        assert expected.startswith(actual)
    else:
        assert actual == expected
    assert (state.mask() == fresh).all()
    assert state.is_accepting() == accepting
    if actual:
        assert state.copy().commit_bytes(actual)


@pytest.mark.parametrize('kind', ['token', 'grammar'])
@pytest.mark.parametrize('lifecycle', LIFECYCLES)
@pytest.mark.parametrize('strategy', STRATEGIES)
def test_shared_exact_event_preserves_both_continuations(kind, lifecycle, strategy):
    v = g.Vocab(VOCAB)
    m = prepared(f'extern {kind} A; extern {kind} B; start s; nt s = A "x" | B "y";', v, lifecycle, strategy)
    m.bind(A=g.TokenSet([17]), B=g.TokenSet([17]))
    c = m.compile(v, mask_strategy=strategy)
    verify(c, [], [17], False, [17])
    verify(c, [17], [0, 1], False, [])
    verify(c, [17, 0], [], True, [])
    verify(c, [17, 1], [], True, [])
    assert not c.start().commit_bytes(b'm')


@pytest.mark.parametrize('kind', ['token', 'grammar'])
@pytest.mark.parametrize('lifecycle', LIFECYCLES)
@pytest.mark.parametrize('strategy', STRATEGIES)
def test_zero_byte_event_and_empty_set_are_not_epsilon(kind, lifecycle, strategy):
    v = g.Vocab(VOCAB)
    source = f'extern {kind} MARK; start s; nt s = MARK "x";'
    m = prepared(source, v, lifecycle, strategy)
    empty = m.copy()
    m.bind('MARK', g.TokenSet([19]))
    c = m.compile(v, mask_strategy=strategy)
    verify(c, [], [19], False, [19, 0])
    verify(c, [19], [0], False, [0])
    verify(c, [19, 0], [], True, [])
    # Empty TokenSet must never complete MARK as epsilon. A typed rejection is also valid.
    try:
        empty.bind('MARK', g.TokenSet([]))
    except g.GLRMaskError as e:
        assert e.code == 'InvalidInput'
        return
    c = empty.compile(v, mask_strategy=strategy)
    verify(c, [], [], False, [])


@pytest.mark.parametrize('kind', ['token', 'grammar'])
@pytest.mark.parametrize('lifecycle', LIFECYCLES)
@pytest.mark.parametrize('strategy', STRATEGIES)
def test_clone_bindings_and_scoped_children_do_not_leak(kind, lifecycle, strategy):
    v = g.Vocab(VOCAB)
    m = prepared(f'extern {kind} MARK; start s; nt s = MARK;', v, lifecycle, strategy)
    left, right = m.copy(), m.copy()
    left.bind('MARK', g.TokenSet([17]))
    right.bind('MARK', g.TokenSet([18]))
    a, b = left.compile(v, mask_strategy=strategy), right.compile(v, mask_strategy=strategy)
    verify(a, [], [17], False, [17])
    verify(b, [], [18], False, [18])
    parent = prepared('extern grammar LEFT; extern grammar RIGHT; start s; nt s = LEFT RIGHT;', v, lifecycle, strategy)
    parent.bind(LEFT=left, RIGHT=right)
    c = parent.compile(v, mask_strategy=strategy)
    verify(c, [], [17], False, [17, 18])
    verify(c, [17], [18], False, [18])
    verify(c, [17, 18], [], True, [])
    verify(a, [], [17], False, [17])
    verify(b, [], [18], False, [18])


@pytest.mark.parametrize('kind', ['token', 'grammar'])
@pytest.mark.parametrize('lifecycle', LIFECYCLES)
@pytest.mark.parametrize('strategy', STRATEGIES)
def test_nullable_payload_allows_exact_end_immediately(kind, lifecycle, strategy):
    v = g.Vocab(VOCAB)
    parent = prepared(f'extern grammar PAYLOAD; extern {kind} END; start s; nt s = PAYLOAD END;', v, lifecycle, strategy)
    parent.bind(PAYLOAD=g.Grammar.from_lark('start: "x"?'), END=g.TokenSet([19]))
    c = parent.compile(v, mask_strategy=strategy)
    verify(c, [], [0, 19], False, [])
    verify(c, [0], [19], False, [19], conservative=True)
    verify(c, [19], [], True, [])
    verify(c, [0, 19], [], True, [])
