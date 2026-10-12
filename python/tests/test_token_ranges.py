"""Exact token sets are model-ID sets, not byte strings or token sequences."""
import pytest
import glrmask

ONE = "glrm 1; start start; extern token T; nt start = T;"
MODES = [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
         glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME]


def vocab(n=128):
    # All IDs alias the same byte spelling; ranges must not leak other IDs.
    return glrmask.Vocab.from_id_to_bytes({i: b"x" for i in range(n)})


def accepted_ids(state):
    return set(state.mask().nonzero()[0].tolist())


def test_range_inputs_are_normalized_and_introspection_stays_compact():
    v = vocab()
    tokens = v.token_ranges([range(20, 40), range(10, 25), range(40, 50), range(90, 95)])
    assert tokens.ranges == [range(10, 50), range(90, 95)]
    assert len(tokens) == 45
    assert tokens.ids == list(range(10, 50)) + list(range(90, 95))
    assert v.tokens(range(30, 10, -1)).ranges == [range(11, 31)]
    assert v.token_ranges(r for r in [range(3, 5), range(5, 9)]).ranges == [range(3, 9)]
    assert v.tokens(iter([8, 3, 4, 5])).ranges == [range(3, 6), range(8, 9)]
    assert v.tokens(range(2, 12, 3)).ids == [2, 5, 8, 11]


@pytest.mark.parametrize("ranges", [[], [range(0)], [range(7, 2)], [range(-1, 2)],
                                    [range(127, 129)], [range(2, 10, 2)], [(2, 10)]])
def test_invalid_compact_ranges_are_rejected(ranges):
    with pytest.raises((ValueError, OverflowError)):
        vocab().token_ranges(ranges)


def test_missing_ids_and_duplicate_explicit_ids_are_rejected():
    v = glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"x", 3: b"x"})
    with pytest.raises(ValueError):
        v.tokens(range(4))
    with pytest.raises(ValueError):
        v.token_ranges([range(1, 4)])
    with pytest.raises(ValueError):
        v.tokens([1, 1])
    with pytest.raises(ValueError):
        v.tokens(range(0))
    assert v.token_ranges([range(0, 2), range(3, 4)]).ids == [0, 1, 3]


@pytest.mark.parametrize("mode", MODES)
def test_exact_ranges_preserve_id_identity_repetition_and_overlap(mode):
    v = vocab()
    source = "glrm 1; start start; extern token A; extern token B; nt start = A B;"
    c = (glrmask.Grammar.from_glrm(source)
         .bind("A", v.tokens(range(10, 31)))
         .bind("B", v.token_ranges([range(20, 41), range(70, 81)]))
         .compile(v, optimization=mode))
    for first in (10, 20, 30):
        state = c.start()
        assert accepted_ids(state) == set(range(10, 31))
        state.commit_token(first)
        assert accepted_ids(state) == set(range(20, 41)) | set(range(70, 81))
        state.commit_token(80)
        assert state.is_accepting()
    loaded = glrmask.Constraint.load(c.save())
    assert accepted_ids(loaded.start()) == set(range(10, 31))


def test_deferred_range_binding_and_serialization():
    v = vocab()
    module = glrmask.Grammar.from_glrm(ONE).compile_unlinked(v)
    bound = module.bind("T", v.token_ranges([range(3, 9), range(90, 100)]))
    expected = set(range(3, 9)) | set(range(90, 100))
    loaded = glrmask.UnlinkedConstraint.load(bound.save())
    for mode in MODES:
        constraint = loaded.link(optimization=mode)
        assert accepted_ids(constraint.start()) == expected
    with pytest.raises(ValueError):
        module.link()


@pytest.mark.parametrize("mode", MODES)
def test_128k_contiguous_and_nearly_all_token_sets(mode):
    n = 131_072
    # Empty byte spellings still denote distinct exact model-token events.
    v = glrmask.Vocab.from_id_to_bytes({i: b"" for i in range(n)})
    for tokens, excluded in [(v.tokens(range(n)), None),
                             (v.token_ranges([range(0, 65_535), range(65_536, n)]), 65_535)]:
        assert len(tokens.ranges) == (1 if excluded is None else 2)
        assert len(tokens) == n - (excluded is not None)
        constraint = glrmask.Grammar.from_glrm(ONE).bind("T", tokens).compile(v, optimization=mode)
        state = constraint.start()
        mask = state.mask()
        assert int(mask.sum()) == len(tokens)
        assert bool(mask[n - 1])
        if excluded is not None:
            assert not bool(mask[excluded])
        state.commit_token(n - 1)
        assert state.is_accepting()
