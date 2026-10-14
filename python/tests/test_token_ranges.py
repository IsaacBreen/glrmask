"""Exact token sets are model-ID sets, not byte strings or token sequences."""


def _bound_snapshot(grammar, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    result = grammar.clone()
    assert result.bind(name, value) is None
    return result

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


@pytest.mark.parametrize("ranges", [[range(0)], [range(7, 2)], [range(-1, 2)],
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
    assert len(v.tokens(range(0))) == 0
    assert len(v.token_ranges([])) == 0
    assert v.token_ranges([range(0, 2), range(3, 4)]).ids == [0, 1, 3]


@pytest.mark.parametrize("mode", MODES)
def test_exact_ranges_preserve_id_identity_repetition_and_overlap(mode):
    v = vocab()
    source = "glrm 1; start start; extern token A; extern token B; nt start = A B;"
    c = (_bound_snapshot(_bound_snapshot(glrmask.Grammar.from_glrm(source), "A", v.tokens(range(10, 31))), "B", v.token_ranges([range(20, 41), range(70, 81)]))
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
    module = glrmask.Grammar.from_glrm(ONE).prepare(v)
    bound = _bound_snapshot(module, "T", v.token_ranges([range(3, 9), range(90, 100)]))
    expected = set(range(3, 9)) | set(range(90, 100))
    loaded = glrmask.PreparedGrammar.load(bound.save())
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
        constraint = _bound_snapshot(glrmask.Grammar.from_glrm(ONE), "T", tokens).compile(v, optimization=mode)
        state = constraint.start()
        mask = state.mask()
        assert int(mask.sum()) == len(tokens)
        assert bool(mask[n - 1])
        if excluded is not None:
            assert not bool(mask[excluded])
        state.commit_token(n - 1)
        assert state.is_accepting()


def pointwise_token_grammar(ids):
    """Independent singleton alternatives; deliberately do not use Vocab.tokens."""
    ids = list(ids)
    names = [f"E{i}" for i in range(len(ids))]
    source = ("glrm 1; start item; " + " ".join(f"extern token {name};" for name in names)
              + " nt item = " + " | ".join(names) + ";")
    grammar = glrmask.Grammar.from_glrm(source)
    for name, token in zip(names, ids):
        grammar = _bound_snapshot(grammar, name, glrmask.Token(token))
    return grammar


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("deferred", [False, True])
def test_compact_ranges_match_pointwise_token_alternatives(mode, deferred):
    # Different model IDs may alias a spelling, including empty bytes. The
    # oracle uses one grammar alternative per exact ID, never a compact set.
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 3: b"ba",
                                      4: b"x", 5: b"", 6: b"a", 7: b"b",
                                      8: b"x", 9: b"x", 10: b"x", 11: b"x"})
    selections = {"A": [range(1, 4), range(6, 10)], "B": [range(3, 8)]}
    for expression in ["A B", "(A | B) A", '\"a\" A \"b\" | \"ab\" B',
                       "A? B+", "(A B)*"]:
        source = "glrm 1; start start; extern token A; extern token B; nt start = " + expression + ";"
        compact = glrmask.Grammar.from_glrm(source)
        pointwise = glrmask.Grammar.from_glrm(source.replace("extern token A;", "extern grammar A;").replace("extern token B;", "extern grammar B;"))
        if deferred:
            compact = compact.prepare(v)
            pointwise = pointwise.prepare(v)
        for name, ranges in selections.items():
            compact = _bound_snapshot(compact, name, v.token_ranges(ranges))
            oracle = pointwise_token_grammar(token for interval in ranges for token in interval)
            pointwise = _bound_snapshot(pointwise, name, oracle.prepare(v) if deferred else oracle)
        if deferred:
            actual = compact.link(optimization=mode)
            expected = pointwise.link(optimization=mode)
        else:
            actual = compact.compile(v, optimization=mode)
            expected = pointwise.compile(v, optimization=mode)
        # Exhaust every legal prefix through two model tokens, comparing the
        # full next-token mask and acceptance at every node.
        prefixes = [()]
        for depth in range(3):
            next_prefixes = []
            for prefix in prefixes:
                states = [actual.start(), expected.start()]
                for state in states:
                    for token in prefix:
                        state.commit_token(token)
                masks = [accepted_ids(state) for state in states]
                assert masks[0] == masks[1], (mode, deferred, expression, prefix, masks)
                assert states[0].is_accepting() == states[1].is_accepting(), (expression, prefix)
                if depth < 2:
                    next_prefixes.extend(prefix + (token,) for token in sorted(masks[0]))
            prefixes = next_prefixes


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("compact", [False, True])
def test_ambiguous_two_events_cannot_admit_a_third_token(mode, compact):
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 3: b"aa", 4: b"ba",
                                      5: b"x", 6: b"a", 7: b"", 8: b"b", 9: b"c",
                                      10: b"bb", 11: b"x"})
    grammar = glrmask.Grammar.from_glrm('glrm 1; start start; extern token A; extern token B; nt start = (A | B) (A | "a");')
    grammar = _bound_snapshot(grammar, "A", v.tokens(range(6)) if compact else glrmask.Token(0))
    grammar = _bound_snapshot(grammar, "B", v.tokens(range(3, 10)) if compact else glrmask.Token(1))
    constraint = grammar.compile(v, optimization=mode, end_tokens=[15])
    state = constraint.start()
    state.commit_token(0)
    state.commit_token(0)
    assert state.is_accepting()
    assert accepted_ids(state) == {15}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("a,b", [(range(1), range(1, 2)), (range(6), range(3, 10)),
                                  (range(12), range(11))])
def test_ambiguous_exact_byte_choices_stop_after_two_tokens(mode, a, b):
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 3: b"aa",
                                      4: b"ba", 5: b"x", 6: b"a", 7: b"",
                                      8: b"b", 9: b"c", 10: b"bb", 11: b"x"})
    grammar = glrmask.Grammar.from_glrm(
        'glrm 1; start start; extern token A; extern token B; nt start = (A | B) (A | "a");')
    constraint = _bound_snapshot(_bound_snapshot(grammar, "A", v.tokens(a)), "B", v.tokens(b)).compile(
        v, optimization=mode, end_tokens=[15])
    state = constraint.start()
    state.commit_token(0)
    state.commit_token(0)
    assert state.is_accepting()
    assert accepted_ids(state) == {15}
    with pytest.raises(ValueError):
        state.commit_token(0)
