"""Token sets are model-ID sets, not byte strings or token sequences."""


def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, value) is None
    return m

import pytest
import glrmask

ONE = "glrm 1; start start; extern token T; nt start = T;"
MODES = [glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_BUILD,
         glrmask.MaskStrategy.TOKEN_CLASS_TRIE, glrmask.MaskStrategy.FAST_RUNTIME]


def vocab(n=128):
    # All IDs alias the same byte spelling; ranges must not leak other IDs.
    return glrmask.Vocab.from_id_to_bytes({i: b"x" for i in range(n)})


def accepted_ids(state):
    return set(state.mask().nonzero()[0].tolist())


def test_range_inputs_are_normalized_to_one_set():
    tokens = glrmask.TokenSet.from_ranges([range(20, 40), range(10, 25), range(40, 50), range(90, 95)])
    assert tokens == glrmask.TokenSet.from_ranges([range(10, 50), range(90, 95)])
    assert repr(tokens) == "TokenSet[10..=49, 90..=94]"
    assert glrmask.TokenSet.from_ranges(r for r in [range(3, 5), range(5, 9)]) == glrmask.TokenSet.from_ranges([range(3, 9)])
    assert glrmask.TokenSet(iter([8, 3, 4, 5])) == glrmask.TokenSet.from_ranges([range(3, 6), range(8, 9)])
    assert glrmask.TokenSet(range(4)) != glrmask.TokenSet(range(5))


@pytest.mark.parametrize("ranges", [[range(2, 10, 2)], [(2, 10)]])
def test_invalid_compact_ranges_are_rejected(ranges):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        glrmask.TokenSet.from_ranges(ranges)


def test_empty_and_reversed_ranges_contribute_nothing():
    assert glrmask.TokenSet.from_ranges([range(0)]) == glrmask.TokenSet([])
    assert glrmask.TokenSet.from_ranges([range(7, 2)]) == glrmask.TokenSet([])
    assert glrmask.TokenSet.from_ranges([range(3, 3), range(5, 5)]) == glrmask.TokenSet([])


def test_membership_is_validated_at_compile_and_duplicates_are_ignored():
    v = glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"x", 3: b"x"})
    grammar = glrmask.Grammar.from_glrm(ONE)
    for value in [glrmask.TokenSet(range(4)), glrmask.TokenSet.from_ranges([range(1, 4)])]:
        with pytest.raises(ValueError):
            _bound_snapshot(grammar, "T", value).compile(v)
    assert glrmask.TokenSet([1, 1]) == glrmask.TokenSet([1])
    assert _bound_snapshot(grammar, "T", glrmask.TokenSet([])).compile(v) is not None
    assert _bound_snapshot(grammar, "T", glrmask.TokenSet.from_ranges([])).compile(v) is not None
    assert glrmask.TokenSet.from_ranges([range(0, 2), range(3, 4)]) == glrmask.TokenSet([0, 1, 3])


@pytest.mark.parametrize("mode", MODES)
def test_exact_ranges_preserve_id_identity_repetition_and_overlap(mode):
    v = vocab()
    source = "glrm 1; start start; extern token A; extern token B; nt start = A B;"
    module = _bound_snapshot(_bound_snapshot(glrmask.Grammar.from_glrm(source), "A", glrmask.TokenSet(range(10, 31))), "B", glrmask.TokenSet.from_ranges([range(20, 41), range(70, 81)]))
    c = module.compile(v, mask_strategy=mode)
    for first in (10, 20, 30):
        state = c.start()
        assert accepted_ids(state) == set(range(10, 31))
        assert state.commit_token(first)
        assert accepted_ids(state) == set(range(20, 41)) | set(range(70, 81))
        assert state.commit_token(80)
        assert state.is_accepting()
    loaded = glrmask.Constraint.load(c.save(), v)
    assert accepted_ids(loaded.start()) == set(range(10, 31))


def test_deferred_range_binding_and_serialization():
    v = vocab()
    module = glrmask.Module(glrmask.Grammar.from_glrm(ONE))
    module.precompile(v)
    bound = _bound_snapshot(module, "T", glrmask.TokenSet.from_ranges([range(3, 9), range(90, 100)]))
    expected = set(range(3, 9)) | set(range(90, 100))
    loaded = glrmask.Module.load(bound.save())
    for mode in MODES:
        constraint = loaded.compile(v, mask_strategy=mode)
        assert accepted_ids(constraint.start()) == expected
    with pytest.raises(ValueError):
        module.compile(v)


@pytest.mark.parametrize("mode", MODES)
def test_128k_contiguous_and_nearly_all_token_sets(mode):
    n = 131_072
    # Empty byte spellings still denote distinct exact model-token events.
    v = glrmask.Vocab.from_id_to_bytes({i: b"" for i in range(n)})
    for tokens, excluded in [(glrmask.TokenSet(range(n)), None),
                             (glrmask.TokenSet.from_ranges([range(0, 65_535), range(65_536, n)]), 65_535)]:
        module = _bound_snapshot(glrmask.Grammar.from_glrm(ONE), "T", tokens)
        constraint = module.compile(v, mask_strategy=mode)
        state = constraint.start()
        mask = state.mask()
        assert int(mask.sum()) == n - (excluded is not None)
        assert bool(mask[n - 1])
        if excluded is not None:
            assert not bool(mask[excluded])
        assert state.commit_token(n - 1)
        assert state.is_accepting()


def pointwise_token_grammar(ids):
    """Independent singleton alternatives; deliberately do not use Vocab.tokens."""
    ids = list(ids)
    names = [f"E{i}" for i in range(len(ids))]
    source = ("glrm 1; start item; " + " ".join(f"extern token {name};" for name in names)
              + " nt item = " + " | ".join(names) + ";")
    grammar = glrmask.Grammar.from_glrm(source)
    for name, token in zip(names, ids):
        grammar = _bound_snapshot(grammar, name, glrmask.TokenSet([token]))
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
        compact = glrmask.Module(glrmask.Grammar.from_glrm(source))
        pointwise = glrmask.Module(glrmask.Grammar.from_glrm(source.replace("extern token A;", "extern grammar A;").replace("extern token B;", "extern grammar B;")))
        if deferred:
            compact.precompile(v)
            pointwise.precompile(v)
        for name, ranges in selections.items():
            compact = _bound_snapshot(compact, name, glrmask.TokenSet.from_ranges(ranges))
            oracle = pointwise_token_grammar(token for interval in ranges for token in interval)
            pointwise = _bound_snapshot(pointwise, name, oracle)
        actual = compact.compile(v, mask_strategy=mode)
        expected = pointwise.compile(v, mask_strategy=mode)
        # Exhaust every legal prefix through two model tokens, comparing the
        # full next-token mask and acceptance at every node.
        prefixes = [()]
        for depth in range(3):
            next_prefixes = []
            for prefix in prefixes:
                states = [actual.start(), expected.start()]
                for state in states:
                    for token in prefix:
                        assert state.commit_token(token)
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
    grammar = _bound_snapshot(grammar, "A", glrmask.TokenSet(range(6)) if compact else glrmask.TokenSet([0]))
    grammar = _bound_snapshot(grammar, "B", glrmask.TokenSet(range(3, 10)) if compact else glrmask.TokenSet([1]))
    constraint = glrmask.Grammar.from_expr(glrmask.Expr.sequence(grammar, glrmask.TokenSet([15]))).compile(v, mask_strategy=mode)
    state = constraint.start()
    assert state.commit_token(0)
    assert state.commit_token(0)
    # The grammar is complete; acceptance needs the end token.
    assert not state.is_accepting()
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
    module = _bound_snapshot(_bound_snapshot(grammar, "A", glrmask.TokenSet(a)), "B", glrmask.TokenSet(b))
    constraint = glrmask.Grammar.from_expr(glrmask.Expr.sequence(module, glrmask.TokenSet([15]))).compile(v, mask_strategy=mode)
    state = constraint.start()
    assert state.commit_token(0)
    assert state.commit_token(0)
    # The grammar is complete; acceptance needs the end token.
    assert not state.is_accepting()
    assert accepted_ids(state) == {15}
    assert not state.commit_token(0)