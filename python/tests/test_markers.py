"""Streaming marker batches: IDs and byte positions, never parse-tree captures."""
import pytest
import glrmask

MODES = [glrmask.MaskStrategy.FAST_BUILD, glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
         glrmask.MaskStrategy.FAST_RUNTIME, glrmask.MaskStrategy.AUTO]
SOURCE = 'glrm 1; start root; nt root = @mark(10) "a" @mark(20) "b" @mark(30);'


@pytest.mark.parametrize("mode", MODES)
def test_cold_canonical_forcing_is_silent_across_exact_events_and_markers(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 10: b""})
    grammar = glrmask.Module(glrmask.Grammar.from_glrm(
        'extern token EVENT start root '
        'nt root=@mark(1) EVENT @mark(2) "a" @mark(3);'
    ))
    grammar.bind("EVENT", v.tokens([10]))
    compiled = grammar.compile(v, mask_strategy=mode)
    for constraint in [compiled, glrmask.Constraint.load(compiled.save())]:
        state = constraint.start()
        before = state.marker_events()
        assert before == [(1, 0)]
        # No mask() call precedes this query: internal lookahead must be silent.
        assert state.forced_tokens() == [10, 0]
        assert state.marker_events() == before
        state.commit_token(10)
        assert state.marker_events() == [(2, 0)]
        assert state.forced_tokens() == [0]
        assert state.marker_events() == [(2, 0)]
        state.commit_token(0)
        assert state.marker_events() == [(3, 1)]
        assert state.is_accepting()
        assert state.forced_tokens() == []


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"ab", 3: b"x", 4: b"y",
        5: b"xaby", 6: b"pxabyq", 7: b"p", 8: b"q", 10: b"", 11: b"<event>"})


def events(state):
    result = state.marker_events()
    assert all(isinstance(e, tuple) and len(e) == 2 and all(type(v) is int for v in e) for e in result)
    assert result == sorted(set(result), key=lambda event: (event[1], event[0]))
    return result


@pytest.mark.parametrize("mode", MODES)
def test_initial_split_and_fused_commits(mode):
    v = vocab()
    c = glrmask.Grammar.from_glrm(SOURCE).compile(v, mask_strategy=mode)
    assert c.marker_ids() == [10, 20, 30]
    for trace in [[0, 1], [2]]:
        s = c.start()
        observed = events(s)
        assert observed == [(10, 0)]
        for token in trace:
            previous = events(s)
            s.mask(); s.forced_tokens()
            assert events(s) == previous  # Internal probes must be silent.
            assert s.commit_token(token) is None
            observed.extend(events(s))
        assert observed == [(10, 0), (20, 1), (30, 2)]
        assert s.is_accepting()
        # Python receives a value copy, not a mutable alias of the Rust buffer.
        copied = events(s); copied.clear()
        assert events(s)


@pytest.mark.parametrize("mode", MODES)
def test_save_load_external_vocab_and_compiled_children(mode):
    v = vocab()
    c = glrmask.Grammar.from_glrm(SOURCE).compile(v, mask_strategy=mode)
    for restored in [glrmask.Constraint.load(c.save()),
                     glrmask.Constraint.load(c.save(external_vocab=True), vocab=v)]:
        assert restored.marker_ids() == [10, 20, 30]
        s = restored.start(); s.commit_token(2)
        assert events(s) == [(20, 1), (30, 2)]
    # Prepared components retain marker programs without a second user grammar.
    parent = glrmask.Module(glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar child; nt root = @mark(40) "x" child "y" @mark(50);'
    ))
    child = glrmask.Module(glrmask.Grammar.from_glrm(SOURCE))
    assert parent.bind("child", child) is None
    parent.precompile(v, mask_strategy=mode)
    prepared = glrmask.Module.load(parent.save(), vocab=v)
    linked = prepared.compile(v, mask_strategy=mode)
    assert linked.marker_ids() == [10, 20, 30, 40, 50]
    s = linked.start(); assert events(s) == [(40, 0)]
    s.commit_token(5)
    assert events(s) == [(10, 1), (20, 2), (30, 3), (50, 4)]


@pytest.mark.parametrize("mode", MODES)
def test_markers_are_not_emitted_inside_an_incomplete_terminal(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"c", 3: b"abc"})
    c = glrmask.Grammar.from_glrm(
        'glrm 1; start root; t WORD = "abc"; nt root = @mark(0) WORD @mark(1);'
    ).compile(v, mask_strategy=mode)
    s = c.start(); assert events(s) == [(0, 0)]
    s.commit_token(0); assert events(s) == []
    s.commit_token(1); assert events(s) == []
    s.commit_token(2); assert events(s) == [(1, 3)]


@pytest.mark.parametrize("mode", MODES)
def test_byte_offsets_can_cross_utf8_model_token_boundaries(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"\xc3", 1: b"\xa9", 2: b"x", 3: "éx".encode()})
    c = glrmask.Grammar.from_glrm(
        'glrm 1; start root; nt root = "é" @mark(1) "x" @mark(2);'
    ).compile(v, mask_strategy=mode)
    s = c.start(); s.commit_token(0); assert events(s) == []
    s.commit_token(1); assert events(s) == [(1, 2)]
    s.commit_token(2); assert events(s) == [(2, 3)]
    s = c.start(); s.commit_token(3); assert events(s) == [(1, 2), (2, 3)]


@pytest.mark.parametrize("mode", MODES)
def test_ambiguity_is_a_union_of_observations_not_an_error(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b", 2: b"c"})
    c = glrmask.Grammar.from_glrm(
        'glrm 1; start root; nt root = "a" @mark(1) "b" | "a" @mark(2) "c";'
    ).compile(v, mask_strategy=mode)
    s = c.start(); s.commit_token(0)
    before = events(s); assert before == [(1, 1), (2, 1)]
    s.commit_token(1)
    assert s.is_accepting() and events(s) == []
    assert before == [(1, 1), (2, 1)]  # No retraction when the other branch fails.


@pytest.mark.parametrize("mode", MODES)
def test_clone_and_failed_commits_do_not_leak_marker_batches(mode):
    v = vocab(); c = glrmask.Grammar.from_glrm(SOURCE).compile(v, mask_strategy=mode)
    original = c.start(); branch = original.clone()
    branch.commit_token(0)
    assert events(branch) == [(20, 1)] and events(original) == [(10, 0)]
    with pytest.raises(ValueError): branch.commit_token(9999)
    assert events(branch) == [] and not branch.is_rejected()
    branch.commit_token(1); assert events(branch) == [(30, 2)]
    with pytest.raises(ValueError): original.commit_token(3)
    assert original.is_rejected() and events(original) == []


@pytest.mark.parametrize("mode", MODES)
def test_empty_spelling_special_events_do_not_become_marker_tokens(mode):
    v = vocab()
    grammar = glrmask.Module(glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern token EVENT; nt root = @mark(0) EVENT @mark(1) "a" @mark(2);'
    ))
    assert grammar.bind("EVENT", v.tokens([10])) is None
    c = grammar.compile(v, mask_strategy=mode)
    s = c.start(); assert events(s) == [(0, 0)]
    assert s.mask()[10]
    s.commit_token(10); assert events(s) == [(1, 0)]
    s.commit_token(0); assert events(s) == [(2, 1)]
    assert s.is_accepting()


@pytest.mark.parametrize("mode", MODES)
def test_repeated_and_sparse_ids_are_deduplicated_without_id_sized_storage(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"aa"})
    c = glrmask.Grammar.from_glrm(
        'glrm 1; start root; nt root = @mark(4294967295) @mark(4294967295) ("a" @mark(7))+;'
    ).compile(v, mask_strategy=mode)
    assert c.marker_ids() == [7, 4294967295]
    s = c.start(); assert events(s) == [(4294967295, 0)]
    s.commit_token(1); assert events(s) == [(7, 1), (7, 2)]


def test_lark_and_ebnf_use_the_same_marker_api():
    v = glrmask.Vocab.from_dict({b"a": 0})
    grammars = [glrmask.Grammar.from_ebnf('start ::= @mark(1) "a" @mark(2)'),
                glrmask.Grammar.from_lark('start: @mark(1) "a" @mark(2)')]
    for grammar in grammars:
        s = grammar.compile(v).start()
        assert events(s) == [(1, 0)]
        s.commit_token(0); assert events(s) == [(2, 1)]


@pytest.mark.parametrize("mode", MODES)
def test_parent_markers_surround_a_prepared_json_schema_child(mode):
    # The JSON Schema frontend's fixed separators include a space after ':'.
    v = glrmask.Vocab.from_id_to_bytes({0: b"(", 1: b'{"x": 1}', 2: b")",
        3: b'({"x": 1})', 4: b"{", 5: b'"x"', 6: b": ", 7: b"1", 8: b"}"})
    parent = glrmask.Module(glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar C; '
        'nt root=@mark(1) "(" @mark(2) C @mark(3) ")" @mark(4);'
    ))
    child = glrmask.Module(glrmask.Grammar.from_json_schema({"type": "object",
        "properties": {"x": {"type": "integer"}}, "required": ["x"],
        "additionalProperties": False}))
    parent.precompile(v, mask_strategy=mode)
    child.precompile(v, mask_strategy=mode)
    assert parent.bind("C", child) is None
    linked = parent.compile(v, mask_strategy=mode)
    for c in [linked, glrmask.Constraint.load(linked.save())]:
        for trace in [[0, 1, 2], [3], [0, 4, 5, 6, 7, 8, 2]]:
            s = c.start(); observed = events(s)
            for token in trace:
                assert s.mask()[token]
                s.commit_token(token); observed.extend(events(s))
            assert observed == [(1, 0), (2, 1), (3, 9), (4, 10)]
            assert s.is_accepting()


@pytest.mark.parametrize("mode", MODES)
def test_nullable_child_can_report_more_than_one_possible_end(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b"(", 1: b"a", 2: b")", 3: b"(a)", 4: b"()"})
    parent = glrmask.Module(glrmask.Grammar.from_glrm(
        'glrm 1; start root; extern grammar C; '
        'nt root=@mark(1) "(" @mark(2) C @mark(3) ")" @mark(4);'
    ))
    child = glrmask.Module(glrmask.Grammar.from_glrm('glrm 1; start body; nt body="a"?;'))
    parent.precompile(v, mask_strategy=mode)
    child.precompile(v, mask_strategy=mode)
    assert parent.bind("C", child) is None
    linked = parent.compile(v, mask_strategy=mode)
    for c in [linked, glrmask.Constraint.load(linked.save())]:
        for trace in [[0, 1, 2], [3], [4]]:
            s = c.start(); observed = events(s)
            for token in trace:
                s.commit_token(token); observed.extend(events(s))
            expected = [(1, 0), (2, 1), (3, 1), (4, 2)] if trace == [4] else [
                (1, 0), (2, 1), (3, 1), (3, 2), (4, 3)]
            assert observed == expected
            assert s.is_accepting()
