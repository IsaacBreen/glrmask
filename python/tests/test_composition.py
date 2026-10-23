

def _bound_snapshot(module, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    result = module.copy()
    assert result.bind(name, value) is None
    return result

import glrmask


def _prepare_glrm(source, vocab, *, subgrammars=None):
    parent = glrmask.Module(glrmask.Grammar.from_glrm(source))
    for name, child in (subgrammars or {}).items():
        parent.bind(name, child)
    parent.precompile(vocab)
    return parent


def _glrm(source, vocab, *, subgrammars=None, end_tokens=None, mask_strategy=None):
    kwargs = {}
    if end_tokens is not None:
        kwargs["end_tokens"] = end_tokens
    m = _prepare_glrm(source, vocab, subgrammars=subgrammars)
    if mask_strategy is not None:
        kwargs["mask_strategy"] = mask_strategy
    return m.compile(vocab, **kwargs)


def test_from_glrm_grammar_binds_typed_external_subgrammar() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"X", 1: b"ab!", 2: b"Xab!", 3: b"a", 4: b"b", 5: b"!"}
    )
    child = _prepare_glrm(
        '''
        glrm 1;
        start child;
        nt child = "a" "b";
        ''',
        vocab,
    )
    composed = _glrm(
        '''
        glrm 1;
        start document;
        extern grammar payload;
        nt document = "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    inline = _glrm(
        '''
        glrm 1;
        start document;
        grammar payload {
            start child;
            nt child = "a" "b";
        };
        nt document = "X" payload "!";
        ''',
        vocab,
    )

    actual = composed.start()
    expected = inline.start()
    assert actual.mask().tolist() == expected.mask().tolist()
    actual.commit_token(2)
    expected.commit_token(2)
    assert actual.is_accepting()
    assert expected.is_accepting()


def test_external_subgrammar_matches_monolithic_across_token_boundary() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"X", 1: b"ab!", 2: b"a", 3: b"b", 4: b"!"}
    )
    child = _prepare_glrm(
        '''
        glrm 1;
        start child;
        nt child = "a" "b";
        ''',
        vocab,
    )
    composed = _glrm(
        '''
        glrm 1;
        start document;
        extern grammar payload;
        nt document = "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    monolithic = _glrm(
        '''
        glrm 1;
        start document;
        grammar inner {
            start child;
            nt child = "a" "b";
        };
        nt document = "X" inner "!";
        ''',
        vocab,
    )

    expected = monolithic.start()
    actual = composed.start()
    assert actual.mask().tolist() == expected.mask().tolist()
    actual.commit_token(0)
    expected.commit_token(0)
    assert actual.mask().tolist() == expected.mask().tolist()
    assert actual.mask().tolist()[1]
    actual.commit_token(1)
    expected.commit_token(1)
    assert actual.is_accepting()
    assert expected.is_accepting()


def test_external_subgrammar_matches_monolithic_for_nullable_child() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"X!", 1: b"Xa!", 2: b"X", 3: b"a", 4: b"!"}
    )
    child = _prepare_glrm(
        '''
        glrm 1;
        start child;
        nt item = "a";
        nt child = item?;
        ''',
        vocab,
    )
    composed = _glrm(
        '''
        glrm 1;
        start document;
        extern grammar payload;
        nt document = "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    monolithic = _glrm(
        '''
        glrm 1;
        start document;
        nt item = "a";
        nt document = "X" item? "!";
        ''',
        vocab,
    )

    for sequence in ([0], [1], [2, 4], [2, 3, 4]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_accepting()
        assert expected.is_accepting()


def test_nested_nullable_external_subgrammar_survives_save_load() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"X!", 1: b"Xa!", 2: b"X", 3: b"a", 4: b"!"}
    )
    leaf = _prepare_glrm(
        '''
        glrm 1;
        start leaf;
        nt leaf = "a";
        ''',
        vocab,
    )
    middle = _prepare_glrm(
        '''
        glrm 1;
        start middle;
        extern grammar leaf;
        nt middle = leaf?;
        ''',
        vocab,
        subgrammars={"leaf": leaf},
    )
    middle = glrmask.Module.load(middle.save(), vocab)
    composed = _glrm(
        '''
        glrm 1;
        start document;
        extern grammar middle;
        nt document = "X" middle "!";
        ''',
        vocab,
        subgrammars={"middle": middle},
    )
    monolithic = _glrm(
        '''
        glrm 1;
        start document;
        nt item = "a";
        nt document = "X" item? "!";
        ''',
        vocab,
    )

    for sequence in ([0], [1], [2, 4], [2, 3, 4]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_accepting()
        assert expected.is_accepting()


def test_external_subgrammar_handles_ignore_inside_fused_boundary_token() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"X a!",
            1: b"Xa!",
            2: b"X",
            3: b" ",
            4: b"a",
            5: b"!",
            6: b" a",
            7: b"a!",
        }
    )
    child = _prepare_glrm(
        '''
        glrm 1;
        start child;
        nt child = "a";
        ''',
        vocab,
    )
    composed = _glrm(
        '''
        glrm 1;
        start document;
        ignore WS;
        t WS = " "+;
        extern grammar payload;
        nt document = "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    monolithic = _glrm(
        '''
        glrm 1;
        start document;
        ignore WS;
        t WS = " "+;
        nt document = "X" "a" "!";
        ''',
        vocab,
    )

    for sequence in ([0], [1], [2, 3, 4, 5], [2, 6, 5], [2, 3, 7]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_accepting()
        assert expected.is_accepting()


def test_same_compiled_child_can_fill_two_external_subgrammars() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"<a>,<b>",
            1: b"<b>,<a>",
            2: b"<",
            3: b"a",
            4: b"b",
            5: b">,<",
            6: b">",
        }
    )
    child = _prepare_glrm(
        '''
        glrm 1;
        start child;
        nt child = "a" | "b";
        ''',
        vocab,
    )
    composed = _glrm(
        '''
        glrm 1;
        start document;
        extern grammar left;
        extern grammar right;
        nt document = "<" left ">,<" right ">";
        ''',
        vocab,
        subgrammars={"left": child, "right": child},
    )
    monolithic = _glrm(
        '''
        glrm 1;
        start document;
        nt child = "a" | "b";
        nt document = "<" child ">,<" child ">";
        ''',
        vocab,
    )

    for sequence in ([0], [1], [2, 3, 5, 4, 6], [2, 4, 5, 3, 6]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_accepting()
        assert expected.is_accepting()


def test_external_subgrammar_acceptance_supports_caller_end_policy() -> None:
    end_token = 1000
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"Xa!", 1: b"X", 2: b"a", 3: b"!"}
    )
    child = _prepare_glrm(
        """
        glrm 1;
        start child;
        nt child = "a";
        """,
        vocab,
    )
    composed = _glrm(
        """
        glrm 1;
        start document;
        extern grammar payload;
        nt document = "X" payload "!";
        """,
        vocab,
        subgrammars={"payload": child},
    )

    for sequence in ([0], [1, 2, 3]):
        state = composed.start()
        for token in sequence:
            state.commit_token(token)
        assert state.is_accepting()
        mask = state.mask(end_token + 1)
        assert not mask[end_token]
        mask[end_token] = True
        assert mask[end_token]
        assert state.forced_tokens() == []

    loaded = glrmask.Constraint.load(composed.save(), vocab)
    loaded_state = loaded.start()
    loaded_state.commit_token(0)
    assert loaded_state.is_accepting()

def test_manual_subgrammar_composition_api_is_not_exposed() -> None:
    assert not hasattr(glrmask.Constraint, "compose_subgrammars")


def test_compiled_parent_can_be_cached_and_rebound() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes({
        0: b"<",
        1: b">",
        2: b"a",
        3: b"b",
        4: b"x",
    })
    parent = glrmask.Module(glrmask.Grammar.from_glrm(
        """
        glrm 1;
        extern grammar payload;
        start document;
        nt document = "x" | "<" payload ">";
        """
    ))
    parent.precompile(vocab)
    child_a = _prepare_glrm(
        'glrm 1; start value; nt value = "a";',
        vocab,
    )
    child_b = _prepare_glrm(
        'glrm 1; start value; nt value = "b";',
        vocab,
    )

    with_a = _bound_snapshot(parent, "payload", child_a).compile(vocab)
    with_b = _bound_snapshot(parent, "payload", child_b).compile(vocab)

    state = with_a.start()
    state.commit_token(0)
    state.commit_token(2)
    state.commit_token(1)
    assert state.is_accepting()

    state = with_b.start()
    state.commit_token(0)
    state.commit_token(3)
    state.commit_token(1)
    assert state.is_accepting()

    loaded_parent = glrmask.Module.load(parent.save(), vocab=vocab)
    loaded_with_a = _bound_snapshot(loaded_parent, "payload", child_a).compile(vocab)
    state = loaded_with_a.start()
    state.commit_token(0)
    state.commit_token(2)
    state.commit_token(1)
    assert state.is_accepting()
