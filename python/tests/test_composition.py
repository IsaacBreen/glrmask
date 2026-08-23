import glrmask


def test_from_glrm_grammar_binds_typed_external_subgrammar() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"X", 1: b"ab!", 2: b"Xab!", 3: b"a", 4: b"b", 5: b"!"}
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a" "b";
        ''',
        vocab,
    )
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        extern g payload;
        nt document ::= "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    inline = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        g payload ::= {
            start child;
            nt child ::= "a" "b";
        };
        nt document ::= "X" payload "!";
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
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a" "b";
        ''',
        vocab,
    )
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        extern g payload;
        nt document ::= "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        g inner ::= {
            start child;
            nt child ::= "a" "b";
        };
        nt document ::= "X" inner "!";
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
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt item ::= "a";
        nt child ::= item?;
        ''',
        vocab,
    )
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        extern g payload;
        nt document ::= "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        nt item ::= "a";
        nt document ::= "X" item? "!";
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
    leaf = glrmask.Constraint.from_glrm_grammar(
        '''
        start leaf;
        nt leaf ::= "a";
        ''',
        vocab,
    )
    middle = glrmask.Constraint.from_glrm_grammar(
        '''
        start middle;
        extern g leaf;
        nt middle ::= leaf?;
        ''',
        vocab,
        subgrammars={"leaf": leaf},
    )
    middle = glrmask.Constraint.load(middle.save(), vocab)
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        extern g middle;
        nt document ::= "X" middle "!";
        ''',
        vocab,
        subgrammars={"middle": middle},
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        nt item ::= "a";
        nt document ::= "X" item? "!";
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
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a";
        ''',
        vocab,
    )
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        ignore WS;
        t WS ::= " "+;
        extern g payload;
        nt document ::= "X" payload "!";
        ''',
        vocab,
        subgrammars={"payload": child},
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        ignore WS;
        t WS ::= " "+;
        nt document ::= "X" "a" "!";
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
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a" | "b";
        ''',
        vocab,
    )
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        extern g left;
        extern g right;
        nt document ::= "<" left ">,<" right ">";
        ''',
        vocab,
        subgrammars={"left": child, "right": child},
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        nt child ::= "a" | "b";
        nt document ::= "<" child ">,<" child ">";
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


def test_external_subgrammar_preserves_out_of_vocab_end_token() -> None:
    end_token = 1000
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"Xa!", 1: b"X", 2: b"a", 3: b"!"}
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a";
        ''',
        vocab,
    )
    composed = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        extern g payload;
        nt document ::= "X" payload "!";
        ''',
        vocab,
        end_token_ids=[end_token],
        subgrammars={"payload": child},
    )

    for sequence in ([0], [1, 2, 3]):
        state = composed.start()
        for token in sequence:
            state.commit_token(token)
        mask = state.mask().tolist()
        assert mask[end_token]
        assert state.forced() == [end_token]
        state.commit_token(end_token)
        assert state.is_accepting()

    loaded = glrmask.Constraint.load(composed.save(), vocab)
    loaded_state = loaded.start()
    loaded_state.commit_token(0)
    assert loaded_state.forced() == [end_token]


def test_legacy_manual_subgrammar_composition_api_is_not_exposed() -> None:
    assert not hasattr(glrmask.Constraint, "compose_subgrammars")


def test_compose_compiled_subgrammars_links_cached_parent_and_child() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {0: b"X", 1: b"ab!", 2: b"Xab!", 3: b"a", 4: b"b", 5: b"!"}
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a" "b";
        ''',
        vocab,
    )
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t PAYLOAD ::= @token(100);
        nt document ::= "X" PAYLOAD "!";
        ''',
        vocab,
    )
    composed = parent.compose_compiled_subgrammars({"PAYLOAD": child}, vocab)
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        nt document ::= "X" "a" "b" "!";
        ''',
        vocab,
    )

    actual = composed.start()
    expected = monolithic.start()
    assert actual.mask().tolist() == expected.mask().tolist()
    actual.commit_token(2)
    expected.commit_token(2)
    assert actual.is_finished()
    assert expected.is_finished()
