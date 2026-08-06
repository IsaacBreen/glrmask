import glrmask
import pytest


def test_from_glrm_grammar_binds_typed_external_subgrammar() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"X",
            1: b"ab!",
            2: b"Xab!",
            3: b"a",
            4: b"b",
            5: b"!",
        }
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
    assert actual.is_finished()
    assert expected.is_finished()


def test_compose_subgrammars_matches_monolithic_across_token_boundary() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"X",
            1: b"ab!",
            2: b"a",
            3: b"b",
            4: b"!",
        }
    )
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t SUB ::= @token(999);
        nt document ::= "X" SUB "!";
        ''',
        vocab,
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a" "b";
        ''',
        vocab,
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

    composed = parent.compose_subgrammars([("SUB", child)], vocab)
    expected = monolithic.start()
    actual = composed.start()

    assert actual.mask().tolist() == expected.mask().tolist()
    actual.commit_token(0)
    expected.commit_token(0)
    assert actual.mask().tolist() == expected.mask().tolist()
    assert actual.mask().tolist()[1]

    actual.commit_token(1)
    expected.commit_token(1)
    assert actual.is_finished()
    assert expected.is_finished()


def test_compose_subgrammars_matches_monolithic_for_nullable_child() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"X!",
            1: b"Xa!",
            2: b"X",
            3: b"a",
            4: b"!",
        }
    )
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t SUB ::= @token(999);
        nt document ::= "X" SUB "!";
        ''',
        vocab,
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt item ::= "a";
        nt child ::= item?;
        ''',
        vocab,
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        nt item ::= "a";
        nt document ::= "X" item? "!";
        ''',
        vocab,
    )
    composed = parent.compose_subgrammars([("SUB", child)], vocab)

    for sequence in ([0], [1], [2, 4], [2, 3, 4]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_finished()
        assert expected.is_finished()


def test_nested_nullable_composition_survives_save_load() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"X!",
            1: b"Xa!",
            2: b"X",
            3: b"a",
            4: b"!",
        }
    )
    leaf = glrmask.Constraint.from_glrm_grammar(
        '''
        start leaf;
        nt leaf ::= "a";
        ''',
        vocab,
    )
    middle_parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start middle;
        t LEAF ::= @token(998);
        nt middle ::= LEAF?;
        ''',
        vocab,
    )
    middle = middle_parent.compose_subgrammars([("LEAF", leaf)], vocab)
    middle = glrmask.Constraint.load(middle.save(), vocab)
    outer_parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t MIDDLE ::= @token(999);
        nt document ::= "X" MIDDLE "!";
        ''',
        vocab,
    )
    composed = outer_parent.compose_subgrammars([("MIDDLE", middle)], vocab)
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
        assert actual.is_finished()
        assert expected.is_finished()


def test_compose_subgrammars_handles_ignore_inside_fused_boundary_token() -> None:
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
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        ignore WS;
        t WS ::= " "+;
        t SUB ::= @token(999);
        nt document ::= "X" SUB "!";
        ''',
        vocab,
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a";
        ''',
        vocab,
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
    composed = parent.compose_subgrammars([("SUB", child)], vocab)

    for sequence in ([0], [1], [2, 3, 4, 5], [2, 6, 5], [2, 3, 7]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_finished()
        assert expected.is_finished()


def test_compose_subgrammars_rejects_real_vocab_placeholder_token() -> None:
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"X",
            1: b"a",
            2: b"!",
            3: b"<placeholder>",
        }
    )
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t SUB ::= @token(3);
        nt document ::= "X" SUB "!";
        ''',
        vocab,
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a";
        ''',
        vocab,
    )

    with pytest.raises(Exception, match="non-vocabulary sentinels"):
        parent.compose_subgrammars([("SUB", child)], vocab)


def test_same_compiled_child_can_fill_two_python_placeholders() -> None:
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
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t LEFT ::= @token(998);
        t RIGHT ::= @token(999);
        nt document ::= "<" LEFT ">,<" RIGHT ">";
        ''',
        vocab,
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a" | "b";
        ''',
        vocab,
    )
    monolithic = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        nt child ::= "a" | "b";
        nt document ::= "<" child ">,<" child ">";
        ''',
        vocab,
    )
    composed = parent.compose_subgrammars(
        [("LEFT", child), ("RIGHT", child)],
        vocab,
    )

    for sequence in ([0], [1], [2, 3, 5, 4, 6], [2, 4, 5, 3, 6]):
        expected = monolithic.start()
        actual = composed.start()
        for token in sequence:
            assert actual.mask().tolist() == expected.mask().tolist()
            actual.commit_token(token)
            expected.commit_token(token)
        assert actual.is_finished()
        assert expected.is_finished()


def test_composition_preserves_out_of_vocab_end_token() -> None:
    placeholder_token = 999
    end_token = 1000
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"Xa!",
            1: b"X",
            2: b"a",
            3: b"!",
        }
    )
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t SUB ::= @token(999);
        nt document ::= "X" SUB "!";
        ''',
        vocab,
        end_token_ids=[end_token],
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a";
        ''',
        vocab,
    )
    composed = parent.compose_subgrammars([("SUB", child)], vocab)

    for sequence in ([0], [1, 2, 3]):
        state = composed.start()
        assert not state.mask().tolist()[placeholder_token]
        for token in sequence:
            state.commit_token(token)
        mask = state.mask().tolist()
        assert mask[end_token]
        assert not mask[placeholder_token]
        assert state.forced() == [end_token]
        state.commit_token(end_token)
        assert state.is_finished()

    loaded = glrmask.Constraint.load(composed.save(), vocab)
    loaded_state = loaded.start()
    loaded_state.commit_token(0)
    assert loaded_state.forced() == [end_token]


def test_composition_rejects_placeholder_end_token_id_collision() -> None:
    shared_token = 999
    vocab = glrmask.Vocab.from_id_to_bytes(
        {
            0: b"Xa!",
            1: b"X",
            2: b"a",
            3: b"!",
        }
    )
    parent = glrmask.Constraint.from_glrm_grammar(
        '''
        start document;
        t SUB ::= @token(999);
        nt document ::= "X" SUB "!";
        ''',
        vocab,
        end_token_ids=[shared_token],
    )
    child = glrmask.Constraint.from_glrm_grammar(
        '''
        start child;
        nt child ::= "a";
        ''',
        vocab,
    )

    with pytest.raises(Exception, match="grammar-level end token"):
        parent.compose_subgrammars([("SUB", child)], vocab)
