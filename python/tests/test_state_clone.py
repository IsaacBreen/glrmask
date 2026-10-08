import glrmask


def test_clone_preserves_prefix_and_keeps_mutations_independent():
    vocab = glrmask.Vocab.from_dict({b"a": 0, b"b": 1, b"c": 2, b"<eos>": 3})
    constraint = glrmask.Grammar.from_ebnf('start ::= "a" ("b" | "c")').compile(
        vocab, end_tokens=[3]
    )
    state = constraint.start()
    state.commit_token(0)
    branch = state.clone()
    assert branch.mask().tolist() == state.mask().tolist()
    branch.commit_token(1)
    assert branch.is_accepting()
    assert not state.is_accepting()
    assert state.mask().tolist() == [False, True, True, False]
    branch.commit_token(3)
    assert branch.is_terminated()
    assert branch.clone().is_terminated()
    assert not state.is_terminated()
    state.commit_token(2)
    assert state.is_accepting()


def test_clone_preserves_rejection_without_changing_saved_prefix():
    vocab = glrmask.Vocab.from_dict({b"a": 0, b"b": 1})
    constraint = glrmask.Grammar.from_ebnf('start ::= "a"').compile(vocab)
    state = constraint.start()
    snapshot = state.clone()
    try:
        state.commit_token(1)
    except ValueError:
        pass
    assert state.is_rejected()
    assert state.clone().is_rejected()
    assert snapshot.mask().tolist() == [True, False]
    snapshot.commit_token(0)
    assert snapshot.is_accepting()
