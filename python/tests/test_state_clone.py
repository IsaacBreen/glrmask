import glrmask
import pytest


def test_clone_preserves_prefix_and_keeps_mutations_independent():
    vocab = glrmask.Vocab.from_dict({b"a": 0, b"b": 1, b"c": 2, b"<eos>": 3})
    constraint = glrmask.Grammar.from_expr(glrmask.Expr.sequence(glrmask.Grammar.from_ebnf('start ::= "a" ("b" | "c")'), glrmask.TokenSet([3]))).compile(vocab)
    state = constraint.start()
    assert state.commit_token(0)
    branch = state.copy()
    assert branch.mask().tolist() == state.mask().tolist()
    assert branch.commit_token(1)
    assert not branch.is_accepting()
    assert not state.is_accepting()
    assert state.mask().tolist() == [False, True, True, False]
    assert branch.commit_token(3)
    for ended in [branch, branch.copy()]:
        assert ended.is_accepting() and not ended.mask().any()
        assert not ended.mask().any()
        assert not ended.commit_token(0)
        assert not ended.commit_bytes(b"a")
    assert not state.is_accepting() and not (not state.mask().any())
    assert state.mask().tolist() == [False, True, True, False]
    assert state.commit_token(2)
    assert not state.is_accepting()
    assert state.commit_token(3)
    assert state.is_accepting()


def test_clone_preserves_rejection_without_changing_saved_prefix():
    vocab = glrmask.Vocab.from_dict({b"a": 0, b"b": 1})
    constraint = glrmask.Grammar.from_ebnf('start ::= "a"').compile(vocab)
    state = constraint.start()
    snapshot = state.copy()
    assert not state.commit_token(1)
    assert (not state.mask().any())
    assert (not state.copy().mask().any())
    assert snapshot.mask().tolist() == [True, False]
    assert snapshot.commit_token(0)
    assert snapshot.is_accepting()
