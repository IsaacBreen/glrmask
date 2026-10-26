import glrmask


def test_is_terminated_tracks_commit_admission():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a"})
    state = glrmask.Grammar.from_lark('start: "a"').compile(vocab).start()
    assert not state.is_terminated()
    assert state.commit_token(0)
    assert state.is_terminated()
    # A rejected commit leaves the state dead, which is also terminated.
    assert not state.commit_token(0)
    assert state.is_terminated()


def test_empty_commit_on_failed_state_returns_false():
    vocab = glrmask.Vocab.from_id_to_bytes({0: b"a"})
    grammar = glrmask.Grammar.from_lark('start: "a"')

    # Fresh state: empty commits return true
    fresh = grammar.compile(vocab).start()
    assert not fresh.is_terminated()
    assert not fresh.is_accepting()
    assert fresh.commit_bytes(b"")
    assert fresh.commit_tokens([])

    # Finished state: empty commits return true
    finished = grammar.compile(vocab).start()
    assert finished.commit_bytes(b"a")
    assert finished.is_accepting()
    assert finished.is_terminated()
    assert finished.commit_bytes(b"")
    assert finished.commit_tokens([])

    # Failed state: committing a disallowed token
    failed = grammar.compile(vocab).start()
    assert not failed.commit_token(999)
    assert failed.is_terminated()
    assert not failed.is_accepting()
    assert not failed.commit_bytes(b"")
    assert failed.is_terminated()
    assert not failed.is_accepting()
    assert not failed.commit_tokens([])
    assert failed.is_terminated()
    assert not failed.is_accepting()

