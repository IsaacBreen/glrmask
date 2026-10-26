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
