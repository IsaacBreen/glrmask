import glrmask


def test_is_terminated_tracks_commit_admission():
    vocab = glrmask.Vocab({0: b"a"})
    state = glrmask.Grammar.from_lark('start: "a"').compile(vocab).start()
    assert not state.is_terminated()
    assert state.commit_token(0)
    assert state.is_terminated()
    # A rejected commit leaves the state dead, which is also terminated.
    assert not state.commit_token(0)
    assert state.is_terminated()


def test_empty_commit_on_failed_state_returns_false():
    vocab = glrmask.Vocab({0: b"a"})
    grammar = glrmask.Grammar.from_lark('start: "a"')

    # Fresh state: empty commits return true
    fresh = grammar.compile(vocab).start()
    assert not fresh.is_terminated()
    assert not fresh.is_accepting()
    assert fresh.commit_bytes(b"")
    assert fresh.commit_tokens([]) == 0

    # Finished state: empty commits return true
    finished = grammar.compile(vocab).start()
    assert finished.commit_bytes(b"a")
    assert finished.is_accepting()
    assert finished.is_terminated()
    assert finished.commit_bytes(b"")
    assert finished.commit_tokens([]) == 0

    # Failed state: committing a disallowed token
    failed = grammar.compile(vocab).start()
    assert not failed.commit_token(999)
    assert failed.is_terminated()
    assert not failed.is_accepting()
    assert not failed.commit_bytes(b"")
    assert failed.is_terminated()
    assert not failed.is_accepting()
    assert failed.commit_tokens([]) == 0
    assert failed.is_terminated()
    assert not failed.is_accepting()



def test_rejected_commits_terminate_live_state():
    vocab = glrmask.Vocab({0: b"a", 1: b"b", 2: b"z"})
    grammar = glrmask.Grammar.from_lark('start: "a" "b"')
    for strategy in (glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_COMPILE,
                     glrmask.MaskStrategy.FAST_MASKS):
        constraint = grammar.compile(vocab, mask_strategy=strategy)
        for reject, expected in ((lambda s: s.commit_token(2), False),
                                 (lambda s: s.commit_tokens([0, 2, 1]), 1),
                                 (lambda s: s.commit_bytes(b"az"), False)):
            state = constraint.start()
            assert not state.is_terminated()
            assert reject(state) == expected
            assert state.is_terminated()
            assert not state.is_accepting()
            assert not state.commit_token(1)
