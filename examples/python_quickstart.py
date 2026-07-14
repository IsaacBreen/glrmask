"""Minimal public Python first-run example for glrmask."""

import glrmask

vocab = glrmask.Vocab.from_dict(
    {
        b"hello": 0,
        b" ": 1,
        b"world": 2,
    }
)
constraint = glrmask.Constraint.from_ebnf(
    'start ::= "hello" " " "world"',
    vocab,
)
state = constraint.start()

assert state.mask().tolist() == [True, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, False]
state.commit_token(1)
assert state.mask().tolist() == [False, False, True]
state.commit_token(2)
assert state.is_finished()

print("accepted: hello world")
