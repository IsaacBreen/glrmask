```python
import glrmask

vocab = glrmask.Vocab({0: b"hello", 1: b" ", 2: b"world"})
grammar = glrmask.Grammar.from_lark('start: "hello" " " "world"')
constraint = grammar.compile(vocab)
state = constraint.start()
for token_id, expected in [(0, [1]), (1, [2]), (2, [4])]:
    assert state.mask().tolist() == expected
    assert state.commit_token(token_id)
assert state.is_accepting()
```
