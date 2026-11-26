```python
import glrmask

grammar = glrmask.Grammar.from_lark('''
    start: "hello" " " "world"
''')

# A toy vocabulary of token ID -> bytes. With a real model, build it from the tokenizer.
vocab = glrmask.Vocab({0: b"hello", 1: b" ", 2: b"world"})

constraint = grammar.compile(vocab)
state = constraint.start()

# Each mask says which tokens may come next. Apply it to the logits before sampling.
assert state.mask(dtype=bool).tolist() == [True, False, False]  # only "hello"
assert state.commit_token(0)
assert state.mask(dtype=bool).tolist() == [False, True, False]  # only " "
assert state.commit_token(1)
assert state.commit_token(2)
assert state.is_accepting()
```
