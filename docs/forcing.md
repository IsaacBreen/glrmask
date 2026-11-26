# Forced bytes

The query returns a grammar-forced byte prefix without changing the state. It stops before acceptance, any possible exact token-set item, or a byte choice. It may return only part of a long run.

```python
import glrmask
vocab = glrmask.Vocab({0: b"hello", 1: b" world"})
state = glrmask.Grammar.from_lark('start: "hello world"').compile(vocab).start()
assert state.forced_bytes() == b"hello world"
assert state.commit_bytes(state.forced_bytes())
assert state.is_accepting()
```

For model inference, tokenize the returned bytes using the model's tokenizer and commit the selected token IDs. Account for tokenizer boundaries; special tokens are separate from text.
