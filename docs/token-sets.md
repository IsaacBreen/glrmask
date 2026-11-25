# Token sets

A `TokenSet` is a set of token IDs. In a grammar it matches exactly one token from the set. It matches the token itself,
not its bytes: two tokens with the same bytes are different tokens, and committing text with `commit_bytes` never matches
a token set. Grammar source cannot introduce token-level matches: only code binding a TokenSet on a Module can.

```python
import glrmask

begin_call = 128
vocab = glrmask.Vocab({i: bytes([i]) for i in range(128)})
grammar = glrmask.Grammar.from_glrm('''
    extern token BEGIN_CALL
    start reply
    nt reply = /[a-z ]*/ | BEGIN_CALL "weather"
''')
module = glrmask.Module(grammar)
module.bind("BEGIN_CALL", glrmask.TokenSet([begin_call]))
state = module.compile(vocab).start()
assert state.commit_token(begin_call)
assert state.commit_bytes(b"weather")
```

## Creating a token set

| | From IDs | From ranges |
|---|---|---|
| Rust | `TokenSet::new([1, 2, 3])` | `TokenSet::from_ranges([0..10])` or `[0..=9]` |
| Python | `TokenSet([1, 2, 3])` | `TokenSet.from_ranges([range(0, 10)])` |
| Node.js | `new TokenSet([1, 2, 3])` | `TokenSet.fromRanges([{start: 0, end: 10}])` |
| C | `glrmask_token_set_new` | `glrmask_token_set_from_ranges`, with `{start, end}` |
| C++ | `TokenSet({1, 2, 3})` | `TokenSet::from_ranges({{0, 10}})` |

C and C++ use `{start, end}` with `end` excluded, as in every language. Ranges may overlap or touch, and a range that
is empty or reversed adds nothing. Repeated IDs count once.

For large sets, use ranges. A set of all but one of 131,072 tokens is two ranges, and costs the same to compile and run
as any other two ranges; built from IDs, it means reading 131,071 IDs first.

## Using a token set

- Bind it on a Module to an `extern token`, or to an `extern grammar`, in which case it matches one token.
- Put it in an expression: `Expr.sequence(TokenSet([begin_call]), grammar)` requires a tool-call marker before
  `grammar`.
- Use a one-element TokenSet for a single token ID. Token sets cannot be bound on a Grammar or included in Grammar.from_expr, even indirectly.

## Checks

IDs may be absent from the vocabulary or have no byte spelling. An empty set is allowed and matches nothing.
