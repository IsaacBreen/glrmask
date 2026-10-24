# Token sets

A `TokenSet` is a set of token IDs. In a grammar it matches exactly one token from the set. It matches the token itself,
not its bytes: two tokens with the same bytes are different tokens, and committing text with `commit_bytes` never matches
a token set. Token sets are how a grammar uses special tokens, such as end-of-sequence or tool-call tokens.

```python
import glrmask

begin_call = 128
vocab = glrmask.Vocab.from_id_to_bytes({i: bytes([i]) for i in range(128)}, special_token_ids=[begin_call])
grammar = glrmask.Grammar.from_glrm('''
    extern token BEGIN_CALL
    start reply
    nt reply = /[a-z ]*/ | BEGIN_CALL "weather"
''')
grammar.bind("BEGIN_CALL", begin_call)
state = grammar.compile(vocab).start()
assert state.commit_token(begin_call)
assert state.commit_bytes(b"weather")
```

## Creating a token set

| | From IDs | From ranges |
|---|---|---|
| Rust | `TokenSet::new([1, 2, 3])` | `TokenSet::from_ranges([0..10])` or `[0..=9]` |
| Python | `TokenSet([1, 2, 3])` | `TokenSet.from_ranges([range(0, 10)])` |
| Node.js | `new TokenSet([1, 2, 3])` | `TokenSet.fromRanges([{start: 0, end: 10}])` |
| C | `glrmask_token_set_new` | `glrmask_token_set_from_ranges`, with `{first, last}` |
| C++ | `TokenSet({1, 2, 3})` | `TokenSet::from_ranges({{0, 9}})` |

Python and Node.js ranges exclude their end; C and C++ ranges include it. Ranges may overlap or touch, and a range that
is empty or reversed adds nothing. Repeated IDs count once.

For large sets, use ranges. A set of all but one of 131,072 tokens is two ranges, and costs the same to compile and run
as any other two ranges; built from IDs, it means reading 131,071 IDs first.

## Using a token set

- Bind it to an `extern token`, or to an `extern grammar`, in which case it matches one token.
- Put it in an expression: `Expr.sequence(grammar, TokenSet([eos]))` requires the end-of-sequence token after
  `grammar`.
- A single token ID binds the same as a set of one.

In Lark, `<[5]>` matches token 5, `<[1,3-8]>` a list or inclusive range of IDs, `<[^1,3-8]>` any token except those, and
`<[*]>` any token. These IDs are written into the grammar, so they are specific to one vocabulary; an extern keeps the
grammar independent of it.

## Checks

When you compile, every ID in a set must be in the vocabulary, and a range can't run across IDs the vocabulary lacks. An
empty set is allowed and matches nothing.
