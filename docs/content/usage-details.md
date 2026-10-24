### Composing grammars

A grammar can declare externs, named slots that you bind later: `extern grammar NAME` takes a grammar, and
`extern token NAME` takes token IDs. Each grammar keeps its own rules, including the whitespace it ignores. A `Module`
holds a grammar, its bindings and any precompiled work, so parts you reuse, such as each tool's schema, are compiled once:

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({i: bytes([i]) for i in range(256)})
weather = glrmask.Module(glrmask.Grammar.from_json_schema(
    '{"type": "object", "properties": {"city": {"type": "string"}},'
    ' "required": ["city"], "additionalProperties": false}'))
weather.precompile(vocab)

call = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS "</weather>"
'''))
call.bind("ARGS", weather)
state = call.compile(vocab).start()
assert state.commit_bytes(b'<weather>{"city": "Perth"}</weather>')
assert state.is_accepting()
```

`Expr.choice`, `Expr.sequence` and `Expr.literal` combine grammars without writing a new one.

### End tokens

GLRMask has no end-token setting. To require an end-of-sequence token, put it after the grammar:
`Grammar.from_expr(Expr.sequence(grammar, TokenSet([eos_id])))`. `is_accepting()` says whether the text so far is a
complete match, and `is_terminated()` whether nothing more can be committed.

### Mask strategies

`MaskStrategy` trades compile time against mask speed. `ParserDwa` is slowest to compile, with the fastest masks;
`TokenTrie` is fastest to compile, with slower masks; `Auto`, the default, reuses precompiled work and otherwise uses
`ParserDwa`, or `TokenTrie` when that would be too large. `BoundaryMaskStrategy` makes the same choice where a grammar
meets a bound extern. Every strategy produces the same masks.

The [full guide](GUIDE.md) covers generation, composition, token sets, strategies, markers and errors.
