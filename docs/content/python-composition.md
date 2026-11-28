```python
import glrmask
vocab = glrmask.Vocab({i: bytes([i]) for i in range(256)})
weather = glrmask.Grammar.from_json_schema(
    '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}'
).compile(vocab)
call = glrmask.Module(glrmask.Grammar.from_glrm('''extern grammar ARGS
    start call
    nt call = "<weather>" ARGS "</weather>"'''))
call.bind("ARGS", weather)
constraint = call.compile(vocab)
```
