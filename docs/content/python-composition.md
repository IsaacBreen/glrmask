```python
# [hidden]
from glrmask import Grammar, Module, Vocab
# Replace with your tokenizer's token bytes
token_bytes = {i: bytes([i]) for i in range(256)}
# [/hidden]
# Token ID -> exact bytes from tokenizer
vocab = Vocab(token_bytes)

weather = Grammar.from_json_schema({
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"],
    "additionalProperties": False,
}).compile(vocab)

parent = Grammar.from_glrm('''
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS
              "</weather>"
''')
call = Module(parent)
call.bind("ARGS", weather)
constraint = call.compile(vocab)
```
