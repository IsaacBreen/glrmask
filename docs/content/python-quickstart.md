```python
import numpy as np
import glrmask

grammar = glrmask.Grammar.from_json_schema({
    "type": "object",
    "properties": {"color": {"enum": ["red", "green", "blue"]}},
    "required": ["color"],
    "additionalProperties": False,
})

# One token per byte, plus EOS. With a real model, build this from its tokenizer.
vocab = glrmask.Vocab({i: bytes([i]) for i in range(256)})
EOS = 256

state = grammar.compile(vocab).start()
rng = np.random.default_rng(0)
output = []
while True:
    logits = rng.standard_normal(257)  # stand-in for your model
    allowed = np.zeros(257, dtype=bool)
    allowed[:state.mask_bit_len] = state.mask(dtype=bool)
    allowed[EOS] = state.is_accepting()
    token = int(np.argmax(np.where(allowed, logits, -np.inf)))
    if token == EOS:
        break
    state.commit_token(token)
    output.append(token)

print(bytes(output).decode())  # e.g. {"color": "blue"}
```
