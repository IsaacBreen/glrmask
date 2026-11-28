```python
# [hidden]
from glrmask import Grammar, Vocab
import numpy as np

# Replace with your tokenizer's token bytes
token_bytes = {i: bytes([i]) for i in range(256)}
EOS = 256
rng = np.random.default_rng(0)
def model(output):
    return rng.standard_normal(257)
def sample(logits, mask, accepting):
    allowed = np.zeros(len(logits), dtype=bool)
    bits = np.unpackbits(mask.astype('<u4').view(np.uint8), bitorder='little')
    allowed[:EOS] = bits[:EOS]
    allowed[EOS] = accepting
    if not allowed.any():
        raise RuntimeError('no allowed token and cannot stop')
    return int(np.argmax(np.where(allowed, logits, -np.inf)))
# [/hidden]
grammar = Grammar.from_json_schema({
    "type": "object",
    "properties": {
        "color": {
            "enum": ["red", "green", "blue"]
        }
    },
    "required": ["color"],
    "additionalProperties": False,
})

# Token ID -> exact bytes from tokenizer
vocab = Vocab(token_bytes)
state = grammar.compile(vocab).start()

output = []
while True:
    logits = model(output)
    mask = state.mask()
    token = sample(
        logits, mask, state.is_accepting()
    )
    if token == EOS:
        break
    if not state.commit_token(token):
        raise RuntimeError("Rejected token")
    output.append(token)
# [hidden]
assert state.is_accepting()
print(bytes(output).decode())
# [/hidden]
```
