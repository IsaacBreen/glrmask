```bash
python -m pip install glrmask llama-cpp-python torch
```

```python
import numpy as np
from llama_cpp import Llama
from torch import from_numpy
from torch.distributions import Categorical

import glrmask

llm = Llama(model_path="model.gguf", logits_all=True)
vocab = glrmask.Vocab.from_llama_cpp(llm)
end_token_ids = vocab.llama_cpp_end_token_ids

schema = {"type": "string", "enum": ["positive", "negative", "neutral"]}
constraint = glrmask.Grammar.from_json_schema(schema).compile(
    vocab,
    end_tokens=end_token_ids,
    optimization=glrmask.Optimization.AUTO,
)

prompt = "Classify this review: The story dragged badly. Sentiment: "
input_tokens = llm.tokenize(prompt.encode())
llm.reset()
llm.eval(input_tokens)

state = constraint.start()
generated = []

for _ in range(64):
    logits = llm.scores[llm.n_tokens - 1]
    mask = state.mask(llm.n_vocab())
    logits[~mask] = -np.inf
    token_id = Categorical(logits=from_numpy(logits)).sample().item()

    llm.eval([token_id])
    generated.append(token_id)
    state.commit_token(token_id)
    if token_id in end_token_ids:
        break

print(llm.detokenize(generated).decode())
```

`state.mask()` returns a NumPy Boolean array indexed by model token ID. Pass a size when the model's logits vector is wider than the constraint's natural token coordinate.
