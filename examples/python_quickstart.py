"""Incremental llama.cpp generation with a GLRMask JSON Schema constraint."""

import ctypes
import numpy as np
import llama_cpp
from torch import from_numpy
from torch.distributions import Categorical

import glrmask


llm = llama_cpp.Llama(model_path="model.gguf", logits_all=True)
llama_vocab = llama_cpp.llama_model_get_vocab(llm.model)
tokens = range(llm.n_vocab())
end_token_ids = [
    token for token in tokens
    if llama_cpp.llama_vocab_is_eog(llama_vocab, token)
]
end_tokens = set(end_token_ids)

def token_bytes(token):
    size = -llama_cpp.llama_token_to_piece(llama_vocab, token, None, 0, 0, False)
    buffer = ctypes.create_string_buffer(size)
    length = llama_cpp.llama_token_to_piece(llama_vocab, token, buffer, size, 0, False)
    return buffer.raw[:length]

vocab = glrmask.Vocab.from_id_to_bytes({
    token: token_bytes(token)
    for token in tokens
    if token not in end_tokens
    and not (
        llama_cpp.llama_vocab_get_attr(llama_vocab, token)
        & (llama_cpp.LLAMA_TOKEN_ATTR_CONTROL | llama_cpp.LLAMA_TOKEN_ATTR_UNUSED)
    )
})

schema = '{"type":"string","enum":["positive","negative","neutral"]}'
constraint = glrmask.Constraint.from_json_schema(
    schema,
    vocab,
    end_token_ids=end_token_ids,
)
state = constraint.start()

get_logits = lambda: llm.scores[llm.n_tokens - 1]
sample = lambda logits: Categorical(logits=from_numpy(logits)).sample().item()

prompt = "Classify this review: The story dragged badly. Sentiment: "
llm.eval(llm.tokenize(prompt.encode()))

generated = []

for _ in range(64):
    logits = get_logits()
    mask = state.mask(llm.n_vocab())
    logits[~mask] = -np.inf

    token = sample(logits)
    llm.eval([token])
    state.commit_token(token)
    generated.append(token)

    if token in end_tokens:
        break

print(llm.detokenize(generated).decode())
