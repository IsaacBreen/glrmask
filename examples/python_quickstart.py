"""Incremental llama.cpp generation with a GLRMask JSON Schema constraint."""

import numpy as np
from llama_cpp import Llama
from torch import from_numpy
from torch.distributions import Categorical

import glrmask


llm = Llama(model_path="model.gguf", logits_all=True)
vocab = glrmask.Vocab.from_llama_cpp(llm)
schema = '{"type":"string","enum":["positive","negative","neutral"]}'
grammar = glrmask.Grammar.from_expr(glrmask.Expr.sequence(
    glrmask.Grammar.from_json_schema(schema),
    glrmask.TokenSet(vocab.llama_cpp_end_token_ids),
))
constraint = grammar.compile(vocab)
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
    assert state.commit_token(token)
    generated.append(token)

    if state.is_terminated():
        break

print(llm.detokenize(generated).decode())
