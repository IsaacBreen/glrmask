GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Constraint`: a compiled grammar, ready to use.
- `ConstraintState`: the state of one generated sequence.
- `Module`: a partially compiled grammar ([see below](#composing-grammars)).

In the quickstart, supply the tokenizer's exact token bytes and the model's EOS ID. The model produces logits; the sampling routine must apply the packed mask and permit EOS only when the state is accepting.
