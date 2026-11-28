GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Constraint`: a compiled grammar, ready to use.
- `ConstraintState`: the state of one generated sequence.
- `Module`: a partially compiled grammar ([see below](#composing-grammars)).

For Hugging Face tokenizers, use `Vocab.from_huggingface(tokenizer)` in Python, or `Vocab::from_tokenizer_file(path, None)` in Rust (enable `hf` and import `VocabFromTokenizer`). Pass the model vocabulary size to include padded IDs. For other tokenizers, supply their exact token bytes. The quickstart also needs the model's EOS ID. The model produces logits; the sampling routine must apply the packed mask and permit EOS only when the state is accepting.
