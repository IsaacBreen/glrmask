## Generating

Use one `ConstraintState` per sequence. Allow EOS when `is_accepting()` is true; stop if EOS is sampled. `is_terminated()` indicates that no further tokens or bytes are accepted. A rejected commit terminates the state.

For PyTorch serving, allocate a reusable packed mask on the CPU, with `V` equal to the logits width (at least each state's mask bit length):

```python
bitmask = torch.empty((len(states), (V + 31) // 32), dtype=torch.int32, pin_memory=True)
for i, state in enumerate(states):
    state.fill_mask(bitmask[i].numpy().view(np.uint32))
mask_on_device = bitmask.to(device, non_blocking=True)
# Apply the packed mask on the device. Wait for the transfer before reusing the CPU buffer.
```

### Forced bytes

`forced_bytes()` gives the bytes that must come next, stopping before a choice, an accepting position or a token-set item. To feed these bytes to a token-based model, tokenize them and commit the resulting tokens. Long forced runs may be returned in parts.

{{composition}}

### Modules and expressions

A grammar extern accepts a grammar, module, constraint, expression or token set. A token extern accepts only a token set. Each extern is bound once; copy a module to use different bindings. Bindings capture the supplied values. A model token can span the boundary between grammar parts.

`precompile(vocab)` compiles as much of a module as possible with its current bindings. Precompiled work is tied to the vocabulary. Modules can be saved with unbound slots and loaded later.

Combine parts using `Expr.choice`, `Expr.sequence` and `Expr.literal`. Use `Grammar.from_expr` for expressions made entirely of grammar and literal parts; bind expressions containing compiled parts or token sets on a module.

## Token sets

A `TokenSet` matches one of its token IDs, including special tokens with no bytes. Token IDs can be allowed through an extern binding even if absent from the vocabulary. See [token sets](docs/token-sets.md).

## Mask strategies

`FAST_MASKS` favors fast mask computation; `FAST_COMPILE` favors fast compilation. `AUTO`, the default, currently prefers `FAST_MASKS`. Pass the preference when compiling or precompiling.

## Grammar formats

JSON Schema supports drafts 4, 6, 7, 2019-09 and 2020-12. Generated JSON follows the layout of Python's default `json.dumps`, including a space after commas and colons. Unsupported features raise errors.

See the [GLRM format](docs/glrm-format.md) and [Lark dialect](docs/lark-dialect.md) for grammar syntax and extensions.
