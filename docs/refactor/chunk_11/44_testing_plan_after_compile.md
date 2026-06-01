# Testing plan after compile repair

After the no-compile architectural phase ends, use this test order.

## Unit-level

- DenseMaskAcc intersection tests.
- bitset helper tests for EOS and ordinary tokens.
- Commit parser-advance template/reference equivalence tests.
- Commit mask assertion enabled on small examples.

## Integration-level

- EBNF hello-world example.
- JSON object schema example.
- Ambiguous grammar with multiple parser stacks.
- Lexer partial-match test where Commit bytes do not immediately emit a terminal.
- EOS completion test.

## Differential

For each small state and each vocabulary token:

```text
membership = token in Mask(S)
commit_ok = Commit(clone(S), bytes(token)).is_ok()
assert membership == commit_ok
```

Run the same differential with template DFA advance enabled and disabled.

## Performance smoke

Only after correctness:

- compare mask timings before/after split;
- compare commit timings before/after split;
- ensure profiling output fields are still emitted.
