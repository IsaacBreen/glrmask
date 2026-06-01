# Runtime Commit state space

## Live map

The live Commit map is keyed by tokenizer state rather than parser state. This is deliberate. The tokenizer state records how much of the next terminal has already been scanned; the parser frontier records which grammar stacks remain viable after completed terminals only.

A single tokenizer state may own a large parser frontier. A single parser frontier may also appear under multiple tokenizer states when the same parser stacks can be paired with different incomplete scanner contexts. That is why `ParserStatesByTokenizer = FxHashMap<u32, ParserGSS>` remains a local alias and why the public runtime state uses a deterministic `BTreeMap<u32, ParserGSS>`.

## Publication invariant

The runtime state should be read as:

```text
for tokenizer state q, parser frontier G:
  q says where lexical scanning may resume;
  G says which parser stacks are possible before the next completed terminal;
  G annotations say which longest-match exclusions are delayed by q.
```

No module outside Commit should mutate this structure directly during byte acceptance.
