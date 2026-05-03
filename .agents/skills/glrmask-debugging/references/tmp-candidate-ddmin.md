# Temporary-Candidate DDMin Runbook

This note captures the useful mechanics from the `Github_hard---o1052` mask/commit MRE reduction.

The key trick is to compile the Rust test once, but make the active artifacts come from stable `/tmp` files. The minimizer then overwrites those files and reruns the same release test binary. This avoids recompiling for every schema, prefix, grammar, or vocab candidate.

## Rust Test Hook

Use inline originals as fallbacks so the test remains self-contained after the temporary reducer files are deleted:

```rust
let vocab_json = fs::read_to_string("/tmp/glrmask_o1052_vocab_candidate.json")
    .unwrap_or_else(|_| {
        let vocab_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".cache/vocab_cache/llama3_vocab.json");
        fs::read_to_string(&vocab_path).expect("read cached vocab")
    });

let schema_json =
    fs::read_to_string("/tmp/glrmask_o1052_schema_candidate.json").unwrap_or_else(|_| {
        r#"{
  "type": "object",
  "properties": {
    "gender": {
      "type": "string",
      "enum": ["mand", "kvinde", "ukendt"]
    }
  }
}
"#
        .to_string()
    });

let prefix = fs::read("/tmp/glrmask_o1052_prefix_candidate.bin")
    .unwrap_or_else(|_| b"{\"gender\": \"".to_vec());
```

Then keep the oracle exact. For a mask false-negative:

```rust
let mut mask_state = constraint.start();
mask_state.commit_bytes(&prefix).unwrap();
let mask_accepts = token_allowed(&mask_state.mask(), disputed_token_id as usize);

let mut commit_state = constraint.start();
commit_state.commit_bytes(&prefix).unwrap();
let commit_accepts = commit_state.commit_bytes(b"mand").is_ok();

assert_eq!(
    (mask_accepts, commit_accepts),
    (true, true),
    "token b\"mand\" should be mask-visible because commit_bytes accepts it",
);
```

Build and run in release mode:

```bash
cargo test -p glrmask --release --test integration test_name -- --ignored --nocapture
```

After reduction, inline the minimized artifacts back into the test and remove the `/tmp` overrides.

## Reducer Oracle

The reducer should detect the exact mismatch, not merely any test failure:

```python
cmd = [
    "cargo", "test", "-q", "-p", "glrmask", "--release",
    "--test", "integration", "test_json_schema_enum_mand_mask_false_negative",
    "--", "--ignored", "--nocapture",
]

def interesting():
    out = subprocess.run(
        cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=120,
    ).stdout
    return "left: (false, true)" in out and "right: (true, true)" in out
```

Do not treat compile errors, parser errors, or unrelated panics as interesting.

## Vocab First

Minimize vocab before schema and prefix because it controls the cost of every later constraint build.

The sequence that helped on `o1052`:

1. Try only the disputed token.
2. Try obvious local token families, such as tokens containing or starting with the disputed bytes.
3. Reproduce the compiler's vocab partitioning and test partitions or partition combinations.
4. Run ddmin deletion inside the surviving partition set.

For the default seven char-type partitions, `classify_vocab_char_type` lives in `src/compiler/stages/id_map_and_terminal_dwa/classify.rs`. Reimplement it in Python closely enough for reduction, or temporarily instrument Rust if exactness matters.

The `o1052` run found:

- one-token vocab with token `1969` / `b"mand"` did not reproduce
- lowercase / ASCII-alpha local subsets did not reproduce
- compiler partition P2 alone did reproduce
- all other char-type partitions were removable
- ddmin inside P2 reduced the vocab from `76848` tokens to much smaller surviving sets
- after the schema was replaced by minimized GLRM, another vocab pass reduced the survivor set much further; always recurse back to vocab after grammar/schema cuts

## DDMin Skeleton

Use deletion ddmin from a known-interesting set. Subset selection can be useful, but deletion preserves distributed interactions better.

```python
def write_vocab(ids):
    ids = sorted(set(ids) | {KEEP}, key=lambda x: int(x))
    path.write_text(
        json.dumps({i: full[i] for i in ids}, separators=(",", ":")) + "\n"
    )

def try_candidate(ids, label):
    write_vocab(ids)
    ok = interesting()
    print(f"{'KEEP' if ok else 'drop'} size={len(set(ids))} {label}", flush=True)
    return ok

cur = sorted(initial_ids, key=lambda x: int(x))
assert try_candidate(cur, "baseline")

while True:
    ids = [i for i in cur if i != KEEP]
    if not ids:
        break

    progress = False
    n = 2
    while n <= len(ids):
        chunk = max(1, len(ids) // n)
        removed = False

        for start in range(0, len(ids), chunk):
            group = set(ids[start:start + chunk])
            cand = [i for i in cur if i not in group]

            if try_candidate(cand, f"delete {start}:{start + chunk} of {len(ids)} n={n}"):
                cur = sorted(set(cand) | {KEEP}, key=lambda x: int(x))
                progress = True
                removed = True
                break

        if removed:
            break
        if chunk == 1:
            break
        n *= 2

    if not progress:
        break
```

For schemas, use the same skeleton over root keys, property keys, nested object keys, enum values, bounds, and prefix fields. After any accepted deletion, restart from the smaller artifact.

## Recursive Loop

Do not treat vocab, schema, prefix, and grammar as independent one-shot phases. Cuts in one artifact can make new cuts possible in another.

The `o1052` sequence was:

1. Add `/tmp` hooks for schema, prefix, and vocab.
2. Reduce root schema metadata and obvious dead properties.
3. Switch back to vocab because the smaller schema made oracle runs cheaper.
4. Reduce by compiler vocab partition, then ddmin the surviving token set.
5. Dump generated GLRM with `GLRMASK_PRINT_GRAMMAR_GLRM=1`.
6. Add a temporary GLRM `/tmp` hook and reduce grammar directly.
7. Re-run vocab ddmin after the GLRM cut; this found a much smaller vocab than the schema-based grammar needed.
8. Inline the final GLRM, prefix, and vocab; remove the `/tmp` hooks.

The important lesson is step 7. If a GLRM reduction removes terminals or alternatives, rerun vocab minimization immediately.

## Inlining Vocab

During reduction, hex JSON is convenient because it mirrors cached vocab files. Once the vocab is small enough to inline in Rust, prefer readable token strings and assign compact IDs automatically:

```rust
let vocab_entries = [
    "mand",
    "owa\u{0107}",
];
let vocab = Vocab::new(
    vocab_entries
        .iter()
        .enumerate()
        .map(|(token_id, token)| (token_id as u32, token.as_bytes().to_vec()))
        .collect(),
    None,
);
```

Use byte literals only if a survivor token is not valid UTF-8. Avoid keeping hex decoding or original LLM token IDs in the final MRE unless either is load-bearing.

## Practical Notes

Keep a small log of accepted and rejected cuts. The point of recursive minimization is not just "smaller"; it is evidence that each remaining piece survived deletion, weakening, or literalization.

When candidate runs become cheap enough, switch from coarse partition cuts to nested cuts immediately. When a large deletion unexpectedly fixes the bug, test single deletions and random or semantic buckets to distinguish one load-bearing token from aggregate vocab-shape effects.

If you interrupt a reducer, remember that the `/tmp` file may contain the last rejected candidate, not the last accepted candidate. Either write every accepted candidate to a checkpoint file immediately, or reconstruct/replay the accepted cuts before continuing. A simple pattern is:

```python
if ok:
    cur = sorted(set(cand) | {KEEP}, key=lambda x: int(x))
    write_vocab(cur)
    checkpoint_path.write_text(vocab_path.read_text())
```
