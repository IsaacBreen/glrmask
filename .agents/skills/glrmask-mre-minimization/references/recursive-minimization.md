# How to Minimize Properly, Recursively

This note is based on the actual `glrmask2` commit sequence from `da8bc0bc` forward, with emphasis on the minimization chain for the split-token-boundary MRE in `tests/o82710_repro.rs`.

The important sequence is:

- `1226d1500` inline stuff, remove bracket
- `80ac208d1` simplify
- `e52a6f25e` simplify
- `030b17345` simplify
- `c0f13ce64` simplify
- `d5806cf89` simplify
- `3f893cc32` simplify
- `a8034005c` simplify
- `80093d5ba` simplify
- `fbb8e12f9` ugh
- `9a377d4b6` simplify
- `0e2b3e645` Shrink split-token boundary MRE further
- `3ed3c064d` Simplify split-boundary start rule
- `7e8577b9a` Further shrink split-boundary MRE
- `40f2b4bc9` Minimize split-boundary branch structure
- `8f4e750c8` Inline split-boundary witness logic
- `2f5042d15` Record failed simplification paths for split-boundary MRE
- `05bd6872b` Inline close expression in split-boundary MRE
- `ed3b61215` Tighten standalone split-boundary witness
- `d450b6731` Probe local minimum for split-boundary witness

The note I had before described the pattern, but not the actual mechanics. That was not good enough. The whole point here is to spell out exactly what changed, exactly why it was the right next cut, and exactly what the recursive part means in practice.

---

## The Standard

The standard is not:

- the repro got noticeably smaller
- the code looks cleaner
- the grammar still resembles the original bug source

The standard is:

- every remaining piece survived an explicit attempt to delete it, inline it, weaken it, literalize it, or scale it down

That is what `recursive` means here.

You do not minimize once.

You minimize, then treat the new smaller thing as the next target.

---

## The Oracle

The oracle was not vague similarity to the original JSON-schema path. The oracle was the exact mismatch shape:

```rust
!mask && commit_token && commit_bytes && complete
```

In the earlier phase the test used:

```rust
let (full_mask, full_commit_token, full_commit_bytes, full_complete) =
	classify_constraint(&constraint, &prefix, v, 0, Some(b""));

assert!(!full_mask && full_commit_token && full_commit_bytes && full_complete);
```

Later, after the witness was reduced to a direct standalone mask-vs-commit check, the oracle was intentionally tightened to just:

```rust
assert!(!mask && commit_token && commit_bytes);
```

That was not sloppiness. It was minimization. `complete` had stopped being essential to the witness being studied in that later shape, so it got deleted.

Everything outside the oracle is disposable.

---

## The Mistake Pattern

What I had been doing wrong was preserving provenance instead of preserving the mechanism.

I kept too much of the source story:

- object semantics
- field names
- helper structure
- realistic JSON wrappers
- larger token sets
- larger prefix scaffolding
- original branch layouts
- numbers that were merely inherited from the original witness

Proper minimization aggressively removes all of that.

The final split-boundary family is much closer to the real mechanism:

- repeated exact runs of `a`
- one close expression ending in `"`
- a token that straddles that boundary
- one prefix length sitting at the failure residue

That is the thing to preserve.

---

## Current Nice Form

The current standalone witness family around `scan_o82710_inline_glrm_split_token_boundary` is much nicer than the older note captured.

This is the key standalone form that the later commits converged on:

```rust
let token = b"aa\"";
let vocab = Vocab::new(vec![(0, token.to_vec())], None);
let constraint = Constraint::from_glrm_grammar(r#"
start start;
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} ("a"{0,32} "\"") | A_EXACT{5}) ("a"{0,32} "\"");
"#, &vocab).unwrap();
let prefix = [b'a'; 159];

let mut mask_state = constraint.start();
mask_state.commit_bytes(&prefix).unwrap();
let mask = mask_state.mask().first().is_some_and(|word| (word & 1) != 0);

let mut commit_token_state = constraint.start();
commit_token_state.commit_bytes(&prefix).unwrap();
let commit_token = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
	Ok(Ok(())) => true,
	Ok(Err(_)) => false,
	Err(_) => true,
};

let mut commit_bytes_state = constraint.start();
commit_bytes_state.commit_bytes(&prefix).unwrap();
let commit_bytes = commit_bytes_state.commit_bytes(token).is_ok();

assert!(!mask && commit_token && commit_bytes);
```

This is better than the earlier `json_string_bounded_split_5+ | "," ? ...` witness because it has already deleted several layers that turned out not to matter.

---

## Phase 1: Inline The Original Active Artifact

### `1226d1500`

The first correct move was to stop minimizing through a helper and inline the grammar literal directly in the test.

Before:

```rust
fn a_only_inline_glrm() -> &'static str {
	r#"
start start;

t JSON_STRING_CHAR ::= /a/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
internal t JSON_STRING_CHAR_UPTO_256_0 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= JSON_STRING_CHAR_UPTO_256_0 "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_3 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_4 ::= JSON_STRING_CHAR_UPTO_136_3 "\"";
nt json_string_bounded_split_5 ::= "\"" (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19} JSON_STRING_CHAR_UPTO_CLOSE_4);
nt obj_open_reqmask_0_nc_0 ::= ...
nt start ::= "{" obj_open_reqmask_0_nc_0 "}";
"#
}
```

After, the grammar lived directly inside the active test.

Why this was right:

- it removed reuse pressure
- it made edits local and ugly
- it removed the psychological barrier against deleting pieces

Rule:

**Inline the active artifact first.**

Pretty helpers are hostile to minimization.

---

## Phase 2: Delete Witness Families

### `1226d1500`

Another early correct cut was collapsing a multi-witness scan to one witness.

Before:

```rust
let tokens: [&[u8]; 3] = [b"aaaaa\"", b"aaaa", b"a\""];
```

After:

```rust
let tokens: [&[u8]; 1] = [b"aaaaa\""];
```

Why this was right:

- family comparisons are useful during diagnosis
- family comparisons are usually noise during minimization

Rule:

**Choose one witness and delete the siblings unless the relationship among them is itself the phenomenon.**

---

## Phase 3: Strip Semantics And Wrappers

### `80ac208d1`

The next good move was deleting the object-key story and the realistic JSON-field prefix.

Before:

```rust
nt obj_open_reqmask_0_nc_0 ::= 
	(("\"" "description\"" ": ") json_string_bounded_split_5) obj_open_reqmask_0_c_0 |
	(("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;

let content = &vec![b'a'; 2300];
let mut prefix = b"\"description\": \"".to_vec();
prefix.extend_from_slice(content);
```

After:

```rust
nt obj_open_reqmask_0_nc_0 ::= 
	(json_string_bounded_split_5) obj_open_reqmask_0_c_0 |
	(("\"" "id\"" ": ") json_string) obj_open_reqmask_0_c_1;

let mut prefix = b"\"".to_vec();
prefix.extend_from_slice(&vec![b'a'; 2300]);
```

This was the correct direction because the bug was not about `description`, not about object keys, and not about preserving the source schema story.

Rule:

**Delete semantic names early.**

---

## Phase 4: Delete The Completion Story When It Stops Mattering

### `e52a6f25e`

The tail was reduced to empty.

Before:

```rust
let tail = b", \"id\": \"\"}";
```

After:

```rust
let tail = b"";
```

This is a small diff with a large implication.

It proves that the remaining JSON continuation narrative was fake complexity.

Rule:

**If an empty tail preserves the oracle, the larger tail story should die immediately.**

---

## Phase 5: Replace Named Structure With A Weaker Shell

### `030b17345`

The old explicit continuation graph was replaced with a weaker repetition shell.

Before:

```rust
nt obj_open_reqmask_0_nc_0 ::= (json_string_bounded_split_5) obj_open_reqmask_0_c_0;
nt obj_open_reqmask_0_c_0 ::= (json_string_bounded_split_5) obj_open_reqmask_0_c_0;
nt obj_open_reqmask_0_c_1 ::= (json_string_bounded_split_5) obj_open_reqmask_0_c_1 | ;
nt start ::= obj_open_reqmask_0_nc_0 "}";
```

After:

```rust
nt obj_open_reqmask_0_nc_0 ::= json_string_bounded_split_5+ | obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= "," (json_string_bounded_split_5) obj_open_reqmask_0_c_1 | ;
nt start ::= obj_open_reqmask_0_nc_0 "}";
```

This is proper minimization because it attacks the structure itself, not just the atoms inside the structure.

Rule:

**Do not only shrink inside the current scaffold. Replace the scaffold with a weaker one whenever the oracle survives.**

---

## Phase 6: Collapse The Shell Further

### `c0f13ce64`

The named scaffold then got absorbed almost completely.

Before:

```rust
nt obj_open_reqmask_0_nc_0 ::= json_string_bounded_split_5+ | obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= "," (json_string_bounded_split_5) obj_open_reqmask_0_c_1 | ;
nt start ::= obj_open_reqmask_0_nc_0 "}";
```

After:

```rust
nt start ::= json_string_bounded_split_5+ | ( "," json_string_bounded_split_5 ) * ;
```

That is the exact pattern I had repeatedly failed to push hard enough.

Rule:

**Every named scaffold should be considered guilty until proven necessary.**

---

## Phase 7: Literalize The Alphabet And Remove Syntax Theater

### `d5806cf89`

The witness stopped pretending to be a normal JSON string.

Before:

```rust
t JSON_STRING_CHAR ::= /a/;
t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
nt json_string ::= "\"" JSON_STRING_BODY;
nt json_string_bounded_split_5 ::= "\"" (...);
let mut prefix = b"\"".to_vec();
prefix.extend_from_slice(&vec![b'a'; 2300]);
```

After:

```rust
t JSON_STRING_CHAR ::= "a";
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19});
nt start ::= json_string_bounded_split_5+ | "," json_string_bounded_split_5 ;
let mut prefix = b"".to_vec();
prefix.extend_from_slice(&vec![b'a'; 2300]);
```

The opening quote and the generic string machinery were not essential, so they were removed.

Rule:

**If wrappers are not the mechanism, kill them.**

---

## Phase 8: Inline Chunk Arithmetic And Remove Internal Indirection

### `3f893cc32`

The grammar stopped naming intermediate count fragments and started spelling the count structure directly.

Before:

```rust
t JSON_STRING_CHAR ::= "a";
internal t JSON_STRING_CHAR_UPTO_256_0 ::= JSON_STRING_CHAR{0,256};
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= JSON_STRING_CHAR_UPTO_256_0 "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= JSON_STRING_CHAR{256};
internal t JSON_STRING_CHAR_UPTO_136_3 ::= JSON_STRING_CHAR{0,136};
t JSON_STRING_CHAR_UPTO_CLOSE_4 ::= JSON_STRING_CHAR_UPTO_136_3 "\"";
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19} JSON_STRING_CHAR_UPTO_CLOSE_4);
```

After:

```rust
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,256} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{256};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
```

This is the recursive part in action: once the outer shell is small, you recurse inward and simplify the representation of the remaining shell.

Rule:

**After simplifying the outer structure, recurse into the representation of the inner structure.**

---

## Phase 9: Flatten Harness Noise

### `a8034005c`

The local test harness got flattened.

Before:

```rust
let mut prefix = b"".to_vec();
prefix.extend_from_slice(&vec![b'a'; 2300]);
let tail = b"";

classify_constraint(&constraint, &prefix, [b"aaaaa\""][0], 0, Some(tail));
```

After:

```rust
let prefix = vec![b'a'; 2300];

classify_constraint(&constraint, &prefix, [b"aaaaa\""][0], 0, Some(b""));
```

This is not the deepest step, but it matters because it strips away temporary scaffolding that makes the witness look more complicated than it is.

Rule:

**Once the witness is simple, flatten the harness too.**

---

## Phase 10: Shrink Numbers Coherently

### `80093d5ba`, `fbb8e12f9`, `9a377d4b6`

The next correct attack was on scale.

Before:

```rust
let vocab = make_vocab(&[b"aaaaa\""]);
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,256} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{256};
let prefix = vec![b'a'; 2300];
```

After:

```rust
let v = b"aaa\"";
let vocab = make_vocab(&[v]);
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,128} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{128};
let prefix = vec![b'a'; 1150];
```

The point was not to pick smaller numbers at random.

The point was to preserve the same structural relationship while scaling the witness down.

Rule:

**Shrink constants only after the structure is skeletal, and shrink them coherently.**

---

## Phase 11: Attack Large Numbers Again

### `0e2b3e645`

This was already a major improvement, and the earlier note stopped too early here.

Before:

```rust
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,128} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{128};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,18} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{19});
let prefix = vec![b'a'; 1150];
let v = b"aaa\"";
assert!(!full_mask && full_commit_token && full_commit_bytes && full_complete);
```

After:

```rust
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,32} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{32};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,4} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});
let prefix = vec![b'a'; 158];
let v = b"aaa\"";
assert!(!full_mask && full_commit_token && full_commit_bytes && full_complete);
```

That already proves an important lesson: a weird-looking large number is usually not sacred. It is often just a scaled residue of the same mechanism.

This commit also added search machinery aimed at minimization instead of diagnosis:

- `scan_o82710_inline_glrm_split_token_boundary_small_chunks`
- `scan_o82710_inline_glrm_split_token_boundary_parameter_search`
- `scan_o82710_inline_glrm_split_token_boundary_token_lengths`
- `scan_o82710_inline_glrm_split_token_boundary_under_32_search`

That is how scanning should be used during minimization: as a narrowing instrument.

---

## Phase 12: Build A Fresh Reduction Ladder From The Live Failure

The old note was still too anchored to the earlier `tests/o82710_repro.rs` family.

The later `tests/o82710_current_mre.rs` work on the current live `Github_medium---o82710` discrepancy added a lesson that matters more than any individual cut:

**when the old minimized witness stops being the real bug, restart from the live failure and rebuild the ladder.**

The correct sequence was:

- reproduce the discrepancy from the real benchmark harness
- capture the exact schema, prefix position, token id, and token bytes
- create a staged chain: exact schema plus full vocab, same schema plus reduced vocab, sparse schema, inline GLRM
- validate each rung with the exact oracle before cutting the next rung

This matters because minimization is not preservation of an old artifact. It is preservation of the active mechanism.

Rule:

**If HEAD no longer matches the old minimized repro, stop polishing the fossil and rebuild the chain from the live failure.**

---

## Phase 13: Convert Diagnosis Into Targeted Scans

During the current `o82710` minimization, the useful scans were not broad exploration. They were local deletion attempts expressed as executable variants.

The important ones were:

- `scan_o82710_fixed_order_field_subsets`
- `scan_o82710_recursive_known_field_subsets`
- `scan_o82710_desc_id_reqmask_structure`
- `scan_o82710_description_only_ascii_prefix_lengths`
- `scan_o82710_control_token_variants`
- `scan_o82710_token_id_variants`
- `scan_o82710_all_ascii_prefix_lengths_up_to_current`
- `scan_o82710_disputed_token_variants`
- `scan_o82710_second_split_arm_variants`

These were not diagnostics for their own sake. Each one asked one deletion question:

- can I remove these fields?
- can I weaken this recursive shell?
- can I shorten this prefix?
- can I shrink these token bytes?
- can I densify these ids?
- can I replace this branch with a smaller equivalent?

That is recursive minimization in practice. You do not stare at the witness and speculate. You generate the smallest local family that can falsify the current idea.

Rule:

**When you think some piece might be removable, write the smallest scan that distinguishes the variants and let the oracle choose.**

---

## Phase 14: Delete Provenance In Layers, Not Just Once

The current live witness did not collapse in one jump. It lost provenance in layers.

The successful cuts were:

- benchmark prose in `description` replaced with pure ASCII filler
- control token reduced from `b" Vimeo"` to `b"a"`
- token ids reduced from sparse benchmark ids to dense `0/1`
- disputed token reduced from the benchmark punctuation token to synthetic `b"aa\""`
- `id` value reduced from generic `json_string` to literal `"x"`
- string alphabet reduced to just `"a"`
- the second bounded-string split arm reduced from a bulky unreachable arm to a tiny direct-quote arm

None of those cuts were safe to assume up front. Several things that looked semantically important turned out to be dead provenance.

Rule:

**Delete source-story residue one layer at a time: content, token bytes, token ids, literal values, alphabet, branch form.**

---

## Phase 15: Re-Ask Old Questions After Every Big Simplification

One of the easiest mistakes is to decide too early that some branch or wrapper is essential and never revisit it.

That is wrong.

After the current witness switched to synthetic token bytes and a one-byte alphabet, old conclusions changed. Some earlier simplifications still failed, but others became newly possible, including the shorter second split arm.

This is the recursive part people skip.

You do not merely simplify the remaining structure.

You also rerun failed deletion ideas against the new, smaller witness, because the dependency graph changes as the witness shrinks.

Rule:

**After every material simplification, revisit previously failed deletions. Smaller witnesses often free cuts that were impossible one step earlier.**

---

## Phase 16: Treat Prefix Length As A First-Class Surface

For the current synthetic witness, the description payload became just repeated `a` bytes, but the count still mattered. The useful move was not to keep the inherited benchmark length. The useful move was to scan lengths directly until the smallest reproducing residue remained.

That is how the current family landed on `description_only_prefix_with_ascii_repeat(2303)` for the shorter disputed token.

The point is not that `2303` is magical.

The point is that once the content becomes uniform, the only remaining information may be the count residue itself.

Rule:

**When the payload is uniform, search the count boundary directly. Prefix length is part of the witness, not just harness setup.**

---

## Phase 17: Preserve Only The Mechanism Summary

The current minimized inline candidate is still not minimal enough, but it already tells the right story.

The surviving mechanism is roughly this:

- a required-field object shell whose recursive follow-set still matters
- a bounded-string split with one active arm and one currently unreachable but still semantically relevant companion arm
- a uniform `a` payload long enough to hit the residue
- a two-token vocab where one byte token is ordinary control flow and the other synthetic token straddles the close boundary
- an oracle that only cares about `!mask && commit_token && commit_bytes`

That is the right summary level.

It is not "a Vimeo schema bug" anymore. It is a tiny commit-vs-mask witness with just enough shell left to express the failure.

Rule:

**At every stage, rewrite the one-sentence mechanism summary. If a surviving piece does not appear in that summary, attack it next.**

---

## The Practical Loop

For this kind of work, the loop should be:

1. Choose one live witness.
2. State one falsifiable deletion or weakening hypothesis.
3. Make the smallest local edit or scan that tests it.
4. Run the narrowest exact oracle.
5. If it survives, commit and recurse from the smaller witness.
6. If it fails, keep the smaller understanding and attack the next surface.

What went right in the later `o82710` work was not taste. It was discipline:

- exact oracle first
- one local hypothesis at a time
- aggressive provenance deletion
- repeated rescans after each simplification
- regular commits so every successful reduction becomes a stable rung

That is what minimizing properly, recursively, actually looks like.

---

## Phase 18: Delete The Fake Surface Syntax Even When It Still Feels Structural

The next big lesson from the current `o82710` work is that a witness can look like it still needs a shell when it really only needs a language shape.

The current live witness lost, in order:

- long field names
- `id` as a semantic concept
- spaces after separators
- braces
- quoted keys
- colons
- commas
- the side-branch marker byte
- the explicit three-state recursive shell

The important point is not that JSON syntax was removable.

The important point is that every removal happened by testing one local language difference at a time.

The shell went through forms like:

- object-key branches
- one-character keys with tight separators
- bare keys and then bare key markers
- adjacency-only branch sequencing
- an explicit recursive three-state shell
- a flat repetition form

That is recursive minimization done correctly.

You do not just strip punctuation because it looks decorative.

You strip one class of punctuation, validate, then ask what abstract language shape remains.

Rule:

**When a witness still looks like a source-language artifact, attack the shell as a language, not as typography.**

---

## Phase 19: Replace Stateful Shells With Equivalent Flat Languages

One particularly important later cut was replacing the explicit recursive `obj_open_reqmask_*` state chain with a direct flat language.

Before, the witness still used named shell states:

```rust
nt obj_open_reqmask_0_nc_0 ::= ("a" json_string_bounded_split_6) obj_open_reqmask_0_c_0 | "\"\"" obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_0 ::= ("a" json_string_bounded_split_6) obj_open_reqmask_0_c_0 | "\"\"" obj_open_reqmask_0_c_1;
nt obj_open_reqmask_0_c_1 ::= ("a" json_string_bounded_split_6) obj_open_reqmask_0_c_1 | ;
nt start ::= obj_open_reqmask_0_nc_0;
```

The focused scan showed the same behavior could be expressed more directly as a flat language:

```rust
nt start ::= ("a" json_string_bounded_split_6)+ "\"\"" ("a" json_string_bounded_split_6)*;
```

Then even the leading marker `"a"` turned out to be dead, leaving the even flatter shape:

```rust
nt start ::= json_string_bounded_split_6+ "\"\"" json_string_bounded_split_6*;
```

This is the right standard.

If multiple named states only encode a regular-language split point, flatten them.

Rule:

**Whenever a shell state machine remains, ask whether it can be replaced by one direct regular-language expression.**

---

## Phase 20: Separate Semantic Cuts From Representational Cuts

Later in the process there were two different kinds of reductions, and they should not be confused.

Semantic cuts changed the witness language:

- removing the leading marker before each bounded string
- removing the commas and branch markers
- shrinking the chunk size from `256` to `32`
- shrinking the default reproducing prefix from `2303` to `287`

Representational cuts kept the language the same but made the grammar easier to read and smaller to stare at:

- inlining helper nonterminals and aliases
- replacing long inherited names with short names like `q`, `qq`, `u`, `e`, `p`
- factoring the repeated quoted fragment into a named `piece`

Both matter.

The semantic cuts shrink the actual witness. The representational cuts make the next semantic cuts easier to see.

Rule:

**Alternate between shrinking the language and shrinking the notation. Cleaner notation exposes the next real deletion.**

---

## Phase 21: Coherent Scale Reduction Is Usually Worth Another Pass

Once the current witness became a flat, abstract split language, the old `256` scale looked suspicious again.

The correct move was not to guess a new number. The correct move was to preserve the same residue shape while scaling everything together:

- chunk size `256 -> 128 -> 64 -> 32`
- default reproducing prefix `2303 -> 1151 -> 575 -> 287`

The key point is that all of those still reproduced.

That means the original scale was just residue, not mechanism.

This same pass also showed the opposite on the split counts: `18/19` survived, but `8/9`, `4/5`, `2/3`, and `1/2` all killed the bug. That is equally important.

Minimization is not just finding what can be deleted. It is also finding what cannot yet be deleted.

Rule:

**After every major structural simplification, rerun coherent scale reduction. Large inherited constants are guilty until proven necessary.**

---

## Phase 22: Harden Diagnostic Scans So They Do Not Abort Early

Another practical lesson from the later work is that scans need to survive invalid variants.

Several scans initially failed because some variants made the prefix itself invalid, and `classify_constraint` still unwrapped those commits. That hides information.

The correct repair was local:

- pre-check whether the candidate prefix commits at all
- print `prefix_rejected=...` when it does not
- continue scanning the rest of the family

That matters because a failed variant is still useful evidence. It should be classified, not crash the scan.

Rule:

**Diagnostic scans should classify dead variants, not abort on them. Invalid prefixes are evidence too.**

---

## Phase 23: Keep Rewriting The Mechanism Summary As The Witness Shrinks

The mechanism summary for the current witness changed dramatically over time.

Earlier it was something like:

- recursive object follow-set
- required known-key shell
- bounded string arm split
- disputed token crossing the close boundary

Later it became much simpler:

- a repeated quoted piece language
- a required middle `""` separator between two piece families
- a bounded split whose second arm still matters
- a disputed token `aa"` that crosses the close boundary

That rewrite matters because it tells you where to cut next.

Once the summary no longer mentions JSON, keys, ids, commas, braces, or object semantics, those should already be gone.

Once the summary no longer mentions large chunk sizes, those should be attacked next.

Rule:

**Keep rewriting the mechanism summary in one sentence. Whatever is missing from that sentence is your next deletion target.**

---

## Phase 24: Revalidate Old Structural Conclusions Against The New Piece Language

The later witness got small enough that some earlier shell conclusions stopped being trustworthy.

At one stage the witness was roughly:

```rust
t q ::= "\"";
t qq ::= q q;
t u ::= "a"{0,32} q;
t e ::= "a"{32};
nt p ::= q (e{0,18} u | e{19} q);
nt start ::= p+ qq p*;
```

That still carried assumptions from an older shell.

The right move was not to preserve them. The right move was to rescan them locally.

That is how the witness got reduced to:

```rust
t q ::= "\"";
t e ::= "a"{32};
nt p ::= (e{0,18} "a"{0,32} | e{19}) q;
nt start ::= p* q p*;
```

The important lesson is not just the final form.

The important lesson is that the smaller `p` language changed what the top-level shell needed, so the shell had to be tested again rather than inherited.

Rule:

**Whenever the inner piece language changes materially, rerun top-level shell scans. Old shell results expire.**

---

## Phase 25: Remove Dead Prefixing Syntax Inside The Remaining Piece

A particularly clean late cut was deleting the outer quote from the piece itself.

Before:

```rust
nt p ::= q (e{0,18} u | e{19} q);
```

After:

```rust
nt p ::= e{0,18} u | e{19} q;
```

That cut matters because the witness had already become abstract enough that the piece no longer needed to pretend to open a fresh quoted fragment itself. The required separator quote in the surrounding shell was enough.

This is a good example of a late recursive cut: you do not stop after flattening the shell. You go back into the last surviving piece and ask whether it still contains inherited syntax theater.

Rule:

**After the shell becomes abstract, attack the syntax inside the remaining piece again. Do not assume earlier inner wrappers are still real.**

---

## Phase 26: Factor Shared Structure Even When It Does Not Change The Language

The final late cleanup was representational, but still important.

Before:

```rust
t u ::= "a"{0,32} q;
nt p ::= e{0,18} u | e{19} q;
```

After:

```rust
nt p ::= (e{0,18} "a"{0,32} | e{19}) q;
```

This did not change the witness language.

It removed one helper and made the shared trailing close marker explicit.

That matters because the next real question is easier to see when the common suffix is written once instead of twice.

Rule:

**If two branches end the same way, factor the shared suffix and delete the helper. Better spelling is often the gateway to the next real cut.**

---

## Phase 27: Keep Diagnostic Helpers Aligned With The Current Witness

Another later lesson was procedural rather than grammatical.

Several scans were briefly answering the wrong question because their helper grammars still encoded older shapes like `p+ q p*` or the older quoted piece form.

That had to be fixed before trusting their output.

This matters a lot during recursive minimization. Once the witness is small, a stale helper is effectively a different witness.

The safe pattern is:

- simplify the witness
- update any scan helpers that encode that witness
- only then trust the next scan result

Rule:

**A minimization scan is only as good as the witness it still matches. After each real cut, repair stale helper grammars before drawing conclusions from them.**

---

## Phase 28: The Current Mechanism Summary Got Smaller Again

The latest mechanism summary is now smaller than the one in the previous section.

It is no longer:

- repeated quoted pieces with a double-quote separator
- explicit helper nonterminals for the bounded-close arm

It is now closer to:

- two families of bounded `a`-runs that both end in a quote
- a required single middle quote between the two piece regions
- a prefix-length residue of `287`
- a disputed token `aa"` whose bytes still commit while the token mask says no

That rewrite is exactly why the witness improved.

The double-quote separator stopped being part of the mechanism summary, so it shrank to one quote.

The extra helper stopped being part of the mechanism summary, so it disappeared.

Rule:

**Keep shrinking the one-sentence mechanism summary. Every noun that disappears from the summary should be attacked in the witness immediately.**

---

## Phase 12: Second Minimization Wave On The Nicer Standalone Witness

This is the part the old note did not cover well enough.

The important point is that minimization did not stop after the `32 / 5 / 158 / aaa"` witness. The smaller witness itself got attacked again.

That is the core lesson.

### `3ed3c064d` Simplify split-boundary start rule

Before:

```rust
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,32} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{32};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,4} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});
nt start ::= json_string_bounded_split_5+ | "," ? json_string_bounded_split_5 ;
let prefix = vec![b'a'; 158];
let v = b"aaa\"";
```

After:

```rust
t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,32} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{32};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,4} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});
nt start ::= json_string_bounded_split_5 JSON_STRING_CHAR_UPTO_CLOSE_1 ;
let prefix = vec![b'a'; 158];
let v = b"aaa\"";
```

This is a very clean example of proper recursion.

The previous witness had already been minimized a lot, but the `+ | "," ? ...` start rule still looked suspiciously general. So it got attacked directly.

That commit also added the right scanners to test alternative shapes rather than just stare at the current one:

- `scan_o82710_inline_glrm_split_token_boundary_constant_total_budget`
- `scan_o82710_inline_glrm_split_token_boundary_direct_two_piece_sequences`
- `scan_o82710_inline_glrm_split_token_boundary_start_rule_shapes`

Those scanners encode the actual minimization question:

- do we need the whole start-language family?
- can the witness be reduced to a two-piece sequence?
- is the total exact-run budget the real thing to preserve?

Rule:

**When a minimized witness still has a general top-level language, replace it with the smallest concrete top-level sequence that still works.**

### `7e8577b9a` Further shrink split-boundary MRE

Before:

```rust
let v = b"aaa\"";

t JSON_STRING_CHAR_UPTO_CLOSE_1 ::= "a"{0,32} "\"";
t JSON_STRING_CHAR_EXACT_256_2 ::= "a"{32};
nt json_string_bounded_split_5 ::= (JSON_STRING_CHAR_EXACT_256_2{0,4} JSON_STRING_CHAR_UPTO_CLOSE_1 | JSON_STRING_CHAR_EXACT_256_2{5});
nt start ::= json_string_bounded_split_5 JSON_STRING_CHAR_UPTO_CLOSE_1 ;

let prefix = vec![b'a'; 158];
```

After:

```rust
let v = b"aa\"";

t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{0,4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;

let prefix = vec![b'a'; 159];
```

This is a really important step.

Several things happened at once:

- the token got shorter: `aaa"` to `aa"`
- the nonterminal disappeared entirely
- the count structure moved directly into `start`
- names got shorter and less narrative
- the prefix changed from `158` to `159`, which is exactly what you want to see when reducing token length against a fixed total exact-run budget of `160`

That `159` is not random. It says the witness is sitting one byte before the close consumes the token boundary.

This commit also added the next correct scanners:

- `scan_o82710_inline_glrm_split_token_boundary_left_chunk_shapes`
- `scan_o82710_inline_glrm_split_token_boundary_chunk_then_close_scale`
- `scan_o82710_inline_glrm_split_token_boundary_inline_start`

Those are not generic explorations. They are precise minimization questions:

- do we need the left-chunk nonterminal shape?
- what is the smallest scale where `chunk then close` still reproduces?
- can the left chunk be inlined directly into `start`?

Rule:

**Once the outer shell is smaller, shorten the token too. The token length is part of the witness and should be attacked like any other constant.**

### `40f2b4bc9` Minimize split-boundary branch structure

Before:

```rust
nt start ::= (A_EXACT{0,4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
```

After:

```rust
nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
```

This is one of the clearest single-line minimization wins in the whole sequence.

The range `0,4` looked plausible, but it turned out only one branch out of that range mattered. So the whole range got collapsed to exactly the surviving branch.

This commit also added scanners that prove that the deleted branches were truly nonessential:

- `scan_o82710_inline_glrm_split_token_boundary_inline_start_exact_range`
- `scan_o82710_inline_glrm_split_token_boundary_inline_start_repeat_caps`
- `scan_o82710_inline_glrm_split_token_boundary_inline_start_sparse_branches`
- `scan_o82710_inline_glrm_split_token_boundary_factored_prefix`

Those are excellent minimization tools because they test exact hypotheses:

- can exact size go below `32`?
- do we need all those repeat-cap branches?
- which branch IDs actually matter?
- can the common prefix be factored out without changing the effect?

Rule:

**If a branch set looks like a convenience range, assume most of it is dead until a scan proves otherwise.**

### `8f4e750c8` Inline split-boundary witness logic

Before:

```rust
let v = b"aa\"";
let vocab = make_vocab(&[v]);
let prefix = vec![b'a'; 159];

let (full_mask, full_commit_token, full_commit_bytes, full_complete) =
	classify_constraint(&constraint, &prefix, v, 0, Some(b""));

println!(
	"split_full_token mask={} commit_token={} commit_bytes={} complete_after_token={}",
	full_mask,
	full_commit_token,
	full_commit_bytes,
	full_complete,
);

assert!(!full_mask && full_commit_token && full_commit_bytes && full_complete);
```

After:

```rust
let token = b"aa\"";
let vocab = Vocab::new(vec![(0, token.to_vec())], None);
let prefix = [b'a'; 159];

let mut mask_state = constraint.start();
mask_state.commit_bytes(&prefix).unwrap();
let full_mask = mask_state.mask().first().map(|word| (word & 1) != 0).unwrap_or(false);

let mut commit_token_state = constraint.start();
commit_token_state.commit_bytes(&prefix).unwrap();
let full_commit_token = match catch_unwind(AssertUnwindSafe(|| commit_token_state.commit_token(0))) {
	Ok(Ok(())) => true,
	Ok(Err(_)) => false,
	Err(_) => true,
};

let mut commit_bytes_state = constraint.start();
commit_bytes_state.commit_bytes(&prefix).unwrap();
let full_commit_bytes = commit_bytes_state.commit_bytes(token).is_ok();

assert!(!full_mask && full_commit_token && full_commit_bytes);
```

This is important because the witness stopped going through a higher-level helper that bundled extra behavior.

That is not just test style. It is minimization.

Now the witness is explicitly about three independent operations:

- what the mask says
- what `commit_token` does
- what `commit_bytes` does

Rule:

**If your helper is computing more than the minimal witness needs, inline the operations and keep only the pieces that define the mismatch.**

### `2f5042d15` Record failed simplification paths for split-boundary MRE

This commit matters even though it did not shrink the main witness directly.

It added concrete dead-end probes such as:

- `scan_o82710_inline_glrm_split_token_boundary_explicit_long_terminals`
- `scan_o82710_inline_glrm_split_token_boundary_explicit_chunk_sequence`
- `scan_o82710_inline_glrm_split_token_boundary_single_branch_token_lengths`
- `scan_o82710_inline_glrm_split_token_boundary_single_branch_close_caps`

That is good minimization practice.

A failed simplification path is still valuable when it is recorded precisely, because it prevents re-running the same bad intuition later.

In particular, these probes test wrong-but-tempting ideas:

- replacing counted repeats with explicit long terminals
- spelling out the chunk sequence by hand
- shrinking the token further in the single-branch family
- shrinking the close cap while holding the rest fixed

Rule:

**Record dead ends when they answer a real minimization question. Do not just abandon them mentally.**

### `05bd6872b` Inline close expression in split-boundary MRE

Before:

```rust
t A_UPTO_CLOSE ::= "a"{0,32} "\"";
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} A_UPTO_CLOSE | A_EXACT{5}) A_UPTO_CLOSE;
```

After:

```rust
t A_EXACT ::= "a"{32};
nt start ::= (A_EXACT{4} ("a"{0,32} "\"") | A_EXACT{5}) ("a"{0,32} "\"");
```

This is exactly the kind of recursive cut I had been failing to do consistently.

Even after the nonterminal had been removed, the named terminal `A_UPTO_CLOSE` was still a helper-shaped thing. So it got attacked too.

This commit also added the next family of excellent probes:

- `scan_o82710_inline_glrm_split_token_boundary_inline_close_expression`
- `scan_o82710_inline_glrm_split_token_boundary_fully_inline_start`

Those ask the right questions:

- do we need the named close terminal at all?
- can the whole witness live directly in `start`?

Rule:

**After deleting named nonterminals, keep going. Named terminals can also be unnecessary abstraction.**

### `ed3b61215` Tighten standalone split-boundary witness

This is a small but disciplined cleanup.

Before:

```rust
let full_mask = ...;
let full_commit_token = ...;
let full_commit_bytes = ...;

println!(
	"split_full_token mask={} commit_token={} commit_bytes={}",
	full_mask,
	full_commit_token,
	full_commit_bytes,
);

assert!(!full_mask && full_commit_token && full_commit_bytes);
```

After:

```rust
let mask = ...;
let commit_token = ...;
let commit_bytes = ...;

assert!(!mask && commit_token && commit_bytes);
```

This matters because once the witness is local and stable, even variable names and debug output should be minimized.

Rule:

**If a local name still carries historical baggage, shorten it once the witness no longer needs the distinction.**

### `d450b6731` Probe local minimum for split-boundary witness

This is another key commit missing from the earlier note.

By this point the witness was already small, but instead of declaring victory, the sequence did the right thing and tested for a local minimum.

It added:

- `scan_o82710_inline_glrm_split_token_boundary_inline_close_exact_range`
- `scan_o82710_inline_glrm_split_token_boundary_inline_close_token_lengths`
- `scan_o82710_inline_glrm_split_token_boundary_simpler_shapes`

Those are exactly the questions you ask near a local minimum:

- can the exact count go below `32` once the close is fully inline?
- can the token go below `aa"` in this final family?
- do apparently simpler equivalent shapes still preserve the mismatch?

This is the right endgame behavior.

You do not stop because the witness looks small.

You stop when the next-layer probes start telling you the remaining pieces are genuinely load-bearing.

Rule:

**Near the end, switch from broad shrinking to local-minimum probes.**

---

## What The Later Scanners Prove

The later scanner family around `scan_o82710_inline_glrm_split_token_boundary` is important because it shows what good recursive minimization looks like operationally.

It is not just “keep tweaking stuff.”

It systematically attacks the witness along separate axes:

- top-level language shape
- left-branch shape
- exact-count size
- repeat-cap size
- token length
- close-cap size
- named helper survival
- factored vs unfactored prefix structure
- explicit vs counted representation
- local-minimum variants that only look simpler

That is what I need to emulate.

If the witness still has multiple dimensions, minimization should become a small battery of focused scans over those dimensions.

---

## The Practical Procedure

If I want to minimize properly next time, this is the procedure.

### 1. Freeze the oracle first

Write the exact condition being preserved.

Example:

```rust
!mask && commit_token && commit_bytes
```

or, if completion still matters in that stage:

```rust
!mask && commit_token && commit_bytes && complete
```

Everything else is now disposable.

### 2. Pick one witness

Choose one:

- grammar
- token
- prefix
- test path

Delete sibling witnesses unless the contrast between them is part of the phenomenon.

### 3. Inline before shrinking

Inline:

- grammar helpers
- builder helpers
- classification helpers that bundle extra checks
- named wrappers that are only convenience

The active witness should become local, literal, and ugly.

### 4. Attack outer story before inner constants

Attack in this order:

1. semantics
2. wrappers
3. named recursive scaffolding
4. named terminal scaffolding
5. branch families
6. token family
7. counts and caps

After each successful shrink, return to the top.

That return is the recursive part.

### 5. Prefer deletion over clever rewrites

Preferred order:

1. delete
2. inline
3. literalize
4. weaken
5. merge
6. scale down

Deletion gives the strongest evidence that a piece was unnecessary.

### 6. Turn minimization questions into scanners

Good scanner questions look like this:

- can exact size go below `32`?
- which branch IDs actually matter?
- can token length go from `aaa"` to `aa"` or lower?
- can a named helper be inlined completely?
- can a counted-repeat family be replaced with one direct sequence?

Bad scanner questions are vague or story-preserving.

### 7. Record dead ends

If a tempting simplification fails, record it as a named probe if it answers a real question.

That prevents repeatedly rediscovering the same non-improvement.

### 8. Probe for a local minimum explicitly

When the witness is already small, stop doing broad moves and start doing exact local probes:

- smaller exact count
- smaller token length
- fewer branches
- fewer helpers
- equivalent-looking simpler shapes

Stop only when those probes stop succeeding.

---

## Main Lessons

### Lesson 1

The goal is not to preserve the original story. The goal is to preserve the oracle.

### Lesson 2

Every smaller witness is a new target. If I stop after one good shrink, I have not minimized recursively.

### Lesson 3

If a witness still looks realistic, it is probably still too big.

### Lesson 4

If a branch range looks convenient, most of it is probably dead.

### Lesson 5

If a number looks weirdly large, it is probably inherited scale rather than a true floor.

### Lesson 6

Good minimization often makes the test uglier and more explicit. That is not a problem. During minimization, ugly often means honest.

### Lesson 7

A good scan file is not random experimentation. It is a map of targeted attacks on separate dimensions of the witness.

---

## Final Standard

The repro is not finished when it is “pretty small.”

It is finished when every remaining piece has already survived a direct attempt to remove it.

That is what “minimize properly, recursively” means.