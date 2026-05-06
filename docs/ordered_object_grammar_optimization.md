# Ordered Object Grammar Optimization — Notes

## Problem Statement

Optimize `glrmask_native` commit latency for complex JSON schemas, specifically
`jsb/data/Github_hard---o67291` (the "monster" FlatBuffers schema). Target: max
commit time < 5µs per token step.

Run via:
```
make example-specific PROBLEM=o67291 FRAMEWORKS=glrmask_native
```

---

## The Schema (`def_monster_2`)

The problem schema contains an object type `def_monster_2` with **49 ordered,
all-optional keys** (pos, mana, hp, name, friendly, color, inventory, test_type,
test, test4, testarrayofstring, testarrayoftables, enemy, testnestedflatbuffer,
testempty, testbool, ...).

Key properties:
- Ordered (keys must appear in schema-declaration order)
- All optional (any subset is valid)
- Closed (no additionalProperties)
- Large (49 keys — enough to stress any grammar approach)

---

## Starting Point

**Grammar used**: Right-recursive binary tree (default `GLRMASK_ORDERED_OBJECT_SHAPE=right`)

Stack depth at `}` closing the monster object: **~91 states deep**
Max commit time: **~38µs** (step 657, token `"}"`)

Profile of the worst step:
```
advance[1]  terminal=7  ts=0  deterministic  38µs
  GSS: depth=91
  ops: det(actions=91, gotos=90, popn=0)
```

90 gotos at ~400ns each (L3 cache misses on a large grammar table) = ~36µs.

---

## Understanding the Grammar Approaches

### Approach 1: Right-recursive binary tree (original)

Structure (for 4 keys k0..k3, all optional):
```
tree → k0_opt k1_opt k2_opt k3_opt
k0_opt → "k0": v0
        | ε
k1_opt → ...
```
Actually it builds a balanced or right-skewed tree:
```
obj[0..3] → k0 obj[1..3]  |  obj[1..3]
obj[1..3] → k1 obj[2..3]  |  obj[2..3]
...
```
- Stack depth at `}`: O(N)
- Nondeterminism: O(log N) GLR forks at each `,` (binary tree splits)
- Goto cost: very high (large LR tables, many cache misses)

### Approach 2: After[j]-per-state (separator-factored dispatch)

Structure:
```
body  → KV_k after[k+1]   for each k reachable from state 0
after[j] → ", " KV_k after[k+1]  for each k reachable from state j
           | ε   (if can_close_at(j))
```

Key property: factoring the separator `,` INTO the `after[j]` rule means the
NEXT token (the key literal `"key_name": `) is the disambiguation point. No
GLR forking at `,`.

Results:
- Deterministic at all `,` steps ✓  
- Stack depth at `}`: O(N) — 48 nested `after[j]` frames ✗
- Step 212 commit: 3µs ✓
- Step 657 commit: 13µs ✗ (48 gotos, 150ns each = L2 cache miss level)

Why 150ns per goto? With N=49 `after[j]` nonterminals, the LR goto table has
many distinct rows. Traversing 47 different rows during the `}` reduction causes
~47 L2/L3 cache misses.

### Approach 3: Single shared `tail` rule (O(1) depth)

Structure:
```
prefix[j] = left-recursive ordered prefix ending at key j-1
body → prefix[j] tail      for each closeable j
tail → ", " free_expr  |  ε
```

Key property: `prefix[j]` reduces eagerly (left-recursive), so at `}` only 2
reductions are needed regardless of N.

Results:
- O(1) stack depth at `}` ✓
- Step 657 commit: 2µs ✓
- Nondeterminism at `, "`: GLR forks (tail has both known-key and free alts) ✗
- Step 212 commit: 15µs ✗ (3-path fork from ambiguous `, "`)

### Approach 4: Separator-merged left-recursive (FINAL SOLUTION)

The insight that breaks the deadlock:

**Merge the separator `, ` directly into the continuation key literal.**

Instead of `prefix[j] → prefix[i] COMMA KV_k`, use:
```
prefix[k] → ", \"key_k\": " value_k    merged into a single token
           | prefix[j] + above          left-recursive extension
```

Because `", \"key_k\": "` is unique per key (each key has a different name),
after seeing `prefix[j]`, the parser's next token uniquely identifies which
`prefix[k]` to build. No ambiguity, no fork.

And because `prefix[k]` is left-recursive, it reduces to a single GSS node
after each key. At `}` time there are at most 2 reductions regardless of N.

**This achieves both properties simultaneously.**

---

## Implementation Details

### File modified: `src/import/json_schema.rs`

Function: `try_build_factored_ordered_object`

#### Grammar emitted (simplified)

For ordered keys [k0, k1, ..., k_{N-1}], all optional:

```
// Base rules (no separator needed for the first key in a sequence)
prefix_p1 → build_merged_literal_key_value_expr(b"", "k0", v0)
prefix_p2 → build_merged_literal_key_value_expr(b"", "k1", v1)
           | prefix_p1  build_merged_literal_key_value_expr(b", ", "k1", v1)
prefix_p3 → build_merged_literal_key_value_expr(b"", "k2", v2)
           | prefix_p1  build_merged_literal_key_value_expr(b", ", "k2", v2)
           | prefix_p2  build_merged_literal_key_value_expr(b", ", "k2", v2)
...

// Body rule (all valid terminal states)
body → prefix_p1 | prefix_p2 | ... | prefix_pN
     | prefix_pJ ", " free_expr   (for open objects)
     | ε                           (if all optional)

// Object
Object → "{" body "}"
```

The `build_merged_literal_key_value_expr(b", ", key, value)` call merges `, `
into the key literal: produces a single expression beginning with `, "key": `.

#### Gap handling

For the prefix rules, we allow "gaps" — skipping optional keys. A gap from
state i to key j-1 is valid if all keys ordered[i..j-1] are optional.

Example: if k1 is optional, then prefix_p3 can skip directly from prefix_p1
(after k0) to k2, without going through prefix_p2.

#### Open objects

For open objects (additionalProperties or patternProperties), the body also
has alternatives for free properties:
```
body → prefix[j]                  (close after known keys)
     | prefix[j] ", " free_expr   (continue with free properties)
     | free_expr                   (only free properties)
     | ε
```

The `, ` separator before `free_expr` is NOT merged (it's a separate literal),
so at `, ` after a prefix, there could be a fork between:
- Continue with another known key (via merged `, "key": ` token)
- Go to free property

In practice this fork is typically 2 paths and resolves immediately on the next
character.

---

## Key Insights

### 1. Left-recursion collapses O(N) history into O(1) stack nodes

A left-recursive rule `A → A t₁ | t₀` means: after parsing t₀ t₁ t₁ ... t₁
(N times), the parser reduces to a SINGLE `A` node after each step. The GSS
accumulates only 1 node per level of left-recursion, not N.

Compare to right-recursion `A → t₁ A | ε`: every reduction of `A → ε` triggers
a cascade of N parent reductions. Stack depth at the base case = O(N).

**For JSON ordered objects**: left-recursive `prefix[j]` is the right structure
because we're building a prefix cumulatively, and the ONLY time we reduce the
full structure is at `}` — which should be O(1) work.

### 2. Merged literals eliminate the LR(1) ambiguity at `, `

Classic problem: after `prefix[j]`, seeing `, ` creates a shift-reduce conflict:
- Shift `, ` as the start of the next key extension `prefix[j] ", " KV_k`
- Reduce `body → prefix[j]` (ending the known-key sequence)

In LR(1), this is a genuine conflict because you need 2 tokens of lookahead to
resolve: the `, ` AND the key name.

By merging `, ` into the key literal (making `, "key": ` one token), the SINGLE
lookahead token already includes the key identity. The parser sees `, "hp": `
and immediately knows it's extending with key "hp". No conflict.

### 3. Goto table size → cache pressure → latency

Each distinct nonterminal adds columns to the LR goto table. With N=49
`after[j]` rules (approach 2), the goto table has 49 extra columns. After 47
sequential reductions at `}`, each hitting a different column = 47 cache misses
at L2 latency (~150ns each) = 7µs just for gotos.

The separator-merged approach has only ONE set of `prefix[j]` rules total, and
they reduce in ONE step at `}`. Goto table is tiny, all L1 cache = ~27ns per op.

### 4. `factored_ordered_object_enabled()` shape dependency removed

Originally required `GLRMASK_ORDERED_OBJECT_SHAPE=right`. This was an artifact
of the previous implementation gating. The new left-recursive approach doesn't
depend on the tree shape — it's always better. Removed the shape requirement.

### 5. The 6µs floor on example 0

Step 94 (`}],` token) costs 6µs and is NOT related to the object grammar:
- Token closes `{a: 1, b: 2}` (nested small object) + `]` (array end) + `,`
- 2 advances, 11 LR ops total, 27ns per op = 4.7µs core + measurement overhead
- The small nested object uses a different grammar path (exact closed object or
  simple required-key path, not the 49-key optional path)
- This is the L1 cache floor for 11 sequential LR operations

To go below 6µs on example 0 would require either reducing the nesting depth of
`test4: [{a,b},{a,b}]` in the schema representation, or reducing the overhead
of the tokenizer dispatch (2 tokenizer states → split commit).

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GLRMASK_ORDERED_OBJECT_SHAPE` | balanced | Shape for fallback tree (right/balanced/left/left-balanced) |

---

## Results

### `make example-specific PROBLEM=o67291 FRAMEWORKS=glrmask_native`

| Example | Before | After | Notes |
|---------|--------|-------|-------|
| 0 | 38µs | 6µs | Step 94 (`}],`) is the floor — nested array close |
| 1 | ~16µs | 4µs | |
| 2 | ~16µs | 4µs | |
| 3 | ~16µs | 4µs | |
| 4 | ~16µs | 3µs | |
| 5 | ~16µs | 4µs | |

### Correctness

- 0 discrepancies vs `llguidance_native` on full JSB suite
- 5/5 problems ok
- Pre-existing failure: `jsb/data/Github_trivial---o66073` (oneOf constraints,
  unrelated)

---

## Commits

```
66c1306ed  feat: separator-merged left-recursive grammar for ordered objects
e79a61046  feat: enable separator-merged LR grammar by default for ordered objects
```

---

## What to Try Next (if more headroom needed)

1. **Step 94 optimization**: Profile whether the 2 tokenizer states at step 94
   cause the split commit overhead. If `ts=0` and `ts=6` can be unified earlier,
   it might save ~1µs.

2. **Grammar for `test4` array elements**: `{a: 1, b: 2}` uses a small 2-key
   object. Verify it's going through the fast exact-closed-object path
   (`try_build_exact_closed_object` for ≤16 required keys).

3. **Measurement precision**: The 6µs reported is a max over the example. The
   p50 commit across all tokens is ~1µs. Consider whether max or p99 is the
   right metric.

4. **Build time**: Grammar build for `def_monster_2` takes ~430ms (one-time
   cost). This is acceptable for inference but worth watching if schemas get
   larger.

5. **Broader benchmarks**: Run `make bench` (or equivalent) on a wider set of
   schemas to verify the left-recursive grammar doesn't regress simpler cases.
