# Resolve-Negatives Graph Trait Plan

## Goal

Refactor `resolve_negatives.rs` so the core algorithms operate over a graph/view trait instead of concrete `NWA`, with **zero behavior change** for the current path.

This is the prerequisite for an exact lazy parser graph path.

## 1. What `resolve_negatives.rs` currently needs from `NWA`

The module uses a surprisingly small concrete surface.

### Read-only graph access

Used by cancellation, finality, and terminal-default analysis:

- state count
  - `nwa.states().len()`
- indexed state access by dense `u32` id
  - `nwa.states()[state as usize]`
- per-state final weight
  - `state.final_weight`
- per-state epsilon edges
  - `state.epsilons`
- per-state labeled transitions
  - `state.transitions`
- negative-label range scan
  - `state.transitions.range(..0)`
- single-label lookup
  - `state.transitions.get(&label)`
  - `state.transitions.get(&DEFAULT_LABEL)`
- whole-map iteration
  - `for (&label, targets) in &state.transitions`

### Concrete mutation

Used only in the public `NWA` entrypoints, not in the pure graph logic itself:

- add derived epsilon
  - `nwa.add_epsilon(from, to, weight)`
- overwrite final weight
  - `nwa.states_mut()[id].final_weight = ...`
- remove negative transitions
  - `state.transitions.retain(|label, _| !is_negative_label(*label))`
- prune default targets and empty transition buckets
  - `state.transitions.get_mut(&DEFAULT_LABEL)`
  - `targets.retain(...)`
  - `state.transitions.retain(|_, targets| !targets.is_empty())`

### What it does **not** need

- `start_states`
- `append_with_body`
- `add_transition`
- any NWA minimization or determinization API

## 2. Recommended trait split

The first extraction should be **read-only only**.

Do not start by forcing the entire mutating `resolve_negative_codes_in_nwa(...)` pipeline behind a mutable trait. That creates the wrong abstraction for the lazy graph and increases refactor risk.

### 2.1 Read-only trait

Use a dense-id view trait for the first step.

```rust
pub trait ResolveNegativesView {
    fn state_count(&self) -> usize;
    fn final_weight(&self, state: u32) -> Option<&Weight>;
    fn epsilons(&self, state: u32) -> &[(u32, Weight)];
    fn transitions(&self, state: u32) -> &BTreeMap<i32, Vec<(u32, Weight)>>;
}
```

Why keep the ids dense `u32` in the first version:

- current kernels are heavily `Vec`-indexed
- it keeps the first commit mechanical
- future lazy graph can still intern symbolic nodes to dense `u32` ids

Do **not** introduce generic associated `StateId` / `Label` / `Weight` types in the first extraction.

That is attractive in the abstract, but it adds type churn without helping the immediate lazy-parser use case. The actual future lazy graph can use the same concrete:

- `state id = u32`
- `label = i32`
- `weight = Weight`

### 2.2 No mutable trait in the first commit

For commit 1, mutation stays concrete in the `NWA` wrapper.

That means the public entrypoint remains:

```rust
pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA)
```

but its internals call generic kernels over `impl ResolveNegativesView` and then apply the results back to `NWA` exactly as today.

This is the smallest behavior-preserving extraction.

## 3. Kernel split after trait introduction

The current file should be split conceptually into:

### 3.1 Pure read-only kernels

Make these generic over `V: ResolveNegativesView`:

- `compute_cancellations_range_inner`
- `compute_cancellations_range`
- `build_finality_preds_and_outdegree`
- `build_finality_reverse_topo_order`
- `collect_initial_final_weights`
- `transition_counts`
- `epsilon_count`
- terminal/default analysis helpers that only read graph shape

These kernels should return plain data structures:

- derived epsilon edges as `Vec<(u32, u32, Weight)>`
- predecessor lists / outdegrees
- reachable final weights as `Vec<Option<GuardedFinalWeight>>`
- terminal-state flags / prune decisions

### 3.2 Concrete application layer for `NWA`

Keep these concrete initially:

- apply derived epsilons to `NWA`
- write final weights into `NWA`
- remove negative transitions in place
- prune default targets in place

This yields the first safe milestone:

- generic read-only resolve kernels
- concrete `NWA` mutation wrappers
- zero change to external behavior or call sites

## 4. Proposed concrete adapter for current behavior

Add a tiny adapter in `resolve_negatives.rs` or nearby:

```rust
struct NwaResolveView<'a> {
    nwa: &'a NWA,
}

impl ResolveNegativesView for NwaResolveView<'_> {
    fn state_count(&self) -> usize { ... }
    fn final_weight(&self, state: u32) -> Option<&Weight> { ... }
    fn epsilons(&self, state: u32) -> &[(u32, Weight)] { ... }
    fn transitions(&self, state: u32) -> &BTreeMap<i32, Vec<(u32, Weight)>> { ... }
}
```

Then the existing entrypoint becomes structurally:

```rust
pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let view = NwaResolveView { nwa };
    let derived = compute_cancellations_range(&view, ...);
    apply_derived_epsilons_to_nwa(nwa, derived);

    let view = NwaResolveView { nwa };
    let reachable_finals = compute_finality_fixpoint(&view);
    write_final_weights_to_nwa(nwa, reachable_finals);

    remove_negative_transitions_in_nwa(nwa);

    let view = NwaResolveView { nwa };
    let terminal_flags = compute_terminal_default_prune_flags(&view);
    apply_default_prune_to_nwa(nwa, terminal_flags);
}
```

Behavior stays the same. Only ownership boundaries change.

## 5. How a future lazy parser graph would implement the trait

The future lazy path should not implement mutation against the raw symbolic graph.

Instead it should implement the same **read-only** trait over an interned symbolic graph.

### 5.1 Symbolic node identity

Intern:

- `Continuation { td_state }`
- `TemplateState { terminal, local_state, target_td_state }`
- `BundleState { bundle_id, local_state, target_td_state }`

to dense `u32` ids.

### 5.2 Backing storage the trait needs

For each interned symbolic node, the lazy graph needs cached raw-node records that can answer:

- `final_weight(node)`
- `epsilons(node)`
- `transitions(node)`

That does **not** require a flat parser-NWA append/copy step.

It only requires:

- a node intern table
- cached per-node raw successor records derived from reusable fragment NWAs and continuation summaries

### 5.3 How resolve-negative output is consumed later

The lazy graph should consume kernel outputs into a separate resolved-view cache, not mutate the raw graph in place.

That is why the first extraction should keep mutation out of the trait.

## 6. Smallest first commit

The first commit that should compile and preserve current behavior is:

### Commit 1: trait extraction plus NWA adapter only

Contents:

- add `ResolveNegativesView`
- add `NwaResolveView`
- make cancellation/finality/default-analysis kernels generic over the trait
- keep public entrypoint concrete on `&mut NWA`
- keep negative-removal and default-pruning mutation concrete
- no lazy graph code yet

Why this is the right first commit:

- mechanical
- low semantic risk
- compiles with zero caller churn
- gives the next commit a stable generic kernel surface

## 7. Mechanical refactor steps

### Step 1. Add the trait and NWA adapter

- define `ResolveNegativesView`
- implement `NwaResolveView<'_>`

### Step 2. Convert read-only helpers first

Generic over `V: ResolveNegativesView`:

- `transition_counts`
- `epsilon_count`
- `build_finality_preds_and_outdegree`
- `collect_initial_final_weights`

This is the cheapest compile-preserving start.

### Step 3. Convert cancellation kernel

Genericize `compute_cancellations_range_inner` and wrappers.

Keep output as plain `Vec<(u32, u32, Weight)>`.

### Step 4. Convert finality kernel

Make the predecessor build and fixpoint generic over the trait.

Return reachable final weights as a plain vector.

### Step 5. Convert terminal/default analysis to read-only kernel + concrete apply

Do not mutate inside the generic algorithm.

Instead:

- compute terminal-state flags or prune mask generically
- apply them concretely to `NWA`

### Step 6. Keep the public entrypoint concrete

The end of commit 1 should still expose only the current `NWA` entrypoint to the rest of the compiler.

## 8. Risk points

### Borrowing / lifetime pressure

Returning borrowed `BTreeMap` and slice references from the trait is fine for the first extraction, but it will tighten lifetime coupling in helper signatures.

Mitigation:

- use simple borrowed references, not custom iterator traits, in commit 1

### Over-generalizing too early

If the first commit introduces associated types or mutation traits, it will add churn without immediate value.

Mitigation:

- keep `u32`, `i32`, and `Weight` concrete in commit 1

### Terminal/default pruning is partly mutating

This is the phase most likely to tempt an over-wide trait.

Mitigation:

- split it into read-only terminal classification + concrete apply

### Dense-id assumption must stay explicit

The generic kernels still assume `Vec` indexing by dense ids.

Mitigation:

- make that part of the trait contract in commit 1
- future lazy graph interns symbolic nodes to dense ids

## 9. Test list

### Compile safety

- `cargo check`

### Existing behavior

- run current resolve-negative related tests unchanged
- run current parser-DWA tests unchanged

### Targeted unit tests to preserve through extraction

1. empty NWA
2. NWA with no negative transitions
3. negative cancellation through positive edge
4. negative cancellation through `DEFAULT_LABEL`
5. finality propagation across epsilon/default/negative cycle
6. redundant default pruning on terminal-shaped states
7. adapter parity test:
   - current `NWA` path vs generic-kernel-through-`NwaResolveView`
   - identical derived epsilons / final weights / pruned defaults

### Post-extraction confidence run

- `cargo check`
- one reduced JS compile/profile run on the unchanged concrete path

The point is not to improve anything yet, only to prove zero behavior change.

## 10. Bottom line

The safest Option 1 refactor is:

- **read-only graph/view trait first**
- **generic resolve kernels second**
- **keep mutation concrete on `NWA` for commit 1**

That produces the exact seam the future lazy parser graph needs, without prematurely forcing the wrong mutable abstraction onto the current resolver.