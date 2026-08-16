# Exact structural sharing for compiled subgrammars

This note states the correctness argument for the structural-sharing quotient
used by compiled-constraint composition.  The optimization is intentionally
grammar-agnostic.  It does not recognize JSON, schemas, or any other domain.

## 1. Setting

After ordinary component compilation and table splicing, let

- `T` be the finite set of terminal IDs,
- `N` the finite set of nonterminal IDs,
- `S` the finite set of LR stack-state IDs,
- `A(s,t)` the optimized table action at state `s` on terminal `t`,
- `G(s,n)` the goto edge from state `s` on nonterminal `n`, and
- `Adv(s)` the exact admission set captured by the table.

The parser stack is a word in `S*`.  Optimized actions may be ordinary shifts,
reductions, finite sets of stack rewrites, guarded stack rewrites, split
actions, accept, or identity skip.

The optimization constructs three equivalence relations, in this order:

1. a sufficient terminal-language relation `~T`;
2. a structural nonterminal relation `~N`;
3. a row-bisimulation relation `~S` on LR states.

Only the last relation physically removes LR states. Terminal and nonterminal
IDs are retained. `~T` lets one shared LR row carry several byte-language
aliases. `~N` is used only as a structural-isomorphism certificate when
matching independently compiled child machines; it is deliberately *not* used
to identify goto columns.

## 2. Terminal relation

For an ordinary byte terminal `t`, let `E(t)` be the retained lexer `Expr`.
The implementation relates two terminals only if neither terminal carries side
semantics not represented by the byte language (control, scoped skip, ignore,
placeholder, or special-token semantics), and one of these sufficient identity
certificates holds:

1. `E(t1) == E(t2)` structurally; or
2. both terminals are the same local terminal of the exact same compiled
   `Constraint` artifact reused at two composition sites.

Current constraint artifact version 11 persists the terminal `Expr` sidecar in
the versioned outer artifact, so independently compiled and independently
loaded current artifacts normally use certificate 1 directly. Older v10/v9/v7
artifacts remain loadable but do not carry this proof metadata. Distinct legacy
artifacts therefore remain separate by default.

An explicitly enabled legacy-artifact fallback can additionally prove equality
from the compiled tokenizer automata. It first tries exact rooted scalar-DFA
isomorphism and may then run bounded exact NFA-language equivalence. These are
proof mechanisms, never heuristics: budget exhaustion returns `Unknown`, which
means "leave distinct". Necessary-condition byte/vocabulary fingerprints are
used only to reject impossible candidates cheaply; equality of a fingerprint
never causes a merge. The fallback is intentionally not enabled by default,
because rediscovering proof structure from large old artifacts can be much more
expensive than carrying the compact v11 sidecar.

All other terminals are singleton classes.

### Lemma 1a — optional exact projected-NFA equivalence certificate

For a serialized tokenizer `L`, terminal `t`, and raw state `r`, retain `r` iff
`t` is a finalizer at `r` or `t` occurs in the exact possible-future metadata at
`r`. Let `P_t(L)` be the epsilon-NFA obtained by this live-state projection,
with the tokenizer reset as start and `t`-finalizing states as accepting.
Then `P_t(L)` recognizes exactly the byte language of terminal `t`.

**Proof.** A removed state neither accepts `t` now nor has a path to a state
that accepts `t`; by the definition of possible-future metadata it cannot
participate in any accepting `t` path. Conversely every state on an accepting
`t` path is either the final state itself or has `t` in its possible future, so
it is retained. Epsilon closure is computed before each projection, preserving
all zero-byte reachability among live states. Therefore projection removes
exactly states irrelevant to `t` and leaves its language unchanged. ∎

The checker explores the product of the on-the-fly subset constructions of
`P_t1(L1)` and `P_t2(L2)`. At every reachable subset pair it compares whether
the left and right subsets contain an accepting state. For every byte with an
outgoing transition on either side it advances all member states, takes exact
epsilon closure, projects to terminal-live states, and interns the resulting
pair. If acceptance ever differs, the BFS path is a concrete distinguishing
word. If the finite reachable product is exhausted without such a pair, the
symmetric difference is empty and the languages are equal. This is the
standard finite-NFA language-equivalence decision procedure performed lazily;
the implementation may abort with `Unknown` before exhaustion but never
returns `true` without exhausting the reachable product.

### Lemma 1 — terminal-language equality

If `t1 ~T t2`, then the set of byte strings recognized by `t1` equals the set
recognized by `t2`, including the accepted widths relevant to longest-match
execution.

**Proof.** In case 1, `Expr` is the denotational lexer expression. Structural
equality of `Expr` values implies equality under its language interpretation by
immediate induction over the expression constructors (`U8Seq`, `U8Class`,
`Dfa`, `Intersect`, `Seq`, `Choice`, `Exclude`, `Repeat`, `Shared`, and
`Epsilon`). `Exclude` and `Intersect` remain inside the expression, so equality
does not discard those semantics. In case 2 there is no semantic comparison at
all: both global aliases refer to the same local terminal in the same immutable
compiled tokenizer artifact, hence to the identical transition/finalizer
machine and accepted widths. If the explicitly enabled legacy fallback is used,
equality of canonical rooted scalar-DFA certificates is an exact labelled-graph
isomorphism proof; otherwise Lemma 1a plus exhaustion of the subset-product
search proves equality of the two compiled byte languages. The exclusion of
terminals with side semantics
removes every known terminal behavior not represented by that byte matcher.
Therefore the byte match relation and every match width are equal. ∎

This is intentionally incomplete. Legacy artifacts without retained proof
metadata stay unmerged by default, and optional exact search may also decline on
its resource bound. Both failure modes lose optimization only, never semantics.

## 3. Structural nonterminal relation

For every `n ∈ N`, write `P(n)` for the ordered-symbol right-hand sides of all
productions whose left-hand side is `n`.  Production order is ignored, while
duplicate productions are retained.

Starting from a coarse partition that distinguishes

- the augmented start nonterminal from every other nonterminal, and
- boundary roots from non-boundary nonterminals,

the implementation repeatedly refines classes by the signature

```
{ map(rhs, T -> class_T, N -> previous_class_N) : rhs in P(n) }.
```

The finite refinement terminates at a fixed point `~N`.

### Lemma 2 — equal grammar equations

If `n1 ~N n2`, their production equations are identical after quotienting
terminal symbols by `~T` and nonterminal variables by `~N`.

**Proof.** This is exactly the fixed-point condition of the refinement. ∎

### Lemma 3 — structural nonterminal language equality

Members of one `~N` class generate the same byte language, modulo substitution
of `~T`-equivalent terminal aliases.

**Proof.** A CFG defines the least fixed point of a monotone system of language
equations over the complete lattice of tuples of languages.  By Lemma 2,
members of one `~N` class have the same polynomial equation after quotienting
variables by `~N`; by Lemma 1, related terminal constants have equal byte
languages.  The subspace in which all variables in one class are equal is
therefore closed under the grammar functional.  Kleene iteration from the
bottom element remains in that subspace at every finite iteration, so the least
fixed point does as well. Thus every member of one class has the same generated
byte language. ∎

The augmented-start nonterminal anchor preserves the grammar-root semantics.
Separately, the LR-state refinement anchors parser state `0` so a quotient that
is later reused as a subgrammar cannot identify its entry row with an internal
row. The
boundary anchor is stronger than required for grammar language equality; it
prevents candidate matching from enlarging boundary-analysis scope.

Crucially, Lemma 3 does **not** justify physically identifying the nonterminal
IDs in an LR table. One caller state may legitimately have `goto(n1) !=
goto(n2)` even when `L(n1) = L(n2)`. The implementation therefore retains
concrete nonterminal identity in the ordinary LR quotient. `~N` is used later
only to establish that two independently compiled child states have the same
standalone grammar shape before caller-specific behavior is considered.

## 4. LR-state relation

The LR-state relation is the greatest fixed point reached by finite partition
refinement from one coarse state class.  A state's refinement signature contains
all execution observations, normalized through the current state partition:

- action columns keyed by `~T` class;
- every action target state mapped through the current state partition;
- concrete reduction nonterminal IDs unchanged;
- concrete goto columns with mapped target states;
- the exact `advance` set, projected through `~T`;
- forwarded-shift status;
- direct-regular wide-frontier descriptors with mapped targets; and
- exact membership in every guarded-stack-shift predicate; and
- the distinguished parser entry state `0` is kept in its own initial colour.

If two different terminal aliases in one physical row have different normalized
actions, that row is forced to remain a singleton. Thus every non-singleton
quotient class has one well-defined action per terminal-language class and one
well-defined goto for every concrete nonterminal ID.

The previous partition class is part of every refinement signature, so the
algorithm only splits classes.  It terminates after at most `|S|-1` strict
refinements.

### Lemma 4 — guarded predicates descend exactly

For every guarded stack predicate `H ⊆ S`, `H` is a union of final `~S`
classes.

**Proof.** Exact membership in every guard occurrence is part of the initial
observable color and is retained by all later refinements.  Hence no final class
contains one member of `H` and one non-member. ∎

This condition is essential.  Without it, mapping guard-state sets through the
quotient could turn a predicate that distinguished two old stack states into a
predicate that accepted their common quotient state.

## 5. Quotient construction

Let `q : S -> S/~S` map an old LR state to its quotient state.

For each quotient state `Q`, the implementation unions the concrete terminal
columns of all `s ∈ Q`. State targets are mapped by `q`; reduction and goto
nonterminal IDs remain concrete. The fixed-point signature guarantees that any
collision has exactly the same normalized action/goto. The implementation
checks this again while materializing the quotient and treats a disagreement as
an internal proof violation.

`advance`, forwarded shifts, guarded state sets, wide-frontier state IDs, and
every component parser-state relation are transported by `q`. Rules,
nonterminal IDs, and boundary nonterminal IDs remain unchanged. Guard indices
are then rebuilt from the quotient table.

### Lemma 5 — action homomorphism

Let `q*` map every LR state in a stack to its `~S` class.  For any old stack
`σ·s` and terminal `t`, every result of applying `A(s,t)` and then mapping the
result stack by `q*` is exactly a result of applying the quotient action at
`q(s)` on `t`; conversely every quotient result is the image of an old result
for some member of the quotient class with a `~T`-equivalent terminal alias.

**Proof by action cases.**

- **Shift / replace shift.** Targets are mapped directly by `q`; the replace
  flag and forwarded-shift observation are part of the signature.
- **StackShifts / ReplaceShifts.** Pop counts are unchanged and each pushed
  state is mapped by `q`.  Sorting/deduplication removes only duplicate
  alternatives that became identical after quotienting.
- **GuardedStackShifts.** The same stack rewrite argument applies.  By Lemma 4,
  every guard predicate is saturated by quotient classes, so evaluating the
  mapped guard on `q*(σ)` gives exactly the old truth value.
- **Reduce.** The pop count and reduced nonterminal are unchanged. The concrete
  goto column is part of the stable row signature and its target is mapped by
  `q`; therefore the post-reduction quotient state is identical.
- **Split.** It is the union of the preceding shift/reduce cases.  Duplicate
  alternatives that become identical may be deduplicated because recognition
  weights use idempotent union.
- **Accept / Skip.** They contain no mapped identity and are unchanged.

Thus the action relation commutes with `q*`. ∎

### Lemma 6 — row bisimulation

If `s1 ~S s2`, then for every terminal-language class the two states have the
same admission decision and the same quotient action, and for every concrete
nonterminal ID they have the same quotient goto.

**Proof.** This is exactly equality of the stable structural state signatures;
rows with non-uniform aliases are excluded from non-singleton classes. ∎

### Theorem 1 — parser preservation

Starting from the mapped initial stack, the quotient parser recognizes exactly
the same byte language as the unquotiented composed parser.

**Proof.** Induct on consuming parser steps, using Lemma 5 for the transition
step and Lemma 6 for availability of that step.  Reductions are internal steps
covered by the same action homomorphism.  Acceptance is unchanged.  The
quotient may admit additional *terminal-ID spellings* obtained by substituting
one `~T` alias for another, but by Lemma 1 those spellings have exactly the same
byte language and match widths.  Therefore their projection to byte strings is
unchanged. ∎

## 6. Context-distinguishable sharing

The row bisimulation above is intentionally conservative. Two copies of one
child machine can acquire different return lookaheads when linked at different
call sites. Ordinary row bisimulation then propagates that difference backwards
through the copied child even though the parser stack still contains enough
caller context to distinguish the copies.

The second quotient recovers exactly this case without erasing that context.

### 6.1 Candidate correspondence

Candidates are found from the independently compiled child tables, *before*
caller-specific return lookaheads are introduced. Child states are compared by
finite partition refinement over their complete table behavior:

- terminal columns are compared modulo `~T`;
- reduction/goto nonterminals are compared modulo `~N`;
- target and guard states are compared modulo the previous state partition;
- `advance` and forwarded-shift observations are included; and
- start states are kept in a separate initial color from internal states.

Only a class containing states from at least two different child components is
proposed to the contextual quotient. This is a structural-isomorphism
certificate for the standalone parser submachines, not a source-domain or name
heuristic.

### 6.2 Predecessor provenance

For an LR state `s`, let `Pred(s)` be a conservative set of LR states that can
occur immediately below `s` on a reachable parser stack.

For an ordinary push edge, the source state is inserted into the target's
predecessor set. For a replace edge, predecessor sets propagate through the edge
to a fixed point. Goto edges are treated identically. This recurrence is
complete for states not entered by an arbitrary precompiled stack effect. Any
target of `StackShifts`, `GuardedStackShifts`, or `ReplaceShifts`, and every
state that inherits such provenance through a replace edge, is therefore marked
unsafe and is not contextually merged.

Hence every concrete immediate predecessor of every accepted state `s` belongs
to `Pred(s)`.

An accepted candidate class `C` additionally satisfies:

1. `Pred(s)` is non-empty for every `s ∈ C`;
2. `Pred(s1) ∩ Pred(s2) = ∅` for distinct `s1,s2 ∈ C`;
3. every state in every `Pred(s)` is frozen and cannot itself be merged;
4. no state observed by an existing guarded predicate is merged; and
5. every LR state mentioned by a *new* guard produced while compiling the
   candidate macro row is also frozen before candidate classes are accepted;
6. no generated macro guard may observe parser state `0`, the distinguished
   standalone subgrammar entry state; and
7. the concrete goto rows of all members agree exactly, because gotos cannot
   carry the terminal-action provenance guard used below.

These conditions make the immediate predecessor a stable, injective provenance
tag for the old top-state identity.

### 6.3 Guarded macro state

For an accepted class `C`, introduce one shared LR state `Q_C`. Every incoming
edge whose old target was `s ∈ C` is redirected to `Q_C`.

For each old action `A(s,t)`, construct the symbolic stack-effect frame

```
pop    = 1
pushes = [s]
guard  = (after popping 1, state ∈ Pred(s)).
```

Operationally, this says: replace the shared top `Q_C` by its concrete old
representative `s`, but only in stack contexts that could have produced `s`.
The existing exact stack-effect compiler then executes `A(s,t)`, including all
reduction/goto closure needed before `t` is consumed, and returns the equivalent
`GuardedStackShifts` macro effect. Pushed LR-state IDs are finally transported
through all accepted sharing classes.

The composed table uses `AdmissionPolicy::ExactSimulation`; its `advance` row
for `Q_C` is therefore safely the union of constituent admission rows. That
union is only a prefilter. The guard-bearing action is simulated before a
terminal is declared admissible.

### Lemma 7 — unique source recovery

Let `α p Q_C` be a reachable stack whose top was obtained by redirecting an old
state in `C`. There is exactly one `s ∈ C` whose provenance guard accepts it,
and that `s` is the old state represented by this occurrence of `Q_C`.

**Proof.** The redirected occurrence originated from some concrete member
`s ∈ C`. Completeness of the safe predecessor analysis gives
`p ∈ Pred(s)`. Pairwise disjointness gives `p ∉ Pred(s')` for every
`s' != s`. The predecessor IDs are frozen, so later quotients cannot destroy
that distinction. ∎

### Lemma 8 — contextual action equivalence

For every reachable `α p Q_C` and terminal `t`, applying the guarded macro row
at `Q_C` produces exactly the image of applying the old action at the concrete
member represented by that occurrence.

**Proof.** By Lemma 7 exactly one member's initial frame is enabled. Replacing
`Q_C` by that member reconstructs the old stack before the action. The
stack-effect compiler is the same exact reduction/shift closure used by the
ordinary table optimizer, so its emitted macro effect has the same result as
the old action through the first consuming step. All other members' frames are
eliminated by their predecessor guards. Mapping pushed target states through
the sharing map merely applies the same representation invariant recursively.
∎

The viability pass performs this macro compilation before the accepted quotient
is fixed and records every LR-state ID appearing in any generated guard. Those
states are frozen along with the predecessor provenance states. Consequently a
later shared-state mapping cannot broaden a generated predicate by identifying
a guarded state with an unguarded one.

### Theorem 2 — contextual table sharing is exact

Redirecting every accepted candidate class to its shared guarded state preserves
the parser's byte language and exact admission relation on every reachable
stack.

**Proof.** Induct over consuming parser steps. The base stack contains no shared
state. For a non-shared top, table behavior is unchanged except that target IDs
may be represented by their shared state. For a shared top, Lemma 8 gives exact
step equivalence. The representation invariant is preserved by transported
pushes. Exact admission performs the same guarded simulation, so the unioned
`advance` row cannot introduce a false admitted terminal. Accepting actions are
deliberately not contextually merged because `Accept` has no guard-bearing
representation. ∎

The safety restrictions are intentionally one-sided. If predecessor provenance
is complex, overlapping, or entangled with existing guards, the candidate is
simply left unmerged.

### Lemma 9 — closure under later subgrammar composition

The legacy table splice gives standalone parser state `0` an additional
*compositional* meaning: that row is not copied into the parent as an ordinary
child state. Instead it is overlaid onto each parent call site. Consequently, a
standalone table can be language-correct while still being unsuitable as a
future child if one of its internal guarded macro actions observes state `0`.
The observation would have to be translated to caller-state provenance at the
next composition level, which the current splice representation does not encode.

The contextual quotient therefore rejects any candidate whose generated macro
row contains state `0` in a guard. Rejection leaves the pre-quotient table
unchanged, so standalone semantics are unchanged. Since the pre-quotient
composition has no internal reference to its distinguished entry state, the
optimization cannot introduce one. Thus contextual sharing preserves the table
invariant required for exact later composition. ∎

This restriction is deliberately conservative. A future linker could transport
such a guard by explicitly representing caller provenance, but merely treating
state `0` as a normal child state or broadening it to all callers without a
correlation proof would not be sufficient.

## 7. Nested tokenizer-coordinate transport

A composed constraint may itself carry the exact runtime lexer product. Such a
product state can represent several pre-product tokenizer-state classes (TSIDs),
so nested composition must not assume a raw tokenizer state belongs to exactly
one local TSID.

For a component raw tokenizer state `s`, let `T(s)` be the finite set of local
TSIDs represented by `s`. Nested composition defines the global raw-state
coordinate by the exact membership signature:

```text
s1 ~ s2  iff  T(s1) = T(s2)
```

within each independently compiled component. Let `G_S` denote the resulting
global class for signature `S`. A local TSID `t` is transported to every global
class whose signature contains it:

```text
t  ->  { G_S | t in S }.
```

The merged reset state is additionally included in the image of every TSID
represented by a component reset.

### Lemma 10 — exactness of TSID membership-signature lifting

For every component raw tokenizer state `s` and every local parser-DWA weight
`W`, evaluating the remapped weight at `G_T(s)` is exactly the union of the
local weight contributions for the TSIDs represented by `s`.

**Proof.** By construction, a local TSID `t` maps to `G_T(s)` exactly when
`t in T(s)`. Therefore every local contribution present at `s` is transported
to the global class, and no contribution from a TSID absent at `s` is
transported there. Weight reconciliation combines colliding transported
contributions with the same semiring union already used for tokenizer-state
ambiguity. Hence the remapped global weight is exactly the local multi-TSID
weight at `s`. Grouping two raw states only when their complete `T` signatures
are equal preserves this observation for every parser-DWA weight. ∎

When every raw state has singleton membership, `T(s) = {t}`, this construction
is exactly the previous one-global-class-per-local-TSID coordinate. Thus the
general nested path is a conservative extension of the original fast case.

## 8. Parser-DWA transport

Compiled component parser DWAs do not contain terminal IDs.  Their labels are
positive and negative LR-state observations.  Composition already transports a
local LR state through a relation

```
local state -> one or more composed states.
```

After structural sharing this relation is simply post-composed with `q` and
deduplicated.  Both positive and negative labels use the same mapping.  The
existing NWA union/determinization then combines any transitions whose labels
became identical.

### Theorem 3 — parser-DWA reuse remains exact

Transporting an already compiled component parser DWA through the updated state
relation denotes the same weighted token relation as compiling that component's
behavior against the quotient table.

**Proof.** Parser-DWA state labels observe only LR stack-state identity.  By
Theorem 1, `q` is a congruence for every parser transition and acceptance
observation.  Replacing each label by its image under `q` is therefore a
homomorphic relabeling of the recognized stack-effect language.  When multiple
old labels acquire one quotient label, NWA union followed by determinization
computes the idempotent union of those equal quotient behaviors, which is
exactly the quotient relation.  TSID × vocabulary weights are unchanged by the
LR-state quotient. ∎

For context-distinguishable sharing, labels are merged only when their
standalone child states lie in the same structural class from §6.1. That
standalone table/terminal isomorphism induces identical component parser-DWA
behavior modulo the same LR-state, TSID, and vocabulary renamings. The
caller-specific differences introduced only by linking are handled by the
guarded composed-table behavior of Theorem 2; they are not part of the reused
child DWA.

### 8.1 Exact additive boundary repair

Reusing a component parser DWA supplies the component-local baseline. Linking
can enlarge the stack-effect template of an ordinary terminal `t` from
`O_t` (the transported component template) to `N_t` (the composed-table
template). The optimized boundary builder uses an additive repair only after
proving

```text
O_t ⊆ N_t.
```

It then materializes the exact remainder

```text
Δ_t = N_t \ O_t,
```

so `N_t = O_t ⊎ Δ_t`, where `⊎` denotes disjoint union. If the inclusion proof
fails, the terminal is marked unsafe and the ordinary full boundary
construction is retained.

For a component-local terminal word `w = t1 ... tn`, let `C` be the positions
whose terminal template changed. The all-old product

```text
O_t1 ... O_tn
```

is already recognized by the transported component parser DWA. Every point of
the new product that is absent from the all-old product has a unique first
changed position `j ∈ C` at which it chooses `Δ_tj`. Hence

```text
N_t1 ... N_tn \ O_t1 ... O_tn

  = ⊎_{j ∈ C}
      P_1 ... P_{j-1} Δ_{t_j} N_{t_{j+1}} ... N_{t_n},
```

where `P_i = O_{t_i}` for changed positions before `j`, and
`P_i = N_{t_i} = O_{t_i}` for unchanged positions. The boundary builder implements exactly this
"first-delta" decomposition with a two-state boolean product recording whether
the first delta has already occurred. It therefore adds every newly legal
component-local stack effect exactly once and does not rebuild the cached
all-old branch.

Scoped skip terminals require one extra distinction. A component's top-level
ignore has its unqualified empty-word identity support removed from the cached
standalone parser artifact; a token consisting only of that scoped ignore must
therefore restore precisely that identity support at the linked boundary. In a
mixed local token, that same scoped `Skip` is parser identity and can be erased
inside the first-delta product. Inherited scoped skips from an already-composed
component may encode a real phase transition and are not erased by this rule.
Any lexical path that actually crosses component ownership remains on the full
boundary lane.

**Theorem 3a — additive boundary repair is exact.** The union of transported
component parser behavior and the additive boundary repair recognizes exactly
the parser behavior obtained by compiling the same composed table directly.

**Proof.** Component-local all-old behavior is supplied by Theorem 3. For every
safe changed ordinary terminal, `N_t = O_t ⊎ Δ_t` by construction. The unique
first-delta partition above is therefore an exhaustive and disjoint partition
of the new component-local behavior not already cached. Pure stripped-ignore
identity is restored separately, mixed stripped-ignore occurrences are parser
identity, and inherited scoped skips are never erased without the corresponding
proof. Cross-component and unsafe paths use the full composed templates, so no
factorization assumption applies to them. These cases partition all accepting
boundary-token paths. Their union is consequently exactly the direct composed
template semantics. ∎

### 8.2 Exact direct union and deferred support evaluation

The final parser DWA is the weighted union of the transported component DWA and
the boundary-repair DWA. A determinized state is represented by a finite
residual subset

```text
{ (q_i, W_i) },
```

where `q_i` is a source state and `W_i` is the residual positive support for
that source lane. Residual subsets are interned by `(q_i, W_i)` using structural
`Weight` equality and hashing; storage-pointer identity is not semantic state.

For a label with exactly two non-empty source contributions
`(q1, A)` and `(q2, B)`, the outgoing edge support is `A ∪ B`, while the target
residual subset is determined by the contributions themselves. Under the
positive-support representation used here, support outside `A ∪ B` is never
observed after taking the edge. Thus construction of the graph topology does
not depend on materializing `A ∪ B`: the union may be evaluated later and
patched onto the already-created ordinary DWA edge. The same observation holds
for a state's final support, which affects acceptance weight but no successor.
Independent support unions are therefore evaluated in parallel after topology
construction. Cases with more than two contributions retain the general eager
path.

The pair-state row merge may additionally process maximal consecutive label
intervals on which both source target/weight vectors are constant. Such an
interval is only a run-length representation of repeated identical transition
equations: its successor and support are solved once and then expanded back to
the ordinary per-label DWA representation. The symbolic `DEFAULT` transition
is kept separate; interval boundaries include the point at which non-negative
labels become eligible for DEFAULT fallback.

**Theorem 3b — direct union scheduling preserves the weighted language.**
Structural residual interning, interval-wise row processing, and deferred
two-way edge/final support unions produce the same ordinary weighted DWA as the
corresponding eager per-label construction, up to deterministic state naming.

**Proof.** Structural residual interning identifies exactly equal residual
relations, so it changes representation only. On each maximal label interval,
the source transition equations are constant by definition, and expansion
reinstates the same equation for every member label; DEFAULT-only gaps are not
expanded. Deferred support evaluation changes only evaluation order:
set/weight union is a pure operation on immutable operands, and the computed
support is written to the same edge or final state whose topology was derived
from those operands. No deferred value participates in discovery of another
state. Therefore every transition target, edge support, and final support is
identical to the eager construction. ∎

## 9. Exact runtime lexer product

The parser quotients above retain logical terminal IDs. To reduce duplicated
*lexer frontier* states, composition optionally installs the same exact runtime
product representation already used by glrmask's generic full-runtime lexer
determinization.

Let the composed source tokenizer be the epsilon-NFA `L` with raw states `R`.
Exact subset construction creates a deterministic product state for each
reachable epsilon-closed subset `P ⊆ R`. For every product state the artifact
stores

- the exact source subset `P`;
- an exact scalar source representative when `P` is one source state's epsilon
  closure;
- the union of the already-compiled TSID classes represented by states in `P`.

The complete original source tokenizer is then appended unchanged after the
product states. Runtime masking may keep a single product key while parser
histories are uniform across its source subset. Before commit needs per-source
longest-match provenance, `expand_runtime_product_states` expands that product
key back to the exact source states in `P`; the source tokenizer then executes
unchanged. After commit, compatible source lanes may be coalesced again.

### Lemma 9 — tokenizer subset equivalence

For every byte word `w`, the deterministic product state reached from subset
`P` is exactly the epsilon closure of the union of source states reachable from
members of `P` on `w`.

**Proof.** This is the standard subset-construction invariant. The builder
computes, for each byte, the union of all source transitions and then takes the
exact epsilon closure before interning the target subset. Induction on `|w|`
gives the claim. ∎

### Lemma 10 — TSID observation preservation

Every product state exposes exactly the union of TSID observations of the
source states in its subset.

**Proof.** Before installing the product, composition snapshots the exact
raw-state-to-TSID relation. For every subset `P` it unions and deduplicates the
TSIDs of all `r ∈ P`; runtime cache finalization rebuilds the general
many-TSID-per-state relation from that mapping. No TSID is invented or removed.
∎

### Theorem 4 — runtime lexer product is exact

Installing the product+source-fallback representation preserves masks, token
acceptance, longest-match behavior, commits, and the recognized byte/token
language.

**Proof.** By Lemma 9, a product state is an exact compact representation of its
source lexer lanes. By Lemma 10, parser-DWA/possible-match weight queries see
exactly the union of the source TSID observations while the state remains
compact. Whenever a consuming operation requires lane-specific provenance, the
runtime expands the product state to exactly its stored source subset and runs
the unchanged source tokenizer. Thus no longest-match/source history is lost;
coalescing is only a reversible representation change. The reset product state
has special handling because it represents one historical reset lane whose
epsilon closure contains all component roots. Therefore every mask and commit
step has the same source semantics as the unproductized composition. ∎

### Adaptive selection is performance-only

Correctness does not depend on the selection heuristic. By default composition
attempts the runtime product only when there are structurally equal terminal
aliases and the source tokenizer is small enough. It selects the product only
when

1. product-state count is reduced by the configured minimum (25% by default),
2. transition growth stays bounded, and
3. excluding the reset product state, some product subset contains at least two
   concrete consuming source lanes whose future terminals are simultaneously
   admitted by one parser row.

Condition 3 prevents the common sequential case where duplicate lexer machines
are statically isomorphic but the parser can enable only one copy at a time.
An explicit environment override may force the exact product on or off for
measurement; this changes performance only, not semantics.

The representation reduces the visible/persistent lexer frontier, not total
stored tokenizer memory: the exact source tokenizer is intentionally retained
as the commit fallback.

## 10. What this proof does *not* assume

The proof does not assume anything about JSON, JSON Schema, programming
languages, or the source that generated a grammar.  It also does not assume
that independently optimized components chose identical raw LR state numbers.
Only the finite structural relations above are used.

The proof intentionally does **not** claim completeness: semantically equal but
structurally different terminals/nonterminals/states may remain separate.  That
is the preferred failure mode.  Structural sharing is an optimization, so a
missed merge costs performance; an unjustified merge would cost correctness.
