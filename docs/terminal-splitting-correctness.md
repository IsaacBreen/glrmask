# Correctness proof for exact terminal splitting

This document proves semantic correctness of the terminal-splitting pass in
`src/compiler/grammar/terminal_splitting.rs`.

The result is a **relative compiler-pass theorem**: for every valid input
`GrammarDef`, the transformed grammar denotes exactly the same language as the
input grammar. Consequently, its ideal next-token mask and EOS decision are
identical at every prefix. This theorem does not assume that the heuristic which
chooses whether splitting is profitable is accurate; profitability affects only
whether the proved transformation is applied.

The theorem is independent of the current default setting. The pass remains
disabled by default for deployment and performance-policy reasons.

## 1. Semantic model

Let `B = {0, ..., 255}` be the byte alphabet. To include special-token
terminals, let `S` be a set of symbols disjoint from `B`, and define the grammar
alphabet

```text
Ω = B ∪ S.
```

A regular `Expr` denotes a language over `B*`. Write its denotation as `⟦E⟧`.
The relevant constructors have their standard meanings:

```text
⟦Epsilon⟧             = {ε}
⟦Seq(E1, ..., Em)⟧    = ⟦E1⟧ ... ⟦Em⟧
⟦Choice(E1, ..., Em)⟧ = ⟦E1⟧ ∪ ... ∪ ⟦Em⟧
⟦Repeat(E, i, j)⟧     = ⋃{⟦E⟧^n : i ≤ n ≤ j}
⟦Shared(E)⟧           = ⟦E⟧.
```

`Intersect` and `Exclude` denote language intersection and difference.

A regular grammar terminal containing expression `E` denotes `⟦E⟧`. A special
terminal denotes one singleton symbol from `S`. Thus every grammar terminal
`t` denotes a language `λ(t) ⊆ Ω*`.

The language of a `GrammarDef` is the ordinary context-free derivation language
with terminal symbols interpreted by language substitution through `λ`. Write
that language as `L(G) ⊆ Ω*`.

For integers `i ≤ j`, use the notation

```text
A^[i,j] = ⋃{A^n : i ≤ n ≤ j}.
```

## 2. The theorem

**Theorem 1 — terminal-splitting language preservation.**

Let `G` be a valid `GrammarDef`. Let `Split(G, V, C)` be the result of
`split_with_config` for any vocabulary `V` and configuration `C`, assuming the
function returns normally. Then

```text
L(Split(G, V, C)) = L(G).
```

The vocabulary and cost configuration may change which terminals are selected
and which legal block size is chosen, but they cannot change the language.

The proof is built from the following lemmas.

## 3. Correctness of repeat-context extraction

When `extract_repeat_context(E, min_repeat)` returns `One(C)`, define

```text
P  = C.prefix
A  = C.body
N  = C.max_repeat
Q  = C.passthrough
Sx = C.suffix.
```

**Lemma 2 — extraction invariant.**

For every successful `One(C)` result,

```text
⟦E⟧ = (⋃{⟦Qj⟧ : Qj ∈ Q}) ∪ ⟦P⟧ ⟦A⟧^[0,N] ⟦Sx⟧.     (1)
```

**Proof.** By induction over the recursive return path of
`extract_repeat_context`.

1. **Bounded-repeat base case.** For `E = Repeat(A, 0, Some(N))`, the function
   returns `P = ε`, `Sx = ε`, and `Q = ∅`. Equation (1) is exactly the semantic
   definition of bounded repetition.

2. **Sequence case.** Suppose

   ```text
   E = B · Ei · D
   ```

   where `Ei` is the unique child returning a context and `B`, `D` are the
   concatenations before and after it. By the induction hypothesis,

   ```text
   ⟦Ei⟧ = ⋃j ⟦Qj⟧ ∪ ⟦P⟧⟦A⟧^[0,N]⟦Sx⟧.
   ```

   Concatenation distributes over union, so

   ```text
   ⟦E⟧ = ⋃j ⟦B Qj D⟧ ∪ ⟦B P⟧⟦A⟧^[0,N]⟦Sx D⟧.
   ```

   This is precisely the context constructed by the sequence branch.

3. **Choice case.** Suppose one option `Ei` returns a context and all other
   options are copied into `passthrough`. The language of `Choice` is union.
   Adding the untouched options to the induction-hypothesis passthrough union
   gives exactly (1).

4. **Shared case.** `Shared(Ei)` has the same language as `Ei`, and the function
   recurses without changing the returned context.

If two recursive children contain candidates, the function returns `Multiple`
and no transformation is performed. All other expression forms return `None`.
Therefore every context used by the pass satisfies (1). ∎

The non-nullability and minimum-length checks are conservative admission checks.
They are not needed for equation (1), but they ensure that generated terminal
pieces are nonempty and satisfy downstream compiler invariants.

## 4. Soundness of intersection selection

The pass sometimes sees

```text
E = X ∩ Y
```

where only one side has the required repeat shape. It selects `X` only after
obtaining a positive certificate for `⟦X⟧ ⊆ ⟦Y⟧`, or symmetrically selects `Y`
after proving `⟦Y⟧ ⊆ ⟦X⟧`.

**Lemma 3 — certified intersection elimination.**

If `⟦X⟧ ⊆ ⟦Y⟧`, then

```text
⟦X ∩ Y⟧ = ⟦X⟧.
```

The symmetric statement also holds.

**Proof.** `⟦X ∩ Y⟧ = ⟦X⟧ ∩ ⟦Y⟧`; intersecting a set with a superset leaves the
set unchanged. ∎

It remains to prove that every returned positive subset certificate is sound.

### 4.1 Exact DFA-product certificate

For deterministic byte automata `DL` and `DR`, the product search starts at the
pair of start states. A missing right transition is represented by a dead state.
It explores every pair reachable by a word accepted along the left automaton. If
it reaches a pair where the left state accepts and the right state does not, that
word is a counterexample. If the finite reachable product is exhausted without
such a pair, every word accepted by the left automaton is accepted by the right.
Therefore the returned `Holds` result proves inclusion.

A state-pair budget can stop the search, but budget exhaustion returns
`BudgetExceeded`, never `Holds`.

### 4.2 Common-context reduction

The implementation may reduce

```text
P X S ⊆ P Y S
```

to the sufficient obligation `X ⊆ Y` when the surrounding expression atoms are
syntactically identical.

**Lemma 4 — context monotonicity.** If `X ⊆ Y`, then for arbitrary languages
`P` and `S`,

```text
P X S ⊆ P Y S.
```

**Proof.** Every element of `P X S` has a factorization `p x s` with `x ∈ X`.
Since `X ⊆ Y`, the same factorization belongs to `P Y S`. ∎

The converse is not true in general because language concatenation is not
cancellative. Accordingly, the code exposes a failed residual check only as
`NotProved`; it does not claim that the original inclusion is false.

### 4.3 Kleene-closure certificates

Let `A*` be a Kleene closure.

- `ε ∈ A*`.
- If `Xi ⊆ A*` for every sequence factor, then
  `X1 ... Xm ⊆ A*`, because `A* A* = A*`.
- If `X ⊆ A*`, then every finite or unbounded repetition of `X` is also a
  subset of `A*`.

All other forms are checked by the exact DFA-product procedure against `A*`.
Thus every structural `Holds` result is sound.

For a target `A+`, if `A` is nullable then `A+ = A*`. If `A` is non-nullable,
then `A* = {ε} ∪ A+`; the implementation additionally requires the left
language to be non-nullable. Hence a proved inclusion in `A*` implies inclusion
in `A+` in either case.

Combining these arguments proves:

**Lemma 5 — positive-certificate soundness.** Whenever `certify_subset` returns
`Holds` for expressions `X` and `Y`, `⟦X⟧ ⊆ ⟦Y⟧`.

`NotProved` and `BudgetExceeded` carry no positive or negative semantic claim.
They simply cause the terminal to remain unchanged.

By Lemmas 3 and 5, candidate selection preserves the original intersection
language.

## 5. Exactness of the counted decomposition

By Lemma 2, after any certified intersection elimination the selected language
has the form

```text
Q ∪ P A^[0,N] S,
```

where `Q` is the passthrough union.

Let the selected block size be `K`, with `1 ≤ K ≤ N`. The implementation uses
`K ≥ 2`, which is stronger than needed for language correctness. By Euclidean
division, there are unique integers `q ≥ 1` and `0 ≤ r < K` such that

```text
N = qK + r.
```

The generated counted alternatives denote the following repetition-count
intervals:

```text
I0 = [0, K-1]
Ij = [jK, (j+1)K-1]       for 1 ≤ j ≤ q-1
Iq = [qK, qK+r].
```

The middle family is empty when `q = 1`.

**Lemma 6 — exact count partition.**

```text
I0 ∪ I1 ∪ ... ∪ Iq = [0, N],
```

and the intervals are pairwise disjoint.

**Proof.** `I0` begins at zero. For each `j < q`, the endpoint of `Ij` is
`(j+1)K-1`, exactly one less than the start `(j+1)K` of the next interval. The
last interval ends at `qK+r = N`. Thus the intervals are contiguous, disjoint,
and cover every integer from zero through `N` exactly once. ∎

For every `j`, the corresponding generated grammar alternative has language

```text
⟦P⟧ ⟦A⟧^Ij ⟦S⟧.
```

For example, an intermediate alternative contains `j` exact `A^K` chunks and a
final `A^[0,K-1]` tail, producing precisely counts `jK` through
`(j+1)K-1`. The final alternative contains `q` exact chunks and an
`A^[0,r]` tail.

Whether the prefix is fused into the first generated terminal changes only the
parenthesization:

```text
(P A^K) A^K ... T = P A^K A^K ... T.
```

Language concatenation is associative, so both modes denote the same language.

Using Lemma 6 and distributivity of concatenation over union,

```text
⋃j ⟦P⟧⟦A⟧^Ij⟦S⟧
  = ⟦P⟧ (⋃j ⟦A⟧^Ij) ⟦S⟧
  = ⟦P⟧⟦A⟧^[0,N]⟦S⟧.
```

Adding the untouched passthrough union `Q` yields exactly the selected source
expression. Ambiguity inside `A` is irrelevant: this is an equality of
languages and concatenation powers, not of parse counts or unique
factorizations.

Therefore:

**Lemma 7 — replacement-expression equality.** For every admitted split plan,

```text
⟦replacement_expr(plan)⟧ = ⟦source_expr(plan)⟧.
```

## 6. Grammar substitution

For a selected source terminal `t`, the pass creates a fresh nonterminal `Xt`.
Each production of `Xt` is a sequence of generated terminals whose concatenated
language is one alternative from Lemma 7. Hence

```text
L_G'(Xt) = λ_G(t).                                      (2)
```

The pass then replaces every occurrence of `t` in every pre-existing rule by
`Xt`.

**Lemma 8 — terminal-to-nonterminal substitution.** Replacing a terminal `t` by
a fresh nonterminal `Xt` satisfying (2) preserves the grammar language.

**Proof.** Consider any derivation in the original grammar. Whenever it uses
`t` to emit a word `w ∈ λ(t)`, equation (2) supplies a derivation of `Xt` that
emits the same `w`; replacing each such step gives a derivation in the new
grammar with the same output word.

Conversely, whenever a new derivation expands `Xt`, equation (2) says that the
emitted word belongs to `λ(t)`. Replacing that expansion by one use of `t`
gives an original-grammar derivation with the same output. Thus the two start
languages are mutually included. ∎

All new nonterminals are fresh, and their productions contain generated
terminals rather than source terminals selected for replacement. Therefore
multiple terminal replacements are independent; applying Lemma 8 repeatedly
proves language preservation for all planned splits.

The old terminal definitions may remain temporarily in `grammar.terminals`, but
no transformed rule refers to them. Removing unused terminal definitions later
does not change derivations.

This completes the proof of Theorem 1. ∎

## 7. Metadata and implementation obligations

The following implementation details do not alter the language proof, but they
must preserve the preconditions of later compiler stages.

1. **Ignore terminal.** The pass never splits `grammar.ignore_terminal`, so
   ignore semantics are unchanged.
2. **Fresh IDs.** New terminal IDs are appended after the original terminal
   vector; new nonterminal IDs begin at `grammar.num_nonterminals()`.
3. **`Shared`.** Wrapping the repeated body in `Expr::Shared` preserves its
   language by definition.
4. **Generated-terminal reuse.** Reusing one terminal ID for structurally equal
   expressions in the same partition is sound because their terminal languages
   are identical.
5. **Lexer partitions.** Partition labels are compilation metadata, not grammar
   operators. Terminal compaction must nevertheless preserve them. The proof
   audit changed compaction so identical terminals are merged only when their
   explicit partition assignments are also equal. This prevents a panic and
   avoids silently discarding partition structure.
6. **Names.** Terminal and nonterminal display names carry no language
   semantics. The generated name prefix is used only to select the conservative
   all-terminal equivalence observer described below.
7. **Arithmetic checker.** `count_cover_is_exact` now checks contiguous intervals
   without allocating `N+1` booleans or computing an unchecked `N+1`.

## 8. Next-token-mask corollary

Let a vocabulary map each token `v` to a word `β(v) ∈ Ω*`; a special token maps
to its special symbol. For a language `L` and committed prefix `u`, define the
ideal next-token mask

```text
M_L(u) = {v : there exists w such that u β(v) w ∈ L}.
```

Define EOS acceptance by

```text
EOS_L(u) ⇔ u ∈ L.
```

**Corollary 9 — exact mask and EOS preservation.** For every prefix `u` and
vocabulary token `v`,

```text
v ∈ M_L(G)(u)  ⇔  v ∈ M_L(Split(G,V,C))(u),
EOS_L(G)(u)    ⇔  EOS_L(Split(G,V,C))(u).
```

**Proof.** Theorem 1 gives equality of the two languages. Substitution of equal
sets into the two definitions gives both equivalences immediately. ∎

This is stronger than agreement along sampled walks: it quantifies over every
prefix, every token, and every continuation.

## 9. Supporting compiler changes

The terminal-splitting commit also changed three internal optimizations. Their
correctness arguments are separate from the grammar-language theorem.

### 9.1 All-terminal observation for transformed grammars

Let `Obs_O(x)` be the vector of lexer-state observations restricted to terminal
set `O`, and define

```text
x ≡O y  ⇔  Obs_O(x) = Obs_O(y).
```

If `O ⊆ O'`, then

```text
x ≡O' y  ⇒  x ≡O y.
```

Thus adding observers refines the equivalence relation: it can split classes but
cannot merge any pair that the smaller relation distinguished. The transformed
grammar uses all terminal residuals rather than one family-local projection.
This is a conservative refinement. It fixed the discovered case where a future
split fragment caused a 64-underscore token to be merged with 22,034 otherwise
valid tokens.

The failed narrower observer is not part of the implementation.

### 9.2 Reusing the original epsilon tokenizer during classification

The combined tokenizer represents a vector of terminal languages. Restricting
its matched and possible-future terminal sets to a candidate subset computes the
same candidate-language residual observations as constructing a tokenizer from
only that subset. Adding unrelated terminal languages does not change the
Brzozowski residual of any candidate language after a byte prefix; it only adds
additional coordinates to the vector.

`CandidatePrefixPowerset` applies the explicit `terminal_to_local` projection to
both matched and future masks at every interned state. Its downstream result
depends only on those projected observations and byte transitions. Therefore
using the original tokenizer and projecting to candidates is observationally
equivalent to the candidate-only tokenizer for classification. It avoids a
large determinization cliff but does not change the classified candidate set.

### 9.3 Compacting sparse equivalence labels

Suppose worklist blocks have labels `ℓi`, possibly with gaps. The fixed builder
assigns a fresh dense ID `ρ(ℓ)` to each occurring label in first-occurrence
order. `ρ` is injective on occurring labels, so for any blocks `i` and `j`,

```text
ρ(ℓi) = ρ(ℓj)  ⇔  ℓi = ℓj.
```

Consequently the partition of raw states is unchanged; only class names are
renumbered. Member sets are unioned by the same original label, and the first
representative for each nonempty class is retained. Empty holes and their
invalid `u32::MAX` representatives disappear. This is a representation
isomorphism, not a semantic quotient change.

## 10. Proof boundary

This document proves the new pass preserves the mathematical `GrammarDef`
language and therefore the specified mask language. It also proves the local
semantic equivalence of the supporting projection/refinement/renaming changes.

It does **not** constitute a machine-checked proof of every pre-existing GLRMask
compiler stage, the Rust compiler, or the runtime. An unconditional theorem
about emitted machine code would require formalizing the complete baseline
compiler. The standard and meaningful compiler-pass claim is therefore:

```text
If the baseline compiler correctly implements GrammarDef semantics,
then compiling the transformed grammar produces exactly the same masks and EOS
semantics as compiling the original grammar.
```

The implementation is additionally guarded by focused DFA equivalence tests,
systematic small-parameter tests, the complete Rust suite, full-vocabulary
static/dynamic differential walks, and CFA native/dynamic comparisons. Those
tests are evidence that the code realizes the proved transformation; they are
not substitutes for the proof above.
