# Exact compression of component-local parser-DWA defaults

## 1. Setting

Let the composed LR table have concrete parser-state set

\[
G = \{0,\ldots,n-1\}.
\]

The parent is component `0`; separately compiled children are components
`1..k`.  Component `i` has local LR-state set `L_i` and the table linker
already computes an exact relation

\[
R_i : L_i \to \mathcal P(G),
\]

where `g in R_i(l)` means that local LR state `l` is represented by composed LR
state `g`.  The relation need not be injective or functional in the reverse
direction: child start/accept states can map to caller states, structural sharing
can identify corresponding states, and one local state can map to more than one
composed state.

A component parser DWA is a weighted deterministic automaton over encoded LR
state labels.  In one DWA row `q`, an explicit positive label `l` overrides the
row's `DEFAULT`; otherwise `DEFAULT`, if present, is the transition for local
state `l`.  Negative labels are push operations and are unrelated to this
optimization.

Historically composition transported a component `DEFAULT` by enumerating every
local LR state without an explicit override and then every state in `R_i(l)`.
This is exact but can be huge: a DWA with hundreds of default-bearing rows and a
child with thousands of LR states creates millions of explicit edges.

## 2. Unambiguous composed-state domains

For a composed state `g`, define its complete component/local preimage

\[
P(g) = \{(i,l) \mid g \in R_i(l)\}.
\]

For child `i > 0`, define

\[
D_i = \{g \in G \mid P(g)=\{(i,l)\}\text{ for some }l\in L_i\}.
\]

Thus `D_i` contains exactly those composed LR states with one and only one
component/local interpretation, and that interpretation belongs to child `i`.
The parent is deliberately not assigned a domain.

### Lemma 1 — uniqueness and disjointness

For every `g in D_i` there is a unique local state `pi_i(g)` with
`g in R_i(pi_i(g))`.  For `i != j`, `D_i` and `D_j` are disjoint.

**Proof.** Both statements follow immediately from `P(g)` being a singleton.
If two local states, or two components, represented `g`, then `|P(g)| >= 2` and
`g` would belong to no child domain. QED.

This definition deliberately excludes all difficult cases rather than proving
special cases about them: caller/return aliases, cross-child sharing,
same-child many-to-one sharing, and any other state with multiple preimages stay
on the old explicit representation.

## 3. Synthetic domain labels

For each selected non-empty `D_i`, choose a fresh nonnegative label `delta_i`
strictly above all concrete composed LR-state labels and below `DEFAULT_LABEL`.
Different child domains receive different labels.

Runtime interpretation of a concrete parser state `g` in a parser-DWA row is:

1. use the explicit concrete label `g`, if present;
2. otherwise, if `g in D_i`, use `delta_i`, if present;
3. otherwise use the ordinary global `DEFAULT`, if present.

This is an input-alphabet quotient only.  The LR stack still contains ordinary
state IDs from `G`; synthetic labels are never pushed onto the parser stack.

## 4. Transport construction

Consider one source DWA row `q` of child `i`, with explicit transition function
`E_q(l)` and optional default transition `d_q`.

Explicit local labels are transported exactly as before: `E_q(l)` is emitted on
every concrete `g in R_i(l)`.

If `d_q` exists, the compressed transport additionally emits one edge

\[
\delta_i \mapsto d_q,
\]

and retains ordinary explicit default materialization for every mapped state
outside `D_i`.  Default edges for states in `D_i` are omitted.

A profitability threshold gates creation of the domain map as a whole. If no
child amortizes that table-sized map, composition uses the old materialization.
Once the map is worthwhile, every child domain with positive predicted savings
is represented symbolically; this changes only representation, never semantics.

## 5. Pointwise equivalence of a transported component row

### Theorem 1

For every child component `i`, every source DWA row `q`, and every concrete
composed parser state `g`, runtime lookup in the compressed row yields exactly
the same target and weight as the historical fully materialized transport.

**Proof.** Split on whether `g` belongs to `D_i`.

### Case A: `g in D_i`

By Lemma 1 there is exactly one local state `l = pi_i(g)` and no other local or
component preimage.

If `q` has an explicit edge `E_q(l)`, transport emits that edge on concrete
label `g`.  Runtime checks concrete labels before the domain fallback, so it
returns `E_q(l)`.  Historical materialization also returns `E_q(l)` because an
explicit edge overrides `DEFAULT`.

If `q` has no explicit edge on `l`, there is no concrete child edge on `g`.
If `d_q` exists, runtime falls through to `delta_i` and returns exactly `d_q`;
historical materialization emitted exactly `d_q` on `g`.  If `d_q` is absent,
both representations have no child transition.

### Case B: `g notin D_i`

The compression omits no historical child-default edge on `g`; defaults are
materialized explicitly exactly as before.  Explicit edges are also unchanged.
Therefore lookup is identical.

The cases exhaust `G`, and targets and weights are copied without alteration.
QED.

This theorem is stronger than a reachable-stack argument: it holds for every
possible sequence of concrete composed LR-state IDs, including impossible
parser stacks.

## 6. Union with parent and boundary parser automata

The final composed parser DWA is the exact weighted union/determinization of
transported component automata and the independently compiled boundary repair.
A boundary/global `DEFAULT` must apply to synthetic labels just as it applies to
all other nonnegative parser-input labels.

The overlap-local determinizer already has this semantics: while processing any
explicit nonnegative label, a source row with no explicit edge on that label
contributes its `DEFAULT`.  Therefore when child `i` contributes `delta_i`, a
boundary `DEFAULT` contributes on the same symbol automatically.

The generic reference/fallback path explicitly expands boundary defaults onto
all synthetic domain labels before generic determinization, giving the same
semantics.

### Theorem 2 — union preservation

For every concrete parser-state word `g_1 ... g_m`, evaluation of the final
compressed weighted parser DWA under the runtime embedding

\[
g \mapsto (g\text{ first},\; \delta_i\text{ on miss if }g\in D_i,\;
DEFAULT\text{ last})
\]

has exactly the same weight as the fully materialized composition.

**Proof.** Theorem 1 gives pointwise equality of every transported child row on
every concrete input `g`.  Parent behavior is unchanged.  Boundary/global
wildcard contributions on `delta_i` are preserved as described above.  Weighted
union and determinization are language preserving, and synthetic domains are
pairwise disjoint, so no contribution can be attributed to the wrong child.
Induction over the concrete input word therefore gives identical active weighted
states after every symbol and hence identical final weight. QED.

## 7. Consequence for constrained decoding

The parser DWA is consumed only through concrete LR states from parser stacks.
The optimization changes only how the transition for such a state is looked up;
it does not alter the LR table, parser stack, tokenizer coordinates, or DWA
weights.  By Theorem 2 every parser-stack evaluation has identical weight.
Therefore every token mask, acceptance result, and commit decision produced from
that parser-DWA semantics is unchanged.

The runtime lookup order is part of the representation invariant and is applied
consistently by the normal mask evaluator, indexed-DAG evaluator, and
single-path/direct fast paths.

## 8. Serialization

Synthetic domain labels are meaningful only together with the map from concrete
parser state to fallback label.  Constraint artifact version 12 stores this map
in the versioned outer artifact envelope.  The new field is skipped in the
inner `Constraint` bincode representation, preserving the byte layout needed to
load version-11 artifacts.  Loading v11 supplies an empty domain map; v12
restores the exact map before runtime cache finalization.

Thus artifact-version compatibility cannot silently reinterpret an old parser
DWA using new domain semantics.

## 9. Fail-closed properties

The optimization declines compression rather than weakening its proof when:

- a composed state has zero or multiple component/local preimages;
- the state belongs to the parent rather than a child;
- synthetic labels would collide with the reserved `DEFAULT_LABEL` range;
- no child reaches the configured profitability threshold needed to amortize
  the table-sized domain map; or
- the optimization is explicitly disabled.

Every declined case uses the historical exact materialization path.
