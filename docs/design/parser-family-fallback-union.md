> Design notes from development. Type and option names may be out of date; [GUIDE.md](../../GUIDE.md) describes the current API.

# Parser-family union and fallback transitions

Parser DWAs interpret `DEFAULT_LABEL` as a row-local fallback: an explicit
parser-state label wins when present, even if its weight is empty. This is
not the ordinary alphabet semantics used by a generic NWA union.

Consequently, source-local transition selection must precede union. Suppose
one family follows a default edge carrying token A, while another follows an
explicit edge carrying token B for the same parser-state label. A raw union
contains both an explicit edge and a default edge. The runtime selects the
explicit edge and loses A. The correct merged edge carries both contributions,
with each source's destination and residual weight retained separately.

The compiler's fallback-aware deterministic product selects each operand's
explicit edge or its own fallback, intersects that selected weight with the
operand's residual, and only then combines the contributions. This preserves
correlation along later edges. Singleton-state reuse keeps the incoming weight
on the merged edge. Explicit rejection barriers use nonaccepting sink states
with nonempty edges so that subsequent empty-weight cleanup cannot accidentally
expose a default. The merge continues to use its existing raw-alphabet paths
when no input contains fallback transitions. Default-labelled immediate
families do not enter the explicit-key top-accept overlay optimization.

No parser, lexer, vocabulary-equivalence relation, runtime masking algorithm,
or public optimization mode is replaced. In particular, Static and Balanced
retain their native parser representation; a dynamic-reference mask is used
only in tests, not as a production fallback or postfilter.

## Reproduction and regression coverage

The integration fixture in
`tests/data/parser_family_fallback/reduced_sql.lark` was reduced from the SQL
benchmark while preserving the failing parser state. After `SELECT u.name`,
a comma is legal. With a vocabulary containing only `,`, the old Static mask
was correct. Adding the unrelated token ` =` introduced a second parser DWA
family and incorrectly removed `,` from that mask. Two vocabulary entries
therefore suffice to reproduce the issue; the original full-vocabulary SQL
fixtures are not changed by this reduction.

`cargo test --release --locked -p glrmask-dwa-merge` checks the reported
source-fallback case and independently compares the result with the union of
the original operands, including overlapping weights, zero-weight barriers,
cycles, more than two families, and singleton-normalization boundaries.

`cargo test --release --locked --features internal-api --test parser_family_fallback_union`
checks the public compilation and state API in FastBuild, Balanced, and
FastRuntime modes with several vocabulary extensions, both before and after
serialization. It verifies complete mask agreement with the same-state native
reference and then commits the legal comma.

Previously compiled artifacts contain previously compiled automata. Recompile
cached constraints with the corrected compiler; loading an old artifact alone
does not repair its stored automaton. The artifact encoding is unchanged.
