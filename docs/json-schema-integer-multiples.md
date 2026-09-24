# Exact integer multiples

Integer `multipleOf` must constrain the generated language. The importer must
not silently replace a divisor it cannot represent with an unrestricted integer
or a range alone.

## Construction

The existing compact power-of-ten regular expressions and finite-range
enumeration remain the first choices. For other positive integral divisors up
to 4,096, the importer constructs a decimal remainder automaton. Its state
records the absolute value of the consumed nonzero-leading decimal prefix
modulo the divisor. Reading digit `d` from remainder `r` gives
`(10 * r + d) % divisor`; only remainder zero accepts.

Separate start, minus-sign, and zero states preserve the existing canonical
integer spelling: an optional minus followed by either a single zero or a
nonzero digit and any number of additional digits. In particular, a lone minus,
a plus sign, and leading zeroes do not become valid. The automaton has no limit
on input digit count and never converts the full input to a machine integer.
Range constraints are intersected with the divisibility language rather than
substituted for it.

The induction invariant is exact: after each digit, the stored remainder equals
the consumed magnitude modulo the divisor. Sign does not affect whether that
remainder is zero. This proves both exclusion of nonmultiples and inclusion of
all multiples in the existing integer lexical language.

## Resource boundary and compatibility

The generic construction uses at most `divisor + 3` states and about
`10 * divisor` byte transitions. The 4,096-state remainder budget is a bound on
compiler resources, not a limit on values or input length. Larger divisors can
still use the existing compact power-of-ten or finite-range constructions.
When none applies, compilation reports an explicit unsupported-schema error.
It does not drop the constraint or construct a partial automaton.

This removes a former broadening compromise. For example, an integer constrained
to multiples of three must not accept seven; a nonnegative integer constrained
to multiples of twelve must not accept twenty-five. Consumers that previously
relied on those invalid values being accepted must correct their inputs or
schema. There is no opt-in flag for enforcement.

This change does not broaden the importer's integer spelling policy to decimal
or exponent aliases such as `3.0` or `3e0`, nor does it change noninteger
`number` schemas, format policy, or LLguidance compatibility settings. It is
not a claim that every JSON Schema feature or spelling is implemented.

## Validation

Regression tests first reproduce the old constraint loss. The new automata are
compared against arithmetic for 548,931 signed value/divisor pairs, including
divisors that share factors with ten and divisors coprime to ten. Additional
tests cover malformed spellings, negative zero, and 2,048-digit prefixes and
exact multiples. Public API tests exercise complete-token masks under Auto,
FastBuild, and FastRuntime, and saved/loaded constraints. A large-divisor test
distinguishes exact finite-bound support from explicit unsupported unbounded
construction.
