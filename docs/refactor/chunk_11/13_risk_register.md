# Risk register

## Risk: private-field visibility after DenseMaskAcc extraction

`runtime/mask/mod.rs` still constructs cache keys and reads dense accumulator
entries.  The extracted structs therefore expose fields as `pub(super)`, not
`pub`.  If compile repair reports privacy errors, prefer narrowing the public
helper methods over widening visibility outside `runtime/mask`.

## Risk: unused helper after bitset extraction

`for_each_set_token_bit` was moved with the bitset group even if not currently
used by the parent module.  Because the crate currently allows dead code, this
is acceptable for now.  A later warning-cleanup chunk may remove it or add a
focused diagnostic use.

## Risk: Commit fast paths remain tangled

This chunk does not split the body of `commit/mod.rs` by fast path.  That is
intentional.  The file still has several thousand lines and should be addressed
in a later chunk with tests nearby.

## Risk: profile terminology still says `may_advance`

Internal exact predicates now use `can_advance`, but profile field names are
left unchanged.  Renaming profile fields may break benchmark parsers or Python
bindings, so it should be a deliberate public diagnostics migration.

## Risk: Force still knows about token bytes

That is acceptable because Force is derived.  It should not leak into core Mask
or Commit definitions.

## Risk: environment variables remain in other runtime areas

This chunk only localizes Commit template-DFA env reads.  Mask profiling env
reads are already mostly in `profile.rs`; other runtime env reads should be
audited in a later global configuration chunk.
