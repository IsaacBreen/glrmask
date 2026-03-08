// SEP1_MAP: `runtime/actions` is a glrmask-only split. sep1 keeps the nearest
// equivalents inline on `GrammarConstraintState` across
// `grammars2024/src/constraint.rs` and `grammars2024/src/constraint_fns.rs`.
pub(crate) mod commit;
pub(crate) mod force;
pub(crate) mod mask;
