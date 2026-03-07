//! GLR grammar with numeric IDs, FIRST and FOLLOW sets.
//!
//! Converts a [`GrammarDef`] into a canonical form with an augmented start
//! rule (S' → S), computes nullable nonterminals, FIRST and FOLLOW sets
//! for SLR(1) table construction.
//!
//! # Acyclicity invariant
//!
//! The compiled parser NWA/DWA must be acyclic.  Per Aycock et al. "Even Faster
//! Generalized LR Parsing", for an ε-free CFG without right recursion and without
//! hidden left recursion, consecutive reductions between shifts are bounded by a
//! constant — so the NWA/DWA is acyclic.  Direct left recursion (A → A α) is
//! explicitly safe and does NOT need elimination.
//!
//! [`normalize_for_mask`] applies two transformations before building any table:
//! 1. **Epsilon elimination** via [`inline_epsilon_rules`] — removes ε-productions.
//! 2. **Right recursion elimination** via [`eliminate_right_recursion`] — removes all
//!    direct and indirect right recursion (and hidden left recursion) that would
//!    otherwise create unbounded reduction chains and NWA back-edges.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeSet, BTreeMap};

use crate::compiler::grammar::ast::{GrammarDef, NonterminalId, Rule, Symbol, TerminalID};

/// EOF pseudo-terminal. Must not collide with any real terminal.
pub const EOF: TerminalID = u32::MAX;

/// An augmented GLR grammar ready for table generation.
#[derive(Debug, Clone)]
pub struct GLRGrammar {
    /// All production rules (augmented start is at index 0).
    pub rules: Vec<Rule>,
    #[allow(dead_code)]
    /// Start nonterminal of the augmented grammar.
    pub start: NonterminalId,
    /// Number of real terminals (not counting EOF).
    pub num_terminals: u32,
    /// Number of nonterminals (including augmented start).
    pub num_nonterminals: u32,
    /// Nullable nonterminals.
    pub nullable: BTreeSet<NonterminalId>,
    /// FIRST(A) for each nonterminal A (indexed by NT id).
    pub first: Vec<BTreeSet<TerminalID>>,
    /// FOLLOW(A) for each nonterminal A (indexed by NT id).
    pub follow: Vec<BTreeSet<TerminalID>>,
}

impl GLRGrammar {
    /// Build from a [`GrammarDef`].
    ///
    /// Augments the grammar with a new start rule `S' → S`.
    ///
    /// # Acyclicity contract
    ///
    /// # Acyclicity
    ///
    /// To guarantee an acyclic DWA, call [`normalize_for_mask`] on the
    /// `GrammarDef` BEFORE passing it to this function.  The compilation
    /// pipeline in `pipeline.rs` does this automatically.
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        unimplemented!()
    }

    /// FIRST set for a sequence of symbols.
    pub fn first_of_seq(&self, seq: &[Symbol]) -> BTreeSet<TerminalID> {
        unimplemented!()
    }

    /// Whether a symbol sequence can derive ε.
    pub fn seq_is_nullable(&self, seq: &[Symbol]) -> bool {
        unimplemented!()
    }
}

// ---------------------------------------------------------------------------
// Grammar normalization for mask computation
// ---------------------------------------------------------------------------

/// Normalize a [`GrammarDef`] for mask computation.
///
/// Returns a new `GrammarDef` with:
/// 1. All epsilon productions inlined away (via [`inline_epsilon_rules`]).
/// 2. All right recursion (and hidden left recursion) eliminated (via [`eliminate_right_recursion`]).
///
/// Per Aycock et al. "Even Faster Generalized LR Parsing", for an ε-free CFG
/// without right recursion and without hidden left recursion, the number of
/// consecutive reductions between shifts of two adjacent symbols is bounded by
/// a constant.  This boundedness prevents cycles in the NWA/DWA.
///
/// **Direct left recursion (A → A α) is explicitly safe** — the paper notes it
/// "requires no such special treatment" and a manual SLR-state analysis confirms
/// that direct-left-recursive rules produce only `nt_escapes` (never `nt_rereduce`
/// self-loops) in `characterize_terminal`.  Do NOT eliminate DLR here.
///
/// The resulting grammar is semantically equivalent (accepts the same language,
/// modulo empty-string cases for epsilon-only grammars) and its SLR parse table
/// produces an acyclic NWA/DWA, which is required for correct mask computation.
///
/// Call this once in the compilation pipeline BEFORE [`GLRGrammar::from_grammar_def`].
pub fn normalize_for_mask(g: &GrammarDef) -> GrammarDef {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Direct left recursion elimination
// ---------------------------------------------------------------------------

/// Eliminate direct left recursion from an epsilon-free rule set.
///
/// **NOTE: This function is NOT used in the normalization pipeline.**
///
/// Per Aycock et al. "Even Faster Generalized LR Parsing", direct left
/// recursion (A → A α) does NOT cause unbounded reduction chains and therefore
/// does NOT cause NWA/DWA cycles.  A manual SLR-state analysis confirms that
/// direct-left-recursive rules produce only `nt_escapes` (never `nt_rereduce`
/// self-loops) in `characterize_terminal`.
///
/// This function is retained for reference only.  Calling it before
/// `eliminate_right_recursion` would incorrectly introduce new right-recursive
/// auxiliaries that `eliminate_right_recursion` would then transform into
/// left-recursive auxiliaries, creating an infinite regress.
///
/// If you need to eliminate DLR for some other reason (e.g. LL parsing), this
/// applies the standard epsilon-free transformation:
///
/// Given:
/// ```text
/// A → A α₁    (direct-left-recursive cases)
/// A → A α₂
/// A → β₁      (non-recursive base cases)
/// A → β₂
/// ```
/// Produces:
/// ```text
/// A  → β₁         A  → β₁ A'
/// A  → β₂         A  → β₂ A'
/// A' → α₁         A' → α₁ A'
/// A' → α₂         A' → α₂ A'
/// ```
#[allow(dead_code)]
pub(crate) fn eliminate_direct_left_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalId,
) {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Right recursion elimination
// ---------------------------------------------------------------------------

/// Eliminate right recursion from an epsilon-free rule set.
///
/// Right recursion creates cycles in the NWA (e.g. A → … B … A), which
/// make the determinized DWA cyclic and break minimization.  This function
/// converts indirect right recursion into left recursion (which is fine for
/// LR parsing) viaa two-step process:
///
/// 1. Inline indirect right-recursive edges until only direct (self-loop)
///    right recursion remains.
/// 2. Eliminate direct right recursion with an epsilon-free transformation
///    that introduces a fresh left-recursive auxiliary NT.
///
/// `fresh_nt()` must return a fresh `NonterminalId` each time it is called.
/// All returned IDs must be distinct and greater than any ID already present
/// in `rules`.
pub(crate) fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalId,
) {
    unimplemented!()
}

/// Return the highest NT id present in `rules` (0 if empty).
fn max_nt_id(rules: &[Rule]) -> u32 {
    unimplemented!()
}

/// Build the right-reachability graph: A → B if A has a production where B
/// is the rightmost NT (i.e. B appears at position i in the RHS and every
/// symbol after position i is a nullable NT).
///
/// Self-loops (A → A) represent direct right recursion and ARE included.
fn build_right_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalId>,
) -> BTreeMap<NonterminalId, BTreeSet<NonterminalId>> {
    unimplemented!()
}

/// Find an indirect right-recursion cycle (length > 1, no self-loops).
///
/// Returns the cycle as a Vec where `cycle[i]` → `cycle[(i+1) % len]`.
fn find_indirect_rr_cycle(
    graph: &BTreeMap<NonterminalId, BTreeSet<NonterminalId>>,
) -> Option<Vec<NonterminalId>> {
    unimplemented!()
}

/// Inline `to_nt`'s productions into every rule of `from_nt` that ends with `to_nt`
/// (considering nullable suffix).
///
/// The original `from_nt → … to_nt (nullable)*` rule is REPLACED by the
/// expanded variants.  Rules for `to_nt` itself are unchanged.
fn inline_right_end(
    rules: &mut Vec<Rule>,
    from_nt: NonterminalId,
    to_nt: NonterminalId,
    nullable: &BTreeSet<NonterminalId>,
) {
    unimplemented!()
}

/// Returns `true` if `rule` has direct right recursion (its last RHS symbol
/// is the NT equal to `rule.lhs`).
fn is_direct_right_recursive(rule: &Rule) -> bool {
    unimplemented!()
}

/// Eliminate direct right recursion for a single NT `nt` using the
/// epsilon-free transformation.
///
/// Given:
/// ```text
/// A → α₁ A  (recursive, prefix = α₁)
/// A → β₁    (non-recursive base case)
/// ```
/// Produces:
/// ```text
/// A   → β₁          (base cases unchanged)
/// A   → A' β₁       (left-recursive case: one or more prefixes before base)
/// A'  → α₁          (A' base: just the prefix)
/// A'  → A' α₁       (A' recursive: left-recursive accumulation)
/// ```
/// This converts right recursion into left recursion without epsilon.
fn resolve_direct_rr_single_nt(
    rules: &mut Vec<Rule>,
    nt: NonterminalId,
    new_nt: NonterminalId,
) {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Epsilon (null) rule inlining
// ---------------------------------------------------------------------------

/// Eliminate all epsilon productions from `rules` by inlining them.
///
/// For each rule `A → α B β` where `B` is nullable (can produce ε), produce
/// two variants: one with `B` and one without.  Duplicate variants and the
/// original epsilon productions are dropped.
///
/// The result contains no rules with an empty RHS.
///
/// # Complexity
///
/// O(R × 2^K) where R is the number of rules and K is the maximum number of
/// nullable nonterminals in any single rule's RHS.  For grammars produced by
/// normal EBNF with `?`, `*`, `+` operators the worst case is small in practice.
///
/// # Panics
///
/// Panics if the result still contains epsilon productions (indicates a bug).
pub(crate) fn inline_epsilon_rules(rules: &[Rule]) -> Vec<Rule> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Nullable (fixed-point)
// ---------------------------------------------------------------------------

fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalId> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// FIRST sets (fixed-point)
// ---------------------------------------------------------------------------

fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    nullable: &BTreeSet<NonterminalId>,
) -> Vec<BTreeSet<TerminalID>> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// FOLLOW sets (fixed-point)
// ---------------------------------------------------------------------------

fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    start: NonterminalId,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalId>,
) -> Vec<BTreeSet<TerminalID>> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::ast::tests::*;

    #[test]
    fn test_glr_grammar_simple() {
        let g = GLRGrammar::from_grammar_def(&simple_ab_grammar());
        // Augmented: S' → S;  S → a b
        assert_eq!(g.rules.len(), 2);
        assert_eq!(g.num_nonterminals, 2); // S, S'
        assert_eq!(g.num_terminals, 2);
        assert!(g.nullable.is_empty());
        // FIRST(S) = {a}
        assert!(g.first[0].contains(&0));
        assert!(!g.first[0].contains(&1));
        // FOLLOW(S) = {$}
        assert!(g.follow[0].contains(&EOF));
    }

    #[test]
    fn test_glr_grammar_choice() {
        let g = GLRGrammar::from_grammar_def(&choice_grammar());
        // S → a | b  →  FIRST(S) = {a, b}
        assert!(g.first[0].contains(&0));
        assert!(g.first[0].contains(&1));
    }

    #[test]
    fn test_glr_grammar_two_nt() {
        let g = GLRGrammar::from_grammar_def(&two_nt_grammar());
        // S → A b, A → a.  FIRST(A) = {a}, FIRST(S) = FIRST(A) = {a}
        assert!(g.first[0].contains(&0)); // FIRST(S) has 'a'
        assert!(g.first[1].contains(&0)); // FIRST(A) has 'a'
        // FOLLOW(A) = FIRST(b...) = {b}
        assert!(g.follow[1].contains(&1)); // 'b' follows A
    }
}
