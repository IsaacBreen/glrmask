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

use std::collections::{BTreeSet, BTreeMap};

use super::super::grammar_def::{GrammarDef, NonterminalId, Rule, Symbol, TerminalId};

/// EOF pseudo-terminal. Must not collide with any real terminal.
pub const EOF: TerminalId = u32::MAX;

/// An augmented GLR grammar ready for table generation.
#[derive(Debug, Clone)]
pub struct GlrGrammar {
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
    pub first: Vec<BTreeSet<TerminalId>>,
    /// FOLLOW(A) for each nonterminal A (indexed by NT id).
    pub follow: Vec<BTreeSet<TerminalId>>,
}

impl GlrGrammar {
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
        let orig_nt_count = g.num_nonterminals();
        let num_terminals = g.num_terminals();

        // Augmented start NT = one past the highest original NT id.
        let aug_start = orig_nt_count;
        let num_nonterminals = orig_nt_count + 1;

        // Augmented rule: S' → S   (index 0)
        let mut rules = vec![Rule {
            lhs: aug_start,
            rhs: vec![Symbol::Nonterminal(g.start)],
        }];
        rules.extend_from_slice(&g.rules);

        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, &nullable);
        let follow = compute_follow(&rules, num_nonterminals, aug_start, &first, &nullable);

        GlrGrammar {
            rules,
            start: aug_start,
            num_terminals,
            num_nonterminals,
            nullable,
            first,
            follow,
        }
    }

    /// FIRST set for a sequence of symbols.
    pub fn first_of_seq(&self, seq: &[Symbol]) -> BTreeSet<TerminalId> {
        let mut result = BTreeSet::new();
        let n = self.num_nonterminals as usize;
        for sym in seq {
            match sym {
                Symbol::Terminal(t) => {
                    result.insert(*t);
                    return result;
                }
                Symbol::Nonterminal(nt) => {
                    if (*nt as usize) < n {
                        result.extend(&self.first[*nt as usize]);
                    }
                    if !self.nullable.contains(nt) {
                        return result;
                    }
                }
            }
        }
        result
    }

    /// Whether a symbol sequence can derive ε.
    pub fn seq_is_nullable(&self, seq: &[Symbol]) -> bool {
        seq.iter().all(|s| match s {
            Symbol::Terminal(_) => false,
            Symbol::Nonterminal(nt) => self.nullable.contains(nt),
        })
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
/// Call this once in the compilation pipeline BEFORE [`GlrGrammar::from_grammar_def`].
pub fn normalize_for_mask(g: &GrammarDef) -> GrammarDef {
    // Step 1: Inline epsilon rules (prevents epsilon-cycle NWA).
    let mut rules = inline_epsilon_rules(&g.rules);

    let mut next_fresh = max_nt_id(&rules) + 1;
    let mut fresh_nt = || {
        let id = next_fresh;
        next_fresh += 1;
        id
    };

    // Step 2: Eliminate right recursion + hidden left recursion.
    // Direct left recursion is safe (per Aycock et al.) and must NOT be eliminated.
    eliminate_right_recursion(&mut rules, &mut fresh_nt);

    GrammarDef {
        rules,
        start: g.start,
        terminals: g.terminals.clone(),
    }
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
    // Collect all NTs that have at least one directly left-recursive rule.
    let directly_lr_nts: Vec<NonterminalId> = {
        let mut nts: BTreeSet<NonterminalId> = BTreeSet::new();
        for r in rules.iter() {
            if matches!(r.rhs.first(), Some(Symbol::Nonterminal(nt)) if *nt == r.lhs) {
                nts.insert(r.lhs);
            }
        }
        nts.into_iter().collect()
    };

    if directly_lr_nts.is_empty() {
        return;
    }

    let mut new_rules: Vec<Rule> = rules.iter().filter(|r| {
        !directly_lr_nts.contains(&r.lhs)
    }).cloned().collect();

    for nt in directly_lr_nts {
        let prods_for_nt: Vec<Rule> = rules.iter().filter(|r| r.lhs == nt).cloned().collect();
        let (recursive, non_recursive): (Vec<_>, Vec<_>) = prods_for_nt
            .iter()
            .cloned()
            .partition(|r| matches!(r.rhs.first(), Some(Symbol::Nonterminal(n)) if *n == nt));

        if recursive.is_empty() || non_recursive.is_empty() {
            // Can't transform without both parts; keep as-is.
            new_rules.extend(prods_for_nt);
            continue;
        }

        let new_nt = fresh_nt();

        // A → βⱼ  (base cases, unchanged)
        for rule in &non_recursive {
            new_rules.push(rule.clone());
        }
        // A → βⱼ A'  (base case + optional suffix chain)
        for rule in &non_recursive {
            let mut rhs = rule.rhs.clone();
            rhs.push(Symbol::Nonterminal(new_nt));
            new_rules.push(Rule { lhs: nt, rhs });
        }
        // A' → αᵢ  (suffix alone = A' base case)
        for rule in &recursive {
            // suffix = rhs[1..] (everything after the leading self-ref)
            let suffix = rule.rhs[1..].to_vec();
            if suffix.is_empty() {
                // A → A with no suffix would mean A = A, skip.
                continue;
            }
            new_rules.push(Rule { lhs: new_nt, rhs: suffix });
        }
        // A' → αᵢ A'  (suffix + recursion = right-recursive)
        for rule in &recursive {
            let suffix = rule.rhs[1..].to_vec();
            if suffix.is_empty() {
                continue;
            }
            let mut rhs = suffix;
            rhs.push(Symbol::Nonterminal(new_nt));
            new_rules.push(Rule { lhs: new_nt, rhs });
        }
    }

    // Deduplicate while preserving order.
    let mut seen: BTreeSet<(NonterminalId, Vec<Symbol>)> = BTreeSet::new();
    *rules = new_rules.into_iter().filter(|r| seen.insert((r.lhs, r.rhs.clone()))).collect();
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
    // Phase 1: eliminate indirect right recursion by inlining until convergence.
    for _ in 0..200 {
        let nullable = compute_nullable(rules, max_nt_id(rules) + 1);
        let graph = build_right_reachability_graph(rules, &nullable);
        let cycle = find_indirect_rr_cycle(&graph);
        let Some(cycle) = cycle else { break };

        // Pick the edge (from → to) to inline.
        // Prefer the edge where `to` has NO direct right recursion
        // (so inlining won't immediately re-introduce it).
        let safe_idx = (0..cycle.len()).find(|&i| {
            let to = cycle[(i + 1) % cycle.len()];
            !rules.iter().any(|r| r.lhs == to && is_direct_right_recursive(r))
        });
        let edge_idx = safe_idx.unwrap_or(0);
        let from = cycle[edge_idx];
        let to = cycle[(edge_idx + 1) % cycle.len()];
        inline_right_end(rules, from, to, &nullable);
    }

    // Phase 2: eliminate any remaining direct right recursion.
    let mut i = 0;
    while i < rules.len() {
        let nt = rules[i].lhs;
        let has_direct_rr = rules.iter().any(|r| r.lhs == nt && is_direct_right_recursive(r));
        if has_direct_rr {
            let new_nt = fresh_nt();
            resolve_direct_rr_single_nt(rules, nt, new_nt);
            // Restart from beginning since the rule list changed.
            i = 0;
        } else {
            i += 1;
        }
    }
}

/// Return the highest NT id present in `rules` (0 if empty).
fn max_nt_id(rules: &[Rule]) -> u32 {
    let mut max = 0u32;
    for r in rules {
        max = max.max(r.lhs);
        for s in &r.rhs {
            if let Symbol::Nonterminal(nt) = s {
                max = max.max(*nt);
            }
        }
    }
    max
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
    let mut graph: BTreeMap<NonterminalId, BTreeSet<NonterminalId>> = BTreeMap::new();
    for r in rules {
        for i in (0..r.rhs.len()).rev() {
            if let Symbol::Nonterminal(nt) = &r.rhs[i] {
                // Check whether the suffix r.rhs[i+1..] is all nullable.
                let suffix_nullable = r.rhs[i + 1..].iter().all(|s| {
                    matches!(s, Symbol::Nonterminal(n) if nullable.contains(n))
                });
                if suffix_nullable {
                    graph.entry(r.lhs).or_default().insert(*nt);
                }
                // Stop scanning right-to-left once we hit a non-nullable NT.
                if !nullable.contains(nt) {
                    break;
                }
            } else {
                // Terminal: not a right-reachable NT.
                break;
            }
        }
    }
    graph
}

/// Find an indirect right-recursion cycle (length > 1, no self-loops).
///
/// Returns the cycle as a Vec where `cycle[i]` → `cycle[(i+1) % len]`.
fn find_indirect_rr_cycle(
    graph: &BTreeMap<NonterminalId, BTreeSet<NonterminalId>>,
) -> Option<Vec<NonterminalId>> {
    let mut visited = BTreeSet::new();
    let mut in_stack = BTreeSet::new();
    let mut path = Vec::new();

    fn dfs(
        node: NonterminalId,
        graph: &BTreeMap<NonterminalId, BTreeSet<NonterminalId>>,
        visited: &mut BTreeSet<NonterminalId>,
        in_stack: &mut BTreeSet<NonterminalId>,
        path: &mut Vec<NonterminalId>,
    ) -> Option<Vec<NonterminalId>> {
        visited.insert(node);
        in_stack.insert(node);
        path.push(node);

        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if neighbor == node {
                    // Skip self-loops; they are direct RR handled separately.
                    continue;
                }
                if in_stack.contains(&neighbor) {
                    let start = path.iter().position(|&n| n == neighbor).unwrap();
                    return Some(path[start..].to_vec());
                }
                if !visited.contains(&neighbor) {
                    if let Some(cycle) = dfs(neighbor, graph, visited, in_stack, path) {
                        return Some(cycle);
                    }
                }
            }
        }
        path.pop();
        in_stack.remove(&node);
        None
    }

    for &node in graph.keys() {
        if !visited.contains(&node) {
            if let Some(cycle) = dfs(node, graph, &mut visited, &mut in_stack, &mut path) {
                return Some(cycle);
            }
        }
    }
    None
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
    let to_prods: Vec<Rule> = rules.iter().filter(|r| r.lhs == to_nt).cloned().collect();
    if to_prods.is_empty() {
        return;
    }

    let mut new_rules: Vec<Rule> = Vec::new();
    for rule in rules.iter() {
        if rule.lhs != from_nt {
            new_rules.push(rule.clone());
            continue;
        }
        // Find the rightmost position of `to_nt` with a nullable suffix.
        let pos = (0..rule.rhs.len()).rev().find(|&i| {
            matches!(&rule.rhs[i], Symbol::Nonterminal(nt) if *nt == to_nt)
                && rule.rhs[i + 1..].iter().all(|s| {
                    matches!(s, Symbol::Nonterminal(n) if nullable.contains(n))
                })
        });
        if let Some(pos) = pos {
            for to_prod in &to_prods {
                let mut rhs = rule.rhs[..pos].to_vec();
                rhs.extend_from_slice(&to_prod.rhs);
                rhs.extend_from_slice(&rule.rhs[pos + 1..]);
                // Skip unit self-loops: A → A
                if rhs.len() == 1 && matches!(&rhs[0], Symbol::Nonterminal(n) if *n == from_nt) {
                    continue;
                }
                new_rules.push(Rule { lhs: from_nt, rhs });
            }
        } else {
            new_rules.push(rule.clone());
        }
    }
    // Deduplicate while preserving order.
    let mut seen: BTreeMap<(NonterminalId, Vec<Symbol>), ()> = BTreeMap::new();
    *rules = new_rules.into_iter().filter(|r| seen.insert((r.lhs, r.rhs.clone()), ()).is_none()).collect();
}

/// Returns `true` if `rule` has direct right recursion (its last RHS symbol
/// is the NT equal to `rule.lhs`).
fn is_direct_right_recursive(rule: &Rule) -> bool {
    matches!(rule.rhs.last(), Some(Symbol::Nonterminal(nt)) if *nt == rule.lhs)
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
    let prods_for_nt: Vec<Rule> = rules.iter().filter(|r| r.lhs == nt).cloned().collect();
    let (recursive, non_recursive): (Vec<_>, Vec<_>) =
        prods_for_nt.iter().cloned().partition(|r| is_direct_right_recursive(r));

    if recursive.is_empty() || non_recursive.is_empty() {
        // Can't apply the transformation without both parts.
        return;
    }

    let mut new_rules: Vec<Rule> = rules.iter().filter(|r| r.lhs != nt).cloned().collect();

    // A → β  (keep non-recursive base cases)
    for rule in &non_recursive {
        new_rules.push(rule.clone());
    }
    // A → A' β  (left-recursive: prepend the new auxiliary NT)
    for rule in &non_recursive {
        let mut rhs = vec![Symbol::Nonterminal(new_nt)];
        rhs.extend_from_slice(&rule.rhs);
        new_rules.push(Rule { lhs: nt, rhs });
    }
    // A' → α  (base case for auxiliary: just the recursive prefix)
    for rule in &recursive {
        let prefix = rule.rhs[..rule.rhs.len() - 1].to_vec();
        new_rules.push(Rule { lhs: new_nt, rhs: prefix });
    }
    // A' → A' α  (recursive case for auxiliary: left-recursive)
    for rule in &recursive {
        let prefix = rule.rhs[..rule.rhs.len() - 1].to_vec();
        let mut rhs = vec![Symbol::Nonterminal(new_nt)];
        rhs.extend_from_slice(&prefix);
        new_rules.push(Rule { lhs: new_nt, rhs });
    }

    *rules = new_rules;
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
    // Quick exit: if no rule has an empty RHS, nothing to do.
    if rules.iter().all(|r| !r.rhs.is_empty()) {
        return rules.to_vec();
    }

    // Compute the set of nullable NTs (those that can produce ε).
    let num_nt = rules.iter()
        .flat_map(|r| {
            let lhs_iter = std::iter::once(r.lhs + 1);
            let rhs_iter = r.rhs.iter().filter_map(|s| {
                if let Symbol::Nonterminal(nt) = s { Some(nt + 1) } else { None }
            });
            lhs_iter.chain(rhs_iter).collect::<Vec<_>>().into_iter()
        })
        .max()
        .unwrap_or(0);
    let nullable = compute_nullable(rules, num_nt);

    if nullable.is_empty() {
        // No nullable NTs means no ε rules can cascade → return as-is.
        return rules.to_vec();
    }

    let mut seen: BTreeMap<(NonterminalId, Vec<Symbol>), ()> = BTreeMap::new();
    let mut result: Vec<Rule> = Vec::new();

    for rule in rules {
        // Collect positions in the RHS that refer to nullable NTs.
        let nullable_positions: Vec<usize> = rule.rhs
            .iter()
            .enumerate()
            .filter_map(|(i, sym)| {
                if let Symbol::Nonterminal(nt) = sym {
                    if nullable.contains(nt) { Some(i) } else { None }
                } else {
                    None
                }
            })
            .collect();

        let n = nullable_positions.len();

        // Guard against exponential blowup (should not happen in practice).
        // Rules with > 20 nullable NTs are extremely unlikely; 2^20 ≈ 1M variants.
        assert!(
            n <= 20,
            "inline_epsilon_rules: rule has {} nullable NTs — too many for exhaustive expansion. \
             Consider refactoring the grammar to reduce optional chains.",
            n
        );

        let num_variants = 1usize << n; // 2^n

        for mask in 0..num_variants {
            // Build the RHS variant: bit i of `mask` = 1 means include nullable_positions[i].
            let new_rhs: Vec<Symbol> = rule
                .rhs
                .iter()
                .enumerate()
                .filter_map(|(i, sym)| {
                    if let Symbol::Nonterminal(nt) = sym {
                        if nullable.contains(nt) {
                            let bit = nullable_positions
                                .iter()
                                .position(|&p| p == i)
                                .unwrap();
                            if (mask >> bit) & 1 == 0 {
                                // Omit this nullable NT in this variant.
                                return None;
                            }
                        }
                    }
                    Some(sym.clone())
                })
                .collect();

            if new_rhs.is_empty() {
                // Never emit new epsilon productions; they are what we're removing.
                continue;
            }

            let key = (rule.lhs, new_rhs.clone());
            if seen.insert(key, ()).is_none() {
                result.push(Rule { lhs: rule.lhs, rhs: new_rhs });
            }
        }
    }

    // Hard invariant: no epsilon productions should remain.
    assert!(
        result.iter().all(|r| !r.rhs.is_empty()),
        "inline_epsilon_rules: epsilon productions remain after inlining — this is a bug"
    );

    result
}

// ---------------------------------------------------------------------------
// Nullable (fixed-point)
// ---------------------------------------------------------------------------

fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalId> {
    let mut nullable = BTreeSet::new();
    loop {
        let mut changed = false;
        for r in rules {
            if r.lhs < num_nt
                && !nullable.contains(&r.lhs)
                && r.rhs.iter().all(|s| match s {
                    Symbol::Terminal(_) => false,
                    Symbol::Nonterminal(nt) => nullable.contains(nt),
                })
            {
                nullable.insert(r.lhs);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    nullable
}

// ---------------------------------------------------------------------------
// FIRST sets (fixed-point)
// ---------------------------------------------------------------------------

fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    nullable: &BTreeSet<NonterminalId>,
) -> Vec<BTreeSet<TerminalId>> {
    let n = num_nt as usize;
    let mut first: Vec<BTreeSet<TerminalId>> = vec![BTreeSet::new(); n];
    loop {
        let mut changed = false;
        for r in rules {
            let lhs = r.lhs as usize;
            if lhs >= n {
                continue;
            }
            for sym in &r.rhs {
                match sym {
                    Symbol::Terminal(t) => {
                        if first[lhs].insert(*t) {
                            changed = true;
                        }
                        break;
                    }
                    Symbol::Nonterminal(nt) => {
                        let nt_idx = *nt as usize;
                        if nt_idx < n {
                            let ext: Vec<TerminalId> = first[nt_idx].iter().copied().collect();
                            for t in ext {
                                if first[lhs].insert(t) {
                                    changed = true;
                                }
                            }
                        }
                        if !nullable.contains(nt) {
                            break;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    first
}

// ---------------------------------------------------------------------------
// FOLLOW sets (fixed-point)
// ---------------------------------------------------------------------------

fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    start: NonterminalId,
    first: &[BTreeSet<TerminalId>],
    nullable: &BTreeSet<NonterminalId>,
) -> Vec<BTreeSet<TerminalId>> {
    let n = num_nt as usize;
    let mut follow: Vec<BTreeSet<TerminalId>> = vec![BTreeSet::new(); n];

    // $ ∈ FOLLOW(S').
    if (start as usize) < n {
        follow[start as usize].insert(EOF);
    }

    loop {
        let mut changed = false;
        for r in rules {
            let lhs = r.lhs as usize;
            for (i, sym) in r.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nt) = sym else {
                    continue;
                };
                let nt_idx = *nt as usize;
                if nt_idx >= n {
                    continue;
                }

                // β = rhs[i+1 ..]
                let beta = &r.rhs[i + 1..];
                let mut beta_nullable = true;
                for s in beta {
                    match s {
                        Symbol::Terminal(t) => {
                            if follow[nt_idx].insert(*t) {
                                changed = true;
                            }
                            beta_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(b) => {
                            let b_idx = *b as usize;
                            if b_idx < n {
                                let ext: Vec<TerminalId> = first[b_idx].iter().copied().collect();
                                for t in ext {
                                    if follow[nt_idx].insert(t) {
                                        changed = true;
                                    }
                                }
                            }
                            if !nullable.contains(b) {
                                beta_nullable = false;
                                break;
                            }
                        }
                    }
                }

                // If β is empty or entirely nullable → FOLLOW(nt) ∪= FOLLOW(lhs)
                if beta_nullable && lhs < n {
                    let ext: Vec<TerminalId> = follow[lhs].iter().copied().collect();
                    for t in ext {
                        if follow[nt_idx].insert(t) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    follow
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::tests::*;

    #[test]
    fn test_glr_grammar_simple() {
        let g = GlrGrammar::from_grammar_def(&simple_ab_grammar());
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
        let g = GlrGrammar::from_grammar_def(&choice_grammar());
        // S → a | b  →  FIRST(S) = {a, b}
        assert!(g.first[0].contains(&0));
        assert!(g.first[0].contains(&1));
    }

    #[test]
    fn test_glr_grammar_two_nt() {
        let g = GlrGrammar::from_grammar_def(&two_nt_grammar());
        // S → A b, A → a.  FIRST(A) = {a}, FIRST(S) = FIRST(A) = {a}
        assert!(g.first[0].contains(&0)); // FIRST(S) has 'a'
        assert!(g.first[1].contains(&0)); // FIRST(A) has 'a'
        // FOLLOW(A) = FIRST(b...) = {b}
        assert!(g.follow[1].contains(&1)); // 'b' follows A
    }
}
