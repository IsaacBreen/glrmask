//! Alternative strategies for nullable-nonterminal (epsilon) elimination.
//!
//! The standard `inline_null_productions` function enumerates all 2^N subsets of
//! nullable nonterminals in each production.  When a production has many nullable
//! nonterminals in sequence—e.g. `S → a NT_1 NT_2 … NT_33 b`—this creates an
//! exponential number of variants, causing memory and time blowups.
//!
//! This module offers several alternative strategies that **pre-process** a grammar
//! to factor out long nullable runs into compact auxiliary rules, then apply the
//! standard exhaustive elimination on the resulting (smaller) grammar:
//!
//! | Strategy          | How it works                                              | Max nullable NTs in any production after preprocessing |
//! |-------------------|-----------------------------------------------------------|---------------------------------------------------------|
//! | `Exhaustive`      | Current behaviour — no preprocessing                     | unchanged (unbounded)                                   |
//! | `RightChain`      | Replace each nullable run with a right-linear chain rule  | 1 per run (just the chain root)                         |
//! | `LeftChain`       | Same, but left-linear                                     | 1 per run                                               |
//! | `BalancedTree(k)` | Replace each run with a balanced k-ary tree of group rules| ⌈N/k⌉ per run (one per tree-node on the path to root)  |
//!
//! Usage: pick a strategy via the `NULL_INLINE_STRATEGY` environment variable
//! (see [`NullableInliningStrategy::from_env`]) and call [`run_null_inline`].

use std::collections::{BTreeMap, BTreeSet};
use crate::glr::grammar::{NonTerminal, Production, Symbol};
use crate::glr::automaton::{compute_nonterminal_nullability, Nullability};
use crate::glr::analyze::inline_null_productions;

// ---------------------------------------------------------------------------
// Strategy enum
// ---------------------------------------------------------------------------

/// Describes how to eliminate nullable-nonterminal (epsilon) productions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NullableInliningStrategy {
    /// Enumerate all 2^N variants per production (current default).
    /// Safe only when no production contains many consecutive nullable NTs.
    Exhaustive,

    /// Pre-process: replace each consecutive run of nullable NTs with a
    /// new right-linear chain auxiliary rule, then run exhaustive elimination.
    ///
    /// Right-linear: `chain_0 → NT_0 chain_1 | chain_1`, …, `chain_{n-1} → NT_{n-1} | ""`
    ///
    /// The outer production gets at most 1 nullable NT per run, so exhaustive
    /// expansion creates only O(1) variants instead of O(2^N).
    RightChain,

    /// Same as `RightChain` but builds a left-linear chain:
    /// `chain_0 → chain_1 NT_{n-1} | chain_1`, …, `chain_{n-1} → NT_0 | ""`
    ///
    /// Semantically equivalent to `RightChain`; may have different GLR state counts.
    LeftChain,

    /// Pre-process: split each nullable run into chunks of `k` consecutive NTs,
    /// create a "leaf" rule per chunk (all 2^k ordered subsets), then combine
    /// chunks with a single "root" production.  After preprocessing the outer
    /// production has ⌈N/k⌉ nullable NTs per run; choose k large enough that
    /// 2^(N/k) stays manageable.
    ///
    /// Example: k=4 for N=33 → 9 leaf rules (each ≤16 alternatives) + outer
    /// production with 9 nullable NTs → 2^9 = 512 variants (vs 2^33 exhaustive).
    BalancedTree(usize),
}

impl NullableInliningStrategy {
    /// Read the strategy from the `NULL_INLINE_STRATEGY` environment variable.
    ///
    /// Default (when unset): `balanced_tree_2`.  This strategy is strictly safer than
    /// `exhaustive` because it produces identical results on grammars with no long nullable
    /// runs while avoiding exponential blowup on grammars with many consecutive nullable NTs.
    /// It automatically falls back to `exhaustive` when the grammar has pre-existing right
    /// recursion (to avoid creating problematic right-reachability cycles).
    ///
    /// Recognised values:
    /// - `exhaustive`   – all 2^N variants (original behaviour; may cause OOM/timeout)
    /// - `right_chain` / `rightchain`
    /// - `left_chain` / `leftchain`
    /// - `balanced_tree_N` where N is the group size (e.g. `balanced_tree_4`)
    pub fn from_env() -> Self {
        let val = std::env::var("NULL_INLINE_STRATEGY")
            .unwrap_or_else(|_| "balanced_tree_2".to_string());
        let val = val.trim().to_lowercase();
        if val == "exhaustive" {
            return Self::Exhaustive;
        }
        if val == "right_chain" || val == "rightchain" {
            return Self::RightChain;
        }
        if val == "left_chain" || val == "leftchain" {
            return Self::LeftChain;
        }
        if let Some(rest) = val.strip_prefix("balanced_tree_") {
            if let Ok(k) = rest.parse::<usize>() {
                if k >= 1 {
                    return Self::BalancedTree(k);
                }
            }
        }
        if val == "balanced_tree" {
            return Self::BalancedTree(2);
        }
        // Unknown → safe default
        Self::RightChain
    }

    /// Short description for logging.
    pub fn name(&self) -> String {
        match self {
            Self::Exhaustive      => "exhaustive".to_string(),
            Self::RightChain      => "right_chain".to_string(),
            Self::LeftChain       => "left_chain".to_string(),
            Self::BalancedTree(k) => format!("balanced_tree_{k}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run epsilon elimination using the chosen strategy.
///
/// All strategies ultimately call the standard [`inline_null_productions`]
/// (exhaustive expansion) on the (possibly pre-processed) grammar; the
/// strategies differ only in how they rewrite productions *before* that step.
pub fn run_null_inline(
    productions: &[Production],
    strategy: &NullableInliningStrategy,
    new_name_gen: &mut impl FnMut(&str) -> String,
) -> Vec<Production> {
    match strategy {
        NullableInliningStrategy::Exhaustive => {
            inline_null_productions(productions, new_name_gen)
        }
        NullableInliningStrategy::RightChain => {
            let preprocessed = preprocess_runs(productions, new_name_gen, 1, Direction::Right);
            inline_null_productions(&preprocessed, new_name_gen)
        }
        NullableInliningStrategy::LeftChain => {
            let preprocessed = preprocess_runs(productions, new_name_gen, 1, Direction::Left);
            inline_null_productions(&preprocessed, new_name_gen)
        }
        NullableInliningStrategy::BalancedTree(k) => {
            let preprocessed = preprocess_balanced_tree(productions, new_name_gen, *k);
            inline_null_productions(&preprocessed, new_name_gen)
        }
    }
}

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
enum Direction { Left, Right }

/// Find contiguous runs of nullable NTs in `rhs`, returning `(start, end)` pairs
/// (both inclusive) for runs with length > `threshold`.
fn find_nullable_runs(
    rhs: &[Symbol],
    nullability: &BTreeMap<NonTerminal, Nullability>,
    threshold: usize,
) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut run_start: Option<usize> = None;

    for (i, sym) in rhs.iter().enumerate() {
        let is_null = matches!(sym, Symbol::NonTerminal(nt)
            if matches!(nullability.get(nt), Some(Nullability::Nullable) | Some(Nullability::Null)));
        if is_null {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else {
            if let Some(start) = run_start.take() {
                let len = i - start;
                if len > threshold {
                    runs.push((start, i - 1));
                }
            }
        }
    }
    // Handle run ending at the last symbol
    if let Some(start) = run_start {
        let len = rhs.len() - start;
        if len > threshold {
            runs.push((start, rhs.len() - 1));
        }
    }
    runs
}

// ---------------------------------------------------------------------------
// Chain pre-processing (RightChain / LeftChain)
// ---------------------------------------------------------------------------

fn preprocess_runs(
    productions: &[Production],
    new_name_gen: &mut impl FnMut(&str) -> String,
    threshold: usize,
    direction: Direction,
) -> Vec<Production> {
    let nullability = compute_nonterminal_nullability(productions);
    let mut result: Vec<Production> = Vec::new();

    for prod in productions {
        let runs = find_nullable_runs(&prod.rhs, &nullability, threshold);
        if runs.is_empty() {
            result.push(prod.clone());
            continue;
        }

        // Process runs right-to-left to preserve indices after splice
        let mut new_rhs = prod.rhs.clone();
        for &(start, end) in runs.iter().rev() {
            let segment: Vec<Symbol> = new_rhs.drain(start..=end).collect();
            if segment.len() <= 1 {
                // Singleton nullable runs do not benefit from chain encoding and can
                // introduce unit-derivation artifacts (chain -> NT) after epsilon
                // elimination. Keep them as-is and let exhaustive inlining handle them.
                if let Some(sym) = segment.into_iter().next() {
                    new_rhs.insert(start, sym);
                }
                continue;
            }
            let chain_root = match direction {
                Direction::Right => build_right_chain(&segment, &prod.lhs.0, new_name_gen, &mut result),
                Direction::Left  => build_left_chain(&segment, &prod.lhs.0, new_name_gen, &mut result),
            };
            new_rhs.insert(start, Symbol::NonTerminal(chain_root));
        }
        result.push(Production { lhs: prod.lhs.clone(), rhs: new_rhs });
    }
    result
}

/// Build a right-linear chain that accepts any ordered subsequence of `segment`.
///
/// For `segment = [A, B, C]` creates:
/// ```text
/// __chain_X_0 → A __chain_X_1 | __chain_X_1
/// __chain_X_1 → B __chain_X_2 | __chain_X_2
/// __chain_X_2 → C | ""
/// ```
/// Returns the root nonterminal (`__chain_X_0`).
fn build_right_chain(
    segment: &[Symbol],
    base: &str,
    gen: &mut impl FnMut(&str) -> String,
    new_prods: &mut Vec<Production>,
) -> NonTerminal {
    let n = segment.len();
    assert!(n > 0);

    // Allocate names
    let names: Vec<String> = (0..n).map(|_| gen(base)).collect();

    for (i, sym) in segment.iter().enumerate() {
        let lhs = NonTerminal(names[i].clone());
        if i + 1 < n {
            let next = NonTerminal(names[i + 1].clone());
            // sym next | next
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![sym.clone(), Symbol::NonTerminal(next.clone())] });
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![Symbol::NonTerminal(next)] });
        } else {
            // Last: sym | ""  (epsilon production — will be removed by inline_null_productions)
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![sym.clone()] });
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![] });
        }
    }

    NonTerminal(names[0].clone())
}

/// Build a left-linear chain that accepts any ordered subsequence of `segment`.
///
/// For `segment = [A, B, C]` creates:
/// ```text
/// __chain_X_2 → C | ""
/// __chain_X_1 → __chain_X_2 B | __chain_X_2
/// __chain_X_0 → __chain_X_1 A... wait, left-linear means A first.
/// ```
///
/// Actually for a LEFT-linear chain accumulating A then B then C:
/// ```text
/// __chain_X_0 → __chain_X_1 C | __chain_X_1
/// __chain_X_1 → __chain_X_2 B | __chain_X_2
/// __chain_X_2 → A | ""
/// ```
/// This reads: chain_0 = the full sequence; chain_2 decides A; chain_1 decides B; chain_0 decides C.
/// Returns the root nonterminal (`__chain_X_0`).
fn build_left_chain(
    segment: &[Symbol],
    base: &str,
    gen: &mut impl FnMut(&str) -> String,
    new_prods: &mut Vec<Production>,
) -> NonTerminal {
    let n = segment.len();
    assert!(n > 0);

    let names: Vec<String> = (0..n).map(|_| gen(base)).collect();

    // chain_{n-1} handles segment[0], chain_{n-2} handles segment[1], ..., chain_0 handles segment[n-1]
    // But we emit names[0] = root and names[n-1] = deepest.
    // Let's map: names[k] handles segment[n-1-k].
    // chain_0 (root): chain_1 segment[n-1] | chain_1
    // chain_1:        chain_2 segment[n-2] | chain_2
    // ...
    // chain_{n-2}:    chain_{n-1} segment[1] | chain_{n-1}
    // chain_{n-1}:    segment[0] | ""

    for k in 0..n {
        let lhs = NonTerminal(names[k].clone());
        let sym = segment[n - 1 - k].clone();  // reversed!
        if k + 1 < n {
            let next = NonTerminal(names[k + 1].clone());
            // next sym | next  (next first, then sym)
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![Symbol::NonTerminal(next.clone()), sym] });
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![Symbol::NonTerminal(next)] });
        } else {
            // Deepest: sym | ""
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![sym] });
            new_prods.push(Production { lhs: lhs.clone(), rhs: vec![] });
        }
    }

    NonTerminal(names[0].clone())
}

// ---------------------------------------------------------------------------
// Balanced-tree pre-processing
// ---------------------------------------------------------------------------

/// For a nullable NT, create a non-nullable version (`a_nn`) that has all
/// epsilon-derivable alternatives removed.  Returns the original NT unchanged
/// if it is already non-nullable or if no non-nullable alternatives are found.
///
/// An alternative is epsilon-derivable iff it is empty OR every symbol in it
/// is itself a nullable NT (terminals are never nullable in this sense).
fn create_nn_nt(
    nt: &NonTerminal,
    nullability: &BTreeMap<NonTerminal, Nullability>,
    prods_by_lhs: &BTreeMap<NonTerminal, Vec<Vec<Symbol>>>,
    gen: &mut impl FnMut(&str) -> String,
    new_prods: &mut Vec<Production>,
) -> NonTerminal {
    // If not in the nullability map or not nullable, use it as-is.
    if !matches!(nullability.get(nt), Some(Nullability::Nullable) | Some(Nullability::Null)) {
        return nt.clone();
    }

    let Some(alts) = prods_by_lhs.get(nt) else {
        return nt.clone();
    };

    let sym_is_nullable = |sym: &Symbol| -> bool {
        matches!(sym, Symbol::NonTerminal(inner)
            if matches!(nullability.get(inner), Some(Nullability::Nullable) | Some(Nullability::Null)))
    };

    // Keep alternatives that are NOT epsilon-derivable:
    // non-empty AND containing at least one non-nullable symbol.
    let kept: Vec<Vec<Symbol>> = alts.iter()
        .filter(|alt| !alt.is_empty() && alt.iter().any(|s| !sym_is_nullable(s)))
        .cloned()
        .collect();

    if kept.is_empty() {
        // All alternatives are epsilon-derivable; can't create a meaningful nn version.
        return nt.clone();
    }

    let nn_name = gen(&nt.0);
    let nn_nt = NonTerminal(nn_name);
    for alt in kept {
        new_prods.push(Production { lhs: nn_nt.clone(), rhs: alt });
    }
    nn_nt
}

/// Replaces each nullable run of N symbols with a **non-nullable** balanced k-ary tree
/// wrapped in an explicit optional (`opt → tree | ""`).
///
/// Using non-nullable leaf and internal nodes prevents `inline_null_productions`
/// from multiplying productions inside the tree (it only sees one `opt` NT to
/// expand at the parent level, adding one extra variant instead of 2^N).
///
/// Structure for segment [A, B, C, D] with k=2 (each A–D is nullable):
/// ```text
/// A_nn  → "a"                        (non-nullable: A with "" removed)
/// B_nn  → "b"
/// leaf_0 → A_nn | B_nn | A_nn B_nn   (non-empty subsets of chunk {A_nn, B_nn})
/// leaf_1 → C_nn | D_nn | C_nn D_nn   (non-empty subsets of chunk {C_nn, D_nn})
/// mid    → leaf_0 | leaf_1 | leaf_0 leaf_1  (non-empty subsets of {leaf_0, leaf_1})
/// opt    → mid | ""                  (nullable optional wrapper — only place for "")
/// ```
/// Parent `X → … A B C D …` becomes `X → … opt …`.
fn preprocess_balanced_tree(
    productions: &[Production],
    new_name_gen: &mut impl FnMut(&str) -> String,
    k: usize,
) -> Vec<Production> {
    // Use k >= 2 to guarantee termination (k=1 would never reduce chunk count).
    let k = k.max(2);
    let nullability = compute_nonterminal_nullability(productions);

    // Build a lookup map for create_nn_nt.
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<Vec<Symbol>>> = BTreeMap::new();
    for prod in productions {
        prods_by_lhs.entry(prod.lhs.clone()).or_default().push(prod.rhs.clone());
    }

    let mut result: Vec<Production> = Vec::new();

    for prod in productions {
        let runs = find_nullable_runs(&prod.rhs, &nullability, 1); // only runs of 2+ nullable NTs
        if runs.is_empty() {
            result.push(prod.clone());
            continue;
        }

        let mut new_rhs = prod.rhs.clone();
        for &(start, end) in runs.iter().rev() {
            let segment: Vec<Symbol> = new_rhs.drain(start..=end).collect();

            // Build a non-nullable balanced tree for this nullable run.
            let root_nn = build_non_nullable_tree(
                &segment, k, &prod.lhs.0, new_name_gen, &mut result,
                &nullability, &prods_by_lhs,
            );

            // Wrap with an explicit nullable optional: opt → root_nn | ""
            let opt_name = new_name_gen(&prod.lhs.0);
            let root_opt = NonTerminal(opt_name);
            result.push(Production {
                lhs: root_opt.clone(),
                rhs: vec![Symbol::NonTerminal(root_nn)],
            });
            result.push(Production { lhs: root_opt.clone(), rhs: vec![] }); // epsilon

            new_rhs.insert(start, Symbol::NonTerminal(root_opt));
        }
        result.push(Production { lhs: prod.lhs.clone(), rhs: new_rhs });
    }
    result
}

/// Build a non-nullable balanced k-ary tree of rules covering all **non-empty**
/// ordered subsequences of `segment`.
///
/// Returns a NON-NULLABLE NT (no epsilon production).  The top-level caller
/// (`preprocess_balanced_tree`) wraps the result in `opt → result | ""`.
///
/// At the leaf level, `_nn` (non-nullable) versions of each input symbol are
/// created/looked-up to ensure `inline_null_productions` cannot expand leaf
/// productions.
fn build_non_nullable_tree(
    segment: &[Symbol],
    k: usize,
    base: &str,
    gen: &mut impl FnMut(&str) -> String,
    new_prods: &mut Vec<Production>,
    nullability: &BTreeMap<NonTerminal, Nullability>,
    prods_by_lhs: &BTreeMap<NonTerminal, Vec<Vec<Symbol>>>,
) -> NonTerminal {
    let n = segment.len();

    if n == 0 {
        // Degenerate: shouldn't happen with threshold=1 but create an epsilon NT as fallback.
        let name = gen(base);
        let nt = NonTerminal(name.clone());
        new_prods.push(Production { lhs: nt.clone(), rhs: vec![] });
        return nt;
    }

    // For each symbol, obtain a non-nullable version.
    // - For nullable NTs: call create_nn_nt to strip epsilon-derivable alternatives.
    // - For terminals or non-nullable NTs: use as-is.
    // - For freshly-created tree NTs (not in nullability map): use as-is (already non-nullable).
    let nn_segment: Vec<Symbol> = segment.iter().map(|sym| match sym {
        Symbol::NonTerminal(nt) =>
            Symbol::NonTerminal(create_nn_nt(nt, nullability, prods_by_lhs, gen, new_prods)),
        Symbol::Terminal(_) => sym.clone(),
    }).collect();

    if n <= k {
        // Leaf: enumerate all NON-EMPTY ordered subsequences (mask 1 to 2^n-1).
        // No epsilon alternative → this NT is non-nullable.
        let name = gen(base);
        let leaf_nt = NonTerminal(name.clone());
        for mask in 1u64..(1u64 << n) {
            let rhs: Vec<Symbol> = nn_segment.iter().enumerate()
                .filter(|(i, _)| (mask >> i) & 1 == 1)
                .map(|(_, sym)| sym.clone())
                .collect();
            new_prods.push(Production { lhs: leaf_nt.clone(), rhs });
        }
        return leaf_nt;
    }

    // Split nn_segment into chunks of size k and build a non-nullable NT for each chunk.
    let chunk_nts: Vec<NonTerminal> = nn_segment.chunks(k)
        .map(|chunk| build_non_nullable_tree(chunk, k, base, gen, new_prods, nullability, prods_by_lhs))
        .collect();

    // Recurse over the chunk NTs (they are non-nullable by construction).
    // Since k >= 2, chunk_nts.len() < n, so the recursion terminates.
    let chunk_syms: Vec<Symbol> = chunk_nts.iter()
        .map(|nt| Symbol::NonTerminal(nt.clone()))
        .collect();
    build_non_nullable_tree(&chunk_syms, k, base, gen, new_prods, nullability, prods_by_lhs)
}

// ---------------------------------------------------------------------------
// Utility: build a name generator scoped to an additional existing-names set
// ---------------------------------------------------------------------------

/// Wraps a mutable counter to generate names of the form `__nl_{base}_{i}`.
pub fn make_null_inline_name_gen(
    existing: &BTreeSet<NonTerminal>,
) -> impl FnMut(&str) -> String + '_ {
    // We keep a counter per base prefix to generate short, sorted names.
    let mut seen: BTreeSet<String> = existing.iter().map(|nt| nt.0.clone()).collect();
    let mut counters: BTreeMap<String, usize> = BTreeMap::new();
    move |base: &str| {
        let key = format!("__nl_{base}");
        let c = counters.entry(key.clone()).or_insert(0);
        loop {
            let candidate = format!("{key}_{c}");
            *c += 1;
            if seen.insert(candidate.clone()) {
                return candidate;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};

    fn nt(s: &str) -> Symbol { Symbol::NonTerminal(NonTerminal(s.to_string())) }
    fn t(s: &str) -> Symbol { Symbol::Terminal(Terminal::Literal(s.as_bytes().to_vec())) }
    fn prod(lhs: &str, rhs: Vec<Symbol>) -> Production {
        Production { lhs: NonTerminal(lhs.to_string()), rhs }
    }

    /// Grammar: S → a X Y Z b, X → "" | "x", Y → "" | "y", Z → "" | "z"
    fn make_test_grammar() -> Vec<Production> {
        vec![
            prod("S", vec![t("a"), nt("X"), nt("Y"), nt("Z"), t("b")]),
            prod("X", vec![t("x")]),
            prod("X", vec![]),
            prod("Y", vec![t("y")]),
            prod("Y", vec![]),
            prod("Z", vec![t("z")]),
            prod("Z", vec![]),
        ]
    }

    /// Collect all strings derivable from `start` in `prods`.
    /// Limits depth to avoid infinite loops.
    fn all_strings(prods: &[Production], start: &str) -> BTreeSet<String> {
        fn derive(
            rhs: &[Symbol], prods: &[Production], depth: usize, buf: &mut String,
            results: &mut BTreeSet<String>,
        ) {
            if depth > 20 { return; }
            if rhs.is_empty() { results.insert(buf.clone()); return; }
            match &rhs[0] {
                Symbol::Terminal(Terminal::Literal(bytes)) => {
                    let s = String::from_utf8_lossy(bytes).to_string();
                    buf.push_str(&s);
                    derive(&rhs[1..], prods, depth, buf, results);
                    buf.truncate(buf.len() - s.len());
                }
                Symbol::NonTerminal(nt) => {
                    for p in prods.iter().filter(|p| &p.lhs == nt) {
                        let combined: Vec<Symbol> = p.rhs.iter().chain(&rhs[1..]).cloned().collect();
                        derive(&combined, prods, depth + 1, buf, results);
                    }
                }
                _ => {} // ignore other terminal kinds in tests
            }
        }

        let mut results = BTreeSet::new();
        let start_nt = NonTerminal(start.to_string());
        for p in prods.iter().filter(|p| p.lhs == start_nt) {
            let mut buf = String::new();
            derive(&p.rhs, prods, 0, &mut buf, &mut results);
        }
        results
    }

    fn expected_strings() -> BTreeSet<String> {
        let mut s = BTreeSet::new();
        for x in &["", "x"] {
            for y in &["", "y"] {
                for z in &["", "z"] {
                    s.insert(format!("a{x}{y}{z}b"));
                }
            }
        }
        s
    }

    fn test_strategy(strategy: &NullableInliningStrategy) {
        let grammar = make_test_grammar();
        let all_nts: BTreeSet<NonTerminal> = grammar.iter().map(|p| p.lhs.clone()).collect();
        let mut gen = make_null_inline_name_gen(&all_nts);
        let result = run_null_inline(&grammar, strategy, &mut gen);
        let strings = all_strings(&result, "S");
        assert_eq!(
            strings, expected_strings(),
            "Strategy {:?} produced wrong language: {:?}",
            strategy, strings
        );
    }

    #[test]
    fn test_exhaustive()     { test_strategy(&NullableInliningStrategy::Exhaustive); }
    #[test]
    fn test_right_chain()    { test_strategy(&NullableInliningStrategy::RightChain); }
    #[test]
    fn test_left_chain()     { test_strategy(&NullableInliningStrategy::LeftChain); }
    #[test]
    fn test_balanced_tree_1(){ test_strategy(&NullableInliningStrategy::BalancedTree(1)); }
    #[test]
    fn test_balanced_tree_2(){ test_strategy(&NullableInliningStrategy::BalancedTree(2)); }
    #[test]
    fn test_balanced_tree_3(){ test_strategy(&NullableInliningStrategy::BalancedTree(3)); }
}