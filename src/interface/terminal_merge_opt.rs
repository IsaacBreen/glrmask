use std::collections::{BTreeMap, BTreeSet};

use super::GrammarExpr;
use crate::finite_automata::Expr;

/// Lightweight grammar-level optimization that merges adjacent terminal references
/// into a single terminal, when it is safe to do so.
///
/// Heuristic:
/// - Work on `(grammar_exprs, terminal_defs)` as used by `GrammarDefinition::from_exprs`.
/// - Consider only terminals that:
///     * Have a definition in `terminal_defs`.
///     * That definition is a simple `Expr::U8Seq` (a fixed byte string).
/// - In each pass, for every `GrammarExpr::Sequence` we look for adjacent
///   `GrammarExpr::Ref` elements `T1 T2` such that:
///     * Both `T1` and `T2` refer to terminals.
///     * Each of `T1` and `T2` is referenced exactly once in the whole grammar.
/// - Such a pair is replaced by a single `Ref` to a freshly created terminal
///   whose definition is the concatenation of the underlying byte sequences.
/// - We repeat passes until no further merges are possible.
///
/// This does not remove the old terminal definitions; it only stops using them in
/// productions. This keeps compatibility with callers that expect every
/// terminal definition they provided to still exist.
pub fn optimize_terminals(
    grammar_exprs: Vec<(String, GrammarExpr)>,
    regex_exprs: Vec<(String, Expr)>,
) -> (Vec<(String, GrammarExpr)>, Vec<(String, Expr)>) {
    if grammar_exprs.is_empty() || regex_exprs.is_empty() {
        return (grammar_exprs, regex_exprs);
    }

    let mut grammar_exprs = grammar_exprs;
    let mut terminal_defs: BTreeMap<String, Expr> = regex_exprs.into_iter().collect();

    // Collect all names (non-terminals + terminals) to avoid clashes when creating
    // merged terminal names.
    let mut all_names: BTreeSet<String> =
        grammar_exprs.iter().map(|(name, _)| name.clone()).collect();
    all_names.extend(terminal_defs.keys().cloned());

    // Fixed-point iteration with a conservative upper bound on passes.
    const MAX_PASSES: usize = 8;
    for _pass in 0..MAX_PASSES {
        let usage_counts = compute_terminal_ref_counts(&grammar_exprs, &terminal_defs);

        let mut changed_any = false;
        let new_grammar_exprs: Vec<(String, GrammarExpr)> = grammar_exprs
            .into_iter()
            .map(|(name, expr)| {
                let new_expr = rewrite_expr_for_pass(
                    &expr,
                    &usage_counts,
                    &mut terminal_defs,
                    &mut all_names,
                    &mut changed_any,
                );
                (name, new_expr)
            })
            .collect();

        grammar_exprs = new_grammar_exprs;

        if !changed_any {
            break;
        }
    }

    // Convert terminal definitions back to a Vec in deterministic order.
    let mut out_defs: Vec<(String, Expr)> = terminal_defs.into_iter().collect();
    out_defs.sort_by(|(a, _), (b, _)| a.cmp(b));
    (grammar_exprs, out_defs)
}

/// Count how many times each terminal is referenced in the non-terminal rules.
///
/// Only `GrammarExpr::Ref(name)` that refers to a terminal name (present in
/// `terminal_defs`) are counted; references to non-terminals are ignored.
fn compute_terminal_ref_counts(
    grammar_exprs: &[(String, GrammarExpr)],
    terminal_defs: &BTreeMap<String, Expr>,
) -> BTreeMap<String, usize> {
    let terminal_names: BTreeSet<&str> =
        terminal_defs.keys().map(|s| s.as_str()).collect();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();

    for (_rule_name, expr) in grammar_exprs {
        collect_refs(expr, &terminal_names, &mut counts);
    }

    counts
}

fn collect_refs(
    expr: &GrammarExpr,
    terminal_names: &BTreeSet<&str>,
    counts: &mut BTreeMap<String, usize>,
) {
    match expr {
        GrammarExpr::Ref(name) => {
            if terminal_names.contains(name.as_str()) {
                *counts.entry(name.clone()).or_insert(0) += 1;
            }
        }
        GrammarExpr::Sequence(exprs) | GrammarExpr::Choice(exprs) => {
            for e in exprs {
                collect_refs(e, terminal_names, counts);
            }
        }
        GrammarExpr::Optional(inner) | GrammarExpr::Repeat(inner) => {
            collect_refs(inner, terminal_names, counts);
        }
        GrammarExpr::Literal(_)
        | GrammarExpr::CharClass(_)
        | GrammarExpr::AnyChar => {}
    }
}

fn rewrite_expr_for_pass(
    expr: &GrammarExpr,
    usage_counts: &BTreeMap<String, usize>,
    terminal_defs: &mut BTreeMap<String, Expr>,
    all_names: &mut BTreeSet<String>,
    changed_any: &mut bool,
) -> GrammarExpr {
    match expr {
        GrammarExpr::Sequence(items) => {
            let mut new_items: Vec<GrammarExpr> = Vec::with_capacity(items.len());
            let mut i = 0;
            while i < items.len() {
                if i + 1 < items.len() {
                    if let (GrammarExpr::Ref(name1), GrammarExpr::Ref(name2)) =
                        (&items[i], &items[i + 1])
                    {
                        if can_merge_pair(name1, name2, usage_counts, terminal_defs) {
                            let merged_name =
                                generate_merged_name(name1, name2, all_names);
                            ensure_merged_definition(
                                terminal_defs,
                                &merged_name,
                                name1,
                                name2,
                            );
                            new_items.push(GrammarExpr::Ref(merged_name));
                            *changed_any = true;
                            i += 2;
                            continue;
                        }
                    }
                }

                let rewritten =
                    rewrite_expr_for_pass(&items[i], usage_counts, terminal_defs, all_names, changed_any);
                new_items.push(rewritten);
                i += 1;
            }
            GrammarExpr::Sequence(new_items)
        }
        GrammarExpr::Choice(exprs) => GrammarExpr::Choice(
            exprs
                .iter()
                .map(|e| {
                    rewrite_expr_for_pass(
                        e,
                        usage_counts,
                        terminal_defs,
                        all_names,
                        changed_any,
                    )
                })
                .collect(),
        ),
        GrammarExpr::Optional(inner) => GrammarExpr::Optional(Box::new(
            rewrite_expr_for_pass(inner, usage_counts, terminal_defs, all_names, changed_any),
        )),
        GrammarExpr::Repeat(inner) => GrammarExpr::Repeat(Box::new(
            rewrite_expr_for_pass(inner, usage_counts, terminal_defs, all_names, changed_any),
        )),
        // Literals / char-classes / any-char / bare refs (that are not part of a
        // mergeable pair) are left as-is.
        _ => expr.clone(),
    }
}

/// Returns true if `name1` and `name2` form a mergeable adjacent pair.
fn can_merge_pair(
    name1: &str,
    name2: &str,
    usage_counts: &BTreeMap<String, usize>,
    terminal_defs: &BTreeMap<String, Expr>,
) -> bool {
    if name1 == name2 {
        return false;
    }

    let count1 = match usage_counts.get(name1) {
        Some(c) => *c,
        None => return false,
    };
    let count2 = match usage_counts.get(name2) {
        Some(c) => *c,
        None => return false,
    };

    if count1 != 1 || count2 != 1 {
        return false;
    }

    get_literal_bytes(terminal_defs, name1).is_some()
        && get_literal_bytes(terminal_defs, name2).is_some()
}

/// Extracts the literal byte sequence for a terminal, if it is a simple `U8Seq`
/// and non-empty.
fn get_literal_bytes<'a>(
    terminal_defs: &'a BTreeMap<String, Expr>,
    name: &str,
) -> Option<&'a [u8]> {
    match terminal_defs.get(name) {
        Some(Expr::U8Seq(bytes)) if !bytes.is_empty() => Some(bytes.as_slice()),
        _ => None,
    }
}

/// Ensures that a merged terminal definition exists for `merged_name`.
fn ensure_merged_definition(
    terminal_defs: &mut BTreeMap<String, Expr>,
    merged_name: &str,
    left: &str,
    right: &str,
) {
    if terminal_defs.contains_key(merged_name) {
        return;
    }

    let left_bytes = get_literal_bytes(terminal_defs, left)
        .expect("left terminal should be a literal at this point");
    let right_bytes = get_literal_bytes(terminal_defs, right)
        .expect("right terminal should be a literal at this point");

    let mut merged = Vec::with_capacity(left_bytes.len() + right_bytes.len());
    merged.extend_from_slice(left_bytes);
    merged.extend_from_slice(right_bytes);

    terminal_defs.insert(merged_name.to_string(), Expr::U8Seq(merged));
}

/// Generates a fresh merged terminal name that does not collide with any
/// existing non-terminal or terminal names.
fn generate_merged_name(
    left: &str,
    right: &str,
    all_names: &mut BTreeSet<String>,
) -> String {
    let mut base = format!("{}+{}", left, right);
    if all_names.insert(base.clone()) {
        return base;
    }

    let mut idx = 1usize;
    loop {
        let candidate = format!("{}+{}#{}", left, right, idx);
        if all_names.insert(candidate.clone()) {
            return candidate;
        }
        idx += 1;
    }
}
