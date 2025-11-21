use std::collections::{BTreeMap, BTreeSet, HashMap};
use bimap::BiBTreeMap;
use crate::finite_automata::Expr;
use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::interface::{GrammarDefinition, GrammarExpr};

//======================================================================================
// PART 1: From src/interface/terminal_merge_opt.rs
//======================================================================================

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
    crate::debug!(3, "Optimizing terminals");
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

//======================================================================================
// PART 2: From src/glr/optimization.rs
//======================================================================================

pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    // Fixed-point iteration for optimizations.
    for _ in 0..8 { // Limit passes to avoid infinite loops on tricky cases.
        let mut changed = false;
        changed |= convert_leaf_nts_to_terminals(grammar);
        changed |= merge_adjacent_terminals(grammar);

        if !changed {
            break;
        }
    }
}

/// Converts non-terminals whose productions are all sequences of terminals
/// into a single new terminal. Repeats until no such non-terminals are found.
fn convert_leaf_nts_to_terminals(grammar: &mut GrammarDefinition) -> bool {
    let mut changed_in_pass = false;

    loop {
        let mut changed_this_iteration = false;
        let non_terminals: BTreeSet<_> = grammar.productions.iter().map(|p| p.lhs.clone()).collect();
        let mut convertible_nts = BTreeSet::new();

        for nt in &non_terminals {
            let prods_for_nt: Vec<_> = grammar.productions.iter().filter(|p| &p.lhs == nt).collect();
            if prods_for_nt.is_empty() { continue; }
            if prods_for_nt.iter().all(|p| p.rhs.iter().all(|s| matches!(s, Symbol::Terminal(_)))) {
                convertible_nts.insert(nt.clone());
            }
        }

        if convertible_nts.is_empty() {
            break;
        }

        for nt in convertible_nts {
            let prods_for_nt: Vec<_> = grammar.productions.iter().filter(|p| p.lhs == nt).cloned().collect();

            let choice_of_seqs: Vec<Expr> = prods_for_nt.iter().map(|p| {
                let term_exprs: Vec<Expr> = p.rhs.iter().map(|s| {
                    if let Symbol::Terminal(t) = s {
                        get_expr_for_terminal(t, grammar)
                    } else {
                        unreachable!("Expected only terminals at this stage for {}", nt.0);
                    }
                }).collect();

                if term_exprs.len() == 1 {
                    term_exprs.into_iter().next().unwrap()
                } else if term_exprs.is_empty() {
                    Expr::Epsilon
                } else {
                    Expr::Seq(term_exprs)
                }
            }).collect();

            let nt_expr = if choice_of_seqs.len() == 1 {
                choice_of_seqs.into_iter().next().unwrap()
            } else {
                Expr::Choice(choice_of_seqs)
            };

            let new_terminal_name = nt.0.clone();
            let new_terminal = Terminal::RegexName(new_terminal_name.clone());
            
            let new_group_id = grammar.group_id_to_expr.keys().max().map(|id| id + 1).unwrap_or(0);
            
            grammar.regex_name_to_group_id.insert(new_terminal_name, new_group_id);
            grammar.group_id_to_expr.insert(new_group_id, nt_expr);

            grammar.productions.retain(|p| p.lhs != nt);

            for prod in &mut grammar.productions {
                for symbol in &mut prod.rhs {
                    if let Symbol::NonTerminal(rhs_nt) = symbol {
                        if rhs_nt == &nt {
                            *symbol = Symbol::Terminal(new_terminal.clone());
                        }
                    }
                }
            }
            changed_this_iteration = true;
        }
        
        if changed_this_iteration {
            changed_in_pass = true;
        } else {
            break;
        }
    }

    changed_in_pass
}

fn merge_adjacent_terminals(grammar: &mut GrammarDefinition) -> bool {
    let mut changed = false;
    let mut terminal_counts = HashMap::new();
    for prod in &grammar.productions {
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                *terminal_counts.entry(t.clone()).or_insert(0) += 1;
            }
        }
    }

    let productions = std::mem::take(&mut grammar.productions);
    let mut new_productions = Vec::with_capacity(productions.len());

    for prod in &productions {
        let mut new_rhs = Vec::new();
        let mut i = 0;
        while i < prod.rhs.len() {
            if i + 1 < prod.rhs.len() {
                if let (Symbol::Terminal(t1), Symbol::Terminal(t2)) = (&prod.rhs[i], &prod.rhs[i+1]) {
                    // Only merge if they appear together and nowhere else. This is a simple heuristic.
                    if *terminal_counts.get(t1).unwrap_or(&0) == 1 && *terminal_counts.get(t2).unwrap_or(&0) == 1 {
                        let new_terminal = merge_terminals_internal(t1, t2, grammar);
                        new_rhs.push(Symbol::Terminal(new_terminal));
                        i += 2;
                        changed = true;
                        continue;
                    }
                }
            }
            new_rhs.push(prod.rhs[i].clone());
            i += 1;
        }
        new_productions.push(Production {
            lhs: prod.lhs.clone(),
            rhs: new_rhs,
        });
    }

    grammar.productions = new_productions;
    changed
}

fn get_expr_for_terminal(t: &Terminal, grammar: &GrammarDefinition) -> Expr {
    let group_id_opt = match t {
        Terminal::Literal(bytes) => grammar.literal_to_group_id.get_by_left(bytes),
        Terminal::RegexName(name) => grammar.regex_name_to_group_id.get_by_left(name),
    };

    let group_id = group_id_opt.unwrap_or_else(|| panic!("Terminal {:?} not found in grammar terminal maps", t));
    grammar.group_id_to_expr.get(group_id).cloned().unwrap_or_else(|| panic!("No expr for terminal {:?}", t))
}

fn merge_terminals_internal(t1: &Terminal, t2: &Terminal, grammar: &mut GrammarDefinition) -> Terminal {
    let expr1 = get_expr_for_terminal(t1, grammar);
    let expr2 = get_expr_for_terminal(t2, grammar);

    let new_expr = match (expr1, expr2) {
        (Expr::U8Seq(mut v1), Expr::U8Seq(v2)) => {
            v1.extend(v2);
            Expr::U8Seq(v1)
        },
        (e1, e2) => Expr::Seq(vec![e1, e2])
    };

    let new_terminal = match (t1, t2) {
        (Terminal::Literal(l1), Terminal::Literal(l2)) => {
            let mut new_bytes = l1.clone();
            new_bytes.extend(l2);
            Terminal::Literal(new_bytes)
        },
        _ => {
            let name1 = match t1 { Terminal::RegexName(n) => n.clone(), Terminal::Literal(l) => String::from_utf8_lossy(l).to_string() };
            let name2 = match t2 { Terminal::RegexName(n) => n.clone(), Terminal::Literal(l) => String::from_utf8_lossy(l).to_string() };
            let new_name = format!("({}+{})", name1, name2);
            Terminal::RegexName(new_name)
        }
    };
    
    let new_group_id = grammar.group_id_to_expr.keys().max().map(|id| id + 1).unwrap_or(0);

    let final_terminal = match new_terminal {
        Terminal::Literal(bytes) => {
            grammar.literal_to_group_id.insert(bytes.clone(), new_group_id);
            Terminal::Literal(bytes)
        },
        Terminal::RegexName(name) => {
            let mut final_name = name.clone();
            let mut idx = 1;
            while grammar.regex_name_to_group_id.contains_left(&final_name) {
                final_name = format!("{}_{}", name, idx);
                idx += 1;
            }
            grammar.regex_name_to_group_id.insert(final_name.clone(), new_group_id);
            Terminal::RegexName(final_name)
        }
    };

    grammar.group_id_to_expr.insert(new_group_id, new_expr);
    final_terminal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};

    #[test]
    fn test_converts_leaf_nt_to_terminal() {
        let (grammar_exprs, regex_exprs) = (
            vec![
                ("start".to_string(), GrammarExpr::Ref("A".to_string())),
                ("A".to_string(), GrammarExpr::Choice(vec![GrammarExpr::Ref("B".to_string()), GrammarExpr::Ref("C".to_string())])),
            ],
            vec![
                ("B".to_string(), Expr::U8Seq(b"b".to_vec())),
                ("C".to_string(), Expr::U8Seq(b"c".to_vec())),
            ]
        );
        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
        optimize_grammar(&mut grammar);

        // The NT 'A' should be converted to a terminal, and start should now reference it.
        let start_prod = grammar.productions.iter().find(|p| p.lhs.0 == "start'").unwrap(); // Augmented start
        let root_prod = grammar.productions.iter().find(|p| p.lhs.0 == "start").unwrap();
        assert_eq!(root_prod.rhs.len(), 1);
        assert!(matches!(&root_prod.rhs[0], Symbol::Terminal(Terminal::RegexName(name)) if name == "A"));
        
        // Productions for A should be gone.
        assert!(grammar.productions.iter().find(|p| p.lhs.0 == "A").is_none());

        // A new terminal "A" should exist.
        assert!(grammar.regex_name_to_group_id.contains_left("A"));
        let group_id = grammar.regex_name_to_group_id.get_by_left("A").unwrap();
        let expr = grammar.group_id_to_expr.get(group_id).unwrap();
        assert!(matches!(expr, Expr::Choice(_)));
    }

    #[test]
    fn test_merge_adjacent_terminals() {
        let (grammar_exprs, regex_exprs) = (
            vec![
                ("start".to_string(), GrammarExpr::Sequence(vec![GrammarExpr::Ref("A".to_string()), GrammarExpr::Ref("B".to_string())])),
            ],
            vec![
                ("A".to_string(), Expr::U8Seq(b"a".to_vec())),
                ("B".to_string(), Expr::U8Seq(b"b".to_vec())),
            ]
        );
        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
        optimize_grammar(&mut grammar);

        let prod = grammar.productions.iter().find(|p| p.lhs.0 == "start").unwrap();
        assert_eq!(prod.rhs.len(), 1);
        let merged_terminal = prod.rhs[0].clone();

        assert!(matches!(merged_terminal, Symbol::Terminal(Terminal::RegexName(_))));
        if let Symbol::Terminal(t) = merged_terminal {
             let expr = get_expr_for_terminal(&t, &grammar);
             assert_eq!(expr, Expr::Seq(vec![Expr::U8Seq(b"a".to_vec()), Expr::U8Seq(b"b".to_vec())]));
        }
    }

    #[test]
    fn test_rolls_up_chain_of_regular_rules() {
        let mut grammar_exprs = vec![("start".to_string(), GrammarExpr::Ref("s0".to_string()))];
        let mut regex_exprs = vec![("C".to_string(), Expr::U8Seq(b"c".to_vec()))];

        let chain_len = 20;

        for i in 0..chain_len {
             let char_val = (b'a' + i as u8) as char;
             let term_name = format!("T{}", i);
             regex_exprs.push((term_name.clone(), Expr::U8Seq(vec![char_val as u8])));

            let next_s = if i < chain_len -1 {
                GrammarExpr::Ref(format!("s{}", i + 1))
            } else {
                GrammarExpr::Ref("C".to_string())
            };

            grammar_exprs.push((
                format!("s{}", i),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(term_name),
                    next_s
                ])
            ));
        }

        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
        optimize_grammar(&mut grammar);

        assert_eq!(grammar.productions.len(), 2); // start' -> start, and start -> <rolled_up_terminal>
        let prod = grammar.productions.iter().find(|p| p.lhs.0 == "start").unwrap();
        assert_eq!(prod.rhs.len(), 1);

        let final_terminal = match &prod.rhs[0] {
            Symbol::Terminal(t) => t,
            _ => panic!("Expected terminal"),
        };

        let final_expr = get_expr_for_terminal(final_terminal, &grammar);

        let expected_bytes: Vec<u8> = (0..chain_len).map(|i| b'a' + i).chain(std::iter::once(b'c')).collect();
        assert_eq!(final_expr, Expr::Seq(vec![Expr::U8Seq(expected_bytes)]));
    }
}
