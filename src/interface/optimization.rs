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

/// Optimizes the grammar by merging adjacent terminals that appear only once.
pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    let mut changed = true;
    while changed {
        changed = false;
        // 1. Count occurrences of each terminal
        let mut terminal_counts = HashMap::new();
        for prod in &grammar.productions {
            for symbol in &prod.rhs {
                if let Symbol::Terminal(t) = symbol {
                    *terminal_counts.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }

        // 2. Identify mergeable pairs and merge them
        // We'll do this by iterating over productions and modifying them in place if possible,
        // or building a new list of productions.
        // Since we need to update the grammar's terminal definitions as well, we might need to be careful.

        let mut new_productions = Vec::new();
        let mut merged_any_in_pass = false;
        let productions = std::mem::take(&mut grammar.productions);

        for prod in &productions {
            let mut new_rhs = Vec::new();
            let mut i = 0;
            while i < prod.rhs.len() {
                let symbol = &prod.rhs[i];
                
                // Check if we can merge current symbol with the next one
                if i + 1 < prod.rhs.len() {
                    let next_symbol = &prod.rhs[i+1];
                    
                    if let (Symbol::Terminal(t1), Symbol::Terminal(t2)) = (symbol, next_symbol) {
                        if terminal_counts.get(&t1) == Some(&1) && terminal_counts.get(&t2) == Some(&1) {
                            // Merge t1 and t2
                            let new_terminal = merge_terminals(&t1, &t2, grammar);
                            new_rhs.push(Symbol::Terminal(new_terminal));
                            i += 2; // Skip next symbol
                            merged_any_in_pass = true;
                            changed = true;
                            continue;
                        }
                    }
                }
                
                new_rhs.push(symbol.clone());
                i += 1;
            }
            new_productions.push(Production {
                lhs: prod.lhs.clone(),
                rhs: new_rhs,
            });
        }
        
        grammar.productions = new_productions;
        
        if merged_any_in_pass {
            // We might want to clean up unused terminals from the grammar definitions here,
            // but it's not strictly necessary for correctness, just for cleanliness.
            // The loop will continue until no more merges are possible.
        }
    }
}

fn merge_terminals(t1: &Terminal, t2: &Terminal, grammar: &mut GrammarDefinition) -> Terminal {
    // 1. Get Exprs for t1 and t2
    let expr1 = get_expr_for_terminal(t1, grammar);
    let expr2 = get_expr_for_terminal(t2, grammar);
    
    // 2. Create new Expr
    let new_expr = match (expr1.clone(), expr2.clone()) {
        (Expr::U8Seq(mut v1), Expr::U8Seq(v2)) => {
            v1.extend(v2);
            Expr::U8Seq(v1)
        },
        (e1, e2) => {
             Expr::Seq(vec![e1, e2])
        }
    };
    
    // 3. Create new Terminal
    // If both are literals, we might make a new literal terminal.
    // Otherwise, it's a regex terminal.
    let new_terminal = match (t1, t2) {
        (Terminal::Literal(l1), Terminal::Literal(l2)) => {
            let mut new_bytes = l1.clone();
            new_bytes.extend(l2);
            Terminal::Literal(new_bytes)
        },
        _ => {
             // Generate a new name
             let name1 = match t1 { Terminal::RegexName(n) => n.clone(), Terminal::Literal(l) => format!("{:?}", l) }; // Simplified name generation
             let name2 = match t2 { Terminal::RegexName(n) => n.clone(), Terminal::Literal(l) => format!("{:?}", l) };
             let new_name = format!("{}_{}", name1, name2); 
             // Ensure uniqueness? The grammar usually handles unique names, but here we are synthesizing.
             // Let's use a simpler approach or ensure uniqueness if needed.
             // For now, let's just use a combined name.
             Terminal::RegexName(new_name)
        }
    };
    
    // 4. Register new terminal in grammar
    // We need a new group_id
    let new_group_id = grammar.group_id_to_expr.keys().max().cloned().unwrap_or(0) + 1;
    
    match &new_terminal {
        Terminal::Literal(bytes) => {
            grammar.literal_to_group_id.insert(bytes.clone(), new_group_id);
        },
        Terminal::RegexName(name) => {
            // Check if name exists, if so, append index
            let mut final_name = name.clone();
            let mut idx = 1;
            while grammar.regex_name_to_group_id.contains_left(&final_name) {
                final_name = format!("{}_{}", name, idx);
                idx += 1;
            }
            let new_terminal_fixed = Terminal::RegexName(final_name.clone());
             grammar.regex_name_to_group_id.insert(final_name, new_group_id);
             // Update return value if name changed
             if let Terminal::RegexName(_) = new_terminal {
                 // This is a bit messy because we return 'new_terminal' which might have the old name.
                 // Let's just return the fixed one.
                 grammar.group_id_to_expr.insert(new_group_id, new_expr);
                 return new_terminal_fixed;
             }
        }
    }
    
    grammar.group_id_to_expr.insert(new_group_id, new_expr);
    
    new_terminal
}

fn get_expr_for_terminal(t: &Terminal, grammar: &GrammarDefinition) -> Expr {
    let group_id = match t {
        Terminal::Literal(bytes) => grammar.literal_to_group_id.get_by_left(bytes),
        Terminal::RegexName(name) => grammar.regex_name_to_group_id.get_by_left(name),
    };
    
    if let Some(gid) = group_id {
        grammar.group_id_to_expr.get(gid).cloned().unwrap_or(Expr::Epsilon) // Should not happen
    } else {
        // Fallback or error? 
        Expr::Epsilon
    }
}

impl GrammarDefinition {
    pub fn optimize(&mut self) {
        optimize_grammar(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
    use crate::datastructures::u8set::U8Set;

    fn create_dummy_grammar() -> GrammarDefinition {
        // S -> A B
        // A -> "a"
        // B -> "b"
        let mut grammar = GrammarDefinition {
            productions: vec![
                Production {
                    lhs: NonTerminal("S".to_string()),
                    rhs: vec![
                        Symbol::Terminal(Terminal::literal(b"a".to_vec())),
                        Symbol::Terminal(Terminal::literal(b"b".to_vec())),
                    ],
                }
            ],
            start_production_id: 0,
            literal_to_group_id: BiBTreeMap::new(),
            regex_name_to_group_id: BiBTreeMap::new(),
            group_id_to_expr: BTreeMap::new(),
            ignore_terminal_id: None,
            external_name_to_group_id: BiBTreeMap::new(),
        };

        // Register terminals
        grammar.literal_to_group_id.insert(b"a".to_vec(), 1);
        grammar.group_id_to_expr.insert(1, Expr::U8Seq(b"a".to_vec()));

        grammar.literal_to_group_id.insert(b"b".to_vec(), 2);
        grammar.group_id_to_expr.insert(2, Expr::U8Seq(b"b".to_vec()));

        grammar
    }

    #[test]
    fn test_optimize_grammar_merges_literals() {
        let mut grammar = create_dummy_grammar();
        optimize_grammar(&mut grammar);

        // Should now be S -> "ab"
        assert_eq!(grammar.productions.len(), 1);
        let prod = &grammar.productions[0];
        assert_eq!(prod.rhs.len(), 1);
        
        if let Symbol::Terminal(Terminal::Literal(bytes)) = &prod.rhs[0] {
            assert_eq!(*bytes, b"ab".to_vec());
        } else {
            panic!("Expected merged literal terminal, got {:?}", prod.rhs[0]);
        }
        
        // Check if new terminal is registered
        assert!(grammar.literal_to_group_id.contains_left(&b"ab".to_vec()));
    }

    #[test]
    fn test_optimize_grammar_merges_regex() {
        let mut grammar = create_dummy_grammar();
        // Add a regex terminal
        // S -> R1 R2
        // R1 -> [a-z] (appears once)
        // R2 -> [0-9] (appears once)
        
        let r1_name = "R1".to_string();
        let r2_name = "R2".to_string();
        
        grammar.productions.push(Production {
            lhs: NonTerminal("S2".to_string()),
            rhs: vec![
                Symbol::Terminal(Terminal::RegexName(r1_name.clone())),
                Symbol::Terminal(Terminal::RegexName(r2_name.clone())),
            ],
        });
        
        grammar.regex_name_to_group_id.insert(r1_name.clone(), 3);
        grammar.group_id_to_expr.insert(3, Expr::U8Class(U8Set::from_u8(b'a'))); // Simplified regex
        
        grammar.regex_name_to_group_id.insert(r2_name.clone(), 4);
        grammar.group_id_to_expr.insert(4, Expr::U8Class(U8Set::from_u8(b'0'))); // Simplified regex
        
        optimize_grammar(&mut grammar);
        
        // Find S2 production
        let prod = grammar.productions.iter().find(|p| p.lhs.0 == "S2").expect("S2 production not found");
        assert_eq!(prod.rhs.len(), 1);
        
        if let Symbol::Terminal(Terminal::RegexName(name)) = &prod.rhs[0] {
            assert!(name.contains("R1"));
            assert!(name.contains("R2"));
            assert!(grammar.regex_name_to_group_id.contains_left(name));
        } else {
            panic!("Expected merged regex terminal, got {:?}", prod.rhs[0]);
        }
    }
}