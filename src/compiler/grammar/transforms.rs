use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::automata::regex::Expr;
use crate::automata::lexer::regex::parse_regex;
use crate::compiler::compile::build_tokenizer;
use crate::compiler::glr::analysis::{merge_identical_nonterminals, normalize_grammar};
use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Terminal};
use crate::compiler::grammar_def::{Rule, Symbol, TerminalID};
use crate::automata::lexer::tokenizer::Tokenizer;

const MAX_RUNTIME_REDUCTION_LEN: usize = 5;

// ── Nullable terminal expansion ─────────────────────────────────────────────

/// Rewrite grammar rules so that nullable terminals (those matching the empty
/// string) are treated as optional.  Operates in place on owned rule data.
///
/// For each nullable terminal `T`, a fresh nonterminal is allocated with two
/// productions: `NT → ε` and `NT → T`.  Every occurrence of `T` in the
/// existing rules is replaced by `NT`.  The tokenizer's start-state finalizer
/// for `T` is assumed to already be drained before this function is called.
pub(crate) fn expand_nullable_terminals(
    rules: &mut Vec<Rule>,
    nullable_terminals: &BTreeSet<TerminalID>,
) {
    if nullable_terminals.is_empty() {
        return;
    }

    // Compute next available nonterminal ID from existing rules.
    let mut next_nt = rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|sym| match sym {
                Symbol::Nonterminal(id) => Some(*id),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .map(|id| id + 1)
        .unwrap_or(0);

    // Map: nullable terminal id → fresh nonterminal id.
    let mut nt_for_terminal = BTreeMap::<TerminalID, NonterminalID>::new();
    let mut extra_rules = Vec::new();

    for &tid in nullable_terminals {
        let fresh_nt = next_nt;
        next_nt += 1;
        nt_for_terminal.insert(tid, fresh_nt);

        // NT → ε
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![],
        });
        // NT → T
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![Symbol::Terminal(tid)],
        });
    }

    // Rewrite existing rules in place: replace nullable Terminal(T) with Nonterminal(NT).
    for rule in rules.iter_mut() {
        for sym in rule.rhs.iter_mut() {
            if let Symbol::Terminal(tid) = sym {
                if let Some(&nt) = nt_for_terminal.get(tid) {
                    *sym = Symbol::Nonterminal(nt);
                }
            }
        }
    }

    rules.extend(extra_rules);
}

fn remap_terminal_id(terminal: &Terminal, new_id: TerminalID) -> Terminal {
    match terminal {
        Terminal::Literal { bytes, .. } => Terminal::Literal {
            id: new_id,
            bytes: bytes.clone(),
        },
        Terminal::Pattern { pattern, utf8, .. } => Terminal::Pattern {
            id: new_id,
            pattern: pattern.clone(),
            utf8: *utf8,
        },
        Terminal::Expr { expr, .. } => Terminal::Expr {
            id: new_id,
            expr: expr.clone(),
        },
    }
}

fn terminal_is_nullable(terminal: &Terminal) -> bool {
    match terminal {
        Terminal::Literal { bytes, .. } => bytes.is_empty(),
        Terminal::Pattern { pattern, utf8, .. } => parse_regex(pattern, *utf8).is_nullable(),
        Terminal::Expr { expr, .. } => expr.is_nullable(),
    }
}

fn nullable_terminals_for_grammar(grammar: &GrammarDef) -> BTreeSet<TerminalID> {
    grammar
        .terminals
        .iter()
        .filter_map(|terminal| terminal_is_nullable(terminal).then_some(terminal.id()))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TerminalIdentity {
    Literal { bytes: Vec<u8>, is_ignore: bool },
    Pattern { pattern: String, utf8: bool, is_ignore: bool },
    Expr { expr: Expr, is_ignore: bool },
}

fn terminal_identity(terminal: &Terminal, is_ignore: bool) -> TerminalIdentity {
    match terminal {
        Terminal::Literal { bytes, .. } => TerminalIdentity::Literal {
            bytes: bytes.clone(),
            is_ignore,
        },
        Terminal::Pattern { pattern, utf8, .. } => TerminalIdentity::Pattern {
            pattern: pattern.clone(),
            utf8: *utf8,
            is_ignore,
        },
        Terminal::Expr { expr, .. } => TerminalIdentity::Expr {
            expr: expr.clone(),
            is_ignore,
        },
    }
}

/// Remove terminals that are no longer referenced by any normalized rule,
/// merge identical terminals, and compact the remaining terminal IDs to a
/// dense 0..N-1 range.  Mutates the grammar in place.
pub(crate) fn compact_unused_terminals(grammar: &mut GrammarDef) {
    let mut used = BTreeSet::<TerminalID>::new();
    for rule in grammar.rules.iter() {
        for symbol in &rule.rhs {
            if let Symbol::Terminal(terminal_id) = symbol {
                used.insert(*terminal_id);
            }
        }
    }
    if let Some(ignore_terminal) = grammar.ignore_terminal {
        used.insert(ignore_terminal);
    }

    let mut remap = BTreeMap::<TerminalID, TerminalID>::new();
    let mut compacted = Vec::with_capacity(used.len());
    let mut canonical_ids = HashMap::<TerminalIdentity, TerminalID>::new();

    for old_id in used {
        let terminal = grammar.terminals.get(old_id as usize).unwrap_or_else(|| {
            panic!("terminal id {} referenced by a rule but missing from grammar.terminals", old_id)
        });
        let is_ignore = grammar.ignore_terminal == Some(old_id);
        let identity = terminal_identity(terminal, is_ignore);
        if let Some(&existing_id) = canonical_ids.get(&identity) {
            remap.insert(old_id, existing_id);
            continue;
        }
        let new_id = compacted.len() as TerminalID;
        canonical_ids.insert(identity, new_id);
        remap.insert(old_id, new_id);
        compacted.push(remap_terminal_id(terminal, new_id));
    }

    for rule in grammar.rules.iter_mut() {
        for symbol in rule.rhs.iter_mut() {
            if let Symbol::Terminal(terminal_id) = symbol {
                *terminal_id = *remap
                    .get(terminal_id)
                    .expect("used terminal must have been assigned a compacted id");
            }
        }
    }

    grammar.terminals = compacted;
    grammar.ignore_terminal = grammar.ignore_terminal.and_then(|old_id| remap.get(&old_id).copied());
    grammar.terminal_names = remap_terminal_names(&grammar.terminal_names, &remap);
}

fn remap_terminal_names(
    terminal_names: &BTreeMap<TerminalID, String>,
    remap: &BTreeMap<TerminalID, TerminalID>,
) -> BTreeMap<TerminalID, String> {
    terminal_names
        .iter()
        .filter_map(|(old_id, name)| remap.get(old_id).map(|new_id| (*new_id, name.clone())))
        .collect()
}

pub(crate) fn inline_single_use_nonterminals(
    rules: &mut Vec<Rule>,
    protected_nonterminals: &BTreeSet<NonterminalID>,
) {
    loop {
        // Build indexes
        let mut productions_by_lhs = BTreeMap::<NonterminalID, Vec<usize>>::new();
        let mut use_counts = BTreeMap::<NonterminalID, usize>::new();
        // For each nonterminal, track which rule LHS values have it at RHS position 0.
        let mut position_0_users = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();

        for (index, rule) in rules.iter().enumerate() {
            productions_by_lhs.entry(rule.lhs).or_default().push(index);
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    *use_counts.entry(*nonterminal).or_default() += 1;
                }
            }
            if let Some(Symbol::Nonterminal(first_nt)) = rule.rhs.first() {
                position_0_users.entry(*first_nt).or_default().insert(rule.lhs);
            }
        }

        // Collect inline candidates in one pass.
        let mut inline_candidates = BTreeMap::<NonterminalID, (usize, Vec<Symbol>)>::new();

        for (&nonterminal, production_indexes) in &productions_by_lhs {
            if protected_nonterminals.contains(&nonterminal) || production_indexes.len() != 1 {
                continue;
            }

            let rule = &rules[production_indexes[0]];
            if rule.rhs.is_empty()
                || rule
                    .rhs
                    .iter()
                    .any(|symbol| matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal))
            {
                continue;
            }

            let use_count = use_counts.get(&nonterminal).copied().unwrap_or(0);
            let should_inline = rule.rhs.len() == 1 || use_count == 1;
            if !should_inline {
                continue;
            }

            // Inlining `nonterminal → first ...` into a rule `X → nonterminal ...`
            // creates direct left recursion if X == first. Use precomputed index.
            let creates_direct_left_recursion =
                if let Some(Symbol::Nonterminal(first)) = rule.rhs.first() {
                    position_0_users
                        .get(&nonterminal)
                        .map_or(false, |lhss| lhss.contains(first))
                } else {
                    false
                };
            if creates_direct_left_recursion {
                continue;
            }

            inline_candidates.insert(nonterminal, (production_indexes[0], rule.rhs.clone()));
        }

        if inline_candidates.is_empty() {
            break;
        }

        // Transitively expand candidate RHS: if a candidate's RHS references
        // another candidate, substitute it. Iterate until stable.
        let inline_candidate_ids: BTreeSet<NonterminalID> =
            inline_candidates.keys().copied().collect();
        let mut expanded = true;
        while expanded {
            expanded = false;
            let snapshot: Vec<(NonterminalID, Vec<Symbol>)> = inline_candidates
                .iter()
                .map(|(&nt, (_, rhs))| (nt, rhs.clone()))
                .collect();
            for (nt, rhs) in snapshot {
                if rhs.iter().any(|s| {
                    matches!(s, Symbol::Nonterminal(id) if inline_candidate_ids.contains(id) && *id != nt)
                }) {
                    let mut new_rhs = Vec::with_capacity(rhs.len());
                    for symbol in &rhs {
                        if let Symbol::Nonterminal(id) = symbol {
                            if *id != nt {
                                if let Some((_, sub_rhs)) = inline_candidates.get(id) {
                                    new_rhs.extend(sub_rhs.iter().cloned());
                                    continue;
                                }
                            }
                        }
                        new_rhs.push(symbol.clone());
                    }
                    if new_rhs != rhs {
                        inline_candidates
                            .get_mut(&nt)
                            .expect("inline candidate should still exist")
                            .1 = new_rhs;
                        expanded = true;
                    }
                }
            }
        }

        // Collect production indexes to remove
        let remove_indexes: BTreeSet<usize> =
            inline_candidates.values().map(|(idx, _)| *idx).collect();

        // Rewrite all rules in one pass
        let mut rewritten = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            if remove_indexes.contains(&index) {
                continue;
            }

            let has_candidate = rule.rhs.iter().any(|s| {
                matches!(s, Symbol::Nonterminal(id) if inline_candidates.contains_key(id))
            });

            if has_candidate {
                let mut new_rhs = Vec::with_capacity(rule.rhs.len());
                for symbol in &rule.rhs {
                    if let Symbol::Nonterminal(id) = symbol {
                        if let Some((_, replacement_rhs)) = inline_candidates.get(id) {
                            new_rhs.extend(replacement_rhs.iter().cloned());
                            continue;
                        }
                    }
                    new_rhs.push(symbol.clone());
                }
                rewritten.push(Rule {
                    lhs: rule.lhs,
                    rhs: new_rhs,
                });
            } else {
                rewritten.push(rule.clone());
            }
        }

        *rules = rewritten;
    }
}

pub(crate) fn bound_runtime_reduction_length(
    grammar: &mut GrammarDef,
    max_rhs_len: usize,
) {
    if max_rhs_len < 2 {
        return;
    }

    let mut next_nt = grammar.num_nonterminals();
    let mut rewritten = Vec::with_capacity(grammar.rules.len());

    for rule in grammar.rules.drain(..) {
        if rule.rhs.len() <= max_rhs_len {
            rewritten.push(rule);
            continue;
        }

        let lhs_name = grammar
            .nonterminal_names
            .get(&rule.lhs)
            .cloned()
            .unwrap_or_else(|| format!("N{}", rule.lhs));
        let symbols = rule.rhs;
        let mut consumed = 1usize;
        let mut stage = 0usize;

        let first_helper = next_nt;
        next_nt += 1;
        stage += 1;
        grammar
            .nonterminal_names
            .entry(first_helper)
            .or_insert_with(|| format!("{lhs_name}__prefix_{stage}"));
        rewritten.push(Rule {
            lhs: first_helper,
            rhs: vec![symbols[0].clone()],
        });

        let mut prefix_nt = first_helper;
        while symbols.len() - consumed > max_rhs_len - 1 {
            let helper = next_nt;
            next_nt += 1;
            stage += 1;
            grammar
                .nonterminal_names
                .entry(helper)
                .or_insert_with(|| format!("{lhs_name}__prefix_{stage}"));

            let take = max_rhs_len - 1;
            let mut rhs = Vec::with_capacity(max_rhs_len);
            rhs.push(Symbol::Nonterminal(prefix_nt));
            rhs.extend(symbols[consumed..consumed + take].iter().cloned());
            rewritten.push(Rule { lhs: helper, rhs });
            prefix_nt = helper;
            consumed += take;
        }

        let mut final_rhs = Vec::with_capacity(1 + symbols.len() - consumed);
        final_rhs.push(Symbol::Nonterminal(prefix_nt));
        final_rhs.extend(symbols[consumed..].iter().cloned());
        rewritten.push(Rule {
            lhs: rule.lhs,
            rhs: final_rhs,
        });
    }

    grammar.rules = rewritten;
}

fn collect_protected_nonterminals(grammar: &GrammarDef) -> BTreeSet<NonterminalID> {
    grammar
        .nonterminal_names
        .keys()
        .copied()
        .chain(std::iter::once(grammar.start))
        .collect()
}

pub(crate) fn prepare_grammar_for_compile(grammar: &GrammarDef) -> (GrammarDef, Tokenizer) {
    // Probe nullability against the original terminal set first; nullable
    // terminals are expanded into optional grammar structure before we compact
    // away any terminals that normalization proves unreachable.
    let nullable_terminals = nullable_terminals_for_grammar(grammar);

    let mut normalized = grammar.clone();

    prepare_owned_grammar_for_compile_impl(&mut normalized, &nullable_terminals)
}

pub(crate) fn prepare_owned_grammar_for_compile(grammar: GrammarDef) -> (GrammarDef, Tokenizer) {
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    let mut normalized = grammar;

    prepare_owned_grammar_for_compile_impl(&mut normalized, &nullable_terminals)
}

/// Run only the grammar transforms without building the tokenizer.
/// The caller is responsible for calling `build_tokenizer` on the result.
pub(crate) fn prepare_grammar_transforms_only(grammar: GrammarDef) -> GrammarDef {
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    let mut normalized = grammar;
    prepare_grammar_transforms_impl(&mut normalized, &nullable_terminals);
    std::mem::take(&mut normalized)
}

/// The shared grammar transform steps (without tokenizer build).
fn prepare_grammar_transforms_impl(
    normalized: &mut GrammarDef,
    nullable_terminals: &BTreeSet<TerminalID>,
) {
    let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
        .unwrap_or(false);
    let t0 = std::time::Instant::now();
    expand_nullable_terminals(&mut normalized.rules, nullable_terminals);
    let expand_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    normalize_grammar(&mut normalized.rules, normalized.start);
    let normalize_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = std::time::Instant::now();
    let protected_nonterminals = collect_protected_nonterminals(normalized);
    inline_single_use_nonterminals(&mut normalized.rules, &protected_nonterminals);
    let inline_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = std::time::Instant::now();
    normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
    let merge1_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let t4 = std::time::Instant::now();
    bound_runtime_reduction_length(normalized, MAX_RUNTIME_REDUCTION_LEN);
    let bound_ms = t4.elapsed().as_secs_f64() * 1000.0;

    let t5 = std::time::Instant::now();
    normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
    let merge2_ms = t5.elapsed().as_secs_f64() * 1000.0;

    let t6 = std::time::Instant::now();
    compact_unused_terminals(normalized);
    let compact_ms = t6.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][prepare_transforms] expand_ms={:.3} normalize_ms={:.3} inline_ms={:.3} merge1_ms={:.3} bound_ms={:.3} merge2_ms={:.3} compact_ms={:.3}",
            expand_ms, normalize_ms, inline_ms, merge1_ms, bound_ms, merge2_ms, compact_ms,
        );
    }
}

fn prepare_owned_grammar_for_compile_impl(
    normalized: &mut GrammarDef,
    nullable_terminals: &BTreeSet<TerminalID>,
) -> (GrammarDef, Tokenizer) {
    prepare_grammar_transforms_impl(normalized, nullable_terminals);

    let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
        .unwrap_or(false);

    // Build the real tokenizer only from the compacted live terminal set so
    // dead terminals never make it into downstream lexer/parser stages.
    let t7 = std::time::Instant::now();
    let mut tokenizer = build_tokenizer(normalized);
    let tokenizer_ms = t7.elapsed().as_secs_f64() * 1000.0;

    let t8 = std::time::Instant::now();
    let _ = tokenizer.isolate_start_state_and_drain_nullable_terminals();
    let isolate_ms = t8.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][prepare_tokenizer] tokenizer_ms={:.3} isolate_ms={:.3}",
            tokenizer_ms, isolate_ms,
        );
    }

    (std::mem::take(normalized), tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::Terminal;

    fn literal(id: u32, bytes: &[u8]) -> Terminal {
        Terminal::Literal {
            id,
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn test_bound_runtime_reduction_length_rewrites_long_rule_to_prefix_chain() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Terminal(2),
                    Symbol::Terminal(3),
                    Symbol::Terminal(4),
                ],
            }],
            start: 0,
            terminals: vec![
                literal(0, b"a"),
                literal(1, b"b"),
                literal(2, b"c"),
                literal(3, b"d"),
                literal(4, b"e"),
            ],
            nonterminal_names: std::collections::BTreeMap::from([(0, "Start".to_string())]),
            ..Default::default()
        };

        bound_runtime_reduction_length(&mut grammar, 3);

        assert!(grammar.rules.iter().all(|rule| rule.rhs.len() <= 3));
        assert_eq!(grammar.rules.len(), 3);
        assert_eq!(grammar.rules[0].lhs, 1);
        assert_eq!(grammar.rules[0].rhs, vec![Symbol::Terminal(0)]);
        assert_eq!(grammar.rules[1].lhs, 2);
        assert_eq!(
            grammar.rules[1].rhs,
            vec![
                Symbol::Nonterminal(1),
                Symbol::Terminal(1),
                Symbol::Terminal(2),
            ]
        );
        assert_eq!(
            grammar.rules[2],
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(3),
                    Symbol::Terminal(4),
                ],
            }
        );
        assert_eq!(grammar.nonterminal_names.get(&1).map(String::as_str), Some("Start__prefix_1"));
        assert_eq!(grammar.nonterminal_names.get(&2).map(String::as_str), Some("Start__prefix_2"));
    }

    #[test]
    fn test_bound_runtime_reduction_length_leaves_short_rules_unchanged() {
        let original_rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            },
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(2)],
            },
        ];
        let mut grammar = GrammarDef {
            rules: original_rules.clone(),
            start: 0,
            terminals: vec![literal(0, b"a"), literal(1, b"b"), literal(2, b"c")],
            ..Default::default()
        };

        bound_runtime_reduction_length(&mut grammar, 2);

        assert_eq!(grammar.rules, original_rules);
    }
}
