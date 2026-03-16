#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::HashMap;

use crate::automata::regex::Expr;
use crate::automata::lexer::regex::parse_regex;
use crate::compiler::compile::build_tokenizer;
use crate::compiler::glr::analysis::{merge_identical_nonterminals, normalize_grammar};
use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Terminal};
use crate::compiler::grammar_def::{Rule, Symbol, TerminalID};
use crate::automata::lexer::tokenizer::Tokenizer;

fn max_runtime_reduction_len_from_env() -> Option<usize> {
    let raw = std::env::var("GLRMASK_MAX_RUNTIME_REDUCTION_LEN").ok()?;
    let value = raw.parse::<usize>().unwrap_or_else(|_| {
        panic!(
            "GLRMASK_MAX_RUNTIME_REDUCTION_LEN must parse as usize, got {raw:?}"
        )
    });
    assert!(
        value >= 2,
        "GLRMASK_MAX_RUNTIME_REDUCTION_LEN must be at least 2; 1 cannot preserve general CFG concatenation"
    );
    Some(value)
}

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
    nullable_terminals: &std::collections::BTreeSet<TerminalID>,
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
    let mut nt_for_terminal = std::collections::BTreeMap::<TerminalID, NonterminalID>::new();
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

fn nullable_terminals_for_grammar(grammar: &GrammarDef) -> std::collections::BTreeSet<TerminalID> {
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
    let mut used = std::collections::BTreeSet::<TerminalID>::new();
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

    let mut remap = std::collections::BTreeMap::<TerminalID, TerminalID>::new();
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
    terminal_names: &std::collections::BTreeMap<TerminalID, String>,
    remap: &std::collections::BTreeMap<TerminalID, TerminalID>,
) -> std::collections::BTreeMap<TerminalID, String> {
    terminal_names
        .iter()
        .filter_map(|(old_id, name)| remap.get(old_id).map(|new_id| (*new_id, name.clone())))
        .collect()
}

pub(crate) fn inline_single_use_nonterminals(
    rules: &mut Vec<Rule>,
    protected_nonterminals: &std::collections::BTreeSet<NonterminalID>,
) {
    loop {
        // Build indexes
        let mut productions_by_lhs = std::collections::BTreeMap::<NonterminalID, Vec<usize>>::new();
        let mut consumer_occurrences = std::collections::BTreeMap::<NonterminalID, usize>::new();

        for (index, rule) in rules.iter().enumerate() {
            productions_by_lhs.entry(rule.lhs).or_default().push(index);
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    *consumer_occurrences.entry(*nonterminal).or_default() += 1;
                }
            }
        }

        // Collect ALL candidates at once
        let mut candidates: std::collections::BTreeMap<NonterminalID, (usize, Vec<Symbol>)> =
            std::collections::BTreeMap::new();

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

            let consumer_count = consumer_occurrences.get(&nonterminal).copied().unwrap_or(0);
            let should_inline = rule.rhs.len() == 1 || consumer_count == 1;
            if !should_inline {
                continue;
            }

            let creates_direct_left_recursion =
                rules.iter().enumerate().any(|(index, outer_rule)| {
                    if index == production_indexes[0] {
                        return false;
                    }
                    outer_rule.rhs.iter().enumerate().any(|(position, symbol)| {
                        matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal)
                            && position == 0
                            && matches!(rule.rhs.first(), Some(Symbol::Nonterminal(first)) if *first == outer_rule.lhs)
                    })
                });
            if creates_direct_left_recursion {
                continue;
            }

            candidates.insert(nonterminal, (production_indexes[0], rule.rhs.clone()));
        }

        if candidates.is_empty() {
            break;
        }

        // Transitively expand candidate RHS: if a candidate's RHS references
        // another candidate, substitute it. Iterate until stable.
        let candidate_nts: std::collections::BTreeSet<NonterminalID> =
            candidates.keys().copied().collect();
        let mut expanded = true;
        while expanded {
            expanded = false;
            let snapshot: Vec<(NonterminalID, Vec<Symbol>)> = candidates
                .iter()
                .map(|(&nt, (_, rhs))| (nt, rhs.clone()))
                .collect();
            for (nt, rhs) in snapshot {
                if rhs.iter().any(|s| {
                    matches!(s, Symbol::Nonterminal(id) if candidate_nts.contains(id) && *id != nt)
                }) {
                    let mut new_rhs = Vec::with_capacity(rhs.len());
                    for symbol in &rhs {
                        if let Symbol::Nonterminal(id) = symbol {
                            if *id != nt {
                                if let Some((_, sub_rhs)) = candidates.get(id) {
                                    new_rhs.extend(sub_rhs.iter().cloned());
                                    continue;
                                }
                            }
                        }
                        new_rhs.push(symbol.clone());
                    }
                    if new_rhs != rhs {
                        candidates.get_mut(&nt).unwrap().1 = new_rhs;
                        expanded = true;
                    }
                }
            }
        }

        // Collect production indexes to remove
        let remove_indexes: std::collections::BTreeSet<usize> =
            candidates.values().map(|(idx, _)| *idx).collect();

        // Rewrite all rules in one pass
        let mut rewritten = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            if remove_indexes.contains(&index) {
                continue;
            }

            let has_candidate = rule.rhs.iter().any(|s| {
                matches!(s, Symbol::Nonterminal(id) if candidates.contains_key(id))
            });

            if has_candidate {
                let mut new_rhs = Vec::with_capacity(rule.rhs.len());
                for symbol in &rule.rhs {
                    if let Symbol::Nonterminal(id) = symbol {
                        if let Some((_, replacement_rhs)) = candidates.get(id) {
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

#[derive(Clone, PartialEq, Eq)]
enum BoundedRepeatTail {
    Next(NonterminalID),
    Base,
}

#[derive(Clone)]
struct BoundedRepeatNodeInfo {
    chunk_nt: NonterminalID,
    chunk_rhs: Vec<Symbol>,
    tail: BoundedRepeatTail,
}

struct BoundedRepeatFamilyBuilder {
    chunk_nt: NonterminalID,
    free_ids: std::vec::IntoIter<NonterminalID>,
    generated_rules: Vec<Rule>,
    pow_ids: std::collections::BTreeMap<usize, NonterminalID>,
    upto_ids: std::collections::BTreeMap<usize, NonterminalID>,
}

impl BoundedRepeatFamilyBuilder {
    fn new(chunk_nt: NonterminalID, reusable_ids: Vec<NonterminalID>) -> Self {
        Self {
            chunk_nt,
            free_ids: reusable_ids.into_iter(),
            generated_rules: Vec::new(),
            pow_ids: std::collections::BTreeMap::new(),
            upto_ids: std::collections::BTreeMap::new(),
        }
    }

    fn alloc(&mut self) -> Option<NonterminalID> {
        self.free_ids.next()
    }

    fn pow_symbol(&mut self, exp: usize) -> Option<Symbol> {
        if exp == 0 {
            Some(Symbol::Nonterminal(self.chunk_nt))
        } else {
            Some(Symbol::Nonterminal(self.ensure_pow(exp)?))
        }
    }

    fn ensure_pow(&mut self, exp: usize) -> Option<NonterminalID> {
        if exp == 0 {
            return Some(self.chunk_nt);
        }
        if let Some(&id) = self.pow_ids.get(&exp) {
            return Some(id);
        }
        let id = self.alloc()?;
        let prev = self.pow_symbol(exp - 1)?;
        self.generated_rules.push(Rule {
            lhs: id,
            rhs: vec![prev.clone(), prev],
        });
        self.pow_ids.insert(exp, id);
        Some(id)
    }

    fn ensure_upto(&mut self, exp: usize) -> Option<NonterminalID> {
        if exp == 0 {
            return Some(self.chunk_nt);
        }
        if let Some(&id) = self.upto_ids.get(&exp) {
            return Some(id);
        }
        let id = self.alloc()?;
        self.emit_upto_into(id, exp)?;
        self.upto_ids.insert(exp, id);
        Some(id)
    }

    fn emit_upto_into(&mut self, target: NonterminalID, exp: usize) -> Option<()> {
        if exp == 0 {
            self.generated_rules.push(Rule {
                lhs: target,
                rhs: vec![Symbol::Nonterminal(self.chunk_nt)],
            });
            return Some(());
        }

        let prev_upto = if exp == 1 {
            Symbol::Nonterminal(self.chunk_nt)
        } else {
            Symbol::Nonterminal(self.ensure_upto(exp - 1)?)
        };
        let current_pow = self.pow_symbol(exp)?;
        self.generated_rules.push(Rule {
            lhs: target,
            rhs: vec![prev_upto.clone()],
        });
        self.generated_rules.push(Rule {
            lhs: target,
            rhs: vec![current_pow.clone(), prev_upto],
        });
        self.generated_rules.push(Rule {
            lhs: target,
            rhs: vec![current_pow],
        });
        Some(())
    }

    fn emit_range_into(&mut self, target: NonterminalID, max_count: usize) -> Option<()> {
        if max_count == 0 {
            return None;
        }
        if max_count == 1 {
            self.generated_rules.push(Rule {
                lhs: target,
                rhs: vec![Symbol::Nonterminal(self.chunk_nt)],
            });
            return Some(());
        }

        let highest_bit = usize::BITS as usize - 1 - max_count.leading_zeros() as usize;
        let largest_power = 1usize << highest_bit;
        if max_count == largest_power {
            let largest_power_symbol = self.pow_symbol(highest_bit)?;
            self.generated_rules.push(Rule {
                lhs: target,
                rhs: vec![largest_power_symbol],
            });
            return Some(());
        }

        if highest_bit == 0 {
            return None;
        }

        let lower_range_symbol = if highest_bit == 1 {
            Symbol::Nonterminal(self.chunk_nt)
        } else {
            Symbol::Nonterminal(self.ensure_upto(highest_bit - 1)?)
        };
        self.generated_rules.push(Rule {
            lhs: target,
            rhs: vec![lower_range_symbol],
        });

        let remainder = max_count - largest_power;
        let remainder_symbol = if remainder == 1 {
            Symbol::Nonterminal(self.chunk_nt)
        } else {
            let remainder_id = self.alloc()?;
            self.emit_range_into(remainder_id, remainder)?;
            Symbol::Nonterminal(remainder_id)
        };
        let largest_power_symbol = self.pow_symbol(highest_bit)?;
        self.generated_rules.push(Rule {
            lhs: target,
            rhs: vec![largest_power_symbol.clone(), remainder_symbol],
        });
        self.generated_rules.push(Rule {
            lhs: target,
            rhs: vec![largest_power_symbol],
        });
        Some(())
    }
}

pub(crate) fn compact_bounded_repeat_ladders(
    rules: &mut Vec<Rule>,
    start: NonterminalID,
    nonterminal_names: &std::collections::BTreeMap<NonterminalID, String>,
) {
    const MIN_FAMILY_LEN: usize = 4;

    loop {
        let mut productions_by_lhs = std::collections::BTreeMap::<NonterminalID, Vec<usize>>::new();
        let mut consumer_occurrences = std::collections::BTreeMap::<NonterminalID, usize>::new();
        for (index, rule) in rules.iter().enumerate() {
            productions_by_lhs.entry(rule.lhs).or_default().push(index);
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    *consumer_occurrences.entry(*nonterminal).or_default() += 1;
                }
            }
        }

        let single_rhs_by_lhs = productions_by_lhs
            .iter()
            .filter_map(|(&lhs, indexes)| {
                if indexes.len() == 1 {
                    Some((lhs, rules[indexes[0]].rhs.clone()))
                } else {
                    None
                }
            })
            .collect::<std::collections::BTreeMap<_, _>>();

        let node_infos = productions_by_lhs
            .iter()
            .filter_map(|(&lhs, indexes)| {
                if lhs == start || nonterminal_names.contains_key(&lhs) || indexes.len() != 2 {
                    return None;
                }

                let first = &rules[indexes[0]].rhs;
                let second = &rules[indexes[1]].rhs;
                let (short_rhs, long_rhs) = if first.len() <= second.len() {
                    (first, second)
                } else {
                    (second, first)
                };

                let [Symbol::Nonterminal(chunk_nt)] = short_rhs.as_slice() else {
                    return None;
                };
                let chunk_rhs = single_rhs_by_lhs.get(chunk_nt)?.clone();
                if chunk_rhs.is_empty() || !long_rhs.starts_with(short_rhs) {
                    return None;
                }

                let tail_rhs = &long_rhs[short_rhs.len()..];
                let tail = if tail_rhs == chunk_rhs.as_slice() || tail_rhs == [Symbol::Nonterminal(*chunk_nt)] {
                    BoundedRepeatTail::Base
                } else if let [Symbol::Nonterminal(next)] = tail_rhs {
                    BoundedRepeatTail::Next(*next)
                } else {
                    return None;
                };

                Some((
                    lhs,
                    BoundedRepeatNodeInfo {
                        chunk_nt: *chunk_nt,
                        chunk_rhs,
                        tail,
                    },
                ))
            })
            .collect::<std::collections::BTreeMap<_, _>>();

        let mut family_incoming = std::collections::BTreeMap::<NonterminalID, usize>::new();
        for info in node_infos.values() {
            if let BoundedRepeatTail::Next(next) = info.tail {
                if node_infos
                    .get(&next)
                    .is_some_and(|child| child.chunk_nt == info.chunk_nt && child.chunk_rhs == info.chunk_rhs)
                {
                    *family_incoming.entry(next).or_default() += 1;
                }
            }
        }

        let candidate_family = node_infos.iter().filter_map(|(&top, info)| {
            if family_incoming.get(&top).copied().unwrap_or(0) != 0 {
                return None;
            }
            if consumer_occurrences.get(&top).copied().unwrap_or(0) == 0 {
                return None;
            }

            let mut chain = vec![top];
            let mut current = top;
            let chunk_nt = info.chunk_nt;
            let chunk_rhs = info.chunk_rhs.clone();

            loop {
                let current_info = node_infos.get(&current)?;
                if current_info.chunk_nt != chunk_nt || current_info.chunk_rhs != chunk_rhs {
                    return None;
                }

                match current_info.tail {
                    BoundedRepeatTail::Base => break,
                    BoundedRepeatTail::Next(next) => {
                        let next_info = node_infos.get(&next)?;
                        if next_info.chunk_nt != chunk_nt || next_info.chunk_rhs != chunk_rhs {
                            return None;
                        }
                        if family_incoming.get(&next).copied().unwrap_or(0) != 1 {
                            return None;
                        }
                        if consumer_occurrences.get(&next).copied().unwrap_or(0) != 1 {
                            return None;
                        }
                        if chain.contains(&next) {
                            return None;
                        }
                        chain.push(next);
                        current = next;
                    }
                }
            }

            if chain.len() < MIN_FAMILY_LEN {
                return None;
            }

            Some((top, chunk_nt, chain))
        }).max_by_key(|(_, _, chain)| chain.len());

        let Some((top, chunk_nt, chain)) = candidate_family else {
            break;
        };

        let family_set = chain.iter().copied().collect::<std::collections::BTreeSet<_>>();
        let mut rebuilt_rules = rules
            .iter()
            .filter(|rule| !family_set.contains(&rule.lhs))
            .cloned()
            .collect::<Vec<_>>();

        let mut builder = BoundedRepeatFamilyBuilder::new(chunk_nt, chain[1..].to_vec());
        if builder.emit_range_into(top, chain.len() + 1).is_none() {
            break;
        }
        rebuilt_rules.extend(builder.generated_rules);
        *rules = rebuilt_rules;
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

pub(crate) fn prepare_grammar_for_compile(grammar: &GrammarDef) -> (GrammarDef, Tokenizer) {
    // Probe nullability against the original terminal set first; nullable
    // terminals are expanded into optional grammar structure before we compact
    // away any terminals that normalization proves unreachable.
    let nullable_terminals = nullable_terminals_for_grammar(grammar);

    let mut normalized = grammar.clone();

    expand_nullable_terminals(&mut normalized.rules, &nullable_terminals);
    normalize_grammar(&mut normalized.rules, normalized.start);
    let protected_nonterminals = normalized
        .nonterminal_names
        .keys()
        .copied()
        .chain(std::iter::once(normalized.start))
        .collect::<std::collections::BTreeSet<_>>();
    inline_single_use_nonterminals(&mut normalized.rules, &protected_nonterminals);
    normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
    compact_bounded_repeat_ladders(&mut normalized.rules, normalized.start, &normalized.nonterminal_names);
    normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
    if let Some(max_rhs_len) = max_runtime_reduction_len_from_env() {
        bound_runtime_reduction_length(&mut normalized, max_rhs_len);
        normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
    }
    compact_unused_terminals(&mut normalized);

    // Build the real tokenizer only from the compacted live terminal set so
    // dead terminals never make it into downstream lexer/parser stages.
    let mut tokenizer = build_tokenizer(&normalized);
    let _ = tokenizer.drain_nullable_terminals();

    (normalized, tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::Expr;
    use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Rule, Symbol, Terminal};

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

    #[test]
    fn test_expand_nullable_terminals_no_nullables() {
        let gdef = crate::compiler::grammar::model::tests::simple_ab_grammar();
        let nullable = std::collections::BTreeSet::new();
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);
        assert_eq!(rules.len(), gdef.rules.len());
        assert_eq!(rules[0].rhs, gdef.rules[0].rhs);
    }

    #[test]
    fn test_expand_nullable_terminals_single_nullable() {
        let gdef = crate::compiler::grammar::model::tests::simple_ab_grammar();
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        assert_eq!(rules.len(), 3);
        let fresh_nt = gdef.num_nonterminals();
        assert_eq!(rules[0].lhs, 0);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Terminal(1)]
        );

        let fresh_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);
        let rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            fresh_rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(rhs_set.contains(&vec![]));
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0)]));
    }

    #[test]
    fn test_expand_nullable_terminals_both_nullable() {
        let gdef = crate::compiler::grammar::model::tests::simple_ab_grammar();
        let nullable = std::collections::BTreeSet::from([0u32, 1u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        assert_eq!(rules.len(), 5);

        let nt0 = gdef.num_nonterminals();
        let nt1 = gdef.num_nonterminals() + 1;
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(nt0), Symbol::Nonterminal(nt1)]
        );
    }

    #[test]
    fn test_expand_nullable_terminals_nonterminal_untouched() {
        let gdef = crate::compiler::grammar::model::tests::two_nt_grammar();
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals();

        let s_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(s_rules.len(), 1);
        assert_eq!(
            s_rules[0].rhs,
            vec![Symbol::Nonterminal(1), Symbol::Terminal(1)]
        );

        let n1_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 1).collect();
        assert_eq!(n1_rules.len(), 1);
        assert_eq!(n1_rules[0].rhs, vec![Symbol::Nonterminal(fresh_nt)]);

        let fresh_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);
    }

    #[test]
    fn test_expand_nullable_terminals_multiple_occurrences() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals();
        assert_eq!(rules.len(), 3);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Nonterminal(fresh_nt)]
        );
    }

    #[test]
    fn test_compact_unused_terminals_remaps_rules_and_terminal_ids() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"dead".to_vec() },
                Terminal::Literal { id: 2, bytes: b"b".to_vec() },
            ],
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }]
        );
        assert_eq!(grammar.terminals.len(), 2);
        assert_eq!(grammar.terminals[0].id(), 0);
        assert_eq!(grammar.terminals[1].id(), 1);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        assert_eq!(grammar.ignore_terminal, None);
    }

    #[test]
    fn test_compact_unused_terminals_preserves_ignore_terminal_and_remaps_it() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"dead".to_vec() },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal { id: 3, bytes: b"b".to_vec() },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }]
        );
        assert_eq!(grammar.terminals.len(), 3);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), " +");
        assert_eq!(grammar.terminals[2].name(), "b");
        assert_eq!(grammar.ignore_terminal, Some(1));
    }

    #[test]
    fn test_compact_unused_terminals_merges_identical_terminals() {
        let mut grammar = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(2)] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() },
                Terminal::Literal { id: 2, bytes: b"a".to_vec() },
            ],
            nonterminal_names: std::collections::BTreeMap::new(),
            terminal_names: std::collections::BTreeMap::new(),
            ignore_terminal: None,
        };
        compact_unused_terminals(&mut grammar);
        assert_eq!(grammar.terminals.len(), 2);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        assert_eq!(grammar.rules[0].rhs, vec![Symbol::Terminal(0), Symbol::Terminal(1)]);
        assert_eq!(grammar.rules[1].rhs, vec![Symbol::Terminal(0)]);
    }

    #[test]
    fn test_prepare_grammar_for_compile_retains_and_remaps_names() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"dead".to_vec() },
                Terminal::Literal { id: 2, bytes: b"b".to_vec() },
            ],
            nonterminal_names: std::collections::BTreeMap::from([(0, "start".to_string())]),
            terminal_names: std::collections::BTreeMap::from([
                (0, "A".to_string()),
                (1, "DEAD".to_string()),
                (2, "B".to_string()),
            ]),
            ignore_terminal: None,
        };

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&grammar);

        assert_eq!(normalized.nonterminal_names.get(&0).map(String::as_str), Some("start"));
        assert_eq!(normalized.terminal_names.get(&0).map(String::as_str), Some("A"));
        assert_eq!(normalized.terminal_names.get(&1).map(String::as_str), Some("B"));
        assert!(!normalized.terminal_names.values().any(|name| name == "DEAD"));
    }

    #[test]
    fn test_drain_nullable_terminals_from_tokenizer() {
        let exprs = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),
        ];
        let mut tok = crate::compiler::compile::build_tokenizer_from_exprs(&exprs);

        assert!(tok.matched_terminals(tok.start_state()).contains(&0));

        let nullable = tok.drain_nullable_terminals();
        assert_eq!(nullable, std::collections::BTreeSet::from([0u32]));
        assert!(!tok.matched_terminals(tok.start_state()).contains(&0));
    }

    #[test]
    fn test_inline_single_use_nonterminals_compacts_repetition_tail_chain() {
        let mut rules = vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(3)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(1), Symbol::Terminal(1)] },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(4),
                    Symbol::Terminal(1),
                ],
            },
            Rule { lhs: 4, rhs: vec![Symbol::Nonterminal(5)] },
            Rule { lhs: 4, rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)] },
            Rule { lhs: 5, rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(7)] },
            Rule { lhs: 6, rhs: vec![Symbol::Terminal(2)] },
            Rule { lhs: 7, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 8, rhs: vec![Symbol::Nonterminal(9)] },
            Rule { lhs: 8, rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)] },
            Rule { lhs: 9, rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(2)] },
            Rule {
                lhs: 10,
                rhs: vec![
                    Symbol::Terminal(3),
                    Symbol::Nonterminal(2),
                    Symbol::Nonterminal(8),
                    Symbol::Terminal(4),
                ],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "json_kv".to_string()),
            (2, "json_value".to_string()),
            (3, "json_object".to_string()),
            (10, "json_array".to_string()),
        ]);
        let protected: std::collections::BTreeSet<NonterminalID> =
            names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(!rules.iter().any(|rule| matches!(rule.lhs, 6 | 7)));
        assert!(rules.contains(&Rule {
            lhs: 5,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(1)],
        }));
        assert!(rules.contains(&Rule { lhs: 4, rhs: vec![Symbol::Nonterminal(5)] }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 9,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule { lhs: 8, rhs: vec![Symbol::Nonterminal(9)] }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
        }));
    }

    #[test]
    fn test_inline_single_use_nonterminals_keeps_multi_symbol_helper_with_multiple_occurrences() {
        let mut rules = vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)] },
            Rule { lhs: 2, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(3)] },
            Rule { lhs: 3, rhs: vec![Symbol::Terminal(1)] },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "root".to_string()),
        ]);
        let protected: std::collections::BTreeSet<NonterminalID> =
            names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(rules.iter().any(|rule| rule.lhs == 2));
        assert!(rules.contains(&Rule {
            lhs: 1,
            rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 2,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
        }));
    }

    fn derivable_chunk_counts(
        rules: &[Rule],
        nonterminal: NonterminalID,
        chunk_nt: NonterminalID,
        memo: &mut std::collections::BTreeMap<NonterminalID, std::collections::BTreeSet<usize>>,
    ) -> std::collections::BTreeSet<usize> {
        if let Some(existing) = memo.get(&nonterminal) {
            return existing.clone();
        }
        if nonterminal == chunk_nt {
            return std::collections::BTreeSet::from([1]);
        }

        let mut result = std::collections::BTreeSet::new();
        for rule in rules.iter().filter(|rule| rule.lhs == nonterminal) {
            let mut totals = std::collections::BTreeSet::from([0usize]);
            for symbol in &rule.rhs {
                let counts = match symbol {
                    Symbol::Terminal(_) => std::collections::BTreeSet::from([0usize]),
                    Symbol::Nonterminal(id) => derivable_chunk_counts(rules, *id, chunk_nt, memo),
                };
                let mut next_totals = std::collections::BTreeSet::new();
                for left in &totals {
                    for right in &counts {
                        next_totals.insert(left + right);
                    }
                }
                totals = next_totals;
            }
            result.extend(totals);
        }

        memo.insert(nonterminal, result.clone());
        result
    }

    #[test]
    fn test_compact_bounded_repeat_ladders_rewrites_linear_family() {
        let chunk_nt = 20;
        let family = [30, 31, 32, 33, 34, 35, 36, 37];
        let mut rules = vec![
            Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(family[0])] },
            Rule { lhs: chunk_nt, rhs: vec![Symbol::Terminal(0)] },
        ];
        for (index, lhs) in family.iter().copied().enumerate() {
            rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(chunk_nt)] });
            let long_rhs = if index + 1 == family.len() {
                vec![Symbol::Nonterminal(chunk_nt), Symbol::Nonterminal(chunk_nt)]
            } else {
                vec![Symbol::Nonterminal(chunk_nt), Symbol::Nonterminal(family[index + 1])]
            };
            rules.push(Rule { lhs, rhs: long_rhs });
        }

        let original_family_rule_count = rules.iter().filter(|rule| family.contains(&rule.lhs)).count();
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "root".to_string()),
        ]);

        compact_bounded_repeat_ladders(&mut rules, 0, &names);

        let rewritten_family_rule_count = rules.iter().filter(|rule| family.contains(&rule.lhs)).count();
        assert!(rewritten_family_rule_count < original_family_rule_count);
        assert_eq!(rules.iter().filter(|rule| rule.lhs == family[0]).count(), 3);

        let mut memo = std::collections::BTreeMap::new();
        let counts = derivable_chunk_counts(&rules, family[0], chunk_nt, &mut memo);
        assert_eq!(counts, (1..=family.len() + 1).collect::<std::collections::BTreeSet<_>>());
    }
}
