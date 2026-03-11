#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::compiler::compile::build_tokenizer;
use crate::compiler::glr::analysis::merge_identical_nonterminals;
use crate::compiler::glr::analysis::normalize_grammar;
use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Terminal};
use crate::compiler::grammar_def::{Rule, Symbol, TerminalID};

#[derive(Debug, Clone)]
pub(crate) struct NonterminalAllocator {
    next_id: NonterminalID,
}

impl NonterminalAllocator {
    pub(crate) fn new(next_id: NonterminalID) -> Self {
        Self { next_id }
    }

    fn fresh(&mut self) -> NonterminalID {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Rewrite grammar rules so that nullable terminals (those matching the empty
/// string) are treated as optional. Operates in place on owned rule data.
pub(crate) fn expand_nullable_terminals(
    rules: &mut Vec<Rule>,
    allocator: &mut NonterminalAllocator,
    nullable_terminals: &std::collections::BTreeSet<TerminalID>,
) {
    if nullable_terminals.is_empty() {
        return;
    }

    let mut nt_for_terminal = std::collections::BTreeMap::<TerminalID, NonterminalID>::new();
    let mut extra_rules = Vec::new();

    for &tid in nullable_terminals {
        let fresh_nt = allocator.fresh();
        nt_for_terminal.insert(tid, fresh_nt);

        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![],
        });
        extra_rules.push(Rule {
            lhs: fresh_nt,
            rhs: vec![Symbol::Terminal(tid)],
        });
    }

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

#[derive(Debug, Clone)]
pub(crate) struct LiveTerminalRemap {
    pub(crate) terminals: Vec<Terminal>,
    pub(crate) ignore_terminal: Option<TerminalID>,
    pub(crate) remap: std::collections::BTreeMap<TerminalID, TerminalID>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TerminalIdentity {
    Literal {
        bytes: Vec<u8>,
        is_ignore: bool,
    },
    Pattern {
        pattern: String,
        utf8: bool,
        is_ignore: bool,
    },
    Expr {
        expr: Expr,
        is_ignore: bool,
    },
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

fn expr_accepts_empty(expr: &Expr) -> bool {
    match expr {
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false,
        Expr::Seq(parts) => parts.iter().all(expr_accepts_empty),
        Expr::Choice(options) => options.iter().any(expr_accepts_empty),
        Expr::Repeat { min, .. } => *min == 0,
        Expr::Shared(inner) => expr_accepts_empty(inner),
        Expr::Epsilon => true,
    }
}

pub(crate) fn nullable_terminal_ids(
    grammar: &GrammarDef,
) -> std::collections::BTreeSet<TerminalID> {
    grammar
        .terminals
        .iter()
        .filter_map(|terminal| {
            let accepts_empty = match terminal {
                Terminal::Literal { bytes, .. } => bytes.is_empty(),
                Terminal::Pattern { pattern, utf8, .. } => {
                    expr_accepts_empty(&parse_regex(pattern, *utf8))
                }
                Terminal::Expr { expr, .. } => expr_accepts_empty(expr),
            };
            accepts_empty.then_some(terminal.id())
        })
        .collect()
}

pub(crate) fn remap_live_terminals(
    rules: &mut Vec<Rule>,
    terminals: &[Terminal],
    ignore_terminal: Option<TerminalID>,
) -> LiveTerminalRemap {
    let mut used = std::collections::BTreeSet::<TerminalID>::new();
    for rule in rules.iter() {
        for symbol in &rule.rhs {
            if let Symbol::Terminal(terminal_id) = symbol {
                used.insert(*terminal_id);
            }
        }
    }
    if let Some(ignore_terminal) = ignore_terminal {
        used.insert(ignore_terminal);
    }

    let mut remap = std::collections::BTreeMap::<TerminalID, TerminalID>::new();
    let mut compacted = Vec::with_capacity(used.len());
    let mut canonical_ids = std::collections::HashMap::<TerminalIdentity, TerminalID>::new();

    for old_id in used {
        let terminal = terminals.get(old_id as usize).unwrap_or_else(|| {
            panic!(
                "terminal id {} referenced by a rule but missing from grammar.terminals",
                old_id
            )
        });
        let identity = terminal_identity(terminal, ignore_terminal == Some(old_id));
        if let Some(&existing_id) = canonical_ids.get(&identity) {
            remap.insert(old_id, existing_id);
            continue;
        }

        let new_id = compacted.len() as TerminalID;
        canonical_ids.insert(identity, new_id);
        remap.insert(old_id, new_id);
        compacted.push(remap_terminal_id(terminal, new_id));
    }

    for rule in rules.iter_mut() {
        for symbol in rule.rhs.iter_mut() {
            if let Symbol::Terminal(terminal_id) = symbol {
                *terminal_id = *remap
                    .get(terminal_id)
                    .expect("used terminal must have been assigned a compacted id");
            }
        }
    }

    let remapped_ignore_terminal = ignore_terminal.and_then(|old_id| remap.get(&old_id).copied());

    LiveTerminalRemap {
        terminals: compacted,
        ignore_terminal: remapped_ignore_terminal,
        remap,
    }
}

pub(crate) fn remap_terminal_names(
    terminal_names: &std::collections::BTreeMap<TerminalID, String>,
    remap: &std::collections::BTreeMap<TerminalID, TerminalID>,
) -> std::collections::BTreeMap<TerminalID, String> {
    let mut remapped = std::collections::BTreeMap::new();
    for (old_id, name) in terminal_names {
        if let Some(&new_id) = remap.get(old_id) {
            remapped.entry(new_id).or_insert_with(|| name.clone());
        }
    }
    remapped
}

pub(crate) fn inline_autogenerated_helper_nonterminals(
    rules: &mut Vec<Rule>,
    start: NonterminalID,
    nonterminal_names: &std::collections::BTreeMap<NonterminalID, String>,
) {
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

        let candidate = productions_by_lhs
            .iter()
            .filter_map(|(&nonterminal, production_indexes)| {
                if nonterminal == start
                    || nonterminal_names.contains_key(&nonterminal)
                    || production_indexes.len() != 1
                {
                    return None;
                }

                let rule = &rules[production_indexes[0]];
                if rule.rhs.is_empty()
                    || rule
                        .rhs
                        .iter()
                        .any(|symbol| matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal))
                {
                    return None;
                }

                let consumer_count = consumer_occurrences.get(&nonterminal).copied().unwrap_or(0);
                let should_inline = rule.rhs.len() == 1 || consumer_count == 1;
                if !should_inline {
                    return None;
                }

                let creates_direct_left_recursion = rules.iter().enumerate().any(|(index, outer_rule)| {
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
                    return None;
                }

                Some((nonterminal, production_indexes[0], rule.rhs.clone()))
            })
            .next();

        let Some((nonterminal, production_index, replacement_rhs)) = candidate else {
            break;
        };

        let mut rewritten = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            if index == production_index {
                continue;
            }

            let mut changed = false;
            let mut new_rhs = Vec::with_capacity(rule.rhs.len());
            for symbol in &rule.rhs {
                if matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal) {
                    new_rhs.extend(replacement_rhs.iter().cloned());
                    changed = true;
                } else {
                    new_rhs.push(symbol.clone());
                }
            }

            if changed {
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
                let tail = if tail_rhs == chunk_rhs.as_slice()
                    || tail_rhs == [Symbol::Nonterminal(*chunk_nt)]
                {
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
                if node_infos.get(&next).is_some_and(|child| {
                    child.chunk_nt == info.chunk_nt && child.chunk_rhs == info.chunk_rhs
                }) {
                    *family_incoming.entry(next).or_default() += 1;
                }
            }
        }

        let candidate_family = node_infos
            .iter()
            .filter_map(|(&top, info)| {
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
            })
            .max_by_key(|(_, _, chain)| chain.len());

        let Some((top, chunk_nt, chain)) = candidate_family else {
            break;
        };

        let family_set = chain
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
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

pub(crate) fn prepare_grammar_for_compile(grammar: &GrammarDef) -> (GrammarDef, Tokenizer) {
    let nullable_terminals = nullable_terminal_ids(grammar);

    let mut rules = grammar.rules.clone();
    let start = grammar.start;
    let mut nonterminals = NonterminalAllocator::new(grammar.num_nonterminals());

    expand_nullable_terminals(&mut rules, &mut nonterminals, &nullable_terminals);
    normalize_grammar(&mut rules, start);
    inline_autogenerated_helper_nonterminals(&mut rules, start, &grammar.nonterminal_names);
    rules = merge_identical_nonterminals(&rules, start);
    compact_bounded_repeat_ladders(&mut rules, start, &grammar.nonterminal_names);
    rules = merge_identical_nonterminals(&rules, start);

    let live_terminals =
        remap_live_terminals(&mut rules, &grammar.terminals, grammar.ignore_terminal);
    let normalized = GrammarDef {
        rules,
        start,
        terminals: live_terminals.terminals,
        nonterminal_names: grammar.nonterminal_names.clone(),
        terminal_names: remap_terminal_names(&grammar.terminal_names, &live_terminals.remap),
        ignore_terminal: live_terminals.ignore_terminal,
    };

    let mut tokenizer = build_tokenizer(&normalized);
    let _ = tokenizer.drain_nullable_terminals();

    (normalized, tokenizer)
}
