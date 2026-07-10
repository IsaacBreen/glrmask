use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use rustc_hash::FxHashSet;

use crate::automata::regex::Expr;
use crate::automata::lexer::regex::parse_regex;
use crate::compiler::glr::analysis::{
    eliminate_right_recursion, has_indirect_left_recursion, merge_identical_nonterminals,
    normalize_grammar,
};
use crate::grammar::flat::{GrammarDef, NonterminalID, Terminal};
use crate::grammar::flat::{Rule, Symbol, TerminalID};

const MAX_RUNTIME_REDUCTION_LEN: usize = 5;
const INLINE_PROTECTED_NONTERMINALS_ENV: &str = "GLRMASK_INLINE_PROTECTED_NONTERMINALS";

fn env_var_enabled(key: &str, default: bool) -> bool {
    std::env::var(key)
        .map(|v| {
            let n = v.trim().to_ascii_lowercase();
            !matches!(n.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn emit_grammar_transform_profile(
    stage: &str,
    elapsed_ms: f64,
    rules_before: usize,
    rules_after: usize,
    extra: &str,
) {
    eprintln!(
        "[glrmask-profile] grammar_transform stage={} ms={:.3} rules_before={} rules_after={}{}",
        stage,
        elapsed_ms,
        rules_before,
        rules_after,
        extra,
    );
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

struct SingleUseInlineIndexes {
    only_production_index: Vec<Option<usize>>,
    has_multiple_productions: Vec<bool>,
    use_counts: Vec<usize>,
    sole_user_index: Vec<Option<usize>>,
    position_0_edges: FxHashSet<(NonterminalID, NonterminalID)>,
}

impl SingleUseInlineIndexes {
    fn build(rules: &[Rule]) -> Self {
        let max_nonterminal = rules
            .iter()
            .flat_map(|rule| {
                std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                    Symbol::Terminal(_) => None,
                }))
            })
            .max()
            .unwrap_or(0) as usize;
        let len = max_nonterminal + 1;
        let mut only_production_index = vec![None; len];
        let mut has_multiple_productions = vec![false; len];
        let mut use_counts = vec![0usize; len];
        let mut sole_user_index = vec![None; len];
        let mut position_0_edges = FxHashSet::default();
        position_0_edges.reserve(rules.len());

        for (index, rule) in rules.iter().enumerate() {
            let lhs = rule.lhs as usize;
            if only_production_index[lhs].replace(index).is_some() {
                has_multiple_productions[lhs] = true;
            }
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    let nonterminal = *nonterminal as usize;
                    use_counts[nonterminal] += 1;
                    if use_counts[nonterminal] == 1 {
                        sole_user_index[nonterminal] = Some(index);
                    } else {
                        sole_user_index[nonterminal] = None;
                    }
                }
            }
            if let Some(Symbol::Nonterminal(first_nonterminal)) = rule.rhs.first() {
                position_0_edges.insert((*first_nonterminal, rule.lhs));
            }
        }

        Self {
            only_production_index,
            has_multiple_productions,
            use_counts,
            sole_user_index,
            position_0_edges,
        }
    }

    fn single_production(&self, nonterminal: NonterminalID) -> Option<usize> {
        let index = nonterminal as usize;
        (!self.has_multiple_productions[index])
            .then_some(self.only_production_index[index])
            .flatten()
    }

    fn use_count(&self, nonterminal: NonterminalID) -> usize {
        self.use_counts[nonterminal as usize]
    }

    fn sole_user(&self, nonterminal: NonterminalID) -> Option<usize> {
        self.sole_user_index[nonterminal as usize]
    }

    fn creates_direct_left_recursion(
        &self,
        nonterminal: NonterminalID,
        replacement_first: NonterminalID,
    ) -> bool {
        self.position_0_edges
            .contains(&(nonterminal, replacement_first))
    }

    fn nonterminals(&self) -> impl Iterator<Item = NonterminalID> + '_ {
        (0..self.only_production_index.len()).map(|index| index as NonterminalID)
    }
}

pub(crate) fn inline_single_use_nonterminals(
    rules: &mut Vec<Rule>,
    protected_nonterminals: &BTreeSet<NonterminalID>,
) {
    let inline_protected_nonterminals = env_var_enabled(INLINE_PROTECTED_NONTERMINALS_ENV, true);

    loop {
        let indexes = SingleUseInlineIndexes::build(rules);
        let mut inline_candidates = BTreeMap::<NonterminalID, (usize, Vec<Symbol>)>::new();

        for nonterminal in indexes.nonterminals() {
            let Some(production_index) = indexes.single_production(nonterminal) else {
                continue;
            };
            let rule = &rules[production_index];
            if rule.rhs.is_empty()
                || rule
                    .rhs
                    .iter()
                    .any(|symbol| matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal))
            {
                continue;
            }
            let use_count = indexes.use_count(nonterminal);
            if use_count == 0 {
                continue;
            }
            if protected_nonterminals.contains(&nonterminal) && !inline_protected_nonterminals {
                continue;
            }
            if rule.rhs.len() != 1 && use_count != 1 {
                continue;
            }
            if let Some(Symbol::Nonterminal(first)) = rule.rhs.first()
                && indexes.creates_direct_left_recursion(nonterminal, *first)
            {
                continue;
            }
            inline_candidates.insert(nonterminal, (production_index, rule.rhs.clone()));
        }

        if inline_candidates.is_empty() {
            break;
        }
        remove_cyclic_inline_candidates(&mut inline_candidates);
        if inline_candidates.is_empty() {
            break;
        }

        expand_inline_candidates(&mut inline_candidates);

        let remove_indexes: BTreeSet<usize> =
            inline_candidates.values().map(|(index, _)| *index).collect();
        let mut rewritten = Vec::with_capacity(rules.len());
        for (index, rule) in rules.iter().enumerate() {
            if remove_indexes.contains(&index) {
                continue;
            }
            let has_candidate = rule.rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(id) if inline_candidates.contains_key(id))
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

fn remove_cyclic_inline_candidates(
    inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
) {
    if inline_candidates.is_empty() {
        return;
    }

    // Preserve the old per-candidate reachability semantics, but avoid building
    // ordered dependency maps and allocating a new visited set for each start.
    // Candidate IDs are grammar nonterminal IDs, so one dense index plus visit
    // stamps gives the same search with substantially less bookkeeping.
    let candidate_ids: Vec<NonterminalID> = inline_candidates.keys().copied().collect();
    let max_nonterminal = candidate_ids.last().copied().unwrap_or(0) as usize;
    let mut candidate_index_by_nonterminal = vec![usize::MAX; max_nonterminal + 1];
    for (candidate_index, &nonterminal) in candidate_ids.iter().enumerate() {
        candidate_index_by_nonterminal[nonterminal as usize] = candidate_index;
    }

    let mut visited_at = vec![0usize; candidate_ids.len()];
    let mut search_stamp = 0usize;
    let mut stack = Vec::new();
    let mut cyclic = vec![false; candidate_ids.len()];

    for (start_index, &start_nonterminal) in candidate_ids.iter().enumerate() {
        search_stamp = search_stamp.checked_add(1).expect("cycle-search stamp overflow");
        stack.clear();
        stack.push(start_index);
        visited_at[start_index] = search_stamp;

        let mut reaches_start = false;
        while let Some(current_index) = stack.pop() {
            let current_nonterminal = candidate_ids[current_index];
            let (_, rhs) = inline_candidates
                .get(&current_nonterminal)
                .expect("candidate id must be present");
            for symbol in rhs {
                let Symbol::Nonterminal(child) = symbol else {
                    continue;
                };
                if *child == start_nonterminal {
                    reaches_start = true;
                    break;
                }
                let child_index = candidate_index_by_nonterminal
                    .get(*child as usize)
                    .copied()
                    .unwrap_or(usize::MAX);
                if child_index != usize::MAX && visited_at[child_index] != search_stamp {
                    visited_at[child_index] = search_stamp;
                    stack.push(child_index);
                }
            }
            if reaches_start {
                break;
            }
        }

        cyclic[start_index] = reaches_start;
    }

    for (candidate_index, nonterminal) in candidate_ids.into_iter().enumerate() {
        if cyclic[candidate_index] {
            inline_candidates.remove(&nonterminal);
        }
    }
}


/// Expand acyclic inline candidates bottom-up.
///
/// Candidate cycles are removed before this runs.  The old implementation
/// repeatedly cloned every candidate map snapshot until no replacement was
/// possible.  Scheduling candidates after their dependencies computes the same
/// transitive substitutions once per candidate, without recursive traversal.
fn expand_inline_candidates(
    inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
) {
    if inline_candidates.is_empty() {
        return;
    }

    let candidate_ids: Vec<NonterminalID> = inline_candidates.keys().copied().collect();
    let max_nonterminal = candidate_ids.last().copied().unwrap_or(0) as usize;
    let mut candidate_index_by_nonterminal = vec![usize::MAX; max_nonterminal + 1];
    for (candidate_index, &nonterminal) in candidate_ids.iter().enumerate() {
        candidate_index_by_nonterminal[nonterminal as usize] = candidate_index;
    }

    let mut remaining_dependencies = vec![0usize; candidate_ids.len()];
    let mut dependents = vec![Vec::<usize>::new(); candidate_ids.len()];
    for (candidate_index, &nonterminal) in candidate_ids.iter().enumerate() {
        let (_, rhs) = inline_candidates
            .get(&nonterminal)
            .expect("candidate id must be present");
        let mut dependencies = Vec::new();
        for symbol in rhs {
            let Symbol::Nonterminal(child) = symbol else {
                continue;
            };
            let child_index = candidate_index_by_nonterminal
                .get(*child as usize)
                .copied()
                .unwrap_or(usize::MAX);
            // Self references are intentionally left untouched, matching the
            // legacy loop. They should already have been removed as cycles.
            if child_index != usize::MAX
                && child_index != candidate_index
                && !dependencies.contains(&child_index)
            {
                dependencies.push(child_index);
            }
        }
        remaining_dependencies[candidate_index] = dependencies.len();
        for dependency in dependencies {
            dependents[dependency].push(candidate_index);
        }
    }

    let mut ready: Vec<usize> = remaining_dependencies
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, &count)| (count == 0).then_some(candidate_index))
        .collect();
    let mut expanded_rhs: Vec<Option<Vec<Symbol>>> = vec![None; candidate_ids.len()];
    let mut expanded_count = 0usize;

    while let Some(candidate_index) = ready.pop() {
        let nonterminal = candidate_ids[candidate_index];
        let (_, rhs) = inline_candidates
            .get(&nonterminal)
            .expect("candidate id must be present");
        let mut expanded = Vec::with_capacity(rhs.len());
        for symbol in rhs {
            let replacement_index = match symbol {
                Symbol::Nonterminal(child) => candidate_index_by_nonterminal
                    .get(*child as usize)
                    .copied()
                    .filter(|&index| index != usize::MAX && index != candidate_index),
                Symbol::Terminal(_) => None,
            };
            if let Some(replacement_index) = replacement_index {
                let replacement = expanded_rhs[replacement_index]
                    .as_ref()
                    .expect("dependency must be expanded before its dependent");
                expanded.extend(replacement.iter().cloned());
            } else {
                expanded.push(symbol.clone());
            }
        }
        expanded_rhs[candidate_index] = Some(expanded);
        expanded_count += 1;

        for &dependent in &dependents[candidate_index] {
            remaining_dependencies[dependent] -= 1;
            if remaining_dependencies[dependent] == 0 {
                ready.push(dependent);
            }
        }
    }

    if expanded_count != candidate_ids.len() {
        // `remove_cyclic_inline_candidates` should make this unreachable. Keep
        // the legacy iterative algorithm as an exact fallback if a malformed
        // candidate graph somehow violates that invariant.
        expand_inline_candidates_iteratively(inline_candidates);
        return;
    }

    for (candidate_index, nonterminal) in candidate_ids.into_iter().enumerate() {
        inline_candidates
            .get_mut(&nonterminal)
            .expect("candidate id must be present")
            .1 = expanded_rhs[candidate_index]
            .take()
            .expect("all candidates must have expanded RHS");
    }
}

fn expand_inline_candidates_iteratively(
    inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
) {
    let inline_candidate_ids: BTreeSet<NonterminalID> =
        inline_candidates.keys().copied().collect();
    let mut expanded = true;
    while expanded {
        expanded = false;
        let snapshot: Vec<(NonterminalID, Vec<Symbol>)> = inline_candidates
            .iter()
            .map(|(&nonterminal, (_, rhs))| (nonterminal, rhs.clone()))
            .collect();
        for (nonterminal, rhs) in snapshot {
            if rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(id) if inline_candidate_ids.contains(id) && *id != nonterminal)
            }) {
                let mut new_rhs = Vec::with_capacity(rhs.len());
                for symbol in &rhs {
                    if let Symbol::Nonterminal(id) = symbol {
                        if *id != nonterminal {
                            if let Some((_, replacement_rhs)) = inline_candidates.get(id) {
                                new_rhs.extend(replacement_rhs.iter().cloned());
                                continue;
                            }
                        }
                    }
                    new_rhs.push(symbol.clone());
                }
                if new_rhs != rhs {
                    inline_candidates
                        .get_mut(&nonterminal)
                        .expect("inline candidate should still exist")
                        .1 = new_rhs;
                    expanded = true;
                }
            }
        }
    }
}

fn inline_post_bound_single_use_nonterminals(
    rules: &mut Vec<Rule>,
    protected_nonterminals: &BTreeSet<NonterminalID>,
    max_rhs_len: usize,
) {
    let inline_protected_nonterminals = env_var_enabled(INLINE_PROTECTED_NONTERMINALS_ENV, true);

    loop {
        let indexes = SingleUseInlineIndexes::build(rules);
        let mut candidate = None;

        for nonterminal in indexes.nonterminals() {
            let Some(candidate_rule_index) = indexes.single_production(nonterminal) else {
                continue;
            };
            if indexes.use_count(nonterminal) != 1 {
                continue;
            }
            if protected_nonterminals.contains(&nonterminal) && !inline_protected_nonterminals {
                continue;
            }
            let candidate_rule = &rules[candidate_rule_index];
            if candidate_rule.rhs.is_empty()
                || candidate_rule
                    .rhs
                    .iter()
                    .any(|symbol| matches!(symbol, Symbol::Nonterminal(id) if *id == nonterminal))
            {
                continue;
            }
            if let Some(Symbol::Nonterminal(first)) = candidate_rule.rhs.first()
                && indexes.creates_direct_left_recursion(nonterminal, *first)
            {
                continue;
            }

            let user_index = indexes
                .sole_user(nonterminal)
                .expect("one use must have a recorded user rule");
            let user_rule = &rules[user_index];
            let mut new_rhs = Vec::with_capacity(user_rule.rhs.len() + candidate_rule.rhs.len());
            for symbol in &user_rule.rhs {
                if let Symbol::Nonterminal(id) = symbol
                    && *id == nonterminal
                {
                    new_rhs.extend(candidate_rule.rhs.iter().cloned());
                    continue;
                }
                new_rhs.push(symbol.clone());
            }
            if new_rhs.len() > max_rhs_len {
                continue;
            }
            candidate = Some((candidate_rule_index, user_index, new_rhs));
            break;
        }

        let Some((candidate_rule_index, user_index, new_rhs)) = candidate else {
            break;
        };
        let mut rewritten = Vec::with_capacity(rules.len().saturating_sub(1));
        for (index, rule) in rules.iter().enumerate() {
            if index == candidate_rule_index {
                continue;
            }
            if index == user_index {
                rewritten.push(Rule {
                    lhs: rule.lhs,
                    rhs: new_rhs.clone(),
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
        let mut symbols = rule.rhs;
        let mut lhs = rule.lhs;

        if symbols[0] == Symbol::Nonterminal(lhs) {
            let tail_nt = next_nt;
            next_nt += 1;
            grammar.nonterminal_names.insert(tail_nt, format!("{lhs_name}__tail"));

            rewritten.push(Rule {
                lhs,
                rhs: vec![Symbol::Nonterminal(lhs), Symbol::Nonterminal(tail_nt)],
            });

            symbols = symbols[1..].to_vec();
            lhs = tail_nt;

            if symbols.len() <= max_rhs_len {
                rewritten.push(Rule { lhs, rhs: symbols });
                continue;
            }
        }

        let first_chunk_len = max_rhs_len.min(symbols.len());
        let mut consumed = first_chunk_len;
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
            rhs: symbols[..first_chunk_len].to_vec(),
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
            lhs,
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

/// Run only the grammar transforms without building the tokenizer.
/// The caller is responsible for calling `build_tokenizer` on the result.
pub(crate) fn prepare_grammar_transforms_only(grammar: GrammarDef) -> GrammarDef {
    let profiling = compile_profile_enabled();
    let nullable_rules_before = grammar.rules.len();
    let nullable_started_at = profiling.then(Instant::now);
    let nullable_terminals = nullable_terminals_for_grammar(&grammar);
    if let Some(started_at) = nullable_started_at {
        emit_grammar_transform_profile(
            "nullable_terminals_for_grammar",
            elapsed_ms(started_at),
            nullable_rules_before,
            nullable_rules_before,
            &format!(" nullable_terminals={}", nullable_terminals.len()),
        );
    }
    let mut normalized = grammar;
    prepare_grammar_transforms_impl(&mut normalized, &nullable_terminals, profiling);
    std::mem::take(&mut normalized)
}

/// The shared grammar transform steps (without tokenizer build).
fn prepare_grammar_transforms_impl(
    normalized: &mut GrammarDef,
    nullable_terminals: &BTreeSet<TerminalID>,
    profiling: bool,
) {
    let expand_rules_before = normalized.rules.len();
    let expand_started_at = profiling.then(Instant::now);
    expand_nullable_terminals(&mut normalized.rules, nullable_terminals);
    if let Some(started_at) = expand_started_at {
        emit_grammar_transform_profile(
            "expand_nullable_terminals",
            elapsed_ms(started_at),
            expand_rules_before,
            normalized.rules.len(),
            &format!(" nullable_terminals={}", nullable_terminals.len()),
        );
    }

    let normalize_rules_before = normalized.rules.len();
    let normalize_started_at = profiling.then(Instant::now);
    normalize_grammar(&mut normalized.rules, normalized.start);
    if let Some(started_at) = normalize_started_at {
        emit_grammar_transform_profile(
            "normalize_grammar",
            elapsed_ms(started_at),
            normalize_rules_before,
            normalized.rules.len(),
            "",
        );
    }

    let protected_rules_before = normalized.rules.len();
    let protected_started_at = profiling.then(Instant::now);
    let protected_nonterminals = collect_protected_nonterminals(normalized);
    if let Some(started_at) = protected_started_at {
        emit_grammar_transform_profile(
            "collect_protected_nonterminals",
            elapsed_ms(started_at),
            protected_rules_before,
            normalized.rules.len(),
            &format!(" protected_nonterminals={}", protected_nonterminals.len()),
        );
    }

    let inline_rules_before = normalized.rules.len();
    let inline_started_at = profiling.then(Instant::now);
    inline_single_use_nonterminals(&mut normalized.rules, &protected_nonterminals);
    if let Some(started_at) = inline_started_at {
        emit_grammar_transform_profile(
            "inline_single_use_nonterminals",
            elapsed_ms(started_at),
            inline_rules_before,
            normalized.rules.len(),
            " pass=pre_bound",
        );
    }

    let mut merge_iteration = 0usize;
    loop {
        merge_iteration += 1;
        let prev_len = normalized.rules.len();
        let merge_started_at = profiling.then(Instant::now);
        normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
        if let Some(started_at) = merge_started_at {
            emit_grammar_transform_profile(
                "merge_identical_nonterminals",
                elapsed_ms(started_at),
                prev_len,
                normalized.rules.len(),
                &format!(" pass=pre_bound iteration={merge_iteration}"),
            );
        }
        if normalized.rules.len() == prev_len {
            break;
        }
    }

    let max_reduction_len = std::env::var("GLRMASK_MAX_RUNTIME_REDUCTION_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_RUNTIME_REDUCTION_LEN);

    let bound_rules_before = normalized.rules.len();
    let bound_started_at = profiling.then(Instant::now);
    bound_runtime_reduction_length(normalized, max_reduction_len);
    if let Some(started_at) = bound_started_at {
        emit_grammar_transform_profile(
            "bound_runtime_reduction_length",
            elapsed_ms(started_at),
            bound_rules_before,
            normalized.rules.len(),
            &format!(" max_reduction_len={max_reduction_len}"),
        );
    }

    let post_inline_rules_before = normalized.rules.len();
    let post_inline_started_at = profiling.then(Instant::now);
    inline_post_bound_single_use_nonterminals(
        &mut normalized.rules,
        &protected_nonterminals,
        max_reduction_len,
    );
    if let Some(started_at) = post_inline_started_at {
        emit_grammar_transform_profile(
            "inline_post_bound_single_use_nonterminals",
            elapsed_ms(started_at),
            post_inline_rules_before,
            normalized.rules.len(),
            &format!(" max_reduction_len={max_reduction_len}"),
        );
    }

    let mut final_merge_iteration = 0usize;
    loop {
        final_merge_iteration += 1;
        let prev_len = normalized.rules.len();
        let merge_started_at = profiling.then(Instant::now);
        normalized.rules = merge_identical_nonterminals(&normalized.rules, normalized.start);
        if let Some(started_at) = merge_started_at {
            emit_grammar_transform_profile(
                "merge_identical_nonterminals",
                elapsed_ms(started_at),
                prev_len,
                normalized.rules.len(),
                &format!(" pass=post_bound iteration={final_merge_iteration}"),
            );
        }
        if normalized.rules.len() == prev_len {
            break;
        }
    }

    let post_merge_rr_rules_before = normalized.rules.len();
    let post_merge_rr_started_at = profiling.then(Instant::now);
    let mut next_nt = normalized.num_nonterminals();
    let mut fresh_nt = || {
        let id = next_nt;
        next_nt += 1;
        id
    };
    eliminate_right_recursion(&mut normalized.rules, &mut fresh_nt);
    // Inlining and identical-nonterminal merging can recreate indirect left
    // recursion after the initial normalization pass. Most grammars do not,
    // so pay for the full fixed-point normalization only when the cheap
    // left-reachability check finds such a cycle.
    if has_indirect_left_recursion(&normalized.rules) {
        normalize_grammar(&mut normalized.rules, normalized.start);
    }
    if let Some(started_at) = post_merge_rr_started_at {
        emit_grammar_transform_profile(
            "post_merge_normalize",
            elapsed_ms(started_at),
            post_merge_rr_rules_before,
            normalized.rules.len(),
            " pass=post_merge",
        );
    }

    let compact_rules_before = normalized.rules.len();
    let compact_started_at = profiling.then(Instant::now);
    compact_unused_terminals(normalized);
    if let Some(started_at) = compact_started_at {
        emit_grammar_transform_profile(
            "compact_unused_terminals",
            elapsed_ms(started_at),
            compact_rules_before,
            normalized.rules.len(),
            "",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nt(id: NonterminalID) -> Symbol {
        Symbol::Nonterminal(id)
    }

    fn t(id: TerminalID) -> Symbol {
        Symbol::Terminal(id)
    }

    fn remove_cyclic_inline_candidates_reference(
        inline_candidates: &mut BTreeMap<NonterminalID, (usize, Vec<Symbol>)>,
    ) {
        fn reaches_start(
            start: NonterminalID,
            current: NonterminalID,
            deps: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
            seen: &mut BTreeSet<NonterminalID>,
        ) -> bool {
            if !seen.insert(current) {
                return false;
            }
            deps.get(&current).is_some_and(|nexts| {
                nexts
                    .iter()
                    .any(|&next| next == start || reaches_start(start, next, deps, seen))
            })
        }

        let inline_candidate_ids: BTreeSet<NonterminalID> =
            inline_candidates.keys().copied().collect();
        let deps: BTreeMap<NonterminalID, BTreeSet<NonterminalID>> = inline_candidates
            .iter()
            .map(|(&nt, (_, rhs))| {
                let rhs_deps = rhs
                    .iter()
                    .filter_map(|symbol| match symbol {
                        Symbol::Nonterminal(id) if inline_candidate_ids.contains(id) => Some(*id),
                        _ => None,
                    })
                    .collect();
                (nt, rhs_deps)
            })
            .collect();

        let cyclic: Vec<NonterminalID> = inline_candidate_ids
            .iter()
            .copied()
            .filter(|&nt| reaches_start(nt, nt, &deps, &mut BTreeSet::new()))
            .collect();
        for nt in cyclic {
            inline_candidates.remove(&nt);
        }
    }

    #[test]
    fn inline_candidate_expansion_matches_reference_on_varied_graphs() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *seed
        }

        let mut seed = 0x1a11_e7a0_5eed_u64;
        for case_index in 0..512 {
            let nonterminals = (next(&mut seed) % 10 + 1) as u32;
            let mut candidates = BTreeMap::new();
            for nonterminal in 0..nonterminals {
                if next(&mut seed) % 4 == 0 {
                    continue;
                }
                let rhs_len = (next(&mut seed) % 5) as usize;
                let mut rhs = Vec::with_capacity(rhs_len);
                for _ in 0..rhs_len {
                    if next(&mut seed) & 1 == 0 {
                        rhs.push(Symbol::Terminal((next(&mut seed) % 5) as u32));
                    } else {
                        rhs.push(Symbol::Nonterminal(
                            (next(&mut seed) % u64::from(nonterminals + 2)) as u32,
                        ));
                    }
                }
                candidates.insert(nonterminal, (nonterminal as usize, rhs));
            }

            remove_cyclic_inline_candidates(&mut candidates);
            let mut expected = candidates.clone();
            expand_inline_candidates_iteratively(&mut expected);
            expand_inline_candidates(&mut candidates);
            assert_eq!(candidates, expected, "case_index={case_index}");
        }
    }

    #[test]
    fn inline_cycle_pruning_matches_reference_on_varied_candidate_graphs() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *seed
        }

        let mut seed = 0x1a11_c7c1_e55_u64;
        for case_index in 0..512 {
            let nonterminals = (next(&mut seed) % 9 + 1) as u32;
            let mut candidates = BTreeMap::new();
            for nonterminal in 0..nonterminals {
                if next(&mut seed) % 4 == 0 {
                    continue;
                }
                let rhs_len = (next(&mut seed) % 5) as usize;
                let mut rhs = Vec::with_capacity(rhs_len);
                for _ in 0..rhs_len {
                    if next(&mut seed) & 1 == 0 {
                        rhs.push(Symbol::Terminal((next(&mut seed) % 5) as u32));
                    } else {
                        rhs.push(Symbol::Nonterminal(
                            (next(&mut seed) % u64::from(nonterminals + 2)) as u32,
                        ));
                    }
                }
                candidates.insert(nonterminal, (nonterminal as usize, rhs));
            }

            let mut expected = candidates.clone();
            remove_cyclic_inline_candidates_reference(&mut expected);
            remove_cyclic_inline_candidates(&mut candidates);
            assert_eq!(candidates, expected, "case_index={case_index}");
        }
    }

    #[test]
    fn inline_single_use_nonterminals_skips_candidate_cycles() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![nt(1), t(0)],
            },
            Rule {
                lhs: 1,
                rhs: vec![nt(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![nt(1)],
            },
        ];

        inline_single_use_nonterminals(&mut rules, &BTreeSet::new());

        assert_eq!(
            rules,
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![nt(1), t(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![nt(2)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![nt(1)],
                },
            ]
        );
    }

    #[test]
    fn inline_single_use_nonterminals_still_expands_acyclic_chains() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![nt(1), t(2)],
            },
            Rule {
                lhs: 1,
                rhs: vec![nt(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![t(0), t(1)],
            },
        ];

        inline_single_use_nonterminals(&mut rules, &BTreeSet::new());

        assert_eq!(
            rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![t(0), t(1), t(2)],
            }]
        );
    }
}
