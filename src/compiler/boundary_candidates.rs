//! Grammar-aware reusable candidate summaries for composition boundaries.
//!
//! The summary answers one deliberately narrow question in original model-token
//! coordinates: can a non-empty proper byte prefix of this model token reach a
//! component's public outward interface from some component-local history?
//! It is an upper bound. Every uncertainty widens; none is interpreted as an
//! empty language.

use std::collections::{BTreeMap, BTreeSet};

use crate::automata::lexer::Lexer;
use crate::grammar::flat::{Rule, Symbol, TerminalID};
use crate::runtime::{
    BoundaryCandidateFingerprint, BoundaryCandidateSummary, Constraint, OriginalTokenSet,
    SummaryPrecision, SummaryUnavailable,
};

const BOUNDARY_CANDIDATE_ALGORITHM_VERSION: u16 = 2;
const DEFAULT_MAX_FRONTIER_PAIRS: usize = 250_000;
const DEFAULT_MAX_INITIAL_FRONTIER_PROBES: usize = 1_000_000;
// Frontier steps manipulate grammar-position sets rather than raw bytes, so
// one "step" is materially more expensive than a tokenizer transition. Keep
// the reusable-summary budget intentionally small: on large components the
// safe fallback (all multibyte model tokens) is preferable to making a later
// composition pay seconds to save a modest fraction of the vocabulary.
const DEFAULT_MAX_BYTE_STEPS: usize = 10_000;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BoundaryCandidateStats {
    pub(crate) input_tokens: usize,
    pub(crate) candidate_tokens: usize,
    pub(crate) byte_steps: usize,
    pub(crate) widened_subtrees: usize,
    pub(crate) peak_frontier_pairs: usize,
}

#[derive(Clone, Default)]
struct Frontier {
    by_lexer_state: BTreeMap<u32, BTreeSet<usize>>,
}

impl Frontier {
    fn insert_controls(&mut self, lexer_state: u32, controls: &BTreeSet<usize>) {
        if controls.is_empty() {
            return;
        }
        self.by_lexer_state
            .entry(lexer_state)
            .or_default()
            .extend(controls.iter().copied());
    }

    fn pair_count(&self) -> usize {
        self.by_lexer_state.values().map(BTreeSet::len).sum()
    }

    fn is_empty(&self) -> bool {
        self.by_lexer_state.is_empty()
    }
}

struct GrammarMachine<'a> {
    rules: &'a [Rule],
    rule_offsets: Vec<usize>,
    position_rule: Vec<usize>,
    position_dot: Vec<usize>,
    entries: BTreeMap<u32, Vec<usize>>,
    returns: BTreeMap<u32, Vec<usize>>,
    outward_terminals: BTreeSet<TerminalID>,
    skip_terminals: BTreeSet<TerminalID>,
    control_terminals: BTreeSet<TerminalID>,
}

impl<'a> GrammarMachine<'a> {
    fn new(
        rules: &'a [Rule],
        outward_terminals: BTreeSet<TerminalID>,
        skip_terminals: BTreeSet<TerminalID>,
        control_terminals: BTreeSet<TerminalID>,
    ) -> Result<Self, SummaryUnavailable> {
        if rules.is_empty() {
            return Err(SummaryUnavailable::MissingGrammarMetadata);
        }
        let mut rule_offsets = Vec::with_capacity(rules.len());
        let mut position_rule = Vec::new();
        let mut position_dot = Vec::new();
        let mut entries = BTreeMap::<u32, Vec<usize>>::new();
        for (rule_index, rule) in rules.iter().enumerate() {
            let offset = position_rule.len();
            rule_offsets.push(offset);
            entries.entry(rule.lhs).or_default().push(offset);
            for dot in 0..=rule.rhs.len() {
                position_rule.push(rule_index);
                position_dot.push(dot);
            }
        }
        let mut returns = BTreeMap::<u32, Vec<usize>>::new();
        for (rule_index, rule) in rules.iter().enumerate() {
            for (dot, symbol) in rule.rhs.iter().enumerate() {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    returns
                        .entry(*nonterminal)
                        .or_default()
                        .push(rule_offsets[rule_index] + dot + 1);
                }
            }
        }
        Ok(Self {
            rules,
            rule_offsets,
            position_rule,
            position_dot,
            entries,
            returns,
            outward_terminals,
            skip_terminals,
            control_terminals,
        })
    }

    fn symbol_at(&self, position: usize) -> Option<&Symbol> {
        let rule_index = *self.position_rule.get(position)?;
        let dot = *self.position_dot.get(position)?;
        self.rules.get(rule_index)?.rhs.get(dot)
    }

    fn next_position(&self, position: usize) -> usize {
        position + 1
    }

    fn is_augmented_completion(&self, position: usize) -> bool {
        self.position_rule.get(position).copied() == Some(0)
            && self
                .position_dot
                .get(position)
                .copied()
                .is_some_and(|dot| dot == self.rules[0].rhs.len())
    }

    /// Epsilon-close parser control after a committed terminal. Public outward
    /// portals are reported and not traversed. Nonterminal return is deliberately
    /// context-insensitive (K=0): it is an over-approximation, never a pruning
    /// assumption.
    fn operational_closure(&self, seeds: impl IntoIterator<Item = usize>) -> (BTreeSet<usize>, bool) {
        let mut stable = BTreeSet::new();
        let mut pending = seeds.into_iter().collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        let mut outward = false;
        while let Some(position) = pending.pop() {
            if !seen.insert(position) {
                continue;
            }
            let rule_index = self.position_rule[position];
            let dot = self.position_dot[position];
            let rule = &self.rules[rule_index];
            if dot == rule.rhs.len() {
                if self.is_augmented_completion(position) {
                    outward = true;
                } else if let Some(continuations) = self.returns.get(&rule.lhs) {
                    pending.extend(continuations.iter().copied());
                }
                continue;
            }
            match &rule.rhs[dot] {
                Symbol::Nonterminal(nonterminal) => {
                    if let Some(entries) = self.entries.get(nonterminal) {
                        pending.extend(entries.iter().copied());
                    }
                }
                Symbol::Terminal(terminal) if self.outward_terminals.contains(terminal) => {
                    outward = true;
                }
                Symbol::Terminal(terminal) if self.control_terminals.contains(terminal) => {
                    pending.push(self.next_position(position));
                }
                Symbol::Terminal(_) => {
                    stable.insert(position);
                }
            }
        }
        (stable, outward)
    }

    /// Parser positions at which an arbitrary *later* model token may start.
    /// Previous ordinary terminals and unresolved slots are traversed as
    /// history edges. An unresolved slot therefore does not hide positions that
    /// become reachable after a previously completed child invocation.
    fn history_starts(&self) -> BTreeSet<usize> {
        let mut stable = BTreeSet::new();
        let mut pending = vec![self.rule_offsets[0]];
        let mut seen = BTreeSet::new();
        while let Some(position) = pending.pop() {
            if !seen.insert(position) {
                continue;
            }
            let rule_index = self.position_rule[position];
            let dot = self.position_dot[position];
            let rule = &self.rules[rule_index];
            if dot == rule.rhs.len() {
                if !self.is_augmented_completion(position) {
                    if let Some(continuations) = self.returns.get(&rule.lhs) {
                        pending.extend(continuations.iter().copied());
                    }
                }
                continue;
            }
            match &rule.rhs[dot] {
                Symbol::Nonterminal(nonterminal) => {
                    if let Some(entries) = self.entries.get(nonterminal) {
                        pending.extend(entries.iter().copied());
                    }
                }
                Symbol::Terminal(terminal) if self.control_terminals.contains(terminal) => {
                    pending.push(self.next_position(position));
                }
                Symbol::Terminal(terminal) if self.outward_terminals.contains(terminal) => {
                    // A prior token may already have traversed the child. Do not
                    // count the portal itself as a zero-offset candidate, but
                    // retain its grammar position as a possible token-start
                    // control. A real local Skip/IGNORE may consume positive
                    // bytes while the grammar remains parked immediately
                    // before this portal; that positive prefix then witnesses
                    // the first outward event. `initial_frontier` admits this
                    // seed only when the lexer can actually scan the expected
                    // terminal or a local skip from the raw state.
                    stable.insert(position);
                    pending.push(self.next_position(position));
                }
                Symbol::Terminal(_) => {
                    stable.insert(position);
                    pending.push(self.next_position(position));
                }
            }
        }
        stable
    }

    fn expected_terminal(&self, position: usize) -> Option<TerminalID> {
        match self.symbol_at(position) {
            Some(Symbol::Terminal(terminal)) => Some(*terminal),
            _ => None,
        }
    }

    fn advance_terminal(&self, position: usize, terminal: TerminalID) -> Option<usize> {
        (self.expected_terminal(position) == Some(terminal)).then(|| self.next_position(position))
    }
}

struct CandidateMachine<'a> {
    tokenizer: &'a crate::automata::lexer::tokenizer::Tokenizer,
    grammar: GrammarMachine<'a>,
    reset_states: Vec<u32>,
    max_frontier_pairs: usize,
    max_byte_steps: usize,
    byte_steps: usize,
    peak_frontier_pairs: usize,
}

impl<'a> CandidateMachine<'a> {
    fn initial_frontier(&mut self) -> Result<Frontier, ()> {
        let starts = self.grammar.history_starts();
        // The frontier pair cap is a memory/result-size budget, but applying it
        // only after probing every raw-state × grammar-position pair lets a
        // large legacy component spend tens of seconds discovering that the
        // result must be widened anyway.  Bound the probe work itself first.
        // Widening is always sound for this summary: callers fall back to all
        // multibyte model tokens, never to an empty candidate set.
        if starts
            .len()
            .saturating_mul(self.tokenizer.num_states() as usize)
            > DEFAULT_MAX_INITIAL_FRONTIER_PROBES
        {
            return Err(());
        }
        let mut frontier = Frontier::default();
        for raw in 0..self.tokenizer.num_states() {
            let futures = self.tokenizer.possible_future_terminals(raw);
            let can_skip = self
                .grammar
                .skip_terminals
                .iter()
                .any(|&terminal| (terminal as usize) < futures.len() && futures.get(terminal as usize));
            let controls = starts
                .iter()
                .filter_map(|&position| {
                    let expected = self.grammar.expected_terminal(position)?;
                    (((expected as usize) < futures.len() && futures.get(expected as usize)) || can_skip)
                        .then_some(position)
                })
                .collect::<BTreeSet<_>>();
            frontier.insert_controls(raw, &controls);
        }
        self.observe_frontier(&frontier)?;
        // Accepting lexer states may commit exactly at the model-token boundary.
        // Such an offset-zero outward event is not itself a candidate, but an
        // internal commit can change the parser/lexer state from which byte 0
        // of this token is consumed. Close those internal commits here.
        self.close_zero_commits(frontier)
    }

    fn observe_frontier(&mut self, frontier: &Frontier) -> Result<(), ()> {
        let pairs = frontier.pair_count();
        self.peak_frontier_pairs = self.peak_frontier_pairs.max(pairs);
        (pairs <= self.max_frontier_pairs && self.byte_steps <= self.max_byte_steps)
            .then_some(())
            .ok_or(())
    }

    fn close_zero_commits(&mut self, mut frontier: Frontier) -> Result<Frontier, ()> {
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot = frontier.by_lexer_state.clone();
            for (raw, controls) in snapshot {
                let matched = self.tokenizer.matched_terminals_iter(raw).collect::<Vec<_>>();
                if matched.is_empty() {
                    continue;
                }
                for terminal in matched {
                    let mut advanced = BTreeSet::new();
                    for &position in &controls {
                        if self.grammar.skip_terminals.contains(&terminal) {
                            // Skip consumes a lexer terminal without advancing
                            // the grammar symbol, but it can expose an outward
                            // portal (or another internal nullable/control
                            // path) at the same grammar position. Offset zero
                            // outward is intentionally not a candidate and is
                            // not traversed; retain any simultaneous internal
                            // stable branches conservatively.
                            let (closed, _outward) =
                                self.grammar.operational_closure([position]);
                            advanced.extend(closed);
                        }
                        if let Some(next) = self.grammar.advance_terminal(position, terminal) {
                            let (closed, outward) = self.grammar.operational_closure([next]);
                            // Outward at offset zero is deliberately ignored and
                            // not traversed. Internal closure remains live.
                            let _ = outward;
                            advanced.extend(closed);
                        }
                    }
                    if advanced.is_empty() {
                        continue;
                    }
                    for &reset in &self.reset_states {
                        let before = frontier
                            .by_lexer_state
                            .get(&reset)
                            .map_or(0, BTreeSet::len);
                        frontier.insert_controls(reset, &advanced);
                        let after = frontier
                            .by_lexer_state
                            .get(&reset)
                            .map_or(0, BTreeSet::len);
                        changed |= after != before;
                    }
                }
            }
            self.observe_frontier(&frontier)?;
        }
        Ok(frontier)
    }

    fn consume(&mut self, frontier: &Frontier, byte: u8) -> Result<(Frontier, bool), ()> {
        self.byte_steps = self.byte_steps.saturating_add(frontier.by_lexer_state.len());
        if self.byte_steps > self.max_byte_steps {
            return Err(());
        }
        let mut next = Frontier::default();
        let mut outward = false;
        for (&raw, controls) in &frontier.by_lexer_state {
            let targets = self.tokenizer.step_all(&[raw], byte);
            for target in targets {
                // The terminal need not commit here; retain the ordinary
                // continuing lexer branch.
                next.insert_controls(target, controls);
                let matched = self
                    .tokenizer
                    .matched_terminals_iter(target)
                    .collect::<Vec<_>>();
                for terminal in matched {
                    let mut advanced = BTreeSet::new();
                    for &position in controls {
                        if self.grammar.skip_terminals.contains(&terminal) {
                            let (closed, did_outward) =
                                self.grammar.operational_closure([position]);
                            outward |= did_outward;
                            advanced.extend(closed);
                        }
                        if let Some(after_terminal) =
                            self.grammar.advance_terminal(position, terminal)
                        {
                            let (closed, did_outward) =
                                self.grammar.operational_closure([after_terminal]);
                            outward |= did_outward;
                            advanced.extend(closed);
                        }
                    }
                    for &reset in &self.reset_states {
                        next.insert_controls(reset, &advanced);
                    }
                }
            }
        }
        self.observe_frontier(&next)?;
        Ok((next, outward))
    }
}

fn fingerprint(
    constraint: &Constraint,
    vocab: &crate::Vocab,
    rules: &[Rule],
) -> Result<BoundaryCandidateFingerprint, SummaryUnavailable> {
    let exprs = constraint
        .retained_terminal_exprs()
        .ok_or(SummaryUnavailable::MissingGrammarMetadata)?;
    let mut semantics = blake3::Hasher::new();
    semantics.update(b"glrmask-boundary-component-semantics-v1\0");
    semantics.update(
        &bincode::serialize(rules).map_err(|_| SummaryUnavailable::MalformedMetadata)?,
    );
    semantics.update(
        &bincode::serialize(exprs).map_err(|_| SummaryUnavailable::MalformedMetadata)?,
    );
    semantics.update(&constraint.ignore_terminal.unwrap_or(u32::MAX).to_le_bytes());
    for terminal in &constraint.table.skip_terminals {
        semantics.update(&terminal.to_le_bytes());
    }
    for terminal in &constraint.table.control_terminals {
        semantics.update(&terminal.to_le_bytes());
    }

    let mut interface = blake3::Hasher::new();
    interface.update(b"glrmask-boundary-public-interface-v1\0");
    for (name, terminal) in &constraint.unbound_grammar_placeholders {
        interface.update(&(name.len() as u64).to_le_bytes());
        interface.update(name.as_bytes());
        interface.update(&terminal.to_le_bytes());
    }
    let mut late = constraint
        .late_grammar_slots
        .iter()
        .map(|slot| (slot.name.as_str(), slot.terminal_id))
        .collect::<Vec<_>>();
    late.sort_unstable();
    for (name, terminal) in late {
        interface.update(&(name.len() as u64).to_le_bytes());
        interface.update(name.as_bytes());
        interface.update(&terminal.to_le_bytes());
    }
    let mut specials = constraint
        .special_token_terminals
        .iter()
        .filter(|special| {
            constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                || constraint.token_bytes_for_id(special.token_id).is_none()
        })
        .map(|special| (special.terminal_id, special.token_id))
        .collect::<Vec<_>>();
    specials.sort_unstable();
    for (terminal, token) in specials {
        interface.update(&terminal.to_le_bytes());
        interface.update(&token.to_le_bytes());
    }

    Ok(BoundaryCandidateFingerprint {
        algorithm_version: BOUNDARY_CANDIDATE_ALGORITHM_VERSION,
        component_semantics: *semantics.finalize().as_bytes(),
        public_interface: *interface.finalize().as_bytes(),
        vocabulary: crate::compiler::compile::vocab_content_digest(vocab),
    })
}

fn outward_terminals(constraint: &Constraint) -> BTreeSet<TerminalID> {
    constraint
        .unbound_grammar_placeholders
        .values()
        .copied()
        .chain(constraint.late_grammar_slots.iter().map(|slot| slot.terminal_id))
        .chain(
            constraint
                .special_token_terminals
                .iter()
                .filter(|special| {
                    constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                        || constraint.token_bytes_for_id(special.token_id).is_none()
                })
                .map(|special| special.terminal_id),
        )
        .collect()
}

fn add_descendant_ids(
    entries: &[(u32, &[u8])],
    prefix_len: usize,
    candidates: &mut BTreeSet<u32>,
) {
    candidates.extend(
        entries
            .iter()
            .filter_map(|(id, bytes)| (bytes.len() > prefix_len).then_some(*id)),
    );
}

fn scan_group(
    machine: &mut CandidateMachine<'_>,
    entries: &[(u32, &[u8])],
    depth: usize,
    frontier: &Frontier,
    candidates: &mut BTreeSet<u32>,
    widened_subtrees: &mut usize,
) {
    let mut index = 0usize;
    while index < entries.len() {
        while index < entries.len() && entries[index].1.len() <= depth {
            index += 1;
        }
        if index == entries.len() {
            return;
        }
        let byte = entries[index].1[depth];
        let start = index;
        index += 1;
        while index < entries.len()
            && entries[index].1.len() > depth
            && entries[index].1[depth] == byte
        {
            index += 1;
        }
        let group = &entries[start..index];
        match machine.consume(frontier, byte) {
            Err(()) => {
                *widened_subtrees += 1;
                add_descendant_ids(group, depth + 1, candidates);
            }
            Ok((next, outward)) => {
                let prefix_len = depth + 1;
                if outward {
                    add_descendant_ids(group, prefix_len, candidates);
                } else if !next.is_empty() {
                    scan_group(
                        machine,
                        group,
                        prefix_len,
                        &next,
                        candidates,
                        widened_subtrees,
                    );
                }
            }
        }
    }
}

fn compute_summary(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (BoundaryCandidateSummary, BoundaryCandidateStats) {
    let mut stats = BoundaryCandidateStats {
        input_tokens: vocab.len(),
        ..BoundaryCandidateStats::default()
    };
    if constraint.uses_compact_segmented_parser_runtime() {
        match constraint.recursive_parser_layout() {
            Ok(Some(layout))
                if constraint.tokenizer.num_states() != layout.total_tokenizer_states =>
            {
                return (
                    BoundaryCandidateSummary::Unknown {
                        reason: SummaryUnavailable::Deferred,
                    },
                    stats,
                );
            }
            Err(_) => {
                return (
                    BoundaryCandidateSummary::Unknown {
                        reason: SummaryUnavailable::MalformedMetadata,
                    },
                    stats,
                );
            }
            Ok(_) => {}
        }
    }
    let rules = match constraint.retained_table_rules() {
        Ok(rules) if !rules.is_empty() => rules,
        _ => {
            return (
                BoundaryCandidateSummary::Unknown {
                    reason: SummaryUnavailable::MissingGrammarMetadata,
                },
                stats,
            );
        }
    };
    let fp = match fingerprint(constraint, vocab, rules) {
        Ok(fp) => fp,
        Err(reason) => return (BoundaryCandidateSummary::Unknown { reason }, stats),
    };
    let mut local_skip_terminals = constraint.table.skip_terminals.clone();
    if let Some(ignore) = constraint.ignore_terminal {
        local_skip_terminals.insert(ignore);
    }
    let grammar = match GrammarMachine::new(
        rules,
        outward_terminals(constraint),
        local_skip_terminals,
        constraint.table.control_terminals.clone(),
    ) {
        Ok(grammar) => grammar,
        Err(reason) => return (BoundaryCandidateSummary::Unknown { reason }, stats),
    };
    let reset_states = constraint
        .tokenizer
        .deterministic_reset_states()
        .into_iter()
        .collect::<Vec<_>>();
    let mut machine = CandidateMachine {
        tokenizer: &constraint.tokenizer,
        grammar,
        reset_states,
        max_frontier_pairs: DEFAULT_MAX_FRONTIER_PAIRS,
        max_byte_steps: DEFAULT_MAX_BYTE_STEPS,
        byte_steps: 0,
        peak_frontier_pairs: 0,
    };
    let initial = match machine.initial_frontier() {
        Ok(frontier) => frontier,
        Err(()) => {
            stats.widened_subtrees = 1;
            stats.candidate_tokens = vocab.iter().filter(|(_, bytes)| bytes.len() >= 2).count();
            return (
                BoundaryCandidateSummary::Known {
                    fingerprint: fp,
                    tokens: OriginalTokenSet::AllByteTokensAtLeastTwo,
                    precision: SummaryPrecision::BudgetWidenedUpperBound,
                },
                stats,
            );
        }
    };

    let mut entries = vocab.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| {
        left.1
            .cmp(right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut candidates = BTreeSet::new();
    let mut widened_subtrees = 0usize;
    if !initial.is_empty() {
        scan_group(
            &mut machine,
            &entries,
            0,
            &initial,
            &mut candidates,
            &mut widened_subtrees,
        );
    }
    let ids = candidates.into_iter().collect::<Vec<_>>();
    stats.candidate_tokens = ids.len();
    stats.byte_steps = machine.byte_steps;
    stats.peak_frontier_pairs = machine.peak_frontier_pairs;
    stats.widened_subtrees = widened_subtrees;
    let precision = if widened_subtrees == 0 {
        SummaryPrecision::RegularUpperBound
    } else {
        SummaryPrecision::BudgetWidenedUpperBound
    };
    (
        BoundaryCandidateSummary::Known {
            fingerprint: fp,
            tokens: OriginalTokenSet::from_sorted_unique(ids, vocab.max_token_id()),
            precision,
        },
        stats,
    )
}

pub(crate) fn boundary_candidate_summary(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (BoundaryCandidateSummary, BoundaryCandidateStats) {
    let rules = match constraint.retained_table_rules() {
        Ok(rules) if !rules.is_empty() => rules,
        _ => return compute_summary(constraint, vocab),
    };
    let wanted = fingerprint(constraint, vocab, rules).ok();
    if let Some(existing) = constraint.boundary_candidate_summary.get() {
        if wanted
            .as_ref()
            .is_some_and(|fp| existing.known_tokens_for(fp).is_some())
        {
            let count = match existing {
                BoundaryCandidateSummary::Known { tokens, .. } => tokens
                    .canonical_ids(vocab.iter())
                    .len(),
                BoundaryCandidateSummary::Unknown { .. } => 0,
            };
            return (
                existing.clone(),
                BoundaryCandidateStats {
                    input_tokens: vocab.len(),
                    candidate_tokens: count,
                    ..BoundaryCandidateStats::default()
                },
            );
        }
        if matches!(
            existing,
            BoundaryCandidateSummary::Unknown {
                reason: SummaryUnavailable::Disabled
            }
        ) {
            return (
                existing.clone(),
                BoundaryCandidateStats {
                    input_tokens: vocab.len(),
                    ..BoundaryCandidateStats::default()
                },
            );
        }
    }
    let (summary, stats) = compute_summary(constraint, vocab);
    // A loaded legacy artifact may retain an unchanged-resave byte cache.  Do
    // not install a newly computed summary behind that cache: doing so would
    // make the in-memory metadata differ from save() without a mutable cache
    // invalidation path.  Current artifacts already carry their summary; fresh
    // compiler-owned constraints have no cache here and may memoize normally.
    if constraint.serialized_artifact_cache.is_none() {
        let _ = constraint.boundary_candidate_summary.set(summary.clone());
    }
    (summary, stats)
}

pub(crate) fn restricted_boundary_vocab(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (crate::Vocab, BoundaryCandidateSummary, BoundaryCandidateStats) {
    let (summary, mut stats) = boundary_candidate_summary(constraint, vocab);
    let entries = match &summary {
        BoundaryCandidateSummary::Known { tokens, .. } => vocab
            .iter()
            .filter(|(id, bytes)| tokens.contains(*id, bytes))
            .map(|(id, bytes)| (id, bytes.to_vec()))
            .collect::<Vec<_>>(),
        BoundaryCandidateSummary::Unknown { .. } => vocab
            .iter()
            .map(|(id, bytes)| (id, bytes.to_vec()))
            .collect::<Vec<_>>(),
    };
    stats.candidate_tokens = entries.len();
    (crate::Vocab::new(entries), summary, stats)
}

pub(crate) fn prepare_boundary_candidate_summary(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> BoundaryCandidateStats {
    boundary_candidate_summary(constraint, vocab).1
}

/// Return the checked original-ID candidate set for one component. `None`
/// means the summary is unavailable and the caller must widen to the full
/// vocabulary; `Some(empty)` is a proved empty proper-prefix candidate set.
pub(crate) fn boundary_candidate_ids(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> (Option<Vec<u32>>, BoundaryCandidateStats) {
    let (summary, stats) = boundary_candidate_summary(constraint, vocab);
    let ids = match summary {
        BoundaryCandidateSummary::Known { tokens, .. } => {
            Some(tokens.canonical_ids(vocab.iter()))
        }
        BoundaryCandidateSummary::Unknown { .. } => None,
    };
    (ids, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proper_prefix_widening_excludes_token_ending_at_prefix() {
        let entries = vec![
            (10u32, b"ab".as_slice()),
            (11u32, b"abc".as_slice()),
            (12u32, b"abd".as_slice()),
            (13u32, b"ab".as_slice()),
        ];
        let mut out = BTreeSet::new();
        add_descendant_ids(&entries, 2, &mut out);
        assert_eq!(out, BTreeSet::from([11, 12]));
    }

    #[test]
    fn root_completion_requires_proper_prefix_and_preserves_duplicate_ids() {
        let vocab = crate::Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, stats) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1, 2]));
        assert_eq!(stats.input_tokens, 4);
        assert_eq!(stats.candidate_tokens, 2);
    }

    #[test]
    fn live_out_of_vocab_special_slot_is_an_outward_portal() {
        let vocab = crate::Vocab::new(vec![
            (0, b"m".to_vec()),
            (1, b"mg".to_vec()),
            (2, b"mx".to_vec()),
            (3, b"g".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(998);
                nt document ::= "m" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1, 2]));
    }

    #[test]
    fn stale_known_empty_fingerprint_is_recomputed_not_trusted() {
        let vocab = crate::Vocab::new(vec![
            (0, b"m".to_vec()),
            (1, b"mg".to_vec()),
            (2, b"mx".to_vec()),
            (3, b"g".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(998);
                nt document ::= "m" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        constraint
            .boundary_candidate_summary
            .set(BoundaryCandidateSummary::Known {
                fingerprint: BoundaryCandidateFingerprint {
                    algorithm_version: BOUNDARY_CANDIDATE_ALGORITHM_VERSION,
                    component_semantics: [0; 32],
                    public_interface: [0; 32],
                    vocabulary: [0; 32],
                },
                tokens: OriginalTokenSet::Empty,
                precision: SummaryPrecision::RegularUpperBound,
            })
            .unwrap();

        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(
            ids,
            Some(vec![1, 2]),
            "fingerprint mismatch must widen/recompute instead of trusting stale empty",
        );
    }

    #[test]
    fn leading_component_ignore_can_precede_positive_outward_cut() {
        let vocab = crate::Vocab::new(vec![
            (0, b" X a".to_vec()),
            (1, b"X a".to_vec()),
            (2, b" ".to_vec()),
            (3, b"X".to_vec()),
            (4, b"a".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB;
            "#,
            &vocab,
        )
        .unwrap();

        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        let ids = ids.expect("component summary should be known");
        assert!(ids.contains(&0), "leading ignore must not hide the later outward cut");
        assert!(ids.contains(&1), "direct outward-prefix token must remain a candidate");
    }

    #[test]
    fn ignore_commit_can_expose_outward_cut_before_next_byte() {
        let vocab = crate::Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" \ta".to_vec()),
            (2, b" ".to_vec()),
            (3, b"\t".to_vec()),
            (4, b"a".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB;
            "#,
            &vocab,
        )
        .unwrap();

        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert!(
            ids.expect("component summary should be known").contains(&1),
            "committing WS after its positive-byte prefix must expose SUB before the following byte",
        );
    }

    #[test]
    fn multiple_local_terminal_commits_can_precede_outward_cut() {
        let vocab = crate::Vocab::new(vec![
            (0, b"ab".to_vec()),
            (1, b"abx".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "a" "b" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1]));
    }

    #[test]
    fn utf8_literal_prefix_uses_byte_offsets_without_rounding() {
        let vocab = crate::Vocab::new(vec![
            (0, "é".as_bytes().to_vec()),
            (1, "éx".as_bytes().to_vec()),
            (2, b"x".to_vec()),
        ]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "é" SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let (ids, _) = boundary_candidate_ids(&constraint, &vocab);
        assert_eq!(ids, Some(vec![1]));
    }
}
