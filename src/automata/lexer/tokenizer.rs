//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::automata::regex::Expr;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub(crate) dfa: DFA,
    pub num_terminals: u32,
    /// Per-terminal regex expressions used to (re)build this tokenizer.
    /// Skipped during (de)serialization because they are only needed during
    /// compile-time simplification for active-terminal rebuilds.
    #[serde(default, skip)]
    pub(crate) exprs: Option<Arc<[Expr]>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMatch {
    pub id: TerminalID,
    pub width: usize,
    pub end_state: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerExecResult {
    pub end_state: Option<u32>,
    pub matches: Vec<TokenizerMatch>,
}

fn into_longest_matches(matches: FxHashMap<TerminalID, (usize, u32)>) -> Vec<TokenizerMatch> {
    matches
        .into_iter()
        .map(|(id, (width, end_state))| TokenizerMatch {
            id,
            width,
            end_state,
        })
        .collect()
}

fn group_matches_by_width(matches: Vec<TokenizerMatch>) -> Vec<(usize, BTreeSet<TerminalID>)> {
    let mut grouped = std::collections::BTreeMap::<usize, BTreeSet<TerminalID>>::new();
    for matched in matches {
        grouped.entry(matched.width).or_default().insert(matched.id);
    }
    grouped.into_iter().collect()
}

struct TerminalFilteredDfa {
    dfa: DFA,
    active_bitset: BitSet,
    any_cleared: bool,
    transitions_pruned: bool,
}

impl Tokenizer {
    pub fn start_state(&self) -> u32 {
        0
    }

    /// Rebuild a tokenizer from the active terminal regexes and map original
    /// states to rebuilt states by walking both DFAs in lockstep.
    pub fn simplified_from_active_exprs(
        &self,
        active_terminals: &[bool],
    ) -> Option<(Tokenizer, ManyToOneIdMap)> {
        use std::collections::VecDeque;

        let exprs = self.exprs.as_ref()?;
        if exprs.len() != active_terminals.len() {
            return None;
        }

        let local_to_orig: Vec<u32> = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(i, &active)| active.then_some(i as u32))
            .collect();
        if local_to_orig.is_empty() {
            return None;
        }

        let active_exprs: Vec<Expr> = local_to_orig
            .iter()
            .map(|&orig| exprs[orig as usize].clone())
            .collect();
        let regex = crate::automata::lexer::compile::build_regex(&active_exprs);
        let mut fresh_dfa: DFA = regex.dfa;

        let num_terminals = self.num_terminals as usize;
        fresh_dfa.ensure_group_capacity(num_terminals);
        let num_states = fresh_dfa.num_states();
        for state_id in 0..num_states {
            let old_finalizers = fresh_dfa.states()[state_id].finalizers.clone();
            let mut new_finalizers = BitSet::new(num_terminals);
            for local in old_finalizers.iter_ones() {
                if let Some(&orig) = local_to_orig.get(local) {
                    new_finalizers.set(orig as usize);
                }
            }

            let old_futures = fresh_dfa
                .possible_future_group_ids(state_id as u32)
                .clone();
            let mut new_futures = BitSet::new(num_terminals);
            for local in old_futures.iter_ones() {
                if let Some(&orig) = local_to_orig.get(local) {
                    new_futures.set(orig as usize);
                }
            }
            fresh_dfa.overwrite_state_metadata(state_id as u32, new_finalizers, new_futures);
        }

        let mut mapping = vec![u32::MAX; self.dfa.num_states()];
        mapping[0] = 0;
        let mut queue = VecDeque::from([0u32]);
        while let Some(original) = queue.pop_front() {
            let fresh = mapping[original as usize];
            let original_state = &self.dfa.states()[original as usize];
            for (byte, &original_next) in original_state.transitions.iter() {
                if let Some(fresh_next) = fresh_dfa.step(fresh, byte) {
                    let slot = &mut mapping[original_next as usize];
                    if *slot == u32::MAX {
                        *slot = fresh_next;
                        queue.push_back(original_next);
                    } else {
                        debug_assert_eq!(
                            *slot, fresh_next,
                            "parallel BFS inconsistency: original state {} mapped to both {} and {}",
                            original_next, *slot, fresh_next
                        );
                    }
                }
            }
        }

        let tok = Tokenizer {
            dfa: fresh_dfa,
            num_terminals: self.num_terminals,
            exprs: self.exprs.clone(),
        };
        Some((
            tok.clone(),
            ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                mapping,
                tok.num_states(),
            ),
        ))
    }

    /// Detect nullable terminals (those that match the empty string) by
    /// inspecting start-state finalizers, remove them from the DFA, and return
    /// the set.  After this call the tokenizer no longer reports those
    /// terminals as matched at state 0.
    pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {
        self.isolate_start_state();
        self.dfa
            .clear_finalizers_for_state(self.start_state())
            .iter()
            .map(|terminal| terminal as TerminalID)
            .collect()
    }

    /// Ensure that no byte transition in the DFA targets the start state.
    ///
    /// If any transition does, a copy of the start state is created and all
    /// such transitions are redirected to the copy.  This keeps the DFA
    /// equivalent while guaranteeing the start state is only reachable at
    /// position 0.
    fn isolate_start_state(&mut self) {
        let start = self.start_state();
        if !self.has_incoming_start_transitions(start) {
            return;
        }
        let clone_id = self.dfa.clone_state(start);
        self.dfa.redirect_transitions(start, clone_id);
    }

    pub fn step(&self, state: u32, byte: u8) -> Option<u32> {
        self.dfa.step(state, byte)
    }

    pub fn get_transition(&self, state: u32, byte: u8) -> u32 {
        self.dfa.get_transition(state, byte)
    }

    pub fn run(&self, input: &[u8]) -> u32 {
        input
            .iter()
            .try_fold(self.start_state(), |state, &byte| self.step(state, byte))
            .unwrap_or(self.start_state())
    }

    pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals_iter(state).collect()
    }

    pub(crate) fn matched_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .finalizers(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    pub(crate) fn possible_future_terminals_iter(
        &self,
        state: u32,
    ) -> impl Iterator<Item = TerminalID> + '_ {
        self.dfa
            .possible_future_group_ids(state)
            .iter()
            .map(|terminal| terminal as TerminalID)
    }

    pub fn all_matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {
        self.matched_terminals(state)
    }

    pub fn possible_future_terminals(&self, state: u32) -> &BitSet {
        self.dfa.possible_future_group_ids(state)
    }

    pub fn is_end(&self, state: u32) -> bool {
        self.possible_future_terminals(state).is_empty()
    }

    pub fn num_states(&self) -> u32 {
        self.dfa.num_states() as u32
    }

    pub(crate) fn execute_from_state_all_widths(
        &self,
        input: &[u8],
        start: u32,
    ) -> TokenizerExecResult {
        let mut matches = Vec::new();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_all_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state: end_state.filter(|&state| !self.is_end(state)),
            matches,
        }
    }

    pub fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {
        let mut matches = FxHashMap::<TerminalID, (usize, u32)>::default();
        let end_state = self.scan_input(input, start, &mut matches, |tokenizer, matches, state, width| {
            tokenizer.record_longest_matches(matches, state, width);
        });

        TokenizerExecResult {
            end_state,
            matches: into_longest_matches(matches),
        }
    }

    pub(crate) fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> Option<u32> {
        self.scan_input(input, start, &mut (), |_, _, _, _| {})
    }

    pub fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {
        let exec = self.execute_from_state_all_widths(input, start);
        let end_state = exec.end_state.unwrap_or(start);
        TokenizerResult {
            end_state,
            matches: group_matches_by_width(exec.matches),
        }
    }

    pub fn initial_state(&self) -> u32 {
        self.start_state()
    }

    pub fn initial_state_id(&self) -> u32 {
        self.initial_state()
    }

    pub fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {
        self.possible_future_terminals(state)
    }

    /// Scan input bytes and report which terminals of interest matched/finalized.
    ///
    /// Returns a bitset of matched terminals and an optional end state.
    ///
    /// Algorithm:
    /// 1. `remaining = terminals_of_interest`.
    /// 2. `matched = empty`.
    /// 3. For each byte:
    ///    - Check if current state's possible futures overlap `remaining`.
    ///      If not, return `(matched, None)`.
    ///    - Consume byte → next state.
    ///    - If no transition, return `(matched, None)`.
    ///    - Get finalizers at next state, intersect with `remaining`.
    ///    - Add intersection to `matched`, remove from `remaining`.
    /// 4. After all bytes, check futures at end state overlap `remaining`.
    ///    If not, return `(matched, None)`. Otherwise `(matched, Some(end_state))`.
    ///
    /// Important: initial-state finalizers are intentionally ignored.
    /// Only post-byte finalizers count.
    ///
    /// `terminals_of_interest` must have length equal to `self.num_terminals`.
    pub fn scan_terminal_matches_from_state(
        &self,
        input: &[u8],
        start: u32,
        terminals_of_interest: &BitSet,
    ) -> (BitSet, Option<u32>) {
        debug_assert_eq!(terminals_of_interest.len(), self.num_terminals as usize);
        let mut remaining = terminals_of_interest.clone();
        let mut matched = BitSet::new(self.num_terminals as usize);
        let mut state = start;

        for &byte in input {
            let futures = self.possible_future_terminals(state);
            if futures.is_disjoint(&remaining) {
                return (matched, None);
            }

            let next = match self.step(state, byte) {
                Some(s) => s,
                None => return (matched, None),
            };

            let finals = self.dfa.finalizers(next).intersection(&remaining);
            matched.union_with(&finals);
            remaining = remaining.difference(&finals);
            state = next;
        }

        let futures = self.possible_future_terminals(state);
        if futures.is_disjoint(&remaining) {
            (matched, None)
        } else {
            (matched, Some(state))
        }
    }

    fn has_incoming_start_transitions(&self, start: u32) -> bool {
        self.dfa
            .states()
            .iter()
            .any(|state| state.transitions.values().any(|&target| target == start))
    }

    fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: u32, width: usize) {
        matches.extend(self.matched_terminals_iter(state).map(|id| TokenizerMatch {
            id,
            width,
            end_state: state,
        }));
    }

    fn record_longest_matches(
        &self,
        matches: &mut FxHashMap<TerminalID, (usize, u32)>,
        state: u32,
        width: usize,
    ) {
        for terminal in self.matched_terminals_iter(state) {
            matches.insert(terminal, (width, state));
        }
    }

    fn scan_input<R>(
        &self,
        input: &[u8],
        start: u32,
        mut matches: &mut R,
        mut record_matches: impl FnMut(&Self, &mut R, u32, usize),
    ) -> Option<u32> {
        let mut state = start;
        for (index, &byte) in input.iter().enumerate() {
            let next = self.step(state, byte)?;
            state = next;
            record_matches(self, &mut matches, state, index + 1);
        }
        Some(state)
    }

    fn filter_dfa_for_terminals(
        &self,
        active_terminals: &[bool],
        relevant_bytes: Option<&[bool; 256]>,
    ) -> TerminalFilteredDfa {
        let mut dfa = self.dfa.clone();

        let num_groups = self.num_terminals as usize;
        let mut active_bitset = BitSet::new(num_groups);
        for (tid, &active) in active_terminals.iter().enumerate() {
            if active {
                active_bitset.set(tid);
            }
        }

        let mut any_cleared = false;
        let mut transitions_pruned = false;
        for state in dfa.states_mut().iter_mut() {
            if let Some(relevant_bytes) = relevant_bytes {
                let mut filtered_transitions = Vec::with_capacity(state.transitions.len());
                for (byte, &target) in state.transitions.iter() {
                    if relevant_bytes[byte as usize] {
                        filtered_transitions.push((byte, target));
                    }
                }
                if filtered_transitions.len() != state.transitions.len() {
                    state.transitions =
                        crate::ds::char_transitions::CharTransitions::from_sorted_entries(
                            filtered_transitions,
                        );
                    transitions_pruned = true;
                }
            }

            if state.finalizers.len() == active_bitset.len()
                && !state.finalizers.is_subset(&active_bitset)
            {
                state.finalizers.intersect_with(&active_bitset);
                any_cleared = true;
            } else {
                for (terminal_id, active) in active_terminals.iter().enumerate() {
                    if !active
                        && terminal_id < state.finalizers.len()
                        && state.finalizers.contains(terminal_id)
                    {
                        state.finalizers.clear(terminal_id);
                        any_cleared = true;
                    }
                }
            }
        }

        let num_states_after_filter = dfa.num_states();
        let mut is_dead = vec![false; num_states_after_filter];
        for state_id in 0..num_states_after_filter {
            let state = &dfa.states()[state_id];
            let final_active = !state.finalizers.is_disjoint(&active_bitset);
            let future_active = !self
                .dfa
                .possible_future_group_ids(state_id as u32)
                .is_disjoint(&active_bitset);
            if !final_active && !future_active {
                is_dead[state_id] = true;
            }
        }
        let mut coreach_pruned = false;
        for state in dfa.states_mut().iter_mut() {
            let original_len = state.transitions.len();
            if original_len == 0 {
                continue;
            }
            let mut filtered = Vec::with_capacity(original_len);
            for (byte, &target) in state.transitions.iter() {
                if !is_dead[target as usize] {
                    filtered.push((byte, target));
                }
            }
            if filtered.len() != original_len {
                state.transitions =
                    crate::ds::char_transitions::CharTransitions::from_sorted_entries(filtered);
                coreach_pruned = true;
            }
        }
        if coreach_pruned {
            any_cleared = true;
        }

        TerminalFilteredDfa {
            dfa,
            active_bitset,
            any_cleared,
            transitions_pruned,
        }
    }

    pub(crate) fn clone_filtered_for_terminals(
        &self,
        active_terminals: &[bool],
        relevant_bytes: &[bool; 256],
    ) -> Tokenizer {
        let mut filtered = self.filter_dfa_for_terminals(active_terminals, Some(relevant_bytes));
        filtered.dfa.recompute_possible_futures();
        Tokenizer {
            dfa: filtered.dfa,
            num_terminals: self.num_terminals,
            exprs: self.exprs.clone(),
        }
    }

    /// Check whether filtering to `active_terminals` can produce a total
    /// original-state map.  Returns `false` if any original DFA state has
    /// neither an active finalizer nor an active future (i.e., the state
    /// would be dead after filtering), which makes the simplified state map
    /// non-total.
    pub fn active_terminal_filter_can_preserve_total_state_map(
        &self,
        active_terminals: &[bool],
    ) -> bool {
        let num_groups = self.num_terminals as usize;
        let mut active_bitset = BitSet::new(num_groups);
        for (terminal_id, &active) in active_terminals.iter().enumerate() {
            if active {
                active_bitset.set(terminal_id);
            }
        }

        for state_id in 0..self.dfa.num_states() {
            let state = &self.dfa.states()[state_id];
            let final_active = !state.finalizers.is_disjoint(&active_bitset);
            let future_active = !self
                .dfa
                .possible_future_group_ids(state_id as u32)
                .is_disjoint(&active_bitset);
            if !final_active && !future_active {
                return false;
            }
        }
        true
    }

    /// Create a simplified tokenizer that only knows about `active_terminals`.
    ///
    /// Non-active terminal bits are cleared from finalizers and the DFA is
    /// minimized. The `relevant_bytes` parameter is accepted for API
    /// compatibility, but transition-byte pruning is only honored when the
    /// `GLRMASK_FORCE_RELEVANT_BYTES` environment variable is set; by default
    /// the method clears inactive terminal metadata and minimizes without byte
    /// pruning, to preserve commit/mask equivalence.
    pub fn simplify_for_terminals(
        &self,
        active_terminals: &[bool],
        relevant_bytes: Option<&[bool; 256]>,
    ) -> (Tokenizer, ManyToOneIdMap) {
        let compile_profile = std::env::var("GLRMASK_PROFILE_COMPILE")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false);

        let use_from_scratch =
            std::env::var_os("GLRMASK_SIMPLIFY_FROM_SCRATCH").is_some() && self.exprs.is_some() && {
            let num_active = active_terminals.iter().filter(|&&active| active).count();
            let total = active_terminals.len();
            num_active * 2 <= total
        };
        if use_from_scratch {
            if let Some(result) = self.simplified_from_active_exprs(active_terminals) {
                if compile_profile {
                    eprintln!(
                        "[glrmask/profile][simplify_detail] from_scratch states={} active={}",
                        result.0.num_states(),
                        active_terminals.iter().filter(|&&active| active).count(),
                    );
                }
                return result;
            }
        }

        let relevant_bytes = if std::env::var_os("GLRMASK_FORCE_RELEVANT_BYTES").is_some() {
            relevant_bytes
        } else {
            None
        };

        let started_at = std::time::Instant::now();
        let TerminalFilteredDfa {
            mut dfa,
            active_bitset,
            any_cleared,
            transitions_pruned,
        } = self.filter_dfa_for_terminals(active_terminals, relevant_bytes);
        let clear_elapsed = started_at.elapsed();

        if !any_cleared && !transitions_pruned {
            let num_states = dfa.num_states();
            let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                (0..num_states as u32).collect(),
                num_states as u32,
            );
            if compile_profile {
                eprintln!(
                    "[glrmask/profile][simplify_detail] states={} no_change clear_ms={:.1}",
                    num_states,
                    clear_elapsed.as_secs_f64() * 1000.0,
                );
            }
            return (
                Tokenizer {
                    dfa,
                    num_terminals: self.num_terminals,
                    exprs: self.exprs.clone(),
                },
                identity,
            );
        }

        let pre_minimize_states = dfa.num_states();
        let num_active = active_terminals.iter().filter(|&&active| active).count();
        if pre_minimize_states > 1000 && num_active > 32 && !transitions_pruned {
            let distinct = dfa.distinct_fingerprint_count();
            if distinct > pre_minimize_states * 9 / 10 {
                dfa.mask_possible_futures(&active_bitset);
                let identity = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                    (0..pre_minimize_states as u32).collect(),
                    pre_minimize_states as u32,
                );
                if compile_profile {
                    eprintln!(
                        "[glrmask/profile][simplify_detail] states={} active={} clear_ms={:.1} skip_minimize(distinct={}/{}) total_ms={:.1}",
                        pre_minimize_states,
                        num_active,
                        clear_elapsed.as_secs_f64() * 1000.0,
                        distinct,
                        pre_minimize_states,
                        started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                return (
                    Tokenizer {
                        dfa,
                        num_terminals: self.num_terminals,
                        exprs: self.exprs.clone(),
                    },
                    identity,
                );
            }
        }

        let minimize_started_at = std::time::Instant::now();
        let preserve_all_original_states = transitions_pruned;
        let (minimized, state_mapping) = if preserve_all_original_states {
            dfa.minimize_with_state_mapping_preserve_all_states()
        } else {
            dfa.minimize_with_state_mapping()
        };
        let minimize_elapsed = minimize_started_at.elapsed();
        let post_minimize_states = minimized.num_states();

        if compile_profile {
            eprintln!(
                "[glrmask/profile][simplify_detail] states={} active={} clear_ms={:.1} minimize_ms={:.1} total_ms={:.1} pre={} post={} reduction={}",
                pre_minimize_states,
                num_active,
                clear_elapsed.as_secs_f64() * 1000.0,
                minimize_elapsed.as_secs_f64() * 1000.0,
                started_at.elapsed().as_secs_f64() * 1000.0,
                pre_minimize_states,
                post_minimize_states,
                pre_minimize_states - post_minimize_states,
            );
        }

        (
            Tokenizer {
                dfa: minimized,
                num_terminals: self.num_terminals,
                exprs: self.exprs.clone(),
            },
            ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                state_mapping,
                post_minimize_states as u32,
            ),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerResult {
    pub end_state: u32,
    pub matches: Vec<(usize, BTreeSet<TerminalID>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::bytes;
    use crate::automata::lexer::regex::parse_regex;
    use crate::compiler::compile::build_tokenizer_from_exprs;

    #[test]
    fn test_execute_from_state_keeps_only_longest_match_per_terminal() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"aa")]);

        let exec = tokenizer.execute_from_state(b"aa", tokenizer.start_state());

        assert_eq!(
            exec.matches,
            vec![
                TokenizerMatch {
                    id: 0,
                    width: 1,
                    end_state: tokenizer.run(b"a"),
                },
                TokenizerMatch {
                    id: 1,
                    width: 2,
                    end_state: tokenizer.run(b"aa"),
                },
            ]
        );
    }

    #[test]
    fn test_execute_from_state_replaces_shorter_match_for_same_terminal() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), parse_regex("a+", true)]);

        let exec = tokenizer.execute_from_state(b"aa", tokenizer.start_state());

        assert_eq!(
            exec.matches,
            vec![
                TokenizerMatch {
                    id: 0,
                    width: 1,
                    end_state: tokenizer.run(b"a"),
                },
                TokenizerMatch {
                    id: 1,
                    width: 2,
                    end_state: tokenizer.run(b"aa"),
                },
            ]
        );
    }

    #[test]
    fn test_execute_all_matches_keeps_all_widths() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), parse_regex("a+", true)]);

        let result = tokenizer.execute_all_matches(b"aa", tokenizer.start_state());

        assert_eq!(
            result.matches,
            vec![
                (1, BTreeSet::from([0, 1])),
                (2, BTreeSet::from([1])),
            ]
        );
    }

    #[test]
    fn test_simplify_for_terminals_preserves_continuation_states_behind_pruned_bytes() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b": "), bytes(b"true")]);
        let colon_state = tokenizer.step(tokenizer.start_state(), b':').unwrap();

        let mut relevant_bytes = [false; 256];
        for byte in [b' ', b't', b'r', b'u', b'e'] {
            relevant_bytes[byte as usize] = true;
        }

        let (simplified, mapping) =
            tokenizer.simplify_for_terminals(&[true, true], Some(&relevant_bytes));
        let simplified_colon_state = mapping.original_to_internal[colon_state as usize];

        assert_ne!(
            simplified_colon_state,
            u32::MAX,
            "continuation states reached through pruned bytes must remain addressable"
        );

        let exec = simplified.execute_from_state_all_widths(b" true", simplified_colon_state);
        assert!(
            exec.matches
                .iter()
                .any(|matched| matched.id == 0 && matched.width == 1),
            "the bridge terminal must still match from the preserved continuation state"
        );
    }

    #[test]
    fn test_scan_terminal_matches_ignores_initial_state_finalizer() {
        // Terminal 0 matches "" or "a"; terminal 1 matches "b".
        let tokenizer = build_tokenizer_from_exprs(&[
            parse_regex("a?", true),
            bytes(b"b"),
        ]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"", tokenizer.start_state(), &interest);
        assert!(matched.is_empty(), "initial state finalizer should be ignored");
        assert!(
            end_state.is_some(),
            "futures should still overlap since terminal 0 can match 'a'"
        );
    }

    #[test]
    fn test_scan_terminal_matches_returns_matched_terminals_after_bytes() {
        // Terminal 0 = "a", terminal 1 = "aa".
        // After consuming "a", terminal 0 matched and terminal 1 remains
        // with a future (can continue with another "a").
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"aa")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        assert!(matched.contains(0), "terminal 0 ('a') should match");
        assert!(!matched.contains(1), "terminal 1 ('aa') should not match yet");
        assert!(end_state.is_some(), "end state should exist because terminal 1 still has a future");
    }

    #[test]
    fn test_scan_terminal_matches_returns_none_when_futures_diverge() {
        // Terminal 0 = "a", terminal 1 = "b".
        // After consuming "a", terminal 1 has no future from the end state.
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        // Terminal 0 matched, terminal 1 remained in `remaining` but has no future.
        assert!(matched.contains(0));
        assert!(!matched.contains(1));
        assert!(
            end_state.is_none(),
            "end state should be None because futures of end state do not overlap remaining"
        );
    }

    #[test]
    fn test_scan_terminal_matches_returns_some_when_end_state_still_has_future() {
        // Terminal 0 = "a", terminal 1 = "ab".
        // After consuming "a", terminal 0 matched. Terminal 1 remains and
        // the end state still has a future for terminal 1 (can continue with 'b').
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"ab")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        assert!(matched.contains(0), "terminal 0 should match");
        assert!(!matched.contains(1), "terminal 1 should not have matched yet");
        assert!(
            end_state.is_some(),
            "end state should be Some because terminal 1 still has a future"
        );
    }

    #[test]
    fn test_scan_terminal_matches_respects_terminals_of_interest_filter() {
        // Terminal 0 = "a", terminal 1 = "b", terminal 2 = "c".
        // We only care about terminal 1 ("b"). Input "a" does not match it,
        // and from the state-after-a there is no future for "b".
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b"), bytes(b"c")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(1);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"a", tokenizer.start_state(), &interest);
        assert!(matched.is_empty(), "no terminal of interest should have matched");
        assert!(
            end_state.is_none(),
            "end state should be None because no future for terminal 1 after 'a'"
        );
    }

    #[test]
    fn test_scan_terminal_matches_early_stop_on_no_transition() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a")]);
        let mut interest = BitSet::new(tokenizer.num_terminals as usize);
        interest.set(0);
        let (matched, end_state) =
            tokenizer.scan_terminal_matches_from_state(b"ab", tokenizer.start_state(), &interest);
        // After "a", terminal 0 matched. Then on "b" there is no transition.
        assert!(matched.contains(0));
        assert!(end_state.is_none());
    }
}
