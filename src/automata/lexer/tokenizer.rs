//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::grammar::flat::TerminalID;
use crate::ds::bitset::BitSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub(crate) dfa: DFA,
    pub num_terminals: u32,
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

fn remap_masked_possible_futures(
    tokenizer: &Tokenizer,
    active_groups: &BitSet,
    state_mapping: &[u32],
    num_new_states: usize,
) -> Vec<BitSet> {
    let mut remapped = (0..num_new_states)
        .map(|_| BitSet::new(active_groups.len()))
        .collect::<Vec<_>>();

    for (old_state, &new_state) in state_mapping.iter().enumerate() {
        if new_state == u32::MAX {
            continue;
        }

        let mut masked = tokenizer.dfa.possible_future_group_ids(old_state as u32).clone();
        masked.intersect_with(active_groups);
        remapped[new_state as usize].union_with(&masked);
    }

    remapped
}

fn state_has_active_continuation(dfa: &DFA, state: usize, active_groups: &BitSet) -> bool {
    !dfa.states()[state].finalizers.is_disjoint(active_groups)
        || !dfa.possible_future_group_ids(state as u32).is_disjoint(active_groups)
}

fn state_needs_preserved_root(dfa: &DFA, state: usize, active_groups: &BitSet) -> bool {
    let dfa_state = &dfa.states()[state];
    !dfa_state.finalizers.is_disjoint(active_groups)
        || (!dfa_state.transitions.is_empty()
            && !dfa.possible_future_group_ids(state as u32).is_disjoint(active_groups))
}

fn collect_pruned_continuation_roots(
    dfa: &DFA,
    pruned_targets: &[Vec<u32>],
    active_groups: &BitSet,
) -> Vec<u32> {
    let num_states = dfa.num_states();
    if num_states == 0 || pruned_targets.len() != num_states {
        return Vec::new();
    }

    let mut reachable = vec![false; num_states];
    let mut queue = vec![0usize];
    reachable[0] = true;
    while let Some(state) = queue.pop() {
        for (_, &next) in dfa.states()[state].transitions.iter() {
            let next = next as usize;
            if !reachable[next] {
                reachable[next] = true;
                queue.push(next);
            }
        }
    }

    let mut visited = vec![false; num_states];
    let mut preserved = vec![false; num_states];
    let mut queue = Vec::new();
    for targets in pruned_targets {
        for &target in targets {
            let target = target as usize;
            if target < num_states && !reachable[target] && !visited[target] {
                visited[target] = true;
                queue.push(target);
            }
        }
    }

    while let Some(state) = queue.pop() {
        if state_needs_preserved_root(dfa, state, active_groups) {
            preserved[state] = true;
        }
        for &next in &pruned_targets[state] {
            let next = next as usize;
            if next < num_states
                && !reachable[next]
                && !visited[next]
                && state_has_active_continuation(dfa, next, active_groups)
            {
                visited[next] = true;
                queue.push(next);
            }
        }
    }

    preserved
        .into_iter()
        .enumerate()
        .filter_map(|(state, keep)| keep.then_some(state as u32))
        .collect()
}

fn append_continuation_alias_states(
    minimized: &mut DFA,
    pruned_dfa: &DFA,
    continuation_roots: &[u32],
    state_mapping: &mut Vec<u32>,
) {
    if continuation_roots.is_empty() {
        return;
    }

    fn ensure_continuation_alias_state(
        minimized: &mut DFA,
        pruned_dfa: &DFA,
        state_mapping: &mut Vec<u32>,
        building: &mut [bool],
        original_state: usize,
    ) -> u32 {
        if state_mapping[original_state] != u32::MAX {
            return state_mapping[original_state];
        }

        if building[original_state] {
            return state_mapping[original_state];
        }

        building[original_state] = true;
        let alias_state = minimized.add_state();
        state_mapping[original_state] = alias_state;

        let original = &pruned_dfa.states()[original_state];
        let mut transitions = Vec::with_capacity(original.transitions.len());
        for (byte, &target) in original.transitions.iter() {
            let mapped_target = ensure_continuation_alias_state(
                minimized,
                pruned_dfa,
                state_mapping,
                building,
                target as usize,
            );
            transitions.push((byte, mapped_target));
        }

        let alias = &mut minimized.states_mut()[alias_state as usize];
        alias.transitions = crate::ds::char_transitions::CharTransitions::from_sorted_entries(transitions);
        alias.finalizers = original.finalizers.clone();
        building[original_state] = false;
        alias_state
    }

    let mut building = vec![false; pruned_dfa.num_states()];
    for &root in continuation_roots {
        let root = root as usize;
        if root < pruned_dfa.num_states() {
            ensure_continuation_alias_state(
                minimized,
                pruned_dfa,
                state_mapping,
                &mut building,
                root,
            );
        }
    }
}

struct TerminalFilteredDfa {
    dfa: DFA,
    active_bitset: BitSet,
    any_cleared: bool,
    transitions_pruned: bool,
    pruned_targets: Option<Vec<Vec<u32>>>,
}

impl Tokenizer {
    pub fn start_state(&self) -> u32 {
        0
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
        let mut active_bitset = crate::ds::bitset::BitSet::new(num_groups);
        for (tid, &active) in active_terminals.iter().enumerate() {
            if active {
                active_bitset.set(tid);
            }
        }

        let mut any_cleared = false;
        let mut transitions_pruned = false;
        let mut pruned_targets = relevant_bytes.map(|_| vec![Vec::new(); dfa.num_states()]);
        for (state_id, state) in dfa.states_mut().iter_mut().enumerate() {
            if let Some(relevant_bytes) = relevant_bytes {
                let mut filtered_transitions = Vec::with_capacity(state.transitions.len());
                for (byte, &target) in state.transitions.iter() {
                    if relevant_bytes[byte as usize] {
                        filtered_transitions.push((byte, target));
                    } else if let Some(pruned_targets) = pruned_targets.as_mut() {
                        pruned_targets[state_id].push(target);
                    }
                }
                if filtered_transitions.len() != state.transitions.len() {
                    state.transitions = crate::ds::char_transitions::CharTransitions::from_sorted_entries(
                        filtered_transitions,
                    );
                    transitions_pruned = true;
                }
            }
            if state.finalizers.len() == active_bitset.len() && !state.finalizers.is_subset(&active_bitset) {
                state.finalizers.intersect_with(&active_bitset);
                any_cleared = true;
            } else {
                for (terminal_id, active) in active_terminals.iter().enumerate() {
                    if !active && terminal_id < state.finalizers.len() && state.finalizers.contains(terminal_id) {
                        state.finalizers.clear(terminal_id);
                        any_cleared = true;
                    }
                }
            }
        }

        TerminalFilteredDfa {
            dfa,
            active_bitset,
            any_cleared,
            transitions_pruned,
            pruned_targets,
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
        }
    }

    /// Create a simplified tokenizer that only knows about `active_terminals`.
    ///
    /// Non-active terminal bits are cleared from all finalizers. When
    /// `relevant_bytes` is provided, transitions on bytes outside that set are
    /// also removed; the resulting DFA is only expected to be used on the
    /// partition's vocab bytes. The DFA is then minimized.
    ///
    /// Returns `(simplified_tokenizer, original_to_simplified_state_map)`.
    /// Unreachable original states map to `u32::MAX`.
    pub fn simplify_for_terminals(
        &self,
        active_terminals: &[bool],
        relevant_bytes: Option<&[bool; 256]>,
    ) -> (Tokenizer, Vec<u32>) {
        let compile_profile = std::env::var("GLRMASK_PROFILE_COMPILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);

        let t_start = std::time::Instant::now();
        let t_clone = t_start.elapsed();
        let TerminalFilteredDfa {
            mut dfa,
            active_bitset,
            any_cleared,
            transitions_pruned,
            pruned_targets,
        } = self.filter_dfa_for_terminals(active_terminals, relevant_bytes);
        let t_clear = t_start.elapsed();

        let continuation_roots = if transitions_pruned {
            collect_pruned_continuation_roots(
                &dfa,
                pruned_targets.as_deref().unwrap_or(&[]),
                &active_bitset,
            )
        } else {
            Vec::new()
        };

        if !any_cleared && !transitions_pruned {
            let n = dfa.num_states();
            let identity: Vec<u32> = (0..n as u32).collect();
            if compile_profile {
                eprintln!(
                    "[glrmask/profile][simplify_detail] states={} no_change clone_ms={:.1} clear_ms={:.1}",
                    n, t_clone.as_secs_f64()*1000.0, (t_clear - t_clone).as_secs_f64()*1000.0,
                );
            }
            return (Tokenizer { dfa, num_terminals: self.num_terminals }, identity);
        }

        let pre_minimize_states = dfa.num_states();

        let num_active = active_terminals.iter().filter(|&&b| b).count();
        if pre_minimize_states > 1000 && num_active > 32 && !transitions_pruned {
            let distinct = dfa.distinct_fingerprint_count();
            let n = pre_minimize_states;
            if distinct > n * 9 / 10 {
                dfa.mask_possible_futures(&active_bitset);
                let identity: Vec<u32> = (0..n as u32).collect();
                if compile_profile {
                    let total = t_start.elapsed();
                    eprintln!(
                        "[glrmask/profile][simplify_detail] states={} active={} clone_ms={:.1} clear_ms={:.1} skip_minimize(distinct={}/{}) total_ms={:.1}",
                        n, num_active, t_clone.as_secs_f64()*1000.0, (t_clear - t_clone).as_secs_f64()*1000.0,
                        distinct, n, total.as_secs_f64()*1000.0,
                    );
                }
                return (Tokenizer { dfa, num_terminals: self.num_terminals }, identity);
            }
        }

        let t_pre_min = std::time::Instant::now();
        let (mut minimized, mut state_mapping) = if num_active <= 16 && pre_minimize_states <= 20_000 {
            match dfa.try_minimize_full_with_state_mapping() {
                Some(result) => result,
                None => {
                    if compile_profile {
                        eprintln!(
                            "[glrmask/profile][simplify_detail] states={} active={} iterative_bail_ms={:.1} falling_through_to_hopcroft",
                            pre_minimize_states, num_active,
                            t_pre_min.elapsed().as_secs_f64()*1000.0,
                        );
                    }
                    dfa.minimize_with_state_mapping()
                }
            }
        } else {
            dfa.minimize_with_state_mapping()
        };

        if transitions_pruned {
            append_continuation_alias_states(
                &mut minimized,
                &dfa,
                &continuation_roots,
                &mut state_mapping,
            );
        };

        if transitions_pruned {
            let remapped_futures = remap_masked_possible_futures(
                self,
                &active_bitset,
                &state_mapping,
                minimized.num_states() as usize,
            );
            for (state, futures) in remapped_futures.into_iter().enumerate() {
                minimized.set_possible_future_group_ids(state as u32, futures);
            }
        }

        let t_minimize = t_pre_min.elapsed();
        let post_minimize_states = minimized.num_states();

        if compile_profile {
            let total = t_start.elapsed();
            eprintln!(
                "[glrmask/profile][simplify_detail] states={} active={} clone_ms={:.1} clear_ms={:.1} minimize_ms={:.1} total_ms={:.1} pre={} post={} reduction={}",
                pre_minimize_states, num_active,
                t_clone.as_secs_f64()*1000.0,
                (t_clear - t_clone).as_secs_f64()*1000.0,
                t_minimize.as_secs_f64()*1000.0,
                total.as_secs_f64()*1000.0,
                pre_minimize_states, post_minimize_states, pre_minimize_states - post_minimize_states,
            );
        }

        let simplified = Tokenizer {
            dfa: minimized,
            num_terminals: self.num_terminals,
        };

        (simplified, state_mapping)
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
    fn test_simplify_for_terminals_preserves_futures_for_pruned_bytes() {
        let tokenizer = build_tokenizer_from_exprs(&[parse_regex("-[0-9]", true)]);
        let dash_state = tokenizer.step(tokenizer.start_state(), b'-').unwrap();
        assert!(tokenizer.possible_future_terminals(dash_state).contains(0));

        let mut relevant_bytes = [false; 256];
        relevant_bytes[b'-' as usize] = true;

        let (simplified, mapping) = tokenizer.simplify_for_terminals(&[true], Some(&relevant_bytes));
        let simplified_dash_state = mapping[dash_state as usize];

        assert_ne!(simplified_dash_state, u32::MAX);
        assert!(simplified.possible_future_terminals(simplified_dash_state).contains(0));
    }

    #[test]
    fn test_simplify_for_terminals_preserves_continuation_states_behind_pruned_bytes() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b": "), bytes(b"true")]);
        let colon_state = tokenizer.step(tokenizer.start_state(), b':').unwrap();

        let mut relevant_bytes = [false; 256];
        for byte in [b' ', b't', b'r', b'u', b'e'] {
            relevant_bytes[byte as usize] = true;
        }

        let (simplified, mapping) = tokenizer.simplify_for_terminals(&[true, true], Some(&relevant_bytes));
        let simplified_colon_state = mapping[colon_state as usize];

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
}
