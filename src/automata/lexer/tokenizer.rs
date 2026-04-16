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
        let mut dfa = self.dfa.clone();
        let t_clone = t_start.elapsed();

        // Build active-groups BitSet for masking possible_future_group_ids.
        let num_groups = self.num_terminals as usize;
        let mut active_bitset = crate::ds::bitset::BitSet::new(num_groups);
        for (tid, &active) in active_terminals.iter().enumerate() {
            if active {
                active_bitset.set(tid);
            }
        }

        let mut any_cleared = false;
        let mut transitions_pruned = false;
        for state in dfa.states_mut() {
            if let Some(relevant_bytes) = relevant_bytes {
                let filtered_transitions: Vec<(u8, u32)> = state
                    .transitions
                    .iter()
                    .filter(|(byte, _)| relevant_bytes[*byte as usize])
                    .map(|(byte, &target)| (byte, target))
                    .collect();
                if filtered_transitions.len() != state.transitions.len() {
                    state.transitions = crate::ds::char_transitions::CharTransitions::from_sorted_entries(
                        filtered_transitions,
                    );
                    transitions_pruned = true;
                }
            }
            for (terminal_id, active) in active_terminals.iter().enumerate() {
                if !active && terminal_id < state.finalizers.len() && state.finalizers.contains(terminal_id) {
                    state.finalizers.clear(terminal_id);
                    any_cleared = true;
                }
            }
        }
        let t_clear = t_start.elapsed();

        if !any_cleared && !transitions_pruned {
            // No finalizer bits or transitions changed.
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
        
        // For large DFAs, check if minimize would early-return with zero
        // reduction. minimize_impl skips Hopcroft when topology_prerefine
        // finds >90% of blocks are unique. We can predict this cheaply using
        // fingerprints: if >90% of (transitions, finalizers) fingerprints are
        // distinct, minimize will certainly early-return unchanged.
        //
        // This check applies to ALL cases (including few active groups).
        // Counting DFAs (from maxLength constraints) have genuinely distinct
        // transitions even with few active groups, making minimize O(n log n)
        // on 77K+ states with 0 reduction.
        let num_active = active_terminals.iter().filter(|&&b| b).count();
        // For large active sets the masking changes very few finalizer bits, so
        // the transition topology already distinguishes states and the
        // fingerprint check is a valid fast-path. But for small active sets
        // the fingerprint check is wrong: it hashes raw target state IDs, so
        // two states A→C and A→D look distinct even when C and D are
        // themselves equivalent after masking (a deep equivalence the local
        // check cannot see). Skip the fingerprint heuristic for small active
        // sets and let Hopcroft discover actual merges.
        if pre_minimize_states > 1000 && num_active > 32 && !transitions_pruned {
            let distinct = dfa.distinct_fingerprint_count();
            let n = pre_minimize_states;
            if distinct > n * 9 / 10 {
                // minimize would early-return with no reduction. Skip the
                // expensive clone + SCC + partition work.
                // Instead of recompute_possible_futures (which does full SCC),
                // just mask existing possible_futures with active_groups.
                // This is correct because only finalizer bits changed; active
                // groups' reachability is unchanged.
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
        // When few groups are active, try fast iterative refinement to
        // discover deep equivalences. For very large DFAs (>20K states),
        // iterative refinement rarely converges in the 6-iteration budget
        // due to deep chain structure, so go straight to Hopcroft minimize
        // which converges regardless. For smaller DFAs, try iterative first
        // (fast when it converges) and fall through to Hopcroft if it doesn't.
        let (mut minimized, state_mapping) = if num_active <= 16 && pre_minimize_states <= 20_000 {
            match dfa.try_minimize_full_with_state_mapping() {
                Some(result) => result,
                None => {
                    // Iterative refinement didn't converge — use full minimize.
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
}
