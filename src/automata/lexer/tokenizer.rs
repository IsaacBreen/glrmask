//! Runtime-facing tokenizer API built on top of the lexer DFA.

use std::collections::BTreeSet;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::automata::dfa::DFA;
use crate::automata::regex::Expr;
use crate::grammar::flat::TerminalID;
use crate::ds::bitset::BitSet;

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

/// Merge alias states that are byte-for-byte identical, never touching
/// main-DFA states (indices `< main_state_count`). Uses a single pass over
/// alias states with a hash signature; redirects redundant aliases to a
/// canonical representative and compacts the DFA.
///
/// This is a cheap structural dedup — it does not collapse transitive
/// equivalences that full Hopcroft minimization would find, but it catches
/// the common case where `append_continuation_alias_states` clones the same
/// original state multiple times via different paths.
fn dedup_alias_states(
    minimized: &mut DFA,
    main_state_count: usize,
    state_mapping: &mut [u32],
) {
    use crate::ds::char_transitions::CharTransitions;
    use rustc_hash::FxHashMap;
    use std::hash::{Hash, Hasher};

    let n = minimized.num_states() as usize;
    if n <= main_state_count {
        return;
    }

    // Hash signature = FxHash of (finalizer bits + sorted transitions).
    // On collision we verify equality.
    let mut sig_hash: FxHashMap<u64, u32> = FxHashMap::default();
    let mut redirect = vec![u32::MAX; n];
    let mut any = false;

    for s in main_state_count..n {
        let st = &minimized.states()[s];
        let mut hasher = rustc_hash::FxHasher::default();
        for bit in st.finalizers.iter_ones() {
            bit.hash(&mut hasher);
        }
        0xAAAA_u32.hash(&mut hasher);
        for (b, &t) in st.transitions.iter() {
            b.hash(&mut hasher);
            t.hash(&mut hasher);
        }
        let key = hasher.finish();
        match sig_hash.get(&key) {
            Some(&canon) => {
                // Verify equality to guard against hash collisions.
                let c = &minimized.states()[canon as usize];
                let s_state = &minimized.states()[s];
                let fins_eq = c.finalizers == s_state.finalizers;
                let trans_eq = c.transitions.iter().count() == s_state.transitions.iter().count()
                    && c.transitions
                        .iter()
                        .zip(s_state.transitions.iter())
                        .all(|((b1, &t1), (b2, &t2))| b1 == b2 && t1 == t2);
                if fins_eq && trans_eq {
                    redirect[s] = canon;
                    any = true;
                }
            }
            None => {
                sig_hash.insert(key, s as u32);
            }
        }
    }

    if !any {
        return;
    }

    // Compact: surviving states get new indices; redirected states inherit
    // their canonical's new index.
    let mut old_to_new = vec![u32::MAX; n];
    let mut new_idx: u32 = 0;
    for s in 0..n {
        if redirect[s] == u32::MAX {
            old_to_new[s] = new_idx;
            new_idx += 1;
        }
    }
    for s in 0..n {
        if redirect[s] != u32::MAX {
            old_to_new[s] = old_to_new[redirect[s] as usize];
        }
    }

    let mut new_states = Vec::with_capacity(new_idx as usize);
    let old_states = std::mem::take(minimized.states_mut());
    for (old_idx, mut st) in old_states.into_iter().enumerate() {
        if redirect[old_idx] != u32::MAX {
            continue;
        }
        let entries: Vec<(u8, u32)> = st
            .transitions
            .iter()
            .map(|(b, &t)| (b, old_to_new[t as usize]))
            .collect();
        st.transitions = CharTransitions::from_sorted_entries(entries);
        new_states.push(st);
    }
    *minimized.states_mut() = new_states;

    for slot in state_mapping.iter_mut() {
        if *slot != u32::MAX {
            *slot = old_to_new[*slot as usize];
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

    /// Build a simplified tokenizer by rebuilding the DFA from scratch using
    /// only the active terminal expressions. Finalizer and possible-future
    /// bitsets in the returned tokenizer are in the ORIGINAL terminal-id
    /// space (inactive bits always clear). Returns `None` if this tokenizer
    /// has no cached `exprs` or the active set is empty.
    ///
    /// The `orig_to_simplified` mapping is computed by a parallel BFS over
    /// (original_state, fresh_state) pairs starting at (0, 0), following
    /// bytes that exist in the fresh DFA. Original states unreachable via
    /// any active-language byte sequence are left as `u32::MAX`.
    pub fn simplified_from_active_exprs(
        &self,
        active_terminals: &[bool],
    ) -> Option<(Tokenizer, Vec<u32>)> {
        use std::collections::VecDeque;

        let exprs = match self.exprs.as_ref() {
            Some(e) => e,
            None => {
                return None;
            }
        };
        if exprs.len() != active_terminals.len() {
            return None;
        }
        let local_to_orig: Vec<u32> = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(i as u32))
            .collect();
        if local_to_orig.is_empty() {
            return None;
        }

        // Build fresh DFA from active exprs only. Local terminal IDs 0..k-1.
        let active_exprs: Vec<Expr> = local_to_orig
            .iter()
            .map(|&orig| exprs[orig as usize].clone())
            .collect();
        let regex = crate::automata::lexer::compile::build_regex(&active_exprs);
        let mut fresh_dfa: DFA = regex.dfa;

        // Remap finalizers and possible_future_group_ids from local IDs
        // (0..num_active-1) to original IDs in a num_terminals-wide bitset.
        let num_terminals = self.num_terminals as usize;
        fresh_dfa.ensure_group_capacity(num_terminals);
        let num_states = fresh_dfa.num_states();
        for s in 0..num_states {
            let old_fins = fresh_dfa.states()[s].finalizers.clone();
            let mut new_fins = BitSet::new(num_terminals);
            for local in old_fins.iter_ones() {
                if let Some(&orig) = local_to_orig.get(local) {
                    new_fins.set(orig as usize);
                }
            }
            let old_fut = fresh_dfa
                .possible_future_group_ids(s as u32)
                .clone();
            let mut new_fut = BitSet::new(num_terminals);
            for local in old_fut.iter_ones() {
                if let Some(&orig) = local_to_orig.get(local) {
                    new_fut.set(orig as usize);
                }
            }
            fresh_dfa.overwrite_state_metadata(s as u32, new_fins, new_fut);
        }

        // Parallel BFS: original state 0 maps to fresh state 0.
        let n_orig = self.dfa.num_states();
        let mut mapping = vec![u32::MAX; n_orig];
        mapping[0] = 0;
        let mut queue: VecDeque<u32> = VecDeque::new();
        queue.push_back(0);
        while let Some(o) = queue.pop_front() {
            let f = mapping[o as usize];
            let o_state = &self.dfa.states()[o as usize];
            for (byte, &o_next) in o_state.transitions.iter() {
                if let Some(f_next) = fresh_dfa.step(f, byte) {
                    let slot = &mut mapping[o_next as usize];
                    if *slot == u32::MAX {
                        *slot = f_next;
                        queue.push_back(o_next);
                    } else {
                        debug_assert_eq!(*slot, f_next,
                            "parallel BFS inconsistency: orig {} mapped to both {} and {}",
                            o_next, *slot, f_next);
                    }
                }
            }
        }

        let tok = Tokenizer {
            dfa: fresh_dfa,
            num_terminals: self.num_terminals,
            exprs: self.exprs.clone(),
        };
        Some((tok, mapping))
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

        // Coreachability prune: remove transitions whose target cannot
        // reach any active terminal. Without this, Hopcroft treats
        // "transition to dead state" and "no transition" as distinguishable,
        // keeping states that differ only in their inactive-terminal
        // sub-structure separate. Pruning these transitions turns them into
        // implicit-trap transitions, letting Hopcroft collapse states with
        // equivalent active-terminal futures but different dead-chain
        // structure.
        //
        // IMPORTANT: use ORIGINAL possible_future_group_ids (from self,
        // before relevant-byte filtering), because continuation-alias states
        // reachable only via pruned bytes must remain "live" so their
        // original futures are preserved for downstream lookups.
        let num_states_after_filter = dfa.num_states();
        let mut is_dead = vec![false; num_states_after_filter];
        for s in 0..num_states_after_filter {
            let st = &dfa.states()[s];
            let final_active = !st.finalizers.is_disjoint(&active_bitset);
            let future_active = !self
                .dfa
                .possible_future_group_ids(s as u32)
                .is_disjoint(&active_bitset);
            if !final_active && !future_active {
                is_dead[s] = true;
            }
        }
        let mut coreach_pruned = false;
        for state in dfa.states_mut().iter_mut() {
            let orig_len = state.transitions.len();
            if orig_len == 0 {
                continue;
            }
            let mut filtered = Vec::with_capacity(orig_len);
            for (byte, &target) in state.transitions.iter() {
                if !is_dead[target as usize] {
                    filtered.push((byte, target));
                }
            }
            if filtered.len() != orig_len {
                state.transitions = crate::ds::char_transitions::CharTransitions::from_sorted_entries(
                    filtered,
                );
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
            exprs: self.exprs.clone(),
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

        // Fast path: when we have cached exprs and the active set is a
        // small fraction of all terminals, rebuild the tokenizer directly
        // from the active expressions. This sidesteps the cost of cloning
        // and minimizing the full ~26k-state original DFA for each
        // partition. When less than half the terminals are active, the
        // fresh DFA is dramatically smaller (e.g. 3k states for 7 active
        // vs 25k for 42), making both simplify and id_map much faster.
        let use_from_scratch = self.exprs.is_some() && {
            let num_active = active_terminals.iter().filter(|&&a| a).count();
            let total = active_terminals.len();
            num_active * 2 <= total
        };
        if use_from_scratch {
            if let Some(result) = self.simplified_from_active_exprs(active_terminals) {
                if compile_profile {
                    eprintln!(
                        "[glrmask/profile][simplify_detail] from_scratch states={} active={}",
                        result.0.num_states(),
                        active_terminals.iter().filter(|&&a| a).count(),
                    );
                }
                return result;
            }
        }

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

        // Diagnostic: count states that cannot reach any active finalizer.
        // Hopcroft will NOT collapse these into a single sink if their
        // transitions still distinguish them (even when every transition
        // eventually leads only to dead states, the distinguishing depth keeps
        // them apart until the algorithm fully propagates). We report the
        // split so we can decide whether to add a coreachability prune pass.
        if std::env::var_os("GLRMASK_DEBUG_SIMPLIFY_COREACH").is_some() {
            let num_states = dfa.num_states();
            let mut coreach = 0usize;
            let mut has_active_final = 0usize;
            for s in 0..num_states {
                let st = &dfa.states()[s];
                let final_active = !st.finalizers.is_disjoint(&active_bitset);
                let future_active = !dfa
                    .possible_future_group_ids(s as u32)
                    .is_disjoint(&active_bitset);
                if final_active {
                    has_active_final += 1;
                }
                if final_active || future_active {
                    coreach += 1;
                }
            }
            eprintln!(
                "[glrmask/debug][simplify_coreach] pre_minimize_states={} coreach={} dead={} has_active_final={} active_terminals={}",
                num_states, coreach, num_states - coreach, has_active_final,
                active_terminals.iter().filter(|&&b| b).count(),
            );
        }

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
            return (Tokenizer { dfa, num_terminals: self.num_terminals, exprs: self.exprs.clone() }, identity);
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
                return (Tokenizer { dfa, num_terminals: self.num_terminals, exprs: self.exprs.clone() }, identity);
            }
        }

        let t_pre_min = std::time::Instant::now();
        let (mut minimized, mut state_mapping) = match dfa.try_minimize_full_with_state_mapping() {
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
        };

        if std::env::var_os("GLRMASK_DEBUG_SIMPLIFY_PHASES").is_some() {
            eprintln!(
                "[glrmask/debug][simplify_phase] active={} pre_min={} post_first_min={} roots={}",
                num_active,
                pre_minimize_states,
                minimized.num_states(),
                continuation_roots.len(),
            );
        }

        if transitions_pruned && !continuation_roots.is_empty() {
            let main_state_count = minimized.num_states() as usize;
            append_continuation_alias_states(
                &mut minimized,
                &dfa,
                &continuation_roots,
                &mut state_mapping,
            );
            // Merge identical alias states (aliases with byte-for-byte equal
            // transitions and finalizers). This is a conservative in-place
            // dedup that never touches main DFA states, preserving the
            // transition shape Hopcroft produced for the filtered DFA while
            // collapsing the redundant alias clones that cloning introduces.
            dedup_alias_states(
                &mut minimized,
                main_state_count,
                &mut state_mapping,
            );

            if std::env::var_os("GLRMASK_DEBUG_SIMPLIFY_PHASES").is_some() {
                eprintln!(
                    "[glrmask/debug][simplify_phase] after_append_dedup states={} main_count={}",
                    minimized.num_states(),
                    main_state_count,
                );
            }

            // Optional: full Hopcroft re-minimize after dedup. We prevent
            // main-state/alias-state cross-merging by tagging every alias
            // with a synthetic finalizer bit before the minimize call so
            // Hopcroft's initial partition separates them. The bit is
            // cleared on the resulting DFA before anyone observes it.
            if std::env::var_os("GLRMASK_DEBUG_REMIN_FULL").is_some() {
                let synthetic_bit = self.num_terminals as usize;
                let num_groups = minimized.num_groups().max(synthetic_bit + 1);
                minimized.ensure_group_capacity(num_groups);
                for s in main_state_count..minimized.num_states() as usize {
                    let mut fins = minimized.states()[s].finalizers.clone();
                    if fins.len() <= synthetic_bit {
                        let mut grown =
                            crate::ds::bitset::BitSet::new(synthetic_bit + 1);
                        for b in fins.iter_ones() {
                            grown.set(b);
                        }
                        fins = grown;
                    }
                    fins.set(synthetic_bit);
                    minimized.states_mut()[s].finalizers = fins;
                }
                let mut roots: Vec<u32> = state_mapping
                    .iter()
                    .copied()
                    .filter(|&m| m != u32::MAX)
                    .collect();
                roots.sort_unstable();
                roots.dedup();
                let pre = minimized.num_states() as usize;
                let (mut remin, remap) =
                    minimized.minimize_with_state_mapping_and_roots(&roots);

                for s in 0..remin.num_states() as usize {
                    if remin.states()[s].finalizers.len() > synthetic_bit
                        && remin.states()[s].finalizers.contains(synthetic_bit)
                    {
                        let mut fins = remin.states()[s].finalizers.clone();
                        fins.clear(synthetic_bit);
                        remin.states_mut()[s].finalizers = fins;
                    }
                }
                eprintln!(
                    "[glrmask/debug][remin_tagged] pre={} post={} main_count={}",
                    pre,
                    remin.num_states(),
                    main_state_count,
                );

                for slot in state_mapping.iter_mut() {
                    if *slot != u32::MAX {
                        *slot = remap[*slot as usize];
                    }
                }
                minimized = remin;
            }
        }

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
            exprs: self.exprs.clone(),
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

    #[test]
    #[ignore]
    fn measure_p2_active_terminals_github_hard_o1051() {
        // The 15 terminals active in l2p partition p2 for Github_hard---o1051.
        // Run with: cargo test --release -- --ignored --nocapture measure_p2_active
        let patterns: &[(&str, &str)] = &[
            ("JSON_BOOL", r#"(?:true|false)"#),
            ("JSON_NULL", r#"null"#),
            (":", r#":"#),
            (",", r#","#),
            (
                "JSON_STRING_CHAR_UPTO_CLOSE_3",
                r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,256}""#,
            ),
            (
                "JSON_STRING_CHAR_EXACT_256_4",
                r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){256}"#,
            ),
            (
                "JSON_STRING_CHAR_UPTO_CLOSE_6",
                r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,256}""#,
            ),
            ("hex_char", r#"[0-9a-fA-F]"#),
            ("uri_char_at", r#"[a-zA-Z0-9\-._~!$&'()*+,;=:@]"#),
            ("uri_char_at_slash_q", r#"[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]"#),
            ("alpha", r#"[a-zA-Z]"#),
            ("scheme_char", r#"[a-zA-Z0-9+\-.]"#),
            ("uri_char_colon", r#"[a-zA-Z0-9\-._~!$&'()*+,;=:]"#),
            ("v_literal", r#"v"#),
            ("uri_char_plain", r#"[a-zA-Z0-9\-._~!$&'()*+,;=]"#),
        ];

        let exprs: Vec<_> = patterns
            .iter()
            .map(|(_, p)| parse_regex(p, true))
            .collect();

        let t0 = std::time::Instant::now();
        let tokenizer = build_tokenizer_from_exprs(&exprs);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[measure_p2] tokenizer_states={} terminals={} build_ms={:.3}",
            tokenizer.num_states(),
            patterns.len(),
            ms
        );

        // Measure each terminal alone.
        for (label, pattern) in patterns {
            let t0 = std::time::Instant::now();
            let tok = build_tokenizer_from_exprs(&[parse_regex(pattern, true)]);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[measure_p2] alone={:30} states={} ms={:.3}",
                label,
                tok.num_states(),
                ms,
            );
        }

        // Measure leave-one-out.
        for (i, (label, _)) in patterns.iter().enumerate() {
            let subset: Vec<_> = exprs
                .iter()
                .enumerate()
                .filter_map(|(j, e)| (j != i).then(|| e.clone()))
                .collect();
            let t0 = std::time::Instant::now();
            let tok = build_tokenizer_from_exprs(&subset);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[measure_p2] drop={:30} states={} ms={:.3}",
                label,
                tok.num_states(),
                ms,
            );
        }
    }

    #[test]
    #[ignore]
    fn measure_simplify_vs_from_scratch_realistic() {
        // Build 15 active + MANY quoted-string-literal decoys (mimicking
        // property-key literals in a JSON schema). These overlap heavily
        // with active JSON_STRING_CHAR regexes at the byte level.
        let active_patterns: &[&str] = &[
            r#"(?:true|false)"#,
            r#"null"#,
            r#":"#,
            r#","#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,256}""#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){256}"#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,256}""#,
            r#"[0-9a-fA-F]"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=:@]"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]"#,
            r#"[a-zA-Z]"#,
            r#"[a-zA-Z0-9+\-.]"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=:]"#,
            r#"v"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=]"#,
        ];
        // ~60 property-key literals enclosed in quotes — these are ALL
        // bounded-length prefixes of JSON_STRING_CHAR body patterns.
        let key_literals: &[&str] = &[
            "repository", "owner", "name", "full_name", "description",
            "url", "html_url", "git_url", "ssh_url", "clone_url",
            "homepage", "language", "forks_count", "stargazers_count",
            "watchers_count", "size", "default_branch", "open_issues_count",
            "is_template", "topics", "has_issues", "has_projects", "has_wiki",
            "has_pages", "has_downloads", "archived", "disabled", "visibility",
            "pushed_at", "created_at", "updated_at", "permissions",
            "allow_rebase_merge", "template_repository", "allow_squash_merge",
            "allow_auto_merge", "delete_branch_on_merge", "allow_merge_commit",
            "subscribers_count", "network_count", "license", "forks",
            "open_issues", "watchers", "node_id", "id", "type", "login",
            "gravatar_id", "avatar_url", "followers_url", "following_url",
            "gists_url", "starred_url", "subscriptions_url", "organizations_url",
            "repos_url", "events_url", "received_events_url", "site_admin",
            "spdx_id", "key", "admin", "pull", "push",
        ];

        let active_exprs: Vec<_> = active_patterns
            .iter()
            .map(|p| parse_regex(p, true))
            .collect();
        let from_scratch = build_tokenizer_from_exprs(&active_exprs);
        eprintln!("[real_sim] from_scratch_states={}", from_scratch.num_states());

        let mut combined_exprs: Vec<_> = active_exprs.clone();
        for k in key_literals {
            // Quoted literal: "<key>" — keys are alphanumeric+_, no regex meta
            let pattern = format!(r#""{}""#, k);
            combined_exprs.push(parse_regex(&pattern, true));
        }
        // Add the o1051 "killer" unbounded-body terminals that don't appear in
        // my naive decoys:
        let killer_patterns: &[&str] = &[
            // JSON_STRING_BODY-like: unbounded body star
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"#,
            // UUID-inside-body regex (id=37)
            r#"(?:(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*(?:[\x30-\x39\x61-\x66]{8}-[\x30-\x39\x61-\x66]{4}-[\x30-\x39\x61-\x66]{4}-[\x30-\x39\x61-\x66]{4}-[\x30-\x39\x61-\x66]{12})(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*)""#,
            // Date-inside-body (id=78)
            r#"(?:([\x30-\x39]{4})(-([\x30-\x39]{2}))?(-([\x30-\x39]{2}))?)""#,
            // Unbounded body then close quote (id=6)
            r#"(?:(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*)""#,
            // IPv4 octet
            r#"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"#,
            // JSON integer / number
            r#"\-?(?:0|[1-9][0-9]*)"#,
            r#"\-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?"#,
        ];
        for p in killer_patterns {
            combined_exprs.push(parse_regex(p, true));
        }
        let combined = build_tokenizer_from_exprs(&combined_exprs);
        eprintln!(
            "[real_sim] combined_states={} active={} inactive={}",
            combined.num_states(),
            active_patterns.len(),
            key_literals.len(),
        );

        let n_total = combined_exprs.len();
        let mut active_mask = vec![false; n_total];
        for i in 0..active_patterns.len() {
            active_mask[i] = true;
        }

        let (simplified, _mapping) = combined.simplify_for_terminals(&active_mask, None);
        eprintln!(
            "[real_sim] simplify_for_terminals states={} (target={} overshoot={:.2}x)",
            simplified.num_states(),
            from_scratch.num_states(),
            simplified.num_states() as f64 / from_scratch.num_states() as f64,
        );

        // Also test with relevant_bytes set to the bytes actually used by active
        // terminals (simulating what the real pipeline does for a partition).
        let mut relevant_bytes = [false; 256];
        for b in b'a'..=b'z' { relevant_bytes[b as usize] = true; }
        for b in b'A'..=b'Z' { relevant_bytes[b as usize] = true; }
        for b in b'0'..=b'9' { relevant_bytes[b as usize] = true; }
        for &b in b"-._~!$&'()*+,;=:@/?\"\\" { relevant_bytes[b as usize] = true; }
        let (simplified_rb, _mapping_rb) =
            combined.simplify_for_terminals(&active_mask, Some(&relevant_bytes));
        eprintln!(
            "[real_sim] simplify_for_terminals_with_rb states={} (vs no_rb={})",
            simplified_rb.num_states(),
            simplified.num_states(),
        );
    }

    #[test]
    #[ignore]
    fn measure_simplify_vs_from_scratch() {
        // Build a tokenizer from 15 active + several "inactive decoy" terminals.
        // Then simplify with only the 15 active and compare to from-scratch.
        let active_patterns: &[&str] = &[
            r#"(?:true|false)"#,
            r#"null"#,
            r#":"#,
            r#","#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,256}""#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){256}"#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,256}""#,
            r#"[0-9a-fA-F]"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=:@]"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=:@/?]"#,
            r#"[a-zA-Z]"#,
            r#"[a-zA-Z0-9+\-.]"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=:]"#,
            r#"v"#,
            r#"[a-zA-Z0-9\-._~!$&'()*+,;=]"#,
        ];
        // Inactive decoys: bounded-repeat date-time-like patterns and other
        // chains that would exist in the original full tokenizer.
        let inactive_patterns: &[&str] = &[
            r#"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+\-][0-9]{2}:[0-9]{2})"#,
            r#"[0-9]{4}-[0-9]{2}-[0-9]{2}"#,
            r#"[0-9]{2}:[0-9]{2}:[0-9]{2}"#,
            r#"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){0,512}""#,
            r#"(?:[^\x00-\x1f\x7f"\\]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4}){128}"#,
            r#"\-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?"#,
        ];

        let from_scratch_exprs: Vec<_> = active_patterns
            .iter()
            .map(|p| parse_regex(p, true))
            .collect();
        let from_scratch = build_tokenizer_from_exprs(&from_scratch_exprs);
        eprintln!("[sim] from_scratch_states={}", from_scratch.num_states());

        // Combined tokenizer: active + inactive
        let mut combined_exprs: Vec<_> = from_scratch_exprs.clone();
        for p in inactive_patterns {
            combined_exprs.push(parse_regex(p, true));
        }
        let combined = build_tokenizer_from_exprs(&combined_exprs);
        eprintln!(
            "[sim] combined_states={} (active={} + inactive={})",
            combined.num_states(),
            active_patterns.len(),
            inactive_patterns.len(),
        );

        // Simplify: only first N terminals active.
        let n_active = active_patterns.len();
        let n_total = combined_exprs.len();
        let mut active_mask = vec![false; n_total];
        for i in 0..n_active {
            active_mask[i] = true;
        }

        let t0 = std::time::Instant::now();
        let (simplified, _mapping) = combined.simplify_for_terminals(&active_mask, None);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[sim] simplify_for_terminals states={} ms={:.3}",
            simplified.num_states(),
            ms,
        );

        // Re-minimize the simplified DFA to check if it's already minimal.
        let (remin, _) = simplified.dfa.minimize_with_state_mapping();
        eprintln!(
            "[sim] re_minimize_states={} (same as above means already minimal)",
            remin.num_states(),
        );

        // Expected: simplified should equal from_scratch. If not, simplify is buggy.
        eprintln!(
            "[sim] DIFF: simplified={} from_scratch={} overshoot={}x",
            simplified.num_states(),
            from_scratch.num_states(),
            simplified.num_states() as f64 / from_scratch.num_states() as f64,
        );
    }
}
