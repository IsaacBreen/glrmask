//! NWA builder: trie-walk construction of the terminal NWA.
//!
//! Contains the core `TerminalNwaBuilder` that walks the vocab prefix trie
//! and constructs NWA transitions for each (byte, tokenizer-state) pair.

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerMatch};
use crate::automata::weighted::nwa::NWA;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesByState,
    PossibleMatchesComputer,
};
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::ds::weight::Weight;

use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    ColorId, TerminalColoring, TerminalDwaBuildProfile, TerminalPathLength,
    debug_profile_enabled,
};

/// NWA state identifier (index into `NWA.states`).
type NwaState = u32;
/// Tokenizer state identifier.
type TokenizerState = u32;
type LeafTokenIds = SmallVec<[u32; 8]>;
type FutureTerminalColorGroups = SmallVec<[(ColorId, SmallVec<[TerminalID; 4]>); 8]>;

fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([0..=max_token_id]),
    )
}

#[derive(Clone)]
pub(crate) struct NodesByTokenizerState {
    pub(crate) entries: FxHashMap<TokenizerState, Vec<NwaState>>,
}

impl NodesByTokenizerState {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn merge(&mut self, state: TokenizerState, nodes: &[NwaState]) {
        self.entries.entry(state).or_default().extend_from_slice(nodes);
    }

    fn first(&self, state: TokenizerState) -> Option<NwaState> {
        self.entries.get(&state).and_then(|nodes| nodes.first().copied())
    }

    fn push_one(&mut self, state: TokenizerState, node: NwaState) {
        self.entries.entry(state).or_default().push(node);
    }

    fn iter(&self) -> impl Iterator<Item = (TokenizerState, &[NwaState])> {
        self.entries
            .iter()
            .map(|(&state, nodes)| (state, nodes.as_slice()))
    }
}

impl IntoIterator for NodesByTokenizerState {
    type Item = (TokenizerState, Vec<NwaState>);
    type IntoIter = <FxHashMap<TokenizerState, Vec<NwaState>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

pub(crate) struct TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    tokenizer: &'tok Tokenizer,
    terminal_coloring: TerminalColoring,
    possible_future_terminals: FxHashMap<TokenizerState, Vec<TerminalID>>,
    future_terminal_color_groups: FxHashMap<TokenizerState, FutureTerminalColorGroups>,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    use_terminal_coloring: bool,
    terminal_path_lengths: Option<Vec<TerminalPathLength>>,
    active_terminals: Option<Vec<bool>>,
    self_loop_bytes: FxHashMap<TokenizerState, U8Set>,
    leaf_token_ids_buffer: Vec<Vec<LeafTokenIds>>,
    future_leaf_buffer: FxHashMap<(u32, TokenizerState, ColorId), BufferedLeafTransition>,
    reachable_weight_cache: HashMap<usize, Weight>,
    pruned_weight_cache: HashMap<(usize, u32, TerminalID), Weight>,
    leaf_weight_cache: HashMap<LeafTokenIds, Weight>,
    transition_buffer: FxHashMap<(u32, i32, u32), Weight>,
    epsilon_buffer: FxHashMap<(u32, u32), Weight>,
    pub(crate) profile: TerminalDwaBuildProfile,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
}

#[derive(Default)]
struct BufferedLeafTransition {
    token_ids: LeafTokenIds,
    weight: Option<Weight>,
}

impl<'tok, 'pm, 'nwa> TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    pub(crate) fn new(
        tokenizer: &'tok Tokenizer,
        terminal_coloring: TerminalColoring,
        possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
        nwa: &'nwa mut NWA,
        num_tsids: u32,
        leaf_state: u32,
        ignore_terminal: Option<TerminalID>,
        use_terminal_coloring: bool,
        terminal_path_lengths: Option<Vec<TerminalPathLength>>,
        active_terminals: Option<Vec<bool>>,
        num_tokenizer_states: usize,
    ) -> Self {
        Self {
            tokenizer,
            terminal_coloring,
            possible_future_terminals: FxHashMap::default(),
            future_terminal_color_groups: FxHashMap::default(),
            possible_matches,
            nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            use_terminal_coloring,
            terminal_path_lengths,
            active_terminals,
            self_loop_bytes: FxHashMap::default(),
            leaf_token_ids_buffer: Vec::new(),
            future_leaf_buffer: FxHashMap::default(),
            reachable_weight_cache: HashMap::new(),
            pruned_weight_cache: HashMap::new(),
            leaf_weight_cache: HashMap::new(),
            transition_buffer: FxHashMap::default(),
            epsilon_buffer: FxHashMap::default(),
            profile: TerminalDwaBuildProfile::default(),
            flat_transitions: vec![None; num_tokenizer_states],
        }
    }

    /// O(1) DFA step using lazily-built flat transition table.
    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            let dfa_state = &self.tokenizer.dfa.states()[state_idx];
            let mut flat = Box::new([u32::MAX; 256]);
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            self.flat_transitions[state_idx] = Some(flat);
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn leaf_token_ids_for(&mut self, source: u32, label: TerminalID) -> &mut LeafTokenIds {
        let source_idx = source as usize;
        if source_idx >= self.leaf_token_ids_buffer.len() {
            self.leaf_token_ids_buffer.resize_with(source_idx + 1, Vec::new);
        }

        let labels = &mut self.leaf_token_ids_buffer[source_idx];
        let label_idx = label as usize;
        if label_idx >= labels.len() {
            labels.resize_with(label_idx + 1, SmallVec::new);
        }

        &mut labels[label_idx]
    }

    fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {
        self.leaf_token_ids_for(source, label).push(internal_token_id);
    }

    fn possible_future_terminals_for_state(&mut self, tokenizer_state: TokenizerState) -> Vec<TerminalID> {
        self.possible_future_terminals
            .entry(tokenizer_state)
            .or_insert_with(|| {
                self.tokenizer
                    .possible_future_terminals_iter(tokenizer_state)
                    .collect()
            })
            .clone()
    }

    fn future_terminal_color_groups_for_state(
        &mut self,
        tokenizer_state: TokenizerState,
    ) -> FutureTerminalColorGroups {
        self.future_terminal_color_groups
            .entry(tokenizer_state)
            .or_insert_with(|| {
                let mut groups = BTreeMap::<ColorId, SmallVec<[TerminalID; 4]>>::new();
                for terminal_id in self.tokenizer.possible_future_terminals_iter(tokenizer_state) {
                    if Some(terminal_id) == self.ignore_terminal {
                        continue;
                    }
                    groups
                        .entry(self.terminal_coloring.color_for(terminal_id))
                        .or_default()
                        .push(terminal_id);
                }
                groups.into_iter().collect()
            })
            .clone()
    }

    fn buffer_future_leaf_token_id(
        &mut self,
        source: u32,
        internal_tsid: TokenizerState,
        color: ColorId,
        internal_token_id: u32,
    ) {
        self.profile.future_terminal_additions += 1;
        self.future_leaf_buffer
            .entry((source, internal_tsid, color))
            .or_default()
            .token_ids
            .push(internal_token_id);
    }

    fn add_future_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        internal_token_id: u32,
    ) {
        if !self.use_terminal_coloring {
            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for terminal_id in future_terminals {
                self.add_leaf_token_from_sources(sources, terminal_id, internal_token_id);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal {
            if self
                .possible_future_terminals_for_state(tokenizer_state)
                .contains(&ignore_terminal)
            {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_leaf_token_from_sources(sources, ignore_terminal, internal_token_id);
            }
        }

        let color_groups = self.future_terminal_color_groups_for_state(tokenizer_state);
        for (color, terminals) in color_groups {
            if terminals.is_empty() {
                continue;
            }
            for &source in sources {
                self.buffer_future_leaf_token_id(source, tokenizer_state, color, internal_token_id);
            }
        }
    }

    fn add_future_weighted_match_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        weight: &Weight,
    ) {
        if !self.use_terminal_coloring {
            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for terminal_id in future_terminals {
                self.add_match_from_sources(sources, terminal_id, self.leaf_state, weight);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal {
            if self
                .possible_future_terminals_for_state(tokenizer_state)
                .contains(&ignore_terminal)
            {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_match_from_sources(sources, ignore_terminal, self.leaf_state, weight);
            }
        }

        let color_groups = self.future_terminal_color_groups_for_state(tokenizer_state);
        for (color, terminals) in color_groups {
            if terminals.is_empty() || weight.is_empty() {
                continue;
            }
            for &source in sources {
                self.profile.future_terminal_additions += 1;
                let entry = self.future_leaf_buffer
                    .entry((source, tokenizer_state, color))
                    .or_default();
                if let Some(existing) = &mut entry.weight {
                    *existing = existing.union(weight);
                } else {
                    entry.weight = Some(weight.clone());
                }
            }
        }
    }

    fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {
        let cache_key = token_ids as *const RangeSetBlaze<usize> as usize;
        if let Some(weight) = self.reachable_weight_cache.get(&cache_key) {
            return weight.clone();
        }

        let weight = self.token_set_weight_fast(token_ids);
        self.reachable_weight_cache.insert(cache_key, weight.clone());
        weight
    }

    /// Build a weight covering all tsids for the given set of internal token IDs.
    fn token_set_weight_fast(&self, internal_token_ids: &RangeSetBlaze<usize>) -> Weight {
        if self.num_tsids == 0 || internal_token_ids.is_empty() {
            return Weight::empty();
        }
        let tokens: RangeSetBlaze<u32> = internal_token_ids
            .ranges()
            .map(|r| (*r.start() as u32)..=(*r.end() as u32))
            .collect();
        Weight::from_uniform(0..=self.num_tsids - 1, tokens)
    }

    fn cached_leaf_weight(&mut self, mut token_ids: LeafTokenIds) -> Weight {
        token_ids.sort_unstable();
        token_ids.dedup();

        if let Some(weight) = self.leaf_weight_cache.get(&token_ids) {
            return weight.clone();
        }

        let tokens = RangeSetBlaze::from_iter(token_ids.iter().copied().map(|id| id..=id));
        let weight = Weight::from_uniform(0..=self.num_tsids - 1, tokens);
        self.leaf_weight_cache.insert(token_ids, weight.clone());
        weight
    }

    fn continuation_weight_for_match(
        &mut self,
        child_node: &VocabPrefixTreeNode,
        leaf_token_id: u32,
        terminal_id: TerminalID,
        end_state: Option<u32>,
        completes_segment: bool,
    ) -> Option<Weight> {
        if !(completes_segment && child_node.has_token()) {
            return Some(self.cached_reachable_weight(child_node.reachable_token_ids()));
        }

        let cache_key = (
            child_node as *const VocabPrefixTreeNode as usize,
            end_state.unwrap_or(u32::MAX),
            terminal_id,
        );
        if let Some(weight) = self.pruned_weight_cache.get(&cache_key) {
            return Some(weight.clone());
        }

        let mut remaining = child_node.reachable_token_ids().clone();
        remaining.remove(leaf_token_id as usize);

        if let Some(end_state) = end_state {
            let possible_matches = self
                .possible_matches
                .possible_matches_for_node(child_node, end_state);
            if let Some(matches_for_terminal) = possible_matches.get(&terminal_id) {
                subtract_possible_matches(&mut remaining, matches_for_terminal);
            }
        }

        if remaining.is_empty() {
            return None;
        }

        let weight = self.token_set_weight_fast(&remaining);
        self.pruned_weight_cache.insert(cache_key, weight.clone());
        Some(weight)
    }

    fn add_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        internal_token_id: u32,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = if self.num_tsids == 0 {
                Weight::empty()
            } else {
                Weight::from_uniform(
                    0..=self.num_tsids - 1,
                    RangeSetBlaze::from_iter([internal_token_id..=internal_token_id]),
                )
            };
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id(source, label, internal_token_id);
        }
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: TokenizerState,
    ) -> bool {
        let self_loop_bytes = self.self_loop_bytes.entry(tokenizer_state).or_insert_with(|| {
            let state = &self.tokenizer.dfa.states()[tokenizer_state as usize];
            let mut bytes = U8Set::empty();
            for (byte, &target) in state.transitions.iter() {
                if target == tokenizer_state {
                    bytes.insert(byte);
                }
            }
            bytes
        });
        U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
    }

    fn emit_self_loop_leaf_only_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        let mut accessible = node.reachable_token_ids().clone();
        if node.has_token() {
            accessible.remove(node.token_id() as usize);
        }
        if accessible.is_empty() {
            return;
        }
        let accessible_weight = self.token_set_weight_fast(&accessible);
        for (internal_tsid, source_nodes) in assoc_by_state.iter() {
            self.add_future_weighted_match_from_sources(
                source_nodes,
                internal_tsid,
                &accessible_weight,
            );
        }
    }

    fn add_match_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        target: u32,
        weight: &Weight,
    ) {
        if self.ignore_terminal == Some(label) {
            for &source in sources {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        } else {
            for &source in sources {
                self.transition_buffer
                    .entry((source, label as i32, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
    }

    pub(crate) fn flush_transition_buffer(&mut self) {
        let flush_start = std::time::Instant::now();
        let mut leaf_transition_buckets: Vec<FxHashMap<i32, BufferedLeafTransition>> =
            (0..self.nwa.states.len()).map(|_| FxHashMap::default()).collect();

        let leaf_buf_count = self.leaf_token_ids_buffer.len();
        for (from, labels_vec) in std::mem::take(&mut self.leaf_token_ids_buffer)
            .into_iter()
            .enumerate()
        {
            for (label_idx, token_ids) in labels_vec.into_iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                leaf_transition_buckets[from]
                    .entry(label_idx as i32)
                    .or_default()
                    .token_ids
                    .extend(token_ids);
            }
        }
        let flush_leaf_ms = flush_start.elapsed().as_secs_f64() * 1000.0;

        let future_buf_count = self.future_leaf_buffer.len();
        let flush_future_start = std::time::Instant::now();
        for ((source, tokenizer_state, color), buffered) in
            std::mem::take(&mut self.future_leaf_buffer)
        {
            if buffered.token_ids.is_empty() && buffered.weight.as_ref().map_or(true, |w| w.is_empty()) {
                continue;
            }
            let color_groups = self.future_terminal_color_groups_for_state(tokenizer_state);
            let terminals = color_groups
                .iter()
                .find_map(|(group_color, terminals)| (*group_color == color).then_some(terminals.to_vec()))
                .unwrap_or_default();
            for terminal_id in terminals {
                let entry = leaf_transition_buckets[source as usize]
                    .entry(terminal_id as i32)
                    .or_default();
                if !buffered.token_ids.is_empty() {
                    entry.token_ids.extend_from_slice(&buffered.token_ids);
                }
                if let Some(w) = &buffered.weight {
                    if let Some(existing) = &mut entry.weight {
                        *existing = existing.union(w);
                    } else {
                        entry.weight = Some(w.clone());
                    }
                }
            }
        }
        let flush_future_ms = flush_future_start.elapsed().as_secs_f64() * 1000.0;

        let flush_weight_start = std::time::Instant::now();
        let mut epsilon_entries: Vec<_> = std::mem::take(&mut self.epsilon_buffer).into_iter().collect();
        epsilon_entries.sort_unstable_by_key(|((from, target), _)| (*from, *target));
        for ((from, target), weight) in epsilon_entries {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered epsilon source state must exist");
            state.epsilons.push((target, weight));
        }

        let mut transition_entries: Vec<_> = std::mem::take(&mut self.transition_buffer).into_iter().collect();
        transition_entries.sort_unstable_by_key(|((from, label, target), _)| (*from, *label, *target));
        for ((from, label, target), weight) in transition_entries {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered transition source state must exist");
            state.transitions.entry(label).or_default().push((target, weight));
        }

        for (from, bucket) in leaf_transition_buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            let mut entries: Vec<(i32, BufferedLeafTransition)> = bucket.into_iter().collect();
            entries.sort_unstable_by_key(|(label, _)| *label);

            let mut finalized_entries = Vec::with_capacity(entries.len());
            for (label, mut entry) in entries {
                let mut weight = entry.weight.take().unwrap_or_else(Weight::empty);
                if !entry.token_ids.is_empty() {
                    let token_weight = self.cached_leaf_weight(entry.token_ids);
                    weight = if weight.is_empty() {
                        token_weight
                    } else {
                        weight.union(&token_weight)
                    };
                }
                if !weight.is_empty() {
                    finalized_entries.push((label, weight));
                }
            }

            let state = self
                .nwa
                .states
                .get_mut(from)
                .expect("buffered leaf transition source state must exist");
            for (label, weight) in finalized_entries {
                state.transitions.entry(label).or_default().push((self.leaf_state, weight));
            }
        }
        let flush_weight_ms = flush_weight_start.elapsed().as_secs_f64() * 1000.0;

        if debug_profile_enabled() {
            eprintln!(
                "[glrmask/debug][flush] leaf_buf={} future_buf={} flush_leaf_ms={:.1} flush_future_ms={:.1} flush_weight_ms={:.1}",
                leaf_buf_count, future_buf_count, flush_leaf_ms, flush_future_ms, flush_weight_ms,
            );
        }
    }

    /// Fast NWA construction for L1-only grammars (all terminals have path
    /// length ≤ 1).  Replaces the trie walk with a simple flat loop over
    /// internal vocab × state class representatives.
    ///
    /// Uses a two-phase approach to avoid per-token future-leaf buffering:
    /// Phase 1: Walk all (token, state) pairs, collecting:
    ///   - Terminal matches at token endpoint → direct leaf buffering
    ///   - Alive pairs grouped by (start_state, end_state) → batched token lists
    /// Phase 2: Process grouped alive pairs, adding future leaves in batch.
    pub(crate) fn build_l1_fast(
        &mut self,
        internal_vocab: &[(u32, Vec<u8>)],
        roots_by_tokenizer_state: &NodesByTokenizerState,
        id_map: &InternalIdMap,
    ) {
        // Pre-populate flat transition tables for ALL tokenizer states.
        let num_states = self.tokenizer.num_states() as usize;
        for state_idx in 0..num_states {
            if self.flat_transitions[state_idx].is_none() {
                let dfa_state = &self.tokenizer.dfa.states()[state_idx];
                let mut flat = Box::new([u32::MAX; 256]);
                for (b, &target) in dfa_state.transitions.iter() {
                    flat[b as usize] = target;
                }
                self.flat_transitions[state_idx] = Some(flat);
            }
        }

        // Phase 1: Walk all (token, state) pairs.
        // Group alive pairs by (representative_state, ending_state) to batch future leaves.
        let mut future_groups: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let phase1_start = std::time::Instant::now();
        let mut total_alive: u64 = 0;
        let mut total_pairs: u64 = 0;

        for &(internal_token_id, ref bytes) in internal_vocab {
            for (_tsid_idx, representative_state) in
                id_map.tokenizer_states.iter_representative_ids().enumerate()
            {
                let source_nodes =
                    match roots_by_tokenizer_state.entries.get(&representative_state) {
                        Some(nodes) => nodes.as_slice(),
                        None => continue,
                    };
                if source_nodes.is_empty() {
                    continue;
                }

                total_pairs += 1;

                // Walk bytes using O(1) flat transition table.
                let mut scan_state = representative_state;
                let mut alive = true;
                for &byte in bytes.iter() {
                    let next = unsafe {
                        // SAFETY: scan_state < num_states (maintained by DFA construction)
                        // and flat table is always Some (populated above).
                        *self.flat_transitions
                            .get_unchecked(scan_state as usize)
                            .as_ref()
                            .unwrap_unchecked()
                            .get_unchecked(byte as usize)
                    };
                    if next == u32::MAX {
                        alive = false;
                        break;
                    }
                    scan_state = next;
                }

                if alive {
                    total_alive += 1;
                    // Terminal matches at the exact token endpoint.
                    let finalizers = self.tokenizer.dfa.finalizers(scan_state);
                    for t in finalizers.iter() {
                        let terminal = t as TerminalID;
                        self.profile.match_transition_additions += source_nodes.len() as u64;
                        self.add_leaf_token_from_sources(
                            source_nodes,
                            terminal,
                            internal_token_id,
                        );
                    }

                    // Collect for batched future leaf processing.
                    future_groups
                        .entry((representative_state, scan_state))
                        .or_default()
                        .push(internal_token_id);
                }
            }
        }
        let phase1_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Process future leaves in batch using pre-computed weights.
        // Convert each group's token_ids to a Weight ONCE, then distribute.
        let phase2_start = std::time::Instant::now();
        let num_groups = future_groups.len();
        for ((representative_state, ending_state), token_ids) in future_groups {
            let source_nodes =
                match roots_by_tokenizer_state.entries.get(&representative_state) {
                    Some(nodes) => nodes.as_slice(),
                    None => continue,
                };

            if token_ids.is_empty() {
                continue;
            }

            // Pre-compute weight from token_ids (ONCE per group).
            let leaf_ids: LeafTokenIds = token_ids.into();
            let weight = self.cached_leaf_weight(leaf_ids);

            // Distribute the lightweight Weight (Arc) to all future terminals.
            self.add_future_weighted_match_from_sources(
                source_nodes,
                ending_state,
                &weight,
            );
        }
        let phase2_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;

        if debug_profile_enabled() {
            eprintln!(
                "[glrmask/debug][build_l1_fast] phase1_ms={:.1} phase2_ms={:.1} total_pairs={} alive={} groups={}",
                phase1_ms, phase2_ms, total_pairs, total_alive, num_groups,
            );
        }
    }

    pub(crate) fn build_from_trie(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        let mut recursive_nodes = NodesByTokenizerState::new();
        let mut self_loop_only_nodes = NodesByTokenizerState::new();
        for (tokenizer_state, source_nodes) in assoc_by_state.iter() {
            if self.can_skip_self_loop_subtree(node, tokenizer_state) {
                self_loop_only_nodes.merge(tokenizer_state, source_nodes);
            } else {
                recursive_nodes.merge(tokenizer_state, source_nodes);
            }
        }

        if !self_loop_only_nodes.is_empty() {
            self.emit_self_loop_leaf_only_subtree(node, &self_loop_only_nodes);
        }

        if recursive_nodes.is_empty() {
            return;
        }

        for (segment_bytes, child_node) in node.iter_children() {
            let next_level_nodes = self.process_child_segment(
                segment_bytes,
                child_node,
                &recursive_nodes,
            );
            if !next_level_nodes.is_empty() {
                self.build_from_trie(child_node, &next_level_nodes);
            }
        }
    }

    fn process_child_segment(
        &mut self,
        segment_bytes: &[u8],
        child_node: &VocabPrefixTreeNode,
        initial_nodes: &NodesByTokenizerState,
    ) -> NodesByTokenizerState {
        // Token IDs in the trie are already internal (equivalence class) IDs.
        let leaf_token_id = child_node.token_id() as u32;
        let mut next_level_nodes = NodesByTokenizerState::new();
        let mut pending_by_offset = BTreeMap::<usize, NodesByTokenizerState>::new();
        pending_by_offset.insert(0, initial_nodes.clone());

        // Reusable buffers for DFA execution (avoids per-call allocation)
        let mut match_map_buf = FxHashMap::<TerminalID, (usize, u32)>::default();
        let mut matches_buf: Vec<TokenizerMatch> = Vec::new();

        while let Some((offset, nodes_at_offset)) = pending_by_offset.pop_first() {
            if offset == segment_bytes.len() {
                for (tokenizer_state, nwa_states) in nodes_at_offset {
                    next_level_nodes.merge(tokenizer_state, &nwa_states);
                }
                continue;
            }

            for (tokenizer_state, source_nodes) in nodes_at_offset {
                // Inline DFA scanning with flat transition table for O(1) per-byte stepping
                match_map_buf.clear();
                let mut scan_state = tokenizer_state;
                let mut scan_alive = true;
                for (index, &byte) in segment_bytes[offset..].iter().enumerate() {
                    if let Some(next) = self.fast_step(scan_state, byte) {
                        scan_state = next;
                        // Record longest match per terminal
                        for terminal in self.tokenizer.matched_terminals_iter(scan_state) {
                            // Skip non-active terminals when filtering
                            if let Some(ref active) = self.active_terminals {
                                if !active.get(terminal as usize).copied().unwrap_or(false) {
                                    continue;
                                }
                            }
                            match_map_buf.insert(terminal, (index + 1, scan_state));
                        }
                    } else {
                        scan_alive = false;
                        break;
                    }
                }
                let end_state = if scan_alive { Some(scan_state) } else { None };

                // Collect matches into reusable buffer
                matches_buf.clear();
                for (&id, &(width, end_st)) in match_map_buf.iter() {
                    matches_buf.push(TokenizerMatch { id, width, end_state: end_st });
                }

                if let Some(end_state) = end_state {
                    if child_node.has_token() {
                        self.add_future_leaf_token_from_sources(
                            &source_nodes,
                            end_state,
                            leaf_token_id,
                        );
                    }

                    next_level_nodes.merge(end_state, &source_nodes);
                }

                for matched in &matches_buf {
                    let next_offset = offset + matched.width;

                    if next_offset == segment_bytes.len() && child_node.has_token() {
                        self.profile.match_transition_additions += source_nodes.len() as u64;
                        self.add_leaf_token_from_sources(
                            &source_nodes,
                            matched.id,
                            leaf_token_id,
                        );
                    }

                    // L1 terminals never appear in multi-terminal paths, so
                    // skip continuation processing (no second terminal will
                    // follow within the same token).
                    if let Some(ref lengths) = self.terminal_path_lengths {
                        if let Some(&TerminalPathLength::One) = lengths.get(matched.id as usize) {
                            continue;
                        }
                    }

                    let Some(continuation_weight) = self.continuation_weight_for_match(
                        child_node,
                        leaf_token_id,
                        matched.id,
                        end_state,
                        next_offset == segment_bytes.len(),
                    ) else {
                        continue;
                    };
                    if continuation_weight.is_empty() {
                        continue;
                    }

                    let continuation_nodes = pending_by_offset
                        .entry(next_offset)
                        .or_insert_with(NodesByTokenizerState::new);
                    let destination = ensure_continuation_state(
                        continuation_nodes,
                        self.tokenizer.initial_state_id(),
                        self.nwa,
                    );

                    self.profile.match_transition_additions += source_nodes.len() as u64;
                    self.add_match_from_sources(
                        &source_nodes,
                        matched.id,
                        destination,
                        &continuation_weight,
                    );
                }
            }
        }

        next_level_nodes
    }
}

fn subtract_possible_matches(
    continuation_tokens: &mut RangeSetBlaze<usize>,
    possible_matches: &RangeSetBlaze<u32>,
) {
    for token_id in possible_matches.iter() {
        continuation_tokens.remove(token_id as usize);
    }
}

fn ensure_continuation_state(
    pending: &mut NodesByTokenizerState,
    tokenizer_state: TokenizerState,
    nwa: &mut NWA,
) -> NwaState {
    if let Some(existing) = pending.first(tokenizer_state) {
        return existing;
    }

    let state = nwa.add_state();
    pending.push_one(tokenizer_state, state);
    state
}

pub(crate) fn internal_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {
    id_map
        .vocab_tokens
        .iter_representative_ids()
        .enumerate()
        .filter_map(|(internal_token_id, representative)| {
            vocab
                .entries
                .get(&representative)
                .map(|bytes| (internal_token_id as u32, bytes.clone()))
        })
        .collect()
}

fn root_combined_signature(
    tokenizer: &Tokenizer,
    representative_state: u32,
    internal_tsid: u32,
    terminal_coloring: &TerminalColoring,
    ignore_terminal: Option<TerminalID>,
    possible_matches_by_state: &PossibleMatchesByState,
) -> u64 {
    use std::hash::Hasher;

    let mut future_groups = BTreeMap::<ColorId, smallvec::SmallVec<[TerminalID; 4]>>::new();
    for terminal_id in tokenizer.possible_future_terminals_iter(representative_state) {
        if Some(terminal_id) == ignore_terminal {
            continue;
        }
        future_groups
            .entry(terminal_coloring.color_for(terminal_id))
            .or_default()
            .push(terminal_id);
    }

    let mut future_hasher = std::collections::hash_map::DefaultHasher::new();
    for (color, terminals) in future_groups {
        color.hash(&mut future_hasher);
        terminals.len().hash(&mut future_hasher);
        for terminal_id in terminals {
            terminal_id.hash(&mut future_hasher);
        }
    }
    let future_sig = future_hasher.finish();

    let mut possible_matches_hasher = std::collections::hash_map::DefaultHasher::new();
    if let Some(matches_by_terminal) = possible_matches_by_state.get(&internal_tsid) {
        for (terminal_id, token_ids) in matches_by_terminal {
            terminal_id.hash(&mut possible_matches_hasher);
            for range in token_ids.ranges() {
                range.start().hash(&mut possible_matches_hasher);
                range.end().hash(&mut possible_matches_hasher);
            }
        }
    }
    let possible_matches_sig = possible_matches_hasher.finish();

    let mut combined_hasher = std::collections::hash_map::DefaultHasher::new();
    representative_state.hash(&mut combined_hasher);
    future_sig.hash(&mut combined_hasher);
    possible_matches_sig.hash(&mut combined_hasher);
    combined_hasher.finish()
}

pub(crate) fn seed_root_nodes(
    nwa: &mut NWA,
    start_state: u32,
    tokenizer: &Tokenizer,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    ignore_terminal: Option<TerminalID>,
    possible_matches_by_state: &PossibleMatchesByState,
) -> NodesByTokenizerState {
    let mut roots_by_tokenizer_state = NodesByTokenizerState::new();
    let mut roots_by_signature = HashMap::<u64, NwaState>::new();
    let mut start_weights_by_root = HashMap::<NwaState, Weight>::new();

    for (internal_tsid, representative_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        let combined_sig = root_combined_signature(
            tokenizer,
            representative_state,
            internal_tsid as u32,
            terminal_coloring,
            ignore_terminal,
            possible_matches_by_state,
        );

        let root = *roots_by_signature
            .entry(combined_sig)
            .or_insert_with(|| nwa.add_state());
        let start_weight = all_token_weight(internal_tsid as u32, id_map.max_internal_token_id());
        start_weights_by_root
            .entry(root)
            .and_modify(|existing| *existing = existing.union(&start_weight))
            .or_insert(start_weight);

        roots_by_tokenizer_state.merge(representative_state, &[root]);
    }

    let mut start_weight_entries: Vec<(NwaState, Weight)> = start_weights_by_root.into_iter().collect();
    start_weight_entries.sort_unstable_by_key(|(root, _)| *root);
    for (root, weight) in start_weight_entries {
        nwa.add_epsilon(start_state, root, weight);
    }

    roots_by_tokenizer_state
}

pub(crate) fn build_nwa_via_trie_walk<'a>(
    tokenizer: &'a Tokenizer,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    nwa: &mut NWA,
    leaf_state: u32,
    num_tsids: u32,
    vocab_tree_root: &VocabPrefixTreeNode,
    roots: &NodesByTokenizerState,
    possible_matches: &mut PossibleMatchesComputer<'a>,
    active_terminals: Option<&[bool]>,
) -> TerminalDwaBuildProfile {
    let num_tokenizer_states = tokenizer.num_states() as usize;
    let mut builder = TerminalNwaBuilder::new(
        tokenizer,
        terminal_coloring.clone(),
        possible_matches,
        nwa,
        num_tsids,
        leaf_state,
        ignore_terminal,
        use_terminal_coloring,
        None,
        active_terminals.map(|a| a.to_vec()),
        num_tokenizer_states,
    );
    builder.build_from_trie(vocab_tree_root, roots);
    builder.flush_transition_buffer();
    builder.profile
}
