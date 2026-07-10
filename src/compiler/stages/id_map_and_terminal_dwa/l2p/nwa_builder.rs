//! NWA builder: trie-walk construction of the terminal NWA.
//!
//! Contains the core `TerminalNwaBuilder` that walks the vocab prefix trie
//! and constructs NWA transitions for each (byte, tokenizer-state) pair.

use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerMatch};
use crate::automata::weighted::nwa::NWA;
use crate::grammar::flat::TerminalID;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::ds::weight::Weight;

use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    ColorId, TerminalColoring, TerminalDwaBuildProfile, TerminalPathLength,
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
    future_terminal_colors: FxHashMap<TokenizerState, SmallVec<[ColorId; 8]>>,
    ignore_terminal_possible: FxHashMap<TokenizerState, bool>,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    /// States reached from the NWA start using only epsilon transitions.
    ///
    /// Ignore is transparent only after a terminal boundary. An ignore match
    /// from one of these sources is the token's first terminal and must remain
    /// labelled so the parser DWA can account for it with the ignore template.
    initial_source_states: Vec<bool>,
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
        initial_source_states: Vec<bool>,
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
            future_terminal_colors: FxHashMap::default(),
            ignore_terminal_possible: FxHashMap::default(),
            possible_matches,
            nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            initial_source_states,
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
            self.flat_transitions[state_idx] = Some(self.tokenizer.transition_row(state));
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

    fn terminal_is_active(&self, terminal: TerminalID) -> bool {
        self.active_terminals.as_ref().map_or(true, |active| {
            active.get(terminal as usize).copied().unwrap_or(false)
        })
    }

    fn possible_future_terminals_for_state(&mut self, tokenizer_state: TokenizerState) -> Vec<TerminalID> {
        let active = self.active_terminals.clone();
        self.possible_future_terminals
            .entry(tokenizer_state)
            .or_insert_with(|| {
                self.tokenizer
                    .possible_future_terminals_iter(tokenizer_state)
                    .filter(|&terminal| active.as_ref().map_or(true, |mask| {
                        mask.get(terminal as usize).copied().unwrap_or(false)
                    }))
                    .collect()
            })
            .clone()
    }

    fn populate_future_terminal_color_cache(&mut self, tokenizer_state: TokenizerState) {
        if self.future_terminal_color_groups.contains_key(&tokenizer_state) {
            return;
        }

        let mut groups = BTreeMap::<ColorId, SmallVec<[TerminalID; 4]>>::new();
        let mut colors = SmallVec::<[ColorId; 8]>::new();
        let mut ignore_present = false;

        for terminal_id in self.tokenizer.possible_future_terminals_iter(tokenizer_state) {
            if !self.terminal_is_active(terminal_id) {
                continue;
            }
            if Some(terminal_id) == self.ignore_terminal {
                ignore_present = true;
                continue;
            }
            let color = self.terminal_coloring.color_for(terminal_id);
            let entry = groups.entry(color).or_default();
            if entry.is_empty() {
                colors.push(color);
            }
            entry.push(terminal_id);
        }

        self.future_terminal_color_groups
            .insert(tokenizer_state, groups.into_iter().collect());
        self.future_terminal_colors.insert(tokenizer_state, colors);
        self.ignore_terminal_possible
            .insert(tokenizer_state, ignore_present);
    }

    fn ignore_terminal_possible_for_state(&mut self, tokenizer_state: TokenizerState) -> bool {
        if self.ignore_terminal.is_none() {
            return false;
        }
        self.populate_future_terminal_color_cache(tokenizer_state);
        self.ignore_terminal_possible
            .get(&tokenizer_state)
            .copied()
            .unwrap_or(false)
    }

    fn future_terminal_colors_for_state(
        &mut self,
        tokenizer_state: TokenizerState,
    ) -> SmallVec<[ColorId; 8]> {
        self.populate_future_terminal_color_cache(tokenizer_state);
        self.future_terminal_colors
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default()
    }

    fn future_terminal_color_groups_for_state(
        &mut self,
        tokenizer_state: TokenizerState,
    ) -> FutureTerminalColorGroups {
        self.populate_future_terminal_color_cache(tokenizer_state);
        self.future_terminal_color_groups
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default()
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
            if self.ignore_terminal_possible_for_state(tokenizer_state) {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_leaf_token_from_sources(sources, ignore_terminal, internal_token_id);
            }
        }

        let colors = self.future_terminal_colors_for_state(tokenizer_state);
        for color in colors {
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
            if self.ignore_terminal_possible_for_state(tokenizer_state) {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_match_from_sources(sources, ignore_terminal, self.leaf_state, weight);
            }
        }

        let colors = self.future_terminal_colors_for_state(tokenizer_state);
        for color in colors {
            if weight.is_empty() {
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
        let self_loop_bytes = self
            .self_loop_bytes
            .entry(tokenizer_state)
            .or_insert_with(|| self.tokenizer.self_loop_bytes(tokenizer_state));
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
        let lower_ignore_to_epsilon = self.ignore_terminal == Some(label);
        for &source in sources {
            if lower_ignore_to_epsilon
                && !self
                    .initial_source_states
                    .get(source as usize)
                    .copied()
                    .unwrap_or(false)
            {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            } else {
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
            (0..self.nwa.states().len()).map(|_| FxHashMap::default()).collect();

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
        let buffer = std::mem::take(&mut self.future_leaf_buffer);

        // Pre-compute terminal lookups to avoid repeated clone+find_map (157K→~100 unique keys)
        let mut terminal_cache: FxHashMap<(TokenizerState, ColorId), SmallVec<[TerminalID; 4]>> =
            FxHashMap::default();
        for &(_, tokenizer_state, color) in buffer.keys() {
            terminal_cache
                .entry((tokenizer_state, color))
                .or_insert_with(|| {
                    let groups = self.future_terminal_color_groups_for_state(tokenizer_state);
                    groups
                        .iter()
                        .find_map(|(gc, ts)| (*gc == color).then_some(ts.clone()))
                        .unwrap_or_default()
                });
        }

        for ((source, tokenizer_state, color), buffered) in buffer {
            if buffered.token_ids.is_empty() && buffered.weight.as_ref().map_or(true, |w| w.is_empty()) {
                continue;
            }
            let terminals = &terminal_cache[&(tokenizer_state, color)];
            for &terminal_id in terminals {
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
                .states_mut()
                .get_mut(from as usize)
                .expect("buffered epsilon source state must exist");
            state.epsilons.push((target, weight));
        }

        let mut transition_entries: Vec<_> = std::mem::take(&mut self.transition_buffer).into_iter().collect();
        transition_entries.sort_unstable_by_key(|((from, label, target), _)| (*from, *label, *target));
        for ((from, label, target), weight) in transition_entries {
            let state = self
                .nwa
                .states_mut()
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
                .states_mut()
                .get_mut(from)
                .expect("buffered leaf transition source state must exist");
            for (label, weight) in finalized_entries {
                state.transitions.entry(label).or_default().push((self.leaf_state, weight));
            }
        }
        let flush_weight_ms = flush_weight_start.elapsed().as_secs_f64() * 1000.0;

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
                self.flat_transitions[state_idx] =
                    Some(self.tokenizer.transition_row(state_idx as u32));
            }
        }

        // Phase 1: Walk all (token, state) pairs.
        // Group alive pairs by (representative_state, ending_state) to batch future leaves.
        let mut future_groups: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        let phase1_start = std::time::Instant::now();
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
                    // Terminal matches at the exact token endpoint.
                    for terminal in self.tokenizer.matched_terminals_iter(scan_state) {
                        if !self.terminal_is_active(terminal) {
                            continue;
                        }
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

        let mut matches_buf: Vec<TokenizerMatch> = Vec::new();

        while let Some((offset, nodes_at_offset)) = pending_by_offset.pop_first() {
            if offset == segment_bytes.len() {
                for (tokenizer_state, nwa_states) in nodes_at_offset {
                    next_level_nodes.merge(tokenizer_state, &nwa_states);
                }
                continue;
            }

            for (tokenizer_state, source_nodes) in nodes_at_offset {
                let execution = self.tokenizer.execute_from_state(
                    &segment_bytes[offset..],
                    tokenizer_state,
                );
                matches_buf.clear();
                matches_buf.extend(execution.matches.into_iter().filter(|matched| {
                    self.active_terminals.as_ref().map_or(true, |active| {
                        active
                            .get(matched.id as usize)
                            .copied()
                            .unwrap_or(false)
                    })
                }));
                let end_states = execution.end_state;

                for &end_state in &end_states {
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

                    if next_offset == segment_bytes.len() && child_node.has_token() &&
                        !end_states.iter().copied().any(|s| self.possible_future_terminals_for_state(s).contains(&matched.id)) // This one's optional. Might make it a little faster but functionally no difference.
                    {
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
                        Some(matched.end_state),
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

pub(crate) fn seed_root_nodes(
    nwa: &mut NWA,
    start_state: u32,
    id_map: &InternalIdMap,
) -> NodesByTokenizerState {
    let mut roots_by_tokenizer_state = NodesByTokenizerState::new();

    for (internal_tsid, representative_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        let root = nwa.add_state();
        let start_weight = all_token_weight(internal_tsid as u32, id_map.max_internal_token_id());
        nwa.add_epsilon(start_state, root, start_weight);
        roots_by_tokenizer_state.merge(representative_state, &[root]);
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
    let mut initial_source_states = vec![false; nwa.states().len()];
    for (_, source_nodes) in roots.iter() {
        for &source in source_nodes {
            initial_source_states[source as usize] = true;
        }
    }
    let mut builder = TerminalNwaBuilder::new(
        tokenizer,
        terminal_coloring.clone(),
        possible_matches,
        nwa,
        num_tsids,
        leaf_state,
        ignore_terminal,
        initial_source_states,
        use_terminal_coloring,
        None,
        active_terminals.map(|a| a.to_vec()),
        num_tokenizer_states,
    );
    let trie_start = std::time::Instant::now();
    builder.build_from_trie(vocab_tree_root, roots);
    let trie_ms = trie_start.elapsed().as_secs_f64() * 1000.0;

    let flush_start = std::time::Instant::now();
    builder.flush_transition_buffer();
    let flush_ms = flush_start.elapsed().as_secs_f64() * 1000.0;

    let profile = builder.profile;
    // Drop builder to release the mutable borrow on nwa before reading nwa.states.
    drop(builder);

    profile
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex_monolithic as build_regex;
    use crate::automata::weighted::determinize::determinize;
    use crate::automata::weighted::minimize::minimize_owned;
    use crate::compiler::stages::equiv_types::ManyToOneIdMap;
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::terminal_interchangeability::{
        binary_transport_modes, discover_one_round, visible_output_raw_labels,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::types::{
        LocalIdMapTerminalDwa, TerminalDwaPhaseProfile,
    };
    use crate::ds::vocab_prefix_tree::VocabPrefixTree;

    fn one_terminal_tokenizer() -> Tokenizer {
        let expressions = vec![Expr::U8Seq(b" ".to_vec())];
        build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    #[test]
    fn ignore_matches_are_labelled_initially_and_epsilon_later() {
        let tokenizer = one_terminal_tokenizer();
        let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
        let mut nwa = NWA::new(1, 0);
        let initial_source = nwa.add_state();
        let later_source = nwa.add_state();
        let target = nwa.add_state();

        let mut builder = TerminalNwaBuilder::new(
            &tokenizer,
            TerminalColoring::identity(1),
            &mut possible_matches,
            &mut nwa,
            1,
            target,
            Some(0),
            vec![true, false, false],
            false,
            None,
            None,
            tokenizer.num_states() as usize,
        );
        builder.add_match_from_sources(
            &[initial_source, later_source],
            0,
            target,
            &Weight::all(),
        );
        builder.flush_transition_buffer();
        drop(builder);

        assert!(
            nwa.states()[initial_source as usize]
                .transitions
                .get(&0)
                .is_some_and(|edges| edges.iter().any(|(dst, _)| *dst == target)),
            "an initial ignore match must remain a labelled terminal edge"
        );
        assert!(
            nwa.states()[later_source as usize]
                .epsilons
                .iter()
                .any(|(dst, _)| *dst == target),
            "a non-initial ignore match must remain transparent"
        );
        assert!(
            !nwa.states()[initial_source as usize]
                .epsilons
                .iter()
                .any(|(dst, _)| *dst == target),
            "the initial ignore match must not also be lowered to epsilon"
        );
        assert!(
            !nwa.states()[later_source as usize]
                .transitions
                .contains_key(&0),
            "the non-initial ignore match must not become a terminal edge"
        );
    }

    #[test]
    fn member_reconstruction_emits_only_its_target_member() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let tree = VocabPrefixTree::build(&[(0, b"a".to_vec()), (1, b"c".to_vec())]);
        let state_ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
        let id_map = InternalIdMap {
            tokenizer_states:
                ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                    state_ids.clone(), state_ids,
                ),
            vocab_tokens: ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                vec![0, 1], vec![0, 1],
            ),
        };
        let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);
        let identity = TransportScannerStateMap::Explicit(
            (0..tokenizer.num_states()).collect::<Vec<_>>().into(),
        );
        let modes = [
            TerminalNwaTransportMode::ordinary(tokenizer.num_states() as usize),
            TerminalNwaTransportMode::member(identity, 0, 1),
        ];

        build_transport_nwa_via_trie_walk(
            &tokenizer,
            None,
            &mut nwa,
            start,
            leaf,
            &id_map,
            &tree.root,
            &mut possible_matches,
            &[true, false, true],
            &modes,
        );

        let root = nwa.states()[start as usize].epsilons[0].0;
        let transitions = &nwa.states()[root as usize].transitions;
        assert!(transitions.contains_key(&0), "ordinary representative scan must emit rep");
        assert!(transitions.contains_key(&1), "member reconstruction must emit its target member");
        assert!(transitions.contains_key(&2), "ordinary representative scan must retain unrelated rep");
        let member_weight = transitions
            .get(&1)
            .and_then(|edges| {
                edges
                    .iter()
                    .find_map(|(destination, weight)| (*destination == leaf).then_some(weight))
            })
            .expect("member reconstruction must create a leaf edge");
        let member_tokens = member_weight.tokens_for_tsid(0);
        assert!(member_tokens.contains(0), "member must accept the representative token");
        assert!(
            !member_tokens.contains(1),
            "member reconstruction must not emit its member for unrelated raw outputs",
        );
    }

    fn singleton_id_map(num_tokenizer_states: u32, num_tokens: usize) -> InternalIdMap {
        let tokenizer_states = (0..num_tokenizer_states).collect::<Vec<_>>();
        let vocab_tokens = (0..num_tokens as u32).collect::<Vec<_>>();
        InternalIdMap {
            tokenizer_states:
                ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                    tokenizer_states.clone(),
                    tokenizer_states,
                ),
            vocab_tokens: ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                vocab_tokens.clone(),
                vocab_tokens,
            ),
        }
    }

    fn finish_test_artifact(nwa: &NWA, id_map: InternalIdMap) -> LocalIdMapTerminalDwa {
        LocalIdMapTerminalDwa {
            id_map,
            dwa: minimize_owned(determinize(nwa).expect("test NWA must determinize")),
            profile: TerminalDwaPhaseProfile::default(),
        }
    }

    fn build_baseline_test_artifact(
        tokenizer: &Tokenizer,
        tree: &VocabPrefixTree,
        id_map: &InternalIdMap,
    ) -> LocalIdMapTerminalDwa {
        let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);
        let roots = seed_root_nodes(&mut nwa, start, id_map);
        build_nwa_via_trie_walk(
            tokenizer,
            &TerminalColoring::identity(tokenizer.num_terminals() as usize),
            false,
            None,
            &mut nwa,
            leaf,
            id_map.num_tsids(),
            &tree.root,
            &roots,
            &mut possible_matches,
            None,
        );
        finish_test_artifact(&nwa, id_map.clone())
    }

    fn build_transport_test_artifact(
        tokenizer: &Tokenizer,
        tree: &VocabPrefixTree,
        id_map: &InternalIdMap,
        visible_output_raw_labels: &[bool],
        modes: &[TerminalNwaTransportMode],
    ) -> LocalIdMapTerminalDwa {
        let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);
        build_transport_nwa_via_trie_walk(
            tokenizer,
            None,
            &mut nwa,
            start,
            leaf,
            id_map,
            &tree.root,
            &mut possible_matches,
            visible_output_raw_labels,
            modes,
        );
        finish_test_artifact(&nwa, id_map.clone())
    }

    #[test]
    fn transport_shares_one_nwa_root_per_actual_tsid() {
        // A large class makes the old reference builder allocate one root per
        // (actual TSID, mode). The quotient-union builder must retain the
        // exact language while allocating roots only for actual TSIDs.
        let expressions = (0..16)
            .map(|_| Expr::U8Seq(b"a".to_vec()))
            .collect::<Vec<_>>();
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let partition = discover_one_round(
            &tokenizer,
            &vec![true; tokenizer.num_terminals() as usize],
            &[true; 256],
            None,
        );
        let modes = binary_transport_modes(
            &tokenizer,
            &vec![true; tokenizer.num_terminals() as usize],
            &partition,
            &[true; 256],
            None,
        );
        assert!(modes.len() >= 16, "the test requires many transport modes");

        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"aaaa".to_vec()),
        ]);
        let id_map = singleton_id_map(tokenizer.num_states(), 4);
        let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);

        build_transport_nwa_via_trie_walk(
            &tokenizer,
            None,
            &mut nwa,
            start,
            leaf,
            &id_map,
            &tree.root,
            &mut possible_matches,
            &visible_output_raw_labels(&partition, tokenizer.num_terminals() as usize),
            &modes,
        );

        assert_eq!(
            nwa.states()[start as usize].epsilons.len(),
            id_map.num_tsids() as usize,
            "transport modes must share each actual tokenizer-state root"
        );
    }

    #[test]
    fn compact_transport_output_filter_preserves_the_baseline_language() {
        // These duplicate literals are rooted-interchangeable. Raw
        // nonrepresentatives remain present in the lexer metadata, but their
        // output edges are redundant: the transport mode using its
        // representative edge supplies the corresponding member label.
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let partition = discover_one_round(&tokenizer, &[true, true], &[true; 256], None);
        let modes = binary_transport_modes(
            &tokenizer,
            &[true, true],
            &partition,
            &[true; 256],
            None,
        );
        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"aaa".to_vec()),
            (2, b"aaaaa".to_vec()),
            (3, b"aaaaaaa".to_vec()),
        ]);
        let id_map = singleton_id_map(tokenizer.num_states(), 4);
        let baseline = build_baseline_test_artifact(&tokenizer, &tree, &id_map);
        let full = build_transport_test_artifact(
            &tokenizer,
            &tree,
            &id_map,
            &[true, true],
            &modes,
        );
        super::super::terminal_dwa_equivalence::compare(&baseline, &full)
            .expect("full raw-output transport must reproduce the ordinary NWA");

        let compact = build_transport_test_artifact(
            &tokenizer,
            &tree,
            &id_map,
            &visible_output_raw_labels(&partition, tokenizer.num_terminals() as usize),
            &modes,
        );
        super::super::terminal_dwa_equivalence::compare(&baseline, &compact)
            .expect("raw-edge filtering must preserve the completed terminal language");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TransportScannerStateMap {
    Explicit(Arc<[TokenizerState]>),
    Quotient {
        state_count: usize,
        class_for_original: Arc<[u32]>,
        representative_for_class: Arc<[TokenizerState]>,
        source_class_for_target_deviations: Box<[(u32, u32)]>,
    },
    /// Temporary composition of round-local TI scanner coordinates. Keeping
    /// this lazy lets iterative TI replay share every accepted round map until
    /// the transport NWA actually queries a scanner state.
    Composed {
        outer: Arc<TransportScannerStateMap>,
        inner: Arc<TransportScannerStateMap>,
    },
}

impl TransportScannerStateMap {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Explicit(states) => states.len(),
            Self::Quotient { state_count, .. } => *state_count,
            Self::Composed { outer, inner } => {
                debug_assert_eq!(outer.len(), inner.len());
                outer.len()
            }
        }
    }

    #[inline]
    pub(crate) fn scanner_state(&self, original_state: TokenizerState) -> TokenizerState {
        match self {
            Self::Explicit(states) => states[original_state as usize],
            Self::Quotient {
                class_for_original,
                representative_for_class,
                source_class_for_target_deviations,
                ..
            } => {
                let target_class = class_for_original[original_state as usize];
                let source_class = source_class_for_target_deviations
                    .binary_search_by_key(&target_class, |&(target, _)| target)
                    .map(|index| source_class_for_target_deviations[index].1)
                    .unwrap_or(target_class);
                representative_for_class[source_class as usize]
            }
            Self::Composed { outer, inner } => {
                outer.scanner_state(inner.scanner_state(original_state))
            }
        }
    }

    /// Return the scanner coordinate for applying `inner` first and `outer`
    /// second. TI replay composes every accepted round-local witness this way.
    pub(crate) fn compose(
        outer: Arc<TransportScannerStateMap>,
        inner: Arc<TransportScannerStateMap>,
    ) -> Arc<TransportScannerStateMap> {
        assert_eq!(
            outer.len(),
            inner.len(),
            "TI scanner-coordinate composition requires equal state domains",
        );
        Arc::new(Self::Composed { outer, inner })
    }

    pub(crate) fn materialized(&self) -> Arc<[TokenizerState]> {
        match self {
            Self::Explicit(states) => Arc::clone(states),
            Self::Quotient { state_count, .. } => (0..*state_count)
                .map(|state| self.scanner_state(state as TokenizerState))
                .collect::<Vec<_>>()
                .into(),
            Self::Composed { .. } => (0..self.len())
                .map(|state| self.scanner_state(state as TokenizerState))
                .collect::<Vec<_>>()
                .into(),
        }
    }

    /// A composed TI transform is constant on the source classes of its
    /// innermost map: that map first selects one representative raw coordinate,
    /// and every outer transform is then deterministic.  Transport-coordinate
    /// refinement uses this fact to evaluate a whole mode group once per source
    /// quotient class instead of once per raw scanner state.
    #[inline]
    fn innermost_source_map(&self) -> &Self {
        match self {
            Self::Composed { inner, .. } => inner.innermost_source_map(),
            Self::Explicit(_) | Self::Quotient { .. } => self,
        }
    }

    /// A stable identity for the innermost source partition.  Equal keys mean
    /// the maps share the same raw-state-to-source-class function.  Distinct
    /// but extensionally equal partitions are merely kept in separate groups,
    /// which is conservative and remains exact.
    #[inline]
    pub(crate) fn innermost_source_domain_key(&self) -> usize {
        match self.innermost_source_map() {
            Self::Explicit(_) => 0,
            Self::Quotient {
                class_for_original,
                ..
            } => class_for_original.as_ptr() as usize,
            Self::Composed { .. } => unreachable!("innermost transport map cannot be composed"),
        }
    }

    #[inline]
    pub(crate) fn innermost_source_class_count(&self) -> usize {
        match self.innermost_source_map() {
            Self::Explicit(states) => states.len(),
            Self::Quotient {
                representative_for_class,
                ..
            } => representative_for_class.len(),
            Self::Composed { .. } => unreachable!("innermost transport map cannot be composed"),
        }
    }

    #[inline]
    pub(crate) fn innermost_source_class(&self, original_state: TokenizerState) -> usize {
        match self.innermost_source_map() {
            Self::Explicit(_) => original_state as usize,
            Self::Quotient {
                class_for_original,
                ..
            } => class_for_original[original_state as usize] as usize,
            Self::Composed { .. } => unreachable!("innermost transport map cannot be composed"),
        }
    }

    #[inline]
    pub(crate) fn innermost_source_representative(&self, class: usize) -> TokenizerState {
        match self.innermost_source_map() {
            Self::Explicit(_) => class as TokenizerState,
            Self::Quotient {
                representative_for_class,
                ..
            } => representative_for_class[class],
            Self::Composed { .. } => unreachable!("innermost transport map cannot be composed"),
        }
    }

    /// Append this temporary transport program in application order.  A
    /// composed transform applies its inner map first, then its outer map.
    pub(crate) fn append_atomic_transforms<'a>(
        &'a self,
        output: &mut Vec<&'a TransportScannerStateMap>,
    ) {
        match self {
            Self::Composed { outer, inner } => {
                inner.append_atomic_transforms(output);
                outer.append_atomic_transforms(output);
            }
            Self::Explicit(_) | Self::Quotient { .. } => output.push(self),
        }
    }

    /// Sparse inverse class permutation of a direct certified TI quotient.
    /// Each pair maps an input stable class to the representative class used
    /// by the transport; omitted classes map to themselves.
    #[inline]
    pub(crate) fn quotient_deviations(&self) -> Option<&[(u32, u32)]> {
        match self {
            Self::Quotient {
                source_class_for_target_deviations,
                ..
            } => Some(source_class_for_target_deviations),
            Self::Explicit(_) | Self::Composed { .. } => None,
        }
    }

    #[inline]
    pub(crate) fn direct_quotient_default_key(&self) -> Option<(usize, usize)> {
        match self {
            Self::Quotient {
                class_for_original,
                representative_for_class,
                ..
            } => Some((
                class_for_original.as_ptr() as usize,
                representative_for_class.as_ptr() as usize,
            )),
            Self::Explicit(_) | Self::Composed { .. } => None,
        }
    }

    pub(crate) fn make_explicit_mut(&mut self) -> &mut [TokenizerState] {
        if !matches!(self, Self::Explicit(_)) {
            *self = Self::Explicit(self.materialized());
        }
        match self {
            Self::Explicit(states) => Arc::make_mut(states),
            Self::Quotient { .. } | Self::Composed { .. } => {
                unreachable!("transport map was just materialized")
            }
        }
    }
}

/// One temporary binary-witness scanner mode for strict terminal
/// interchangeability. The ordinary mode emits only final representatives. A
/// member mode scans from one certified binary transport state map and emits
/// exactly one label: its member whenever the transported scan emits that
/// member's representative. It never relabels or emits unrelated terminals.
#[derive(Clone, Debug)]
pub(crate) struct TerminalNwaTransportMode {
    pub(crate) scanner_state_for_original: TransportScannerStateMap,
    member_reconstruction: Option<(TerminalID, TerminalID)>,
}

impl TerminalNwaTransportMode {
    pub(crate) fn ordinary(state_count: usize) -> Self {
        Self {
            scanner_state_for_original: TransportScannerStateMap::Explicit(
                (0..state_count as u32).collect::<Vec<_>>().into(),
            ),
            member_reconstruction: None,
        }
    }

    pub(crate) fn member(
        scanner_state_for_original: TransportScannerStateMap,
        representative: TerminalID,
        member: TerminalID,
    ) -> Self {
        Self {
            scanner_state_for_original,
            member_reconstruction: Some((representative, member)),
        }
    }

    #[inline]
    pub(crate) fn member_reconstruction(&self) -> Option<(TerminalID, TerminalID)> {
        self.member_reconstruction
    }

    #[inline]
    fn emitted_terminal(
        &self,
        raw_terminal: TerminalID,
        visible_representatives: &[bool],
    ) -> Option<TerminalID> {
        match self.member_reconstruction {
            None => visible_representatives
                .get(raw_terminal as usize)
                .copied()
                .unwrap_or(false)
                .then_some(raw_terminal),
            Some((representative, member)) => {
                (raw_terminal == representative).then_some(member)
            }
        }
    }
}

/// A scanner context is a raw lexer state together with the set of terminal
/// permutations that reached it. The parser/NWA state is deliberately *not*
/// duplicated for every permutation: all modes have the same parser-side
/// weight, so the exact union is represented by one NWA source with one context
/// per distinct raw scanner state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TransportContext {
    scanner_state: TokenizerState,
    mode_set: usize,
}

/// Exact mode subset representation. The common scanner context contains all
/// transport modes except a sparse deviation list, avoiding a full mode vector
/// at every root.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum TransportModeSet {
    Explicit(Arc<[usize]>),
    AllExcept(Arc<[usize]>),
}

#[derive(Clone, Default)]
struct NodesByTransportContext {
    entries: FxHashMap<TransportContext, Vec<NwaState>>,
}

impl NodesByTransportContext {
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn merge(&mut self, context: TransportContext, nodes: &[NwaState]) {
        self.entries.entry(context).or_default().extend_from_slice(nodes);
    }

    fn first_any(&self) -> Option<NwaState> {
        self.entries.values().find_map(|nodes| nodes.first().copied())
    }

    fn push_one(&mut self, context: TransportContext, node: NwaState) {
        self.entries.entry(context).or_default().push(node);
    }
}

#[derive(Default)]
struct TransportNwaTiming {
    enabled: bool,
    root_context_ms: f64,
    reset_context_ms: f64,
    mode_set_intern_ms: f64,
    mode_set_intern_hits: u64,
    mode_set_intern_new: u64,
    mapped_label_cache_hit_ms: f64,
    mapped_label_compute_ms: f64,
    mapped_label_cache_hits: u64,
    mapped_label_cache_misses: u64,
    trie_scan_ms: f64,
    trie_scan_calls: u64,
    trie_scan_bytes: u64,
    pending_context_ms: f64,
    pending_context_ops: u64,
    context_merge_ms: f64,
    future_edge_ms: f64,
    endpoint_leaf_edge_ms: f64,
    match_edge_ms: f64,
    edge_flush_ms: f64,
    edge_buffer_insertions: u64,
    total_ms: f64,
}

impl TransportNwaTiming {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some(),
            ..Self::default()
        }
    }

    #[inline]
    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    #[inline]
    fn elapsed_ms(&self, started: Option<Instant>) -> f64 {
        started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
    }

    fn emit(&self, modes: usize, roots: usize, mode_sets: usize) {
        if !self.enabled {
            return;
        }
        let edge_insertion_ms = self.future_edge_ms
            + self.endpoint_leaf_edge_ms
            + self.match_edge_ms
            + self.edge_flush_ms;
        eprintln!(
            "[glrmask/profile][transport_nwa] modes={} roots={} mode_sets={} root_context_ms={:.3} reset_context_ms={:.3} mode_set_intern_ms={:.3} mode_set_intern_hits={} mode_set_intern_new={} mapped_label_cache_hit_ms={:.3} mapped_label_compute_ms={:.3} mapped_label_cache_hits={} mapped_label_cache_misses={} trie_scan_ms={:.3} trie_scan_calls={} trie_scan_bytes={} pending_context_ms={:.3} pending_context_ops={} context_merge_ms={:.3} future_edge_ms={:.3} endpoint_leaf_edge_ms={:.3} match_edge_ms={:.3} edge_flush_ms={:.3} edge_insertion_ms={:.3} edge_buffer_insertions={} total_ms={:.3}",
            modes,
            roots,
            mode_sets,
            self.root_context_ms,
            self.reset_context_ms,
            self.mode_set_intern_ms,
            self.mode_set_intern_hits,
            self.mode_set_intern_new,
            self.mapped_label_cache_hit_ms,
            self.mapped_label_compute_ms,
            self.mapped_label_cache_hits,
            self.mapped_label_cache_misses,
            self.trie_scan_ms,
            self.trie_scan_calls,
            self.trie_scan_bytes,
            self.pending_context_ms,
            self.pending_context_ops,
            self.context_merge_ms,
            self.future_edge_ms,
            self.endpoint_leaf_edge_ms,
            self.match_edge_ms,
            self.edge_flush_ms,
            edge_insertion_ms,
            self.edge_buffer_insertions,
            self.total_ms,
        );
    }
}

impl IntoIterator for NodesByTransportContext {
    type Item = (TransportContext, Vec<NwaState>);
    type IntoIter = <FxHashMap<TransportContext, Vec<NwaState>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Interns subsets of transport modes and gives their exact raw-output label
/// expansion. Modes that enter the same raw scanner state are grouped because
/// a DFA continuation from that point is identical; only their output labels
/// differ.
struct TransportModePlanner<'m> {
    modes: &'m [TerminalNwaTransportMode],
    visible_representatives: &'m [bool],
    mode_set_ids: FxHashMap<TransportModeSet, usize>,
    mode_sets: Vec<TransportModeSet>,
    mapped_labels: Vec<FxHashMap<TerminalID, SmallVec<[TerminalID; 4]>>>,
    timing: TransportNwaTiming,
}

impl<'m> TransportModePlanner<'m> {
    fn new(
        modes: &'m [TerminalNwaTransportMode],
        visible_representatives: &'m [bool],
    ) -> Self {
        assert!(!modes.is_empty());
        Self {
            modes,
            visible_representatives,
            mode_set_ids: FxHashMap::default(),
            mode_sets: Vec::new(),
            mapped_labels: Vec::new(),
            timing: TransportNwaTiming::new(),
        }
    }

    fn intern_mode_set(&mut self, mode_set: TransportModeSet) -> usize {
        let timer = self.timing.start();
        debug_assert!(match &mode_set {
            TransportModeSet::Explicit(modes) | TransportModeSet::AllExcept(modes) => {
                modes.windows(2).all(|pair| pair[0] < pair[1])
            }
        });
        if let Some(&id) = self.mode_set_ids.get(&mode_set) {
            self.timing.mode_set_intern_hits += 1;
            self.timing.mode_set_intern_ms += self.timing.elapsed_ms(timer);
            return id;
        }
        let id = self.mode_sets.len();
        self.mode_set_ids.insert(mode_set.clone(), id);
        self.mode_sets.push(mode_set);
        self.mapped_labels.push(FxHashMap::default());
        self.timing.mode_set_intern_new += 1;
        self.timing.mode_set_intern_ms += self.timing.elapsed_ms(timer);
        id
    }

    /// The original direct grouping is cheaper when there are few transport
    /// modes, because a two-pass batch setup cannot amortize its own work.
    fn contexts_for_original_state_small(
        &mut self,
        original_state: TokenizerState,
    ) -> SmallVec<[TransportContext; 4]> {
        let mut modes_by_scanner_state = BTreeMap::<TokenizerState, Vec<usize>>::new();
        for (mode, transport) in self.modes.iter().enumerate() {
            let scanner_state = transport
                .scanner_state_for_original
                .scanner_state(original_state);
            modes_by_scanner_state
                .entry(scanner_state)
                .or_default()
                .push(mode);
        }
        modes_by_scanner_state
            .into_iter()
            .map(|(scanner_state, modes)| TransportContext {
                scanner_state,
                mode_set: self.intern_mode_set(TransportModeSet::Explicit(modes.into())),
            })
            .collect()
    }

    /// Build all root/reset contexts at once. The modal scanner state for a
    /// raw state is represented as all modes except sparse deviations, which
    /// avoids per-root ordered maps and full mode-vector interning.
    fn contexts_for_original_states(
        &mut self,
        original_states: &[TokenizerState],
    ) -> Vec<SmallVec<[TransportContext; 4]>> {
        const BATCH_MODE_THRESHOLD: usize = 128;
        if self.modes.len() < BATCH_MODE_THRESHOLD {
            return original_states
                .iter()
                .map(|&original_state| self.contexts_for_original_state_small(original_state))
                .collect();
        }

        type ScannerCounts = SmallVec<[(TokenizerState, usize); 4]>;
        type ExplicitGroups = SmallVec<[(TokenizerState, SmallVec<[usize; 4]>); 4]>;

        let state_count = original_states.len();
        if state_count == 0 {
            return Vec::new();
        }

        let mut counts = vec![ScannerCounts::new(); state_count];
        for transport in self.modes {
            for (state_index, &original_state) in original_states.iter().enumerate() {
                let scanner_state = transport
                    .scanner_state_for_original
                    .scanner_state(original_state);
                if let Some((_, count)) = counts[state_index]
                    .iter_mut()
                    .find(|(state, _)| *state == scanner_state)
                {
                    *count += 1;
                } else {
                    counts[state_index].push((scanner_state, 1));
                }
            }
        }

        let modal_scanner_states: Vec<_> = counts
            .iter()
            .map(|scanner_counts| {
                let mut modal = scanner_counts[0];
                for &(scanner_state, count) in scanner_counts.iter().skip(1) {
                    if count > modal.1 || (count == modal.1 && scanner_state < modal.0) {
                        modal = (scanner_state, count);
                    }
                }
                modal.0
            })
            .collect();

        let mut exclusions = vec![SmallVec::<[usize; 4]>::new(); state_count];
        let mut explicit_groups = vec![ExplicitGroups::new(); state_count];
        for (mode, transport) in self.modes.iter().enumerate() {
            for (state_index, &original_state) in original_states.iter().enumerate() {
                let scanner_state = transport
                    .scanner_state_for_original
                    .scanner_state(original_state);
                if scanner_state == modal_scanner_states[state_index] {
                    continue;
                }
                exclusions[state_index].push(mode);
                if let Some((_, mode_indices)) = explicit_groups[state_index]
                    .iter_mut()
                    .find(|(state, _)| *state == scanner_state)
                {
                    mode_indices.push(mode);
                } else {
                    explicit_groups[state_index]
                        .push((scanner_state, SmallVec::from_vec(vec![mode])));
                }
            }
        }

        let mut contexts_by_state = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            let excluded: Vec<_> = std::mem::take(&mut exclusions[state_index]).into_vec();
            let mut contexts = SmallVec::<[TransportContext; 4]>::new();
            contexts.push(TransportContext {
                scanner_state: modal_scanner_states[state_index],
                mode_set: self.intern_mode_set(TransportModeSet::AllExcept(excluded.into())),
            });
            for (scanner_state, mode_indices) in std::mem::take(&mut explicit_groups[state_index]) {
                let mode_indices: Vec<_> = mode_indices.into_vec();
                contexts.push(TransportContext {
                    scanner_state,
                    mode_set: self.intern_mode_set(TransportModeSet::Explicit(mode_indices.into())),
                });
            }
            contexts.sort_unstable_by_key(|context| context.scanner_state);
            contexts_by_state.push(contexts);
        }
        contexts_by_state
    }

    fn mapped_labels_for(
        &mut self,
        mode_set: usize,
        terminal: TerminalID,
    ) -> SmallVec<[TerminalID; 4]> {
        let timer = self.timing.start();
        if let Some(labels) = self.mapped_labels[mode_set].get(&terminal) {
            self.timing.mapped_label_cache_hits += 1;
            self.timing.mapped_label_cache_hit_ms += self.timing.elapsed_ms(timer);
            return labels.clone();
        }
        self.timing.mapped_label_cache_misses += 1;
        let mut labels = SmallVec::<[TerminalID; 4]>::new();
        match &self.mode_sets[mode_set] {
            TransportModeSet::Explicit(modes) => {
                for &mode in modes.iter() {
                    if let Some(label) = self.modes[mode]
                        .emitted_terminal(terminal, self.visible_representatives)
                    {
                        labels.push(label);
                    }
                }
            }
            TransportModeSet::AllExcept(excluded) => {
                let mut excluded_index = 0;
                for mode in 0..self.modes.len() {
                    if excluded.get(excluded_index).copied() == Some(mode) {
                        excluded_index += 1;
                    } else if let Some(label) = self.modes[mode]
                        .emitted_terminal(terminal, self.visible_representatives)
                    {
                        labels.push(label);
                    }
                }
            }
        }
        labels.sort_unstable();
        labels.dedup();
        self.mapped_labels[mode_set].insert(terminal, labels.clone());
        self.timing.mapped_label_compute_ms += self.timing.elapsed_ms(timer);
        labels
    }
}

struct TransportNwaBuilder<'tok, 'pm, 'nwa, 'm> {
    base: TerminalNwaBuilder<'tok, 'pm, 'nwa>,
    transport: TransportModePlanner<'m>,
    /// After every terminal boundary every temporary witness is available again.
    /// This vector partitions those modes by their reset scanner state.
    reset_contexts: SmallVec<[TransportContext; 4]>,
}

impl<'tok, 'pm, 'nwa, 'm> TransportNwaBuilder<'tok, 'pm, 'nwa, 'm> {
    fn mapped_labels(&mut self, context: TransportContext, terminal: TerminalID) -> SmallVec<[TerminalID; 4]> {
        self.transport.mapped_labels_for(context.mode_set, terminal)
    }

    fn add_future_leaf_token_from_sources(
        &mut self,
        sources: &[NwaState],
        context: TransportContext,
        internal_token_id: u32,
    ) {
        let future_edge_timer = self.transport.timing.start();
        let future = self
            .base
            .possible_future_terminals_for_state(context.scanner_state);
        for terminal in future {
            let mapped_labels = self.mapped_labels(context, terminal);
            if mapped_labels.is_empty() {
                continue;
            }
            self.base.profile.future_terminal_additions +=
                (sources.len() * mapped_labels.len()) as u64;
            self.transport.timing.edge_buffer_insertions +=
                (sources.len() * mapped_labels.len()) as u64;
            for mapped_label in mapped_labels {
                self.base
                    .add_leaf_token_from_sources(sources, mapped_label, internal_token_id);
            }
        }
        self.transport.timing.future_edge_ms +=
            self.transport.timing.elapsed_ms(future_edge_timer);
    }

    fn build_from_trie(
        &mut self,
        node: &VocabPrefixTreeNode,
        contexts: &NodesByTransportContext,
    ) {
        if contexts.is_empty() {
            return;
        }
        for (segment_bytes, child) in node.iter_children() {
            let next = self.process_child_segment(segment_bytes, child, contexts);
            if !next.is_empty() {
                self.build_from_trie(child, &next);
            }
        }
    }

    fn process_child_segment(
        &mut self,
        segment_bytes: &[u8],
        child_node: &VocabPrefixTreeNode,
        initial_contexts: &NodesByTransportContext,
    ) -> NodesByTransportContext {
        let leaf_token_id = child_node.token_id() as u32;
        let mut next_level = NodesByTransportContext::default();
        let mut pending = BTreeMap::<usize, NodesByTransportContext>::new();
        let pending_timer = self.transport.timing.start();
        pending.insert(0, initial_contexts.clone());
        self.transport.timing.pending_context_ms +=
            self.transport.timing.elapsed_ms(pending_timer);
        self.transport.timing.pending_context_ops += 1;
        let mut match_map = FxHashMap::<TerminalID, (usize, TokenizerState)>::default();
        let mut matches = Vec::<TokenizerMatch>::new();

        while let Some((offset, contexts_at_offset)) = pending.pop_first() {
            if offset == segment_bytes.len() {
                let merge_timer = self.transport.timing.start();
                for (context, nodes) in contexts_at_offset {
                    next_level.merge(context, &nodes);
                }
                self.transport.timing.context_merge_ms +=
                    self.transport.timing.elapsed_ms(merge_timer);
                continue;
            }

            for (context, source_nodes) in contexts_at_offset {
                match_map.clear();
                let mut scan_state = context.scanner_state;
                let mut alive = true;
                let scan_timer = self.transport.timing.start();
                let mut scanned_bytes = 0_u64;
                for (index, &byte) in segment_bytes[offset..].iter().enumerate() {
                    scanned_bytes += 1;
                    let Some(next) = self.base.fast_step(scan_state, byte) else {
                        alive = false;
                        break;
                    };
                    scan_state = next;
                    for terminal in self.base.tokenizer.matched_terminals_iter(scan_state) {
                        if !self.mapped_labels(context, terminal).is_empty() {
                            match_map.insert(terminal, (index + 1, scan_state));
                        }
                    }
                }
                self.transport.timing.trie_scan_ms +=
                    self.transport.timing.elapsed_ms(scan_timer);
                self.transport.timing.trie_scan_calls += 1;
                self.transport.timing.trie_scan_bytes += scanned_bytes;
                let end_state = alive.then_some(scan_state);

                matches.clear();
                matches.extend(match_map.iter().map(|(&id, &(width, end_state))| TokenizerMatch {
                    id,
                    width,
                    end_state,
                }));

                if let Some(end_state) = end_state {
                    if child_node.has_token() {
                        self.add_future_leaf_token_from_sources(
                            &source_nodes,
                            TransportContext { scanner_state: end_state, ..context },
                            leaf_token_id,
                        );
                    }
                    let merge_timer = self.transport.timing.start();
                    next_level.merge(
                        TransportContext { scanner_state: end_state, ..context },
                        &source_nodes,
                    );
                    self.transport.timing.context_merge_ms +=
                        self.transport.timing.elapsed_ms(merge_timer);
                }

                for matched in &matches {
                    let next_offset = offset + matched.width;
                    let mapped_labels = self.mapped_labels(context, matched.id);
                    if next_offset == segment_bytes.len()
                        && child_node.has_token()
                        && !end_state.is_some_and(|state| {
                            self.base
                                .possible_future_terminals_for_state(state)
                                .contains(&matched.id)
                        })
                    {
                        self.base.profile.match_transition_additions +=
                            (source_nodes.len() * mapped_labels.len()) as u64;
                        let endpoint_edge_timer = self.transport.timing.start();
                        for mapped_label in &mapped_labels {
                            self.base.add_leaf_token_from_sources(
                                &source_nodes,
                                *mapped_label,
                                leaf_token_id,
                            );
                        }
                        self.transport.timing.endpoint_leaf_edge_ms +=
                            self.transport.timing.elapsed_ms(endpoint_edge_timer);
                        self.transport.timing.edge_buffer_insertions +=
                            (source_nodes.len() * mapped_labels.len()) as u64;
                    }

                    let Some(weight) = self.base.continuation_weight_for_match(
                        child_node,
                        leaf_token_id,
                        matched.id,
                        end_state,
                        next_offset == segment_bytes.len(),
                    ) else {
                        continue;
                    };
                    if weight.is_empty() {
                        continue;
                    }

                    // A terminal boundary resets the actual lexer. Every
                    // transport mode is available for the next terminal, so one
                    // NWA continuation state carries their exact union; it is
                    // merely attached to each distinct reset scanner context.
                    let pending_timer = self.transport.timing.start();
                    let continuation_contexts = pending.entry(next_offset).or_default();
                    let destination = ensure_transport_continuation_state(
                        continuation_contexts,
                        self.base.nwa,
                    );
                    for &next_context in &self.reset_contexts {
                        continuation_contexts.push_one(next_context, destination);
                    }
                    self.transport.timing.pending_context_ms +=
                        self.transport.timing.elapsed_ms(pending_timer);
                    self.transport.timing.pending_context_ops += 1;
                    self.base.profile.match_transition_additions +=
                        (source_nodes.len() * mapped_labels.len()) as u64;
                    let match_edge_timer = self.transport.timing.start();
                    let mapped_label_count = mapped_labels.len();
                    for mapped_label in mapped_labels {
                        self.base.add_match_from_sources(
                            &source_nodes,
                            mapped_label,
                            destination,
                            &weight,
                        );
                    }
                    self.transport.timing.match_edge_ms +=
                        self.transport.timing.elapsed_ms(match_edge_timer);
                    self.transport.timing.edge_buffer_insertions +=
                        (source_nodes.len() * mapped_label_count) as u64;
                }
            }
        }

        next_level
    }
}

fn ensure_transport_continuation_state(
    pending: &mut NodesByTransportContext,
    nwa: &mut NWA,
) -> NwaState {
    if let Some(existing) = pending.first_any() {
        return existing;
    }
    let state = nwa.add_state();
    // The caller immediately attaches this state to the complete reset-mode
    // partition. Keeping the helper context-free prevents one identical NFA
    // union state from being cloned per terminal interchangeability mode.
    state
}

/// Slow reference trie walk for strict terminal interchangeability. It is used
/// only while the feature is validation-gated. Normal L2P construction remains
/// on `build_nwa_via_trie_walk` above.
pub(crate) fn build_transport_nwa_via_trie_walk<'a>(
    tokenizer: &'a Tokenizer,
    ignore_terminal: Option<TerminalID>,
    nwa: &mut NWA,
    start_state: u32,
    leaf_state: u32,
    id_map: &InternalIdMap,
    vocab_tree_root: &VocabPrefixTreeNode,
    possible_matches: &mut PossibleMatchesComputer<'a>,
    visible_output_raw_labels: &[bool],
    modes: &[TerminalNwaTransportMode],
) -> TerminalDwaBuildProfile {
    assert!(!modes.is_empty());
    let mut roots = NodesByTransportContext::default();
    let mut initial_source_states = Vec::<bool>::new();
    let mut transport = TransportModePlanner::new(modes, visible_output_raw_labels);
    let build_timer = transport.timing.start();
    let representative_states: Vec<_> = id_map
        .tokenizer_states
        .iter_representative_ids()
        .collect();
    let root_context_timer = transport.timing.start();
    let root_contexts = transport.contexts_for_original_states(&representative_states);

    for (internal_tsid, contexts) in root_contexts.into_iter().enumerate() {
        // All transport modes start from the same parser-side condition for
        // this actual tokenizer state. A single NWA root therefore represents
        // their exact union; contexts retain only the distinct raw scanner
        // continuations needed to discover the next terminal boundary.
        let root = nwa.add_state();
        let weight = all_token_weight(internal_tsid as u32, id_map.max_internal_token_id());
        nwa.add_epsilon(start_state, root, weight);
        if root as usize >= initial_source_states.len() {
            initial_source_states.resize(root as usize + 1, false);
        }
        initial_source_states[root as usize] = true;
        for context in contexts {
            roots.merge(context, &[root]);
        }
    }
    transport.timing.root_context_ms += transport.timing.elapsed_ms(root_context_timer);
    let reset_context_timer = transport.timing.start();
    let reset_contexts = transport
        .contexts_for_original_states(&[tokenizer.initial_state_id()])
        .into_iter()
        .next()
        .expect("one reset state must produce one context set");
    transport.timing.reset_context_ms += transport.timing.elapsed_ms(reset_context_timer);

    let base = TerminalNwaBuilder::new(
        tokenizer,
        TerminalColoring::identity(tokenizer.num_terminals() as usize),
        possible_matches,
        nwa,
        id_map.num_tsids(),
        leaf_state,
        ignore_terminal,
        initial_source_states,
        false,
        None,
        None,
        tokenizer.num_states() as usize,
    );
    let mut builder = TransportNwaBuilder {
        base,
        transport,
        reset_contexts,
    };
    builder.build_from_trie(vocab_tree_root, &roots);
    let flush_timer = builder.transport.timing.start();
    builder.base.flush_transition_buffer();
    builder.transport.timing.edge_flush_ms +=
        builder.transport.timing.elapsed_ms(flush_timer);
    builder.transport.timing.total_ms = builder.transport.timing.elapsed_ms(build_timer);
    builder.transport.timing.emit(
        modes.len(),
        id_map.num_tsids() as usize,
        builder.transport.mode_sets.len(),
    );
    builder.base.profile
}
