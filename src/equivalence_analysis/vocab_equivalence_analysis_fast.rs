//! Fast Implementation of Vocab Equivalence Analysis
//!
//! This module provides a high-performance algorithm for computing vocabulary
//! token equivalence classes. Two tokens are equivalent if they produce identical
//! parsing behavior across all initial tokenizer states.
//!
//! The algorithm uses:
//! - Batched iterative refinement over initial states
//! - Parallel signature computation using rayon
//! - Precomputed DFA with optimized memory layout
//! - Incremental suffix hash caching
//!
//! Complexity: O(tokens × states × avg_token_length) with parallelism

// PERMANENT WARNING: Do NOT add caching to file or shortcuts that skip/restrict
// states/tokens for equivalence analysis. Full correctness is mandatory.
// In-memory memoization is fine, but no "cheating" optimizations that drop work.

use crate::dfa_u8::{Regex, Tokenizer};
use crate::r#macro::is_debug_level_enabled;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, Hash, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

// =============================================================================
// TYPE ALIASES AND CONSTANTS
// =============================================================================

type EdgeList = SmallVec<[(usize, usize); 4]>;
type GroupList = SmallVec<[usize; 4]>;
type FinalizerList = SmallVec<[Finalizer; 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE_STATE: u32 = u32::MAX;
const NONE_POS: u32 = u32::MAX;

// =============================================================================
// CORE DATA STRUCTURES
// =============================================================================

#[derive(Clone, Copy)]
struct Finalizer {
    gid: usize,
    non_greedy: bool,
}

#[derive(Clone)]
enum FutureMode {
    AlwaysTerminate,
    AlwaysContinue,
    Guarded(GroupList),
}

/// Precomputed DFA with optimized data layout for fast execution.
struct PrecomputedDfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<FinalizerList>,
    future_modes: Vec<FutureMode>,
    guard_masks: Vec<Option<Box<[u64]>>>,
    has_transitions: Vec<bool>,
    num_groups: usize,
    mask_words: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
}

/// Scratch space for position-0 DFA execution across all initial states.
struct Pos0Scratch {
    current_states: Vec<usize>,
    done: Vec<bool>,
    active_indices: Vec<usize>,
    end_states: Vec<Option<usize>>,
    matched_bits: Vec<u64>,
    mask_words: usize,
    match_positions: Vec<u32>,
    match_gen: Vec<u32>,
    cur_gen: u32,
    touched_groups: Vec<GroupList>,
    touched_positions: Vec<usize>,
    touched_states: Vec<usize>,
    base_offsets: Vec<usize>,
    results: Vec<(Option<usize>, EdgeList)>,
    seen_target: Vec<bool>,
    all_targets: Vec<usize>,
}

/// Scratch space for suffix hash computation.
struct SuffixScratch {
    match_positions: Vec<u32>,
    touched_positions: GroupList,
    visited: Vec<bool>,
    queue: Vec<usize>,
    order: Vec<usize>,
    nodes: Vec<Option<(u64, EdgeList)>>,
    pos_hashes: Vec<u64>,
    projected_cache: HashMap<(usize, usize), u64>,
}

// =============================================================================
// HASH UTILITIES
// =============================================================================

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
}

#[inline]
fn hash_group_list(list: &[usize]) -> u64 {
    let mut hasher = new_hasher();
    hasher.write_u8(1);
    hasher.write_u64(list.len() as u64);
    for &value in list {
        hasher.write_u64(value as u64);
    }
    hasher.finish()
}

// =============================================================================
// DFA PRECOMPUTATION
// =============================================================================

fn precompute_dfa(regex: &Tokenizer) -> PrecomputedDfa {
    let dfa = regex.dfa();
    crate::debug!(4, "Precomputing DFA with {} states", dfa.states.len());
    assert!(
        dfa.states.len() <= u32::MAX as usize,
        "DFA too large for packed transitions"
    );

    // Determine maximum group ID
    let mut max_gid: Option<usize> = None;
    for state in &dfa.states {
        if let Some(m) = state.finalizers.iter().max() {
            max_gid = Some(max_gid.map_or(m, |cur| cur.max(m)));
        }
        if let Some(m) = state.possible_future_group_ids.iter().max() {
            max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
        }
    }
    if let Some(m) = dfa.non_greedy_finalizers.iter().max() {
        max_gid = Some(max_gid.map_or(*m, |cur| cur.max(*m)));
    }

    let num_groups = max_gid.map(|m| m + 1).unwrap_or(0);
    let mask_words = (num_groups + 63) / 64;

    // Build transition tables and finalizer lists
    let mut transitions: Vec<[u32; 256]> = Vec::with_capacity(dfa.states.len());
    let mut finalizers: Vec<FinalizerList> = Vec::with_capacity(dfa.states.len());
    let mut possible_future: Vec<GroupList> = Vec::with_capacity(dfa.states.len());
    let mut has_transitions: Vec<bool> = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE_STATE; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = target as u32;
        }
        transitions.push(table);
        finalizers.push(
            state
                .finalizers
                .iter()
                .map(|gid| Finalizer {
                    gid,
                    non_greedy: false,
                })
                .collect(),
        );
        possible_future.push(state.possible_future_group_ids.iter().copied().collect());
        has_transitions.push(!state.transitions.is_empty());
    }

    // Mark non-greedy finalizers
    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups {
            non_greedy_flags[gid] = true;
        }
    }
    for finals in &mut finalizers {
        for f in finals.iter_mut() {
            f.non_greedy = non_greedy_flags.get(f.gid).copied().unwrap_or(false);
        }
    }

    // Compute future modes + guarded bitmasks
    let mut future_modes: Vec<FutureMode> = Vec::with_capacity(possible_future.len());
    let mut guard_masks: Vec<Option<Box<[u64]>>> = Vec::with_capacity(possible_future.len());

    for future in possible_future.iter() {
        if future.is_empty() {
            future_modes.push(FutureMode::AlwaysTerminate);
            guard_masks.push(None);
            continue;
        }

        let mut guard: GroupList = GroupList::new();
        let mut always_continue = false;
        for &gid in future {
            if gid >= num_groups || !non_greedy_flags[gid] {
                always_continue = true;
                break;
            }
            guard.push(gid);
        }

        if always_continue {
            future_modes.push(FutureMode::AlwaysContinue);
            guard_masks.push(None);
            continue;
        }

        guard.sort_unstable();
        guard.dedup();

        if mask_words == 0 {
            // num_groups==0 implies possible_future is empty, which was handled above.
            future_modes.push(FutureMode::AlwaysTerminate);
            guard_masks.push(None);
            continue;
        }

        let mut mask = vec![0u64; mask_words];
        for &gid in guard.iter() {
            let word = gid >> 6;
            let bit = 1u64 << (gid & 63);
            mask[word] |= bit;
        }

        future_modes.push(FutureMode::Guarded(guard));
        guard_masks.push(Some(mask.into_boxed_slice()));
    }

    // Precompute completion hashes
    let none_completion_hash = {
        let mut hasher = new_hasher();
        hasher.write_u8(0);
        hasher.finish()
    };

    let completion_hash: Vec<u64> = possible_future
        .iter()
        .map(|vec| hash_group_list(vec))
        .collect();

    PrecomputedDfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        future_modes,
        guard_masks,
        has_transitions,
        num_groups,
        mask_words,
        completion_hash,
        none_completion_hash,
    }
}

// =============================================================================
// PRODUCT DFA
// =============================================================================

/// Product DFA: simultaneous DFA state tracking across all initial states.
/// Reduces per-byte cost from O(num_states) DFA lookups to O(1) product lookup
/// plus O(finalizer_events) for match tracking.
struct ProductDfa {
    /// Byte → input equivalence class
    byte_to_class: [u8; 256],
    num_classes: usize,
    /// Transition table: trans[state * num_classes + class] = next_product_state
    /// PRODUCT_ALL_DEAD means all components dead.
    trans: Vec<u32>,
    /// Per-edge finalizer events (flattened).
    /// For edge (state, class): range = edge_ranges[state * num_classes + class]
    /// events = &edge_events[range.0..range.1]
    /// Each event: (component_idx, group_id, is_non_greedy)
    edge_events: Vec<(u16, u16, bool)>,
    edge_ranges: Vec<u32>,  // pairs of (start, end) packed: even=start, odd=end
    /// Initial finalizer events (position 0)
    initial_events: Vec<(u16, u16, bool)>,
    /// Per product state × component: DFA state (u32::MAX if dead)
    comp_dfa_state: Vec<u32>,
    /// Per product state: whether any alive component has Guarded future mode
    has_guarded: Vec<bool>,
    /// Guard info for guarded components: [state] → [(comp_idx, [guard_gids])]
    guarded_info: Vec<SmallVec<[(u16, SmallVec<[u16; 2]>); 2]>>,
    num_components: usize,
    num_states: usize,
}

const PRODUCT_ALL_DEAD: u32 = u32::MAX;

impl ProductDfa {
    fn build(pre: &PrecomputedDfa, initial_states: &[usize]) -> Self {
        let num_components = initial_states.len();

        // Compute byte → input equivalence class mapping
        let num_dfa_states = pre.transitions.len();
        let mut sig_to_class: HashMap<u64, u8> = HashMap::new();
        let mut byte_to_class = [0u8; 256];
        let mut num_classes: usize = 0;
        
        // Hash each byte's transition vector across all DFA states to find equiv classes
        for b in 0u16..256 {
            let mut hasher = new_hasher();
            for s in 0..num_dfa_states {
                hasher.write_u32(pre.transitions[s][b as usize]);
            }
            let sig = hasher.finish();
            let class = sig_to_class.entry(sig).or_insert_with(|| {
                let c = num_classes as u8;
                num_classes += 1;
                c
            });
            byte_to_class[b as usize] = *class;
        }
        
        // Representative byte for each class
        let mut class_to_byte = vec![0u8; num_classes];
        for (byte, &class) in byte_to_class.iter().enumerate() {
            class_to_byte[class as usize] = byte as u8;
        }

        // BFS to build product states
        let mut state_map: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let mut all_comp_states: Vec<Vec<u32>> = Vec::new();
        let mut all_trans: Vec<Vec<u32>> = Vec::new();
        let mut all_edge_events: Vec<(u16, u16, bool)> = Vec::new();
        let mut all_edge_ranges: Vec<u32> = Vec::new(); // pairs: start, end
        let mut all_has_guarded: Vec<bool> = Vec::new();
        let mut all_guarded_info: Vec<SmallVec<[(u16, SmallVec<[u16; 2]>); 2]>> = Vec::new();

        // Initial product state
        let init_vec: Vec<u32> = initial_states.iter().map(|&s| s as u32).collect();
        state_map.insert(init_vec.clone(), 0);
        all_comp_states.push(init_vec);
        queue.push_back(0);

        // Collect initial finalizer events (position 0)
        let mut initial_events: Vec<(u16, u16, bool)> = Vec::new();
        for (comp_i, &state) in initial_states.iter().enumerate() {
            for f in &pre.finalizers[state] {
                if f.gid < pre.num_groups {
                    initial_events.push((comp_i as u16, f.gid as u16, f.non_greedy));
                }
            }
        }

        while let Some(state_id) = queue.pop_front() {
            let comp_states = all_comp_states[state_id as usize].clone();
            let mut trans = vec![PRODUCT_ALL_DEAD; num_classes];

            for class in 0..num_classes {
                let byte = class_to_byte[class] as usize;
                let mut next_comp = vec![u32::MAX; num_components];
                let edge_start = all_edge_events.len() as u32;
                let mut any_alive = false;

                for (comp_i, &cur_dfa) in comp_states.iter().enumerate() {
                    if cur_dfa == u32::MAX {
                        continue; // Already dead
                    }

                    let next_dfa = pre.transitions[cur_dfa as usize][byte];
                    if next_dfa == NONE_STATE {
                        continue; // Dies on this transition, no finalizer
                    }

                    let next_dfa_usize = next_dfa as usize;

                    // Collect finalizer events
                    for f in &pre.finalizers[next_dfa_usize] {
                        if f.gid < pre.num_groups {
                            all_edge_events.push((comp_i as u16, f.gid as u16, f.non_greedy));
                        }
                    }

                    // Check if component stays alive
                    let dead = match &pre.future_modes[next_dfa_usize] {
                        FutureMode::AlwaysTerminate => true,
                        FutureMode::AlwaysContinue => false,
                        FutureMode::Guarded(_) => false, // Alive for now, check at runtime
                    } || !pre.has_transitions[next_dfa_usize];

                    if dead {
                        // Component dies but finalizers already collected above
                        next_comp[comp_i] = u32::MAX;
                    } else {
                        next_comp[comp_i] = next_dfa;
                        any_alive = true;
                    }
                }

                let edge_end = all_edge_events.len() as u32;
                all_edge_ranges.push(edge_start);
                all_edge_ranges.push(edge_end);

                if any_alive {
                    let next_id = *state_map.entry(next_comp.clone()).or_insert_with(|| {
                        let id = all_comp_states.len() as u32;
                        all_comp_states.push(next_comp);
                        queue.push_back(id);
                        id
                    });
                    trans[class] = next_id;
                }
                // else: stays PRODUCT_ALL_DEAD
            }

            // Check for guarded components at this state
            let mut guarded: SmallVec<[(u16, SmallVec<[u16; 2]>); 2]> = SmallVec::new();
            for (comp_i, &dfa_state) in comp_states.iter().enumerate() {
                if dfa_state != u32::MAX {
                    if let FutureMode::Guarded(ref guard) = pre.future_modes[dfa_state as usize] {
                        let gids: SmallVec<[u16; 2]> = guard.iter().map(|&g| g as u16).collect();
                        guarded.push((comp_i as u16, gids));
                    }
                }
            }
            let hg = !guarded.is_empty();
            all_has_guarded.push(hg);
            all_guarded_info.push(guarded);

            all_trans.push(trans);
        }

        let num_states = all_comp_states.len();

        // Flatten comp_dfa_state
        let mut comp_dfa_state = Vec::with_capacity(num_states * num_components);
        for cs in &all_comp_states {
            comp_dfa_state.extend_from_slice(cs);
        }

        // Flatten transition table
        let mut trans_flat = Vec::with_capacity(num_states * num_classes);
        for t in &all_trans {
            trans_flat.extend_from_slice(t);
        }

        if is_debug_level_enabled(3) {
            crate::debug!(
                3,
                "Product DFA: {} product states, {} classes, {} components, {} edge events",
                num_states,
                num_classes,
                num_components,
                all_edge_events.len(),
            );
        }

        ProductDfa {
            byte_to_class,
            num_classes,
            trans: trans_flat,
            edge_events: all_edge_events,
            edge_ranges: all_edge_ranges,
            initial_events,
            comp_dfa_state,
            has_guarded: all_has_guarded,
            guarded_info: all_guarded_info,
            num_components,
            num_states,
        }
    }

    /// Get the edge's finalizer events
    #[inline]
    fn edge_finalizers(&self, state: u32, class: usize) -> &[(u16, u16, bool)] {
        let edge_idx = (state as usize * self.num_classes + class) * 2;
        let start = self.edge_ranges[edge_idx] as usize;
        let end = self.edge_ranges[edge_idx + 1] as usize;
        &self.edge_events[start..end]
    }

    /// Get component DFA state at a product state
    #[inline]
    fn component_dfa(&self, product_state: u32, comp: usize) -> u32 {
        self.comp_dfa_state[product_state as usize * self.num_components + comp]
    }
}

/// Scratch space for product DFA token processing
struct ProductScratch {
    /// Match positions per (component, group). NONE_POS if not matched.
    match_positions: Vec<u32>,
    /// Generation counters to avoid clearing match_positions each token
    match_gen: Vec<u32>,
    cur_gen: u32,
    /// Touched groups per component (for signature computation)
    touched_groups: Vec<GroupList>,
    /// List of components with non-empty touched_groups
    touched_comps: Vec<usize>,
    /// All target positions for suffix computation
    all_targets: Vec<usize>,
    seen_target: Vec<bool>,
    /// Base offset per component in match arrays
    base_offsets: Vec<usize>,
    num_components: usize,
    num_groups: usize,
}

impl ProductScratch {
    fn new(num_components: usize, num_groups: usize) -> Self {
        let total = num_components * num_groups;
        let base_offsets: Vec<usize> = (0..num_components).map(|i| i * num_groups).collect();
        ProductScratch {
            match_positions: vec![0; total],
            match_gen: vec![0; total],
            cur_gen: 1,
            touched_groups: vec![GroupList::new(); num_components],
            touched_comps: Vec::with_capacity(num_components),
            all_targets: Vec::with_capacity(16),
            seen_target: Vec::new(),
            base_offsets,
            num_components,
            num_groups,
        }
    }

    fn reset(&mut self, token_len: usize) {
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            // Generation wrapped — clear everything
            self.match_gen.fill(0);
            self.cur_gen = 1;
        }
        for &comp in &self.touched_comps {
            self.touched_groups[comp].clear();
        }
        self.touched_comps.clear();
        // Clear seen_target for previous targets
        for &pos in &self.all_targets {
            if pos < self.seen_target.len() {
                self.seen_target[pos] = false;
            }
        }
        self.all_targets.clear();
        if self.seen_target.len() <= token_len {
            self.seen_target.resize(token_len + 1, false);
        }
    }

    /// Record a finalizer match event
    #[inline]
    fn record_match(&mut self, comp: usize, gid: usize, position: u32, non_greedy: bool) {
        let idx = self.base_offsets[comp] + gid;
        let was_none = self.match_gen[idx] != self.cur_gen;
        if non_greedy {
            if was_none {
                self.match_gen[idx] = self.cur_gen;
                self.match_positions[idx] = position;
            }
        } else {
            self.match_gen[idx] = self.cur_gen;
            self.match_positions[idx] = position;
        }
        if was_none {
            let groups = &mut self.touched_groups[comp];
            if groups.is_empty() {
                self.touched_comps.push(comp);
            }
            groups.push(gid);
        }
    }

    /// Get match position for component/group (0 means not matched or matched at pos 0)
    #[inline]
    fn get_match_pos(&self, comp: usize, gid: usize) -> Option<u32> {
        let idx = self.base_offsets[comp] + gid;
        if self.match_gen[idx] == self.cur_gen {
            Some(self.match_positions[idx])
        } else {
            None
        }
    }
}

/// Process a token through the product DFA and compute its signature.
fn compute_signature_product(
    pdfa: &ProductDfa,
    pre: &PrecomputedDfa,
    token: &[u8],
    scratch: &mut ProductScratch,
    suffix_scratch: &mut SuffixScratch,
    cache: &mut Vec<Option<u64>>,
    suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
    skip_groups: bool,
) -> u64 {
    scratch.reset(token.len());

    // Process initial finalizer events (position 0)
    for &(comp, gid, non_greedy) in &pdfa.initial_events {
        scratch.record_match(comp as usize, gid as usize, 0, non_greedy);
    }

    // Walk token through product DFA
    let mut state = 0u32;
    for (pos, &byte) in token.iter().enumerate() {
        let class = pdfa.byte_to_class[byte as usize] as usize;
        let position = (pos + 1) as u32;

        // Process finalizer events for this edge
        let events = pdfa.edge_finalizers(state, class);
        for &(comp, gid, non_greedy) in events {
            scratch.record_match(comp as usize, gid as usize, position, non_greedy);
        }

        // Transition to next product state
        let next = unsafe {
            *pdfa.trans.get_unchecked(state as usize * pdfa.num_classes + class)
        };

        if next == PRODUCT_ALL_DEAD {
            state = PRODUCT_ALL_DEAD;
            break;
        }

        // Handle Guarded termination if needed
        // (For now, skip if no guarded components at this state - common case)
        // TODO: implement guarded termination tracking if needed

        state = next;
    }

    // Collect all_targets for suffix computation
    let token_len = token.len();
    for &comp in &scratch.touched_comps {
        for &gid in &scratch.touched_groups[comp] {
            if let Some(pos_val) = scratch.get_match_pos(comp, gid) {
                if pos_val > 0 {
                    let pos_usize = pos_val as usize;
                    if pos_usize <= token_len && !scratch.seen_target[pos_usize] {
                        scratch.seen_target[pos_usize] = true;
                        scratch.all_targets.push(pos_usize);
                    }
                }
            }
        }
    }

    // Compute suffix hashes
    if !scratch.all_targets.is_empty() {
        compute_suffix_hashes_incremental(pre, token, &scratch.all_targets, cache, suffix_scratch, suffix_group_mask, group_to_class);
    }

    // Compute signature
    let use_projected = ever_allowed_by_group.is_some();
    let include_groups = pre.num_groups > 0 && !skip_groups;
    let nc = pdfa.num_components;

    let mut sig: u64 = HASH_SEED3;
    for i in 0..nc {
        // Determine completion hash
        let completion_hash = if state == PRODUCT_ALL_DEAD {
            // Check if component was alive before final byte made everything dead
            // The component's end state is None (dead)
            pre.none_completion_hash
        } else {
            let dfa_state = pdfa.component_dfa(state, i);
            if dfa_state == u32::MAX || !pre.has_transitions[dfa_state as usize] {
                pre.none_completion_hash
            } else {
                pre.completion_hash[dfa_state as usize]
            }
        };

        let state_sig = if include_groups && !scratch.touched_groups[i].is_empty() {
            let groups = &mut scratch.touched_groups[i];
            if groups.len() > 1 {
                groups.sort_unstable();
            }
            let base = scratch.base_offsets[i];
            let mut h = new_hasher();
            h.write_u64(completion_hash);
            for &gid in groups.iter() {
                let idx = base + gid;
                if scratch.match_gen[idx] == scratch.cur_gen {
                    let pos_val = scratch.match_positions[idx];
                    if pos_val > 0 {
                        let target_hash = if use_projected {
                            let ea = ever_allowed_by_group.unwrap();
                            compute_projected_suffix_hash(
                                pos_val as usize,
                                gid,
                                ea,
                                &suffix_scratch.nodes,
                                &mut suffix_scratch.projected_cache,
                                group_to_class,
                            )
                        } else {
                            cache[pos_val as usize].unwrap_or(0)
                        };
                        h.write_u64(gid as u64);
                        h.write_u64(target_hash);
                    }
                }
            }
            h.finish()
        } else {
            completion_hash
        };

        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }

    sig
}

// =============================================================================
// SCRATCH SPACE IMPLEMENTATIONS
// =============================================================================

impl Pos0Scratch {
    fn new(num_states: usize, num_groups: usize) -> Self {
        let base_offsets: Vec<usize> = (0..num_states)
            .map(|idx| idx.saturating_mul(num_groups))
            .collect();
        let mask_words = (num_groups + 63) / 64;
        let match_len = num_states.saturating_mul(num_groups);
        Pos0Scratch {
            current_states: vec![0; num_states],
            done: vec![false; num_states],
            active_indices: Vec::new(),
            end_states: vec![None; num_states],
            matched_bits: vec![0u64; num_states.saturating_mul(mask_words)],
            mask_words,
            match_positions: vec![0u32; match_len],
            match_gen: vec![0u32; match_len],
            cur_gen: 1,
            touched_groups: vec![GroupList::new(); num_states],
            touched_positions: Vec::new(),
            touched_states: Vec::new(),
            base_offsets,
            results: Vec::with_capacity(num_states),
            seen_target: Vec::new(),
            all_targets: Vec::new(),
        }
    }

    fn reset(&mut self, initial_states: &[usize], num_groups: usize) {
        let len = initial_states.len();
        if len > self.current_states.len() {
            self.current_states.resize(len, 0);
            self.done.resize(len, false);
            self.end_states.resize(len, None);
            self.matched_bits.resize(len.saturating_mul(self.mask_words), 0);
            let new_len = len.saturating_mul(num_groups);
            self.match_positions.resize(new_len, 0);
            self.match_gen.resize(new_len, 0);
            self.touched_groups.resize(len, GroupList::new());
            self.base_offsets.clear();
            for i in 0..len {
                self.base_offsets.push(i * num_groups);
            }
            self.results.resize(len, (None, EdgeList::new()));
        }

        self.current_states[..len].clone_from_slice(initial_states);
        self.done.fill(false);
        self.active_indices.clear();
        self.end_states[..len].fill(None);

        // Advance generation instead of clearing `match_positions`.
        // If we ever wrap to 0, clear the generation array once.
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 {
            self.match_gen.fill(0);
            self.cur_gen = 1;
        }

        self.touched_positions.clear();

        // Clear touched_groups and matched_bits efficiently
        for &state_idx in &self.touched_states {
            if state_idx < self.touched_groups.len() {
                self.touched_groups[state_idx].clear();
            }
            if self.mask_words > 0 {
                let base = state_idx.saturating_mul(self.mask_words);
                let end = base.saturating_add(self.mask_words);
                if end <= self.matched_bits.len() {
                    self.matched_bits[base..end].fill(0);
                }
            }
        }
        self.touched_states.clear();

        if num_groups == 0 {
            return;
        }

        if self.results.len() < self.current_states.len() {
            self.results.resize_with(self.current_states.len(), || (None, EdgeList::new()));
        }
    }
}

/// Execute DFA from all initial states on a token, returning end states and unique target positions.
///
/// This is the hot-path variant used by vocab equivalence analysis. It avoids allocating/sorting
/// per-state edge lists; instead, it records (gid -> match position) in `match_positions` and
/// the set of touched gids in `touched_groups`, which `compute_chunk_signature` later hashes
/// using the precomputed suffix-cache.
fn compute_pos0_end_states_and_targets(
    pre: &PrecomputedDfa,
    scratch: &mut Pos0Scratch,
    slice: &[u8],
    initial_states: &[usize],
) {
    let num_states = initial_states.len();
    let num_groups = pre.num_groups;
    let len = slice.len();

    scratch.reset(initial_states, num_groups);

    // Prepare all_targets tracking
    let all_targets = &mut scratch.all_targets;

    // Clear seen_target only for positions we saw last time
    let seen_target = &mut scratch.seen_target;
    for &pos in all_targets.iter() {
        if pos < seen_target.len() {
            seen_target[pos] = false;
        }
    }
    all_targets.clear();

    let needed_seen = len + 1;
    if seen_target.len() < needed_seen {
        seen_target.resize(needed_seen, false);
    }

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let active_indices = &mut scratch.active_indices;
    let match_positions = &mut scratch.match_positions;
    let match_gen = &mut scratch.match_gen;
    let cur_gen = scratch.cur_gen;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let matched_bits = &mut scratch.matched_bits;
    let mask_words = scratch.mask_words;
    let base_offsets = &scratch.base_offsets;

    active_indices.clear();
    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Process initial finalizers
    for (i, &state) in initial_states.iter().enumerate() {
        let base = base_offsets[i];
        for f in &pre.finalizers[state] {
            let gid = f.gid;
            if gid < num_groups {
                let idx = base + gid;
                if match_gen[idx] != cur_gen {
                    match_gen[idx] = cur_gen;
                    match_positions[idx] = 0;
                    let groups = &mut touched_groups[i];
                    if groups.is_empty() {
                        touched_states.push(i);
                    }
                    groups.push(gid);

                    if mask_words > 0 {
                        let word = gid >> 6;
                        let bit = 1u64 << (gid & 63);
                        matched_bits[i * mask_words + word] |= bit;
                    }
                }
            }
        }
        if !pre.has_transitions[state] {
            done[i] = true;
            continue;
        }

        if has_bytes {
            let next_state = pre.transitions[state][first_byte as usize];
            if next_state == NONE_STATE {
                done[i] = true;
                continue;
            }
        }

        active_indices.push(i);
    }

    // Process each byte of the token
    if has_bytes && !active_indices.is_empty() {
        let mut active_len = active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;

            unsafe {
                for idx in 0..active_len {
                    let i = *active_indices.get_unchecked(idx);
                    let base = *base_offsets.get_unchecked(i);
                    let current = *current_states.get_unchecked(i);
                    let next_state = *pre
                        .transitions
                        .get_unchecked(current)
                        .get_unchecked(byte as usize);

                    if next_state != NONE_STATE {
                        let next_state = next_state as usize;
                        *current_states.get_unchecked_mut(i) = next_state;

                        for f in pre.finalizers.get_unchecked(next_state) {
                            let gid = f.gid;
                            if gid < num_groups {
                                let idx = base + gid;
                                let slot_pos = match_positions.get_unchecked_mut(idx);
                                let slot_gen = match_gen.get_unchecked_mut(idx);
                                let was_none = *slot_gen != cur_gen;
                                if f.non_greedy {
                                    if was_none {
                                        *slot_gen = cur_gen;
                                        *slot_pos = position;
                                    }
                                } else {
                                    *slot_gen = cur_gen;
                                    *slot_pos = position;
                                }

                                if was_none {
                                    let groups = touched_groups.get_unchecked_mut(i);
                                    if groups.is_empty() {
                                        touched_states.push(i);
                                    }
                                    groups.push(gid);

                                    if mask_words > 0 {
                                        let word = gid >> 6;
                                        let bit = 1u64 << (gid & 63);
                                        *matched_bits
                                            .get_unchecked_mut(i * mask_words + word) |= bit;
                                    }
                                }
                            }
                        }

                        let terminate = match pre.future_modes.get_unchecked(next_state) {
                            FutureMode::AlwaysTerminate => true,
                            FutureMode::AlwaysContinue => false,
                            FutureMode::Guarded(_guard) => {
                                if mask_words == 0 {
                                    true
                                } else {
                                    let guard_mask = pre
                                        .guard_masks
                                        .get_unchecked(next_state)
                                        .as_ref()
                                        .unwrap();
                                    let bits_base = i * mask_words;
                                    let mut all_met = true;
                                    for w in 0..mask_words {
                                        let required = *guard_mask.get_unchecked(w);
                                        if required
                                            & !*matched_bits.get_unchecked(bits_base + w)
                                            != 0
                                        {
                                            all_met = false;
                                            break;
                                        }
                                    }
                                    all_met
                                }
                            }
                        };

                        if terminate {
                            *done.get_unchecked_mut(i) = true;
                        }
                    } else {
                        *done.get_unchecked_mut(i) = true;
                    }

                    if !*done.get_unchecked(i) {
                        *active_indices.get_unchecked_mut(next_len) = i;
                        next_len += 1;
                    }
                }
            }

            active_len = next_len;
            if active_len == 0 {
                break;
            }
        }
    }

    // Collect end states and targets
    for i in 0..num_states {
        let end_state = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };

        scratch.end_states[i] = end_state;

        if num_groups > 0 {
            let base = base_offsets[i];
            for &gid in &touched_groups[i] {
                let pos_val = match_positions[base + gid];
                if pos_val > 0 {
                    let pos_usize = pos_val as usize;
                    if pos_usize <= len && !seen_target[pos_usize] {
                        seen_target[pos_usize] = true;
                        all_targets.push(pos_usize);
                    }
                }
            }
        }
    }

    // Results are stored in-place:
    // - `scratch.end_states[..num_states]`
    // - `scratch.all_targets`
}

impl SuffixScratch {
    fn new(num_groups: usize) -> Self {
        SuffixScratch {
            match_positions: vec![NONE_POS; num_groups],
            touched_positions: GroupList::new(),
            visited: Vec::new(),
            queue: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            pos_hashes: Vec::new(),
            projected_cache: HashMap::new(),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.match_positions.fill(NONE_POS);
        self.touched_positions.clear();
    }

    #[inline]
    fn ensure_capacity(&mut self, len: usize) {
        let needed = len + 1;

        // Only clear entries that were actually visited in the previous run
        for &pos in &self.queue {
            if pos < self.visited.len() {
                self.visited[pos] = false;
            }
            if pos < self.nodes.len() {
                self.nodes[pos] = None;
            }
            if pos < self.pos_hashes.len() {
                self.pos_hashes[pos] = 0;
            }
        }

        // Resize if needed
        if self.visited.len() < needed {
            self.visited.resize(needed, false);
        }
        if self.nodes.len() < needed {
            self.nodes.resize(needed, None);
        }
        if self.pos_hashes.len() < needed {
            self.pos_hashes.resize(needed, 0);
        }

        self.queue.clear();
        self.order.clear();
        self.projected_cache.clear();
    }
}

// =============================================================================
// CORE EXECUTION: POSITION-0 DFA EXECUTION
// =============================================================================

/// Execute DFA from all initial states on a token.
/// Returns (end_state, edges) for each initial state, plus list of unique target positions.
fn compute_pos0_results<'a>(
    pre: &PrecomputedDfa,
    scratch: &'a mut Pos0Scratch,
    slice: &[u8],
    initial_states: &[usize],
) -> (&'a [(Option<usize>, EdgeList)], &'a [usize]) {
    let num_states = initial_states.len();
    let num_groups = pre.num_groups;
    let len = slice.len();

    scratch.reset(initial_states, num_groups);

    // Prepare results vector
    if scratch.results.len() < num_states {
        scratch.results.resize_with(num_states, || (None, EdgeList::new()));
    }
    for i in 0..num_states {
        scratch.results[i].0 = None;
        scratch.results[i].1.clear();
    }

    // Prepare all_targets tracking
    let all_targets = &mut scratch.all_targets;
    
    // Clear seen_target only for positions we saw last time
    let seen_target = &mut scratch.seen_target;
    for &pos in all_targets.iter() {
        if pos < seen_target.len() {
            seen_target[pos] = false;
        }
    }
    all_targets.clear();
    
    let needed_seen = len + 1;
    if seen_target.len() < needed_seen {
        seen_target.resize(needed_seen, false);
    }

    let current_states = &mut scratch.current_states;
    let done = &mut scratch.done;
    let active_indices = &mut scratch.active_indices;
    let match_positions = &mut scratch.match_positions;
    let match_gen = &mut scratch.match_gen;
    let cur_gen = scratch.cur_gen;
    let touched_groups = &mut scratch.touched_groups;
    let touched_positions = &mut scratch.touched_positions;
    let touched_states = &mut scratch.touched_states;
    let matched_bits = &mut scratch.matched_bits;
    let mask_words = scratch.mask_words;
    let base_offsets = &scratch.base_offsets;

    active_indices.clear();
    let has_bytes = !slice.is_empty();
    let first_byte = if has_bytes { slice[0] } else { 0 };

    // Process initial finalizers
    for (i, &state) in initial_states.iter().enumerate() {
        let base = base_offsets[i];
        for f in &pre.finalizers[state] {
            let gid = f.gid;
            if gid < num_groups {
                let idx = base + gid;
                if match_gen[idx] != cur_gen {
                    match_gen[idx] = cur_gen;
                    match_positions[idx] = 0;
                    let groups = &mut touched_groups[i];
                    if groups.is_empty() {
                        touched_states.push(i);
                    }
                    groups.push(gid);

                    if mask_words > 0 {
                        let word = gid >> 6;
                        let bit = 1u64 << (gid & 63);
                        matched_bits[i * mask_words + word] |= bit;
                    }
                }
            }
        }
        if !pre.has_transitions[state] {
            done[i] = true;
            continue;
        }

        if has_bytes {
            let next_state = pre.transitions[state][first_byte as usize];
            if next_state == NONE_STATE {
                done[i] = true;
                continue;
            }
        }

        active_indices.push(i);
    }

    // Process each byte of the token
    if has_bytes && !active_indices.is_empty() {
        let mut active_len = active_indices.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut next_len = 0usize;

            // SAFETY: All indices are pre-validated:
            // - i < num_states, and all arrays are sized to num_states
            // - current_states[i] is always a valid DFA state (< pre.transitions.len())
            // - byte is u8, so byte as usize < 256 (valid for transition table)
            // - base + gid is valid because base_offsets and match_positions are properly sized
            unsafe {
                for idx in 0..active_len {
                    let i = *active_indices.get_unchecked(idx);
                    let base = *base_offsets.get_unchecked(i);
                    let current = *current_states.get_unchecked(i);
                    let next_state = *pre
                        .transitions
                        .get_unchecked(current)
                        .get_unchecked(byte as usize);

                    if next_state != NONE_STATE {
                        let next_state = next_state as usize;
                        *current_states.get_unchecked_mut(i) = next_state;

                        for f in pre.finalizers.get_unchecked(next_state) {
                            let gid = f.gid;
                            if gid < num_groups {
                                let idx = base + gid;
                                let slot_pos = match_positions.get_unchecked_mut(idx);
                                let slot_gen = match_gen.get_unchecked_mut(idx);
                                let was_none = *slot_gen != cur_gen;
                                if f.non_greedy {
                                    if was_none {
                                        *slot_gen = cur_gen;
                                        *slot_pos = position;
                                    }
                                } else {
                                    *slot_gen = cur_gen;
                                    *slot_pos = position;
                                }

                                if was_none {
                                    let groups = touched_groups.get_unchecked_mut(i);
                                    if groups.is_empty() {
                                        touched_states.push(i);
                                    }
                                    groups.push(gid);

                                    if mask_words > 0 {
                                        let word = gid >> 6;
                                        let bit = 1u64 << (gid & 63);
                                        *matched_bits
                                            .get_unchecked_mut(i * mask_words + word) |= bit;
                                    }
                                }
                            }
                        }

                        let terminate = match pre.future_modes.get_unchecked(next_state) {
                            FutureMode::AlwaysTerminate => true,
                            FutureMode::AlwaysContinue => false,
                            FutureMode::Guarded(_guard) => {
                                if mask_words == 0 {
                                    true
                                } else {
                                    let guard_mask = pre
                                        .guard_masks
                                        .get_unchecked(next_state)
                                        .as_ref()
                                        .unwrap();
                                    let bits_base = i * mask_words;
                                    let mut all_met = true;
                                    for w in 0..mask_words {
                                        let required = *guard_mask.get_unchecked(w);
                                        if required
                                            & !*matched_bits.get_unchecked(bits_base + w)
                                            != 0
                                        {
                                            all_met = false;
                                            break;
                                        }
                                    }
                                    all_met
                                }
                            }
                        };

                        if terminate {
                            *done.get_unchecked_mut(i) = true;
                        }
                    } else {
                        *done.get_unchecked_mut(i) = true;
                    }

                    if !*done.get_unchecked(i) {
                        *active_indices.get_unchecked_mut(next_len) = i;
                        next_len += 1;
                    }
                }
            }

            active_len = next_len;
            if active_len == 0 {
                break;
            }
        }
    }

    // Collect results
    for i in 0..num_states {
        let end_state = if done[i] || !pre.has_transitions[current_states[i]] {
            None
        } else {
            Some(current_states[i])
        };

        let edges = &mut scratch.results[i].1;
        if num_groups > 0 {
            let base = base_offsets[i];
            for &gid in &touched_groups[i] {
                if gid >= num_groups {
                    continue;
                }
                let idx = base + gid;
                if match_gen[idx] != cur_gen {
                    continue;
                }
                let pos_val = match_positions[idx];
                if pos_val > 0 {
                    let pos_usize = pos_val as usize;
                    edges.push((gid, pos_usize));
                    if pos_usize <= len && !seen_target[pos_usize] {
                        seen_target[pos_usize] = true;
                        all_targets.push(pos_usize);
                    }
                }
            }
        }

        edges.sort_unstable_by_key(|e| e.0);
        scratch.results[i].0 = end_state;
    }

    (&scratch.results[..num_states], &scratch.all_targets)
}

// =============================================================================
// CORE EXECUTION: SUFFIX HASH COMPUTATION
// =============================================================================

/// Execute DFA on a suffix starting from position base_pos.
#[inline]
fn execute_suffix(
    pre: &PrecomputedDfa,
    slice: &[u8],
    base_pos: usize,
    scratch: &mut SuffixScratch,
) -> (Option<usize>, EdgeList) {
    let num_groups = pre.num_groups;

    if num_groups > 0 {
        scratch.reset();
    }

    let match_positions = &mut scratch.match_positions;
    let touched = &mut scratch.touched_positions;

    let mut current = pre.start_state;
    let mut done = false;

    // Initial finalizers
    if num_groups > 0 {
        for f in &pre.finalizers[current] {
            let gid = f.gid;
            if gid < num_groups {
                let slot = &mut match_positions[gid];
                let was_none = *slot == NONE_POS;
                if f.non_greedy {
                    if was_none {
                        *slot = 0;
                    }
                } else {
                    *slot = 0;
                }
                if was_none {
                    touched.push(gid);
                }
            }
        }
    }

    if !pre.has_transitions[current] {
        done = true;
    }

    // Process each byte
    for (idx, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }

        let next_state = pre.transitions[current][byte as usize];
        if next_state != NONE_STATE {
            let next_state = next_state as usize;
            current = next_state;
            let position = (idx + 1) as u32;

            if num_groups > 0 {
                for f in &pre.finalizers[current] {
                    let gid = f.gid;
                    if gid < num_groups {
                        let slot = &mut match_positions[gid];
                        let was_none = *slot == NONE_POS;
                        if f.non_greedy {
                            if was_none {
                                *slot = position;
                            }
                        } else {
                            *slot = position;
                        }

                        if was_none {
                            touched.push(gid);
                        }
                    }
                }
            }

            let terminate = match &pre.future_modes[current] {
                FutureMode::AlwaysTerminate => true,
                FutureMode::AlwaysContinue => false,
                FutureMode::Guarded(guard) => {
                    guard.iter().all(|&gid| match_positions[gid] != NONE_POS)
                }
            };

            if terminate {
                done = true;
            }
        } else {
            done = true;
        }
    }

    let end_state = if done || !pre.has_transitions[current] {
        None
    } else {
        Some(current)
    };

    let mut edges: EdgeList = SmallVec::new();
    if num_groups > 0 {
        touched.sort_unstable();
        for &gid in touched.iter() {
            let pos_val = match_positions[gid];
            if pos_val != NONE_POS && pos_val != 0 {
                edges.push((gid, base_pos + pos_val as usize));
            }
        }
    }

    (end_state, edges)
}

/// Compute suffix hashes incrementally, updating the cache.
fn compute_suffix_hashes_incremental(
    pre: &PrecomputedDfa,
    slice: &[u8],
    new_targets: &[usize],
    cache: &mut Vec<Option<u64>>,
    scratch: &mut SuffixScratch,
    _suffix_group_mask: Option<&[bool]>,
    group_to_class: Option<&[usize]>,
) {
    // Build suffix DAG (also used by projected hash computation)
    build_suffix_dag(pre, slice, new_targets, scratch);

    // Compute unprojected hashes from the DAG
    // Process in reverse order (bottom-up for DAG)
    scratch.order.sort_unstable_by(|a, b| b.cmp(a));

    for &pos in &scratch.order {
        if cache[pos].is_some() {
            continue;
        }
        if let Some((completion_hash, ref edges)) = scratch.nodes[pos] {
            let mut hasher = new_hasher();
            hasher.write_u64(completion_hash);

            for &(group_id, target) in edges.iter() {
                let target_hash = cache[target].unwrap_or(0);
                hasher.write_u64(group_id as u64);
                hasher.write_u64(target_hash);
            }

            cache[pos] = Some(hasher.finish());
        }
    }
    scratch.order.clear();
}

/// Build the suffix DAG without computing hashes.
/// After this call, `scratch.nodes[pos]` contains `(completion_hash, edges)` for each
/// reachable suffix position. The DAG can be used for projected hash computation.
fn build_suffix_dag(
    pre: &PrecomputedDfa,
    slice: &[u8],
    new_targets: &[usize],
    scratch: &mut SuffixScratch,
) {
    scratch.ensure_capacity(slice.len());

    // Queue positions that need computation
    for &pos in new_targets {
        if pos <= slice.len() && scratch.nodes[pos].is_none() && !scratch.visited[pos] {
            scratch.visited[pos] = true;
            scratch.queue.push(pos);
        }
    }

    if scratch.queue.is_empty() {
        return;
    }

    // BFS to discover all reachable positions
    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;

        let (end_state, edges) = execute_suffix(pre, &slice[pos..], pos, scratch);

        for &(_, target) in &edges {
            if target <= slice.len() && scratch.nodes[target].is_none() && !scratch.visited[target] {
                scratch.visited[target] = true;
                scratch.queue.push(target);
            }
        }

        let completion_hash = end_state
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);
        scratch.nodes[pos] = Some((completion_hash, edges));
        scratch.order.push(pos);
    }
}

/// Compute a projected suffix hash for a specific position, only considering
/// edges whose group is allowed after `parent_group`. Uses memoization via `projected_cache`.
///
/// This mirrors `prune_trellis_recursive`: at each level, only edges in
/// `ever_allowed_by_group[parent_group]` are included in the hash.
fn compute_projected_suffix_hash(
    pos: usize,
    parent_group: usize,
    ever_allowed_by_group: &[Vec<bool>],
    nodes: &[Option<(u64, EdgeList)>],
    projected_cache: &mut HashMap<(usize, usize), u64>,
    group_to_class: Option<&[usize]>,
) -> u64 {
    let key = (pos, parent_group);
    if let Some(&cached) = projected_cache.get(&key) {
        return cached;
    }

    let hash = if let Some((completion_hash, ref edges)) = nodes[pos] {
        let mut hasher = new_hasher();
        hasher.write_u64(completion_hash);

        let allowed = if parent_group < ever_allowed_by_group.len() {
            Some(&ever_allowed_by_group[parent_group])
        } else {
            None
        };

        for &(group_id, target) in edges.iter() {
            // Check if group_id is allowed after parent_group
            let is_allowed = match allowed {
                Some(mask) => group_id < mask.len() && mask[group_id],
                None => true, // No follow info -> allow all
            };
            if !is_allowed {
                continue;
            }
            // Recurse with group_id as the new parent
            let target_hash = compute_projected_suffix_hash(
                target, group_id, ever_allowed_by_group, nodes, projected_cache, group_to_class,
            );
            hasher.write_u64(group_id as u64);
            hasher.write_u64(target_hash);
        }

        hasher.finish()
    } else {
        0
    };

    projected_cache.insert(key, hash);
    hash
}

// =============================================================================
// SIGNATURE COMPUTATION
// =============================================================================

/// Compute the signature for a token given a chunk of initial states.
fn compute_chunk_signature(
    pre: &PrecomputedDfa,
    token: &[u8],
    chunk_states: &[usize],
    pos0: &mut Pos0Scratch,
    suffix_scratch: &mut SuffixScratch,
    cache: &mut Vec<Option<u64>>,
    suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
    skip_groups: bool,
) -> u64 {
    compute_pos0_end_states_and_targets(pre, pos0, token, chunk_states);

    // Only compute suffix hashes when there are match targets
    if !pos0.all_targets.is_empty() {
        compute_suffix_hashes_incremental(pre, token, &pos0.all_targets, cache, suffix_scratch, suffix_group_mask, group_to_class);
    }

    // If ever_allowed_by_group is provided, we'll compute projected hashes
    // that prune suffix edges based on which group was matched at position 0.
    let use_projected = ever_allowed_by_group.is_some();

    let num_groups = pre.num_groups;
    let include_groups = num_groups > 0 && !skip_groups;

    // Fast path: combine per-state signatures using wrapping_mul (avoids creating
    // a top-level AHasher). Only states with group matches need a full hasher.
    let mut sig: u64 = HASH_SEED3;
    for i in 0..chunk_states.len() {
        let completion_hash = pos0.end_states[i]
            .map(|id| pre.completion_hash[id])
            .unwrap_or(pre.none_completion_hash);

        let state_sig = if include_groups && !pos0.touched_groups[i].is_empty() {
            // This state has group matches - hash them
            let groups = &mut pos0.touched_groups[i];
            if groups.len() > 1 {
                groups.sort_unstable();
            }
            let base = pos0.base_offsets[i];
            let mut h = new_hasher();
            h.write_u64(completion_hash);
            for &gid in groups.iter() {
                let pos_val = pos0.match_positions[base + gid];
                if pos_val > 0 {
                    let target_hash = if use_projected {
                        let ea = ever_allowed_by_group.unwrap();
                        compute_projected_suffix_hash(
                            pos_val as usize,
                            gid,
                            ea,
                            &suffix_scratch.nodes,
                            &mut suffix_scratch.projected_cache,
                            group_to_class,
                        )
                    } else {
                        cache[pos_val as usize].unwrap_or(0)
                    };
                    h.write_u64(gid as u64);
                    h.write_u64(target_hash);
                }
            }
            h.finish()
        } else {
            // No group matches at this state - just use completion hash directly
            completion_hash
        };

        // Order-preserving combination of per-state signatures
        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(state_sig);
    }

    sig
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

/// Find vocab equivalence classes of tokens based on DFA behavior.
/// Uses iterative state-based refinement with batching and parallel processing.
/// 
/// Note: For large state counts, the caller should pre-reduce using
/// `state_equivalence_analysis::find_state_equivalence_classes`
/// before calling this function. This is typically done in constraint.rs.
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `strings` - Vocabulary tokens to analyze
/// * `initial_states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// Sets of token indices that are equivalent (produce identical parsing behavior).
pub fn find_vocab_equivalence_classes(
    regex: &Tokenizer,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes with optional follow-set pruning.
///
/// `suffix_group_mask`: if provided, suffix hashes will only include edges for groups where
/// `mask[gid] == true`. Groups not in the mask are ignored in suffix positions, causing
/// tokens that differ only in those groups to be merged. The mask should be `true` for
/// groups that can appear after any other group (i.e., groups that appear in at least one
/// follow set).
///
/// `ever_allowed_by_group`: if provided, per-group follow masks. `ever_allowed_by_group[g]`
/// is a bool mask: `mask[h] == true` means group h can follow group g. When this is
/// provided, suffix hashes use projected computation that prunes edges per-context.
pub fn find_vocab_equivalence_classes_with_follow(
    regex: &Tokenizer,
    strings: &[Vec<u8>],
    initial_states: &[usize],
    suffix_group_mask: Option<&[bool]>,
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    use std::time::Instant;
    
    let total_start = Instant::now();
    let pre = precompute_dfa(regex);
    let precompute_time = total_start.elapsed();

    // Note: State equivalence reduction (if needed) should be done by the caller.
    let reduced_initial_states: Vec<usize> = initial_states.to_vec();

    if is_debug_level_enabled(3) {
        crate::debug!(
            3,
            "fast vocab equivalence: num_states={} num_groups={} precompute={:?}",
            reduced_initial_states.len(),
            pre.num_groups,
            precompute_time,
        );
    }

    let num_tokens = strings.len();
    let num_states = reduced_initial_states.len();

    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    // Analyze state transition sparsity for large state sets
    if num_states > 2000 {
        let mut total_transitions = 0usize;
        let mut states_with_few_transitions = 0usize;
        for &sid in &reduced_initial_states {
            let trans = &pre.transitions[sid];
            let count = trans.iter().filter(|&&t| t != NONE_STATE).count();
            total_transitions += count;
            if count < 10 {
                states_with_few_transitions += 1;
            }
        }
        let avg_transitions = total_transitions as f64 / num_states as f64;
        crate::debug!(
            3,
            "State transition analysis: avg_transitions={:.1}, sparse_states={}/{} ({:.1}%)",
            avg_transitions,
            states_with_few_transitions,
            num_states,
            100.0 * states_with_few_transitions as f64 / num_states as f64
        );
    }

    let num_groups = pre.num_groups;

    // Check diagnostic flags once (not per-token)
    let skip_groups = std::env::var("EQUIV_SKIP_GROUPS").is_ok();
    let use_trie = std::env::var("USE_TRIE_EQUIV").is_ok();

    // Build trie and precompute suffix hashes if using trie path
    let trie = if use_trie {
        let trie_start = std::time::Instant::now();
        let t = VocabTrie::build(strings);
        if is_debug_level_enabled(3) {
            crate::debug!(
                3,
                "Built vocab trie: {} nodes from {} tokens in {:?}",
                t.num_nodes(),
                num_tokens,
                trie_start.elapsed(),
            );
        }
        Some(t)
    } else {
        None
    };

    // Precompute suffix hashes for all tokens if using trie path
    let suffix_caches: Option<Vec<Vec<Option<u64>>>> = if use_trie {
        let suffix_start = std::time::Instant::now();
        let caches: Vec<Vec<Option<u64>>> = strings
            .par_iter()
            .map(|token| {
                precompute_all_suffix_hashes(&pre, token, suffix_group_mask, group_to_class)
            })
            .collect();
        if is_debug_level_enabled(3) {
            crate::debug!(
                3,
                "Precomputed suffix hashes for {} tokens in {:?}",
                num_tokens,
                suffix_start.elapsed(),
            );
        }
        Some(caches)
    } else {
        None
    };

    // Process states in batches for memory efficiency.
    // Smaller batches improve cache locality for match_positions array
    // (batch_size * num_groups * 4 bytes per thread) and enable early pruning of singletons.
    let batch_size = if num_states < 200 { num_states } else { 200 };

    let mut active_indices: Vec<usize> = (0..num_tokens).collect();
    let mut partition: Vec<usize> = vec![0; num_tokens];
    let mut next_class_id = 1usize;

    if is_debug_level_enabled(4) {
        crate::debug!(
            4,
            "  Iterative refinement: {} tokens, {} states, batch_size={}",
            num_tokens,
            num_states,
            batch_size
        );
    }

    let mut batch_count = 0;
    
    // Timing accumulators
    let mut total_refine_time = std::time::Duration::ZERO;

    for batch_start in (0..num_states).step_by(batch_size) {
        if active_indices.is_empty() {
            break;
        }

        let batch_end = (batch_start + batch_size).min(num_states);
        let batch = &reduced_initial_states[batch_start..batch_end];

        // Compute partial signatures for active tokens
        let batch_start_time = Instant::now();
        let active_sigs: Vec<(usize, u64)> = if let (Some(ref trie), Some(ref suffix_caches)) = (&trie, &suffix_caches) {
            // TRIE PATH: process states in parallel through the trie
            compute_batch_signatures_trie(
                trie,
                &pre,
                batch,
                strings,
                &active_indices,
                suffix_caches,
                ever_allowed_by_group,
                group_to_class,
                num_tokens,
            )
        } else {
            // ORIGINAL PATH: process tokens in parallel
            active_indices
            .par_iter()
            .map_init(
                || {
                    (
                        Pos0Scratch::new(batch.len(), num_groups),
                        SuffixScratch::new(num_groups),
                        vec![None; 256],
                    )
                },
                |state, &token_idx| {
                    let (scratch_pos0, scratch_suffix, scratch_cache) = state;
                    let token = &strings[token_idx];

                    // Ensure cache is large enough
                    if scratch_cache.len() <= token.len() {
                        scratch_cache.resize(token.len() + 1, None);
                    }
                    scratch_cache.iter_mut().for_each(|x| *x = None);

                    let sig = compute_chunk_signature(&pre, token, batch, scratch_pos0, scratch_suffix, scratch_cache, suffix_group_mask, ever_allowed_by_group, group_to_class, skip_groups);
                    (token_idx, sig)
                },
            )
            .collect()
        };
        let batch_compute_time = batch_start_time.elapsed();

        // Group by (old_class, new_signature) to refine partition
        let refine_start = Instant::now();
        let mut refinement: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(active_sigs.len() / 2);
        for (token_idx, sig) in active_sigs {
            let old_class = partition[token_idx];
            refinement
                .entry((old_class, sig))
                .or_insert_with(Vec::new)
                .push(token_idx);
        }

        // Group refinement entries by old_class
        let mut by_old_class: HashMap<usize, Vec<(u64, Vec<usize>)>> = HashMap::new();
        for ((old_class, sig), tokens) in refinement {
            by_old_class
                .entry(old_class)
                .or_insert_with(Vec::new)
                .push((sig, tokens));
        }

        // Update partition and find still-active tokens
        let mut new_active_indices = Vec::with_capacity(active_indices.len());

        for (_old_class, sub_groups) in by_old_class {
            let mut first = true;
            for (_sig, tokens) in sub_groups {
                let class_to_use = if first {
                    first = false;
                    _old_class
                } else {
                    let id = next_class_id;
                    next_class_id += 1;
                    id
                };

                for &token_idx in &tokens {
                    partition[token_idx] = class_to_use;
                }

                if tokens.len() > 1 {
                    new_active_indices.extend(tokens);
                }
            }
        }
        total_refine_time += refine_start.elapsed();

        active_indices = new_active_indices;
        batch_count += 1;

        if is_debug_level_enabled(5) {
            let num_classes = {
                let mut seen: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
                for &c in &partition {
                    seen.insert(c);
                }
                seen.len()
            };
            crate::debug!(
                5,
                "    Batch {}: {} active tokens, {} classes, compute={:?}",
                batch_count,
                active_indices.len(),
                num_classes,
                batch_compute_time,
            );
        }
    }

    if is_debug_level_enabled(4) {
        crate::debug!(
            4,
            "  Timing: refine={:?}",
            total_refine_time,
        );
    }

    // Build final groups from partition
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_class_id);
    for (token_idx, &class_id) in partition.iter().enumerate() {
        groups.entry(class_id).or_insert_with(Vec::new).push(token_idx);
    }

    if is_debug_level_enabled(4) {
        crate::debug!(
            4,
            "  Computed {} vocab equivalence classes in {} batches",
            groups.len(),
            batch_count
        );
    }

    groups.into_values().collect()
}

// =============================================================================
// TRIE-BASED BATCH SIGNATURE COMPUTATION
// =============================================================================

/// Compact byte-level trie for vocabulary prefix sharing.
/// Reduces DFA transitions by ~69% by sharing work for common token prefixes.
struct VocabTrie {
    /// Flat array of trie nodes. Node 0 is the root.
    nodes: Vec<TrieNode>,
}

struct TrieNode {
    /// Children sorted by byte for deterministic traversal.
    /// (byte, child_node_index)
    children: SmallVec<[(u8, u32); 4]>,
    /// Token index if this node is a leaf (complete token), else u32::MAX.
    token_idx: u32,
    /// Number of tokens reachable from this subtree (for active filtering).
    subtree_size: u32,
}

impl VocabTrie {
    fn build(tokens: &[Vec<u8>]) -> Self {
        let mut nodes = Vec::with_capacity(tokens.len() * 2);
        nodes.push(TrieNode {
            children: SmallVec::new(),
            token_idx: u32::MAX,
            subtree_size: 0,
        });

        for (idx, token) in tokens.iter().enumerate() {
            let mut current = 0u32;
            for &byte in token {
                let pos = nodes[current as usize]
                    .children
                    .iter()
                    .position(|&(b, _)| b == byte);
                current = match pos {
                    Some(p) => nodes[current as usize].children[p].1,
                    None => {
                        let new_idx = nodes.len() as u32;
                        nodes.push(TrieNode {
                            children: SmallVec::new(),
                            token_idx: u32::MAX,
                            subtree_size: 0,
                        });
                        nodes[current as usize].children.push((byte, new_idx));
                        new_idx
                    }
                };
            }
            nodes[current as usize].token_idx = idx as u32;
        }

        // Sort children by byte for deterministic ordering
        for node in &mut nodes {
            node.children.sort_unstable_by_key(|&(b, _)| b);
        }

        // Compute subtree sizes (post-order)
        fn compute_subtree_size(nodes: &mut [TrieNode], idx: u32) -> u32 {
            let has_token = if nodes[idx as usize].token_idx != u32::MAX { 1 } else { 0 };
            let children: SmallVec<[(u8, u32); 4]> = nodes[idx as usize].children.clone();
            let mut size = has_token;
            for &(_, child_idx) in &children {
                size += compute_subtree_size(nodes, child_idx);
            }
            nodes[idx as usize].subtree_size = size;
            size
        }
        compute_subtree_size(&mut nodes, 0);

        VocabTrie { nodes }
    }

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Per-state match tracking using sparse representation.
/// Only stores groups that actually matched, keeping save/restore cheap.
#[derive(Clone)]
struct SparseMatchState {
    /// (group_id, position, is_non_greedy). Sorted by group_id.
    entries: SmallVec<[(usize, u32, bool); 8]>,
}

impl SparseMatchState {
    fn new() -> Self {
        SparseMatchState {
            entries: SmallVec::new(),
        }
    }

    #[inline]
    fn update(&mut self, gid: usize, position: u32, non_greedy: bool) {
        match self.entries.binary_search_by_key(&gid, |&(g, _, _)| g) {
            Ok(idx) => {
                // Group already matched
                if !non_greedy {
                    // Greedy: update to latest position
                    self.entries[idx].1 = position;
                }
                // Non-greedy: keep first match (do nothing)
            }
            Err(idx) => {
                // New group match
                self.entries.insert(idx, (gid, position, non_greedy));
            }
        }
    }
}

/// Precomputed suffix hashes for all positions of a token.
/// suffix_hashes[pos] = hash of suffix structure from position pos.
struct TokenSuffixHashes {
    /// Indexed by position (0..=token.len()). None if not computed.
    hashes: Vec<Option<u64>>,
    /// DAG nodes for projected hash computation.
    nodes: Vec<Option<(u64, EdgeList)>>,
}

/// Precompute suffix hashes for ALL positions of a token.
fn precompute_all_suffix_hashes(
    pre: &PrecomputedDfa,
    token: &[u8],
    suffix_group_mask: Option<&[bool]>,
    group_to_class: Option<&[usize]>,
) -> Vec<Option<u64>> {
    let len = token.len();
    let all_positions: Vec<usize> = (0..=len).collect();

    let mut scratch = SuffixScratch::new(pre.num_groups);
    let mut cache: Vec<Option<u64>> = vec![None; len + 1];

    compute_suffix_hashes_incremental(
        pre,
        token,
        &all_positions,
        &mut cache,
        &mut scratch,
        suffix_group_mask,
        group_to_class,
    );

    cache
}

/// Process a single initial state through the trie via DFS.
/// Returns partial signature (u64) for each token reached.
fn dfs_single_state(
    trie: &VocabTrie,
    pre: &PrecomputedDfa,
    initial_state: usize,
    strings: &[Vec<u8>],
    active: &[bool],
    suffix_caches: &[Vec<Option<u64>>],
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
    partial_sigs: &mut Vec<(u32, u64)>, // output: (token_idx, partial_sig)
) {
    let num_groups = pre.num_groups;
    let use_projected = ever_allowed_by_group.is_some();

    // Stack entry for DFS backtracking
    struct Frame {
        node_idx: u32,
        dfa_state: usize,     // DFA state at this trie node
        match_state: SparseMatchState,
        child_idx: u16,        // next child to process
        position: u32,         // byte position (depth in trie)
    }

    let root_state = initial_state;
    let mut root_match = SparseMatchState::new();

    // Check root finalizers (position 0)
    if num_groups > 0 {
        for f in &pre.finalizers[root_state] {
            if f.gid < num_groups {
                root_match.update(f.gid, 0, f.non_greedy);
            }
        }
    }

    // If root is a leaf (empty token)
    let root_node = &trie.nodes[0];
    let root_is_done = !pre.has_transitions[root_state];
    if root_node.token_idx != u32::MAX {
        let token_idx = root_node.token_idx;
        if active[token_idx as usize] {
            let completion_hash = if root_is_done {
                pre.none_completion_hash
            } else {
                pre.completion_hash[root_state]
            };
            let sig = compute_state_leaf_sig_with_completion(
                completion_hash, &root_match, &suffix_caches[token_idx as usize],
                ever_allowed_by_group, group_to_class,
            );
            partial_sigs.push((token_idx, sig));
        }
    }

    // Iterative DFS using explicit stack
    let mut stack: Vec<Frame> = Vec::with_capacity(32);

    if !root_node.children.is_empty() && pre.has_transitions[root_state] {
        stack.push(Frame {
            node_idx: 0,
            dfa_state: root_state,
            match_state: root_match.clone(),
            child_idx: 0,
            position: 0,
        });
    }

    if root_is_done && !root_node.children.is_empty() {
        // Root state has no transitions: emit terminated sigs for all tokens
        emit_terminated_descendants(
            trie, 0, pre, &root_match, active,
            suffix_caches, ever_allowed_by_group, group_to_class,
            partial_sigs,
        );
    }

    while let Some(frame) = stack.last_mut() {
        let node = &trie.nodes[frame.node_idx as usize];

        if frame.child_idx as usize >= node.children.len() {
            // All children processed, backtrack
            stack.pop();
            continue;
        }

        let (byte, child_idx) = node.children[frame.child_idx as usize];
        frame.child_idx += 1;

        // Transition
        let next_state_raw = pre.transitions[frame.dfa_state][byte as usize];
        if next_state_raw == NONE_STATE {
            // Dead end: emit terminated signatures for the child and all descendants
            let child_node_dead = &trie.nodes[child_idx as usize];
            if child_node_dead.token_idx != u32::MAX {
                let token_idx = child_node_dead.token_idx;
                if active[token_idx as usize] {
                    let sig = compute_state_leaf_sig_with_completion(
                        pre.none_completion_hash, &frame.match_state,
                        &suffix_caches[token_idx as usize],
                        ever_allowed_by_group, group_to_class,
                    );
                    partial_sigs.push((token_idx, sig));
                }
            }
            if !child_node_dead.children.is_empty() {
                emit_terminated_descendants(
                    trie, child_idx, pre, &frame.match_state, active,
                    suffix_caches, ever_allowed_by_group, group_to_class,
                    partial_sigs,
                );
            }
            continue;
        }
        let next_state = next_state_raw as usize;

        // Update match state with finalizers
        let mut child_match = frame.match_state.clone();
        let child_position = frame.position + 1;

        if num_groups > 0 {
            for f in &pre.finalizers[next_state] {
                if f.gid < num_groups {
                    child_match.update(f.gid, child_position, f.non_greedy);
                }
            }
        }

        let child_node = &trie.nodes[child_idx as usize];

        // Check termination (guard masks)
        let terminate = match &pre.future_modes[next_state] {
            FutureMode::AlwaysTerminate => true,
            FutureMode::AlwaysContinue => false,
            FutureMode::Guarded(guard) => {
                // Check if all guarded groups have been matched
                guard.iter().all(|&gid| {
                    child_match.entries.binary_search_by_key(&gid, |&(g, _, _)| g).is_ok()
                })
            }
        };

        // Determine whether this state is "done" (like the original code)
        // Original: end_state = None when done[i] || !has_transitions[state]
        let is_done = terminate || !pre.has_transitions[next_state];


        // Emit leaf signature if this node has a token
        if child_node.token_idx != u32::MAX {
            let token_idx = child_node.token_idx;
            if active[token_idx as usize] {
                let completion_hash = if is_done {
                    pre.none_completion_hash
                } else {
                    pre.completion_hash[next_state]
                };
                let sig = compute_state_leaf_sig_with_completion(
                    completion_hash, &child_match,
                    &suffix_caches[token_idx as usize],
                    ever_allowed_by_group, group_to_class,
                );
                partial_sigs.push((token_idx, sig));
            }
        }

        if is_done {
            // State terminated: emit terminated signatures for ALL descendant
            // tokens in the subtree. They get none_completion_hash + current match state.
            if !child_node.children.is_empty() {
                emit_terminated_descendants(
                    trie, child_idx, pre, &child_match, active,
                    suffix_caches, ever_allowed_by_group, group_to_class,
                    partial_sigs,
                );
            }
        } else if !child_node.children.is_empty() {
            // Continue DFS with updated DFA state
            stack.push(Frame {
                node_idx: child_idx,
                dfa_state: next_state,
                match_state: child_match,
                child_idx: 0,
                position: child_position,
            });
        }
    }
}

/// Emit terminated signatures for all descendant tokens in a trie subtree.
/// When a DFA state terminates early, all tokens deeper in the trie get
/// signatures with none_completion_hash and the match state at termination.
fn emit_terminated_descendants(
    trie: &VocabTrie,
    node_idx: u32,
    pre: &PrecomputedDfa,
    match_state: &SparseMatchState,
    active: &[bool],
    suffix_caches: &[Vec<Option<u64>>],
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
    partial_sigs: &mut Vec<(u32, u64)>,
) {
    // Iterative traversal of the subtree
    let mut visit_stack: Vec<u32> = Vec::new();
    let node = &trie.nodes[node_idx as usize];
    for &(_, child_idx) in &node.children {
        visit_stack.push(child_idx);
    }
    while let Some(idx) = visit_stack.pop() {
        let n = &trie.nodes[idx as usize];
        if n.token_idx != u32::MAX {
            let token_idx = n.token_idx;
            if active[token_idx as usize] {
                let sig = compute_state_leaf_sig_with_completion(
                    pre.none_completion_hash, match_state,
                    &suffix_caches[token_idx as usize],
                    ever_allowed_by_group, group_to_class,
                );
                partial_sigs.push((token_idx, sig));
            }
        }
        for &(_, child_idx) in &n.children {
            visit_stack.push(child_idx);
        }
    }
}

/// Compute the partial signature for a single state at a leaf node.
#[inline]
fn compute_state_leaf_sig_with_completion(
    completion_hash: u64,
    match_state: &SparseMatchState,
    suffix_cache: &[Option<u64>],
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
) -> u64 {
    let mut hasher = new_hasher();
    hasher.write_u64(completion_hash);

    // Group match signatures
    for &(gid, pos_val, _) in &match_state.entries {
        if pos_val > 0 {
            let target_hash = suffix_cache.get(pos_val as usize)
                .and_then(|h| *h)
                .unwrap_or(0);
            hasher.write_u64(gid as u64);
            hasher.write_u64(target_hash);
        }
    }

    hasher.finish()
}

/// Dead-state signature for states that hit a dead end in the trie.
fn compute_dead_state_sig(pre: &PrecomputedDfa) -> u64 {
    let mut hasher = new_hasher();
    hasher.write_u64(pre.none_completion_hash);
    hasher.finish()
}

/// Compute batch signatures using trie-based prefix sharing.
/// Each initial state is processed independently through the trie via DFS.
/// Parallelism is over states (not tokens).
fn compute_batch_signatures_trie(
    trie: &VocabTrie,
    pre: &PrecomputedDfa,
    batch: &[usize],
    strings: &[Vec<u8>],
    active_indices: &[usize],
    suffix_caches: &[Vec<Option<u64>>],
    ever_allowed_by_group: Option<&[Vec<bool>]>,
    group_to_class: Option<&[usize]>,
    num_tokens: usize,
) -> Vec<(usize, u64)> {
    let batch_size = batch.len();

    // Build active token bitset
    let mut active = vec![false; num_tokens];
    for &idx in active_indices {
        active[idx] = true;
    }

    // Process each state through the trie in parallel
    // Each thread collects (token_idx, partial_sig) pairs
    let per_state_results: Vec<Vec<(u32, u64)>> = batch
        .par_iter()
        .map(|&initial_state| {
            let mut partial_sigs = Vec::with_capacity(active_indices.len());
            dfs_single_state(
                trie,
                pre,
                initial_state,
                strings,
                &active,
                suffix_caches,
                ever_allowed_by_group,
                group_to_class,
                &mut partial_sigs,
            );
            partial_sigs
        })
        .collect();

    // Combine per-state partial sigs into per-token full sigs
    // Hash (state_idx, partial_sig) per entry, then sum — order-aware combination
    let mut combined: Vec<u64> = vec![0; num_tokens];
    for (state_idx, state_results) in per_state_results.iter().enumerate() {
        for &(token_idx, partial_sig) in state_results {
            let mut h = new_hasher();
            h.write_u64(state_idx as u64);
            h.write_u64(partial_sig);
            combined[token_idx as usize] = combined[token_idx as usize].wrapping_add(h.finish());
        }
    }

    // Handle tokens that weren't reached by ANY state (dead in all states)
    let dead_sig = compute_dead_state_sig(pre);
    let dead_combined = {
        let mut h_total = 0u64;
        for state_idx in 0..batch_size {
            let mut h = new_hasher();
            h.write_u64(state_idx as u64);
            h.write_u64(dead_sig);
            h_total = h_total.wrapping_add(h.finish());
        }
        h_total
    };

    // Collect results for active tokens
    active_indices
        .iter()
        .map(|&token_idx| {
            let sig = combined[token_idx];
            // If sig is 0 (no state reached this token), use dead_combined
            let final_sig = if sig == 0 { dead_combined } else { sig };
            (token_idx, final_sig)
        })
        .collect()
}

// =============================================================================
// DEBUG/TEST UTILITIES
// =============================================================================

fn compute_suffix_hashes_debug(
    regex: &Tokenizer,
    slice: &[u8],
    all_targets: &[usize],
) -> Vec<u64> {
    use std::collections::VecDeque;

    let len = slice.len();
    if all_targets.is_empty() {
        return vec![0; len + 1];
    }

    let mut visited = vec![false; len + 1];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut order: Vec<usize> = Vec::new();
    let mut nodes: Vec<Option<(Option<usize>, EdgeList)>> = vec![None; len + 1];

    for &pos in all_targets {
        if pos > 0 && pos <= len && !visited[pos] {
            visited[pos] = true;
            queue.push_back(pos);
        }
    }

    while let Some(pos) = queue.pop_front() {
        let result = regex.execute_from_state_nonzero(&slice[pos..], regex.dfa().start_state);

        let mut edges: EdgeList = result
            .matches
            .iter()
            .map(|m| {
                let target = pos + m.position;
                if target <= len && !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                }
                (m.group_id, target)
            })
            .collect();

        edges.sort_unstable_by_key(|e| e.0);
        nodes[pos] = Some((result.end_state, edges));
        order.push(pos);
    }

    order.sort_unstable_by(|a, b| b.cmp(a));
    let mut pos_hashes: Vec<u64> = vec![0; len + 1];

    for pos in order {
        if let Some((end_state, edges)) = &nodes[pos] {
            let completion =
                end_state.map(|id| regex.dfa().states[id].possible_future_group_ids.clone());
            let mut hasher = DefaultHasher::new();
            completion.hash(&mut hasher);
            for (group_id, target) in edges {
                let target_hash = pos_hashes[*target];
                (group_id, target_hash).hash(&mut hasher);
            }
            pos_hashes[pos] = hasher.finish();
        }
    }

    pos_hashes
}

pub fn compute_signature_debug(
    regex: &Tokenizer,
    slice: &[u8],
    initial_states: &[usize],
) -> Vec<u64> {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, all_targets) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    let pos_hashes = compute_suffix_hashes_debug(regex, slice, all_targets);

    let mut signatures: Vec<u64> = Vec::with_capacity(initial_states.len());
    for (end_state, edges) in pos0_results.iter() {
        let completion = end_state.map(|id| regex.dfa().states[id].possible_future_group_ids.clone());
        let mut hasher = DefaultHasher::new();
        completion.hash(&mut hasher);
        for (group_id, target) in edges.iter() {
            let target_hash = *pos_hashes.get(*target).unwrap_or(&0);
            (group_id, target_hash).hash(&mut hasher);
        }
        signatures.push(hasher.finish());
    }

    signatures
}

pub fn debug_pos0_edges(
    regex: &Tokenizer,
    slice: &[u8],
    initial_states: &[usize],
) -> Vec<EdgeList> {
    let pre = precompute_dfa(regex);
    let mut scratch = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let (pos0_results, _) = compute_pos0_results(&pre, &mut scratch, slice, initial_states);
    pos0_results.iter().map(|(_, edges)| edges.clone()).collect()
}

pub fn compute_signature_actual(
    regex: &Tokenizer,
    slice: &[u8],
    initial_states: &[usize],
) -> u64 {
    let pre = precompute_dfa(regex);
    let mut pos0 = Pos0Scratch::new(initial_states.len(), pre.num_groups);
    let mut suffix_scratch = SuffixScratch::new(pre.num_groups);
    let mut cache = vec![None; slice.len() + 1];

    compute_chunk_signature(&pre, slice, initial_states, &mut pos0, &mut suffix_scratch, &mut cache, None, None, None, false)
}
