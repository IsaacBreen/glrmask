// PERMANENT WARNING: Do NOT add any form of caching or shortcuts that skip or restrict
// states/tokens for equivalence analysis. Full correctness is mandatory; no "cheating"
// optimizations that drop work are allowed here.

//! Fast equivalence analysis using trie-based DFS traversal.
//!
//! This module implements equivalence class computation for tokens based on their
//! behavior with respect to a tokenizer DFA. The algorithm mirrors the structure of
//! `constraint_precompute.rs` but instead of building an NWA, it computes hash
//! signatures that identify equivalent tokens.
//!
//! The key insight is that two tokens are equivalent if and only if they produce
//! the same "behavior" from every possible tokenizer state. The behavior is captured
//! as a hash that encodes:
//! - Which grammar terminals match, and at what positions
//! - What tokenizer state we end up in (or if we terminate)
//! - The completion potential (what terminals could match in the future)

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::BitOrAssign;

use ahash::{AHasher, RandomState};
use range_set_blaze::RangeSetBlaze;

use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;
use crate::types::TerminalID as GrammarTokenID;

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

// =============================================================================
// HASH UTILITIES
// =============================================================================

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;

#[inline]
fn new_hasher() -> AHasher {
    RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4).build_hasher()
}

// =============================================================================
// POSSIBLE MATCHES CACHE (handles greedy matching)
// =============================================================================

/// Cache for possible matches at each (vocab_node, tokenizer_state) pair.
/// This is key to correctly handling greedy matching behavior.
struct PossibleMatchesCache<'r> {
    tokenizer: &'r Regex,
    cache: HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, RangeSetBlaze<usize>>>,
}

impl<'r> PossibleMatchesCache<'r> {
    fn new(tokenizer: &'r Regex) -> Self {
        PossibleMatchesCache {
            tokenizer,
            cache: HashMap::new(),
        }
    }

    /// Compute which grammar terminals could match descendant tokens when starting
    /// from the given tokenizer state at the given vocab node position.
    fn possible_matches(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
    ) -> BTreeMap<GrammarTokenID, RangeSetBlaze<usize>> {
        let cache_key = (vocab_node as *const VocabPrefixTreeNode, tokenizer_state_id);
        
        if let Some(cached) = self.cache.get(&cache_key) {
            return cached.clone();
        }

        let mut result_map: BTreeMap<GrammarTokenID, RangeSetBlaze<usize>> = BTreeMap::new();

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                result_map
                    .entry(grammar_token_id)
                    .or_insert_with(RangeSetBlaze::new)
                    .bitor_assign(applicable_tokens);
            }
            
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_state: BTreeSet<_> = self
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(final_state_val))
                    .into_iter()
                    .collect();
                let matches_here: BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                let possible_new_matches = &matches_possible_from_state - &matches_here;
                
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(
                        child_vocab_node,
                        TokenizerStateID(final_state_val),
                    );
                    for (token, bv) in next_results {
                        result_map
                            .entry(token)
                            .or_insert_with(RangeSetBlaze::new)
                            .bitor_assign(&bv);
                    }
                }
            }
        }

        self.cache.insert(cache_key, result_map.clone());
        result_map
    }
}

// =============================================================================
// EQUIVALENCE SIGNATURE COMPUTATION
// =============================================================================

/// Represents the accumulated behavior signature for a token from a specific initial state.
/// This captures everything needed to determine equivalence.
#[derive(Clone, Default)]
struct StateBehavior {
    /// Hash encoding the behavior from this initial state
    hash: u64,
}

impl StateBehavior {
    fn new() -> Self {
        StateBehavior { hash: 0 }
    }

    /// Record a match at the given position for the given terminal
    fn record_match(&mut self, terminal_id: usize, position: usize, is_final: bool) {
        let mut h = new_hasher();
        h.write_u8(1); // Match marker
        h.write_u64(self.hash);
        h.write_usize(terminal_id);
        h.write_usize(position);
        h.write_u8(is_final as u8);
        self.hash = h.finish();
    }

    /// Record the final state or termination
    fn record_end_state(&mut self, end_state: Option<usize>, completion_hash: u64) {
        let mut h = new_hasher();
        h.write_u8(2); // End state marker
        h.write_u64(self.hash);
        if let Some(state) = end_state {
            h.write_u8(1);
            h.write_u64(completion_hash);
        } else {
            h.write_u8(0);
        }
        self.hash = h.finish();
    }
}

/// Per-initial-state tracking during DFS traversal
struct StateTracker {
    /// Current tokenizer state for this initial state
    current_tokenizer_state: TokenizerStateID,
    /// Accumulated behavior signature
    behavior: StateBehavior,
    /// Whether this state has terminated (no more transitions possible)
    done: bool,
}

// =============================================================================
// MAIN DFS ALGORITHM
// =============================================================================

/// Main entry point for equivalence class computation.
pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    use std::time::Instant;
    let total_start = Instant::now();
    
    let num_tokens = strings.len();
    let num_states = initial_states.len();
    
    if num_states == 0 || num_tokens == 0 {
        return BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    }

    // Build vocab prefix tree from token strings
    let tokens: Vec<(usize, Vec<u8>)> = strings
        .iter()
        .enumerate()
        .map(|(idx, bytes)| (idx, bytes.clone()))
        .collect();
    
    crate::debug!(3, "Building vocab prefix tree for {} tokens", num_tokens);
    let vocab_tree = VocabPrefixTree::build(&tokens);
    let tree_build_time = total_start.elapsed();
    
    crate::debug!(3, "Vocab tree built in {:?}", tree_build_time);

    // Compute completion hashes for all DFA states
    let completion_hashes = compute_completion_hashes(regex);
    
    // Initialize possible matches cache
    let mut pm_cache = PossibleMatchesCache::new(regex);

    // Token index -> accumulated signature hash (combined across all initial states)
    let mut token_signatures: Vec<u64> = vec![0; num_tokens];

    // Process the tree using DFS, mirroring constraint_precompute.rs structure
    crate::debug!(3, "Starting trie DFS with {} initial states", num_states);
    
    dfs_compute_signatures(
        regex,
        &vocab_tree.root,
        initial_states,
        &completion_hashes,
        &mut pm_cache,
        &mut token_signatures,
    );

    let dfs_time = total_start.elapsed() - tree_build_time;
    crate::debug!(3, "Trie DFS completed in {:?}", dfs_time);

    // Group tokens by their signature
    let mut sig_to_tokens: HashMap<u64, Vec<usize>> = HashMap::new();
    for (token_idx, &sig) in token_signatures.iter().enumerate() {
        sig_to_tokens
            .entry(sig)
            .or_insert_with(Vec::new)
            .push(token_idx);
    }

    crate::debug!(
        3,
        "Computed {} equivalence classes in {:?}",
        sig_to_tokens.len(),
        total_start.elapsed()
    );

    sig_to_tokens.into_values().collect()
}

/// Compute completion hashes for all DFA states.
/// Two states with the same completion hash have the same possible_future_group_ids.
fn compute_completion_hashes(regex: &Regex) -> Vec<u64> {
    let dfa = &regex.dfa;
    dfa.states
        .iter()
        .map(|state| {
            let mut h = new_hasher();
            for &gid in &state.possible_future_group_ids {
                h.write_usize(gid);
            }
            h.finish()
        })
        .collect()
}

/// DFS through the vocab trie, computing behavior signatures for all tokens.
fn dfs_compute_signatures(
    regex: &Regex,
    vocab_node: &VocabPrefixTreeNode,
    initial_states: &[usize],
    completion_hashes: &[u64],
    pm_cache: &mut PossibleMatchesCache,
    token_signatures: &mut [u64],
) {
    // Map: TokenizerStateID -> Vec<(initial_state_index, accumulated_behavior_hash)>
    let mut assoc: BTreeMap<TokenizerStateID, Vec<(usize, u64)>> = BTreeMap::new();
    
    // Initialize: all initial states start at their respective positions
    for (idx, &state) in initial_states.iter().enumerate() {
        assoc
            .entry(TokenizerStateID(state))
            .or_insert_with(Vec::new)
            .push((idx, 0));
    }

    dfs_inner(
        regex,
        vocab_node,
        assoc,
        completion_hashes,
        pm_cache,
        token_signatures,
    );
}

/// Inner DFS function - processes one level of the vocab trie.
fn dfs_inner(
    regex: &Regex,
    vocab_node: &VocabPrefixTreeNode,
    assoc_by_state: BTreeMap<TokenizerStateID, Vec<(usize, u64)>>,
    completion_hashes: &[u64],
    pm_cache: &mut PossibleMatchesCache,
    token_signatures: &mut [u64],
) {
    for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
        let child_token_id = child_vocab_node.token_id();
        let child_reachable = child_vocab_node.reachable_token_ids();

        // Track states that will continue to the next level
        let mut next_level_assoc: BTreeMap<TokenizerStateID, Vec<(usize, u64)>> = BTreeMap::new();

        // Collect contributions from each initial state in sorted order
        let mut contributions: BTreeMap<usize, u64> = BTreeMap::new();
        
        // Process all (tokenizer_state, initial_state_indices) pairs
        for (tokenizer_state_id, state_entries) in &assoc_by_state {
            // Execute the tokenizer on this segment
            let exec_result = regex.execute_from_state(&segment_bytes, *tokenizer_state_id);

            // Get possible matches at end for greedy handling
            let possible_matches_at_end = if let Some(end_val) = exec_result.end_state {
                let ts = TokenizerStateID(end_val);
                Some(pm_cache.possible_matches(child_vocab_node, ts))
            } else {
                None
            };

            // Process each initial state that was at this tokenizer state
            for &(initial_state_idx, current_hash) in state_entries {
                let mut h = new_hasher();
                h.write_u64(current_hash);

                // Record matches
                for match_info in &exec_result.matches {
                    let terminal_id = match_info.id;
                    let position = match_info.width;
                    let is_final = position == segment_bytes.len();
                    
                    // Handle greedy behavior: check if this match could be extended
                    let should_record = if is_final {
                        if let Some(ref pm) = possible_matches_at_end {
                            // Check if this terminal could match with a longer token
                            !pm.get(&GrammarTokenID(terminal_id))
                                .map(|bv| bv.contains(child_token_id))
                                .unwrap_or(false)
                        } else {
                            true
                        }
                    } else {
                        true
                    };

                    if should_record {
                        h.write_u8(1); // Match marker
                        h.write_usize(terminal_id);
                        h.write_usize(position);
                        h.write_u8(is_final as u8);
                    }
                }

                // Record end state or termination
                if let Some(end_state_val) = exec_result.end_state {
                    h.write_u8(2); // Continue marker
                    h.write_u64(completion_hashes[end_state_val]);
                    
                    // This state continues to the next level
                    let new_hash = h.finish();
                    next_level_assoc
                        .entry(TokenizerStateID(end_state_val))
                        .or_insert_with(Vec::new)
                        .push((initial_state_idx, new_hash));
                    
                    // Record contribution for this initial state
                    contributions.insert(initial_state_idx, new_hash);
                } else {
                    h.write_u8(3); // Terminate marker
                    let final_hash = h.finish();
                    contributions.insert(initial_state_idx, final_hash);
                }
            }
        }

        // Combine contributions from all initial states in sorted order (by initial_state_idx)
        // This matches the reference implementation's combination strategy
        let mut combined = new_hasher();
        for (initial_state_idx, contribution) in contributions.iter() {
            combined.write_u64(*contribution);
        }
        token_signatures[child_token_id] = combined.finish();

        // Recurse to children
        if !next_level_assoc.is_empty() {
            dfs_inner(
                regex,
                child_vocab_node,
                next_level_assoc,
                completion_hashes,
                pm_cache,
                token_signatures,
            );
        }
    }
}
