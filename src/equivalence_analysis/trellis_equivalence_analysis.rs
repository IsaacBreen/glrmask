//! Trellis-based Equivalence Analysis (Very Slow Ground Truth)
//!
//! This is a deliberately slow but DEFINITELY correct implementation
//! used only for testing. It uses `generate_token_trellis_with_completion`
//! to get the full structural parse DAG for each token/state combination,
//! then hashes these to determine equivalence.
//!
//! This is O(tokens × states × token_length × trellis_size) and should
//! only be used when the problem size is small enough (< 1M operations).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::finite_automata::{Regex, TokenTrellisWithCompletion, GroupID, Trellis};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;
pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;

/// Hash a trellis structure recursively.
/// This captures the full structural information including:
/// - End state (possible_future_group_ids)
/// - All edges (group_id -> child trellis hash)
fn hash_trellis(trellis: &TokenTrellisWithCompletion) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_trellis_recursive(trellis, &mut hasher);
    hasher.finish()
}

fn hash_trellis_recursive<H: Hasher>(trellis: &Trellis<BTreeSet<GroupID>>, hasher: &mut H) {
    // Hash the end state (None vs Some(set of group IDs))
    match &trellis.end_state {
        None => {
            0u8.hash(hasher);
        }
        Some(groups) => {
            1u8.hash(hasher);
            groups.len().hash(hasher);
            for gid in groups {
                gid.hash(hasher);
            }
        }
    }
    
    // Hash edges in order (BTreeMap is ordered)
    trellis.edges.len().hash(hasher);
    for (group_id, child) in &trellis.edges {
        group_id.hash(hasher);
        // Recursively hash child trellis
        hash_trellis_recursive(child, hasher);
    }
}

/// Find vocab equivalence classes using trellis hashing.
///
/// For each token, compute a combined hash of its trellis structure
/// across all initial states. Tokens with the same hash are equivalent.
///
/// This is the SLOWEST but most correct implementation.
pub fn find_vocab_equivalence_classes_trellis(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_trellis_with_follow(regex, tokens, initial_states, None)
}

/// Find vocab equivalence classes using trellis hashing, with optional follow-set pruning.
///
/// If `ever_allowed_by_group` is provided, uses the FULL trellis (with intermediate
/// group match positions) and applies NWA-style disallowed-follow pruning:
/// edges whose group is not in `ever_allowed[parent_group]` are removed.
/// This mirrors how `prune_nwa_disallowed_follows` works on the NWA graph.
pub fn find_vocab_equivalence_classes_trellis_with_follow(
    regex: &Regex,
    tokens: &[Vec<u8>],
    initial_states: &[usize],
    ever_allowed_by_group: Option<&[BTreeSet<GroupID>]>,
) -> VocabEquivalenceResult {
    let progress_every = std::env::var("TRELLIS_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let progress_counter = progress_every.map(|_| AtomicUsize::new(0));
    let progress_start = std::time::Instant::now();

    // Compute signature for each token
    let mut token_signatures: Vec<u64> = Vec::with_capacity(tokens.len());

    for token in tokens {
        // Combine trellis hashes across all initial states
        let mut combined_hasher = DefaultHasher::new();

        for &state in initial_states {
            let trellis_to_hash = if let Some(ea) = ever_allowed_by_group {
                // Use FULL trellis (with intermediate positions) for follow pruning
                let trellis = regex.generate_token_trellis_with_completion_full(token, state);
                prune_trellis_disallowed_follows(&trellis, ea)
            } else {
                regex.generate_token_trellis_with_completion(token, state)
            };
            let trellis_hash = hash_trellis(&trellis_to_hash);
            trellis_hash.hash(&mut combined_hasher);
        }
        
        token_signatures.push(combined_hasher.finish());

        if let (Some(every), Some(counter)) = (progress_every, progress_counter.as_ref()) {
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if done % every == 0 || done == tokens.len() {
                let elapsed = progress_start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed.max(1e-9);
                let remaining = tokens.len().saturating_sub(done);
                let eta = if rate > 0.0 {
                    remaining as f64 / rate
                } else {
                    0.0
                };
                eprintln!(
                    "TRELLIS progress: {}/{} tokens ({:.1}%) elapsed={:.1}s rate={:.1} tok/s eta={:.1}s",
                    done,
                    tokens.len(),
                    100.0 * done as f64 / tokens.len().max(1) as f64,
                    elapsed,
                    rate,
                    eta,
                );
            }
        }
    }
    
    // Group tokens by signature
    let mut sig_to_tokens: HashMap<u64, Vec<usize>> = HashMap::new();
    for (token_idx, sig) in token_signatures.into_iter().enumerate() {
        sig_to_tokens.entry(sig).or_default().push(token_idx);
    }
    
    // Collect as result
    sig_to_tokens.into_values().collect()
}

/// Find state equivalence classes using trellis hashing.
///
/// For each state, compute a combined hash of trellis structures
/// across all tokens. States with the same hash are equivalent.
///
/// This is the SLOWEST but most correct implementation.
pub fn find_state_equivalence_classes_trellis(
    regex: &Regex,
    tokens: &[Vec<u8>],
    states: &[usize],
) -> Vec<usize> {
    // Compute signature for each state
    let mut state_signatures: Vec<u64> = Vec::with_capacity(states.len());
    
    for &state in states {
        // Combine trellis hashes across all tokens
        let mut combined_hasher = DefaultHasher::new();
        
        for token in tokens {
            let trellis = regex.generate_token_trellis_with_completion(token, state);
            let trellis_hash = hash_trellis(&trellis);
            trellis_hash.hash(&mut combined_hasher);
        }
        
        state_signatures.push(combined_hasher.finish());
    }
    
    // Group states by signature and get representatives
    let mut sig_to_rep: HashMap<u64, usize> = HashMap::new();
    let mut mapping: Vec<usize> = Vec::with_capacity(states.len());
    
    for (idx, sig) in state_signatures.into_iter().enumerate() {
        let rep = *sig_to_rep.entry(sig).or_insert(states[idx]);
        mapping.push(rep);
    }
    
    mapping
}

/// Convert a state-to-representative mapping to StateEquivalenceResult format.
pub fn mapping_to_equivalence_classes(states: &[usize], mapping: &[usize]) -> StateEquivalenceResult {
    let mut rep_to_class: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    
    for (i, &rep) in mapping.iter().enumerate() {
        rep_to_class.entry(rep).or_default().insert(states[i]);
    }
    
    rep_to_class.into_values().collect()
}

/// Prune a trellis based on disallowed-follows relationships.
///
/// For each edge labeled with group G, if G is not in `ever_allowed[parent_group]`,
/// remove that edge. At the root level (no parent), all groups are allowed.
///
/// `ever_allowed_by_group`: for each group_id G, the set of group_ids that can
/// follow G in the grammar. If a group doesn't have an entry (or is out of bounds),
/// it is treated as allowing all followers.
///
/// Returns a new trellis with disallowed edges removed.
pub fn prune_trellis_disallowed_follows(
    trellis: &TokenTrellisWithCompletion,
    ever_allowed_by_group: &[BTreeSet<GroupID>],
) -> TokenTrellisWithCompletion {
    // At root, no parent group -> all edges are allowed
    // Recurse into children with context of the parent edge's group
    let mut new_edges = BTreeMap::new();
    for (&gid, child) in &trellis.edges {
        let pruned_child = prune_trellis_recursive(child, gid, ever_allowed_by_group);
        new_edges.insert(gid, Arc::new(pruned_child));
    }
    Trellis {
        end_state: trellis.end_state.clone(),
        edges: new_edges,
    }
}

fn prune_trellis_recursive(
    trellis: &Trellis<BTreeSet<GroupID>>,
    parent_group: GroupID,
    ever_allowed_by_group: &[BTreeSet<GroupID>],
) -> Trellis<BTreeSet<GroupID>> {
    let allowed = if (parent_group as usize) < ever_allowed_by_group.len() {
        Some(&ever_allowed_by_group[parent_group as usize])
    } else {
        None // No follow info for this group -> allow everything
    };

    let mut new_edges = BTreeMap::new();
    for (&gid, child) in &trellis.edges {
        // Check if gid is allowed after parent_group
        let is_allowed = match allowed {
            Some(set) => set.contains(&gid),
            None => true,
        };
        if is_allowed {
            let pruned_child = prune_trellis_recursive(child, gid, ever_allowed_by_group);
            new_edges.insert(gid, Arc::new(pruned_child));
        }
    }

    Trellis {
        end_state: trellis.end_state.clone(),
        edges: new_edges,
    }
}