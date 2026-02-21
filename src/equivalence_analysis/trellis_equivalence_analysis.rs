//! Trellis-based Equivalence Analysis (Very Slow Ground Truth)
//!
//! This is a deliberately slow but DEFINITELY correct implementation
//! used only for testing. It uses `generate_token_trellis_with_completion`
//! to get the full structural parse DAG for each token/state combination,
//! then hashes these to determine equivalence.
//!
//! This is O(tokens × states × token_length × trellis_size) and should
//! only be used when the problem size is small enough (< 1M operations).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::Arc;

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

fn generate_trellis_full_token_only(
    regex: &Regex,
    bytes: &[u8],
    start_state: usize,
) -> TokenTrellisWithCompletion {
    let mut flat_trellis: BTreeMap<usize, (Option<BTreeSet<GroupID>>, Vec<(GroupID, usize)>)> = BTreeMap::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();

    queue.push_back(0usize);
    visited.insert(0usize);
    flat_trellis.insert(0usize, (None, Vec::new()));

    while let Some(pos) = queue.pop_front() {
        if pos > bytes.len() {
            continue;
        }

        let exec_start = if pos == 0 {
            start_state
        } else {
            regex.dfa.start_state
        };
        let result = regex.execute_from_state_nonzero(&bytes[pos..], exec_start);

        let mut edges = Vec::new();
        for m in result.matches {
            let target_pos = pos + m.position;
            if target_pos != bytes.len() || target_pos <= pos {
                continue;
            }

            edges.push((m.group_id, target_pos));
            if visited.insert(target_pos) {
                queue.push_back(target_pos);
                flat_trellis.insert(target_pos, (None, Vec::new()));
            }
        }

        let completion = result
            .end_state
            .map(|idx| regex.dfa.states[idx].possible_future_group_ids.clone());
        if let Some(node) = flat_trellis.get_mut(&pos) {
            node.0 = completion;
            node.1 = edges;
        } else {
            flat_trellis.insert(pos, (completion, edges));
        }
    }

    let mut memo: HashMap<usize, Arc<TokenTrellisWithCompletion>> = HashMap::new();
    for (pos, (end_state, edges_list)) in flat_trellis.into_iter().rev() {
        let mut edges = BTreeMap::new();
        for (gid, target_pos) in edges_list {
            if let Some(target_node) = memo.get(&target_pos) {
                if edges.insert(gid, target_node.clone()).is_some() {
                    panic!("Duplicate edge for group ID {} at position {}", gid, pos);
                }
            }
        }
        memo.insert(pos, Arc::new(Trellis { end_state, edges }));
    }

    memo.get(&0)
        .map(|root| root.as_ref().clone())
        .unwrap_or_else(|| Trellis {
            end_state: None,
            edges: BTreeMap::new(),
        })
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
    // Compute signature for each token
    let mut token_signatures: Vec<u64> = Vec::with_capacity(tokens.len());
    
    for token in tokens {
        // Combine trellis hashes across all initial states
        let mut combined_hasher = DefaultHasher::new();
        
        for &state in initial_states {
            let trellis = generate_trellis_full_token_only(regex, token, state);
            let trellis_hash = hash_trellis(&trellis);
            trellis_hash.hash(&mut combined_hasher);
        }
        
        token_signatures.push(combined_hasher.finish());
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
            let trellis = generate_trellis_full_token_only(regex, token, state);
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
