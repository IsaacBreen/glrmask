use crate::finite_automata::{GroupID, Regex};
use hashbrown::HashMap;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

/// Lightweight representation of a node in the token trellis, indexed by
/// byte position in the input.
struct FlatNode {
    /// DFA state index at the end of executing from this position, if any.
    end_state: Option<usize>,
    /// Outgoing edges: (group_id, target_position).
    ///
    /// The vector is kept in ascending `group_id` order, matching the
    /// behaviour of the original BTreeMap-based representation.
    edges: Vec<(GroupID, usize)>,
}

/// Build a flat trellis representation for a single input string and
/// initial DFA state, equivalent to the data used by
/// `Regex::generate_token_trellis_with_completion`, but using simple
/// Vec-based structures instead of a BTreeMap + Arc graph.
///
/// This is logically equivalent to `Regex::generate_flat_trellis`, but
/// implemented here so we can hash the structure directly without ever
/// constructing the full `Trellis` graph.
fn build_flat_trellis(regex: &Regex, bytes: &[u8], start_state: usize) -> Vec<Option<FlatNode>> {
    let len = bytes.len();
    let mut nodes: Vec<Option<FlatNode>> = std::iter::repeat_with(|| None).take(len + 1).collect();
    let mut visited: Vec<bool> = vec![false; len + 1];
    let mut queue: VecDeque<usize> = VecDeque::new();

    visited[0] = true;
    queue.push_back(0);

    while let Some(pos) = queue.pop_front() {
        let slice = if pos <= len {
            &bytes[pos..]
        } else {
            &[]
        };

        let exec_start = if pos == 0 {
            start_state
        } else {
            regex.dfa.start_state
        };

        let result = regex.execute_from_state_nonzero(slice, exec_start);

        let mut edges: Vec<(GroupID, usize)> = Vec::new();
        for m in result.matches {
            let target_pos = pos + m.position;
            debug_assert!(
                target_pos <= len,
                "target_pos {} out of range for input len {}",
                target_pos,
                len
            );
            if target_pos > len {
                // Should not happen given ExecutionResult semantics, but guard just in case.
                continue;
            }

            edges.push((m.group_id, target_pos));

            if !visited[target_pos] {
                visited[target_pos] = true;
                queue.push_back(target_pos);
            }
        }

        // `result.matches` is built from a BTreeMap in RegexState, so it is
        // already sorted by group_id. Keeping that order is important to
        // match the hashing behaviour of the original BTreeMap-based trellis.
        nodes[pos] = Some(FlatNode {
            end_state: result.end_state,
            edges,
        });
    }

    nodes
}

/// Recursively hash the logical trellis structure starting at a given
/// position, emulating the derived `Hash` implementation of
/// `Trellis<BTreeSet<GroupID>>` without constructing that structure.
///
/// This function is designed to be *hash-equivalent* to:
///
/// ```ignore
/// let trellis = regex.generate_token_trellis_with_completion(bytes, start_state);
/// trellis.hash(hasher);
/// ```
///
/// but significantly faster and more memory efficient.
fn hash_trellis_from_pos<H: Hasher>(
    regex: &Regex,
    nodes: &[Option<FlatNode>],
    pos: usize,
    state: &mut H,
) {
    let node = nodes
        .get(pos)
        .and_then(|n| n.as_ref())
        .expect("Missing node for position in trellis hashing");

    // end_state: Option<BTreeSet<GroupID>>
    //
    // In the original implementation, `end_state` is mapped to a
    // BTreeSet via:
    //
    //     end_state.map(|idx| self.dfa.states[idx].possible_future_group_ids.clone())
    //
    // and then hashed. Here we avoid the clone by using
    // Option<&BTreeSet<_>>, which hashes identically to
    // Option<BTreeSet<_>> because &T implements Hash by delegating
    // to T's Hash implementation.
    let end_state_sets: Option<&BTreeSet<GroupID>> = node
        .end_state
        .map(|idx| &regex.dfa.states[idx].possible_future_group_ids);

    end_state_sets.hash(state);

    // edges: logically a BTreeMap<GroupID, Arc<Trellis<_>>>.
    //
    // In the original code, edges are stored in a BTreeMap and then
    // hashed. BTreeMap iteration is in key order, and
    // `result.matches` (from which `edges` is built) is already sorted
    // by group_id because it comes from a BTreeMap. Our `edges` vec is
    // therefore in exactly the same key order.
    //
    // To emulate BTreeMap's Hash, we:
    //
    //   - Hash the number of entries
    //   - For each (key, value) pair in order, hash the key then
    //     recursively hash the child node.
    //
    // This mirrors the behaviour of deriving Hash for the corresponding
    // BTreeMap structure.
    node.edges.len().hash(state);
    for &(group_id, target_pos) in &node.edges {
        group_id.hash(state);
        hash_trellis_from_pos(regex, nodes, target_pos, state);
    }
}

/// Compute a structural hash for a given string and starting DFA state,
/// matching the behaviour of hashing the full completion trellis:
///
/// ```ignore
/// let trellis = regex.generate_token_trellis_with_completion(slice, start_state);
/// trellis.hash(hasher);
/// ```
fn compute_structural_hash(
    regex: &Regex,
    slice: &[u8],
    start_state: usize,
    hasher: &mut DefaultHasher,
) {
    let nodes = build_flat_trellis(regex, slice, start_state);
    // Root of the trellis is always at position 0.
    hash_trellis_from_pos(regex, &nodes, 0, hasher);
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    let signatures: Vec<u64> = strings
        .par_iter()
        .enumerate()
        .map(|(i, s)| {
            if i % 100 == 0 {
                println!(
                    "Computing equivalence signatures: processing string {}/{}",
                    i,
                    strings.len()
                );
            }
            let mut h = DefaultHasher::new();
            for &start in initial_states.iter() {
                compute_structural_hash(regex, s, start, &mut h);
            }
            h.finish()
        })
        .collect();

    let mut groups = HashMap::new();
    for (idx, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(idx);
    }

    groups.into_values().collect()
}