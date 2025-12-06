use crate::finite_automata::Regex;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

/// Per‑position information: what the regex can still match
/// starting from this position, and to which later positions each
/// capturing group can jump.
struct NodeData<'a> {
    /// `None` if the DFA run cannot stop here, otherwise a pointer to
    /// the DFA state's `possible_future_group_ids` vector.
    ///
    /// This is intentionally stored as `&Vec<usize>` so that hashing
    /// uses the same pointer‑based semantics as in the original code.
    completion: Option<&'a BTreeSet<usize>>,
    /// Outgoing edges: (group_id, absolute_target_position_in_string).
    edges: Vec<(usize, usize)>,
}

/// Evaluate the DFA from `regex.dfa.start_state` on `slice[pos..]` and
/// materialize the resulting node description.
///
/// This is the analogue of a single call to the expensive function `f`
/// in the theoretical discussion.
#[inline]
fn build_node<'a>(regex: &'a Regex, slice: &[u8], pos: usize) -> NodeData<'a> {
    debug_assert!(pos <= slice.len());

    let result = regex.execute_from_state_nonzero(&slice[pos..], regex.dfa.start_state);

    let mut edges = Vec::with_capacity(result.matches.len());
    for m in result.matches {
        let target = pos + m.position;
        debug_assert!(target <= slice.len());
        // All edges go strictly forward in the string, so target > pos.
        edges.push((m.group_id, target));
    }

    // Ensure deterministic hash regardless of internal match discovery order.
    edges.sort_unstable_by_key(|e| e.0);

    let completion = result
        .end_state
        .map(|id| &regex.dfa.states[id].possible_future_group_ids);

    NodeData { completion, edges }
}

/// Compute a combined structural signature of `slice` for *all*
/// `initial_states` at once, re‑using work between states.
///
/// This is functionally equivalent to:
///
/// ```ignore
/// let mut hasher = DefaultHasher::new();
/// for &state in initial_states {
///     compute_signature(regex, slice, state).hash(&mut hasher);
/// }
/// hasher.finish()
/// ```
///
/// where `compute_signature` is your original per‑state function.
/// The main differences are:
///
/// * Every position `pos > 0` in the string is evaluated at most once
///   (one call to `execute_from_state_nonzero(&slice[pos..], dfa.start_state)`),
///   no matter how many `initial_states` you use.
/// * We avoid `HashMap`/`HashSet` overhead and use simple `Vec`s.
/// * The hashing order and ingredients are kept identical.
fn compute_combined_signature<'a>(
    regex: &'a Regex,
    slice: &[u8],
    initial_states: &[usize],
) -> u64 {
    let len = slice.len();

    // Per‑position cached node data for positions in 0..=len.
    // Entry 0 is never used here (roots are handled per start state).
    let mut nodes: Vec<Option<NodeData<'a>>> = std::iter::repeat_with(|| None).take(len + 1).collect();
    let mut seen: Vec<bool> = vec![false; len + 1];

    // Work queue for a simple BFS over positions > 0.
    // Implemented as a Vec with a head index instead of VecDeque
    // to keep it very cheap.
    let mut queue: Vec<usize> = Vec::new();
    let mut q_head: usize = 0;

    // Information about the root (position 0) for each requested start state.
    struct RootInfo<'a> {
        completion: Option<&'a BTreeSet<usize>>,
        edges: Vec<(usize, usize)>,
    }
    let mut roots: Vec<RootInfo<'a>> = Vec::with_capacity(initial_states.len());

    // 1. For each initial start state, evaluate position 0 and seed the BFS
    //    with all newly discovered target positions.
    for &start_state in initial_states {
        let result = regex.execute_from_state_nonzero(slice, start_state);

        let mut edges = Vec::with_capacity(result.matches.len());
        for m in result.matches {
            let target = m.position; // pos == 0 here
            debug_assert!(target <= len);
            edges.push((m.group_id, target));

            // For positions > 0 we share work between all start states.
            if target > 0 && !seen[target] {
                seen[target] = true;
                nodes[target] = Some(build_node(regex, slice, target));
                queue.push(target);
            }
        }

        // Deterministic ordering of outgoing edges.
        edges.sort_unstable_by_key(|e| e.0);

        let completion = result
            .end_state
            .map(|id| &regex.dfa.states[id].possible_future_group_ids);

        roots.push(RootInfo { completion, edges });
    }

    // 2. BFS over all positions reachable from any root under capturing‑group
    //    transitions, computing each node at most once.
    while q_head < queue.len() {
        let pos = queue[q_head];
        q_head += 1;

        let (head, tail) = nodes.split_at_mut(pos + 1);
        let node = head[pos].as_ref().expect("node must be present");

        for &(_, target) in &node.edges {
            // Edges always go strictly forward (target > pos).
            debug_assert!(target > pos && target <= len);
            if !seen[target] {
                seen[target] = true;
                // `target` is an index into the original `nodes` slice.
                // `tail` starts at index `pos + 1`, so we adjust the index.
                tail[target - (pos + 1)] = Some(build_node(regex, slice, target));
                queue.push(target);
            }
        }
    }

    // 3. Backward pass: compute a hash for every visited position > 0.
    //
    //    Edges always go strictly forward in the string (target > source),
    //    so processing positions in descending order guarantees that a
    //    target's hash is available before its source is handled.
    let mut node_hashes = vec![0u64; len + 1];

    for pos in (1..=len).rev() {
        if !seen[pos] {
            continue;
        }

        let node = nodes[pos].as_ref().unwrap();
        let mut hasher = DefaultHasher::new();

        // Hash local completion info (pointer to the DFA's
        // `possible_future_group_ids` vector or None). This uses the same
        // semantics as the original code: `&Vec` hashes by address,
        // not by contents.
        node.completion.hash(&mut hasher);

        // Hash the structural edges plus each target's hash.
        for &(group_id, target) in &node.edges {
            let target_hash = node_hashes[target];
            (group_id, target_hash).hash(&mut hasher);
        }

        node_hashes[pos] = hasher.finish();
    }

    // 4. Compute the per‑start‑state root signatures (position 0),
    //    then fold them into one combined signature exactly as the
    //    original `find_equivalence_classes` did.
    let mut combined = DefaultHasher::new();

    for root in roots {
        let mut hasher = DefaultHasher::new();

        root.completion.hash(&mut hasher);

        for (group_id, target) in root.edges {
            // In your original code, a target of 0 here would have created
            // a self‑loop at position 0 and broken the DAG assumption;
            // that is implicitly disallowed by relying on acyclicity.
            //
            // We mirror that assumption. node_hashes[0] stays 0 and will be
            // used only if such a match were ever present (which would have
            // panicked before).
            let target_hash = node_hashes[target];
            (group_id, target_hash).hash(&mut hasher);
        }

        let sig = hasher.finish();
        sig.hash(&mut combined);
    }

    combined.finish()
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Compute one combined signature per string in parallel. For each
    // string we evaluate every position > 0 at most once, even if there
    // are many different `initial_states`.
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| compute_combined_signature(regex, s, initial_states))
        .collect();

    // Group indices by signature.  Instead of a hash map we sort the
    // (signature, index) pairs, which is typically faster and more
    // cache‑friendly for large batches than a hash map, and it
    // preserves the same semantics.
    let mut pairs: Vec<(u64, usize)> = signatures
        .into_iter()
        .enumerate()
        .map(|(idx, sig)| (sig, idx))
        .collect();

    // Sort by (signature, index) so indices inside each group are
    // always in ascending order. This matches (and makes explicit)
    // the behaviour of the original implementation, which pushed
    // indices in increasing order into each group.
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut result: EquivalenceResult = BTreeSet::new();

    let mut current_group: Vec<usize> = Vec::new();
    let mut current_sig: Option<u64> = None;

    for (sig, idx) in pairs {
        match current_sig {
            Some(s) if s == sig => {
                current_group.push(idx);
            }
            _ => {
                if !current_group.is_empty() {
                    result.insert(current_group);
                }
                current_group = vec![idx];
                current_sig = Some(sig);
            }
        }
    }

    if !current_group.is_empty() {
        result.insert(current_group);
    }

    result
}