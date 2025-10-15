// src/constraint_precompute3_challenge_elimination.rs
use crate::constraint::{IntermediateTrie3EdgeKey, StateIDBV};

/// Eliminates adjacent Push/Pop pairs from a stack of intermediate trie edge keys.
/// This is a core part of simplifying the precompute3 graph.
///
/// The logic is as follows:
/// - A `Push` looks for the nearest `Pop` to its right.
/// - If another `Push` is encountered first, the search stops for this `Push`.
/// - If a `Pop(n, pop_states)` is found:
///   - If `n == 0` (a state check):
///     - If the `push_states` and `pop_states` intersect, both operations are removed (they cancel).
///     - If they do not intersect, the entire stack is invalid (`None` is returned).
///   - If `n > 0`:
///     - The `Push` is removed.
///     - The `Pop` is decremented to `Pop(n - 1, ...)`. If `n` becomes 1, the `Pop` is also removed.
/// - This process repeats until no more cancellations can be made.
pub fn eliminate_pushes_and_pops(
    stack: Vec<IntermediateTrie3EdgeKey>,
) -> Option<Vec<IntermediateTrie3EdgeKey>> {
    let mut stack = stack;
    loop {
        let mut changed_in_pass = false;
        let mut i = 0;
        while i < stack.len() {
            if let IntermediateTrie3EdgeKey::Push(push_states) = &stack[i] {
                let push_states = push_states.clone();
                // Find nearest pop to the right, not blocked by another push
                let mut pop_j = None;
                for j in (i + 1)..stack.len() {
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Push(_)) {
                        break; // Blocked
                    }
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Pop(_, _)) {
                        pop_j = Some(j);
                        break;
                    }
                }

                if let Some(j) = pop_j {
                    // Found a pair to cancel
                    let pop_op = stack.remove(j);
                    let _push_op = stack.remove(i); // push is at i

                    if let IntermediateTrie3EdgeKey::Pop(n, pop_states) = pop_op {
                        if n == 0 { // State check
                            if push_states.is_disjoint(&pop_states) {
                                return None; // Mismatch
                            }
                            // Match: both are removed.
                        } else { // n > 0
                            // Push is removed. Pop is decremented.
                            if n > 1 {
                                stack.insert(i, IntermediateTrie3EdgeKey::Pop(n - 1, pop_states));
                            }
                        }
                    }
                    changed_in_pass = true;
                    // Restart scan from beginning of modified stack
                    i = 0;
                    continue;
                }
            }
            i += 1;
        }
        if !changed_in_pass {
            break;
        }
    }
    Some(stack)
}
