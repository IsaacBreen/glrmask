// src/constraint_precompute3_challenge_elim.rs
use crate::constraint::{IntermediateTrie3EdgeKey, StateIDBV};

/// Eliminates push/pop pairs from a stack of intermediate trie3 edge keys.
///
/// This function implements a "challenge" system where `Push` operations look to their
/// right for a `Pop` operation to cancel with. The process repeats until no more
/// cancellations can be made.
///
/// Returns `Some(new_stack)` if successful, or `None` if a state mismatch occurs
/// during a `Pop { k: 0, ... }` cancellation, which invalidates the entire stack.
pub fn stack_challenge_elimination(
    mut stack: Vec<IntermediateTrie3EdgeKey>,
) -> Option<Vec<IntermediateTrie3EdgeKey>> {
    loop {
        let mut changed = false;
        let mut i = 0;
        while i < stack.len() {
            let push_states_clone = if let IntermediateTrie3EdgeKey::Push { states } = &stack[i] {
                Some(states.clone())
            } else {
                None
            };

            if let Some(push_states) = push_states_clone {
                // This is a Push. Look for a Pop to its right.
                let mut found_pop_idx = None;
                for j in (i + 1)..stack.len() {
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Push { .. }) {
                        // Blocked by another Push. Stop searching from this `i`.
                        break;
                    }
                    if matches!(stack[j], IntermediateTrie3EdgeKey::Pop { .. }) {
                        found_pop_idx = Some(j);
                        break;
                    }
                }

                if let Some(j) = found_pop_idx {
                    // We found a pair: Push at `i` and Pop at `j`.
                    changed = true;
                    let pop_op = stack.remove(j);
                    let _push_op = stack.remove(i); // `i` is now the index of the push op

                    if let IntermediateTrie3EdgeKey::Pop { k, states: pop_states } = pop_op {
                        if k == 0 {
                            // State check pop.
                            if push_states.is_disjoint(&pop_states) {
                                // Mismatch. The entire stack is invalid.
                                return None;
                            }
                            // Match. The Pop is consumed, the Push is consumed. Nothing is inserted back.
                        } else {
                            // A pop with k > 0. The Push is consumed.
                            // The Pop is re-inserted with a decremented `k`.
                            if k > 1 {
                                stack.insert(
                                    i,
                                    IntermediateTrie3EdgeKey::Pop {
                                        k: k - 1,
                                        states: pop_states,
                                    },
                                );
                            }
                            // If k was 1, it's now 0 and is fully consumed.
                        }
                    }
                    // After removing two elements and possibly inserting one,
                    // we should restart the scan to handle new adjacencies.
                    // The `continue` will restart the outer `while i < stack.len()` loop with i=0.
                    continue;
                }
            }
            i += 1;
        }

        if !changed {
            break; // Fixpoint reached.
        }
    }

    Some(stack)
}
