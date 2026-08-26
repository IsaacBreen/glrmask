use crate::automata::lexer::Lexer;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::{TokenizerExecResult, TokenizerStateSet};

use super::super::artifact::Constraint;


/// Reusable exact tokenizer execution scratch for the runtime hot path.
/// Capacities are allocated when a ConstraintState is created. If an unusual
/// epsilon frontier or match set exceeds them, execution declines before
/// mutating parser state and the existing general path remains authoritative.
#[derive(Debug)]
pub(crate) struct ReusableTokenizerExecScratch {
    pub states: Vec<u32>,
    next_states: Vec<u32>,
    pub matches: Vec<crate::automata::lexer::tokenizer::TokenizerMatch>,
}

impl Default for ReusableTokenizerExecScratch {
    fn default() -> Self {
        Self {
            states: Vec::with_capacity(64),
            next_states: Vec::with_capacity(64),
            matches: Vec::with_capacity(64),
        }
    }
}

/// Execute the epsilon tokenizer exactly using only preallocated scratch.
/// Returns false if a bounded scratch capacity would be exceeded.
pub(crate) fn execute_tokenizer_reusable(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
    scratch: &mut ReusableTokenizerExecScratch,
) -> bool {
    let cached_closures = constraint.tokenizer.cached_singleton_epsilon_closures();
    scratch.states.clear();
    scratch.matches.clear();

    let owned_start_closure;
    let start_closure: &[u32] = if let Some(closures) = cached_closures {
        let Some(closure) = closures.get(start_state as usize) else {
            return false;
        };
        closure
    } else if constraint.tokenizer.state_has_epsilon_transitions(start_state) {
        owned_start_closure = constraint.tokenizer.singleton_epsilon_closure(start_state);
        &owned_start_closure
    } else {
        owned_start_closure = Box::new([start_state]);
        &owned_start_closure
    };

    let mut byte_start = 0usize;
    if start_closure.len() > scratch.states.capacity() {
        if start_state != constraint.runtime_commit_initial_state() {
            return false;
        }
        let Some(&first_byte) = bytes.first() else {
            return false;
        };
        let frontiers = constraint.tokenizer.initial_byte_frontiers();
        let first = &frontiers[first_byte as usize];
        if first.len() > scratch.states.capacity() {
            return false;
        }
        scratch.states.extend(first.iter().copied());
        if scratch.states.is_empty() {
            return true;
        }
        for &state in &scratch.states {
            // Do not use `matched_terminals_slice` here. On compiler-created
            // tokenizers its first call materializes finalizer lists for every
            // lexer state; source-specialized schemas can have >1M states, so
            // that otherwise puts tens of milliseconds on the first commit.
            for terminal in constraint.tokenizer.matched_terminals_iter(state) {
                if scratch.matches.len() == scratch.matches.capacity() {
                    return false;
                }
                scratch.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: terminal,
                    width: 1,
                    end_state: state,
                });
            }
        }
        byte_start = 1;
    } else {
        scratch.states.extend_from_slice(start_closure);
    }

    for (index, &byte) in bytes.iter().enumerate().skip(byte_start) {
        scratch.next_states.clear();
        for &state in &scratch.states {
            let target = constraint.tokenizer_fast_transitions.transition(
                &constraint.tokenizer,
                state,
                byte,
            );
            if target == u32::MAX {
                continue;
            }
            if let Some(closures) = cached_closures {
                let Some(target_closure) = closures.get(target as usize) else {
                    return false;
                };
                if scratch.next_states.len() + target_closure.len()
                    > scratch.next_states.capacity()
                {
                    return false;
                }
                scratch.next_states.extend_from_slice(target_closure);
            } else if constraint.tokenizer.state_has_epsilon_transitions(target) {
                let target_closure = constraint.tokenizer.singleton_epsilon_closure(target);
                if scratch.next_states.len() + target_closure.len()
                    > scratch.next_states.capacity()
                {
                    return false;
                }
                scratch.next_states.extend_from_slice(&target_closure);
            } else if scratch.next_states.len() == scratch.next_states.capacity() {
                return false;
            } else {
                scratch.next_states.push(target);
            }
        }
        if scratch.next_states.is_empty() {
            scratch.states.clear();
            return true;
        }
        scratch.next_states.sort_unstable();
        scratch.next_states.dedup();
        std::mem::swap(&mut scratch.states, &mut scratch.next_states);

        let width = index + 1;
        for &state in &scratch.states {
            for terminal in constraint.tokenizer.matched_terminals_iter(state) {
                let prior_width = scratch
                    .matches
                    .iter()
                    .find(|matched| matched.id == terminal)
                    .map(|matched| matched.width);
                match prior_width {
                    Some(prior) if prior > width => continue,
                    Some(prior) if prior < width => {
                        scratch.matches.retain(|matched| matched.id != terminal);
                    }
                    Some(_) => continue,
                    _ => {}
                }
                if scratch.matches.len() == scratch.matches.capacity() {
                    return false;
                }
                scratch.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: terminal,
                    width,
                    end_state: state,
                });
            }
        }
    }
    true
}


/// Execute from the exact union of several tokenizer start states using the
/// same bounded scratch. Callers use this only when all starts carry one
/// identical parser language and their terminal-ID futures are pairwise
/// disjoint, so lane-local longest-match decisions cannot interfere.
pub(crate) fn execute_tokenizer_reusable_from_states(
    constraint: &Constraint,
    bytes: &[u8],
    start_states: &[u32],
    scratch: &mut ReusableTokenizerExecScratch,
) -> bool {
    if let [start_state] = start_states {
        return execute_tokenizer_reusable(constraint, bytes, *start_state, scratch);
    }
    if start_states.is_empty() {
        scratch.states.clear();
        scratch.matches.clear();
        return true;
    }
    let cached_closures = constraint.tokenizer.cached_singleton_epsilon_closures();
    scratch.states.clear();
    scratch.matches.clear();
    for &start_state in start_states {
        if let Some(closures) = cached_closures {
            let Some(closure) = closures.get(start_state as usize) else {
                return false;
            };
            if scratch.states.len() + closure.len() > scratch.states.capacity() {
                return false;
            }
            scratch.states.extend_from_slice(closure);
        } else if constraint.tokenizer.state_has_epsilon_transitions(start_state) {
            let closure = constraint.tokenizer.singleton_epsilon_closure(start_state);
            if scratch.states.len() + closure.len() > scratch.states.capacity() {
                return false;
            }
            scratch.states.extend_from_slice(&closure);
        } else if scratch.states.len() == scratch.states.capacity() {
            return false;
        } else {
            scratch.states.push(start_state);
        }
    }
    scratch.states.sort_unstable();
    scratch.states.dedup();

    for (index, &byte) in bytes.iter().enumerate() {
        scratch.next_states.clear();
        for &state in &scratch.states {
            let target = constraint.tokenizer_fast_transitions.transition(
                &constraint.tokenizer,
                state,
                byte,
            );
            if target == u32::MAX {
                continue;
            }
            if let Some(closures) = cached_closures {
                let Some(closure) = closures.get(target as usize) else {
                    return false;
                };
                if scratch.next_states.len() + closure.len() > scratch.next_states.capacity() {
                    return false;
                }
                scratch.next_states.extend_from_slice(closure);
            } else if constraint.tokenizer.state_has_epsilon_transitions(target) {
                let closure = constraint.tokenizer.singleton_epsilon_closure(target);
                if scratch.next_states.len() + closure.len() > scratch.next_states.capacity() {
                    return false;
                }
                scratch.next_states.extend_from_slice(&closure);
            } else if scratch.next_states.len() == scratch.next_states.capacity() {
                return false;
            } else {
                scratch.next_states.push(target);
            }
        }
        if scratch.next_states.is_empty() {
            scratch.states.clear();
            return true;
        }
        scratch.next_states.sort_unstable();
        scratch.next_states.dedup();
        std::mem::swap(&mut scratch.states, &mut scratch.next_states);
        let width = index + 1;
        for &state in &scratch.states {
            for terminal in constraint.tokenizer.matched_terminals_iter(state) {
                let prior_width = scratch
                    .matches
                    .iter()
                    .find(|matched| matched.id == terminal)
                    .map(|matched| matched.width);
                match prior_width {
                    Some(prior) if prior > width => continue,
                    Some(prior) if prior < width => {
                        scratch.matches.retain(|matched| matched.id != terminal);
                    }
                    Some(_) => continue,
                    _ => {}
                }
                if scratch.matches.len() == scratch.matches.capacity() {
                    return false;
                }
                scratch.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: terminal,
                    width,
                    end_state: state,
                });
            }
        }
    }
    true
}

pub(super) struct InitialCommitScan {
	pub exec_results: FxHashMap<u32, TokenizerExecResult>,
}

pub(super) fn execute_tokenizer_from_state_small(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
) -> TokenizerExecResult {
    let mut result = TokenizerExecResult {
        end_state: TokenizerStateSet::new(),
        matches: Vec::with_capacity(8),
    };
    execute_tokenizer_from_state_small_into(constraint, bytes, start_state, &mut result);
    result
}

/// Execute one tokenizer lane in the recursive leaf coordinate while keeping
/// the current outer terminal IDs as a temporary commit-facing facade. Byte
/// transitions, epsilon closure, longest-match choice, and continuation states
/// all come exclusively from the intact leaf tokenizer; the outer union
/// tokenizer is not consulted.
pub(super) fn execute_recursive_tokenizer_from_state_small(
    constraint: &Constraint,
    bytes: &[u8],
    scoped_start_state: u32,
) -> Option<TokenizerExecResult> {
    let (leaf_index, local_start_state) =
        constraint.recursive_tokenizer_leaf_state(scoped_start_state)?;
    let leaf = constraint.recursive_leaf_constraint(leaf_index)?;
    debug_assert!(
        !leaf.uses_compact_segmented_parser_runtime(),
        "recursive tokenizer layout leaves must be intact runtime constraints",
    );
    let local = execute_tokenizer_from_state_small(leaf, bytes, local_start_state);
    let mut result = TokenizerExecResult {
        end_state: TokenizerStateSet::new(),
        matches: Vec::with_capacity(local.matches.len()),
    };
    for local_end_state in local.end_state {
        let scoped = constraint.recursive_tokenizer_scoped_state(leaf_index, local_end_state)?;
        if !result.end_state.contains(&scoped) {
            result.end_state.push(scoped);
        }
    }
    for matched in local.matches {
        let scoped_end_state =
            constraint.recursive_tokenizer_scoped_state(leaf_index, matched.end_state)?;
        let globals = constraint
            .recursive_global_terminals_for_leaf_terminal(leaf_index, matched.id)?;
        for global_terminal in globals {
            if let Some(existing) = result
                .matches
                .iter_mut()
                .find(|existing| existing.id == global_terminal)
            {
                if matched.width >= existing.width {
                    existing.width = matched.width;
                    existing.end_state = scoped_end_state;
                }
            } else {
                result.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: global_terminal,
                    width: matched.width,
                    end_state: scoped_end_state,
                });
            }
        }
    }
    Some(result)
}

pub(super) fn execute_tokenizer_from_state_small_into(
    constraint: &Constraint,
    bytes: &[u8],
    start_state: u32,
    result: &mut TokenizerExecResult,
) {
    if constraint.tokenizer_has_epsilon_transitions {
        *result = constraint.tokenizer.execute_from_state(bytes, start_state);
        return;
    }
    result.end_state.clear();
    result.matches.clear();
    let mut tokenizer_state = start_state;

    for (index, &byte) in bytes.iter().enumerate() {
        let next_state = constraint.tokenizer_fast_transitions.transition(
            &constraint.tokenizer,
            tokenizer_state,
            byte,
        );
        if next_state == u32::MAX {
            return;
        }

        tokenizer_state = next_state;
        let width = index + 1;
        for terminal in constraint.tokenizer.matched_terminals_iter(tokenizer_state) {
            if let Some(existing) = result
                .matches
                .iter_mut()
                .find(|matched| matched.id == terminal)
            {
                existing.width = width;
                existing.end_state = tokenizer_state;
            } else {
                result.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                    id: terminal,
                    width,
                    end_state: tokenizer_state,
                });
            }
        }
    }

    result.end_state.push(tokenizer_state);
}
