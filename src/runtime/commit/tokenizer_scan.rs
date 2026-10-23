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
        Self::with_capacity(512)
    }
}

impl ReusableTokenizerExecScratch {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            matches: Vec::with_capacity(capacity),
        }
    }
}

#[inline]
fn seed_first_byte_frontier(
    constraint: &Constraint,
    first: &[u32],
    scratch: &mut ReusableTokenizerExecScratch,
) -> bool {
    if first.len() > scratch.states.capacity() { return false; }
    scratch.states.extend_from_slice(first);
    for &state in &scratch.states {
        // Do not materialize all lexer finalizer lists to report this byte.
        for terminal in constraint.tokenizer.matched_terminals_iter(state) {
            if scratch.matches.len() == scratch.matches.capacity() { return false; }
            scratch.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                id: terminal, width: 1, end_state: state,
            });
        }
    }
    true
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

    // Check the prepared first-byte frontier BEFORE obtaining the reset
    // epsilon closure. A packed/lazy lexer may not cache that full closure;
    // rebuilding it only to ignore it would retain linear work per commit.
    let first_frontier = if start_state == constraint.tokenizer.initial_state() {
        bytes.first().and_then(|&byte| constraint.tokenizer.cached_initial_byte_frontiers()
            .map(|rows| &rows[byte as usize]))
    } else { None };
    let mut byte_start = 0usize;
    if let Some(first) = first_frontier {
        if !seed_first_byte_frontier(constraint, first, scratch) { return false; }
        if scratch.states.is_empty() { return true; }
        byte_start = 1;
    } else {
        let owned_start_closure;
        let start_closure: &[u32] = if let Some(closure) = cached_closures
            .and_then(|all| all.get(start_state as usize)) {
            closure
        } else if constraint.tokenizer.state_has_epsilon_transitions(start_state) {
            owned_start_closure = constraint.tokenizer.singleton_epsilon_closure(start_state);
            &owned_start_closure
        } else {
            // Stack-owned singleton, including virtual arithmetic lexer states.
            std::slice::from_ref(&start_state)
        };
        if start_closure.len() > scratch.states.capacity() {
            if start_state != constraint.runtime_commit_initial_state() { return false; }
            let Some(&first_byte) = bytes.first() else { return false; };
            // Existing fallback for non-finalized internal artifacts. Published
            // constraints prepare the cache in compile/load, not in start().
            let frontiers = constraint.tokenizer.initial_byte_frontiers_ref();
            if !seed_first_byte_frontier(constraint, &frontiers[first_byte as usize], scratch) { return false; }
            if scratch.states.is_empty() { return true; }
            byte_start = 1;
        } else {
            scratch.states.extend_from_slice(start_closure);
        }
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
                if let Some(target_closure) = closures.get(target as usize) {
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
                    // The frontier already deduplicates states, not terminal
                    // IDs. Keep every accepting end state at the longest width.
                    Some(_) => {},
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

/// Execute one recursive leaf tokenizer into the reusable bounded scratch,
/// then rebase its local continuation states and terminal matches into the
/// live scoped runtime coordinate.
///
/// This is deliberately separate from `execute_tokenizer_reusable`: most
/// callers of that helper are monolithic flat/linear accelerators whose parser
/// logic is not yet provider-aware. The recursive single-lane lexer-only fast
/// path can use this helper safely because it never changes the parser GSS; any
/// ignored or parser-actionable match makes that path decline transactionally.
pub(crate) fn execute_recursive_tokenizer_reusable(
    constraint: &Constraint,
    bytes: &[u8],
    scoped_start_state: u32,
    scratch: &mut ReusableTokenizerExecScratch,
) -> bool {
    let Some((leaf_index, local_start_state)) =
        constraint.recursive_tokenizer_leaf_state(scoped_start_state)
    else {
        return false;
    };
    let Some(leaf) = constraint.recursive_leaf_constraint(leaf_index) else {
        return false;
    };
    debug_assert!(
        !leaf.uses_compact_segmented_parser_runtime(),
        "recursive tokenizer layout leaves must be intact runtime constraints",
    );
    if !execute_tokenizer_reusable(leaf, bytes, local_start_state, scratch) {
        return false;
    }
    for state in &mut scratch.states {
        let Some(scoped) = constraint.recursive_tokenizer_scoped_state(leaf_index, *state) else {
            return false;
        };
        *state = scoped;
    }
    for matched in &mut scratch.matches {
        let Some(scoped_end_state) =
            constraint.recursive_tokenizer_scoped_state(leaf_index, matched.end_state)
        else {
            return false;
        };
        let Some(runtime_terminal) =
            constraint.recursive_terminal_scoped_id(leaf_index, matched.id)
        else {
            return false;
        };
        matched.end_state = scoped_end_state;
        matched.id = runtime_terminal;
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
            if let Some(closure) = closures.get(start_state as usize) {
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
                if let Some(closure) = closures.get(target as usize) {
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
                    // The frontier already deduplicates states, not terminal
                    // IDs. Keep every accepting end state at the longest width.
                    Some(_) => {},
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
    if constraint.uses_compact_segmented_parser_runtime() {
        return execute_recursive_tokenizer_from_state_small(constraint, bytes, start_state)
            .unwrap_or_else(|| TokenizerExecResult {
                end_state: TokenizerStateSet::new(),
                matches: Vec::new(),
            });
    }
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
        let runtime_terminal =
            constraint.recursive_terminal_scoped_id(leaf_index, matched.id)?;
        if let Some(existing) = result
            .matches
            .iter_mut()
            .find(|existing| existing.id == runtime_terminal)
        {
            if matched.width >= existing.width {
                existing.width = matched.width;
                existing.end_state = scoped_end_state;
            }
        } else {
            result.matches.push(crate::automata::lexer::tokenizer::TokenizerMatch {
                id: runtime_terminal,
                width: matched.width,
                end_state: scoped_end_state,
            });
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

#[cfg(test)]
mod reset_projection_tests {
    use super::*;
    use crate::{BuildOptions, Constraint, Grammar, MaskStrategy, Vocab};

    #[test]
    fn medium_and_large_reset_projections_match_generic_lexer_execution() {
        let vocab = Vocab::new((0u32..=255).map(|id| (id, vec![id as u8])).collect());
        let mut projected_cases = 0;
        for width in [16, 128, 512] {
            let source = format!("start root nt root = {}", std::iter::once("\"a\"+".to_owned())
                .chain((0..width).map(|id| format!("\"x{id}:\" @mark({id}) \"b\"")))
                .collect::<Vec<_>>().join(" | "));
            for mode in [MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
                let fresh = Grammar::from_glrm(&source).unwrap().compile_with_build_options(&vocab,
                    BuildOptions::default().optimization(mode)).unwrap();
                let loaded = Constraint::load(fresh.save(), &vocab).unwrap();
                for constraint in [&fresh, &loaded] {
                    let reset = constraint.runtime_commit_initial_state();
                    let closure = constraint.tokenizer.singleton_epsilon_closure(reset);
                    if closure.len() > 16 {
                        assert!(constraint.tokenizer.cached_initial_byte_frontiers().is_some(),
                            "projection must be prepared before start: {width} {mode:?}");
                        projected_cases += 1;
                    }
                    let mut inputs = (0u8..=255).map(|byte| vec![byte]).collect::<Vec<_>>();
                    inputs.extend([b"".as_slice(), b"aa", b"abababab", b"x0:", b"x0:b",
                        b"x127:b", b"x511:b", b"x512:b", b"xx", b"unused"].into_iter().map(Vec::from));
                    let mut scratch = ReusableTokenizerExecScratch::default();
                    for input in inputs {
                        let expected = constraint.tokenizer.execute_from_state(&input, reset);
                        if !execute_tokenizer_reusable(constraint, &input, reset, &mut scratch) {
                            // Empty input has no first-byte projection; a wide
                            // raw epsilon frontier can still exceed scratch.
                            assert!(input.is_empty() && closure.len() > scratch.states.capacity(),
                                "bounded fixture declined: {width} {mode:?} {input:?}");
                            continue;
                        }
                        let mut got_states = scratch.states.clone(); got_states.sort_unstable(); got_states.dedup();
                        let mut expected_states = expected.end_state.to_vec(); expected_states.sort_unstable(); expected_states.dedup();
                        assert_eq!(got_states, expected_states, "end states {width} {mode:?} {input:?}");
                        let mut got = scratch.matches.iter().map(|m| (m.id, m.width)).collect::<Vec<_>>();
                        let mut expected = expected.matches.iter().map(|m| (m.id, m.width)).collect::<Vec<_>>();
                        got.sort_unstable(); got.dedup(); expected.sort_unstable(); expected.dedup();
                        assert_eq!(got, expected, "matches {width} {mode:?} {input:?}");
                    }
                }
            }
        }
        assert!(projected_cases >= 4, "fixtures must exercise medium and large reset projections");
    }
    #[test]
    fn bounded_scan_keeps_all_longest_accepting_end_states() {
        use crate::automata::lexer::tokenizer::Tokenizer;
        use std::sync::Arc;
        let vocab = Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec())]);
        for cached in [false,true] {
            let mut constraint = Grammar::from_glrm("start root; t A=/ab*/; nt root=A;").unwrap()
                .compile_with_build_options(&vocab,BuildOptions::default().optimization(MaskStrategy::TokenTrie)).unwrap();
            // The public analysis union retains distinct states but shares
            // terminal IDs, exactly the overlapping-lane case under test.
            let tokenizer = Tokenizer::disjoint_union_for_analysis(
                &constraint.tokenizer, &constraint.tokenizer).tokenizer;
            if cached { let _=tokenizer.initial_byte_frontiers(); }
            constraint.tokenizer=Arc::new(tokenizer);
            // The union changed state coordinates; do not use the old source
            // constraint's unrelated precomputed transition rows.
            constraint.tokenizer_fast_transitions=Default::default();
            let mut scratch=ReusableTokenizerExecScratch::default();
            for input in [b"a".as_slice(),b"ab",b"abb"] {
                assert!(execute_tokenizer_reusable(&constraint,input,0,&mut scratch));
                let mut got=scratch.matches.iter().map(|m|(m.id,m.width,m.end_state)).collect::<Vec<_>>();
                let mut expected=constraint.tokenizer.execute_from_state(input,0).matches
                    .into_iter().map(|m|(m.id,m.width,m.end_state)).collect::<Vec<_>>();
                got.sort_unstable();expected.sort_unstable();
                assert_eq!(got,expected,"cached={cached} input={input:?}");
                assert!(got.len() >= 2, "fixture must retain distinct accepting endpoints");
            }
        }
    }

}
