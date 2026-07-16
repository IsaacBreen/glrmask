//! Final grammar-ignore elimination.
//!
//! Initial ignore matches deliberately remain labelled while the L2P terminal
//! automaton is built and while terminal interchangeability is expanded. The
//! TI dispatcher/transport construction needs that first raw terminal choice.
//! Once every partition has completed TI expansion and like terminal families
//! have been merged, those initial labels become weighted epsilon edges.

use std::collections::BTreeSet;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

fn union_final_weight(slot: &mut Option<Weight>, add: Weight) {
    if add.is_empty() {
        return;
    }
    *slot = Some(match slot.take() {
        Some(existing) => existing.union(&add),
        None => add,
    });
}

fn dwa_to_nwa(dwa: DWA) -> NWA {
    let start_state = dwa.start_state();
    let states = dwa
        .states()
        .iter()
        .map(|state| NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| (label, vec![(*target, weight.clone())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();
    NWA::from_parts(states, vec![start_state])
}

/// Preserve the common two-state L1 shape without introducing an NWA. The
/// ignore edge reaches the shared terminal leaf, so erasing it means adding its
/// accepted token weight to the start final.
fn try_erase_leaf_ignore_from_dwa(dwa: &mut DWA, ignore_label: i32) -> bool {
    let start = dwa.start_state() as usize;
    let Some((target, edge_weight)) = dwa
        .states()
        .get(start)
        .and_then(|state| state.transitions.get(&ignore_label))
        .cloned()
    else {
        return false;
    };

    if dwa.states().iter().enumerate().any(|(state_id, state)| {
        state_id != start && state.transitions.contains_key(&ignore_label)
    }) {
        return false;
    }

    let Some(target_state) = dwa.states().get(target as usize) else {
        return false;
    };
    if target_state
        .transitions
        .values()
        .any(|(_, weight)| !weight.is_empty())
    {
        return false;
    }

    let accepted = target_state
        .final_weight
        .as_ref()
        .map(|final_weight| edge_weight.intersection(final_weight))
        .unwrap_or_else(Weight::empty);
    let start_state = &mut dwa.states_mut()[start];
    start_state.transitions.remove(&ignore_label);
    union_final_weight(&mut start_state.final_weight, accepted);
    true
}

fn contains_label(automaton: &TerminalAutomaton, label: i32) -> bool {
    match automaton {
        TerminalAutomaton::Dwa(dwa) => dwa
            .states()
            .iter()
            .any(|state| state.transitions.contains_key(&label)),
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa
            .states()
            .iter()
            .any(|state| state.transitions.contains_key(&label)),
    }
}

/// Remove the grammar ignore label only after TI expansion and family merge.
///
/// Every remaining labelled ignore edge must originate at a terminal-automaton
/// start state. It is changed in place to a weighted epsilon edge. We do not
/// take an epsilon closure or duplicate suffix transitions here: parser-DWA
/// composition consumes this post-TI epsilon structure directly.
pub(crate) fn erase_ignore_after_ti(
    automaton: TerminalAutomaton,
    ignore_terminal: Option<TerminalID>,
) -> TerminalAutomaton {
    let Some(ignore_terminal) = ignore_terminal else {
        return automaton;
    };
    let ignore_label = ignore_terminal as i32;
    if !contains_label(&automaton, ignore_label) {
        return automaton;
    }

    if let TerminalAutomaton::Dwa(mut dwa) = automaton {
        if try_erase_leaf_ignore_from_dwa(&mut dwa, ignore_label) {
            debug_assert!(!dwa
                .states()
                .iter()
                .any(|state| state.transitions.contains_key(&ignore_label)));
            return TerminalAutomaton::Dwa(dwa);
        }
        return erase_start_ignore_from_nwa(dwa_to_nwa(dwa), ignore_label);
    }

    match automaton {
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => erase_start_ignore_from_nwa(nwa, ignore_label),
        TerminalAutomaton::Dwa(_) => unreachable!(),
    }
}

fn erase_start_ignore_from_nwa(mut nwa: NWA, ignore_label: i32) -> TerminalAutomaton {
    let starts: BTreeSet<u32> = nwa.start_states().iter().copied().collect();
    for (state_id, state) in nwa.states().iter().enumerate() {
        if state.transitions.contains_key(&ignore_label) {
            assert!(
                starts.contains(&(state_id as u32)),
                "labelled IGNORE survived outside a terminal-family start state after TI expansion",
            );
        }
    }

    for &start in nwa.start_states().to_vec().iter() {
        let start_idx = start as usize;
        let ignore_targets = nwa.states_mut()[start_idx]
            .transitions
            .remove(&ignore_label)
            .unwrap_or_default();
        nwa.states_mut()[start_idx].epsilons.extend(
            ignore_targets
                .into_iter()
                .filter(|(_, weight)| !weight.is_empty()),
        );
    }

    assert!(
        nwa.states()
            .iter()
            .all(|state| !state.transitions.contains_key(&ignore_label)),
        "final terminal family still contains a labelled IGNORE edge",
    );
    TerminalAutomaton::EpsilonNwa(nwa)
}

#[cfg(test)]
mod tests {
    use range_set_blaze::RangeSetBlaze;

    use crate::automata::weighted::determinize::determinize;

    use super::*;

    fn token_weight(tokens: impl IntoIterator<Item = u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, tokens.into_iter().collect::<RangeSetBlaze<_>>())
    }

    #[test]
    fn leaf_ignore_becomes_empty_terminal_word() {
        let ignore = 0;
        let terminal = 1;
        let mut dwa = DWA::new(1, 2);
        let final_state = dwa.add_state();
        dwa.set_final_weight(final_state, Weight::all());
        dwa.add_transition(dwa.start_state(), ignore, final_state, token_weight([0]));
        dwa.add_transition(dwa.start_state(), terminal, final_state, token_weight([1]));

        let TerminalAutomaton::Dwa(erased) =
            erase_ignore_after_ti(TerminalAutomaton::Dwa(dwa), Some(ignore as u32))
        else {
            panic!("leaf fast path must remain a DWA");
        };

        assert_eq!(erased.eval_word(&[]), token_weight([0]));
        assert_eq!(erased.eval_word(&[terminal]), token_weight([1]));
        assert!(erased.eval_word(&[ignore]).is_empty());
    }

    #[test]
    fn ignore_prefixed_suffix_becomes_weighted_epsilon_after_ti() {
        let ignore = 0;
        let terminal = 1;
        let mut dwa = DWA::new(1, 3);
        let after_ignore = dwa.add_state();
        let direct_final = dwa.add_state();
        let suffix_final = dwa.add_state();
        dwa.set_final_weight(after_ignore, token_weight([1]));
        dwa.set_final_weight(direct_final, Weight::all());
        dwa.set_final_weight(suffix_final, Weight::all());
        dwa.add_transition(
            dwa.start_state(),
            ignore,
            after_ignore,
            token_weight([1, 2]),
        );
        dwa.add_transition(
            dwa.start_state(),
            terminal,
            direct_final,
            token_weight([0]),
        );
        dwa.add_transition(after_ignore, terminal, suffix_final, token_weight([2]));

        let TerminalAutomaton::EpsilonNwa(erased) =
            erase_ignore_after_ti(TerminalAutomaton::Dwa(dwa), Some(ignore as u32))
        else {
            panic!("non-leaf ignore must become a post-TI epsilon NWA");
        };
        let start = erased.start_states()[0] as usize;
        assert!(!erased.states()[start].transitions.contains_key(&ignore));
        assert_eq!(erased.states()[start].epsilons.len(), 1);

        let determinized = determinize(&erased).unwrap();
        assert_eq!(determinized.eval_word(&[]), token_weight([1]));
        assert_eq!(determinized.eval_word(&[terminal]), token_weight([0, 2]));
        assert!(determinized.eval_word(&[ignore]).is_empty());
    }

    #[test]
    #[should_panic(expected = "outside a terminal-family start state")]
    fn rejects_non_initial_labelled_ignore() {
        let ignore = 0;
        let mut nwa = NWA::new(1, 1);
        let start = nwa.add_state();
        let middle = nwa.add_state();
        let final_state = nwa.add_state();
        nwa.set_start_states(vec![start]);
        nwa.add_transition(start, 1, middle, Weight::all());
        nwa.add_transition(middle, ignore, final_state, Weight::all());
        nwa.set_final_weight(final_state, Weight::all());

        let _ = erase_ignore_after_ti(
            TerminalAutomaton::TokenDeterministicNwa(nwa),
            Some(ignore as u32),
        );
    }
}
