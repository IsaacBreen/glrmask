//! Resolve negative parser-state labels in weighted NWAs.
//!
//! Cargo-check-only skeleton: signatures and module structure are preserved,
//! but implementation bodies are intentionally gutted.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use crate::automata::weighted::nwa::Nwa;
use crate::automata::weighted::weight::Weight;

pub(crate) fn compute_cancellations(nwa: &Nwa) -> Vec<(u32, u32, Weight)> {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn apply_cancellations(nwa: &mut Nwa) {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut Nwa) {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn remove_negative_transitions(nwa: &mut Nwa) {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn remove_redundant_default_transitions(nwa: &mut Nwa) {
    unimplemented!("cargo-check-only stub")
}

pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut Nwa) {
    unimplemented!("cargo-check-only stub")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted::weight::Weight;
    use crate::automata::weighted::weight::TokenSet;

    fn singleton_weight(token: u32) -> Weight {
        Weight::from_entries(vec![(0, 0, TokenSet::from_iter([token..=token]))])
    }

    #[test]
    fn removes_default_transition_to_terminal_final_state() {
        let mut nwa = Nwa::new(1, 3);
        let start = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let weight = singleton_weight(1);
        nwa.add_transition(start, DEFAULT_LABEL, end, weight.clone());
        nwa.set_final_weight(end, weight.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert_eq!(nwa.states[start as usize].final_weight.as_ref(), Some(&weight));
        assert!(!nwa.states[start as usize].transitions.contains_key(&DEFAULT_LABEL));
    }

    #[test]
    fn removes_default_only_chain_after_finality_propagation() {
        let mut nwa = Nwa::new(1, 3);
        let start = nwa.add_state();
        let mid = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let weight = singleton_weight(2);
        nwa.add_transition(start, DEFAULT_LABEL, mid, weight.clone());
        nwa.add_transition(mid, DEFAULT_LABEL, end, weight.clone());
        nwa.set_final_weight(end, weight.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert_eq!(nwa.states[start as usize].final_weight.as_ref(), Some(&weight));
        assert_eq!(nwa.states[mid as usize].final_weight.as_ref(), Some(&weight));
        assert!(!nwa.states[start as usize].transitions.contains_key(&DEFAULT_LABEL));
        assert!(!nwa.states[mid as usize].transitions.contains_key(&DEFAULT_LABEL));
    }

    #[test]
    fn keeps_default_transition_when_target_is_not_terminal() {
        let mut nwa = Nwa::new(1, 3);
        let start = nwa.add_state();
        let mid = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let weight = singleton_weight(0);
        nwa.add_transition(start, DEFAULT_LABEL, mid, weight.clone());
        nwa.add_transition(mid, 7, end, weight.clone());
        nwa.set_final_weight(end, weight.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert!(nwa.states[start as usize].transitions.contains_key(&DEFAULT_LABEL));
    }
}
