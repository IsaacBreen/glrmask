//! Dummy implementation for pushing weights (state_weight removed).

use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::DWA;

impl DWA {
    pub fn push_weights_into_transitions_and_finals_cyclic(&mut self) -> bool {
        false
    }
}
