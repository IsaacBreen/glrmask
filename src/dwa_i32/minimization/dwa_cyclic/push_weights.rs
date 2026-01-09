//! Dummy implementation for pushing weights (state_weight removed).

use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::DWA;

impl DWA {
    pub fn push_weights_into_transitions_and_finals_cyclic(&mut self) -> bool {
        false
    }
}
