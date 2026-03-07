//! Deterministic Weighted Automaton (DWA).
//!
//! Shape-first compilation-time deterministic weighted automaton.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::nwa::Label;
use super::weight::Weight;

/// A single state in the compilation-time DWA.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DwaState {
    /// Label → (target_state, transition_weight).
    pub transitions: BTreeMap<Label, (u32, Weight)>,
    /// Accepting weight, or `None` if the state is non-accepting.
    pub final_weight: Option<Weight>,
}

/// Compilation-time DWA.
///
/// Each `(state, label)` maps to at most one `(target, weight)`.  The weights
/// are full [`Weight`] sets that track which (token, TSID) positions survive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dwa {
    /// All states.
    pub states: Vec<CompDwaState>,
    /// Start state ID.
    pub start_state: u32,
    /// Number of TSIDs.
    pub num_tsids: u32,
    /// Maximum token ID.
    pub max_token: u32,
}

impl Dwa {
    /// Create a new Dwa with a single (empty) start state.
    pub fn new(num_tsids: u32, max_token: u32) -> Self {
        unimplemented!()
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self) -> u32 {
        unimplemented!()
    }

    /// Number of states.
    pub fn num_states(&self) -> u32 {
        unimplemented!()
    }

    /// Total number of transitions across all states.
    pub fn num_transitions(&self) -> usize {
        unimplemented!()
    }

    /// Set the final weight for a state.
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        unimplemented!()
    }

    /// Add a labelled transition.
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        unimplemented!()
    }

    /// Evaluate a word, returning the surviving weight.
    ///
    /// Follows transitions for each label in the word, intersecting weights.
    /// Returns the intersection of all transition weights and the final weight
    /// of the last state (empty weight if any step fails).
    pub fn eval_word(&self, word: &[Label]) -> Weight {
        unimplemented!()
    }

    /// Collect all distinct labels used in transitions.
    pub fn labels(&self) -> Vec<Label> {
        unimplemented!()
    }

    /// Return a wrapper that prints this DWA using a symbol→name map.
    ///
    /// Labels not present in the map print as raw integers.
    pub fn display_with_symbols<'a>(
        &'a self,
        symbols: &'a BTreeMap<Label, String>,
    ) -> DwaDisplayWithSymbols<'a> {
        unimplemented!()
    }

    /// Return a wrapper that prints this DWA using maps for symbols, TSIDs,
    /// and token IDs.
    pub fn display_with_all_maps<'a>(
        &'a self,
        symbols: &'a BTreeMap<Label, String>,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> DwaDisplayWithAllMaps<'a> {
        unimplemented!()
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Shared formatting logic for CompDwa states.
fn fmt_dwa_states(
    dwa: &Dwa,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    for (i, st) in dwa.states.iter().enumerate() {
        if st.transitions.is_empty() && st.final_weight.is_none() {
            continue;
        }

        let start_mark = if i as u32 == dwa.start_state { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        if let Some(w) = &st.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        for (label, (tgt, w)) in &st.transitions {
            let lbl = label_fn(*label);
            writeln!(f, "    {lbl} → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl std::fmt::Display for Dwa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    unimplemented!()
}
}

/// Wrapper to display a [`Dwa`] with human-readable symbol names.
pub struct DwaDisplayWithSymbols<'a> {
    dwa: &'a Dwa,
    symbols: &'a BTreeMap<Label, String>,
}

impl std::fmt::Display for DwaDisplayWithSymbols<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dwa = self.dwa;
        writeln!(f, "DWA: {} states, start=State {}, tsids={}, max_token={}",
            dwa.states.len(), dwa.start_state, dwa.num_tsids, dwa.max_token)?;
        let syms = self.symbols;
        fmt_dwa_states(dwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{w}"),
        )
    }
}

/// Wrapper to display a [`Dwa`] with maps for symbols, TSIDs, and tokens.
pub struct DwaDisplayWithAllMaps<'a> {
    dwa: &'a Dwa,
    symbols: &'a BTreeMap<Label, String>,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl std::fmt::Display for DwaDisplayWithAllMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

impl PartialEq for Dwa {
    fn eq(&self, other: &Self) -> bool {
        unimplemented!()
    }
}

impl PartialEq for DwaState {
    fn eq(&self, other: &Self) -> bool {
        unimplemented!()
    }
}

pub type CompDwa = Dwa;
pub type CompDwaState = DwaState;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted::weight::TokenSet;

    #[test]
    fn test_comp_dwa_eval_word() {
        // Simple 2-state DWA: s0 --label 0--> s1 (accepting).
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();

        let w_trans = Weight::from_positions(&TokenSet::from_iter([0..=5]), nt);
        let w_final = Weight::from_positions(&TokenSet::from_iter([2..=4]), nt);
        dwa.add_transition(0, 0, s1, w_trans);
        dwa.set_final_weight(s1, w_final);

        let result = dwa.eval_word(&[0]);
        // acc starts as all(5, 1) = {0..=5}, intersect with trans {0..=5} = {0..=5}
        // then intersect with final {2..=4} = {2..=4}
        assert_eq!(result.len(), 3);
        assert!(result.contains(2, nt));
        assert!(result.contains(3, nt));
        assert!(result.contains(4, nt));
    }

    #[test]
    fn test_comp_dwa_eval_word_reject() {
        let nt = 1u32;
        let dwa = CompDwa::new(nt, 5);
        // No transition for label 0 → empty result.
        let result = dwa.eval_word(&[0]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_runtime_dwa() {
        let mut wt = WeightTable::new(3, 2);
        wt.set(0, 1, 2, 5);
        let dwa = Dwa::new(wt, 0, vec![false, false, true]);
        assert_eq!(dwa.step(1, 0), (2, 5));
        assert!(dwa.is_accepting(2));
        assert!(!dwa.is_accepting(0));
    }
}
