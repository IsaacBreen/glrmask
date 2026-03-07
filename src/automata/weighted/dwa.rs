//! Deterministic Weighted Automaton (DWA).
//!
//! Two representations live here:
//!
//! - **[`CompDwa`]** – compilation-time DWA whose transitions carry full
//!   [`Weight`] sets.  Used during determinization and minimization.
//! - **[`Dwa`]** – runtime DWA backed by a flat [`WeightTable`].  Produced at
//!   the end of compilation and used for fast mask computation during
//!   inference.
#![allow(dead_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::nwa::Label;
use super::weight::{Weight, WeightTable};

// ---------------------------------------------------------------------------
// CompDwa — compilation-time DWA
// ---------------------------------------------------------------------------

/// A single state in the compilation-time DWA.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompDwaState {
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
pub struct CompDwa {
    /// All states.
    pub states: Vec<CompDwaState>,
    /// Start state ID.
    pub start_state: u32,
    /// Number of TSIDs.
    pub num_tsids: u32,
    /// Maximum token ID.
    pub max_token: u32,
}

impl CompDwa {
    /// Create a new CompDwa with a single (empty) start state.
    pub fn new(num_tsids: u32, max_token: u32) -> Self {
        Self {
            states: vec![CompDwaState::default()],
            start_state: 0,
            num_tsids,
            max_token,
        }
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(CompDwaState::default());
        id
    }

    /// Number of states.
    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    /// Total number of transitions across all states.
    pub fn num_transitions(&self) -> usize {
        self.states.iter().map(|s| s.transitions.len()).sum()
    }

    /// Set the final weight for a state.
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        self.states[state as usize].final_weight = Some(weight);
    }

    /// Add a labelled transition.
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        self.states[from as usize]
            .transitions
            .insert(label, (to, weight));
    }

    /// Evaluate a word, returning the surviving weight.
    ///
    /// Follows transitions for each label in the word, intersecting weights.
    /// Returns the intersection of all transition weights and the final weight
    /// of the last state (empty weight if any step fails).
    pub fn eval_word(&self, word: &[Label]) -> Weight {
        use crate::compiler::parser_dwa::DEFAULT_LABEL;

        let empty = Weight::empty(self.num_tsids);
        if self.states.is_empty() {
            return empty;
        }

        let max_pos = self
            .max_token
            .saturating_mul(self.num_tsids.max(1))
            .saturating_add(self.num_tsids.max(1) - 1);
        let mut state = self.start_state;
        let mut acc = Weight::all(max_pos, self.num_tsids);

        for &label in word {
            // Try specific transition first, then DEFAULT fallback.
            let resolved = self.states[state as usize]
                .transitions
                .get(&label)
                .or_else(|| self.states[state as usize].transitions.get(&DEFAULT_LABEL));
            match resolved {
                Some(&(target, ref w)) => {
                    acc = acc.intersection(w);
                    if acc.is_empty() {
                        return empty;
                    }
                    state = target;
                }
                None => return empty,
            }
        }

        match &self.states[state as usize].final_weight {
            Some(fw) => {
                let result = acc.intersection(fw);
                if result.is_empty() { empty } else { result }
            }
            None => empty,
        }
    }

    /// Collect all distinct labels used in transitions.
    pub fn labels(&self) -> Vec<Label> {
        let mut labels: Vec<Label> = self
            .states
            .iter()
            .flat_map(|s| s.transitions.keys().copied())
            .collect();
        labels.sort_unstable();
        labels.dedup();
        labels
    }

    /// Return a wrapper that prints this DWA using a symbol→name map.
    ///
    /// Labels not present in the map print as raw integers.
    pub fn display_with_symbols<'a>(
        &'a self,
        symbols: &'a BTreeMap<Label, String>,
    ) -> CompDwaDisplayWithSymbols<'a> {
        CompDwaDisplayWithSymbols { dwa: self, symbols }
    }

    /// Return a wrapper that prints this DWA using maps for symbols, TSIDs,
    /// and token IDs.
    pub fn display_with_all_maps<'a>(
        &'a self,
        symbols: &'a BTreeMap<Label, String>,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> CompDwaDisplayWithAllMaps<'a> {
        CompDwaDisplayWithAllMaps { dwa: self, symbols, tsid_names, token_names }
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Shared formatting logic for CompDwa states.
fn fmt_comp_dwa_states(
    dwa: &CompDwa,
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

impl std::fmt::Display for CompDwa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA: {} states, start=State {}, tsids={}, max_token={}",
            self.states.len(), self.start_state, self.num_tsids, self.max_token)?;
        fmt_comp_dwa_states(self, f,
            &|label| format!("{label}"),
            &|w| format!("{w}"),
        )
    }
}

/// Wrapper to display a [`CompDwa`] with human-readable symbol names.
pub struct CompDwaDisplayWithSymbols<'a> {
    dwa: &'a CompDwa,
    symbols: &'a BTreeMap<Label, String>,
}

impl std::fmt::Display for CompDwaDisplayWithSymbols<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dwa = self.dwa;
        writeln!(f, "DWA: {} states, start=State {}, tsids={}, max_token={}",
            dwa.states.len(), dwa.start_state, dwa.num_tsids, dwa.max_token)?;
        let syms = self.symbols;
        fmt_comp_dwa_states(dwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{w}"),
        )
    }
}

/// Wrapper to display a [`CompDwa`] with maps for symbols, TSIDs, and tokens.
pub struct CompDwaDisplayWithAllMaps<'a> {
    dwa: &'a CompDwa,
    symbols: &'a BTreeMap<Label, String>,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl std::fmt::Display for CompDwaDisplayWithAllMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dwa = self.dwa;
        writeln!(f, "DWA: {} states, start=State {}, tsids={}, max_token={}",
            dwa.states.len(), dwa.start_state, dwa.num_tsids, dwa.max_token)?;
        let syms = self.symbols;
        let tsid_m = self.tsid_names;
        let tok_m = self.token_names;
        fmt_comp_dwa_states(dwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{}", w.display_with_maps(tsid_m, tok_m)),
        )
    }
}

impl PartialEq for CompDwa {
    fn eq(&self, other: &Self) -> bool {
        self.start_state == other.start_state
            && self.num_tsids == other.num_tsids
            && self.states.len() == other.states.len()
            && self
                .states
                .iter()
                .zip(other.states.iter())
                .all(|(a, b)| a.final_weight == b.final_weight && a.transitions == b.transitions)
    }
}

impl PartialEq for CompDwaState {
    fn eq(&self, other: &Self) -> bool {
        self.final_weight == other.final_weight && self.transitions == other.transitions
    }
}

// ---------------------------------------------------------------------------
// Dwa — runtime DWA (flat table)
// ---------------------------------------------------------------------------

/// A Deterministic Weighted Automaton operating over token-set IDs.
///
/// At each step, given the current state and a token-set ID (TSID),
/// the DWA produces a next state and an integer weight.  The weight
/// is used to determine whether the token is allowed (weight ≥ 0 means
/// allowed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dwa {
    /// The weight/transition table.
    pub weights: WeightTable,
    /// The start state.
    pub start_state: u32,
    /// Which states are accepting (valid end-of-sequence states).
    pub accepting: Vec<bool>,
}

impl Dwa {
    /// Create a new DWA.
    pub fn new(weights: WeightTable, start_state: u32, accepting: Vec<bool>) -> Self {
        Self {
            weights,
            start_state,
            accepting,
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> u32 {
        self.weights.num_states
    }

    /// Number of token-set IDs.
    pub fn num_tsids(&self) -> u32 {
        self.weights.num_tsids
    }

    /// Get the transition for `(state, tsid)`.
    #[inline]
    pub fn step(&self, state: u32, tsid: u32) -> (u32, i32) {
        self.weights.get(tsid, state)
    }

    /// Whether a state is accepting.
    pub fn is_accepting(&self, state: u32) -> bool {
        self.accepting.get(state as usize).copied().unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ds::RangeSet;

    #[test]
    fn test_comp_dwa_eval_word() {
        // Simple 2-state DWA: s0 --label 0--> s1 (accepting).
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();

        let w_trans = Weight::from_positions(&RangeSet::from_range(0, 5), nt);
        let w_final = Weight::from_positions(&RangeSet::from_range(2, 4), nt);
        dwa.add_transition(0, 0, s1, w_trans);
        dwa.set_final_weight(s1, w_final);

        let result = dwa.eval_word(&[0]);
        // acc starts as all(5, 1) = {0..=5}, intersect with trans {0..=5} = {0..=5}
        // then intersect with final {2..=4} = {2..=4}
        assert_eq!(result.len(), 3);
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(result.contains(4));
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
