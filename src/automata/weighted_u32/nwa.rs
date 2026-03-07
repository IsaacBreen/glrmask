//! Nondeterministic Weighted Automaton (NWA).
//!
//! The NWA is the intermediate representation produced by the compiler
//! (one NWA per grammar nonterminal, or a combined super-NWA) before
//! determinization into a [`Dwa`](super::dwa::Dwa).
//!
//! Transition labels are `i32` (grammar symbol IDs).  Weights are
//! [`Weight`](super::weight::Weight) sets representing which
//! (token, TSID) positions survive a transition.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use super::weight::Weight;

/// Grammar-symbol label.
pub type Label = i32;

/// A single NWA state.
#[derive(Debug, Clone, Default)]
pub struct NwaState {
    /// Optional final (accepting) weight.  `Some(w)` means the state is
    /// accepting and the set of surviving positions is `w`.
    pub final_weight: Option<Weight>,
    /// Label-keyed transitions: label → list of (target, weight).
    pub transitions: BTreeMap<Label, Vec<(u32, Weight)>>,
    /// ε-transitions: (target, weight).
    pub epsilons: Vec<(u32, Weight)>,
}

/// A Nondeterministic Weighted Automaton.
#[derive(Debug, Clone)]
pub struct Nwa {
    /// All states.
    pub states: Vec<NwaState>,
    /// Start states (subset construction begins from the ε-closure of these).
    pub start_states: Vec<u32>,
    /// Number of TSIDs (dimension bound for weight operations).
    pub num_tsids: u32,
    /// Maximum token ID (dimension bound for weight operations).
    pub max_token: u32,
}

impl Nwa {
    /// Create an empty NWA.
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

    /// Set the final weight for a state (makes it accepting).
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        unimplemented!()
    }

    /// Add a labelled transition.
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        unimplemented!()
    }

    /// Add an ε-transition.
    pub fn add_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        unimplemented!()
    }

    /// Total number of transitions (labelled + ε).
    pub fn num_transitions(&self) -> usize {
        unimplemented!()
    }

    /// Maximum position in the weight space.
    pub fn max_position(&self) -> u32 {
        unimplemented!()
    }

    /// Return a wrapper that prints this NWA using a symbol→name map.
    ///
    /// Labels not present in the map print as raw integers.
    pub fn display_with_symbols<'a>(
        &'a self,
        symbols: &'a std::collections::BTreeMap<Label, String>,
    ) -> NwaDisplayWithSymbols<'a> {
        unimplemented!()
    }

    /// Return a wrapper that prints this NWA using maps for symbols, TSIDs,
    /// and token IDs.
    pub fn display_with_all_maps<'a>(
        &'a self,
        symbols: &'a std::collections::BTreeMap<Label, String>,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> NwaDisplayWithAllMaps<'a> {
        unimplemented!()
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Shared formatting logic for NWA states.
fn fmt_nwa_states(
    nwa: &Nwa,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    let start_set: std::collections::BTreeSet<u32> = nwa.start_states.iter().copied().collect();

    for (i, st) in nwa.states.iter().enumerate() {
        if st.transitions.is_empty() && st.epsilons.is_empty() && st.final_weight.is_none() {
            continue;
        }

        // State header — finality implied by final: block, not header tag
        let start_mark = if start_set.contains(&(i as u32)) { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        // Final weight on its own line
        if let Some(w) = &st.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        // Transitions
        for (label, targets) in &st.transitions {
            let lbl = label_fn(*label);
            for (tgt, w) in targets {
                writeln!(f, "    {lbl} → State {tgt}")?;
                writeln!(f, "      weight: {}", weight_fn(w))?;
            }
        }

        // Epsilon transitions
        for (tgt, w) in &st.epsilons {
            writeln!(f, "    ε → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl std::fmt::Display for Nwa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    unimplemented!()
}
}

/// Wrapper to display an [`Nwa`] with human-readable symbol names.
pub struct NwaDisplayWithSymbols<'a> {
    nwa: &'a Nwa,
    symbols: &'a std::collections::BTreeMap<Label, String>,
}

impl std::fmt::Display for NwaDisplayWithSymbols<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nwa = self.nwa;
        writeln!(f, "NWA: {} states, start={:?}, tsids={}, max_token={}",
            nwa.states.len(), nwa.start_states, nwa.num_tsids, nwa.max_token)?;
        let syms = self.symbols;
        fmt_nwa_states(nwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{w}"),
        )
    }
}

/// Wrapper to display an [`Nwa`] with maps for symbols, TSIDs, and tokens.
pub struct NwaDisplayWithAllMaps<'a> {
    nwa: &'a Nwa,
    symbols: &'a std::collections::BTreeMap<Label, String>,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl std::fmt::Display for NwaDisplayWithAllMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_nwa_basic() {
        let mut nwa = Nwa::new(2, 10);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();

        let w = Weight::all();
        nwa.add_transition(s0, 0, s1, w.clone());
        nwa.add_epsilon(s1, s2, w.clone());
        nwa.set_final_weight(s2, w);

        assert_eq!(nwa.num_states(), 3);
        assert_eq!(nwa.num_transitions(), 2);
        assert!(nwa.states[s2 as usize].final_weight.is_some());
    }
}
