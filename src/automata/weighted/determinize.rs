//! Determinization: NWA → DWA.
//!
//! Converts a nondeterministic weighted automaton into a deterministic one
//! using an acyclic subset-construction variant that tracks weight offsets.

use super::dwa::Dwa;
use super::nwa::Nwa;
use super::weight::WeightTable;

/// Determinize an NWA into a DWA.
///
/// This assumes the NWA is acyclic (which it will be in our compilation
/// pipeline). For acyclic NWAs, the determinization always terminates
/// and the weight semantics are well-defined.
pub fn determinize(nwa: &Nwa) -> Dwa {
    // TODO: Implement acyclic determinization
    // For now, return a trivial single-state DWA
    let weights = WeightTable::new(1, nwa.num_tsids);
    Dwa::new(weights, 0, vec![true])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinize_trivial() {
        let nwa = Nwa::new(1, 1);
        let dwa = determinize(&nwa);
        assert_eq!(dwa.num_states(), 1);
    }
}
