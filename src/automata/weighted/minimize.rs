//! DWA minimization.
//!
//! Partition-based minimization with relax and consolidate passes.

use super::dwa::Dwa;

/// Minimize a DWA by merging equivalent states.
///
/// Uses a partition refinement approach:
/// 1. **Partition**: Group states by transition signature
/// 2. **Relax**: Adjust weights to normalize equivalent behaviors
/// 3. **Consolidate**: Merge states and rebuild the weight table
pub fn minimize(dwa: &Dwa) -> Dwa {
    // TODO: Implement partition-based minimization
    // For now, return the DWA unchanged
    dwa.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::weight::WeightTable;

    #[test]
    fn test_minimize_noop() {
        let weights = WeightTable::new(1, 1);
        let dwa = Dwa::new(weights, 0, vec![true]);
        let min = minimize(&dwa);
        assert_eq!(min.num_states(), 1);
    }
}
