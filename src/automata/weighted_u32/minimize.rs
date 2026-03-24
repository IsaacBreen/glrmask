//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to minimize.
use super::dwa::DWA;

fn should_skip_minimization(dwa: &DWA) -> bool {
    dwa.states.is_empty() || !dwa.is_acyclic()
}

pub fn minimize(dwa: &DWA) -> DWA {
    if should_skip_minimization(dwa) {
        return dwa.clone();
    }

    super::minimize_acyclic::minimize_acyclic(dwa)
}

/// Like [`minimize`], but switches from the O(n²) incompatibility graph to
/// partition-refinement coloring when any height bucket exceeds
/// `partition_refine_threshold` candidates. Produces a slightly larger DWA
/// for those buckets but avoids the quadratic cost.
pub fn minimize_with_threshold(dwa: &DWA, partition_refine_threshold: usize) -> DWA {
    if should_skip_minimization(dwa) {
        return dwa.clone();
    }
    super::minimize_acyclic::minimize_acyclic_with_threshold(dwa, partition_refine_threshold)
}

/// Fast minimize that uses signature-based partition refinement instead of
/// O(n²) pairwise graph coloring. Produces a valid (correct) DWA that may
/// be slightly larger than the graph-coloring result (doesn't merge states
/// with overlapping needed sets). Suitable for bundle minimization where
/// the extra few states are acceptable.
pub fn minimize_fast(dwa: &DWA) -> DWA {
    if should_skip_minimization(dwa) {
        return dwa.clone();
    }
    super::minimize_acyclic::minimize_acyclic_fast(dwa)
}
