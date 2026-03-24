//! Weighted-DWA minimization entry points.
//!
//! These wrappers only minimize acyclic inputs. Cyclic DWAs are returned
//! unchanged and handled by the caller.
use super::dwa::DWA;

fn should_skip_minimization(dwa: &DWA) -> bool {
    dwa.states.is_empty() || !dwa.is_acyclic()
}

fn minimize_if_acyclic(dwa: &DWA, minimize: impl FnOnce(&DWA) -> DWA) -> DWA {
    if should_skip_minimization(dwa) {
        return dwa.clone();
    }

    minimize(dwa)
}

pub fn minimize(dwa: &DWA) -> DWA {
    minimize_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic)
}

/// Like [`minimize`], but switches from the O(n²) incompatibility graph to
/// partition-refinement coloring when any height bucket exceeds
/// `partition_refine_threshold` candidates. Produces a slightly larger DWA
/// for those buckets but avoids the quadratic cost.
pub fn minimize_with_threshold(dwa: &DWA, partition_refine_threshold: usize) -> DWA {
    minimize_if_acyclic(dwa, |dwa| {
        super::minimize_acyclic::minimize_acyclic_with_threshold(dwa, partition_refine_threshold)
    })
}

/// Fast minimize that uses signature-based partition refinement instead of
/// O(n²) pairwise graph coloring. Produces a valid (correct) DWA that may
/// be slightly larger than the graph-coloring result (doesn't merge states
/// with overlapping needed sets). Suitable for bundle minimization where
/// the extra few states are acceptable.
pub fn minimize_fast(dwa: &DWA) -> DWA {
    minimize_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic_fast)
}
