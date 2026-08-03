//! Weighted-DWA minimization entry points.
//!
//! These wrappers only minimize acyclic inputs. Cyclic DWAs are returned
//! unchanged and handled by the caller.
use super::dwa::DWA;
pub use super::minimize_acyclic::PointwiseClassOrder;

fn should_skip_minimization(dwa: &DWA) -> bool {
    dwa.states().is_empty() || !dwa.is_acyclic()
}

fn minimize_if_acyclic(dwa: &DWA, minimize: impl FnOnce(&DWA) -> DWA) -> DWA {
    if should_skip_minimization(dwa) {
        return dwa.clone();
    }

    minimize(dwa)
}

fn minimize_owned_if_acyclic(dwa: DWA, minimize: impl FnOnce(DWA) -> DWA) -> DWA {
    if should_skip_minimization(&dwa) {
        return dwa;
    }

    minimize(dwa)
}

pub fn minimize(dwa: &DWA) -> DWA {
    minimize_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic)
}

pub fn minimize_owned(dwa: DWA) -> DWA {
    minimize_owned_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic_owned)
}

/// Minimize an acyclic DWA using an explicit exact pointwise grouping order.
/// The ordinary entry point keeps the stable historical order.
pub fn minimize_owned_with_pointwise_class_order(
    dwa: DWA,
    pointwise_class_order: PointwiseClassOrder,
) -> DWA {
    minimize_owned_if_acyclic(dwa, |dwa| {
        super::minimize_acyclic::minimize_acyclic_owned_with_pointwise_class_order(
            dwa,
            pointwise_class_order,
        )
    })
}
