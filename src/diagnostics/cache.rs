//! Explicit cache-management utilities.
//!
//! These are intentionally not re-exported from the crate root.  Normal users
//! should not need to think about weight interning or operation memoization.
//! Benchmarks and long-running diagnostic processes may call these functions to
//! make cache effects explicit.

/// Clear stale weak references from the global weight interners.
pub fn clear_stale_weights() {
    crate::sets::weight::clear_stale_weights();
}

/// Clear all global weight interners.
pub fn clear_weight_interners() {
    crate::sets::weight::clear_weight_interners();
}

/// Clear cached weight operations such as unions/intersections.
pub fn clear_weight_op_caches() {
    crate::sets::weight::clear_weight_op_caches();
}
