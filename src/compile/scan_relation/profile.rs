//! Can-match profiling: shared types and helpers.
//!
//! Neutral module. Used by both `can_match` (for the legacy sparse
//! computer) and `scan_relation::collector` (for the dense
//! Constraint collector). Neither direction depends on the other module.

use std::time::Instant;

pub(crate) fn profile_summary_enabled() -> bool {
    crate::compile::profiling::compile_profile_summary_enabled()
}

pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CanMatchProfile {
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) reachable_cache_hits: u64,
    pub(crate) reachable_cache_misses: u64,
    pub(crate) child_segments_visited: u64,
    pub(crate) byte_steps: u64,
    pub(crate) blocked_segments: u64,
    pub(crate) recursive_descents: u64,
    pub(crate) self_loop_subtrees_skipped: u64,
    pub(crate) terminal_insertions: u64,
    pub(crate) cache_entries: usize,
    pub(crate) reachable_cache_entries: usize,
    pub(crate) cache_lookup_ms: f64,
    pub(crate) reachable_lookup_ms: f64,
    pub(crate) node_terminal_insert_ms: f64,
    pub(crate) segment_walk_ms: f64,
    pub(crate) self_loop_check_ms: f64,
    pub(crate) merge_child_matches_ms: f64,
    pub(crate) root_compute_ms: f64,
    pub(crate) materialize_output_ms: f64,
}
