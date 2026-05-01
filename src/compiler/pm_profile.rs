//! Possible-match profiling: shared types and helpers.
//!
//! Neutral module. Used by both `possible_matches` (for the legacy sparse
//! computer) and `constraint_possible_matches::collector` (for the dense
//! Constraint collector). Neither direction depends on the other module.

use std::time::Instant;

pub(crate) fn profile_summary_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_COMPILE_SUMMARY")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PossibleMatchesProfile {
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

pub(crate) fn merge_possible_matches_profile(
    into: &mut PossibleMatchesProfile,
    other: PossibleMatchesProfile,
) {
    into.cache_hits += other.cache_hits;
    into.cache_misses += other.cache_misses;
    into.reachable_cache_hits += other.reachable_cache_hits;
    into.reachable_cache_misses += other.reachable_cache_misses;
    into.child_segments_visited += other.child_segments_visited;
    into.byte_steps += other.byte_steps;
    into.blocked_segments += other.blocked_segments;
    into.recursive_descents += other.recursive_descents;
    into.self_loop_subtrees_skipped += other.self_loop_subtrees_skipped;
    into.terminal_insertions += other.terminal_insertions;
    into.cache_entries += other.cache_entries;
    into.reachable_cache_entries += other.reachable_cache_entries;
    into.cache_lookup_ms += other.cache_lookup_ms;
    into.reachable_lookup_ms += other.reachable_lookup_ms;
    into.node_terminal_insert_ms += other.node_terminal_insert_ms;
    into.segment_walk_ms += other.segment_walk_ms;
    into.self_loop_check_ms += other.self_loop_check_ms;
    into.merge_child_matches_ms += other.merge_child_matches_ms;
    into.root_compute_ms += other.root_compute_ms;
    into.materialize_output_ms += other.materialize_output_ms;
}

pub(crate) fn emit_possible_matches_profile_summary(
    label: &str,
    token_count: usize,
    state_count: u32,
    trie_build_ms: f64,
    collect_ms: f64,
    profile: &PossibleMatchesProfile,
) {
    if !profile_summary_enabled() {
        return;
    }

    eprintln!(
        "[glrmask/profile][possible_matches] label={} tokens={} states={} trie_build_ms={:.3} collect_ms={:.3} root_compute_ms={:.3} materialize_output_ms={:.3} cache_lookup_ms={:.3} reachable_lookup_ms={:.3} node_terminal_insert_ms={:.3} segment_walk_ms={:.3} self_loop_check_ms={:.3} merge_child_matches_ms={:.3} cache_entries={} reachable_cache_entries={} cache_hits={} cache_misses={} reachable_cache_hits={} reachable_cache_misses={} child_segments={} byte_steps={} blocked_segments={} recursive_descents={} self_loop_subtrees_skipped={} terminal_insertions={}",
        label,
        token_count,
        state_count,
        trie_build_ms,
        collect_ms,
        profile.root_compute_ms,
        profile.materialize_output_ms,
        profile.cache_lookup_ms,
        profile.reachable_lookup_ms,
        profile.node_terminal_insert_ms,
        profile.segment_walk_ms,
        profile.self_loop_check_ms,
        profile.merge_child_matches_ms,
        profile.cache_entries,
        profile.reachable_cache_entries,
        profile.cache_hits,
        profile.cache_misses,
        profile.reachable_cache_hits,
        profile.reachable_cache_misses,
        profile.child_segments_visited,
        profile.byte_steps,
        profile.blocked_segments,
        profile.recursive_descents,
        profile.self_loop_subtrees_skipped,
        profile.terminal_insertions,
    );
}
