//! Parser-DWA profiling records and textual emission.
//!
//! Construction code records phase timings into structs.  This file is the only
//! Parser-DWA submodule allowed to print profile lines.  Keeping profile output
//! here prevents the mathematical construction from being interleaved with
//! logging mechanics.

use std::time::Instant;

use crate::compile::template_dfa::BundleBuildProfile;

pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

pub(crate) fn parser_dwa_compose_detail_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_PARSER_DWA_COMPOSE_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ParserNwaBuildProfile {
    pub(crate) state_prep_ms: f64,
    pub(crate) compose_state_ms: f64,
    pub(crate) parser_nwa_build_ms: f64,
}

#[derive(Default)]
pub(crate) struct ParserDwaComposeDetailProfile {
    pub(crate) total_states: usize,
    pub(crate) productive_states: usize,
    pub(crate) total_branches: usize,
    pub(crate) productive_branches: usize,
    pub(crate) unique_bundles: usize,
    pub(crate) accepting_bundles: usize,
    pub(crate) state_init_ms: f64,
    pub(crate) branch_walk_ms: f64,
    pub(crate) memo_hit_clone_ms: f64,
    pub(crate) fragment_build_ms: f64,
    pub(crate) epsilon_link_ms: f64,
    pub(crate) bundle_profile_total_ms: f64,
    pub(crate) bundle_profile_build_group_dfas_ms: f64,
    pub(crate) bundle_profile_union_groups_ms: f64,
    pub(crate) bundle_profile_determinize_ms: f64,
    pub(crate) bundle_profile_minimize_ms: f64,
    pub(crate) bundle_profile_dwa_to_nwa_ms: f64,
    pub(crate) memo_hits: usize,
    pub(crate) memo_misses: usize,
    pub(crate) bundle_cache_builds: usize,
    pub(crate) bundle_profile_result_dwa_states: usize,
    pub(crate) bundle_profile_result_dwa_transitions: usize,
    pub(crate) bundle_profile_result_nwa_states: usize,
    pub(crate) bundle_profile_result_nwa_transitions: usize,
    pub(crate) epsilon_edges_added: usize,
    pub(crate) fragment_start_states_total: usize,
}

impl ParserDwaComposeDetailProfile {
    pub(crate) fn accumulate_bundle_profile(&mut self, bundle_profile: &BundleBuildProfile) {
        self.bundle_profile_total_ms += bundle_profile.total_ms;
        self.bundle_profile_build_group_dfas_ms += bundle_profile.build_group_dfas_ms;
        self.bundle_profile_union_groups_ms += bundle_profile.union_groups_ms;
        self.bundle_profile_determinize_ms += bundle_profile.determinize_bundle_ms;
        self.bundle_profile_minimize_ms += bundle_profile.minimize_ms;
        self.bundle_profile_dwa_to_nwa_ms += bundle_profile.dwa_to_nwa_ms;
        self.bundle_profile_result_dwa_states += bundle_profile.result_dwa_states;
        self.bundle_profile_result_dwa_transitions += bundle_profile.result_dwa_transitions;
        self.bundle_profile_result_nwa_states += bundle_profile.result_nwa_states;
        self.bundle_profile_result_nwa_transitions += bundle_profile.result_nwa_transitions;
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ParserDwaProfile {
    pub(crate) terminal_dwa_states: usize,
    pub(crate) terminal_dwa_transitions: usize,
    pub(crate) terminal_dwa_interned_ranges: usize,
    pub(crate) parser_nwa_built: bool,
    pub(crate) parser_nwa_states: usize,
    pub(crate) parser_nwa_start_states: usize,
    pub(crate) pre_minimize_states: usize,
    pub(crate) pre_minimize_transitions: usize,
    pub(crate) post_minimize_states: usize,
    pub(crate) post_minimize_transitions: usize,
    pub(crate) minimize_skipped: bool,
    pub(crate) state_prep_ms: f64,
    pub(crate) compose_state_ms: f64,
    pub(crate) parser_nwa_build_ms: f64,
    pub(crate) resolve_negative_ms: f64,
    pub(crate) support_determinize_ms: f64,
    pub(crate) possible_outgoing_ms: f64,
    pub(crate) default_opt_ms: f64,
    pub(crate) subtract_final_ms: f64,
    pub(crate) fallback_determinize_ms: f64,
    pub(crate) minimize_ms: f64,
    pub(crate) total_ms: f64,
}

impl ParserDwaProfile {
    pub(crate) fn empty(
        terminal_dwa_states: usize,
        terminal_dwa_transitions: usize,
        terminal_dwa_interned_ranges: usize,
        minimize_skipped: bool,
        total_ms: f64,
    ) -> Self {
        Self {
            terminal_dwa_states,
            terminal_dwa_transitions,
            terminal_dwa_interned_ranges,
            parser_nwa_built: false,
            minimize_skipped,
            total_ms,
            ..Self::default()
        }
    }

    pub(crate) fn emit_detail(&self) {
        eprintln!(
            "[glrmask/profile][parser_dwa_detail] terminal_dwa_states={} terminal_dwa_transitions={} terminal_dwa_interned_ranges={} parser_nwa_built={} parser_nwa_states={} parser_nwa_start_states={} pre_minimize_states={} pre_minimize_transitions={} post_minimize_states={} post_minimize_transitions={} minimize_skipped={} state_prep_ms={:.3} compose_state_ms={:.3} parser_nwa_build_ms={:.3} resolve_negative_ms={:.3} support_determinize_ms={:.3} possible_outgoing_ms={:.3} default_opt_ms={:.3} subtract_final_ms={:.3} fallback_determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            self.terminal_dwa_states,
            self.terminal_dwa_transitions,
            self.terminal_dwa_interned_ranges,
            self.parser_nwa_built,
            self.parser_nwa_states,
            self.parser_nwa_start_states,
            self.pre_minimize_states,
            self.pre_minimize_transitions,
            self.post_minimize_states,
            self.post_minimize_transitions,
            self.minimize_skipped,
            self.state_prep_ms,
            self.compose_state_ms,
            self.parser_nwa_build_ms,
            self.resolve_negative_ms,
            self.support_determinize_ms,
            self.possible_outgoing_ms,
            self.default_opt_ms,
            self.subtract_final_ms,
            self.fallback_determinize_ms,
            self.minimize_ms,
            self.total_ms,
        );
    }
}

pub(crate) fn emit_parser_bundle_profile(bundle_id: usize, bundle_profile: &BundleBuildProfile) {
    eprintln!(
        "[glrmask/profile][parser_bundle] bundle_id={} terminals={} weight_groups={} single_entry_weights={} single_tsid_weights={} total_weight_outer_ranges={} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} det_pop_ms={:.3} det_alive_ms={:.3} det_final_ms={:.3} det_collect_labels_ms={:.3} det_next_state_ms={:.3} det_edge_weight_ms={:.3} det_lookup_ms={:.3} det_add_transition_ms={:.3} det_states={} det_labels={} det_transitions={} det_edge_subset_total={} det_edge_subset_max={} det_edge_cache_hits={} det_edge_cache_misses={} minimize_ms={:.3} minimize_skipped={} dwa_to_nwa_ms={:.3} total_ms={:.3} result_dwa_states={} result_dwa_transitions={} result_nwa_states={} result_nwa_transitions={}",
        bundle_id,
        bundle_profile.input_terminals,
        bundle_profile.weight_groups,
        bundle_profile.single_entry_weights,
        bundle_profile.single_tsid_weights,
        bundle_profile.total_weight_outer_ranges,
        bundle_profile.build_group_dfas_ms,
        bundle_profile.union_groups_ms,
        bundle_profile.determinize_bundle_ms,
        bundle_profile.determinize_pop_state_ms,
        bundle_profile.determinize_alive_groups_ms,
        bundle_profile.determinize_final_weight_ms,
        bundle_profile.determinize_collect_labels_ms,
        bundle_profile.determinize_next_state_ms,
        bundle_profile.determinize_edge_weight_ms,
        bundle_profile.determinize_state_lookup_ms,
        bundle_profile.determinize_add_transition_ms,
        bundle_profile.determinize_states_visited,
        bundle_profile.determinize_labels_processed,
        bundle_profile.determinize_transitions_added,
        bundle_profile.determinize_edge_subset_total,
        bundle_profile.determinize_edge_subset_max,
        bundle_profile.determinize_edge_cache_hits,
        bundle_profile.determinize_edge_cache_misses,
        bundle_profile.minimize_ms,
        bundle_profile.minimize_skipped,
        bundle_profile.dwa_to_nwa_ms,
        bundle_profile.total_ms,
        bundle_profile.result_dwa_states,
        bundle_profile.result_dwa_transitions,
        bundle_profile.result_nwa_states,
        bundle_profile.result_nwa_transitions,
    );
}

pub(crate) fn emit_parser_dwa_compose_profiles(detail: &ParserDwaComposeDetailProfile) {
    eprintln!(
        "[glrmask/profile][parser_dwa_compose] total_states={} productive_states={} total_branches={} productive_branches={} unique_bundles={} accepting_bundles={} state_init_ms={:.3} branch_walk_ms={:.3} memo_hit_clone_ms={:.3} fragment_build_ms={:.3} epsilon_link_ms={:.3} memo_hits={} memo_misses={} bundle_cache_builds={} epsilon_edges_added={} fragment_start_states_total={}",
        detail.total_states,
        detail.productive_states,
        detail.total_branches,
        detail.productive_branches,
        detail.unique_bundles,
        detail.accepting_bundles,
        detail.state_init_ms,
        detail.branch_walk_ms,
        detail.memo_hit_clone_ms,
        detail.fragment_build_ms,
        detail.epsilon_link_ms,
        detail.memo_hits,
        detail.memo_misses,
        detail.bundle_cache_builds,
        detail.epsilon_edges_added,
        detail.fragment_start_states_total,
    );
    eprintln!(
        "[glrmask/profile][parser_dwa_compose_bundles] bundle_cache_builds={} bundle_profile_total_ms={:.3} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} minimize_ms={:.3} dwa_to_nwa_ms={:.3} result_dwa_states_total={} result_dwa_transitions_total={} result_nwa_states_total={} result_nwa_transitions_total={}",
        detail.bundle_cache_builds,
        detail.bundle_profile_total_ms,
        detail.bundle_profile_build_group_dfas_ms,
        detail.bundle_profile_union_groups_ms,
        detail.bundle_profile_determinize_ms,
        detail.bundle_profile_minimize_ms,
        detail.bundle_profile_dwa_to_nwa_ms,
        detail.bundle_profile_result_dwa_states,
        detail.bundle_profile_result_dwa_transitions,
        detail.bundle_profile_result_nwa_states,
        detail.bundle_profile_result_nwa_transitions,
    );
}
