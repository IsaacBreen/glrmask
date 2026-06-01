//! Parser-DWA construction and shared-coordinate reconciliation.
//!
//! The Terminal DWA, Parser DWA, and scan-relation CanMatch artifact are all
//! weighted by sets of `(lexer-state, token)` pairs.  Performance requires each
//! weighted artifact to quotient those pairs aggressively, but correctness
//! requires the final runtime artifacts to agree on one coordinate system.  This
//! module owns that final reconciliation.

use std::time::Instant;

use crate::Vocab;
use crate::compile::options::{
    compact_can_match_before_reconcile_enabled,
    dwa_can_match_mode,
};
use crate::compile::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compile::pipeline::context::{
    GrammarAnalysisOutput,
    ReconciledArtifacts,
    TerminalAndScanOutput,
};
use crate::compile::template_dfa::Templates;
use crate::compile::pipeline::counts::{
    interned_range_count_for_artifact,
    joint_interned_range_count_for_artifacts,
};
use crate::compile::profiling::{CompilePhaseProfile, compile_profile_enabled, elapsed_ms};
use crate::compile::mapped_artifact::MappedArtifact;

/// Reconcile Terminal DWA, Parser DWA, and scan relation into one internal ID space.
pub(crate) fn reconcile_and_build_parser_dwa(
    analysis: &GrammarAnalysisOutput,
    terminal_and_scan: TerminalAndScanOutput,
    templates: Templates,
    vocab: &Vocab,
    profile: &mut CompilePhaseProfile,
) -> ReconciledArtifacts {
    let mut terminal_dwa = terminal_and_scan.terminal_dwa;
    let mut can_match = terminal_and_scan.scan_relation.mapped_can_match;
    let scan_relation_profile = terminal_and_scan.scan_relation.profile;
    let dwa_can_match_mode = dwa_can_match_mode();

    let mut shared_id_reconcile_ms = 0.0;
    if compact_can_match_before_reconcile_enabled() {
        let compact_started_at = Instant::now();
        if compile_profile_enabled() {
            let _ = can_match.compact_dimensions_fast_with_stats();
        } else {
            let _ = can_match.compact_dimensions_fast();
        }
        profile.compact_ms += elapsed_ms(compact_started_at);
    }

    let terminal_dwa_interned_ranges_before_can_match_reconcile =
        interned_range_count_for_artifact(terminal_dwa.artifact_mut());
    let can_match_interned_ranges_before_can_match_reconcile =
        interned_range_count_for_artifact(can_match.artifact_mut());
    let terminal_can_match_joint_interned_ranges_before_reconcile =
        joint_interned_range_count_for_artifacts(terminal_dwa.artifact_mut(), can_match.artifact_mut());

    let mut internal_ids = terminal_dwa.id_map().clone();
    let (mut parser_dwa, parser_dwa_ms) = if dwa_can_match_mode.does_terminal_compact() {
        let shared_id_reconcile_started_at = Instant::now();
        let mut terminal_can_match_pair = MappedArtifact::from((terminal_dwa, can_match));
        shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);

        if dwa_can_match_mode.does_parser_compact() {
            let compact_plan_started_at = Instant::now();
            let terminal_compaction_plan = terminal_can_match_pair.plan_dimensions_compaction(true, true);
            profile.compact_ms += elapsed_ms(compact_plan_started_at);

            let ((terminal_dwa_artifact, can_match_artifact), compacted_ids) =
                terminal_can_match_pair.into_parts();
            terminal_dwa = MappedArtifact::new(terminal_dwa_artifact, compacted_ids.clone());
            can_match = MappedArtifact::new(can_match_artifact, compacted_ids.clone());

            let terminal_apply_started_at = Instant::now();
            terminal_dwa.apply_compaction_plan(&terminal_compaction_plan);
            profile.compact_ms += elapsed_ms(terminal_apply_started_at);
            internal_ids = terminal_dwa.id_map().clone();

            let ((parser_dwa, parser_dwa_ms), can_match_compact_ms) = rayon::join(
                || {
                    let parser_dwa_started_at = Instant::now();
                    let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                        &analysis.table,
                        &analysis.analyzed_grammar,
                        terminal_dwa.artifact(),
                        templates,
                        vocab,
                        &internal_ids,
                    );
                    (parser_dwa, elapsed_ms(parser_dwa_started_at))
                },
                || {
                    let can_match_apply_started_at = Instant::now();
                    can_match.apply_compaction_plan(&terminal_compaction_plan);
                    elapsed_ms(can_match_apply_started_at)
                },
            );
            if can_match_compact_ms > parser_dwa_ms {
                profile.compact_ms += can_match_compact_ms - parser_dwa_ms;
            }
            (MappedArtifact::new(parser_dwa, internal_ids.clone()), parser_dwa_ms)
        } else {
            let pre_compact_ids = terminal_can_match_pair.id_map().clone();
            let ((terminal_compaction_plan, terminal_compaction_plan_ms), (parser_dwa, parser_dwa_ms)) =
                rayon::join(
                    || {
                        let compact_started_at = Instant::now();
                        let plan = terminal_can_match_pair.plan_dimensions_compaction(true, true);
                        (plan, elapsed_ms(compact_started_at))
                    },
                    || {
                        let parser_dwa_started_at = Instant::now();
                        let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                            &analysis.table,
                            &analysis.analyzed_grammar,
                            &terminal_can_match_pair.artifact().0,
                            templates,
                            vocab,
                            &pre_compact_ids,
                        );
                        (parser_dwa, elapsed_ms(parser_dwa_started_at))
                    },
                );

            let compact_apply_started_at = Instant::now();
            terminal_can_match_pair.apply_compaction_plan(&terminal_compaction_plan);
            let mut parser_dwa = MappedArtifact::new(parser_dwa, pre_compact_ids);
            parser_dwa.apply_compaction_plan(&terminal_compaction_plan);
            profile.compact_ms += terminal_compaction_plan_ms + elapsed_ms(compact_apply_started_at);

            let ((terminal_dwa_artifact, can_match_artifact), compacted_ids) =
                terminal_can_match_pair.into_parts();
            terminal_dwa = MappedArtifact::new(terminal_dwa_artifact, compacted_ids.clone());
            can_match = MappedArtifact::new(can_match_artifact, compacted_ids.clone());
            internal_ids = compacted_ids.clone();
            (parser_dwa, parser_dwa_ms)
        }
    } else {
        if dwa_can_match_mode.does_terminal_reconcile() {
            let shared_id_reconcile_started_at = Instant::now();
            internal_ids = terminal_dwa.reconcile_with(&mut can_match);
            shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
        }
        let parser_dwa_started_at = Instant::now();
        let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
            &analysis.table,
            &analysis.analyzed_grammar,
            terminal_dwa.artifact(),
            templates,
            vocab,
            &internal_ids,
        );
        let parser_dwa_ms = elapsed_ms(parser_dwa_started_at);
        (MappedArtifact::new(parser_dwa, internal_ids.clone()), parser_dwa_ms)
    };

    let terminal_can_match_joint_interned_ranges =
        joint_interned_range_count_for_artifacts(terminal_dwa.artifact_mut(), can_match.artifact_mut());

    if dwa_can_match_mode.does_terminal_reconcile() {
        if dwa_can_match_mode.does_parser_compact() {
            let shared_id_reconcile_started_at = Instant::now();
            let mut parser_can_match_pair = MappedArtifact::from((parser_dwa, can_match));
            shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
            let compact_started_at = Instant::now();
            parser_can_match_pair.compact_dimensions();
            profile.compact_ms += elapsed_ms(compact_started_at);
            let ((parser_dwa_artifact, can_match_artifact), compacted_ids) =
                parser_can_match_pair.into_parts();
            parser_dwa = MappedArtifact::new(parser_dwa_artifact, compacted_ids.clone());
            can_match = MappedArtifact::new(can_match_artifact, compacted_ids.clone());
            internal_ids = compacted_ids;
        }
    } else {
        let shared_id_reconcile_started_at = Instant::now();
        let mut parser_can_match_pair = MappedArtifact::from((parser_dwa, can_match));
        shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
        if dwa_can_match_mode.does_parser_compact() {
            let compact_started_at = Instant::now();
            parser_can_match_pair.compact_dimensions();
            profile.compact_ms += elapsed_ms(compact_started_at);
        }
        let ((parser_dwa_artifact, can_match_artifact), reconciled_ids) =
            parser_can_match_pair.into_parts();
        parser_dwa = MappedArtifact::new(parser_dwa_artifact, reconciled_ids.clone());
        can_match = MappedArtifact::new(can_match_artifact, reconciled_ids.clone());
        internal_ids = reconciled_ids;
    }

    let parser_dwa_interned_ranges = parser_dwa.artifact().stats().interned_ranges;
    let (can_match_interned_ranges, parser_can_match_joint_interned_ranges) = {
        let (parser_dwa_artifact, _) = parser_dwa.parts_mut();
        let (can_match_artifact, _) = can_match.parts_mut();
        (
            interned_range_count_for_artifact(can_match_artifact),
            joint_interned_range_count_for_artifacts(parser_dwa_artifact, can_match_artifact),
        )
    };

    profile.parser_dwa_ms = parser_dwa_ms;
    profile.scan_relation_collect_ms = scan_relation_profile.scan_relation_collect_ms;
    profile.can_match_materialize_ms = scan_relation_profile.scan_relation_vocab_ms;
    profile.shared_id_reconcile_ms = shared_id_reconcile_ms;
    profile.can_match_pipeline_ms =
        scan_relation_profile.scan_relation_collect_ms
            + scan_relation_profile.scan_relation_vocab_ms
            + shared_id_reconcile_ms;
    profile.terminal_dwa_interned_ranges_before_can_match_reconcile =
        terminal_dwa_interned_ranges_before_can_match_reconcile;
    profile.can_match_interned_ranges_before_can_match_reconcile =
        can_match_interned_ranges_before_can_match_reconcile;
    profile.terminal_can_match_joint_interned_ranges_before_reconcile =
        terminal_can_match_joint_interned_ranges_before_reconcile;
    profile.terminal_can_match_joint_interned_ranges = terminal_can_match_joint_interned_ranges;
    profile.parser_dwa_interned_ranges = parser_dwa_interned_ranges;
    profile.can_match_interned_ranges = can_match_interned_ranges;
    profile.parser_can_match_joint_interned_ranges = parser_can_match_joint_interned_ranges;

    ReconciledArtifacts {
        parser_dwa: parser_dwa.into_artifact(),
        can_match: can_match.into_artifact(),
        internal_ids,
        parser_dwa_interned_ranges,
        can_match_interned_ranges,
        parser_can_match_joint_interned_ranges,
        terminal_dwa_interned_ranges_before_can_match_reconcile,
        can_match_interned_ranges_before_can_match_reconcile,
        terminal_can_match_joint_interned_ranges_before_reconcile,
        terminal_can_match_joint_interned_ranges,
    }
}
