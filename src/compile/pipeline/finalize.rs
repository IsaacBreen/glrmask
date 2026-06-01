//! Runtime-artifact finalization.
//!
//! This phase is the only place in the compile graph that should know the full
//! layout of [`crate::runtime::Constraint`].  Earlier phases produce mathematical
//! artifacts: tokenizer, table, Parser DWA, CanMatch, and shared internal ID
//! maps.  This phase packages them into the runtime object and rebuilds derived
//! caches.

use std::sync::Arc;
use std::time::Instant;

use crate::Vocab;
use crate::compile::pipeline::context::{
    GrammarAnalysisOutput,
    ReconciledArtifacts,
};
use crate::compile::profiling::{CompilePhaseProfile, elapsed_ms};
use crate::compile::scan_relation::build_internal_token_bytes_from_groups;
use crate::grammar::flat::GrammarDef;
use crate::runtime::{CompiledArtifactParts, Constraint, TemplateDfasByTerminal};

/// Assemble the runtime constraint and rebuild runtime caches.
pub(crate) fn finalize_runtime_constraint(
    prepared_grammar: GrammarDef,
    analysis: GrammarAnalysisOutput,
    reconciled: ReconciledArtifacts,
    template_dfas_by_terminal: TemplateDfasByTerminal,
    vocab: &Vocab,
    profile: &mut CompilePhaseProfile,
) -> Constraint {
    let internal_token_bytes_started_at = Instant::now();
    let internal_token_bytes = build_internal_token_bytes_from_groups(
        vocab,
        &reconciled.internal_ids.vocab_tokens.internal_to_originals,
    );
    profile.internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

    let finalize_started_at = Instant::now();
    let token_bytes = Arc::clone(&vocab.entries);
    let mut constraint = Constraint::from_compiled_parts(CompiledArtifactParts {
        parser_dwa: reconciled.parser_dwa,
        table: analysis.table,
        terminal_display_names: analysis.analyzed_grammar.terminal_display_names.clone(),
        tokenizer: analysis.tokenizer,
        ignore_terminal: prepared_grammar.ignore_terminal,
        can_match: reconciled.can_match,
        state_to_internal_tsid: reconciled.internal_ids.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: reconciled.internal_ids.tokenizer_states.internal_to_originals_vecs(),
        template_dfas_by_terminal,
        original_token_to_internal: reconciled.internal_ids.vocab_tokens.original_to_internal.clone(),
        internal_token_to_tokens: reconciled.internal_ids.vocab_tokens.internal_to_originals_vecs(),
        eos_token_id: vocab.eos_token_id,
        token_bytes,
        internal_token_bytes,
    });

    constraint.rebuild_runtime_caches();
    profile.finalize_ms = elapsed_ms(finalize_started_at);
    constraint
}
