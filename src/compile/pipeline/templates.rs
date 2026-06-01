//! Template-DFA phase.
//!
//! Templates are stack-effect recognizers.  They are deliberately not defined as
//! an LR-only object: they summarize parser stack effects in the form required by
//! downstream weighted automata and commit-time acceleration.

use std::sync::Arc;
use std::time::Instant;

use crate::compile::pipeline::context::{GrammarAnalysisOutput, TemplateOutput};
use crate::compile::profiling::{CompilePhaseProfile, elapsed_ms, emit_template_profile_summary};
use crate::compile::template_dfa::Templates;
use crate::compile::template_dfa::characterize::characterize_terminals_profiled;
use crate::compile::template_dfa::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    split_commit_template_dfas,
};

/// Build parser stack-effect templates and commit-specialized template DFAs.
pub(crate) fn build_templates(
    analysis: &GrammarAnalysisOutput,
    profile: &mut CompilePhaseProfile,
) -> TemplateOutput {
    let templates_started_at = Instant::now();
    let (characterizations, characterization_profile) =
        characterize_terminals_profiled(&analysis.table, &analysis.analyzed_grammar);
    let (templates, template_profile) = Templates::from_characterizations_profiled(&characterizations);
    let mut template_dfas_by_terminal = vec![None; analysis.analyzed_grammar.num_terminals as usize];

    for (&terminal, dfa) in &templates.by_terminal {
        if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
            let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
            let split_commit_dfas = split_commit_template_dfas(&commit_dfa);
            *slot = Some(Arc::new(split_commit_dfas));
        }
    }

    emit_template_profile_summary(&characterization_profile, &template_profile);
    profile.templates_ms = elapsed_ms(templates_started_at);

    TemplateOutput {
        templates,
        template_dfas_by_terminal,
    }
}
