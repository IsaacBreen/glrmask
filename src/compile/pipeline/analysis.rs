//! Grammar-analysis phase group.
//!
//! This phase group computes the parser-side facts that are independent of the
//! vocabulary quotient.  It may run tokenizer construction and grammar/table
//! analysis in parallel, but the output is still a single typed object.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::Vocab;
use crate::compile::pipeline::context::GrammarAnalysisOutput;
use crate::compile::profiling::{CompilePhaseProfile, elapsed_ms};
use crate::compile::terminal_dwa::grammar_helpers::{
    compute_ever_allowed_follows,
    compute_terminal_coloring,
};
use crate::compile::tokenizer::build_tokenizer;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::parser::glr::table::GLRTable;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::GrammarDef;

/// Build tokenizer, analyzed grammar, GLR table, terminal colors, and disallowed follows.
pub(crate) fn build_grammar_analysis(
    prepared_grammar: &GrammarDef,
    _vocab: &Vocab,
    profile: &mut CompilePhaseProfile,
) -> GrammarAnalysisOutput {
    let analysis_started_at = Instant::now();
    let (
        (mut tokenizer, tokenizer_build_ms),
        (
            analyzed_grammar,
            analyze_grammar_ms,
            table,
            glr_table_ms,
            terminal_coloring,
            terminal_coloring_ms,
            disallowed_follows,
            disallowed_follows_ms,
        ),
    ) = rayon::join(
        || {
            let tok_started = Instant::now();
            let mut tokenizer = build_tokenizer(prepared_grammar);
            tokenizer.isolate_start_state_and_drain_nullable_terminals();
            (tokenizer, elapsed_ms(tok_started))
        },
        || {
            let analyze_grammar_started_at = Instant::now();
            let analyzed_grammar = AnalyzedGrammar::from_grammar_def(prepared_grammar);
            let analyze_grammar_ms = elapsed_ms(analyze_grammar_started_at);

            if let Err(message) = analyzed_grammar.check_table_build_normal_form() {
                panic!("[glrmask] grammar precondition violations:\n{}", message);
            }

            let table_started_at = Instant::now();
            let table = GLRTable::build(&analyzed_grammar);
            let glr_table_ms = elapsed_ms(table_started_at);

            let terminal_coloring_started_at = Instant::now();
            let terminal_coloring = compute_terminal_coloring(&table);
            let terminal_coloring_ms = elapsed_ms(terminal_coloring_started_at);

            let disallowed_follows_started_at = Instant::now();
            let disallowed_follows = compute_disallowed_follows(&analyzed_grammar);
            let disallowed_follows_ms = elapsed_ms(disallowed_follows_started_at);

            (
                analyzed_grammar,
                analyze_grammar_ms,
                table,
                glr_table_ms,
                terminal_coloring,
                terminal_coloring_ms,
                disallowed_follows,
                disallowed_follows_ms,
            )
        },
    );

    profile.tokenizer_build_ms = tokenizer_build_ms;
    profile.analyze_grammar_ms = analyze_grammar_ms;
    profile.glr_table_ms = glr_table_ms;
    profile.terminal_coloring_ms = terminal_coloring_ms;
    profile.disallowed_follows_ms = disallowed_follows_ms;
    profile.analysis_wall_ms = elapsed_ms(analysis_started_at);

    GrammarAnalysisOutput {
        tokenizer,
        analyzed_grammar,
        table,
        terminal_coloring,
        disallowed_follows,
    }
}

/// Compute terminals that may not legally follow each completed terminal.
pub(crate) fn compute_disallowed_follows(grammar: &AnalyzedGrammar) -> BTreeMap<u32, BitSet> {
    let ever_allowed = compute_ever_allowed_follows(grammar);
    let num_terminals = grammar.num_terminals as usize;
    let mut disallowed_by_terminal = BTreeMap::new();

    for (terminal_id, allowed) in ever_allowed.iter().enumerate() {
        let allowed_set: BTreeSet<u32> = allowed.iter().copied().collect();
        let mut disallowed = BitSet::new(num_terminals);

        for other in 0..num_terminals {
            if !allowed_set.contains(&(other as u32)) {
                disallowed.set(other);
            }
        }

        if !disallowed.is_zero() {
            disallowed_by_terminal.insert(terminal_id as u32, disallowed);
        }
    }

    disallowed_by_terminal
}
