use std::sync::Arc;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::pipeline::{
    build_vocab_partition_compile_context, compute_disallowed_follows, run_with_compile_thread_pool,
};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::{
    build_global_max_length_state_map_with_initial,
    build_vocab_equivalence_partition_with_precomputed_global_max_length,
    l1,
};
use crate::grammar::flat::GrammarDef;

/// Compile only the grammar-dependent vocabulary equivalence partition.
///
/// This deliberately stops before terminal-DWA, possible-matches, parser-DWA,
/// and runtime-artifact construction. The returned map may be finer than the
/// final Static token quotient, but never coarser than the exact relations used
/// by this fast path.
pub(crate) fn compile_vocab_partition_owned(
    grammar: GrammarDef,
    vocab: &Vocab,
) -> ManyToOneIdMap {
    // Pure vocabulary artifacts are reusable across every grammar using this
    // Vocab. Populate them once so grammar-dependent latency does not repeatedly
    // pay radix/reverse-trie construction.
    crate::compiler::stages::id_map_and_terminal_dwa::prepare_vocab_for_terminal_dwa(vocab);

    let profile = crate::compiler::compile::compile_profile_enabled();
    let prepare_started = std::time::Instant::now();
    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    let grammar_prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;
    run_with_compile_thread_pool(|| {
        let context_started = std::time::Instant::now();
        let (tokenizer, initial_state_map, partition_local_synthesis_plan) =
            build_vocab_partition_compile_context(&prepared_grammar, vocab);
        let context_ms = context_started.elapsed().as_secs_f64() * 1000.0;

        let analysis_started = std::time::Instant::now();
        let analyzed_grammar = AnalyzedGrammar::from_grammar_def(&prepared_grammar);
        let disallowed_follows = compute_disallowed_follows(&analyzed_grammar);
        let analysis_ms = analysis_started.elapsed().as_secs_f64() * 1000.0;

        let flat_started = std::time::Instant::now();
        let flat_trans: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(&tokenizer));
        let flat_ms = flat_started.elapsed().as_secs_f64() * 1000.0;

        let max_length_started = std::time::Instant::now();
        let global_max_length_state_map = build_global_max_length_state_map_with_initial(
            &tokenizer,
            vocab,
            &flat_trans,
            initial_state_map.as_ref(),
        );
        let max_length_ms = max_length_started.elapsed().as_secs_f64() * 1000.0;

        let partition_started = std::time::Instant::now();
        let result = build_vocab_equivalence_partition_with_precomputed_global_max_length(
            &tokenizer,
            vocab,
            prepared_grammar.ignore_terminal,
            &analyzed_grammar,
            &disallowed_follows,
            flat_trans,
            &global_max_length_state_map,
            partition_local_synthesis_plan.as_deref(),
        );
        let partition_ms = partition_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_partition_stages] grammar_prepare_ms={grammar_prepare_ms:.3} context_ms={context_ms:.3} analysis_ms={analysis_ms:.3} flat_ms={flat_ms:.3} max_length_ms={max_length_ms:.3} partition_ms={partition_ms:.3} tokenizer_states={} terminals={}",
                tokenizer.num_states(),
                prepared_grammar.terminals.len(),
            );
        }
        result
    })
}
