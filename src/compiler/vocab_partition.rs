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
    build_vocab_partition_from_static_id_maps,
    build_vocab_equivalence_partition_with_precomputed_global_max_length,
    l1,
    types::TerminalColoring,
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
        // Match Static's compile DAG: grammar analysis is independent of the
        // entire tokenizer -> flat-transition -> max-length chain. Running
        // those lanes concurrently avoids serializing grammar work in front of
        // an otherwise lexer-only API.
        let (
            (
                tokenizer,
                initial_state_map,
                partition_local_synthesis_plan,
                direct_mask_tokenizer,
                flat_trans,
                global_max_length_state_map,
                context_ms,
                flat_ms,
                max_length_ms,
            ),
            (analyzed_grammar, disallowed_follows, analysis_ms),
        ) = crate::compiler::macro_join(
            "vocab_partition_frontend",
            || {
                let context_started = std::time::Instant::now();
                let (
                    tokenizer,
                    initial_state_map,
                    partition_local_synthesis_plan,
                    direct_mask_tokenizer,
                ) =
                    build_vocab_partition_compile_context(&prepared_grammar, vocab);
                if std::env::var_os("GLRMASK_PROFILE_VOCAB_PROJECTED_QUOTIENTS").is_some() {
                    let projected_started = std::time::Instant::now();
                    let quotients = tokenizer
                        .build_shared_component_terminal_projected_quotients(256);
                    eprintln!(
                        "[glrmask/profile][vocab_projected_quotients] retained={} elapsed_ms={:.3}",
                        quotients.len(),
                        projected_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                let context_ms = context_started.elapsed().as_secs_f64() * 1000.0;

                let flat_started = std::time::Instant::now();
                let flat_trans: Arc<[u32]> =
                    Arc::from(l1::build_flat_transition_table(&tokenizer));
                let flat_ms = flat_started.elapsed().as_secs_f64() * 1000.0;

                let max_length_started = std::time::Instant::now();
                let global_max_length_state_map = build_global_max_length_state_map_with_initial(
                    &tokenizer,
                    vocab,
                    &flat_trans,
                    initial_state_map.as_ref(),
                );
                let max_length_ms = max_length_started.elapsed().as_secs_f64() * 1000.0;
                (
                    tokenizer,
                    initial_state_map,
                    partition_local_synthesis_plan,
                    direct_mask_tokenizer,
                    flat_trans,
                    global_max_length_state_map,
                    context_ms,
                    flat_ms,
                    max_length_ms,
                )
            },
            || {
                let analysis_started = std::time::Instant::now();
                let analyzed_grammar = AnalyzedGrammar::from_grammar_def(&prepared_grammar);
                let disallowed_follows = compute_disallowed_follows(&analyzed_grammar);
                let analysis_ms = analysis_started.elapsed().as_secs_f64() * 1000.0;
                (analyzed_grammar, disallowed_follows, analysis_ms)
            },
        );

        if std::env::var_os("GLRMASK_PROFILE_TERMINAL_OBSERVATION_PARTITIONS").is_some() {
            let started = std::time::Instant::now();
            let mut total_configs = 0usize;
            let mut total_classes = 0usize;
            for terminal in 0..tokenizer.num_terminals() {
                let terminal_started = std::time::Instant::now();
                match tokenizer.exact_terminal_observation_partition(
                    terminal,
                    1_000_000,
                    100_000_000,
                ) {
                    Some((classes, configs, rounds)) => {
                        let class_count = classes.iter().copied().max().unwrap_or(0) as usize;
                        total_configs += configs;
                        total_classes += class_count;
                        eprintln!(
                            "[glrmask/profile][terminal_observation_partition] terminal={} configs={} classes={} rounds={} ms={:.3}",
                            terminal,
                            configs,
                            class_count,
                            rounds,
                            terminal_started.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    None => eprintln!(
                        "[glrmask/profile][terminal_observation_partition] terminal={} declined=true ms={:.3}",
                        terminal,
                        terminal_started.elapsed().as_secs_f64() * 1000.0,
                    ),
                }
            }
            eprintln!(
                "[glrmask/profile][terminal_observation_partitions] terminals={} total_configs={} summed_classes={} total_ms={:.3}",
                tokenizer.num_terminals(),
                total_configs,
                total_classes,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let partition_started = std::time::Instant::now();
        // The dedicated vocab-only relation avoids expensive L2 proofs and
        // terminal-DWA artifacts, which is a substantial win once the lexer
        // has enough state/terminal interaction. On small lexers its own
        // classification/setup cost dominates, and the existing id-map-only
        // Static path is both faster and materially closer to Static's final
        // quotient. Select by the size of the observable lexer topology rather
        // than by schema/source kind.
        const DEDICATED_TOPOLOGY_MIN: usize = 100_000;
        let topology = (tokenizer.num_states() as usize)
            .saturating_mul(analyzed_grammar.num_terminals as usize);
        let use_dedicated = std::env::var("GLRMASK_VOCAB_PARTITION_DEDICATED")
            .ok()
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(direct_mask_tokenizer || topology >= DEDICATED_TOPOLOGY_MIN);
        let result = if use_dedicated {
            build_vocab_equivalence_partition_with_precomputed_global_max_length(
                &tokenizer,
                vocab,
                prepared_grammar.ignore_terminal,
                &analyzed_grammar,
                &disallowed_follows,
                Arc::clone(&flat_trans),
                &global_max_length_state_map,
                partition_local_synthesis_plan.as_deref(),
            )
        } else {
            let terminal_coloring =
                TerminalColoring::identity(analyzed_grammar.num_terminals as usize);
            build_vocab_partition_from_static_id_maps(
                &tokenizer,
                vocab,
                &terminal_coloring,
                false,
                prepared_grammar.ignore_terminal,
                &analyzed_grammar,
                &disallowed_follows,
                None,
                Arc::clone(&flat_trans),
                &global_max_length_state_map,
                None,
                None,
                partition_local_synthesis_plan.as_deref(),
                None,
            )
        };
        let partition_ms = partition_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_partition_stages] grammar_prepare_ms={grammar_prepare_ms:.3} context_ms={context_ms:.3} analysis_ms={analysis_ms:.3} flat_ms={flat_ms:.3} max_length_ms={max_length_ms:.3} partition_ms={partition_ms:.3} tokenizer_states={} terminals={} topology={} direct_mask={} dedicated={}",
                tokenizer.num_states(),
                prepared_grammar.terminals.len(),
                topology,
                direct_mask_tokenizer,
                use_dedicated,
            );
        }
        result
    })
}
