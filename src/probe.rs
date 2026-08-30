//! Experimental partial-compile / probe path for vocabulary equivalence analysis.
//!
//! Preserves production static constraint preprocessing, tokenizer planning,
//! determinization, grammar analysis, and classification choices, but stops
//! immediately after exact partition/branch vocabulary equivalence maps and
//! computes their global common-refinement merge.
//!
//! Fully instrumented with wall time and CLOCK_THREAD_CPUTIME_ID CPU time.

#![allow(unused_imports, dead_code, unused_variables)]

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{GLRTable, GlrTableConstruction};
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::pipeline::{
    build_static_compile_tokenizer_for_vocab_equiv_probe,
    plan_synthetic_tokenizer,
    compute_disallowed_follows_from_ever,
};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::{
    build_char_type_sub_vocabs,
    automatic_p2_overflow_threshold,
    use_global_single_terminal_l1,
    split_l2p_vocab_enabled,
    PartitionLocalSynthesisPlan,
    build_partition_local_tokenizer,
    partition_local_synthesis_selected,
    CpuTimer,
    classify::{
        self,
        classify_terminal_path_lengths,
        split_vocab_for_active_l2p_terminals,
    },
    grammar_helpers::{
        compute_allowed_follow_sets,
        ignore_transparent_disallowed_follows,
    },
    l1::{
        self,
        build_projected_vocab_equivalence,
        implementations::BuildInput,
    },
    l2p::{
        self,
        analyze_vocab_equivalence_with_group_filter,
        equivalence_analysis::disallowed_follows::normalize_disallowed_follows,
    },
    types::{TerminalColoring, TerminalPathLength, compile_profile_join},
    build_global_max_length_state_map_with_initial,
    build_branch_active_state_map,
    partition::{
        structural_branch_tokenizer_selected,
        materialize_branch_active_tokenizer_selected,
        branch_active_state_map_selected,
        inactive_component_branch_state_map,
    },
    synthetic_state_map,
};
use crate::grammar::ast;
use crate::grammar::factoring::factor_named_grammar;
use crate::grammar::flat::GrammarDef;
use crate::import::json_schema;
use glrmask_dwa_merge::merge_vocab_token_maps;

fn probe_dedicated_p0_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = match std::env::var("GLRMASK_P0_DEDICATED_THREADS") {
            Ok(value) => value.trim().parse::<usize>().ok().filter(|&value| value > 0)?,
            Err(_) => {
                #[cfg(target_os = "macos")]
                {
                    if rayon::current_num_threads() >= 8 { 2 } else { return None; }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    return None;
                }
            }
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|idx| format!("glrmask-probe-p0-{idx}"))
            .build()
            .ok()
    }).as_ref()
}

fn probe_dedicated_p0_max_tokenizer_states() -> u32 {
    std::env::var("GLRMASK_P0_DEDICATED_MAX_TOKENIZER_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(384)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchTimingRecord {
    pub problem_id: String,
    pub partition_label: String,
    pub branch_type: String,
    pub vocab_tokens: usize,
    pub active_terminals: usize,
    pub source_states: usize,
    pub kernel: String,
    pub prep_wall_ms: f64,
    pub prep_cpu_ms: f64,
    pub pre_state_wall_ms: f64,
    pub pre_state_cpu_ms: f64,
    pub exact_state_wall_ms: f64,
    pub exact_state_cpu_ms: f64,
    pub vocab_equiv_wall_ms: f64,
    pub vocab_equiv_cpu_ms: f64,
    pub finalize_wall_ms: f64,
    pub finalize_cpu_ms: f64,
    pub branch_total_wall_ms: f64,
    pub branch_total_cpu_ms: f64,
    pub branch_vocab_classes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProblemTimingRecord {
    pub problem_id: String,
    pub problem_format: String,
    pub status: String,
    pub error_message: Option<String>,
    pub num_terminals: usize,
    pub tokenizer_states: usize,
    pub final_vocab_classes: usize,
    pub import_parse_wall_ms: f64,
    pub import_parse_cpu_ms: f64,
    pub grammar_prep_wall_ms: f64,
    pub grammar_prep_cpu_ms: f64,
    pub lexer_setup_wall_ms: f64,
    pub lexer_setup_cpu_ms: f64,
    pub grammar_analysis_wall_ms: f64,
    pub grammar_analysis_cpu_ms: f64,
    pub glr_table_wall_ms: f64,
    pub glr_table_cpu_ms: f64,
    pub classify_routing_wall_ms: f64,
    pub classify_routing_cpu_ms: f64,
    pub global_max_len_wall_ms: f64,
    pub global_max_len_cpu_ms: f64,
    pub partition_total_wall_ms: f64,
    pub partition_total_cpu_ms: f64,
    pub global_merge_wall_ms: f64,
    pub global_merge_cpu_ms: f64,
    pub equiv_ready_wall_ms: f64,
    pub equiv_ready_cpu_ms: f64,
    pub instrumented_setup_total_wall_ms: f64,
    pub instrumented_setup_total_cpu_ms: f64,
    pub total_wall_ms: f64,
    pub total_cpu_ms: f64,
}


pub fn prepare_vocab_for_vocab_equiv_probe(vocab: &Vocab) {
    crate::prepare_vocab_for_compile(vocab);
}

pub struct ProbeOutcome {
    pub problem_record: ProblemTimingRecord,
    pub branch_records: Vec<BranchTimingRecord>,
    pub final_vocab_map: Option<ManyToOneIdMap>,
}

fn parse_named_grammar_from_source(format: &str, source_text: &str) -> crate::Result<ast::NamedGrammar> {
    match format {
        "json_schema" => {
            let schema: serde_json::Value = serde_json::from_str(source_text)
                .map_err(|e| crate::GlrMaskError::GrammarParse(format!("invalid JSON schema: {e}")))?;
            let named = json_schema::schema_to_named_grammar(&schema)?;
            Ok(named)
        }
        "glrm" => {
            Ok(crate::grammar::glrm::from_glrm(source_text)?)
        }
        "lark" => {
            Ok(glrmask_grammar::__private::import::lark::parse_lark_to_named(source_text)?)
        }
        "ebnf" => {
            Ok(glrmask_grammar::__private::import::ebnf::parse_ebnf_to_named(source_text)?)
        }
        other => Err(crate::GlrMaskError::GrammarParse(format!("unsupported format: {other}"))),
    }
}

pub fn probe_problem_vocab_equivalence(
    problem_id: &str,
    format: &str,
    source_text: &str,
    vocab: &Vocab,
) -> ProbeOutcome {
    let total_problem_timer = CpuTimer::start();
    let mut branch_records = Vec::new();

    // Stage 1: Import / Parse
    let import_timer = CpuTimer::start();
    let named_res = parse_named_grammar_from_source(format, source_text);
    let (import_parse_wall_ms, import_parse_cpu_ms) = import_timer.elapsed();

    let named = match named_res {
        Ok(named) => named,
        Err(e) => {
            let (total_wall_ms, total_cpu_ms) = total_problem_timer.elapsed();
            return ProbeOutcome {
                problem_record: ProblemTimingRecord {
                    problem_id: problem_id.to_string(),
                    problem_format: format.to_string(),
                    status: "error".to_string(),
                    error_message: Some(format!("import parse error: {e:?}")),
                    num_terminals: 0,
                    tokenizer_states: 0,
                    final_vocab_classes: 0,
                    import_parse_wall_ms,
                    import_parse_cpu_ms,
                    grammar_prep_wall_ms: 0.0,
                    grammar_prep_cpu_ms: 0.0,
                    lexer_setup_wall_ms: 0.0,
                    lexer_setup_cpu_ms: 0.0,
                    grammar_analysis_wall_ms: 0.0,
                    grammar_analysis_cpu_ms: 0.0,
                    glr_table_wall_ms: 0.0,
                    glr_table_cpu_ms: 0.0,
                    classify_routing_wall_ms: 0.0,
                    classify_routing_cpu_ms: 0.0,
                    global_max_len_wall_ms: 0.0,
                    global_max_len_cpu_ms: 0.0,
                    partition_total_wall_ms: 0.0,
                    partition_total_cpu_ms: 0.0,
                    global_merge_wall_ms: 0.0,
                    global_merge_cpu_ms: 0.0,
                    equiv_ready_wall_ms: total_wall_ms,
                    equiv_ready_cpu_ms: total_cpu_ms,
                    instrumented_setup_total_wall_ms: import_parse_wall_ms,
                    instrumented_setup_total_cpu_ms: import_parse_cpu_ms,
                    total_wall_ms,
                    total_cpu_ms,
                },
                branch_records,
                final_vocab_map: None,
            };
        }
    };

    // Stage 2: Grammar Transforms / Preparation
    let prep_timer = CpuTimer::start();
    let mut factored = factor_named_grammar(named);
    if format == "json_schema" {
        if let Err(e) = json_schema::prepare_named_grammar(&mut factored) {
            let (total_wall_ms, total_cpu_ms) = total_problem_timer.elapsed();
            return ProbeOutcome {
                problem_record: ProblemTimingRecord {
                    problem_id: problem_id.to_string(),
                    problem_format: format.to_string(),
                    status: "error".to_string(),
                    error_message: Some(format!("json_schema prep error: {e:?}")),
                    num_terminals: 0,
                    tokenizer_states: 0,
                    final_vocab_classes: 0,
                    import_parse_wall_ms,
                    import_parse_cpu_ms,
                    grammar_prep_wall_ms: 0.0,
                    grammar_prep_cpu_ms: 0.0,
                    lexer_setup_wall_ms: 0.0,
                    lexer_setup_cpu_ms: 0.0,
                    grammar_analysis_wall_ms: 0.0,
                    grammar_analysis_cpu_ms: 0.0,
                    glr_table_wall_ms: 0.0,
                    glr_table_cpu_ms: 0.0,
                    classify_routing_wall_ms: 0.0,
                    classify_routing_cpu_ms: 0.0,
                    global_max_len_wall_ms: 0.0,
                    global_max_len_cpu_ms: 0.0,
                    partition_total_wall_ms: 0.0,
                    partition_total_cpu_ms: 0.0,
                    global_merge_wall_ms: 0.0,
                    global_merge_cpu_ms: 0.0,
                    equiv_ready_wall_ms: total_wall_ms,
                    equiv_ready_cpu_ms: total_cpu_ms,
                    instrumented_setup_total_wall_ms: import_parse_wall_ms,
                    instrumented_setup_total_cpu_ms: import_parse_cpu_ms,
                    total_wall_ms,
                    total_cpu_ms,
                },
                branch_records,
                final_vocab_map: None,
            };
        }
    }
    let grammar_def = match ast::lower(&factored) {
        Ok(gdef) => gdef,
        Err(e) => {
            let (total_wall_ms, total_cpu_ms) = total_problem_timer.elapsed();
            return ProbeOutcome {
                problem_record: ProblemTimingRecord {
                    problem_id: problem_id.to_string(),
                    problem_format: format.to_string(),
                    status: "error".to_string(),
                    error_message: Some(format!("grammar lowering error: {e:?}")),
                    num_terminals: 0,
                    tokenizer_states: 0,
                    final_vocab_classes: 0,
                    import_parse_wall_ms,
                    import_parse_cpu_ms,
                    grammar_prep_wall_ms: 0.0,
                    grammar_prep_cpu_ms: 0.0,
                    lexer_setup_wall_ms: 0.0,
                    lexer_setup_cpu_ms: 0.0,
                    grammar_analysis_wall_ms: 0.0,
                    grammar_analysis_cpu_ms: 0.0,
                    glr_table_wall_ms: 0.0,
                    glr_table_cpu_ms: 0.0,
                    classify_routing_wall_ms: 0.0,
                    classify_routing_cpu_ms: 0.0,
                    global_max_len_wall_ms: 0.0,
                    global_max_len_cpu_ms: 0.0,
                    partition_total_wall_ms: 0.0,
                    partition_total_cpu_ms: 0.0,
                    global_merge_wall_ms: 0.0,
                    global_merge_cpu_ms: 0.0,
                    equiv_ready_wall_ms: total_wall_ms,
                    equiv_ready_cpu_ms: total_cpu_ms,
                    instrumented_setup_total_wall_ms: import_parse_wall_ms,
                    instrumented_setup_total_cpu_ms: import_parse_cpu_ms,
                    total_wall_ms,
                    total_cpu_ms,
                },
                branch_records,
                final_vocab_map: None,
            };
        }
    };
    let prepared_grammar = prepare_grammar_transforms_only(grammar_def);
    let (grammar_prep_wall_ms, grammar_prep_cpu_ms) = prep_timer.elapsed();

    // Stage 3: Lexer / NFA / Tokenizer Setup
    let lexer_timer = CpuTimer::start();
    crate::automata::lexer::compile::install_vocabulary_exact_state_certifier(
        crate::compiler::stages::id_map_and_terminal_dwa::synthetic_state_map::certify_vocabulary_exact_state_candidates,
    );
    let synthetic_tokenizer_plan = plan_synthetic_tokenizer(&prepared_grammar, vocab);
    let partition_local_synthesis_plan = synthetic_tokenizer_plan.as_ref().map(|plan| {
        Arc::new(PartitionLocalSynthesisPlan {
            expressions: Arc::from(plan.synthesized_expressions.clone().into_boxed_slice()),
            partition_ids: Arc::from(plan.partition_ids.clone().into_boxed_slice()),
            residual_isolation_classes: Arc::from(
                plan.residual_isolation_classes.clone().into_boxed_slice(),
            ),
            protected_terminal_ids: Arc::from(
                plan.changed_terminal_ids.clone().into_boxed_slice(),
            ),
            labels: Arc::from(
                prepared_grammar
                    .terminals
                    .iter()
                    .enumerate()
                    .map(|(index, _)| prepared_grammar.terminal_display_name(index as u32))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            adaptive: true,
            global_max_token_len: vocab
                .entries_map()
                .values()
                .map(Vec::len)
                .max()
                .unwrap_or(0),
        })
    });

    let (tokenizer, compile_initial_state_map) =
        build_static_compile_tokenizer_for_vocab_equiv_probe(
            &prepared_grammar,
            vocab,
            synthetic_tokenizer_plan.as_ref(),
        );
    let (lexer_setup_wall_ms, lexer_setup_cpu_ms) = lexer_timer.elapsed();

    // Stage 4: Grammar Analysis
    let analysis_timer = CpuTimer::start();
    let analyzed_grammar = Arc::new(AnalyzedGrammar::from_grammar_def(&prepared_grammar));
    let (ever_allowed_follows, _) = compute_allowed_follow_sets(&analyzed_grammar);
    let disallowed_follows = compute_disallowed_follows_from_ever(
        analyzed_grammar.num_terminals,
        &ever_allowed_follows,
    );
    let token_path_disallowed_follows = Arc::new(
        ignore_transparent_disallowed_follows(&disallowed_follows, prepared_grammar.ignore_terminal),
    );
    let normalized_token_path_disallowed_follows: Arc<[crate::ds::bitset::BitSet]> = Arc::from(
        normalize_disallowed_follows(
            analyzed_grammar.num_terminals as usize,
            &token_path_disallowed_follows,
        )
        .into_boxed_slice(),
    );
    let (grammar_analysis_wall_ms, grammar_analysis_cpu_ms) = analysis_timer.elapsed();

    // Stage 5: GLR Table Construction (Separately measured)
    let glr_timer = CpuTimer::start();
    let _glr_table = GLRTable::build_with_default_construction(
        &analyzed_grammar,
        GlrTableConstruction::ExperimentalCoreMerged,
    );
    let (glr_table_wall_ms, glr_table_cpu_ms) = glr_timer.elapsed();

    // Stage 6: Global Max Length & Flat Transition Table Setup
    let max_len_timer = CpuTimer::start();
    let flat_trans: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(&tokenizer));
    let global_max_length_state_map = build_global_max_length_state_map_with_initial(
        &tokenizer,
        vocab,
        &flat_trans,
        compile_initial_state_map.as_ref(),
    );
    let (global_max_len_wall_ms, global_max_len_cpu_ms) = max_len_timer.elapsed();

    // Stage 7: Partition Classification & Routing
    let classify_timer = CpuTimer::start();
    let terminal_coloring = TerminalColoring::identity(analyzed_grammar.num_terminals as usize);
    let shared_classify_cache = classify::SharedClassifyCache::new();
    let sub_vocabs: Arc<[Vocab]> = build_char_type_sub_vocabs(
        vocab,
        partition_local_synthesis_plan.is_some(),
        automatic_p2_overflow_threshold(tokenizer.num_states()),
    );
    let (classify_routing_wall_ms, classify_routing_cpu_ms) = classify_timer.elapsed();

    // Stage 8: Partition & Branch Vocabulary Equivalence
    let partition_total_timer = CpuTimer::start();
    let mut all_branch_maps: Vec<ManyToOneIdMap> = Vec::new();

    let direct_single_terminal = use_global_single_terminal_l1(&analyzed_grammar, prepared_grammar.ignore_terminal);
    if direct_single_terminal {
        let active_terminals = vec![true];
        let branch_timer = CpuTimer::start();
        let input = BuildInput {
            partition_label: "single_terminal_global",
            tokenizer: &tokenizer,
            vocab,
            terminal_coloring: &terminal_coloring,
            use_terminal_coloring: false,
            ignore_terminal: prepared_grammar.ignore_terminal,
            grammar: &analyzed_grammar,
            active_terminals: &active_terminals,
            flat_trans: &flat_trans,
            transitions_by_byte: None,
            initial_state_map: Some(&global_max_length_state_map),
            shared_generic_nfa_topology: None,
            shared_generic_nfa_trie: None,
            subset_parent_order: None,
        };
        if let Some(res) = build_projected_vocab_equivalence(input) {
            let (branch_total_wall_ms, branch_total_cpu_ms) = branch_timer.elapsed();
            branch_records.push(BranchTimingRecord {
                problem_id: problem_id.to_string(),
                partition_label: "global".to_string(),
                branch_type: "single_terminal_global".to_string(),
                vocab_tokens: vocab.len(),
                active_terminals: 1,
                source_states: tokenizer.num_states() as usize,
                kernel: res.kernel.to_string(),
                prep_wall_ms: res.prep_ms,
                prep_cpu_ms: res.prep_cpu_ms,
                pre_state_wall_ms: 0.0,
                pre_state_cpu_ms: 0.0,
                exact_state_wall_ms: 0.0,
                exact_state_cpu_ms: 0.0,
                vocab_equiv_wall_ms: res.scan_ms + res.compact_ms,
                vocab_equiv_cpu_ms: res.scan_cpu_ms + res.compact_cpu_ms,
                finalize_wall_ms: 0.0,
                finalize_cpu_ms: 0.0,
                branch_total_wall_ms,
                branch_total_cpu_ms,
                branch_vocab_classes: res.vocab_map.num_internal_ids() as usize,
            });
            all_branch_maps.push(res.vocab_map);
        }
    } else {
        let shared_vocab_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
        let shared_original_vocab_analysis_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache::default();
        let shared_transition_cache = OnceLock::new();
        let shared_l1_token_trie = l1::prepared_l1_token_bounded_analysis_trie(vocab);

        let build_partition = |idx: usize, sub_vocab: &Vocab| {
                let mut branch_records = Vec::new();
                let mut all_branch_maps = Vec::new();
            if sub_vocab.is_empty() {
                return (branch_records, all_branch_maps);
            }
            let label = format!("p{}", idx);

            // Mirror production's certified partition-local synthesis before
            // classification/equivalence analysis. The previous probe built the
            // plan but ignored it, causing pathological whole-token scans over
            // the much larger global tokenizer.
            let local = partition_local_synthesis_plan
                .as_deref()
                .and_then(|plan| build_partition_local_tokenizer(&tokenizer, sub_vocab, plan))
                .filter(|_| partition_local_synthesis_selected(&label));
            let local_flat_trans: Option<Arc<[u32]>> = local.as_ref().map(|local| {
                Arc::from(l1::build_flat_transition_table(&local.tokenizer))
            });
            let local_classify_cache = local.as_ref().map(|local| {
                let cache = classify::SharedClassifyCache::new();
                classify::prewarm_shared_classify_cache(
                    &local.tokenizer,
                    local.tokenizer.num_terminals(),
                    &cache,
                );
                cache
            });
            let local_vocab_dfa_cache = local.as_ref().map(|_| l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new());
            let local_original_vocab_analysis_dfa_cache = local.as_ref().map(|_| l2p::equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache::default());
            let local_transition_cache: Option<OnceLock<l2p::equivalence_analysis::compat::FlatTransitionCache>> = local.as_ref().map(|_| OnceLock::new());

            let partition_tokenizer = local.as_ref().map(|local| &local.tokenizer).unwrap_or(&tokenizer);
            let partition_flat_trans: &Arc<[u32]> = local_flat_trans.as_ref().unwrap_or(&flat_trans);
            let partition_classify_cache = local_classify_cache.as_ref().unwrap_or(&shared_classify_cache);
            let partition_vocab_dfa_cache = local_vocab_dfa_cache.as_ref().unwrap_or(&shared_vocab_dfa_cache);
            let partition_original_vocab_analysis_dfa_cache = local_original_vocab_analysis_dfa_cache.as_ref().unwrap_or(&shared_original_vocab_analysis_dfa_cache);
            let partition_transition_cache = local_transition_cache.as_ref().unwrap_or(&shared_transition_cache);
            let partition_initial_state_map = if local.is_some() { None } else { Some(&global_max_length_state_map) };

            let num_terminals = analyzed_grammar.num_terminals;
            let terminal_path_lengths = classify_terminal_path_lengths(
                &label,
                partition_tokenizer,
                sub_vocab,
                token_path_disallowed_follows.as_ref(),
                num_terminals,
                Some(partition_classify_cache),
            );

            let mut l1_mask = vec![false; num_terminals as usize];
            let mut l2p_mask = vec![false; num_terminals as usize];
            let mut has_l1 = false;
            let mut has_l2p = false;
            for (i, len) in terminal_path_lengths.iter().enumerate() {
                match len {
                    TerminalPathLength::One => {
                        l1_mask[i] = true;
                        has_l1 = true;
                    }
                    TerminalPathLength::TwoPlus => {
                        l2p_mask[i] = true;
                        has_l2p = true;
                    }
                    TerminalPathLength::Zero => {}
                }
            }

            let use_l2p_vocab_split = has_l2p && split_l2p_vocab_enabled();
            let l2p_vocab_split = use_l2p_vocab_split.then(|| {
                split_vocab_for_active_l2p_terminals(
                    partition_tokenizer,
                    partition_flat_trans,
                    sub_vocab,
                    &token_path_disallowed_follows,
                    num_terminals,
                    &l2p_mask,
                    Some(partition_classify_cache),
                    shared_l1_token_trie.as_deref(),
                )
            });
            let has_split_l1 = l2p_vocab_split.as_ref().is_some_and(|s| s.single_tokens != 0);
            let combine_l1_single = label == "p1" && has_l1 && has_split_l1;
            let combined_l1_mask = combine_l1_single.then(|| {
                l1_mask.iter().zip(&l2p_mask).map(|(&l1, &l2p)| l1 || l2p).collect::<Vec<_>>()
            });
            let l1_build_mask = combined_l1_mask.as_deref().unwrap_or(&l1_mask);
            let l1_transitions_by_byte = (has_l1 || has_split_l1).then(|| {
                partition_classify_cache.get().map(|bytesets| bytesets.transitions_by_byte())
            }).flatten();

            // Stage 8: Nested branch scheduling inside partition
            let (l1_output, (l2p_boundary_output, l2p_single_output)) = compile_profile_join(
                || {
                    // 8a. L1 Branch
                    if has_l1 {
                        let branch_timer = CpuTimer::start();
                        let branch_label = format!("{label}.l1");
                        let active_terminal_count = l1_mask.iter().filter(|&&active| active).count();
                        let source_states = partition_initial_state_map
                            .map(ManyToOneIdMap::num_internal_ids)
                            .unwrap_or_else(|| partition_tokenizer.num_states()) as usize;
                        let materialization_requested = structural_branch_tokenizer_selected(
                            &branch_label,
                            sub_vocab.len(),
                            active_terminal_count,
                            source_states,
                        ) || materialize_branch_active_tokenizer_selected(&branch_label);
                        let state_map_requested = materialization_requested
                            || branch_active_state_map_selected(
                                &branch_label,
                                sub_vocab.len(),
                                active_terminal_count,
                                source_states,
                            );
                        let branch_state_map = inactive_component_branch_state_map(
                            partition_tokenizer,
                            l1_build_mask,
                            partition_initial_state_map,
                            &branch_label,
                        )
                        .or_else(|| {
                            build_branch_active_state_map(
                                partition_tokenizer,
                                sub_vocab,
                                l1_build_mask,
                                partition_initial_state_map,
                                &branch_label,
                                state_map_requested,
                            )
                        });
                        let materialized = materialization_requested
                            .then(|| {
                                branch_state_map.as_ref().and_then(|(map, _)| {
                                    synthetic_state_map::materialize_active_tokenizer(
                                        partition_tokenizer,
                                        sub_vocab,
                                        l1_build_mask,
                                        map.clone(),
                                    )
                                })
                            })
                            .flatten();

                        let res = if let Some(materialized) = materialized.as_ref() {
                            let branch_flat_trans: Arc<[u32]> =
                                Arc::from(l1::build_flat_transition_table(&materialized.tokenizer));
                            let input = BuildInput {
                                partition_label: &label,
                                tokenizer: &materialized.tokenizer,
                                vocab: sub_vocab,
                                terminal_coloring: &terminal_coloring,
                                use_terminal_coloring: false,
                                ignore_terminal: prepared_grammar.ignore_terminal,
                                grammar: &analyzed_grammar,
                                active_terminals: l1_build_mask,
                                flat_trans: &branch_flat_trans,
                                transitions_by_byte: None,
                                initial_state_map: None,
                                shared_generic_nfa_topology: None,
                                shared_generic_nfa_trie: None,
                                subset_parent_order: None,
                            };
                            build_projected_vocab_equivalence(input)
                        } else {
                            let branch_initial_state_map = branch_state_map
                                .as_ref()
                                .map(|(map, _)| map)
                                .or(partition_initial_state_map);
                            let input = BuildInput {
                                partition_label: &label,
                                tokenizer: partition_tokenizer,
                                vocab: sub_vocab,
                                terminal_coloring: &terminal_coloring,
                                use_terminal_coloring: false,
                                ignore_terminal: prepared_grammar.ignore_terminal,
                                grammar: &analyzed_grammar,
                                active_terminals: l1_build_mask,
                                flat_trans: partition_flat_trans,
                                transitions_by_byte: l1_transitions_by_byte,
                                initial_state_map: branch_initial_state_map,
                                shared_generic_nfa_topology: None,
                                shared_generic_nfa_trie: None,
                                subset_parent_order: None,
                            };
                            build_projected_vocab_equivalence(input)
                        };

                        if let Some(res) = res {
                            let (branch_total_wall_ms, branch_total_cpu_ms) = branch_timer.elapsed();
                            let record = BranchTimingRecord {
                                problem_id: problem_id.to_string(),
                                partition_label: label.clone(),
                                branch_type: "l1".to_string(),
                                vocab_tokens: sub_vocab.len(),
                                active_terminals: active_terminal_count,
                                source_states: partition_tokenizer.num_states() as usize,
                                kernel: res.kernel.to_string(),
                                prep_wall_ms: res.prep_ms,
                                prep_cpu_ms: res.prep_cpu_ms,
                                pre_state_wall_ms: 0.0,
                                pre_state_cpu_ms: 0.0,
                                exact_state_wall_ms: 0.0,
                                exact_state_cpu_ms: 0.0,
                                vocab_equiv_wall_ms: res.scan_ms + res.compact_ms,
                                vocab_equiv_cpu_ms: res.scan_cpu_ms + res.compact_cpu_ms,
                                finalize_wall_ms: 0.0,
                                finalize_cpu_ms: 0.0,
                                branch_total_wall_ms,
                                branch_total_cpu_ms,
                                branch_vocab_classes: res.vocab_map.num_internal_ids() as usize,
                            };
                            Some((record, res.vocab_map))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                || {
                    // 8b. L2P Branch
                    if has_l2p {
                        let active_terminal_count = l2p_mask.iter().filter(|&&active| active).count();
                        if let Some(split) = l2p_vocab_split.as_ref() {
                            compile_profile_join(
                                || {
                                    // Boundary tokens
                                    if split.boundary_tokens > 0 {
                                        let branch_timer = CpuTimer::start();
                                        let boundary_vocab = split.boundary_vocab(sub_vocab);
                                        let branch_label = format!("{label}.l2p");
                                        let active_terminal_count =
                                            l2p_mask.iter().filter(|&&active| active).count();
                                        let source_states = partition_initial_state_map
                                            .map(ManyToOneIdMap::num_internal_ids)
                                            .unwrap_or_else(|| partition_tokenizer.num_states()) as usize;
                                        let materialization_requested = structural_branch_tokenizer_selected(
                                            &branch_label,
                                            boundary_vocab.len(),
                                            active_terminal_count,
                                            source_states,
                                        ) || materialize_branch_active_tokenizer_selected(&branch_label);
                                        let state_map_requested = materialization_requested
                                            || branch_active_state_map_selected(
                                                &branch_label,
                                                boundary_vocab.len(),
                                                active_terminal_count,
                                                source_states,
                                            );
                                        let branch_state_map = inactive_component_branch_state_map(
                                                partition_tokenizer,
                                                &l2p_mask,
                                                partition_initial_state_map,
                                                &branch_label,
                                            )
                                            .or_else(|| {
                                                build_branch_active_state_map(
                                                    partition_tokenizer,
                                                    &boundary_vocab,
                                                    &l2p_mask,
                                                    partition_initial_state_map,
                                                    &branch_label,
                                                    state_map_requested,
                                                )
                                            });
                                        let materialized = materialization_requested
                                            .then(|| {
                                                branch_state_map.as_ref().and_then(|(map, _)| {
                                                    synthetic_state_map::materialize_active_tokenizer(
                                                        partition_tokenizer,
                                                        &boundary_vocab,
                                                        &l2p_mask,
                                                        map.clone(),
                                                    )
                                                })
                                            })
                                            .flatten();

                                        let l2p_res = if let Some(materialized) = materialized.as_ref() {
                                            let branch_flat_trans: Arc<[u32]> = Arc::from(
                                                l1::build_flat_transition_table(&materialized.tokenizer),
                                            );
                                            let local_vocab_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
                                            let local_original_vocab_analysis_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache::default();
                                            let local_transition_cache = std::sync::OnceLock::new();
                                            analyze_vocab_equivalence_with_group_filter(
                                                &label,
                                                &materialized.tokenizer,
                                                &boundary_vocab,
                                                &disallowed_follows,
                                                prepared_grammar.ignore_terminal,
                                                true,
                                                Some(&normalized_token_path_disallowed_follows),
                                                Some(&l2p_mask),
                                                Some(&local_vocab_dfa_cache),
                                                Some(&local_original_vocab_analysis_dfa_cache),
                                                0.0,
                                                Some(&branch_flat_trans),
                                                Some(&local_transition_cache),
                                                None,
                                                false,
                                                None,
                                                None,
                                                shared_l1_token_trie.as_deref(),
                                            )
                                        } else {
                                            let branch_initial_state_map = branch_state_map
                                                .as_ref()
                                                .map(|(map, _)| map)
                                                .or(partition_initial_state_map);
                                            analyze_vocab_equivalence_with_group_filter(
                                                &label,
                                                partition_tokenizer,
                                                &boundary_vocab,
                                                &disallowed_follows,
                                                prepared_grammar.ignore_terminal,
                                                true,
                                                Some(&normalized_token_path_disallowed_follows),
                                                Some(&l2p_mask),
                                                Some(partition_vocab_dfa_cache),
                                                Some(partition_original_vocab_analysis_dfa_cache),
                                                0.0,
                                                Some(partition_flat_trans),
                                                Some(partition_transition_cache),
                                                branch_initial_state_map,
                                                false,
                                                None,
                                                None,
                                                shared_l1_token_trie.as_deref(),
                                            )
                                        };
                                        let (branch_total_wall_ms, branch_total_cpu_ms) = branch_timer.elapsed();
                                        let record = BranchTimingRecord {
                                            problem_id: problem_id.to_string(),
                                            partition_label: label.clone(),
                                            branch_type: "l2p".to_string(),
                                            vocab_tokens: boundary_vocab.len(),
                                            active_terminals: active_terminal_count,
                                            source_states: partition_tokenizer.num_states() as usize,
                                            kernel: if l2p_res.vocab_first { "l2p_vocab_first".to_string() } else { "l2p_state_first".to_string() },
                                            prep_wall_ms: l2p_res.prep_ms,
                                            prep_cpu_ms: l2p_res.prep_cpu_ms,
                                            pre_state_wall_ms: l2p_res.pre_state_ms,
                                            pre_state_cpu_ms: l2p_res.pre_state_cpu_ms,
                                            exact_state_wall_ms: l2p_res.exact_state_refine_ms,
                                            exact_state_cpu_ms: l2p_res.exact_state_refine_cpu_ms,
                                            vocab_equiv_wall_ms: l2p_res.vocab_equiv_ms,
                                            vocab_equiv_cpu_ms: l2p_res.vocab_equiv_cpu_ms,
                                            finalize_wall_ms: l2p_res.finalize_ms,
                                            finalize_cpu_ms: l2p_res.finalize_cpu_ms,
                                            branch_total_wall_ms,
                                            branch_total_cpu_ms,
                                            branch_vocab_classes: l2p_res.vocab_classes_count,
                                        };
                                        Some((record, l2p_res.vocab_map))
                                    } else {
                                        None
                                    }
                                },
                                || {
                                    // Split single tokens
                                    if split.single_tokens > 0 && !combine_l1_single {
                                        let branch_timer = CpuTimer::start();
                                        let single_vocab = split.single_vocab(sub_vocab);
                                        let input = BuildInput {
                                            partition_label: &label,
                                            tokenizer: partition_tokenizer,
                                            vocab: &single_vocab,
                                            terminal_coloring: &terminal_coloring,
                                            use_terminal_coloring: false,
                                            ignore_terminal: prepared_grammar.ignore_terminal,
                                            grammar: &analyzed_grammar,
                                            active_terminals: &l2p_mask,
                                            flat_trans: partition_flat_trans,
                                            transitions_by_byte: l1_transitions_by_byte,
                                            initial_state_map: partition_initial_state_map,
                                            shared_generic_nfa_topology: None,
                                            shared_generic_nfa_trie: None,
                                            subset_parent_order: None,
                                        };
                                        if let Some(res) = build_projected_vocab_equivalence(input) {
                                            let (branch_total_wall_ms, branch_total_cpu_ms) = branch_timer.elapsed();
                                            let record = BranchTimingRecord {
                                                problem_id: problem_id.to_string(),
                                                partition_label: label.clone(),
                                                branch_type: "l2p_single_l1".to_string(),
                                                vocab_tokens: single_vocab.len(),
                                                active_terminals: active_terminal_count,
                                                source_states: partition_tokenizer.num_states() as usize,
                                                kernel: res.kernel.to_string(),
                                                prep_wall_ms: res.prep_ms,
                                                prep_cpu_ms: res.prep_cpu_ms,
                                                pre_state_wall_ms: 0.0,
                                                pre_state_cpu_ms: 0.0,
                                                exact_state_wall_ms: 0.0,
                                                exact_state_cpu_ms: 0.0,
                                                vocab_equiv_wall_ms: res.scan_ms + res.compact_ms,
                                                vocab_equiv_cpu_ms: res.scan_cpu_ms + res.compact_cpu_ms,
                                                finalize_wall_ms: 0.0,
                                                finalize_cpu_ms: 0.0,
                                                branch_total_wall_ms,
                                                branch_total_cpu_ms,
                                                branch_vocab_classes: res.vocab_map.num_internal_ids() as usize,
                                            };
                                            Some((record, res.vocab_map))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                },
                            )
                        } else {
                            let branch_timer = CpuTimer::start();
                            let l2p_res = analyze_vocab_equivalence_with_group_filter(
                                &label,
                                partition_tokenizer,
                                sub_vocab,
                                &disallowed_follows,
                                prepared_grammar.ignore_terminal,
                                true,
                                Some(&normalized_token_path_disallowed_follows),
                                Some(&l2p_mask),
                                Some(partition_vocab_dfa_cache),
                                Some(partition_original_vocab_analysis_dfa_cache),
                                0.0,
                                Some(partition_flat_trans),
                                Some(partition_transition_cache),
                                partition_initial_state_map,
                                false,
                                None,
                                None,
                                shared_l1_token_trie.as_deref(),
                            );
                            let (branch_total_wall_ms, branch_total_cpu_ms) = branch_timer.elapsed();
                            let record = BranchTimingRecord {
                                problem_id: problem_id.to_string(),
                                partition_label: label.clone(),
                                branch_type: "l2p".to_string(),
                                vocab_tokens: sub_vocab.len(),
                                active_terminals: active_terminal_count,
                                source_states: partition_tokenizer.num_states() as usize,
                                kernel: if l2p_res.vocab_first { "l2p_vocab_first".to_string() } else { "l2p_state_first".to_string() },
                                prep_wall_ms: l2p_res.prep_ms,
                                prep_cpu_ms: l2p_res.prep_cpu_ms,
                                pre_state_wall_ms: l2p_res.pre_state_ms,
                                pre_state_cpu_ms: l2p_res.pre_state_cpu_ms,
                                exact_state_wall_ms: l2p_res.exact_state_refine_ms,
                                exact_state_cpu_ms: l2p_res.exact_state_refine_cpu_ms,
                                vocab_equiv_wall_ms: l2p_res.vocab_equiv_ms,
                                vocab_equiv_cpu_ms: l2p_res.vocab_equiv_cpu_ms,
                                finalize_wall_ms: l2p_res.finalize_ms,
                                finalize_cpu_ms: l2p_res.finalize_cpu_ms,
                                branch_total_wall_ms,
                                branch_total_cpu_ms,
                                branch_vocab_classes: l2p_res.vocab_classes_count,
                            };
                            (Some((record, l2p_res.vocab_map)), None)
                        }
                    } else {
                        (None, None)
                    }
                },
            );

            if let Some((rec, map)) = l1_output {
                branch_records.push(rec);
                all_branch_maps.push(map);
            }
            if let Some((rec, map)) = l2p_boundary_output {
                branch_records.push(rec);
                all_branch_maps.push(map);
            }
            if let Some((rec, map)) = l2p_single_output {
                branch_records.push(rec);
                all_branch_maps.push(map);
            }
                (branch_records, all_branch_maps)
            };
        let partition_outputs: Vec<(Vec<BranchTimingRecord>, Vec<ManyToOneIdMap>)> =
            if !sub_vocabs.is_empty()
                && tokenizer.num_states() <= probe_dedicated_p0_max_tokenizer_states()
                && let Some(p0_pool) = probe_dedicated_p0_pool()
            {
                let (p0_result, mut other_results) = rayon::join(
                    || p0_pool.install(|| build_partition(0, &sub_vocabs[0])),
                    || sub_vocabs[1..]
                        .par_iter()
                        .enumerate()
                        .map(|(offset, sub_vocab)| build_partition(offset + 1, sub_vocab))
                        .collect::<Vec<_>>(),
                );
                let mut results = Vec::with_capacity(sub_vocabs.len());
                results.push(p0_result);
                results.append(&mut other_results);
                results
            } else {
                sub_vocabs
                    .par_iter()
                    .enumerate()
                    .map(|(idx, sub_vocab)| build_partition(idx, sub_vocab))
                    .collect()
            };
        for (records, maps) in partition_outputs {
            branch_records.extend(records);
            all_branch_maps.extend(maps);
        }
    }
    let (partition_total_wall_ms, partition_total_cpu_ms) = partition_total_timer.elapsed();

    // Stage 9: Final Global Map Merge
    let merge_timer = CpuTimer::start();
    let map_refs: Vec<&ManyToOneIdMap> = all_branch_maps.iter().collect();
    let final_vocab_map = merge_vocab_token_maps(&map_refs, vocab.max_token_id());
    let (global_merge_wall_ms, global_merge_cpu_ms) = merge_timer.elapsed();

    let final_vocab_classes = final_vocab_map.num_internal_ids() as usize;

    // Stage 10: Total timings & metrics
    let (total_wall_ms, total_cpu_ms) = total_problem_timer.elapsed();
    let equiv_ready_wall_ms = (total_wall_ms - glr_table_wall_ms).max(0.0);
    let equiv_ready_cpu_ms = (total_cpu_ms - glr_table_cpu_ms).max(0.0);
    let instrumented_setup_total_wall_ms = import_parse_wall_ms
        + grammar_prep_wall_ms
        + lexer_setup_wall_ms
        + grammar_analysis_wall_ms
        + glr_table_wall_ms
        + classify_routing_wall_ms
        + global_max_len_wall_ms
        + partition_total_wall_ms
        + global_merge_wall_ms;
    let instrumented_setup_total_cpu_ms = import_parse_cpu_ms
        + grammar_prep_cpu_ms
        + lexer_setup_cpu_ms
        + grammar_analysis_cpu_ms
        + glr_table_cpu_ms
        + classify_routing_cpu_ms
        + global_max_len_cpu_ms
        + partition_total_cpu_ms
        + global_merge_cpu_ms;

    ProbeOutcome {
        problem_record: ProblemTimingRecord {
            problem_id: problem_id.to_string(),
            problem_format: format.to_string(),
            status: "ok".to_string(),
            error_message: None,
            num_terminals: analyzed_grammar.num_terminals as usize,
            tokenizer_states: tokenizer.num_states() as usize,
            final_vocab_classes,
            import_parse_wall_ms,
            import_parse_cpu_ms,
            grammar_prep_wall_ms,
            grammar_prep_cpu_ms,
            lexer_setup_wall_ms,
            lexer_setup_cpu_ms,
            grammar_analysis_wall_ms,
            grammar_analysis_cpu_ms,
            glr_table_wall_ms,
            glr_table_cpu_ms,
            classify_routing_wall_ms,
            classify_routing_cpu_ms,
            global_max_len_wall_ms,
            global_max_len_cpu_ms,
            partition_total_wall_ms,
            partition_total_cpu_ms,
            global_merge_wall_ms,
            global_merge_cpu_ms,
            equiv_ready_wall_ms,
            equiv_ready_cpu_ms,
            instrumented_setup_total_wall_ms,
            instrumented_setup_total_cpu_ms,
            total_wall_ms,
            total_cpu_ms,
        },
        branch_records,
        final_vocab_map: Some(final_vocab_map),
    }
}
