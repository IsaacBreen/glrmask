use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use once_cell::sync::Lazy;
use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::compile::{
    build_regex,
    build_regex_partitioned,
    build_regex_partitioned_with_adaptive,
    build_regex_partitioned_with_adaptive_and_residual_isolation,
    build_regex_partitioned_with_profile_labels,
    build_regex_partitioned_with_profile_labels_and_adaptive,
    build_regex_partitioned_with_profile_labels_and_adaptive_and_residual_isolation,
    build_regex_partitioned_with_profile_labels_and_residual_isolation,
    build_regex_partitioned_with_residual_isolation,
    build_regex_with_profile_labels,
    compile_terminal_expression_pair_with_structural_map,
    factor_regex_expr,
    prepare_partitioned_expression_pair_with_structural_map,
    DeferredPartitionedRegex,
};
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::constraint_possible_matches as cpm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{GLRTable, GlrTableConstruction};
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    SharedClassifyCache,
    prewarm_shared_classify_cache,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::{
    compute_allowed_follow_sets,
    compute_ever_allowed_follows,
    compute_terminal_coloring,
    ignore_transparent_disallowed_follows,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    TerminalColoring,
    TerminalDwaFamilies,
    TerminalDwaPhaseProfile,
};
use crate::compiler::stages::id_map_and_terminal_dwa::synthetic_state_map::{
    BoundedTerminalCandidateScanner, CertifiedFullToSynthesizedStateMap,
    certify_full_to_synthesized_state_map, estimated_synthesis_state_volume,
    synthesize_bounded_terminal_expressions, synthesize_terminal_expressions_for_horizon,
};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::mapped_artifact::{
    MappedArtifact,
    WeightRefs,
    count_interned_ranges_for_weights,
};
use crate::compiler::stages::parser_dwa::{
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    try_build_direct_regular_parser_top_accept_parts, try_build_immediate_parser_dwa,
    try_build_immediate_terminal_completion_weights,
};
use crate::compiler::stages::templates::{Templates, commit_template_dfas_enabled};
use crate::compiler::stages::templates::characterize::characterize_terminals_profiled;
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::{GrammarDef, Terminal, TerminalID};
use crate::runtime::{Constraint, SpecialTokenTerminal};
use crate::DynamicConstraint;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn env_flag_enabled_by_default(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}

fn compact_possible_matches_before_reconcile_enabled() -> bool {
    env_flag_enabled_by_default("GLRMASK_COMPACT_POSSIBLE_MATCHES_BEFORE_RECONCILE")
}

fn terminal_coloring_enabled() -> bool {
    env_flag_enabled("GLRMASK_TERMINAL_COLORING")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DwaPossibleMatchesMode {
    TerminalReconcile,
    TerminalReconcileAndCompact,
    TerminalReconcileAndParserCompact,
    TerminalReconcileAndTerminalCompactAndParserCompact,
    ParserReconcile,
    ParserReconcileAndCompact,
}

impl DwaPossibleMatchesMode {
    fn does_terminal_reconcile(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcile
                | Self::TerminalReconcileAndCompact
                | Self::TerminalReconcileAndParserCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
        )
    }

    fn does_terminal_compact(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcileAndCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
        )
    }

    fn does_parser_compact(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcileAndParserCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
                | Self::ParserReconcileAndCompact
        )
    }
}

fn dwa_possible_matches_mode() -> DwaPossibleMatchesMode {
    match std::env::var("GLRMASK_DWA_PM_MODE")
        .or_else(|_| std::env::var("GLRMASK_PARSER_DWA_PM_COMPACTION"))
    {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "no" | "off" | "terminal" | "term"
            | "term_pm_reconcile" | "terminal_pm_reconcile" => DwaPossibleMatchesMode::TerminalReconcile,
            "term_compact" | "terminal_compact" | "term_pm_compact"
            | "terminal_pm_compact" | "term_pm_reconcile_compact"
            | "terminal_pm_reconcile_compact" => DwaPossibleMatchesMode::TerminalReconcileAndCompact,
            "parser_compact" | "term_parser_compact" | "terminal_parser_compact"
            | "term_pm_reconcile_parser_pm_compact"
            | "terminal_pm_reconcile_parser_pm_compact" => {
                DwaPossibleMatchesMode::TerminalReconcileAndParserCompact
            }
            "both" | "1" | "true" | "yes" | "on" | "term_and_parser_compact"
            | "terminal_and_parser_compact" | "term_pm_compact_parser_pm_compact"
            | "terminal_pm_compact_parser_pm_compact" => {
                DwaPossibleMatchesMode::TerminalReconcileAndTerminalCompactAndParserCompact
            }
            "parser" | "only" | "parser_only" | "replace" | "parser_pm_reconcile" => {
                DwaPossibleMatchesMode::ParserReconcile
            }
            "parser_pm_compact" | "parser_reconcile_compact"
            | "parser_pm_reconcile_compact" => DwaPossibleMatchesMode::ParserReconcileAndCompact,
            _ => DwaPossibleMatchesMode::TerminalReconcile,
        },
        Err(_) => {
            // PM compaction remains available via `GLRMASK_DWA_PM_MODE=terminal_compact`,
            // `parser_pm_compact`, and `both`, but it is not the default because large
            // schemas can pay substantial compile time for small artifact-size wins.
            DwaPossibleMatchesMode::ParserReconcile
        }
    }
}

pub(crate) fn compile_profile_summary_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

pub(crate) fn compile_top_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE_TOP")
}

pub(crate) fn compile_profile_enabled() -> bool {
    compile_profile_summary_enabled()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn compile_thread_count() -> Option<usize> {
    if let Some(value) = std::env::var("GLRMASK_COMPILE_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return Some(value);
    }

    if std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .is_some_and(|value| value > 0)
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        return std::thread::available_parallelism()
            .ok()
            .map(|parallelism| parallelism.get().min(8))
            .filter(|&value| value > 1);
    }

    #[cfg(not(target_os = "macos"))]
    {
        return std::thread::available_parallelism()
            .ok()
            .map(|parallelism| parallelism.get().min(8))
            .filter(|&value| value > 1);
    }
}

static COMPILE_THREAD_POOL: Lazy<Option<rayon::ThreadPool>> = Lazy::new(|| {
    let thread_count = compile_thread_count()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .ok()
});

fn run_with_compile_thread_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if let Some(pool) = &*COMPILE_THREAD_POOL {
        pool.install(f)
    } else {
        f()
    }
}

#[derive(Debug, Default)]
pub(crate) struct CompilePhaseProfile {
    pub(crate) prepare_ms: f64,
    pub(crate) tokenizer_build_ms: f64,
    pub(crate) tokenizer_final_states: usize,
    pub(crate) tokenizer_final_transitions: usize,
    pub(crate) synthetic_candidate_terminals: usize,
    pub(crate) synthetic_certified: bool,
    pub(crate) synthetic_compile_states: usize,
    pub(crate) synthetic_compile_transitions: usize,
    pub(crate) synthetic_certification_ms: f64,
    pub(crate) analyze_grammar_ms: f64,
    pub(crate) glr_table_ms: f64,
    pub(crate) terminal_coloring_ms: f64,
    pub(crate) disallowed_follows_ms: f64,
    pub(crate) analysis_wall_ms: f64,
    pub(crate) classify_ms: f64,
    pub(crate) id_map_ms: f64,
    pub(crate) terminal_dwa_ms: f64,
    pub(crate) templates_ms: f64,
    pub(crate) compact_ms: f64,
    pub(crate) split_terminal_dwa_total_ms: f64,
    pub(crate) global_merge_ms: f64,
    pub(crate) possible_matches_vocab_equiv_ms: f64,
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) possible_matches_materialize_ms: f64,
    pub(crate) shared_id_reconcile_ms: f64,
    pub(crate) possible_matches_pipeline_ms: f64,
    pub(crate) terminal_dwa_interned_ranges_before_pm_reconcile: usize,
    pub(crate) possible_matches_interned_ranges_before_pm_reconcile: usize,
    pub(crate) terminal_pm_joint_interned_ranges_before_reconcile: usize,
    pub(crate) terminal_pm_joint_interned_ranges: usize,
    pub(crate) internal_token_bytes_ms: f64,
    pub(crate) terminal_run_collapse_ms: f64,
    pub(crate) parser_dwa_ms: f64,
    pub(crate) parser_dwa_interned_ranges: usize,
    pub(crate) possible_matches_interned_ranges: usize,
    pub(crate) parser_pm_joint_interned_ranges: usize,
    pub(crate) finalize_ms: f64,
    pub(crate) compile_ms: f64,
    pub(crate) total_ms: f64,
}

pub(crate) fn emit_compile_profile_summary(
    source_kind: Option<&str>,
    import_ms: Option<f64>,
    profile: &CompilePhaseProfile,
) {
    if !compile_profile_summary_enabled() && !compile_top_profile_enabled() {
        return;
    }

    let source = source_kind.unwrap_or("grammar");
    let import_fragment = import_ms
        .map(|ms| format!(" import_ms={ms:.3}"))
        .unwrap_or_default();

    eprintln!(
        "[glrmask/profile][compile] source={}{} prepare_ms={:.3} tokenizer_build_ms={:.3} tokenizer_final_states={} tokenizer_final_transitions={} synthetic_candidate_terminals={} synthetic_certified={} synthetic_compile_states={} synthetic_compile_transitions={} synthetic_certification_ms={:.3} analyze_grammar_ms={:.3} glr_table_ms={:.3} terminal_coloring_ms={:.3} disallowed_follows_ms={:.3} analysis_wall_ms={:.3} classify_ms={:.3} id_map_ms={:.3} terminal_dwa_ms={:.3} split_terminal_dwa_total_ms={:.3} global_merge_ms={:.3} templates_ms={:.3} compact_ms={:.3} possible_matches_vocab_equiv_ms={:.3} possible_matches_collect_ms={:.3} possible_matches_materialize_ms={:.3} shared_id_reconcile_ms={:.3} possible_matches_pipeline_ms={:.3} terminal_dwa_interned_ranges_before_pm_reconcile={} possible_matches_interned_ranges_before_pm_reconcile={} terminal_pm_joint_interned_ranges_before_reconcile={} terminal_pm_joint_interned_ranges={} internal_token_bytes_ms={:.3} terminal_run_collapse_ms={:.3} parser_dwa_ms={:.3} parser_dwa_interned_ranges={} possible_matches_interned_ranges={} parser_pm_joint_interned_ranges={} finalize_ms={:.3} compile_ms={:.3} total_ms={:.3}",
        source,
        import_fragment,
        profile.prepare_ms,
        profile.tokenizer_build_ms,
        profile.tokenizer_final_states,
        profile.tokenizer_final_transitions,
        profile.synthetic_candidate_terminals,
        profile.synthetic_certified,
        profile.synthetic_compile_states,
        profile.synthetic_compile_transitions,
        profile.synthetic_certification_ms,
        profile.analyze_grammar_ms,
        profile.glr_table_ms,
        profile.terminal_coloring_ms,
        profile.disallowed_follows_ms,
        profile.analysis_wall_ms,
        profile.classify_ms,
        profile.id_map_ms,
        profile.terminal_dwa_ms,
        profile.split_terminal_dwa_total_ms,
        profile.global_merge_ms,
        profile.templates_ms,
        profile.compact_ms,
        profile.possible_matches_vocab_equiv_ms,
        profile.possible_matches_collect_ms,
        profile.possible_matches_materialize_ms,
        profile.shared_id_reconcile_ms,
        profile.possible_matches_pipeline_ms,
        profile.terminal_dwa_interned_ranges_before_pm_reconcile,
        profile.possible_matches_interned_ranges_before_pm_reconcile,
        profile.terminal_pm_joint_interned_ranges_before_reconcile,
        profile.terminal_pm_joint_interned_ranges,
        profile.internal_token_bytes_ms,
        profile.terminal_run_collapse_ms,
        profile.parser_dwa_ms,
        profile.parser_dwa_interned_ranges,
        profile.possible_matches_interned_ranges,
        profile.parser_pm_joint_interned_ranges,
        profile.finalize_ms,
        profile.compile_ms,
        profile.total_ms,
    );
}

fn interned_range_count_for_weight_refs(weight_refs: &[&Weight]) -> usize {
    let counts = count_interned_ranges_for_weights(weight_refs.iter().copied());
    counts.tsid_ranges + counts.token_ranges
}

fn interned_range_count_for_artifact<T: WeightRefs>(artifact: &mut T) -> usize {
    let weights = artifact.weight_refs_mut();
    let weight_refs: Vec<_> = weights.iter().map(|weight| &**weight).collect();
    interned_range_count_for_weight_refs(&weight_refs)
}

fn joint_interned_range_count_for_artifacts<L, R>(left: &mut L, right: &mut R) -> usize
where
    L: WeightRefs,
    R: WeightRefs,
{
    let left_weights = left.weight_refs_mut();
    let right_weights = right.weight_refs_mut();
    let mut weight_refs = Vec::with_capacity(left_weights.len() + right_weights.len());
    weight_refs.extend(left_weights.iter().map(|weight| &**weight));
    weight_refs.extend(right_weights.iter().map(|weight| &**weight));
    interned_range_count_for_weight_refs(&weight_refs)
}

pub(crate) fn compute_disallowed_follows(grammar: &AnalyzedGrammar) -> BTreeMap<u32, BitSet> {
    let ever_allowed = compute_ever_allowed_follows(grammar);
    compute_disallowed_follows_from_ever(grammar.num_terminals, &ever_allowed)
}

fn compute_disallowed_follows_from_ever(
    num_terminals: u32,
    ever_allowed: &[Vec<u32>],
) -> BTreeMap<u32, BitSet> {
    let num_terminals = num_terminals as usize;
    let mut disallowed_by_terminal = BTreeMap::new();

    for (terminal_id, allowed) in ever_allowed.iter().enumerate() {
        let mut allowed_bits = BitSet::new(num_terminals);
        for &follow in allowed {
            if (follow as usize) < num_terminals {
                allowed_bits.set(follow as usize);
            }
        }
        let disallowed = allowed_bits.complement();
        if !disallowed.is_zero() {
            disallowed_by_terminal.insert(terminal_id as u32, disallowed);
        }
    }

    disallowed_by_terminal
}

pub(crate) fn build_tokenizer(grammar: &GrammarDef) -> Tokenizer {
    let profile_timing = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
        || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TRACE").is_some();
    let factor_started_at = Instant::now();
    let exprs: Vec<Expr> = grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .map(factor_regex_expr)
        .collect();
    if profile_timing {
        eprintln!(
            "[glrmask/profile][tokenizer] factor_terminals terminals={} elapsed_ms={:.3}",
            exprs.len(),
            elapsed_ms(factor_started_at),
        );
    }
    let terminal_labels: Vec<String> = grammar
        .terminals
        .iter()
        .enumerate()
        .map(|(index, _)| grammar.terminal_display_name(index as u32))
        .collect();
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] terminals={}",
            grammar.terminals.len()
        );
    }
    let partition_ids = lexer_partition_ids(grammar);
    if profile_detail {
        let mut partition_terminals = BTreeMap::<u32, Vec<(u32, &str, &str)>>::new();
        for (terminal, &partition_id) in partition_ids.iter().enumerate() {
            let terminal_id = terminal as u32;
            let partition_name = grammar
                .lexer_partitions
                .get(&terminal_id)
                .map(String::as_str)
                .unwrap_or("<default>");
            partition_terminals.entry(partition_id).or_default().push((
                terminal_id,
                partition_name,
                terminal_labels[terminal].as_str(),
            ));
        }
        for (partition_id, terminals) in partition_terminals {
            let partition_names = terminals
                .iter()
                .map(|(_, partition_name, _)| *partition_name)
                .collect::<BTreeSet<_>>();
            let labels = terminals
                .iter()
                .map(|(terminal, _, label)| format!("{terminal}:{label}"))
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][tokenizer] lexer_partition id={} names={:?} terminals={} labels=[{}]",
                partition_id,
                partition_names,
                terminals.len(),
                labels.join(", "),
            );
        }
    }
    let residual_isolation_classes = lexer_residual_isolation_classes(grammar);
    build_tokenizer_from_exprs_partitioned_impl(
        &exprs,
        Some(&terminal_labels),
        &partition_ids,
        Some(&residual_isolation_classes),
        None,
    )
}

/// Dynamic masking can execute directly over partitioned NFA tokenizer states,
/// so large terminal sets do not need to pay for expensive product DFA
/// construction up front. Keep the historical tokenizer policy for smaller
/// grammars, where the more deterministic shape is useful at mask time.
/// Explicit lexer env overrides continue to take precedence over this default.
fn build_dynamic_tokenizer(grammar: &GrammarDef) -> Tokenizer {
    const LARGE_DYNAMIC_LEXER_TERMINALS: usize = 96;

    let explicit_policy = std::env::var_os("GLRMASK_LEXER_SINGLETONS").is_some()
        || std::env::var_os("GLRMASK_LEXER_ADAPTIVE").is_some();
    if !explicit_policy && grammar.terminals.len() >= LARGE_DYNAMIC_LEXER_TERMINALS {
        build_tokenizer_with_partition_options(grammar, true, false)
    } else {
        build_tokenizer(grammar)
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid {name}={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
        Err(_) => default,
    }
}

fn lexer_partition_ids_with_options(
    grammar: &GrammarDef,
    singleton_all_terminals: bool,
) -> Vec<u32> {
    let residual_isolation_classes = lexer_residual_isolation_classes(grammar);
    lexer_partition_ids_with_residual_classes(
        grammar,
        singleton_all_terminals,
        &residual_isolation_classes,
    )
}

fn lexer_partition_ids_with_residual_classes(
    grammar: &GrammarDef,
    singleton_all_terminals: bool,
    residual_isolation_classes: &[Option<u32>],
) -> Vec<u32> {
    assert_eq!(
        grammar.terminals.len(),
        residual_isolation_classes.len(),
        "one residual-isolation entry is required per terminal",
    );
    // Named lexer groups opt into partitioning. Unspecified terminals remain
    // monolithic by default so existing grammars keep their historical lexer
    // shape. The global singleton override deliberately takes precedence over
    // named groups because it is an exact stress mode.
    let mut ids_by_key = BTreeMap::<String, u32>::new();
    let mut next_id = 0u32;
    (0..grammar.terminals.len())
        .map(|terminal| {
            let terminal = terminal as u32;
            let key = if singleton_all_terminals {
                format!("terminal:{terminal}")
            } else if let Some(class) = residual_isolation_classes[terminal as usize] {
                format!("residual-isolation:{class}")
            } else {
                grammar
                    .lexer_partitions
                    .get(&terminal)
                    .map(|partition| format!("named:{partition}"))
                    .unwrap_or_else(|| "default".to_string())
            };
            *ids_by_key.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
}

fn lexer_residual_isolation_classes(grammar: &GrammarDef) -> Vec<Option<u32>> {
    (0..grammar.terminals.len())
        .map(|terminal| {
            grammar
                .residual_isolation_classes
                .get(&(terminal as u32))
                .copied()
        })
        .collect()
}

fn lexer_partition_ids(grammar: &GrammarDef) -> Vec<u32> {
    let singleton_all_terminals = env_flag("GLRMASK_LEXER_SINGLETONS", false);
    lexer_partition_ids_with_options(grammar, singleton_all_terminals)
}

pub(crate) fn build_tokenizer_with_partition_options(
    grammar: &GrammarDef,
    singleton_all_terminals: bool,
    adaptive: bool,
) -> Tokenizer {
    let exprs = grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let labels = grammar
        .terminals
        .iter()
        .enumerate()
        .map(|(index, _)| grammar.terminal_display_name(index as u32))
        .collect::<Vec<_>>();
    let partition_ids = lexer_partition_ids_with_options(grammar, singleton_all_terminals);
    let residual_isolation_classes = lexer_residual_isolation_classes(grammar);
    build_tokenizer_from_exprs_partitioned_impl(
        &exprs,
        Some(&labels),
        &partition_ids,
        Some(&residual_isolation_classes),
        Some(adaptive),
    )
}

#[cfg(test)]
mod huge_parser_top_accept_collapse_tests {
    use std::collections::BTreeMap;

    use range_set_blaze::RangeSetBlaze;

    use super::{
        collapse_huge_parser_top_accept_parts, ParserTopAccept,
        PARSER_TOP_ACCEPT_COMPILE_UNION_MAX_MISSING_TERMINALS,
        PARSER_TOP_ACCEPT_COMPILE_UNION_MIN_PARTS,
    };
    use crate::ds::weight::Weight;

    fn point_weight(token: u32) -> Weight {
        Weight::from_per_tsid_token_sets([(
            0,
            RangeSetBlaze::from_iter([token..=token]),
        )])
    }

    fn top_accept_with_parts(label: i32, count: usize) -> (ParserTopAccept, Weight) {
        let parts = (0..count as u32).map(point_weight).collect::<Vec<_>>();
        let expected = Weight::union_all(parts.iter());
        (
            ParserTopAccept {
                combined: BTreeMap::new(),
                parts: BTreeMap::from([(label, parts)]),
                direct_l1_complete_by_terminal: BTreeMap::new(),
            },
            expected,
        )
    }

    #[test]
    fn large_near_universal_top_accept_parts_collapse_with_exact_union() {
        let count = PARSER_TOP_ACCEPT_COMPILE_UNION_MIN_PARTS;
        let num_terminals = count + PARSER_TOP_ACCEPT_COMPILE_UNION_MAX_MISSING_TERMINALS;
        let (mut top_accept, expected) = top_accept_with_parts(11, count);

        let report = collapse_huge_parser_top_accept_parts(&mut top_accept, num_terminals);
        let combined = top_accept.combined.get(&11).expect("collapsed label");
        assert!(combined.is_subset(&expected));
        assert!(expected.is_subset(combined));
        assert!(!top_accept.parts.contains_key(&11));
        assert_eq!(report.labels_collapsed, 1);
        assert_eq!(report.part_refs_after, 0);
    }

    #[test]
    fn large_non_universal_top_accept_parts_remain_partitioned() {
        let count = PARSER_TOP_ACCEPT_COMPILE_UNION_MIN_PARTS;
        let num_terminals = count + PARSER_TOP_ACCEPT_COMPILE_UNION_MAX_MISSING_TERMINALS + 1;
        let (mut top_accept, _) = top_accept_with_parts(7, count);

        let report = collapse_huge_parser_top_accept_parts(&mut top_accept, num_terminals);
        assert!(top_accept.combined.is_empty());
        assert_eq!(top_accept.parts[&7].len(), count);
        assert_eq!(report.labels_collapsed, 0);
        assert_eq!(report.part_refs_after, count);
    }

    #[test]
    fn small_near_universal_top_accept_parts_remain_partitioned() {
        let count = PARSER_TOP_ACCEPT_COMPILE_UNION_MIN_PARTS - 1;
        let (mut top_accept, _) = top_accept_with_parts(5, count);

        let report = collapse_huge_parser_top_accept_parts(&mut top_accept, count);
        assert!(top_accept.combined.is_empty());
        assert_eq!(top_accept.parts[&5].len(), count);
        assert_eq!(report.labels_collapsed, 0);
    }
}

#[cfg(test)]
mod lexer_partition_plan_tests {
    use std::collections::BTreeSet;

    use super::{
        lexer_partition_ids_with_options, prepare_structural_tokenizer_pair,
        plan_synthetic_tokenizer_enabled, structural_state_reduction_is_profitable,
    };
    use crate::automata::lexer::Lexer;
    use crate::automata::regex::Expr;
    use crate::grammar::flat::{GrammarDef, Terminal};
    use crate::Vocab;

    fn grammar_with_terminals(count: u32) -> GrammarDef {
        GrammarDef {
            terminals: (0..count)
                .map(|id| Terminal::Literal {
                    id,
                    bytes: vec![b'a' + id as u8],
                })
                .collect(),
            ..GrammarDef::default()
        }
    }

    #[test]
    fn unspecified_terminals_are_monolithic_by_default() {
        let grammar = grammar_with_terminals(3);
        assert_eq!(lexer_partition_ids_with_options(&grammar, false), vec![0, 0, 0]);
    }

    #[test]
    fn global_singleton_override_isolates_named_and_unnamed_terminals() {
        let mut grammar = grammar_with_terminals(3);
        grammar.lexer_partitions.insert(0, "words".to_string());
        grammar.lexer_partitions.insert(1, "words".to_string());

        let ids = lexer_partition_ids_with_options(&grammar, true);
        assert_eq!(ids.iter().copied().collect::<BTreeSet<_>>().len(), 3);
    }

    #[test]
    fn named_partition_membership_is_preserved_by_partition_planning() {
        let mut grammar = grammar_with_terminals(3);
        grammar.lexer_partitions.insert(0, "words".to_string());
        grammar.lexer_partitions.insert(1, "words".to_string());
        grammar.lexer_partitions.insert(2, "numbers".to_string());

        let ids = lexer_partition_ids_with_options(&grammar, false);
        assert_eq!(ids[0], ids[1]);
        assert_ne!(ids[0], ids[2]);
    }

    #[test]
    fn residual_isolation_classes_override_named_partition_membership() {
        let mut grammar = grammar_with_terminals(3);
        for terminal in 0..3 {
            grammar
                .lexer_partitions
                .insert(terminal, "words".to_string());
        }
        grammar.residual_isolation_classes.insert(0, 71);
        grammar.residual_isolation_classes.insert(1, 72);

        let ids = lexer_partition_ids_with_options(&grammar, false);
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[0], ids[2]);
        assert_ne!(ids[1], ids[2]);
    }

    #[test]
    fn exceptional_structural_reduction_accepts_bounded_large_compile_coordinate() {
        assert!(structural_state_reduction_is_profitable(1_437_667, 173_832));
        assert!(!structural_state_reduction_is_profitable(1_000_000, 250_001));
        assert!(!structural_state_reduction_is_profitable(1_000_000, 200_000));
    }

    #[test]
    fn structural_pair_preisolates_ordinary_nullable_components() {
        let grammar = GrammarDef {
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 1,
                        max: Some(5_000),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min: 0,
                        max: Some(1),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, Vec::new()),
            (1, b"a".to_vec()),
            (2, b"aaaa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let plan = plan_synthetic_tokenizer_enabled(
            &grammar,
            &vocab,
            false,
            std::time::Instant::now(),
            false,
            false,
        )
            .expect("large bounded terminal should be selected for synthesis");
        let (synthesized, full, certified) =
            prepare_structural_tokenizer_pair(&grammar, &plan, &vocab, Some(false))
                .expect("nullable structural pair");
        let full = full.finish();

        assert_eq!(
            certified.full_to_synthesized.len(),
            full.num_states() as usize,
        );
        assert!(certified
            .full_to_synthesized
            .iter()
            .all(|&state| state < synthesized.num_states()));
        assert!(full.matched_terminals(full.initial_state()).is_empty());
        assert!(synthesized
            .matched_terminals(synthesized.initial_state())
            .is_empty());
        let full_after_a = full.step_all(&[full.initial_state()], b'a');
        let synthesized_after_a =
            synthesized.step_all(&[synthesized.initial_state()], b'a');
        assert!(full_after_a
            .iter()
            .any(|&state| full.matched_terminals(state).contains(&0)));
        assert!(synthesized_after_a
            .iter()
            .any(|&state| synthesized.matched_terminals(state).contains(&0)));
        let full_after_b = full.step_all(&[full.initial_state()], b'b');
        let synthesized_after_b =
            synthesized.step_all(&[synthesized.initial_state()], b'b');
        assert!(full_after_b
            .iter()
            .any(|&state| full.matched_terminals(state).contains(&1)));
        assert!(synthesized_after_b
            .iter()
            .any(|&state| synthesized.matched_terminals(state).contains(&1)));
    }
}

pub(crate) fn build_tokenizer_from_exprs(
    exprs: &[Expr],
    profile_labels: Option<&[String]>,
) -> Tokenizer {
    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let started_at = Instant::now();
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] combined_build_start terminals={} labels={} ",
            exprs.len(),
            profile_labels.map_or(0, |labels| labels.len())
        );
    }
    let regex = if let Some(labels) = profile_labels {
        build_regex_with_profile_labels(exprs, labels)
    } else {
        build_regex(exprs)
    };
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] combined_build_done terminals={} elapsed_ms={:.3} final_states={} final_transitions={}",
            exprs.len(),
            elapsed_ms(started_at),
            regex.num_states(),
            regex.num_transitions()
        );
    }

    regex.into_tokenizer(
        exprs.len() as u32,
        Some(std::sync::Arc::from(exprs.to_vec())),
    )
}

pub(crate) fn build_tokenizer_from_exprs_partitioned(
    exprs: &[Expr],
    profile_labels: Option<&[String]>,
    partition_ids: &[u32],
) -> Tokenizer {
    build_tokenizer_from_exprs_partitioned_impl(
        exprs,
        profile_labels,
        partition_ids,
        None,
        None,
    )
}

pub(crate) fn build_tokenizer_from_exprs_partitioned_with_adaptive(
    exprs: &[Expr],
    profile_labels: Option<&[String]>,
    partition_ids: &[u32],
    adaptive: bool,
) -> Tokenizer {
    build_tokenizer_from_exprs_partitioned_impl(
        exprs,
        profile_labels,
        partition_ids,
        None,
        Some(adaptive),
    )
}

fn build_tokenizer_from_exprs_partitioned_impl(
    exprs: &[Expr],
    profile_labels: Option<&[String]>,
    partition_ids: &[u32],
    residual_isolation_classes: Option<&[Option<u32>]>,
    adaptive_override: Option<bool>,
) -> Tokenizer {
    let profile_detail = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some();
    let started_at = Instant::now();
    let regex = match (
        adaptive_override,
        profile_labels,
        residual_isolation_classes,
    ) {
        (Some(adaptive), Some(labels), Some(classes)) => {
            build_regex_partitioned_with_profile_labels_and_adaptive_and_residual_isolation(
                exprs,
                labels,
                partition_ids,
                classes,
                adaptive,
            )
        }
        (Some(adaptive), None, Some(classes)) => {
            build_regex_partitioned_with_adaptive_and_residual_isolation(
                exprs,
                partition_ids,
                classes,
                adaptive,
            )
        }
        (Some(adaptive), Some(labels), None) => {
            build_regex_partitioned_with_profile_labels_and_adaptive(
                exprs,
                labels,
                partition_ids,
                adaptive,
            )
        }
        (Some(adaptive), None, None) => {
            build_regex_partitioned_with_adaptive(exprs, partition_ids, adaptive)
        }
        (None, Some(labels), Some(classes)) => {
            build_regex_partitioned_with_profile_labels_and_residual_isolation(
                exprs,
                labels,
                partition_ids,
                classes,
            )
        }
        (None, None, Some(classes)) => build_regex_partitioned_with_residual_isolation(
            exprs,
            partition_ids,
            classes,
        ),
        (None, Some(labels), None) => {
            build_regex_partitioned_with_profile_labels(exprs, labels, partition_ids)
        }
        (None, None, None) => build_regex_partitioned(exprs, partition_ids),
    };
    if profile_detail {
        eprintln!(
            "[glrmask/profile][tokenizer] partitioned_build_done terminals={} partitions={} elapsed_ms={:.3} final_states={} final_transitions={}",
            exprs.len(),
            partition_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            elapsed_ms(started_at),
            regex.num_states(),
            regex.num_transitions(),
        );
    }
    regex.into_tokenizer(
        exprs.len() as u32,
        Some(std::sync::Arc::from(exprs.to_vec())),
    )
}

fn terminal_expr(terminal: &Terminal) -> Expr {
    match terminal {
        Terminal::Literal { bytes, .. } => Expr::U8Seq(bytes.clone()),
        Terminal::Pattern { pattern, utf8, .. } => parse_regex(pattern, *utf8),
        Terminal::Expr { expr, .. } => expr.clone(),
        Terminal::SpecialToken { .. } => Expr::Choice(Vec::new()),
    }
}

struct SyntheticTokenizerPlan {
    full_expressions: Vec<Expr>,
    synthesized_expressions: Vec<Expr>,
    changed_terminal_ids: Vec<u32>,
    partition_ids: Vec<u32>,
    residual_isolation_classes: Vec<Option<u32>>,
    changed_terminal_count: usize,
    repeat_horizons: Arc<crate::automata::lexer::compile::VocabularyRepeatHorizonCache>,
}

fn synthetic_state_reduction_is_profitable(full_states: usize, synthesized_states: usize) -> bool {
    // Certification walks the vocabulary from both state domains. Do not pay
    // that fixed cost for a modest quotient: the exact full tokenizer is
    // already the fallback and is usually faster at this scale. Keep this gate
    // deliberately conservative; it affects only whether the optimization is
    // attempted, never language correctness.
    const MIN_ABSOLUTE_STATE_SAVING: usize = 250_000;
    const MIN_REDUCTION_NUMERATOR: usize = 1;
    const MIN_REDUCTION_DENOMINATOR: usize = 2;

    full_states.saturating_sub(synthesized_states) >= MIN_ABSOLUTE_STATE_SAVING
        && synthesized_states.saturating_mul(MIN_REDUCTION_DENOMINATOR)
            <= full_states.saturating_mul(MIN_REDUCTION_NUMERATOR)
}

fn structural_state_reduction_is_profitable(
    full_states: usize,
    synthesized_states: usize,
) -> bool {
    const SMALL_SYNTHESIZED_COMPILE_STATES: usize = 20_000;
    const MIN_ABSOLUTE_STATE_SAVING: usize = 10_000;
    // A compile tokenizer this large still makes every downstream vocabulary
    // partition pay substantial state-classification and transition-table
    // costs. The exact fallback already has mature max-length/vocabulary
    // quotients, so reject large stencils rather than turning an attempted
    // optimization into a multi-second regression.
    const MAX_SYNTHESIZED_COMPILE_STATES: usize = 100_000;
    // A structurally certified composition can still be the safer compile
    // coordinate above the ordinary stencil limit when it replaces an exact
    // tokenizer that is larger by an order of magnitude. Keep this bounded so
    // large marginal reductions continue to use the mature exact path.
    const MAX_EXCEPTIONAL_SYNTHESIZED_COMPILE_STATES: usize = 250_000;
    const EXCEPTIONAL_REDUCTION_FACTOR: usize = 8;
    let is_reduction = synthesized_states < full_states;
    let small_compile_domain =
        is_reduction && synthesized_states <= SMALL_SYNTHESIZED_COMPILE_STATES;
    let substantial_large_reduction = full_states.saturating_sub(synthesized_states)
        >= MIN_ABSOLUTE_STATE_SAVING
        && (synthesized_states <= MAX_SYNTHESIZED_COMPILE_STATES
            || std::env::var_os("GLRMASK_ALLOW_LARGE_SYNTHETIC").is_some())
        && synthesized_states.saturating_mul(2) <= full_states;
    let exceptional_structural_reduction = synthesized_states
        <= MAX_EXCEPTIONAL_SYNTHESIZED_COMPILE_STATES
        && synthesized_states.saturating_mul(EXCEPTIONAL_REDUCTION_FACTOR) <= full_states;
    small_compile_domain || substantial_large_reduction || exceptional_structural_reduction
}

fn plan_synthetic_tokenizer(
    grammar: &GrammarDef,
    vocab: &Vocab,
) -> Option<SyntheticTokenizerPlan> {
    let profile = std::env::var_os("GLRMASK_PROFILE_SYNTHETIC_PLAN").is_some();
    let plan_started_at = Instant::now();
    let synthesis_enabled = crate::compiler::synthetic_bounded_terminals_enabled();
    let aggressive_partition_horizon =
        std::env::var_os("GLRMASK_AGGRESSIVE_PARTITION_HORIZON").is_some();
    let allow_vocab_only_candidates =
        std::env::var_os("GLRMASK_SYNTHETIC_VOCAB_ONLY_CANDIDATES").is_some();
    let has_candidate = aggressive_partition_horizon
        || allow_vocab_only_candidates
        || grammar_has_potential_bounded_terminal_synthesis(
            grammar,
            vocab.max_token_byte_len(),
        );
    if !has_candidate {
        if profile {
            eprintln!(
                "[glrmask/profile][synthetic_plan_detail] selected=false reason=no_pathological_shape terminals={} changed_terminals=0 full_expressions_ms=0.000 synthesize_ms=0.000 preflight_ms=0.000 total_ms={:.3}",
                grammar.terminals.len(),
                elapsed_ms(plan_started_at),
            );
        }
        return None;
    }
    if !synthesis_enabled {
        if profile {
            eprintln!(
                "[glrmask/profile][synthetic_plan_detail] selected=false reason=disabled terminals={} changed_terminals=0 full_expressions_ms=0.000 synthesize_ms=0.000 preflight_ms=0.000 total_ms={:.3}",
                grammar.terminals.len(),
                elapsed_ms(plan_started_at),
            );
        }
        return None;
    }
    plan_synthetic_tokenizer_enabled(
        grammar,
        vocab,
        profile,
        plan_started_at,
        aggressive_partition_horizon,
        allow_vocab_only_candidates,
    )
}

fn grammar_has_potential_bounded_terminal_synthesis(
    grammar: &GrammarDef,
    max_token_len: usize,
) -> bool {
    let mut scanner = BoundedTerminalCandidateScanner::new(max_token_len);
    grammar.terminals.iter().any(|terminal| match terminal {
        Terminal::Expr { expr, .. } => scanner.is_candidate(expr),
        Terminal::Pattern { pattern, utf8, .. } if pattern.as_bytes().contains(&b'{') => {
            let expr = parse_regex(pattern, *utf8);
            scanner.is_candidate(&expr)
        }
        Terminal::Literal { .. }
        | Terminal::Pattern { .. }
        | Terminal::SpecialToken { .. } => false,
    })
}

fn plan_synthetic_tokenizer_enabled(
    grammar: &GrammarDef,
    vocab: &Vocab,
    profile: bool,
    plan_started_at: Instant,
    aggressive_partition_horizon: bool,
    allow_vocab_only_candidates: bool,
) -> Option<SyntheticTokenizerPlan> {
    // The normal path uses exact vocabulary-relative repeat horizons. The
    // legacy fixed 64-byte candidate remains available only as an explicitly
    // unsafe diagnostic probe whose result must still pass full certification.
    const COMMON_PARTITION_HORIZON: usize = 64;
    let full_expressions_started_at = Instant::now();
    let full_expressions = grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .collect::<Vec<_>>();
    let full_expressions_ms = elapsed_ms(full_expressions_started_at);
    let repeat_horizons = Arc::new(
        crate::automata::lexer::compile::VocabularyRepeatHorizonCache::new(),
    );
    let synthesize_started_at = Instant::now();
    let mut synthesized = if aggressive_partition_horizon {
        // Experimental candidate generation may shorten terminals that are
        // reducible only at the common partition horizon. The resulting
        // tokenizer is never trusted directly: the full-vocabulary
        // certification below remains authoritative and rejects any candidate
        // whose shortened states are observable by a longer token.
        synthesize_terminal_expressions_for_horizon(
            &full_expressions,
            COMMON_PARTITION_HORIZON,
        )
    } else {
        synthesize_bounded_terminal_expressions(
            &full_expressions,
            vocab,
            repeat_horizons.as_ref(),
        )
    };
    let synthesize_ms = elapsed_ms(synthesize_started_at);

    if synthesized.changed_terminals.is_empty() {
        if profile {
            eprintln!(
                "[glrmask/profile][synthetic_plan_detail] selected=false terminals={} changed_terminals=0 full_expressions_ms={:.3} synthesize_ms={:.3} preflight_ms=0.000 total_ms={:.3}",
                grammar.terminals.len(),
                full_expressions_ms,
                synthesize_ms,
                elapsed_ms(plan_started_at),
            );
        }
        return None;
    }

    let preflight_started_at = Instant::now();
    if !aggressive_partition_horizon {
        const MAX_LOCAL_PREFLIGHT_ESTIMATE: u128 = 4_000_000;
        const MIN_LOCAL_STATE_SAVING: usize = 1_024;
        const MAX_SYNTHESIZED_RATIO_NUMERATOR: usize = 3;
        const MAX_SYNTHESIZED_RATIO_DENOMINATOR: usize = 4;

        let max_token_len = vocab.max_token_byte_len();
        let relevant_bytes = vocab.relevant_bytes();

        synthesized.changed_terminals.retain(|&terminal| {
            let terminal = terminal as usize;
            let full = &full_expressions[terminal];
            let candidate = &synthesized.expressions[terminal];
            if estimated_synthesis_state_volume(full) > MAX_LOCAL_PREFLIGHT_ESTIMATE {
                return true;
            }

            let full = factor_regex_expr(full.clone());
            let candidate = factor_regex_expr(candidate.clone());
            let pair = compile_terminal_expression_pair_with_structural_map(
                &full,
                &candidate,
                vocab,
                repeat_horizons.as_ref(),
                max_token_len,
                &relevant_bytes,
            );
            let keep = pair.as_ref().is_some_and(|pair| {
                let full_states = pair.full.num_states();
                let synthesized_states = pair.synthesized.num_states();
                synthesized_states < full_states
                    && full_states.saturating_sub(synthesized_states)
                        >= MIN_LOCAL_STATE_SAVING
                    && synthesized_states
                        .saturating_mul(MAX_SYNTHESIZED_RATIO_DENOMINATOR)
                        <= full_states.saturating_mul(MAX_SYNTHESIZED_RATIO_NUMERATOR)
            });
            if std::env::var_os("GLRMASK_PROFILE_SYNTHETIC_PLAN").is_some() {
                let (full_states, synthesized_states) = pair.as_ref().map_or((0, 0), |pair| {
                    (pair.full.num_states(), pair.synthesized.num_states())
                });
                eprintln!(
                    "[glrmask/profile][synthetic_preflight] terminal={} keep={} full_states={} synthesized_states={} absolute_saving={}",
                    terminal,
                    keep,
                    full_states,
                    synthesized_states,
                    full_states.saturating_sub(synthesized_states),
                );
            }
            if !keep {
                synthesized.expressions[terminal] = full_expressions[terminal].clone();
            }
            keep
        });
    }
    let preflight_ms = elapsed_ms(preflight_started_at);
    let changed_terminal_ids = synthesized.changed_terminals.clone();
    if changed_terminal_ids.is_empty() {
        if profile {
            eprintln!(
                "[glrmask/profile][synthetic_plan_detail] selected=false terminals={} changed_terminals=0 full_expressions_ms={:.3} synthesize_ms={:.3} preflight_ms={:.3} total_ms={:.3}",
                grammar.terminals.len(),
                full_expressions_ms,
                synthesize_ms,
                preflight_ms,
                elapsed_ms(plan_started_at),
            );
        }
        return None;
    }
    let changed_terminal_count = changed_terminal_ids.len();

    let mut residual_isolation_classes = lexer_residual_isolation_classes(grammar);
    let mut next_class = residual_isolation_classes
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |class| class.saturating_add(1));
    for &terminal in &changed_terminal_ids {
        residual_isolation_classes[terminal as usize] = Some(next_class);
        next_class = next_class
            .checked_add(1)
            .expect("residual isolation class id overflow");
    }
    let partition_ids =
        lexer_partition_ids_with_residual_classes(grammar, false, &residual_isolation_classes);

    if profile {
        let changed_terminal_labels = changed_terminal_ids
            .iter()
            .map(|&terminal| format!("{}:{}", terminal, grammar.terminal_display_name(terminal)))
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[glrmask/profile][synthetic_plan_detail] selected=true terminals={} changed_terminals={} changed_terminal_labels=[{}] full_expressions_ms={:.3} synthesize_ms={:.3} preflight_ms={:.3} total_ms={:.3}",
            grammar.terminals.len(),
            changed_terminal_count,
            changed_terminal_labels,
            full_expressions_ms,
            synthesize_ms,
            preflight_ms,
            elapsed_ms(plan_started_at),
        );
    }

    Some(SyntheticTokenizerPlan {
        full_expressions,
        synthesized_expressions: synthesized.expressions,
        changed_terminal_ids,
        partition_ids,
        residual_isolation_classes,
        changed_terminal_count,
        repeat_horizons,
    })
}

fn build_tokenizer_from_planned_expressions(
    grammar: &GrammarDef,
    plan: &SyntheticTokenizerPlan,
    expressions: &[Expr],
    adaptive_override: Option<bool>,
) -> Tokenizer {
    let expressions = expressions
        .iter()
        .cloned()
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let labels = grammar
        .terminals
        .iter()
        .enumerate()
        .map(|(index, _)| grammar.terminal_display_name(index as u32))
        .collect::<Vec<_>>();
    build_tokenizer_from_exprs_partitioned_impl(
        &expressions,
        Some(&labels),
        &plan.partition_ids,
        Some(&plan.residual_isolation_classes),
        adaptive_override,
    )
}

fn build_ordinary_compile_tokenizer(
    grammar: &GrammarDef,
    adaptive_override: Option<bool>,
) -> Tokenizer {
    adaptive_override.map_or_else(
        || build_tokenizer(grammar),
        |adaptive| build_tokenizer_with_partition_options(grammar, false, adaptive),
    )
}

enum DeferredRuntimeTokenizer {
    Ready(Tokenizer),
    Partitioned {
        full: DeferredPartitionedRegex,
        num_terminals: u32,
        expressions: Arc<[Expr]>,
        num_states: usize,
    },
}

impl DeferredRuntimeTokenizer {
    fn num_states(&self) -> usize {
        match self {
            Self::Ready(tokenizer) => tokenizer.num_states() as usize,
            Self::Partitioned { num_states, .. } => *num_states,
        }
    }

    fn default_start_delay_ms(&self) -> u64 {
        let has_deferred_materialization = match self {
            Self::Ready(_) => false,
            Self::Partitioned { full, .. } => full.has_deferred_runtime_materialization(),
        };
        if has_deferred_materialization && rayon::current_num_threads() >= 9 {
            100
        } else {
            0
        }
    }

    fn finish(self) -> Tokenizer {
        match self {
            Self::Ready(tokenizer) => tokenizer,
            Self::Partitioned {
                full,
                num_terminals,
                expressions,
                ..
            } => {
                let mut tokenizer =
                    full.finish_runtime_tokenizer(num_terminals, expressions);
                let nullable = tokenizer.isolate_start_state_and_drain_nullable_terminals();
                debug_assert!(
                    nullable.is_empty(),
                    "prepared protected residual components must be non-nullable"
                );
                tokenizer
            }
        }
    }
}

fn prepare_structural_tokenizer_pair(
    grammar: &GrammarDef,
    plan: &SyntheticTokenizerPlan,
    vocab: &Vocab,
    adaptive_override: Option<bool>,
) -> Option<(
    Tokenizer,
    DeferredRuntimeTokenizer,
    CertifiedFullToSynthesizedStateMap,
)> {
    // Importer-split residual terminals can overlap terminals in another
    // parser construction family. Their compiler contract deliberately
    // requires grammar-wide terminal observation, while the structural pair
    // proof below maps independently isolated lexer components. Do not extend
    // that proof across the stronger cross-family observation boundary.
    if grammar.requires_global_terminal_observation {
        return None;
    }
    let full_expressions = plan
        .full_expressions
        .iter()
        .cloned()
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let synthesized_expressions = plan
        .synthesized_expressions
        .iter()
        .cloned()
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let max_token_len = vocab.max_token_byte_len();
    let relevant_bytes = vocab.relevant_bytes();

    let expression_count = full_expressions.len() as u32;
    let (synthesized_regex, full, full_to_synthesized, effective_synthesized_expressions) =
        if full_expressions.len() == 1 {
            let pair = compile_terminal_expression_pair_with_structural_map(
                &full_expressions[0],
                &synthesized_expressions[0],
                vocab,
                plan.repeat_horizons.as_ref(),
                max_token_len,
                &relevant_bytes,
            )?;
            let mut full = pair.full.into_tokenizer(
                expression_count,
                Some(Arc::from(full_expressions.clone().into_boxed_slice())),
            );
            let full_nullable = full.isolate_start_state_and_drain_nullable_terminals();
            if !full_nullable.is_empty() {
                return None;
            }
            (
                pair.synthesized,
                DeferredRuntimeTokenizer::Ready(full),
                pair.full_to_synthesized,
                vec![pair.synthesized_expression],
            )
        } else {
            let labels = grammar
                .terminals
                .iter()
                .enumerate()
                .map(|(index, _)| grammar.terminal_display_name(index as u32))
                .collect::<Vec<_>>();
            let adaptive = adaptive_override
                .unwrap_or_else(|| env_flag_enabled_by_default("GLRMASK_LEXER_ADAPTIVE"));
            let pair = prepare_partitioned_expression_pair_with_structural_map(
                &full_expressions,
                &synthesized_expressions,
                Some(&labels),
                &plan.partition_ids,
                &plan.residual_isolation_classes,
                adaptive,
                vocab,
                plan.repeat_horizons.as_ref(),
                max_token_len,
                &relevant_bytes,
            )?;
            let full_num_states = pair.full_num_states();
            let (synthesized, full, full_to_synthesized, effective_synthesized_expressions) =
                pair.into_parts();
            (
                synthesized,
                DeferredRuntimeTokenizer::Partitioned {
                    full,
                    num_terminals: expression_count,
                    expressions: Arc::from(full_expressions.clone().into_boxed_slice()),
                    num_states: full_num_states,
                },
                full_to_synthesized,
                effective_synthesized_expressions,
            )
        };

    let mut synthesized = synthesized_regex.into_tokenizer(
        expression_count,
        Some(Arc::from(
            effective_synthesized_expressions.into_boxed_slice(),
        )),
    );
    let synthesized_nullable = synthesized.isolate_start_state_and_drain_nullable_terminals();
    if !synthesized_nullable.is_empty() {
        return None;
    }
    Some((
        synthesized,
        full,
        CertifiedFullToSynthesizedStateMap {
            full_to_synthesized,
        },
    ))
}

fn collect_special_token_terminals(grammar: &GrammarDef) -> Vec<SpecialTokenTerminal> {
    let mut specials = grammar
        .terminals
        .iter()
        .filter_map(|terminal| match terminal {
            Terminal::SpecialToken { id, token_id } => Some(SpecialTokenTerminal {
                terminal_id: *id,
                token_id: *token_id,
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    specials.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    specials
}

fn build_special_token_terminal_family(
    tokenizer: &Tokenizer,
    specials: &[SpecialTokenTerminal],
) -> Option<MappedArtifact<TerminalAutomaton>> {
    if specials.is_empty() {
        return None;
    }

    let mut token_ids = specials
        .iter()
        .map(|special| special.token_id)
        .collect::<Vec<_>>();
    token_ids.sort_unstable();
    token_ids.dedup();

    let max_token_id = *token_ids.last()? as usize;
    let mut original_token_to_internal = vec![u32::MAX; max_token_id + 1];
    for (internal, &token_id) in token_ids.iter().enumerate() {
        original_token_to_internal[token_id as usize] = internal as u32;
    }
    let vocab_tokens = ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
        original_token_to_internal,
        token_ids,
    );

    let initial_state = tokenizer.initial_state();
    let mut original_state_to_internal = vec![u32::MAX; tokenizer.num_states() as usize];
    original_state_to_internal[initial_state as usize] = 0;
    let tokenizer_states =
        ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
            original_state_to_internal,
            vec![initial_state],
        );
    let id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };

    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let final_state = dwa.add_state();
    dwa.set_final_weight(final_state, Weight::all());
    for special in specials {
        let internal_token = id_map.vocab_tokens.original_to_internal[special.token_id as usize];
        let tokens = RangeSetBlaze::from_iter([internal_token..=internal_token]);
        let weight = Weight::from_uniform(0..=0, tokens);
        dwa.add_transition(
            dwa.start_state(),
            special.terminal_id as i32,
            final_state,
            weight,
        );
    }

    Some(MappedArtifact::new(TerminalAutomaton::Dwa(dwa), id_map))
}

fn set_dense_bit(words: &mut [u64], token_id: u32) {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    if let Some(slot) = words.get_mut(word) {
        *slot |= 1u64 << bit;
    }
}

fn finalize_constraint(mut constraint: Constraint) -> Constraint {
    constraint.rebuild_runtime_caches();
    constraint
}

#[derive(Clone, Debug, Default)]
struct ParserTopAccept {
    combined: BTreeMap<i32, Weight>,
    parts: BTreeMap<i32, Vec<Weight>>,
    direct_l1_complete_by_terminal: BTreeMap<TerminalID, Weight>,
}

impl WeightRefs for ParserTopAccept {
    fn weight_refs(&self) -> Vec<&Weight> {
        let mut weights = self.combined.weight_refs();
        weights.extend(self.parts.weight_refs());
        weights.extend(self.direct_l1_complete_by_terminal.weight_refs());
        weights
    }

    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = self.combined.weight_refs_mut();
        weights.extend(self.parts.weight_refs_mut());
        weights.extend(self.direct_l1_complete_by_terminal.weight_refs_mut());
        weights
    }
}

type MappedParserDwa = MappedArtifact<(DWA, ParserTopAccept)>;

fn build_templates_for_compile(
    table: &GLRTable,
    analyzed_grammar: &AnalyzedGrammar,
    _ignore_terminal: Option<u32>,
) -> (
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
) {
    let templates_started_at = Instant::now();
    if analyzed_grammar.direct_regular_automaton.is_some()
        && !commit_template_dfas_enabled()
    {
        let templates_ms = elapsed_ms(templates_started_at);
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][templates_direct_regular] terminals={} skipped=true reason=no_commit_template_dfas total_ms={:.3}",
                analyzed_grammar.num_terminals,
                templates_ms,
            );
        }
        return (
            Templates::default(),
            vec![None; analyzed_grammar.num_terminals as usize],
            templates_ms,
        );
    }
    if analyzed_grammar.direct_regular_automaton.is_some()
        && let Some(templates) =
            Templates::from_direct_regular_table(table, analyzed_grammar.num_terminals)
    {
        let mut template_dfas_by_terminal = vec![None; analyzed_grammar.num_terminals as usize];
        let commit_template_dfas_enabled = commit_template_dfas_enabled();
        let mut commit_template_dfas_built = 0usize;
        if commit_template_dfas_enabled {
            for (&terminal, dfa) in &templates.by_terminal {
                if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
                    let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
                    if let Some(split_commit_dfas) =
                        try_split_commit_template_dfas(&commit_dfa)
                    {
                        *slot = Some(Arc::new(split_commit_dfas));
                        commit_template_dfas_built += 1;
                    }
                }
            }
        }
        let dfa_states = templates
            .by_terminal
            .values()
            .map(|dfa| dfa.states.len())
            .sum::<usize>();
        let dfa_transitions = templates
            .by_terminal
            .values()
            .flat_map(|dfa| dfa.states.iter())
            .map(|state| state.transitions.len())
            .sum::<usize>();
        let templates_ms = elapsed_ms(templates_started_at);
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][templates_direct_regular] terminals={} dfa_states={} dfa_transitions={} commit_template_dfas_enabled={} commit_template_dfas_built={} total_ms={:.3}",
                templates.by_terminal.len(),
                dfa_states,
                dfa_transitions,
                commit_template_dfas_enabled,
                commit_template_dfas_built,
                templates_ms,
            );
        }
        return (templates, template_dfas_by_terminal, templates_ms);
    }

    let (characterizations, characterization_profile) =
        characterize_terminals_profiled(table, analyzed_grammar);
    let (templates, template_profile) =
        Templates::from_characterizations_profiled(&characterizations);
    let mut template_dfas_by_terminal = vec![None; analyzed_grammar.num_terminals as usize];
    let commit_template_dfas_enabled = commit_template_dfas_enabled();
    let mut commit_template_dfas_built = 0usize;
    let mut commit_template_dfas_skipped = 0usize;
    let mut commit_template_specialize_ms = 0.0;
    let mut commit_template_split_ms = 0.0;
    if commit_template_dfas_enabled {
        for (&terminal, dfa) in &templates.by_terminal {
            if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
                let specialize_started_at = Instant::now();
                let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
                commit_template_specialize_ms += elapsed_ms(specialize_started_at);
                let split_started_at = Instant::now();
                let split_commit_dfas = try_split_commit_template_dfas(&commit_dfa);
                commit_template_split_ms += elapsed_ms(split_started_at);
                let Some(split_commit_dfas) = split_commit_dfas else {
                    commit_template_dfas_skipped += 1;
                    continue;
                };
                *slot = Some(Arc::new(split_commit_dfas));
                commit_template_dfas_built += 1;
            }
        }
    }
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][templates] terminals={} action_signature_classes={} action_quotient_hits={} max_action_signature_multiplicity={} characterization_signature_ms={:.3} characterization_ms={:.3} characterization_fanout_ms={:.3} characterization_validation_ms={:.3} characterization_total_ms={:.3} characterization_quotient_disabled={} unique_characterizations={} compiled_characterizations={} template_quotient_hits={} max_characterization_multiplicity={} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} template_fanout_ms={:.3} template_validation_ms={:.3} template_total_ms={:.3} template_wall_ms={:.3} template_minimize_skipped={} avg_nfa_states={:.2} avg_nfa_transitions={:.2} avg_premin_dfa_states={:.2} avg_premin_dfa_transitions={:.2} avg_dfa_states={:.2} avg_dfa_transitions={:.2} max_dfa_states={} max_dfa_transitions={} commit_template_dfas_enabled={} commit_template_dfas_built={} commit_template_dfas_skipped={} commit_template_specialize_ms={:.3} commit_template_split_ms={:.3}",
            characterization_profile.terminals,
            characterization_profile.unique_action_signatures,
            characterization_profile.quotient_hits,
            characterization_profile.max_action_signature_multiplicity,
            characterization_profile.signature_ms,
            characterization_profile.characterize_ms,
            characterization_profile.fanout_ms,
            characterization_profile.validation_ms,
            characterization_profile.total_ms,
            characterization_profile.quotient_disabled,
            template_profile.unique_characterizations,
            template_profile.compiled_characterizations,
            template_profile.quotient_hits,
            template_profile.max_characterization_multiplicity,
            template_profile.build_nfa_ms,
            template_profile.determinize_ms,
            template_profile.minimize_ms,
            template_profile.fanout_ms,
            template_profile.validation_ms,
            template_profile.total_ms,
            template_profile.wall_ms,
            template_profile.minimize_skipped,
            template_profile.avg_nfa_states(),
            template_profile.avg_nfa_transitions(),
            template_profile.avg_premin_dfa_states(),
            template_profile.avg_premin_dfa_transitions(),
            template_profile.avg_dfa_states(),
            template_profile.avg_dfa_transitions(),
            template_profile.max_dfa_states,
            template_profile.max_dfa_transitions,
            commit_template_dfas_enabled,
            commit_template_dfas_built,
            commit_template_dfas_skipped,
            commit_template_specialize_ms,
            commit_template_split_ms,
        );
    }
    (
        templates,
        template_dfas_by_terminal,
        elapsed_ms(templates_started_at),
    )
}

#[derive(Clone)]
struct TokenizerDagLane {
    tokenizer: Arc<Tokenizer>,
    partition_local_synthesis_plan: Option<Arc<
        crate::compiler::stages::id_map_and_terminal_dwa::PartitionLocalSynthesisPlan,
    >>,
    prepared_partition_local_tokenizers: Option<Arc<
        crate::compiler::stages::id_map_and_terminal_dwa::PreparedPartitionLocalTokenizers,
    >>,
    synthetic_candidate_terminals: usize,
    synthetic_certification_ms: f64,
    compile_tokenizer_states: usize,
    compile_tokenizer_transitions: usize,
    tokenizer_build_ms: f64,
    tokenizer_ready_ms: f64,
}

struct RuntimeTokenizerDagResult {
    runtime_tokenizer: Option<Tokenizer>,
    full_to_synthesized_state_map: Option<CertifiedFullToSynthesizedStateMap>,
    finish_ms: f64,
}

struct FlatGlobalDagLane {
    flat_trans: Arc<[u32]>,
    shared_transition_cache: Arc<
        std::sync::OnceLock<
            crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::FlatTransitionCache,
        >,
    >,
    flat_trans_ms: f64,
    global_max_length_state_map: ManyToOneIdMap,
    global_max_length_ms: f64,
    started_ms: f64,
    finished_ms: f64,
}

#[derive(Clone)]
struct AnalysisDagLane {
    analyzed_grammar: Arc<AnalyzedGrammar>,
    analyze_grammar_ms: f64,
    disallowed_follows: Arc<BTreeMap<u32, BitSet>>,
    always_allowed_follows: Arc<[Vec<u32>]>,
    disallowed_follows_ms: f64,
    analysis_ready_ms: f64,
}

struct ClassifyDagLane {
    shared_classify_cache: SharedClassifyCache,
    classify_ms: f64,
    started_ms: f64,
    finished_ms: f64,
}

struct ColoringDagLane {
    terminal_coloring: TerminalColoring,
    terminal_coloring_ms: f64,
}

#[derive(Default)]
struct TerminalDagJoinState {
    tokenizer: Option<TokenizerDagLane>,
    flat_global: Option<FlatGlobalDagLane>,
    analysis: Option<AnalysisDagLane>,
    classify: Option<ClassifyDagLane>,
    coloring: Option<ColoringDagLane>,
    classify_launched: bool,
    terminal_launched: bool,
}

struct TerminalDagResult {
    tokenizer: TokenizerDagLane,
    analysis: AnalysisDagLane,
    ignore_terminal: Option<u32>,
    terminal_coloring_ms: f64,
    terminal_dwas: TerminalDwaFamilies,
    terminal_phase_profile: TerminalDwaPhaseProfile,
    classify_ms: f64,
    flat_trans_ms: f64,
    global_max_length_ms: f64,
    flat_global_started_ms: f64,
    flat_global_finished_ms: f64,
    classify_started_ms: f64,
    classify_finished_ms: f64,
    terminal_dwa_started_ms: f64,
    terminal_dwa_finished_ms: f64,
}

struct TemplatesDagResult {
    table: Arc<GLRTable>,
    glr_table_ms: f64,
    glr_ready_ms: f64,
    templates: Templates,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    templates_ms: f64,
    templates_started_ms: f64,
    templates_finished_ms: f64,
}

#[derive(Default)]
struct ParserDagJoinState {
    terminal: Option<TerminalDagResult>,
    templates: Option<TemplatesDagResult>,
    launched: bool,
}

struct CompileDagResult {
    tokenizer: Arc<Tokenizer>,
    synthetic_candidate_terminals: usize,
    synthetic_certification_ms: f64,
    compile_tokenizer_states: usize,
    compile_tokenizer_transitions: usize,
    tokenizer_build_ms: f64,
    tokenizer_ready_ms: f64,
    analyzed_grammar: Arc<AnalyzedGrammar>,
    analyze_grammar_ms: f64,
    disallowed_follows_ms: f64,
    analysis_ready_ms: f64,
    table: Arc<GLRTable>,
    glr_table_ms: f64,
    glr_ready_ms: f64,
    terminal_coloring_ms: f64,
    terminal_dwas: TerminalDwaFamilies,
    terminal_phase_profile: TerminalDwaPhaseProfile,
    templates: Option<Templates>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    templates_ms: f64,
    classify_ms: f64,
    flat_trans_ms: f64,
    global_max_length_ms: f64,
    flat_global_started_ms: f64,
    flat_global_finished_ms: f64,
    classify_started_ms: f64,
    classify_finished_ms: f64,
    terminal_dwa_started_ms: f64,
    terminal_dwa_finished_ms: f64,
    templates_started_ms: f64,
    templates_finished_ms: f64,
    terminal_run_collapse_ms: f64,
    prebuilt_parser_dwa: Option<(MappedParserDwa, f64, f64, f64)>,
}

fn build_parser_dwa_for_terminal_family(
    family_name: &str,
    family: Option<&MappedArtifact<TerminalAutomaton>>,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    vocab: &Vocab,
    collapse_immediate_acceptance: bool,
) -> Option<MappedArtifact<DWA>> {
    let family = family?;
    let internal_ids = family.id_map().clone();
    let (parser_dwa, immediate_fast_path) =
        if let Some(parser_dwa) = try_build_immediate_parser_dwa(family.artifact(), grammar, table) {
            (parser_dwa, true)
        } else {
            (
                build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                    table,
                    grammar,
                    family.artifact(),
                    templates,
                    vocab,
                    &internal_ids,
                    collapse_immediate_acceptance,
                ),
                false,
            )
        };
    if family_name == "l1"
        && family.artifact().num_states() == 2
        && table.admission_policy
            == crate::compiler::glr::table::AdmissionPolicy::RowPresenceExact
        && parser_dwa.num_transitions() > 0
    {
        debug_assert_eq!(parser_dwa.states().len(), 2, "L1 parser DWA must be depth one");
        let start = parser_dwa.start_state() as usize;
        let final_state = 1usize.wrapping_sub(start);
        debug_assert!(parser_dwa.states()[start]
            .transitions
            .values()
            .all(|(target, weight)| *target as usize == final_state && !weight.is_empty()));
        debug_assert!(parser_dwa.states()[final_state].transitions.is_empty());
        debug_assert!(parser_dwa.states()[final_state]
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty()));
    }
    if compile_profile_enabled() {
        let terminal_stats = family.artifact().stats();
        let parser_stats = parser_dwa.stats();
        eprintln!(
            "[glrmask/profile][parser_dwa_family] family={} terminal_states={} terminal_transitions={} parser_states={} parser_transitions={} immediate_fast_path={}",
            family_name,
            terminal_stats.states,
            terminal_stats.transitions,
            parser_stats.states,
            parser_stats.transitions,
            immediate_fast_path,
        );
    }
    Some(MappedArtifact::new(parser_dwa, internal_ids))
}

fn merge_parser_top_accept(mut left: ParserTopAccept, right: ParserTopAccept) -> ParserTopAccept {
    for (label, weight) in right.combined {
        left.combined
            .entry(label)
            .and_modify(|existing| *existing = existing.union(&weight))
            .or_insert(weight);
    }
    for (label, mut parts) in right.parts {
        left.parts.entry(label).or_default().append(&mut parts);
    }
    for (terminal, weight) in right.direct_l1_complete_by_terminal {
        left.direct_l1_complete_by_terminal
            .entry(terminal)
            .and_modify(|existing| *existing = existing.union(&weight))
            .or_insert(weight);
    }
    left
}

// Directly replaying a large acceptance-part list is linear in the number of
// terminal-family certificates. Collapse only near-universal lists: this catches
// broad parser rows while avoiding the expensive bulk union of ordinary rows.
const PARSER_TOP_ACCEPT_COMPILE_UNION_MIN_PARTS: usize = 128;
const PARSER_TOP_ACCEPT_COMPILE_UNION_MAX_MISSING_TERMINALS: usize = 5;

#[derive(Clone, Copy, Debug, Default)]
struct ParserTopAcceptCollapseReport {
    labels_collapsed: usize,
    part_refs_before: usize,
    part_refs_after: usize,
    max_parts_before: usize,
    unique_unions_built: usize,
}

fn collapse_huge_parser_top_accept_parts(
    top_accept: &mut ParserTopAccept,
    num_terminals: usize,
) -> ParserTopAcceptCollapseReport {
    let mut report = ParserTopAcceptCollapseReport {
        part_refs_before: top_accept.parts.values().map(Vec::len).sum(),
        max_parts_before: top_accept.parts.values().map(Vec::len).max().unwrap_or(0),
        ..ParserTopAcceptCollapseReport::default()
    };
    let mut union_cache = rustc_hash::FxHashMap::<Vec<usize>, Weight>::default();
    let parts = std::mem::take(&mut top_accept.parts);
    for (label, mut weights) in parts {
        weights.sort_unstable_by_key(Weight::ptr_key);
        weights.dedup_by_key(|weight| weight.ptr_key());
        let near_universal = weights.len()
            .saturating_add(PARSER_TOP_ACCEPT_COMPILE_UNION_MAX_MISSING_TERMINALS)
            >= num_terminals;
        if weights.len() < PARSER_TOP_ACCEPT_COMPILE_UNION_MIN_PARTS || !near_universal {
            report.part_refs_after += weights.len();
            top_accept.parts.insert(label, weights);
            continue;
        }
        let key = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
        let union = if let Some(cached) = union_cache.get(&key) {
            cached.clone()
        } else {
            let union = Weight::union_all_direct(weights.iter());
            union_cache.insert(key, union.clone());
            report.unique_unions_built += 1;
            union
        };
        top_accept
            .combined
            .entry(label)
            .and_modify(|existing| *existing = existing.union(&union))
            .or_insert(union);
        report.labels_collapsed += 1;
    }
    report
}

fn reconcile_parser_top_accept_parts(
    mut inputs: Vec<MappedArtifact<ParserTopAccept>>,
) -> Option<MappedArtifact<ParserTopAccept>> {
    if inputs.is_empty() {
        return None;
    }
    inputs.sort_unstable_by_key(|mapped| std::cmp::Reverse(mapped.id_map().num_tsids()));
    let mut merged = inputs.remove(0);
    for next in inputs {
        let ((left, right), id_map) = merged.pair_forced_common(next).into_parts();
        merged = MappedArtifact::new(merge_parser_top_accept(left, right), id_map);
    }
    Some(merged)
}

fn build_and_merge_parser_dwa_families(
    terminal_dwas: &TerminalDwaFamilies,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    _ignore_terminal: Option<u32>,
    templates: Templates,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> MappedParserDwa {
    let total_started_at = Instant::now();
    let collapse_immediate_acceptance = !tokenizer.has_epsilon_transitions();
    let direct_l1_parts = grammar
        .direct_regular_automaton
        .as_ref()
        .and_then(|_| terminal_dwas.l1.as_ref())
        .and_then(|family| {
            try_build_immediate_terminal_completion_weights(
                family.artifact(),
                grammar,
                table,
            )
            .map(|direct_l1_complete_by_terminal| {
                MappedArtifact::new(
                    ParserTopAccept {
                        combined: BTreeMap::new(),
                        parts: BTreeMap::new(),
                        direct_l1_complete_by_terminal,
                    },
                    family.id_map().clone(),
                )
            })
        });
    let direct_l2p_parts = grammar
        .direct_regular_automaton
        .as_ref()
        .and_then(|_| terminal_dwas.l2p.as_ref())
        .and_then(|family| {
            try_build_direct_regular_parser_top_accept_parts(
                family.artifact(),
                grammar,
                table,
            )
            .map(|parts| {
                MappedArtifact::new(
                    ParserTopAccept {
                        combined: BTreeMap::new(),
                        parts,
                        direct_l1_complete_by_terminal: BTreeMap::new(),
                    },
                    family.id_map().clone(),
                )
            })
        });
    let direct_special_parts = grammar
        .direct_regular_automaton
        .as_ref()
        .and_then(|_| terminal_dwas.special.as_ref())
        .and_then(|family| {
            try_build_direct_regular_parser_top_accept_parts(
                family.artifact(),
                grammar,
                table,
            )
            .map(|parts| {
                MappedArtifact::new(
                    ParserTopAccept {
                        combined: BTreeMap::new(),
                        parts,
                        direct_l1_complete_by_terminal: BTreeMap::new(),
                    },
                    family.id_map().clone(),
                )
            })
        });
    let use_direct_l1_parts = direct_l1_parts.is_some();
    let use_direct_l2p_parts = direct_l2p_parts.is_some();
    let use_direct_special_parts = direct_special_parts.is_some();
    if grammar.direct_regular_automaton.is_some() && templates.by_terminal.is_empty() {
        assert_eq!(
            terminal_dwas.l1.is_some(),
            use_direct_l1_parts,
            "direct-regular L1 family must have an exact direct acceptance path",
        );
        assert_eq!(
            terminal_dwas.l2p.is_some(),
            use_direct_l2p_parts,
            "direct-regular L2P family must have an exact direct acceptance path",
        );
        assert_eq!(
            terminal_dwas.special.is_some(),
            use_direct_special_parts,
            "direct-regular special-token family must have an exact direct acceptance path",
        );
    }

    let (l1_parser, l2p_parser) = rayon::join(
        || {
            (!use_direct_l1_parts)
                .then(|| {
                    build_parser_dwa_for_terminal_family(
                        "l1",
                        terminal_dwas.l1.as_ref(),
                        table,
                        grammar,
                        &templates,
                        vocab,
                        collapse_immediate_acceptance,
                    )
                })
                .flatten()
        },
        || {
            (!use_direct_l2p_parts)
                .then(|| {
                    build_parser_dwa_for_terminal_family(
                        "l2p",
                        terminal_dwas.l2p.as_ref(),
                        table,
                        grammar,
                        &templates,
                        vocab,
                        collapse_immediate_acceptance,
                    )
                })
                .flatten()
        },
    );
    let special_parser = (!use_direct_special_parts)
        .then(|| {
            build_parser_dwa_for_terminal_family(
                "special",
                terminal_dwas.special.as_ref(),
                table,
                grammar,
                &templates,
                vocab,
                collapse_immediate_acceptance,
            )
        })
        .flatten();
    let parser_dwas: Vec<MappedArtifact<DWA>> = l1_parser
        .into_iter()
        .chain(l2p_parser)
        .chain(special_parser)
        .collect();
    let direct_parts = reconcile_parser_top_accept_parts(
        direct_l1_parts
            .into_iter()
            .chain(direct_l2p_parts)
            .chain(direct_special_parts)
            .collect(),
    );
    let max_token_id = terminal_dwas
        .max_original_token_id()
        .unwrap_or_else(|| vocab.max_token_id())
        .max(vocab.max_token_id());

    if let Some(mapped_parts) = direct_parts {
        let (dwa, top_accept, id_map, parts_first) = if parser_dwas.is_empty() {
            let (top_accept, id_map) = mapped_parts.into_parts();
            (
                DWA::new(id_map.num_tsids(), id_map.max_internal_token_id()),
                top_accept,
                id_map,
                true,
            )
        } else {
            let mapped_dwa =
                crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_mapped_parser_dwas(
                    parser_dwas,
                    tokenizer.num_states() as usize,
                    max_token_id,
                );
            let parts_first = mapped_parts.id_map().num_tsids() > mapped_dwa.id_map().num_tsids();
            if parts_first {
                let ((top_accept, dwa), id_map) =
                    mapped_parts.pair_forced_common(mapped_dwa).into_parts();
                (dwa, top_accept, id_map, true)
            } else {
                let ((dwa, top_accept), id_map) =
                    mapped_dwa.pair_forced_common(mapped_parts).into_parts();
                (dwa, top_accept, id_map, false)
            }
        };
        let mut top_accept = top_accept;
        let collapse_started_at = Instant::now();
        let collapse_report = collapse_huge_parser_top_accept_parts(
            &mut top_accept,
            grammar.num_terminals as usize,
        );
        let collapse_ms = elapsed_ms(collapse_started_at);
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_dwa_merge] mode=top_accept_parts l1_direct={} l2p_direct={} special_direct={} id_order={} labels={} part_refs={} unique_part_weights={} huge_collapsed_labels={} huge_part_refs_before={} huge_part_refs_after={} huge_max_parts_before={} huge_unique_unions={} huge_collapse_ms={:.3} states={} transitions={} total_ms={:.3}",
                use_direct_l1_parts,
                use_direct_l2p_parts,
                use_direct_special_parts,
                if parts_first { "parts" } else { "primary" },
                top_accept.parts.len(),
                top_accept.parts.values().map(Vec::len).sum::<usize>(),
                top_accept
                    .parts
                    .values()
                    .flatten()
                    .map(Weight::ptr_key)
                    .collect::<rustc_hash::FxHashSet<_>>()
                    .len(),
                collapse_report.labels_collapsed,
                collapse_report.part_refs_before,
                collapse_report.part_refs_after,
                collapse_report.max_parts_before,
                collapse_report.unique_unions_built,
                collapse_ms,
                dwa.num_states(),
                dwa.num_transitions(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return MappedArtifact::new((dwa, top_accept), id_map);
    }

    let (mapped_dwa, combined) =
        crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_mapped_parser_dwas_with_top_accept(
            parser_dwas,
            tokenizer.num_states() as usize,
            max_token_id,
        );
    let (dwa, id_map) = mapped_dwa.into_parts();
    MappedArtifact::new(
        (
            dwa,
            ParserTopAccept {
                combined,
                parts: BTreeMap::new(),
                direct_l1_complete_by_terminal: BTreeMap::new(),
            },
        ),
        id_map,
    )
}

#[derive(Clone, Copy)]
struct TerminalFamilyLayout {
    has_l1: bool,
    has_l2p: bool,
    has_special: bool,
}

fn reconcile_terminal_dwa_families(
    families: TerminalDwaFamilies,
) -> (MappedArtifact<Vec<TerminalAutomaton>>, TerminalFamilyLayout) {
    let layout = TerminalFamilyLayout {
        has_l1: families.l1.is_some(),
        has_l2p: families.l2p.is_some(),
        has_special: families.special.is_some(),
    };
    let mapped = MappedArtifact::reconcile_vec(families.into_vec());
    (mapped, layout)
}

fn restore_terminal_dwa_families(
    mapped: MappedArtifact<Vec<TerminalAutomaton>>,
    layout: TerminalFamilyLayout,
) -> TerminalDwaFamilies {
    let mut pieces = mapped.split_vec().into_iter();
    let l1 = layout.has_l1.then(|| {
        pieces
            .next()
            .expect("L1 terminal family missing after reconciliation")
    });
    let l2p = layout.has_l2p.then(|| {
        pieces
            .next()
            .expect("L2P terminal family missing after reconciliation")
    });
    let special = layout.has_special.then(|| {
        pieces
            .next()
            .expect("special-token terminal family missing after reconciliation")
    });
    assert!(
        pieces.next().is_none(),
        "unexpected extra terminal family after reconciliation"
    );
    TerminalDwaFamilies { l1, l2p, special }
}

fn terminal_family_interned_range_count(families: &TerminalDwaFamilies) -> usize {
    let mut weights = Vec::new();
    if let Some(l1) = &families.l1 {
        weights.extend(l1.artifact().weight_refs());
    }
    if let Some(l2p) = &families.l2p {
        weights.extend(l2p.artifact().weight_refs());
    }
    if let Some(special) = &families.special {
        weights.extend(special.artifact().weight_refs());
    }
    count_interned_ranges_for_weights(weights).total_ranges()
}

fn terminal_family_joint_interned_range_count<T: WeightRefs>(
    families: &TerminalDwaFamilies,
    other: &T,
) -> usize {
    let mut weights = Vec::new();
    if let Some(l1) = &families.l1 {
        weights.extend(l1.artifact().weight_refs());
    }
    if let Some(l2p) = &families.l2p {
        weights.extend(l2p.artifact().weight_refs());
    }
    if let Some(special) = &families.special {
        weights.extend(special.artifact().weight_refs());
    }
    weights.extend(other.weight_refs());
    count_interned_ranges_for_weights(weights).total_ranges()
}

fn launch_parser_dag_if_ready<'scope>(
    scope: &rayon::Scope<'scope>,
    parser_state: &'scope Mutex<ParserDagJoinState>,
    result: &'scope Mutex<Option<CompileDagResult>>,
    vocab: &'scope Vocab,
    dwa_pm_mode: DwaPossibleMatchesMode,
    compile_started_at: Instant,
) {
    let ready = {
        let mut state = parser_state.lock().expect("parser DAG join state poisoned");
        if state.launched || state.terminal.is_none() || state.templates.is_none() {
            None
        } else {
            state.launched = true;
            Some((
                state.terminal.take().expect("terminal DAG result ready"),
                state.templates.take().expect("templates DAG result ready"),
            ))
        }
    };

    let Some((terminal, templates)) = ready else {
        return;
    };

    scope.spawn(move |_| {
        let TerminalDagResult {
            tokenizer,
            analysis,
            ignore_terminal,
            terminal_coloring_ms,
            mut terminal_dwas,
            terminal_phase_profile,
            classify_ms,
            flat_trans_ms,
            global_max_length_ms,
            flat_global_started_ms,
            flat_global_finished_ms,
            classify_started_ms,
            classify_finished_ms,
            terminal_dwa_started_ms,
            terminal_dwa_finished_ms,
        } = terminal;
        let TemplatesDagResult {
            table,
            glr_table_ms,
            glr_ready_ms,
            templates,
            template_dfas_by_terminal,
            templates_ms,
            templates_started_ms,
            templates_finished_ms,
        } = templates;

        let terminal_run_collapse_started_at = Instant::now();
        let terminal_run_collapse_profile =
            crate::compiler::terminal_run_collapse::collapse_certified_terminal_runs(
                &mut terminal_dwas,
                &table,
                &analysis.analyzed_grammar,
                &templates,
                vocab,
            );
        let terminal_run_collapse_ms = elapsed_ms(terminal_run_collapse_started_at);
        debug_assert!(
            terminal_run_collapse_ms + 0.001
                >= terminal_run_collapse_profile.certificate_ms
                    + terminal_run_collapse_profile.rewrite_ms
        );

        let (templates, prebuilt_parser_dwa) = if dwa_pm_mode.does_terminal_reconcile() {
            (Some(templates), None)
        } else {
            let parser_dwa_started_at = Instant::now();
            let parser_dwa_started_ms = elapsed_ms(compile_started_at.clone());
            let parser_dwa = build_and_merge_parser_dwa_families(
                &terminal_dwas,
                &table,
                &analysis.analyzed_grammar,
                ignore_terminal,
                templates,
                &tokenizer.tokenizer,
                vocab,
            );
            let parser_dwa_ms = elapsed_ms(parser_dwa_started_at);
            let parser_dwa_finished_ms = elapsed_ms(compile_started_at);
            (
                None,
                Some((
                    parser_dwa,
                    parser_dwa_ms,
                    parser_dwa_started_ms,
                    parser_dwa_finished_ms,
                )),
            )
        };

        *result.lock().expect("compile DAG result poisoned") = Some(CompileDagResult {
            tokenizer: tokenizer.tokenizer,
            synthetic_candidate_terminals: tokenizer.synthetic_candidate_terminals,
            synthetic_certification_ms: tokenizer.synthetic_certification_ms,
            compile_tokenizer_states: tokenizer.compile_tokenizer_states,
            compile_tokenizer_transitions: tokenizer.compile_tokenizer_transitions,
            tokenizer_build_ms: tokenizer.tokenizer_build_ms,
            tokenizer_ready_ms: tokenizer.tokenizer_ready_ms,
            analyzed_grammar: analysis.analyzed_grammar,
            analyze_grammar_ms: analysis.analyze_grammar_ms,
            disallowed_follows_ms: analysis.disallowed_follows_ms,
            analysis_ready_ms: analysis.analysis_ready_ms,
            table,
            glr_table_ms,
            glr_ready_ms,
            terminal_coloring_ms,
            terminal_dwas,
            terminal_phase_profile,
            templates,
            template_dfas_by_terminal,
            templates_ms,
            classify_ms,
            flat_trans_ms,
            global_max_length_ms,
            flat_global_started_ms,
            flat_global_finished_ms,
            classify_started_ms,
            classify_finished_ms,
            terminal_dwa_started_ms,
            terminal_dwa_finished_ms,
            templates_started_ms,
            templates_finished_ms,
            terminal_run_collapse_ms,
            prebuilt_parser_dwa,
        });
    });
}

fn launch_terminal_dag_if_ready<'scope>(
    scope: &rayon::Scope<'scope>,
    terminal_state: &'scope Mutex<TerminalDagJoinState>,
    parser_state: &'scope Mutex<ParserDagJoinState>,
    result: &'scope Mutex<Option<CompileDagResult>>,
    prepared_grammar: &'scope GrammarDef,
    vocab: &'scope Vocab,
    dwa_pm_mode: DwaPossibleMatchesMode,
    use_terminal_coloring: bool,
    compile_started_at: Instant,
) {
    let ready = {
        let mut state = terminal_state.lock().expect("terminal DAG join state poisoned");
        let coloring_ready = !use_terminal_coloring || state.coloring.is_some();
        if state.terminal_launched
            || state.tokenizer.is_none()
            || state.flat_global.is_none()
            || state.analysis.is_none()
            || state.classify.is_none()
            || !coloring_ready
        {
            None
        } else {
            state.terminal_launched = true;
            Some((
                state.tokenizer.take().expect("tokenizer DAG result ready"),
                state.flat_global.take().expect("flat/global DAG result ready"),
                state.analysis.take().expect("analysis DAG result ready"),
                state.classify.take().expect("classification DAG result ready"),
                state.coloring.take(),
            ))
        }
    };

    let Some((tokenizer, flat_global, analysis, classify, coloring)) = ready else {
        return;
    };

    scope.spawn(move |scope| {
        let ColoringDagLane { terminal_coloring, terminal_coloring_ms } = coloring.unwrap_or_else(|| {
            ColoringDagLane {
                terminal_coloring: TerminalColoring::identity(analysis.analyzed_grammar.num_terminals as usize),
                terminal_coloring_ms: 0.0,
            }
        });
        let terminal_dwa_started_ms = elapsed_ms(compile_started_at.clone());
        let (mut terminal_dwas, mut terminal_phase_profile) =
            crate::compiler::stages::id_map_and_terminal_dwa::build_terminal_dwa_families_with_precomputed_global_max_length(
                &tokenizer.tokenizer,
                vocab,
                &terminal_coloring,
                use_terminal_coloring,
                prepared_grammar.ignore_terminal,
                &analysis.analyzed_grammar,
                &analysis.disallowed_follows,
                Some(&analysis.always_allowed_follows),
                Arc::clone(&flat_global.flat_trans),
                &flat_global.global_max_length_state_map,
                Some(&classify.shared_classify_cache),
                Some(flat_global.shared_transition_cache.as_ref()),
                tokenizer.partition_local_synthesis_plan.as_deref(),
                tokenizer.prepared_partition_local_tokenizers.as_deref(),
                None,
            );
        let special_started_at = Instant::now();
        let special_token_terminals = collect_special_token_terminals(prepared_grammar);
        terminal_dwas.special = build_special_token_terminal_family(
            &tokenizer.tokenizer,
            &special_token_terminals,
        );
        terminal_phase_profile.terminal_dwa_ms += elapsed_ms(special_started_at);
        let terminal_dwa_finished_ms = elapsed_ms(compile_started_at.clone());

        parser_state
            .lock()
            .expect("parser DAG join state poisoned")
            .terminal = Some(TerminalDagResult {
                tokenizer,
                analysis,
                ignore_terminal: prepared_grammar.ignore_terminal,
                terminal_coloring_ms,
                terminal_dwas,
                terminal_phase_profile,
                classify_ms: classify.classify_ms,
                flat_trans_ms: flat_global.flat_trans_ms,
                global_max_length_ms: flat_global.global_max_length_ms,
                flat_global_started_ms: flat_global.started_ms,
                flat_global_finished_ms: flat_global.finished_ms,
                classify_started_ms: classify.started_ms,
                classify_finished_ms: classify.finished_ms,
                terminal_dwa_started_ms,
                terminal_dwa_finished_ms,
            });
        launch_parser_dag_if_ready(
            scope,
            parser_state,
            result,
            vocab,
            dwa_pm_mode,
            compile_started_at,
        );
    });
}

fn launch_classify_dag_if_ready<'scope>(
    scope: &rayon::Scope<'scope>,
    terminal_state: &'scope Mutex<TerminalDagJoinState>,
    parser_state: &'scope Mutex<ParserDagJoinState>,
    result: &'scope Mutex<Option<CompileDagResult>>,
    prepared_grammar: &'scope GrammarDef,
    vocab: &'scope Vocab,
    dwa_pm_mode: DwaPossibleMatchesMode,
    use_terminal_coloring: bool,
    compile_started_at: Instant,
) {
    let ready = {
        let mut state = terminal_state.lock().expect("terminal DAG join state poisoned");
        if state.classify_launched || state.tokenizer.is_none() || state.analysis.is_none() {
            None
        } else {
            state.classify_launched = true;
            Some((
                state.tokenizer.as_ref().expect("tokenizer DAG result ready").clone(),
                state.analysis.as_ref().expect("analysis DAG result ready").clone(),
            ))
        }
    };

    let Some((tokenizer, analysis)) = ready else {
        return;
    };

    scope.spawn(move |scope| {
        let token_path_disallowed_follows = ignore_transparent_disallowed_follows(
            &analysis.disallowed_follows,
            prepared_grammar.ignore_terminal,
        );
        let shared_classify_cache = SharedClassifyCache::new();
        let classify_started_ms = elapsed_ms(compile_started_at.clone());
        let classify_started_at = Instant::now();
        prewarm_shared_classify_cache(
            &tokenizer.tokenizer,
            analysis.analyzed_grammar.num_terminals,
            &shared_classify_cache,
        );
        let classify_ms = elapsed_ms(classify_started_at);
        let classify_finished_ms = elapsed_ms(compile_started_at.clone());

        terminal_state
            .lock()
            .expect("terminal DAG join state poisoned")
            .classify = Some(ClassifyDagLane {
            shared_classify_cache,
            classify_ms,
            started_ms: classify_started_ms,
            finished_ms: classify_finished_ms,
        });
        launch_terminal_dag_if_ready(
            scope,
            terminal_state,
            parser_state,
            result,
            prepared_grammar,
            vocab,
            dwa_pm_mode,
            use_terminal_coloring,
            compile_started_at,
        );
    });
}

fn compile_prepared_with_profile(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
) -> (Constraint, CompilePhaseProfile) {
    compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        GlrTableConstruction::ExperimentalCoreMerged,
        None,
    )
}

fn compile_prepared_with_profile_and_table_construction(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
    lexer_adaptive_override: Option<bool>,
) -> (Constraint, CompilePhaseProfile) {
    let synthetic_plan_started_at = Instant::now();
    let synthetic_tokenizer_plan = plan_synthetic_tokenizer(&prepared_grammar, vocab);
    if std::env::var_os("GLRMASK_PROFILE_SYNTHETIC_PLAN").is_some() {
        eprintln!(
            "[glrmask/profile][synthetic_plan] selected={} terminals={} ms={:.3}",
            synthetic_tokenizer_plan.is_some(),
            prepared_grammar.terminals.len(),
            elapsed_ms(synthetic_plan_started_at),
        );
    }
    let partition_local_synthesis_plan = synthetic_tokenizer_plan.as_ref().map(|plan| {
        Arc::new(
            crate::compiler::stages::id_map_and_terminal_dwa::PartitionLocalSynthesisPlan {
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
                adaptive: lexer_adaptive_override
                    .unwrap_or_else(|| env_flag_enabled_by_default("GLRMASK_LEXER_ADAPTIVE")),
                global_max_token_len: vocab
                    .entries
                    .values()
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0),
            },
        )
    });
    let interner_cleanup = crate::ds::weight::defer_weight_interner_cleanup();
    let result = run_with_compile_thread_pool(|| {
        let compile_started_at = Instant::now();
        let mut profile = CompilePhaseProfile::default();
        let derive_single_use_terminal_possible_matches =
            crate::compiler::stages::id_map_and_terminal_dwa::grammar_def_uses_global_single_terminal_l1(
                &prepared_grammar,
            );

        let analysis_started_at = Instant::now();
        let dwa_pm_mode = dwa_possible_matches_mode();
        let use_terminal_coloring = terminal_coloring_enabled();
        let terminal_state = Mutex::new(TerminalDagJoinState::default());
        let parser_state = Mutex::new(ParserDagJoinState::default());
        let compile_dag_result = Mutex::new(None);
        let runtime_tokenizer_result = Mutex::new(None);
        let cpm_result = Mutex::new(None);

        rayon::scope(|scope| {
            let terminal_state_ref = &terminal_state;
            let parser_state_ref = &parser_state;
            let compile_dag_result_ref = &compile_dag_result;
            let runtime_tokenizer_result_ref = &runtime_tokenizer_result;
            let cpm_result_ref = &cpm_result;
            let prepared_grammar_ref = &prepared_grammar;
            let synthetic_tokenizer_plan_ref = synthetic_tokenizer_plan.as_ref();
            let partition_local_synthesis_plan_ref = partition_local_synthesis_plan.as_ref();
            let analysis_started_for_tokenizer = analysis_started_at.clone();
            let compile_started_for_tokenizer = compile_started_at.clone();
            let lexer_adaptive_override = lexer_adaptive_override;

            scope.spawn(move |scope| {
                let tok_started = Instant::now();
                let build_global_tokenizer = || {
                    if let Some(plan) = synthetic_tokenizer_plan_ref {
                                        if let Some((synthesized, full, certified)) =
                                            prepare_structural_tokenizer_pair(
                                                prepared_grammar_ref,
                                                plan,
                                                vocab,
                                                lexer_adaptive_override,
                                            )
                                        {
                                            let profitable = structural_state_reduction_is_profitable(
                                                full.num_states(),
                                                synthesized.num_states() as usize,
                                            );
                                            if profitable {
                                                (
                                                    synthesized,
                                                    Some(full),
                                                    Some(certified),
                                                    0.0,
                                                )
                                            } else {
                                                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                                                    eprintln!(
                                                        "[glrmask/profile][tokenizer] synthetic_certification_skipped reason=insufficient_state_reduction path=structural_product full_states={} synthesized_states={} absolute_saving={}",
                                                        full.num_states(),
                                                        synthesized.num_states(),
                                                        full.num_states()
                                                            .saturating_sub(synthesized.num_states() as usize),
                                                    );
                                                }
                                                drop(synthesized);
                                                drop(full);
                                                (
                                                    build_ordinary_compile_tokenizer(
                                                        prepared_grammar_ref,
                                                        lexer_adaptive_override,
                                                    ),
                                                    None,
                                                    None,
                                                    0.0,
                                                )
                                            }
                                        } else {
                                            let (mut synthesized, mut full) = rayon::join(
                                                || {
                                                    build_tokenizer_from_planned_expressions(
                                                        prepared_grammar_ref,
                                                        plan,
                                                        &plan.synthesized_expressions,
                                                        lexer_adaptive_override,
                                                    )
                                                },
                                                || {
                                                    build_tokenizer_from_planned_expressions(
                                                        prepared_grammar_ref,
                                                        plan,
                                                        &plan.full_expressions,
                                                        lexer_adaptive_override,
                                                    )
                                                },
                                            );
                                            synthesized.isolate_start_state_and_drain_nullable_terminals();
                                            full.isolate_start_state_and_drain_nullable_terminals();
                                            let profitable = synthetic_state_reduction_is_profitable(
                                                full.num_states() as usize,
                                                synthesized.num_states() as usize,
                                            );
                                            let certification_started_at = Instant::now();
                                            let certification = profitable
                                                .then(|| {
                                                    certify_full_to_synthesized_state_map(
                                                        &full,
                                                        &synthesized,
                                                        vocab,
                                                        None,
                                                    )
                                                })
                                                .flatten();
                                            let certification_ms = elapsed_ms(certification_started_at);
                                            if !profitable
                                                && std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some()
                                            {
                                                eprintln!(
                                                    "[glrmask/profile][tokenizer] synthetic_certification_skipped reason=insufficient_state_reduction full_states={} synthesized_states={} absolute_saving={}",
                                                    full.num_states(),
                                                    synthesized.num_states(),
                                                    full.num_states().saturating_sub(synthesized.num_states()),
                                                );
                                            }
                                            match certification {
                                                Some(certified) => {
                                                    (
                                                        synthesized,
                                                        Some(DeferredRuntimeTokenizer::Ready(full)),
                                                        Some(certified),
                                                        certification_ms,
                                                    )
                                                }
                                                None => {
                                                    drop(synthesized);
                                                    drop(full);
                                                    (
                                                        build_ordinary_compile_tokenizer(
                                                            prepared_grammar_ref,
                                                            lexer_adaptive_override,
                                                        ),
                                                        None,
                                                        None,
                                                        certification_ms,
                                                    )
                                                }
                                            }
                                        }
                                    } else {
                                        let tokenizer = build_ordinary_compile_tokenizer(
                                            prepared_grammar_ref,
                                            lexer_adaptive_override,
                                        );
                                        (tokenizer, None, None, 0.0)
                                    }
                };
                let prebuild_partition_locals = partition_local_synthesis_plan_ref.is_some()
                    && crate::compiler::stages::id_map_and_terminal_dwa::prebuild_partition_local_synthesis_enabled();
                let (global_tokenizer_result, prepared_partition_local_tokenizers) =
                    if prebuild_partition_locals {
                        rayon::join(
                            build_global_tokenizer,
                            || {
                                partition_local_synthesis_plan_ref.and_then(|plan| {
                                    crate::compiler::stages::id_map_and_terminal_dwa::prepare_partition_local_tokenizers(
                                        vocab,
                                        plan,
                                    )
                                })
                            },
                        )
                    } else {
                        (build_global_tokenizer(), None)
                    };
                let (
                    mut tokenizer,
                    deferred_runtime_tokenizer,
                    full_to_synthesized_state_map,
                    synthetic_certification_ms,
                ) = global_tokenizer_result;
                let tokenizer_construct_ms = elapsed_ms(tok_started);
                let isolate_started = Instant::now();
                if deferred_runtime_tokenizer.is_none() {
                    tokenizer.isolate_start_state_and_drain_nullable_terminals();
                }
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] construction_vs_isolation construct_ms={:.3} isolate_ms={:.3} total_ms={:.3}",
                        tokenizer_construct_ms,
                        elapsed_ms(isolate_started),
                        elapsed_ms(tok_started),
                    );
                }
                let compile_tokenizer_states = tokenizer.num_states() as usize;
                let compile_tokenizer_transitions = tokenizer.transition_count();
                let partition_local_synthesis_plan = deferred_runtime_tokenizer
                    .is_some()
                    .then(|| partition_local_synthesis_plan_ref.cloned())
                    .flatten();
                let prepared_partition_local_tokenizers = deferred_runtime_tokenizer
                    .is_some()
                    .then_some(prepared_partition_local_tokenizers)
                    .flatten();

                if let Some(deferred_runtime_tokenizer) = deferred_runtime_tokenizer {
                    // Finishing the exact runtime tokenizer is independent of
                    // terminal-DWA construction. Delaying large tokenizers was
                    // intended to reduce memory-bandwidth contention, but the
                    // slow-build cohort consistently benefits from immediate
                    // overlap, including the protected-residual cases this path
                    // targets. Keep the environment override for diagnostics.
                    let default_start_delay_ms =
                        deferred_runtime_tokenizer.default_start_delay_ms();
                    scope.spawn(move |_| {
                        let start_delay_ms = std::env::var(
                            "GLRMASK_DEFERRED_RUNTIME_START_DELAY_MS",
                        )
                        .ok()
                        .and_then(|value| value.parse::<u64>().ok())
                        .unwrap_or(default_start_delay_ms);
                        if start_delay_ms != 0 {
                            std::thread::sleep(std::time::Duration::from_millis(start_delay_ms));
                        }
                        let runtime_started_at = Instant::now();
                        let runtime_tokenizer = deferred_runtime_tokenizer.finish();
                        let finish_ms = elapsed_ms(runtime_started_at);
                        if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                            eprintln!(
                                "[glrmask/profile][tokenizer] deferred_runtime_finish states={} transitions={} start_delay_ms={} elapsed_ms={:.3}",
                                runtime_tokenizer.num_states(),
                                runtime_tokenizer.transition_count(),
                                start_delay_ms,
                                finish_ms,
                            );
                        }
                        *runtime_tokenizer_result_ref
                            .lock()
                            .expect("runtime tokenizer result slot poisoned") =
                            Some(RuntimeTokenizerDagResult {
                                runtime_tokenizer: Some(runtime_tokenizer),
                                full_to_synthesized_state_map,
                                finish_ms,
                            });
                    });
                } else {
                    *runtime_tokenizer_result_ref
                        .lock()
                        .expect("runtime tokenizer result slot poisoned") =
                        Some(RuntimeTokenizerDagResult {
                            runtime_tokenizer: None,
                            full_to_synthesized_state_map: None,
                            finish_ms: 0.0,
                        });
                }

                let tokenizer_lane = TokenizerDagLane {
                    tokenizer: Arc::new(tokenizer),
                    partition_local_synthesis_plan,
                    prepared_partition_local_tokenizers,
                    synthetic_candidate_terminals: synthetic_tokenizer_plan_ref
                        .map_or(0, |plan| plan.changed_terminal_count),
                    synthetic_certification_ms,
                    compile_tokenizer_states,
                    compile_tokenizer_transitions,
                    tokenizer_build_ms: elapsed_ms(tok_started),
                    tokenizer_ready_ms: elapsed_ms(analysis_started_for_tokenizer),
                };

                let eager_possible_matches =
                    env_flag_enabled_by_default("GLRMASK_EAGER_POSSIBLE_MATCHES")
                        && !derive_single_use_terminal_possible_matches;
                if !eager_possible_matches {
                    let possible_matches_tokenizer = Arc::clone(&tokenizer_lane.tokenizer);
                    let compile_started_for_cpm = compile_started_for_tokenizer.clone();
                    scope.spawn(move |_| {
                        let possible_matches_started_ms = elapsed_ms(compile_started_for_cpm.clone());
                        let result = cpm::compute_constraint_possible_matches_for_vocab(
                            &possible_matches_tokenizer,
                            vocab,
                            cpm::ConstraintPossibleMatchesConfig::DEFER_TO_DYNAMIC_MASK,
                        );
                        let possible_matches_finished_ms = elapsed_ms(compile_started_for_cpm);
                        *cpm_result_ref
                            .lock()
                            .expect("possible-matches result slot poisoned") = Some((
                            result,
                            possible_matches_started_ms,
                            possible_matches_finished_ms,
                        ));
                    });
                }

                let flat_global_tokenizer = Arc::clone(&tokenizer_lane.tokenizer);
                let compile_started_for_terminal = compile_started_for_tokenizer.clone();
                scope.spawn(move |scope| {
                    let flat_global_started_ms = elapsed_ms(compile_started_for_terminal.clone());
                    let flat_trans_started_at = Instant::now();
                    let flat_trans: Arc<[u32]> = Arc::from(
                        crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(
                            &flat_global_tokenizer,
                        ),
                    );
                    let flat_trans_ms = elapsed_ms(flat_trans_started_at);
                    let shared_transition_cache = Arc::new(std::sync::OnceLock::new());

                    if eager_possible_matches {
                        let possible_matches_tokenizer = Arc::clone(&flat_global_tokenizer);
                        let possible_matches_flat_trans = Arc::clone(&flat_trans);
                        let possible_matches_transition_cache = Arc::clone(&shared_transition_cache);
                        let compile_started_for_cpm = compile_started_for_terminal.clone();
                        scope.spawn(move |_| {
                            let possible_matches_started_ms =
                                elapsed_ms(compile_started_for_cpm.clone());
                            let raw_byte_to_class = possible_matches_transition_cache
                                .get_or_init(|| {
                                    crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::derive_flat_transition_cache(
                                        &possible_matches_tokenizer,
                                        possible_matches_flat_trans,
                                    )
                                })
                                .byte_to_class;
                            let result =
                                cpm::compute_constraint_possible_matches_for_vocab_with_raw_byte_classes(
                                    &possible_matches_tokenizer,
                                    vocab,
                                    cpm::ConstraintPossibleMatchesConfig::EAGER,
                                    &raw_byte_to_class,
                                );
                            let possible_matches_finished_ms = elapsed_ms(compile_started_for_cpm);
                            *cpm_result_ref
                                .lock()
                                .expect("possible-matches result slot poisoned") = Some((
                                result,
                                possible_matches_started_ms,
                                possible_matches_finished_ms,
                            ));
                        });
                    }

                    let global_max_length_started_at = Instant::now();
                    let global_max_length_state_map =
                        crate::compiler::stages::id_map_and_terminal_dwa::build_global_max_length_state_map(
                            &flat_global_tokenizer,
                            vocab,
                            &flat_trans,
                        );
                    let global_max_length_ms = elapsed_ms(global_max_length_started_at);
                    let flat_global_finished_ms = elapsed_ms(compile_started_for_terminal.clone());
                    terminal_state_ref
                        .lock()
                        .expect("terminal DAG join state poisoned")
                        .flat_global = Some(FlatGlobalDagLane {
                        flat_trans,
                        shared_transition_cache,
                        flat_trans_ms,
                        global_max_length_state_map,
                        global_max_length_ms,
                        started_ms: flat_global_started_ms,
                        finished_ms: flat_global_finished_ms,
                    });
                    launch_terminal_dag_if_ready(
                        scope,
                        terminal_state_ref,
                        parser_state_ref,
                        compile_dag_result_ref,
                        prepared_grammar_ref,
                        vocab,
                        dwa_pm_mode,
                        use_terminal_coloring,
                        compile_started_for_terminal,
                    );
                });

                terminal_state_ref
                    .lock()
                    .expect("terminal DAG join state poisoned")
                    .tokenizer = Some(tokenizer_lane);
                launch_classify_dag_if_ready(
                    scope,
                    terminal_state_ref,
                    parser_state_ref,
                    compile_dag_result_ref,
                    prepared_grammar_ref,
                    vocab,
                    dwa_pm_mode,
                    use_terminal_coloring,
                    compile_started_for_tokenizer.clone(),
                );
                launch_terminal_dag_if_ready(
                    scope,
                    terminal_state_ref,
                    parser_state_ref,
                    compile_dag_result_ref,
                    prepared_grammar_ref,
                    vocab,
                    dwa_pm_mode,
                    use_terminal_coloring,
                    compile_started_for_tokenizer,
                );
            });

            let terminal_state_ref = &terminal_state;
            let parser_state_ref = &parser_state;
            let compile_dag_result_ref = &compile_dag_result;
            let prepared_grammar_ref = &prepared_grammar;
            let synthetic_tokenizer_plan_for_analysis = synthetic_tokenizer_plan.as_ref();
            let analysis_started_for_analysis = analysis_started_at.clone();
            let compile_started_for_analysis = compile_started_at.clone();
            let table_construction = default_table_construction.clone();

            scope.spawn(move |scope| {
                let analyze_grammar_started_at = Instant::now();
                let analyzed_grammar = Arc::new(AnalyzedGrammar::from_grammar_def(prepared_grammar_ref));
                let analyze_grammar_ms = elapsed_ms(analyze_grammar_started_at);
                if let Err(message) = analyzed_grammar.check_table_build_normal_form() {
                    panic!("[glrmask] grammar precondition violations:\n{}", message);
                }

                let glr_analyzed_grammar = Arc::clone(&analyzed_grammar);
                let analysis_started_for_glr = analysis_started_for_analysis.clone();
                let compile_started_for_glr = compile_started_for_analysis.clone();
                scope.spawn(move |scope| {
                    let table_started_at = Instant::now();
                    let table = Arc::new(GLRTable::build_with_default_construction(
                        &glr_analyzed_grammar,
                        table_construction,
                    ));
                    let glr_table_ms = elapsed_ms(table_started_at);
                    if std::env::var_os("GLRMASK_STOP_AFTER_GLR_TABLE").is_some() {
                        panic!("[glrmask] stopped after GLR table build by GLRMASK_STOP_AFTER_GLR_TABLE");
                    }
                    let glr_ready_ms = elapsed_ms(analysis_started_for_glr);

                    if use_terminal_coloring {
                        let coloring_table = Arc::clone(&table);
                        let mut protected_terminal_ids = glr_analyzed_grammar
                            .residual_isolation_classes
                            .keys()
                            .copied()
                            .collect::<Vec<_>>();
                        if let Some(plan) = synthetic_tokenizer_plan_for_analysis {
                            protected_terminal_ids
                                .extend(plan.changed_terminal_ids.iter().copied());
                            protected_terminal_ids.sort_unstable();
                            protected_terminal_ids.dedup();
                        }
                        let compile_started_for_coloring = compile_started_for_glr.clone();
                        scope.spawn(move |scope| {
                            let terminal_coloring_started_at = Instant::now();
                            let mut terminal_coloring = compute_terminal_coloring(&coloring_table);
                            terminal_coloring
                                .isolate_terminals(protected_terminal_ids.iter().copied());
                            let terminal_coloring_ms = elapsed_ms(terminal_coloring_started_at);
                            terminal_state_ref
                                .lock()
                                .expect("terminal DAG join state poisoned")
                                .coloring = Some(ColoringDagLane {
                                terminal_coloring,
                                terminal_coloring_ms,
                            });
                            launch_terminal_dag_if_ready(
                                scope,
                                terminal_state_ref,
                                parser_state_ref,
                                compile_dag_result_ref,
                                prepared_grammar_ref,
                                vocab,
                                dwa_pm_mode,
                                use_terminal_coloring,
                                compile_started_for_coloring,
                            );
                        });
                    }

                    let templates_table = Arc::clone(&table);
                    let templates_analyzed_grammar = Arc::clone(&glr_analyzed_grammar);
                    let compile_started_for_templates = compile_started_for_glr;
                    scope.spawn(move |scope| {
                        let templates_started_ms = elapsed_ms(compile_started_for_templates.clone());
                        let (templates, template_dfas_by_terminal, templates_ms) =
                            build_templates_for_compile(
                                &templates_table,
                                &templates_analyzed_grammar,
                                prepared_grammar_ref.ignore_terminal,
                            );
                        let templates_finished_ms = elapsed_ms(compile_started_for_templates.clone());
                        parser_state_ref
                            .lock()
                            .expect("parser DAG join state poisoned")
                            .templates = Some(TemplatesDagResult {
                            table: templates_table,
                            glr_table_ms,
                            glr_ready_ms,
                            templates,
                            template_dfas_by_terminal,
                            templates_ms,
                            templates_started_ms,
                            templates_finished_ms,
                        });
                        launch_parser_dag_if_ready(
                            scope,
                            parser_state_ref,
                            compile_dag_result_ref,
                            vocab,
                            dwa_pm_mode,
                            compile_started_for_templates,
                        );
                    });
                });

                let disallowed_follows_started_at = Instant::now();
                let (ever_allowed_follows, always_allowed_follows) =
                    compute_allowed_follow_sets(&analyzed_grammar);
                let disallowed_follows = Arc::new(compute_disallowed_follows_from_ever(
                    analyzed_grammar.num_terminals,
                    &ever_allowed_follows,
                ));
                let analysis_lane = AnalysisDagLane {
                    analyzed_grammar,
                    analyze_grammar_ms,
                    disallowed_follows,
                    always_allowed_follows: always_allowed_follows.into(),
                    disallowed_follows_ms: elapsed_ms(disallowed_follows_started_at),
                    analysis_ready_ms: elapsed_ms(analysis_started_for_analysis),
                };
                terminal_state_ref
                    .lock()
                    .expect("terminal DAG join state poisoned")
                    .analysis = Some(analysis_lane);
                launch_classify_dag_if_ready(
                    scope,
                    terminal_state_ref,
                    parser_state_ref,
                    compile_dag_result_ref,
                    prepared_grammar_ref,
                    vocab,
                    dwa_pm_mode,
                    use_terminal_coloring,
                    compile_started_for_analysis.clone(),
                );
                launch_terminal_dag_if_ready(
                    scope,
                    terminal_state_ref,
                    parser_state_ref,
                    compile_dag_result_ref,
                    prepared_grammar_ref,
                    vocab,
                    dwa_pm_mode,
                    use_terminal_coloring,
                    compile_started_for_analysis,
                );
            });
        });

        let (
            mut cpm_result,
            mut possible_matches_started_ms,
            mut possible_matches_finished_ms,
        ) = cpm_result
            .into_inner()
            .expect("possible-matches result slot poisoned")
            .expect("possible-matches task did not complete");
        let RuntimeTokenizerDagResult {
            mut runtime_tokenizer,
            full_to_synthesized_state_map,
            finish_ms: runtime_tokenizer_finish_ms,
        } = runtime_tokenizer_result
            .into_inner()
            .expect("runtime tokenizer result slot poisoned")
            .expect("runtime tokenizer task did not complete");
        let CompileDagResult {
            tokenizer,
            synthetic_candidate_terminals,
            synthetic_certification_ms,
            compile_tokenizer_states,
            compile_tokenizer_transitions,
            tokenizer_build_ms,
            tokenizer_ready_ms,
            analyzed_grammar,
            analyze_grammar_ms,
            disallowed_follows_ms,
            analysis_ready_ms,
            table,
            glr_table_ms,
            glr_ready_ms,
            terminal_coloring_ms,
            mut terminal_dwas,
            mut terminal_phase_profile,
            mut templates,
            template_dfas_by_terminal,
            templates_ms,
            classify_ms,
            flat_trans_ms,
            global_max_length_ms,
            flat_global_started_ms,
            flat_global_finished_ms,
            classify_started_ms,
            classify_finished_ms,
            terminal_dwa_started_ms,
            terminal_dwa_finished_ms,
            templates_started_ms,
            templates_finished_ms,
            terminal_run_collapse_ms,
            prebuilt_parser_dwa,
        } = compile_dag_result
            .into_inner()
            .expect("compile DAG result slot poisoned")
            .expect("compile DAG did not produce a result");
        if derive_single_use_terminal_possible_matches {
            match cpm::complete_single_use_terminal_possible_matches_from_l1(
                &terminal_dwas,
                cpm_result,
            ) {
                Some(derived) => cpm_result = derived,
                None => {
                    // The pre-analysis gate is deliberately cheap and can be
                    // invalidated by later terminal-family construction (for
                    // example, a projected terminal can disappear). Fail
                    // closed by running the ordinary eager computation rather
                    // than leaving an incomplete runtime fallback table.
                    possible_matches_started_ms = elapsed_ms(compile_started_at.clone());
                    cpm_result = cpm::compute_constraint_possible_matches_for_vocab(
                        tokenizer.as_ref(),
                        vocab,
                        cpm::ConstraintPossibleMatchesConfig::EAGER,
                    );
                    possible_matches_finished_ms = elapsed_ms(compile_started_at.clone());
                }
            }
        }
        let mut tokenizer = Arc::try_unwrap(tokenizer)
            .unwrap_or_else(|_| panic!("tokenizer references outlived compile DAG"));
        let analyzed_grammar = Arc::try_unwrap(analyzed_grammar)
            .unwrap_or_else(|_| panic!("analyzed grammar references outlived compile DAG"));
        let table = Arc::try_unwrap(table)
            .unwrap_or_else(|_| panic!("GLR table references outlived compile DAG"));

        profile.tokenizer_build_ms = tokenizer_build_ms;
        let final_tokenizer = runtime_tokenizer.as_ref().unwrap_or(&tokenizer);
        profile.tokenizer_final_states = final_tokenizer.num_states() as usize;
        profile.tokenizer_final_transitions = final_tokenizer.transition_count();
        profile.synthetic_candidate_terminals = synthetic_candidate_terminals;
        profile.synthetic_certified = runtime_tokenizer.is_some();
        profile.synthetic_compile_states = compile_tokenizer_states;
        profile.synthetic_compile_transitions = compile_tokenizer_transitions;
        profile.synthetic_certification_ms = synthetic_certification_ms;
        profile.analyze_grammar_ms = analyze_grammar_ms;
        profile.glr_table_ms = glr_table_ms;
        profile.terminal_coloring_ms = terminal_coloring_ms;
        profile.disallowed_follows_ms = disallowed_follows_ms;
        profile.analysis_wall_ms = tokenizer_ready_ms.max(analysis_ready_ms).max(glr_ready_ms);
        profile.classify_ms = classify_ms;
        terminal_phase_profile.terminal_dwa_ms += flat_trans_ms;
        terminal_phase_profile.id_map_ms += global_max_length_ms;
        profile.templates_ms = templates_ms;
        profile.id_map_ms = terminal_phase_profile.id_map_ms;
        profile.terminal_dwa_ms = terminal_phase_profile.terminal_dwa_ms;
        profile.compact_ms = terminal_phase_profile.compact_ms;
        profile.split_terminal_dwa_total_ms = terminal_phase_profile.split_terminal_dwa_total_ms;
        profile.global_merge_ms = terminal_phase_profile.global_merge_ms;

        let runtime_dynamic_vocab = cpm_result.runtime_dynamic_vocab;
        let possible_matches_complete = cpm_result.complete;
        let mut possible_matches = cpm_result.mapped_possible_matches;
        let cpm_profile = cpm_result.profile;
        let parser_dag_timing = prebuilt_parser_dwa
            .as_ref()
            .map(|(_, _, started_ms, finished_ms)| (*started_ms, *finished_ms));

        let mut shared_id_reconcile_ms = 0.0;
        if compact_possible_matches_before_reconcile_enabled() {
            let compact_started_at = Instant::now();
            if compile_profile_enabled() {
                let _ = possible_matches.compact_dimensions_fast_with_stats();
            } else {
                let _ = possible_matches.compact_dimensions_fast();
            }
            profile.compact_ms += elapsed_ms(compact_started_at);
        }
        let collect_expensive_profile_stats = compile_profile_summary_enabled();
        let (
            terminal_dwa_interned_ranges_before_pm_reconcile,
            possible_matches_interned_ranges_before_pm_reconcile,
            terminal_pm_joint_interned_ranges_before_reconcile,
        ) = if collect_expensive_profile_stats {
            (
                terminal_family_interned_range_count(&terminal_dwas),
                interned_range_count_for_artifact(possible_matches.artifact_mut()),
                terminal_family_joint_interned_range_count(
                    &terminal_dwas,
                    possible_matches.artifact(),
                ),
            )
        } else {
            (0, 0, 0)
        };

        let (mut parser_dwa, parser_dwa_ms) = if let Some((
            parser_dwa,
            parser_dwa_ms,
            _,
            _,
        )) = prebuilt_parser_dwa
        {
            (parser_dwa, parser_dwa_ms)
        } else {
            let parser_dwa_started_at = Instant::now();
            let retained_templates = templates
                .take()
                .expect("terminal reconciliation mode retains templates");
            let (family_vec, family_layout) = reconcile_terminal_dwa_families(terminal_dwas);
            let shared_id_reconcile_started_at = Instant::now();
            let mut terminal_pm_pair = MappedArtifact::from((family_vec, possible_matches));
            shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);

            let parser_dwa = if dwa_pm_mode.does_terminal_compact() {
                let compact_plan_started_at = Instant::now();
                let terminal_compaction_plan =
                    terminal_pm_pair.plan_dimensions_compaction(true, true);
                profile.compact_ms += elapsed_ms(compact_plan_started_at);

                if dwa_pm_mode.does_parser_compact() {
                    let compact_apply_started_at = Instant::now();
                    terminal_pm_pair.apply_compaction_plan(&terminal_compaction_plan);
                    profile.compact_ms += elapsed_ms(compact_apply_started_at);
                    let ((family_artifacts, possible_matches_artifact), compacted_ids) =
                        terminal_pm_pair.into_parts();
                    terminal_dwas = restore_terminal_dwa_families(
                        MappedArtifact::new(family_artifacts, compacted_ids.clone()),
                        family_layout,
                    );
                    possible_matches =
                        MappedArtifact::new(possible_matches_artifact, compacted_ids);
                    build_and_merge_parser_dwa_families(
                        &terminal_dwas,
                        &table,
                        &analyzed_grammar,
                        prepared_grammar.ignore_terminal,
                        retained_templates,
                        &tokenizer,
                        vocab,
                    )
                } else {
                    let precompact_families = restore_terminal_dwa_families(
                        MappedArtifact::new(
                            terminal_pm_pair.artifact().0.clone(),
                            terminal_pm_pair.id_map().clone(),
                        ),
                        family_layout,
                    );
                    let mut parser_dwa = build_and_merge_parser_dwa_families(
                        &precompact_families,
                        &table,
                        &analyzed_grammar,
                        prepared_grammar.ignore_terminal,
                        retained_templates,
                        &tokenizer,
                        vocab,
                    );
                    let compact_apply_started_at = Instant::now();
                    terminal_pm_pair.apply_compaction_plan(&terminal_compaction_plan);
                    parser_dwa.apply_compaction_plan(&terminal_compaction_plan);
                    profile.compact_ms += elapsed_ms(compact_apply_started_at);
                    let ((family_artifacts, possible_matches_artifact), compacted_ids) =
                        terminal_pm_pair.into_parts();
                    terminal_dwas = restore_terminal_dwa_families(
                        MappedArtifact::new(family_artifacts, compacted_ids.clone()),
                        family_layout,
                    );
                    possible_matches =
                        MappedArtifact::new(possible_matches_artifact, compacted_ids);
                    parser_dwa
                }
            } else {
                let ((family_artifacts, possible_matches_artifact), reconciled_ids) =
                    terminal_pm_pair.into_parts();
                terminal_dwas = restore_terminal_dwa_families(
                    MappedArtifact::new(family_artifacts, reconciled_ids.clone()),
                    family_layout,
                );
                possible_matches =
                    MappedArtifact::new(possible_matches_artifact, reconciled_ids);
                build_and_merge_parser_dwa_families(
                    &terminal_dwas,
                    &table,
                    &analyzed_grammar,
                    prepared_grammar.ignore_terminal,
                    retained_templates,
                    &tokenizer,
                    vocab,
                )
            };
            (parser_dwa, elapsed_ms(parser_dwa_started_at))
        };
        if compile_profile_enabled() || compile_top_profile_enabled() {
            if let Some((parser_dwa_started_ms, parser_dwa_finished_ms)) = parser_dag_timing {
                let overlap_ms = possible_matches_finished_ms.min(parser_dwa_finished_ms)
                    - possible_matches_started_ms.max(parser_dwa_started_ms);
                eprintln!(
                    "[glrmask/profile][compile_dag] tokenizer_ready_ms={:.3} analysis_ready_ms={:.3} glr_ready_ms={:.3} flat_global_started_ms={:.3} flat_global_finished_ms={:.3} classify_started_ms={:.3} classify_finished_ms={:.3} templates_started_ms={:.3} templates_finished_ms={:.3} terminal_dwa_started_ms={:.3} terminal_dwa_finished_ms={:.3} possible_matches_started_ms={:.3} possible_matches_finished_ms={:.3} parser_dwa_started_ms={:.3} parser_dwa_finished_ms={:.3} possible_matches_parser_overlap_ms={:.3} parser_waited_for_possible_matches=false terminal_coloring_enabled={}",
                    tokenizer_ready_ms,
                    analysis_ready_ms,
                    glr_ready_ms,
                    flat_global_started_ms,
                    flat_global_finished_ms,
                    classify_started_ms,
                    classify_finished_ms,
                    templates_started_ms,
                    templates_finished_ms,
                    terminal_dwa_started_ms,
                    terminal_dwa_finished_ms,
                    possible_matches_started_ms,
                    possible_matches_finished_ms,
                    parser_dwa_started_ms,
                    parser_dwa_finished_ms,
                    overlap_ms.max(0.0),
                    use_terminal_coloring,
                );
            }
        }

        let terminal_pm_joint_interned_ranges = if collect_expensive_profile_stats {
            terminal_family_joint_interned_range_count(
                &terminal_dwas,
                possible_matches.artifact(),
            )
        } else {
            0
        };

        // Parser-family union may choose a different but equivalent internal ID
        // numbering from the reconciled terminal families.  Always make the
        // parser/possible-match relationship explicit instead of relying on
        // coincidentally identical numbering.
        let shared_id_reconcile_started_at = Instant::now();
        let mut parser_pm_pair = MappedArtifact::from((parser_dwa, possible_matches));
        shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
        if dwa_pm_mode.does_parser_compact() {
            let compact_started_at = Instant::now();
            parser_pm_pair.compact_dimensions();
            profile.compact_ms += elapsed_ms(compact_started_at);
        }
        let ((parser_dwa_artifact, possible_matches_artifact), internal_ids) =
            parser_pm_pair.into_parts();
        let runtime_state_map_lift_started_at = Instant::now();
        let mut runtime_tokenizer_state_map = match full_to_synthesized_state_map.as_ref() {
            Some(certified) => certified
                .lift_internal_tsid_map(&internal_ids.tokenizer_states)
                .expect("certified full lexer state map must lift the final synthesized TSID map"),
            None => internal_ids.tokenizer_states.clone(),
        };
        let mut runtime_internal_tsid_to_states =
            runtime_tokenizer_state_map.internal_to_originals_vecs();
        let mut runtime_source_state_offset = None;
        let mut runtime_product_source_offsets = Vec::<u32>::new();
        let mut runtime_product_source_states = Vec::<u32>::new();
        let mut runtime_product_exact_source_states = Vec::<u32>::new();
        let runtime_full_adaptive = std::env::var("GLRMASK_RUNTIME_FULL_ADAPTIVE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
            })
            // Exact product states reduce visible lexer-frontier width, but
            // commit must restore per-source longest-match provenance. Current
            // corpus measurements show that restoration costs more than the
            // smaller frontier saves, so retain full runtime determinization as
            // an explicit experiment rather than a production default.
            .unwrap_or(false);
        if runtime_full_adaptive {
            let source_tokenizer = runtime_tokenizer.as_ref().unwrap_or(&tokenizer);
            let source_states = source_tokenizer.num_states() as usize;
            let source_transitions = source_tokenizer.transition_count();
            let state_limit = std::env::var("GLRMASK_ADAPTIVE_LEXER_MAX_STATES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
                // Runtime full determinization is optional. Useful products in
                // the tail-latency cohort are typically hundreds of states;
                // letting a failed large product grow to 32k states can add
                // hundreds of milliseconds while producing no runtime
                // artifact. Keep a bounded default and retain the existing
                // environment override for deliberate larger experiments.
                .unwrap_or(8_192)
                .min(source_states.max(1));
            let transition_growth_percent = std::env::var(
                "GLRMASK_ADAPTIVE_LEXER_MAX_TRANSITION_GROWTH_PERCENT",
            )
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(600);
            let transition_limit = source_transitions
                .saturating_mul(transition_growth_percent)
                / 100;
            let determinize_started_at = Instant::now();
            let profile_runtime_determinization = compile_profile_enabled()
                || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
            if let Some(candidate) = source_tokenizer
                .try_full_determinization(state_limit, transition_limit.max(1))
            {
                let product_states = candidate.source_subsets.len();
                let runtime_state_capacity = product_states.saturating_add(source_states);
                let mut mapped = Vec::with_capacity(runtime_state_capacity);
                let mut state_tsids =
                    Vec::<Vec<u32>>::with_capacity(runtime_state_capacity);
                let mut exact_closure_states = 0usize;
                let mut singleton_states = 0usize;
                let mut multi_tsid_states = 0usize;
                let mut max_distinct_tsids = 0usize;
                let source_subset_memberships = candidate
                    .source_subsets
                    .iter()
                    .map(|subset| subset.len())
                    .sum::<usize>();
                let max_source_subset = candidate
                    .source_subsets
                    .iter()
                    .map(|subset| subset.len())
                    .max()
                    .unwrap_or(0);
                for (subset, &exact_source) in candidate
                    .source_subsets
                    .iter()
                    .zip(&candidate.exact_source_states)
                {
                    if exact_source != u32::MAX {
                        let source_state = exact_source;
                        let tsid = runtime_tokenizer_state_map.original_to_internal
                            [source_state as usize];
                        mapped.push(tsid);
                        state_tsids.push(vec![tsid]);
                        exact_closure_states += 1;
                        continue;
                    }

                    let mut tsids = subset
                        .iter()
                        .map(|&source_state| {
                            runtime_tokenizer_state_map.original_to_internal
                                [source_state as usize]
                        })
                        .collect::<Vec<_>>();
                    tsids.sort_unstable();
                    tsids.dedup();
                    max_distinct_tsids = max_distinct_tsids.max(tsids.len());
                    if tsids.is_empty() {
                        mapped.clear();
                        state_tsids.clear();
                        break;
                    }
                    mapped.push(tsids[0]);
                    if tsids.len() == 1 {
                        singleton_states += 1;
                    } else {
                        multi_tsid_states += 1;
                    }
                    state_tsids.push(tsids);
                }

                if !mapped.is_empty() {
                    for source_state in 0..source_states {
                        let tsid = runtime_tokenizer_state_map.original_to_internal[source_state];
                        mapped.push(tsid);
                        state_tsids.push(vec![tsid]);
                    }
                }

                if !mapped.is_empty() {
                    let candidate = match runtime_tokenizer.as_mut() {
                        Some(source_tokenizer) => source_tokenizer
                            .finish_full_determinization_with_source_fallback(candidate),
                        None => tokenizer
                            .finish_full_determinization_with_source_fallback(candidate),
                    };
                    let num_internal_tsids = runtime_tokenizer_state_map.num_internal_ids();
                    let mut reverse = vec![Vec::<u32>::new(); num_internal_tsids as usize];
                    for (runtime_state, tsids) in state_tsids.iter().enumerate() {
                        for &tsid in tsids {
                            reverse[tsid as usize].push(runtime_state as u32);
                        }
                    }
                    runtime_tokenizer_state_map =
                        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                            mapped,
                            num_internal_tsids,
                        );
                    runtime_internal_tsid_to_states = reverse;
                    runtime_source_state_offset = Some(candidate.source_state_offset);
                    runtime_product_source_offsets.reserve(product_states + 1);
                    runtime_product_source_offsets.push(0);
                    for subset in &candidate.source_subsets {
                        runtime_product_source_states.extend_from_slice(subset);
                        runtime_product_source_offsets
                            .push(runtime_product_source_states.len() as u32);
                    }
                    runtime_product_exact_source_states = candidate.exact_source_states.clone();
                    if profile_runtime_determinization {
                        eprintln!(
                            "[glrmask/profile][runtime_full_adaptive] selected=true source_states={} product_states={} runtime_states={} source_transitions={} runtime_transitions={} source_subset_memberships={} max_source_subset={} exact_closure_states={} singleton_states={} multi_tsid_states={} max_distinct_tsids={} elapsed_ms={:.3}",
                            source_states,
                            product_states,
                            candidate.tokenizer.num_states(),
                            source_transitions,
                            candidate.tokenizer.transition_count(),
                            source_subset_memberships,
                            max_source_subset,
                            exact_closure_states,
                            singleton_states,
                            multi_tsid_states,
                            max_distinct_tsids,
                            elapsed_ms(determinize_started_at),
                        );
                    }
                    runtime_tokenizer = Some(candidate.tokenizer);
                } else if profile_runtime_determinization {
                    eprintln!(
                        "[glrmask/profile][runtime_full_adaptive] selected=false reason=empty_tsid_subset source_states={} candidate_states={} elapsed_ms={:.3}",
                        source_states,
                        candidate.tokenizer.num_states(),
                        elapsed_ms(determinize_started_at),
                    );
                }
            } else if profile_runtime_determinization {
                eprintln!(
                    "[glrmask/profile][runtime_full_adaptive] selected=false reason=determinization_limit source_states={} state_limit={} transition_limit={} elapsed_ms={:.3}",
                    source_states,
                    state_limit,
                    transition_limit.max(1),
                    elapsed_ms(determinize_started_at),
                );
            }
        }
        let runtime_state_map_lift_ms = elapsed_ms(runtime_state_map_lift_started_at);
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][runtime_tokenizer_join] finish_ms={:.3} state_map_lift_ms={:.3} runtime_states={} synthesized_states={}",
                runtime_tokenizer_finish_ms,
                runtime_state_map_lift_ms,
                runtime_tokenizer
                    .as_ref()
                    .map_or(tokenizer.num_states(), Tokenizer::num_states),
                tokenizer.num_states(),
            );
        }
        parser_dwa = MappedArtifact::new(parser_dwa_artifact, internal_ids.clone());
        possible_matches =
            MappedArtifact::new(possible_matches_artifact, internal_ids.clone());

        let (
            parser_dwa_interned_ranges,
            possible_matches_interned_ranges,
            parser_pm_joint_interned_ranges,
        ) = if collect_expensive_profile_stats {
            let parser_dwa_interned_ranges =
                count_interned_ranges_for_weights(parser_dwa.artifact().weight_refs())
                    .total_ranges();
            let (parser_dwa_artifact, _) = parser_dwa.parts_mut();
            let (possible_matches_artifact, _) = possible_matches.parts_mut();
            (
                parser_dwa_interned_ranges,
                interned_range_count_for_artifact(possible_matches_artifact),
                joint_interned_range_count_for_artifacts(
                    parser_dwa_artifact,
                    possible_matches_artifact,
                ),
            )
        } else {
            (0, 0, 0)
        };
        let (
            parser_dwa,
            ParserTopAccept {
                combined: parser_top_accept,
                parts: parser_top_accept_parts,
                direct_l1_complete_by_terminal,
            },
        ) = parser_dwa.into_artifact();

        let internal_token_bytes_started_at = Instant::now();
        let internal_token_bytes = cpm::build_internal_token_bytes_from_groups(
            vocab,
            &internal_ids.vocab_tokens.internal_to_originals,
        );
        let internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

        profile.terminal_run_collapse_ms = terminal_run_collapse_ms;
        profile.parser_dwa_ms = parser_dwa_ms;
        profile.possible_matches_vocab_equiv_ms = cpm_profile.vocab_equiv_ms;
        profile.possible_matches_collect_ms = cpm_profile.possible_matches_collect_ms;
        profile.possible_matches_materialize_ms = cpm_profile.possible_match_vocab_ms;
        profile.shared_id_reconcile_ms = shared_id_reconcile_ms;
        profile.possible_matches_pipeline_ms =
            cpm_profile.vocab_equiv_ms
                + cpm_profile.possible_matches_collect_ms
                + cpm_profile.possible_match_vocab_ms
                + shared_id_reconcile_ms;
        profile.terminal_dwa_interned_ranges_before_pm_reconcile =
            terminal_dwa_interned_ranges_before_pm_reconcile;
        profile.possible_matches_interned_ranges_before_pm_reconcile =
            possible_matches_interned_ranges_before_pm_reconcile;
        profile.terminal_pm_joint_interned_ranges_before_reconcile =
            terminal_pm_joint_interned_ranges_before_reconcile;
        profile.terminal_pm_joint_interned_ranges = terminal_pm_joint_interned_ranges;
        profile.internal_token_bytes_ms = internal_token_bytes_ms;
        profile.parser_dwa_interned_ranges = parser_dwa_interned_ranges;
        profile.possible_matches_interned_ranges = possible_matches_interned_ranges;
        profile.parser_pm_joint_interned_ranges = parser_pm_joint_interned_ranges;

        let finalize_started_at = Instant::now();
        let token_bytes = std::sync::Arc::clone(&vocab.entries);
        let special_token_terminals = collect_special_token_terminals(&prepared_grammar);
        let tokenizer = runtime_tokenizer.unwrap_or(tokenizer);
        let constraint = finalize_constraint(Constraint {
            runtime_backend: crate::runtime::ConstraintRuntimeBackend::Static,
            parser_dwa,
            parser_top_accept,
            parser_top_accept_parts,
            direct_regular_l1_complete_by_terminal: direct_l1_complete_by_terminal,
            direct_regular_wide_frontier_acceptance: Vec::new(),
            direct_regular_parser_state_acceptance: Vec::new(),
            direct_regular_automaton: analyzed_grammar.direct_regular_automaton.clone(),
            table,
            terminal_display_names: analyzed_grammar.terminal_display_names.clone(),
            tokenizer,
            tokenizer_has_epsilon_transitions: false,
            ignore_terminal: prepared_grammar.ignore_terminal,
            special_token_terminals,
            dynamic_mask_vocab: runtime_dynamic_vocab.vocab,
            lazy_dynamic_mask_vocab: std::sync::OnceLock::new(),
            possible_matches: possible_matches.into_artifact(),
            possible_matches_complete,
            state_to_internal_tsid: runtime_tokenizer_state_map.original_to_internal.clone(),
            internal_tsid_to_states: runtime_internal_tsid_to_states,
            terminal_live_states: Vec::new(),
            state_internal_tsid_offsets: Vec::new(),
            state_internal_tsids: Vec::new(),
            runtime_source_state_offset,
            runtime_product_source_offsets,
            runtime_product_source_states,
            runtime_product_exact_source_states,
            runtime_product_state_by_source_subset: Default::default(),
            original_token_to_internal: internal_ids.vocab_tokens.original_to_internal.clone(),
            internal_token_to_tokens: internal_ids.vocab_tokens.internal_to_originals_vecs(),
            template_dfas_by_terminal,
            fast_template_dfas_by_terminal: Vec::new(),
            token_bytes,
            internal_token_bytes,
            token_bytes_dense: Vec::new(),
            internal_token_buf_masks: Vec::new(),
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks: Vec::new(),
            quad_word_group_buf_masks: Vec::new(),
            super_word_group_buf_masks: Vec::new(),
            mega_word_group_buf_masks: Vec::new(),
            giga_word_group_buf_masks: Vec::new(),
            word_group_sparse_masks: Vec::new(),
            word_group_prefix_buf_masks: Vec::new(),
            word_group_sparse_prefix_entries: Vec::new(),
            quad_group_sparse_masks: Vec::new(),
            quad_group_dense_masks: Vec::new(),
            byte_group_sparse_masks: Vec::new(),
            byte_group_dense_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: rustc_hash::FxHashMap::default(),
            weight_token_buf_masks: rustc_hash::FxHashMap::default(),
            weight_token_sparse_buf_masks: rustc_hash::FxHashMap::default(),
            direct_sparse_weight_token_sets: rustc_hash::FxHashSet::default(),
            seed_terminal_dense: rustc_hash::FxHashMap::default(),
            seed_terminal_dense_fallback: Default::default(),
            seed_universe_dense: std::sync::Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Vec::new(),
            indexed_dag_dense_transitions: Vec::new(),
            indexed_dag_dense_finals: Vec::new(),
            tokenizer_fast_transitions: Default::default(),
            heavy_token_dense_masks: Vec::new(),
            heavy_token_indices: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: crate::runtime::mask_mapping::FinalMaskMapping::default(),
        });
        profile.finalize_ms = elapsed_ms(finalize_started_at);
        profile.compile_ms = elapsed_ms(compile_started_at);

        (constraint, profile)
    });
    // Keep the output constraint alive while the final sweep removes only dead
    // weak entries from the compile-time interners.
    interner_cleanup.finish();
    result
}

pub(crate) fn compile_prepared(prepared_grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    let start_nullable = prepared_grammar.start_is_nullable();
    let mut constraint = compile_prepared_with_profile(prepared_grammar, vocab).0;
    constraint
        .table
        .set_embedded_start_nullable(start_nullable);
    constraint
}

fn prepare_grammar(grammar: GrammarDef) -> GrammarDef {
    prepare_grammar_transforms_only(grammar)
}

pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    compile_owned_with_table_construction(
        grammar,
        vocab,
        GlrTableConstruction::ExperimentalCoreMerged,
    )
}

pub(crate) fn compile_dynamic_owned_with_table_construction(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> DynamicConstraint {
    let start_nullable = grammar.start_is_nullable();
    let profile = compile_profile_enabled();
    let total_started_at = profile.then(Instant::now);
    let prepare_started_at = profile.then(Instant::now);
    let prepared_grammar = prepare_grammar(grammar);
    let prepare_ms = prepare_started_at.map_or(0.0, elapsed_ms);
    run_with_compile_thread_pool(|| {
        let analysis_started_at = profile.then(Instant::now);
        let analyzed_grammar = AnalyzedGrammar::from_grammar_def(&prepared_grammar);
        if let Err(message) = analyzed_grammar.check_table_build_normal_form() {
            panic!("[glrmask] grammar precondition violations:\n{}", message);
        }
        let analysis_ms = analysis_started_at.map_or(0.0, elapsed_ms);

        let ((tokenizer, tokenizer_ms), (table, table_ms)) = rayon::join(
            || {
                let started_at = Instant::now();
                let mut tokenizer = build_dynamic_tokenizer(&prepared_grammar);
                tokenizer.isolate_start_state_and_drain_nullable_terminals();
                (tokenizer, elapsed_ms(started_at))
            },
            || {
                let started_at = Instant::now();
                let table = GLRTable::build_with_default_construction(
                    &analyzed_grammar,
                    default_table_construction,
                );
                (table, elapsed_ms(started_at))
            },
        );

        let finalize_started_at = profile.then(Instant::now);
        let mut constraint = DynamicConstraint::from_parts(
            table,
            analyzed_grammar.terminal_display_names.clone(),
            tokenizer,
            prepared_grammar.ignore_terminal,
            collect_special_token_terminals(&prepared_grammar),
            vocab,
        );
        constraint
            .inner
            .table
            .set_embedded_start_nullable(start_nullable);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_compile] prepare_ms={:.3} analysis_ms={:.3} tokenizer_ms={:.3} table_ms={:.3} finalize_ms={:.3} parallel_core_wall_ms={:.3} total_ms={:.3}",
                prepare_ms,
                analysis_ms,
                tokenizer_ms,
                table_ms,
                finalize_started_at.map_or(0.0, elapsed_ms),
                tokenizer_ms.max(table_ms),
                elapsed_ms(total_started_at),
            );
        }
        constraint
    })
}

pub(crate) fn compile_owned_with_table_construction(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> Constraint {
    let start_nullable = grammar.start_is_nullable();
    if compile_profile_summary_enabled() || compile_top_profile_enabled() {
        let (mut constraint, profile) =
            compile_owned_profiled_with_table_construction(grammar, vocab, default_table_construction);
        constraint
            .table
            .set_embedded_start_nullable(start_nullable);
        emit_compile_profile_summary(None, None, &profile);
        return constraint;
    }

    let prepared_grammar = prepare_grammar(grammar);
    let mut constraint =
        compile_prepared_with_table_construction(prepared_grammar, vocab, default_table_construction);
    constraint
        .table
        .set_embedded_start_nullable(start_nullable);
    constraint
}

pub(crate) fn compile_prepared_with_table_construction(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> Constraint {
    let start_nullable = prepared_grammar.start_is_nullable();
    let mut constraint = compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        default_table_construction,
        None,
    )
    .0;
    constraint
        .table
        .set_embedded_start_nullable(start_nullable);
    constraint
}

#[cfg(test)]
pub(crate) fn compile_owned_with_lexer_adaptive(
    grammar: GrammarDef,
    vocab: &Vocab,
    adaptive: bool,
) -> Constraint {
    let start_nullable = grammar.start_is_nullable();
    let prepared_grammar = prepare_grammar(grammar);
    let mut constraint = compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        GlrTableConstruction::ExperimentalCoreMerged,
        Some(adaptive),
    )
    .0;
    constraint
        .table
        .set_embedded_start_nullable(start_nullable);
    constraint
}

pub(crate) fn compile_owned_profiled(
    grammar: GrammarDef,
    vocab: &Vocab,
) -> (Constraint, CompilePhaseProfile) {
    compile_owned_profiled_with_table_construction(
        grammar,
        vocab,
        GlrTableConstruction::ExperimentalCoreMerged,
    )
}

pub(crate) fn compile_owned_profiled_with_table_construction(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> (Constraint, CompilePhaseProfile) {
    let start_nullable = grammar.start_is_nullable();
    let total_started_at = Instant::now();
    let prepare_started_at = Instant::now();
    let prepared_grammar = prepare_grammar(grammar);
    let prepare_ms = elapsed_ms(prepare_started_at);

    let (mut constraint, mut profile) = compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        default_table_construction,
        None,
    );
    constraint
        .table
        .set_embedded_start_nullable(start_nullable);
    profile.prepare_ms = prepare_ms;
    profile.total_ms = elapsed_ms(total_started_at);
    (constraint, profile)
}
