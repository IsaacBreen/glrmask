use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use once_cell::sync::Lazy;
use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::compile::{
    build_exact_partitioned_runtime_tokenizer,
    build_virtual_unit_repeat_tokenizer,
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
    compile_terminal_expression_pair_with_vocabulary_token_quotient,
    expression_contains_large_bounded_repeat,
    expression_supports_bounded_code_residual_runtime,
    expression_supports_deferred_dense_runtime,
    factor_regex_expr,
    prepare_partitioned_expression_pair_with_structural_map,
    prepare_partitioned_expression_pair_with_vocabulary_token_quotient,
    virtual_binary_bounded_repeat_intersection_descriptor,
    virtual_large_bounded_repeat_descriptor,
    virtual_unit_repeat_descriptor,
    virtual_zero_min_unit_repeat_fits_state_ids,
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
    estimated_synthesis_state_volume,
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
use crate::compiler::stages::templates::characterize::{
    TerminalCharacterization, characterize_terminals_profiled,
};
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::ds::u8set::U8Set;
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

fn lexer_adaptive_enabled() -> bool {
    if std::env::var_os("GLRMASK_LEXER_ADAPTIVE").is_some() {
        env_flag_enabled("GLRMASK_LEXER_ADAPTIVE")
    } else {
        // A depth override is itself an explicit request for adaptive lexer
        // construction. With neither variable present, keep the exact
        // partition union: current corpus measurements show it is both faster
        // to build and lower-tail at runtime than the historical depth-one
        // hybrid.
        std::env::var_os("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH").is_some()
    }
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
    TerminalReconcileAndPreParserCompact,
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
                | Self::TerminalReconcileAndPreParserCompact
                | Self::TerminalReconcileAndParserCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
        )
    }

    fn does_terminal_compact(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcileAndCompact
                | Self::TerminalReconcileAndPreParserCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
        )
    }

    fn does_pre_parser_compact(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcileAndPreParserCompact
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
            "preparser" | "pre_parser" | "preparser_compact" | "pre_parser_compact"
            | "parser_build_compact" | "terminal_preparser_compact" => {
                DwaPossibleMatchesMode::TerminalReconcileAndPreParserCompact
            }
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

#[cfg(target_os = "windows")]
const DEFAULT_COMPILE_THREAD_CAP: usize = 6;
#[cfg(not(target_os = "windows"))]
const DEFAULT_COMPILE_THREAD_CAP: usize = 10;

#[cfg(target_os = "windows")]
fn configure_compile_worker_thread() {
    // Windows may dynamically put long-lived worker threads into execution-speed
    // throttling (EcoQoS) after sustained CPU work. A compile pool is latency
    // sensitive, so explicitly opt its owned worker threads out. StateMask=0
    // requests normal execution speed; ControlMask marks that choice explicit.
    #[repr(C)]
    struct ThreadPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }

    const THREAD_POWER_THROTTLING: i32 = 3;
    const THREAD_POWER_THROTTLING_CURRENT_VERSION: u32 = 1;
    const THREAD_POWER_THROTTLING_EXECUTION_SPEED: u32 = 0x1;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadInformation(
            thread: *mut std::ffi::c_void,
            thread_information_class: i32,
            thread_information: *const std::ffi::c_void,
            thread_information_size: u32,
        ) -> i32;
    }

    let state = ThreadPowerThrottlingState {
        version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        control_mask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        state_mask: 0,
    };
    unsafe {
        let _ = SetThreadInformation(
            GetCurrentThread(),
            THREAD_POWER_THROTTLING,
            (&state as *const ThreadPowerThrottlingState).cast(),
            std::mem::size_of::<ThreadPowerThrottlingState>() as u32,
        );
    }
}

#[cfg(not(target_os = "windows"))]
fn configure_compile_worker_thread() {}

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
            .map(|parallelism| parallelism.get().min(DEFAULT_COMPILE_THREAD_CAP))
            .filter(|&value| value > 1);
    }

    #[cfg(not(target_os = "macos"))]
    {
        return std::thread::available_parallelism()
            .ok()
            .map(|parallelism| parallelism.get().min(DEFAULT_COMPILE_THREAD_CAP))
            .filter(|&value| value > 1);
    }
}

static COMPILE_THREAD_POOL: Lazy<Option<rayon::ThreadPool>> = Lazy::new(|| {
    let thread_count = compile_thread_count()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .start_handler(|_| configure_compile_worker_thread())
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
    /// A synthesized tokenizer replaced the raw compile/runtime coordinate
    /// under a full byte-transition homomorphism.
    pub(crate) synthetic_certified: bool,
    /// A whole-vocabulary-exact quotient was certified and used only as the
    /// initial terminal-DWA observation partition while retaining the full
    /// raw tokenizer.
    pub(crate) synthetic_token_quotient_certified: bool,
    pub(crate) synthetic_observation_states: usize,
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
        "[glrmask/profile][compile] source={}{} prepare_ms={:.3} tokenizer_build_ms={:.3} tokenizer_final_states={} tokenizer_final_transitions={} synthetic_candidate_terminals={} synthetic_certified={} synthetic_token_quotient_certified={} synthetic_observation_states={} synthetic_compile_states={} synthetic_compile_transitions={} synthetic_certification_ms={:.3} analyze_grammar_ms={:.3} glr_table_ms={:.3} terminal_coloring_ms={:.3} disallowed_follows_ms={:.3} analysis_wall_ms={:.3} classify_ms={:.3} id_map_ms={:.3} terminal_dwa_ms={:.3} split_terminal_dwa_total_ms={:.3} global_merge_ms={:.3} templates_ms={:.3} compact_ms={:.3} possible_matches_vocab_equiv_ms={:.3} possible_matches_collect_ms={:.3} possible_matches_materialize_ms={:.3} shared_id_reconcile_ms={:.3} possible_matches_pipeline_ms={:.3} terminal_dwa_interned_ranges_before_pm_reconcile={} possible_matches_interned_ranges_before_pm_reconcile={} terminal_pm_joint_interned_ranges_before_reconcile={} terminal_pm_joint_interned_ranges={} internal_token_bytes_ms={:.3} terminal_run_collapse_ms={:.3} parser_dwa_ms={:.3} parser_dwa_interned_ranges={} possible_matches_interned_ranges={} parser_pm_joint_interned_ranges={} finalize_ms={:.3} compile_ms={:.3} total_ms={:.3}",
        source,
        import_fragment,
        profile.prepare_ms,
        profile.tokenizer_build_ms,
        profile.tokenizer_final_states,
        profile.tokenizer_final_transitions,
        profile.synthetic_candidate_terminals,
        profile.synthetic_certified,
        profile.synthetic_token_quotient_certified,
        profile.synthetic_observation_states,
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

pub(crate) fn composition_grammar_summary_from_analysis(
    grammar: &AnalyzedGrammar,
) -> crate::runtime::CompositionGrammarSummary {
    let num_terminals = grammar.num_terminals as usize;
    let root = grammar
        .rules
        .first()
        .and_then(|augmented| match augmented.rhs.as_slice() {
            [crate::grammar::flat::Symbol::Nonterminal(root)] => Some(*root),
            _ => None,
        });
    let allowed_follows = compute_ever_allowed_follows(grammar)
        .into_iter()
        .map(|row| {
            let mut bits = BitSet::new(num_terminals);
            for terminal in row {
                if (terminal as usize) < num_terminals {
                    bits.set(terminal as usize);
                }
            }
            bits
        })
        .collect::<Vec<_>>();

    let mut last = vec![BitSet::new(num_terminals); grammar.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for rule in &grammar.rules {
            let mut additions = BitSet::new(num_terminals);
            for symbol in rule.rhs.iter().rev() {
                match symbol {
                    crate::grammar::flat::Symbol::Terminal(terminal) => {
                        if (*terminal as usize) < num_terminals {
                            additions.set(*terminal as usize);
                        }
                        break;
                    }
                    crate::grammar::flat::Symbol::Nonterminal(nonterminal) => {
                        if let Some(row) = last.get(*nonterminal as usize) {
                            additions.union_with(row);
                        }
                        if !grammar.nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
            let Some(target) = last.get_mut(rule.lhs as usize) else {
                continue;
            };
            let before = target.count_ones();
            target.union_with(&additions);
            changed |= before != target.count_ones();
        }
        if !changed {
            break;
        }
    }

    let root_first = root
        .and_then(|root| grammar.first.get(root as usize).cloned())
        .unwrap_or_else(|| BitSet::new(num_terminals));
    let root_last = root
        .and_then(|root| last.get(root as usize).cloned())
        .unwrap_or_else(|| BitSet::new(num_terminals));
    let root_nullable = root.is_some_and(|root| grammar.nullable.contains(&root));
    crate::runtime::CompositionGrammarSummary {
        allowed_follows,
        root_first,
        root_last,
        root_nullable,
    }
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

fn build_dynamic_virtual_tokenizer(grammar: &GrammarDef) -> crate::Result<Option<Tokenizer>> {
    const HYBRID_MIN_BOUND: usize = 4_096;

    fn contains_bounded_repeat_at_least(expr: &Expr, threshold: usize) -> bool {
        match expr {
            Expr::Repeat { expr, max, .. } => {
                max.is_some_and(|max| max >= threshold)
                    || contains_bounded_repeat_at_least(expr, threshold)
            }
            Expr::Seq(parts) | Expr::Choice(parts) => parts
                .iter()
                .any(|part| contains_bounded_repeat_at_least(part, threshold)),
            Expr::Intersect { expr, intersect } => {
                contains_bounded_repeat_at_least(expr, threshold)
                    || contains_bounded_repeat_at_least(intersect, threshold)
            }
            Expr::Exclude { expr, exclude } => {
                contains_bounded_repeat_at_least(expr, threshold)
                    || contains_bounded_repeat_at_least(exclude, threshold)
            }
            Expr::Shared(expr) => contains_bounded_repeat_at_least(expr, threshold),
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Dfa(_) | Expr::Epsilon => false,
        }
    }

    let expressions = grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    let giant_terminals = expressions
        .iter()
        .enumerate()
        .filter_map(|(terminal, expression)| {
            expression_contains_large_bounded_repeat(expression)
                .then_some(terminal as TerminalID)
        })
        .collect::<Vec<_>>();
    // Experiment only: force ordinary finite bounded-repeat terminals onto the
    // general exact residual runtime below the normal 4096-state protection
    // threshold. This lets us measure the representation independently from
    // JSON-Schema chunking without changing the accepted language.
    let force_residual_bounded_min = std::env::var(
        "GLRMASK_EXPERIMENT_DYNAMIC_FORCE_RESIDUAL_BOUNDED_MIN",
    )
    .ok()
    .and_then(|value| value.trim().parse::<usize>().ok())
    .filter(|&value| value > 0);
    let forced_residual_terminals = force_residual_bounded_min
        .map(|threshold| {
            expressions
                .iter()
                .enumerate()
                .filter_map(|(terminal, expression)| {
                    contains_bounded_repeat_at_least(expression, threshold)
                        .then_some(terminal as TerminalID)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    // A bounded-code intersection can have a declared bound below the generic
    // 4096 giant-repeat threshold and still explode when eagerly materialized
    // (pattern/format + JSON decoded-length envelopes are a common example).
    // When no older virtual family is already required, let the exact oracle
    // itself certify these terminals for the general residual runtime. Keeping
    // this lane dynamic-only leaves static representation policy unchanged.
    let bounded_code_terminals = if giant_terminals.is_empty() {
        expressions
            .iter()
            .enumerate()
            .filter_map(|(terminal, expression)| {
                expression_supports_bounded_code_residual_runtime(expression)
                    .then_some(terminal as TerminalID)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    if giant_terminals.is_empty()
        && bounded_code_terminals.is_empty()
        && forced_residual_terminals.is_empty()
    {
        return Ok(None);
    }
    if !giant_terminals.is_empty()
        && let Some(tokenizer) = build_virtual_unit_repeat_tokenizer(&expressions)
    {
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][dynamic_tokenizer] path=virtual_unit_repeat states={}",
                tokenizer.num_states(),
            );
        }
        return Ok(Some(tokenizer));
    }

    // Hybrid lane: replace every virtualizable pathological terminal with an
    // impossible physical proxy, finish/drain the ordinary tokenizer, then
    // attach exact virtual components. A single unit-width repeat keeps its
    // O(1) arithmetic counter when possible; multi-component tokenizers use
    // the shared lazy product-state allocator so their handles cannot collide.
    let unit_candidates = expressions
        .iter()
        .enumerate()
        .filter_map(|(terminal, expression)| {
            virtual_unit_repeat_descriptor(expression)
                .filter(|(_, _, max)| *max >= HYBRID_MIN_BOUND)
                .map(|(body, min, max)| (terminal as TerminalID, body, min, max))
        })
        .collect::<Vec<_>>();
    let virtual_candidates = expressions
        .iter()
        .enumerate()
        .filter_map(|(terminal, expression)| {
            virtual_binary_bounded_repeat_intersection_descriptor(expression)
                .or_else(|| virtual_large_bounded_repeat_descriptor(expression))
                .map(|descriptor| (terminal as TerminalID, descriptor))
        })
        .collect::<Vec<_>>();

    let specialized_terminals = virtual_candidates
        .iter()
        .map(|(terminal, _)| *terminal)
        .collect::<BTreeSet<_>>();
    let all_giants_specialized = giant_terminals
        .iter()
        .all(|terminal| specialized_terminals.contains(terminal));

    let build_error = |detail: &str| {
        crate::Error::Compilation(format!(
            "validated protected tokenizer component could not stay on its exact virtual runtime path ({detail}); refusing eager materialization"
        ))
    };
    let mut general_residual_terminals_storage = if giant_terminals.is_empty() {
        bounded_code_terminals.clone()
    } else {
        giant_terminals.clone()
    };
    for terminal in forced_residual_terminals {
        if !general_residual_terminals_storage.contains(&terminal) {
            general_residual_terminals_storage.push(terminal);
        }
    }
    let general_residual_terminals = &general_residual_terminals_storage;

    let build_general_residual = || -> crate::Result<Option<Tokenizer>> {
        let mut proxy_expressions = expressions.clone();
        for &terminal in general_residual_terminals {
            proxy_expressions[terminal as usize] = Expr::U8Class(U8Set::empty());
        }
        let terminal_labels = grammar
            .terminals
            .iter()
            .enumerate()
            .map(|(index, _)| grammar.terminal_display_name(index as u32))
            .collect::<Vec<_>>();
        let partition_ids = lexer_partition_ids(grammar);
        let residual_isolation_classes = lexer_residual_isolation_classes(grammar);
        let mut tokenizer = build_tokenizer_from_exprs_partitioned_impl(
            &proxy_expressions,
            Some(&terminal_labels),
            &partition_ids,
            Some(&residual_isolation_classes),
            None,
        );
        tokenizer.isolate_start_state_and_drain_nullable_terminals();
        tokenizer
            .restore_terminal_exprs_without_virtual_runtime(Some(expressions.clone()))
            .map_err(|detail| build_error(&format!("terminal expression restoration failed: {detail}")))?;
        tokenizer
            .install_virtual_residual_components(
                general_residual_terminals
                    .iter()
                    .map(|&terminal| (expressions[terminal as usize].clone(), terminal))
                    .collect(),
            )
            .ok_or_else(|| build_error("general residual component installation failed"))?;
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][dynamic_tokenizer] path=hybrid_virtual_residuals physical_states={} components={}",
                tokenizer.num_states(),
                general_residual_terminals.len(),
            );
        }
        Ok(Some(tokenizer))
    };

    if giant_terminals.is_empty()
        || !all_giants_specialized
        || force_residual_bounded_min.is_some()
    {
        return build_general_residual();
    }

    let mut proxy_expressions = expressions.clone();
    for (terminal, _) in &virtual_candidates {
        proxy_expressions[*terminal as usize] = Expr::U8Class(U8Set::empty());
    }
    let terminal_labels = grammar
        .terminals
        .iter()
        .enumerate()
        .map(|(index, _)| grammar.terminal_display_name(index as u32))
        .collect::<Vec<_>>();
    let partition_ids = lexer_partition_ids(grammar);
    let residual_isolation_classes = lexer_residual_isolation_classes(grammar);
    let mut tokenizer = build_tokenizer_from_exprs_partitioned_impl(
        &proxy_expressions,
        Some(&terminal_labels),
        &partition_ids,
        Some(&residual_isolation_classes),
        None,
    );
    // Drain ordinary nullable terminals before reserving the arithmetic state
    // interval. A second drain by the caller is then a no-op.
    tokenizer.isolate_start_state_and_drain_nullable_terminals();
    let profile_kind;
    let profile_bound;
    if virtual_candidates.len() == 1
        && let Some((virtual_terminal, body, min, max)) = unit_candidates.into_iter().next()
    {
        let Some(physical_state_count) = u32::try_from(tokenizer.num_states())
            .ok()
            .and_then(|states| states.checked_add(1))
        else {
            return build_general_residual();
        };
        if virtual_zero_min_unit_repeat_fits_state_ids(max, physical_state_count) {
            if tokenizer
                .install_virtual_unit_repeat_component(body, min, max, virtual_terminal)
                .is_none()
            {
                return build_general_residual();
            }
            profile_kind = "hybrid_virtual_unit_repeat";
            profile_bound = max;
        } else {
            let Some(descriptor) = virtual_large_bounded_repeat_descriptor(
                &expressions[virtual_terminal as usize],
            ) else {
                return build_general_residual();
            };
            profile_bound = descriptor.left.max as usize;
            if tokenizer
                .install_virtual_binary_repeat_intersection_component(
                    descriptor,
                    virtual_terminal,
                )
                .is_none()
            {
                return build_general_residual();
            }
            profile_kind = "hybrid_virtual_bounded_repeat";
        }
    } else {
        profile_bound = virtual_candidates
            .iter()
            .map(|(_, descriptor)| descriptor.left.max.max(descriptor.right.max) as usize)
            .max()
            .ok_or_else(|| build_error("virtual candidate set unexpectedly became empty"))?;
        if tokenizer
            .install_virtual_binary_repeat_intersection_components(
                virtual_candidates
                    .into_iter()
                    .map(|(terminal, descriptor)| (descriptor, terminal))
                    .collect(),
            )
            .is_none()
        {
            return build_general_residual();
        }
        profile_kind = "hybrid_virtual_repeat_components";
    }
    tokenizer
        .restore_terminal_exprs(Some(expressions))
        .map_err(|detail| build_error(&format!("terminal expression restoration failed: {detail}")))?;
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][dynamic_tokenizer] path={} physical_states={} bound={}",
            profile_kind,
            tokenizer.num_states(),
            profile_bound,
        );
    }
    Ok(Some(tokenizer))
}

fn build_dynamic_tokenizer(grammar: &GrammarDef) -> crate::Result<Tokenizer> {
    const LARGE_DYNAMIC_LEXER_TERMINALS: usize = 96;

    // Select the virtual lane before any general regex/NFA construction: the
    // latter is exactly where a huge bounded repeat would be materialized.
    if let Some(tokenizer) = build_dynamic_virtual_tokenizer(grammar)? {
        return Ok(tokenizer);
    }

    let explicit_policy = std::env::var_os("GLRMASK_LEXER_SINGLETONS").is_some()
        || std::env::var_os("GLRMASK_LEXER_ADAPTIVE").is_some()
        || std::env::var_os("GLRMASK_ADAPTIVE_LEXER_MAX_DEPTH").is_some();
    let expressions = grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .map(factor_regex_expr)
        .collect::<Vec<_>>();
    if !explicit_policy {
        let labels = grammar
            .terminals
            .iter()
            .enumerate()
            .map(|(index, _)| grammar.terminal_display_name(index as u32))
            .collect::<Vec<_>>();
        let mut residual_isolation_classes = lexer_residual_isolation_classes(grammar);
        let mut next_class = residual_isolation_classes
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |class| class.saturating_add(1));
        let mut deferred_terminals = 0usize;
        for (terminal, expression) in expressions.iter().enumerate() {
            if expression_supports_deferred_dense_runtime(expression) {
                residual_isolation_classes[terminal] = Some(next_class);
                next_class = next_class
                    .checked_add(1)
                    .expect("residual isolation class id overflow");
                deferred_terminals += 1;
            }
        }
        if deferred_terminals > 0 {
            let partition_ids = lexer_partition_ids_with_residual_classes(
                grammar,
                false,
                &residual_isolation_classes,
            );
            let tokenizer = build_exact_partitioned_runtime_tokenizer(
                &expressions,
                Some(&labels),
                &partition_ids,
                &residual_isolation_classes,
            );
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][dynamic_tokenizer] path=direct_exact_runtime deferred_terminals={} states={}",
                    deferred_terminals,
                    tokenizer.num_states(),
                );
            }
            return Ok(tokenizer);
        }
    }
    if !explicit_policy && grammar.terminals.len() >= LARGE_DYNAMIC_LEXER_TERMINALS {
        // Large source-state grammars often contain thousands of exact-line
        // terminals with substantial shared prefixes. Keeping each terminal in
        // a singleton partition duplicates those prefixes in the runtime NFA
        // and makes dynamic mask generation walk every partition. Build one
        // combined product tokenizer instead: it preserves terminal identities
        // while sharing prefix states and gives the runtime a deterministic
        // transition structure.
        Ok(build_tokenizer_with_partition_options(grammar, false, false))
    } else {
        Ok(build_tokenizer(grammar))
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
        compile_owned_profiled_with_table_construction, lexer_partition_ids_with_options,
        prepare_structural_tokenizer_pair, plan_synthetic_tokenizer_enabled,
        structural_state_reduction_is_profitable,
    };
    use crate::automata::lexer::Lexer;
    use crate::automata::regex::Expr;
    use crate::compiler::glr::table::GlrTableConstruction;
    use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};
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
    fn profiled_compile_uses_a_certified_synthetic_coordinate() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min: 1,
                    max: Some(5_000),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaaa".to_vec()),
            (3, b"x".to_vec()),
        ]);

        let (_, profile) = compile_owned_profiled_with_table_construction(
            grammar,
            &vocab,
            GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            profile.synthetic_certified || profile.synthetic_token_quotient_certified,
            "synthetic planning must retain either the raw or vocabulary-token certificate",
        );
        assert!(profile.synthetic_candidate_terminals > 0);
        assert!(profile.synthetic_observation_states < profile.tokenizer_final_states);
        assert!(profile.synthetic_compile_states < profile.tokenizer_final_states);
    }

    #[test]
    fn structural_token_quotient_preisolates_ordinary_nullable_components() {
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
            prepare_structural_tokenizer_pair(&grammar, &plan, &vocab, Some(false), true)
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
            // Preflight asks only whether the candidate is a useful exact
            // whole-token quotient. Raw lexer substitution is decided later by
            // the stricter homomorphism constructor; a token-only candidate
            // keeps the full compile/runtime lexer.
            let pair = compile_terminal_expression_pair_with_vocabulary_token_quotient(
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
    vocabulary_token_quotient: bool,
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
            let pair = if vocabulary_token_quotient {
                compile_terminal_expression_pair_with_vocabulary_token_quotient(
                    &full_expressions[0],
                    &synthesized_expressions[0],
                    vocab,
                    plan.repeat_horizons.as_ref(),
                    max_token_len,
                    &relevant_bytes,
                )
            } else {
                compile_terminal_expression_pair_with_structural_map(
                    &full_expressions[0],
                    &synthesized_expressions[0],
                    vocab,
                    plan.repeat_horizons.as_ref(),
                    max_token_len,
                    &relevant_bytes,
                )
            }?;
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
                .unwrap_or_else(lexer_adaptive_enabled);
            let pair = if vocabulary_token_quotient {
                prepare_partitioned_expression_pair_with_vocabulary_token_quotient(
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
                )
            } else {
                prepare_partitioned_expression_pair_with_structural_map(
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
                )
            }?;
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
    if constraint.packed_token_bytes.is_none() {
        constraint.packed_token_bytes = Some(std::sync::Arc::new(
            crate::runtime::PackedTokenBytes::from_runtime_entries(&constraint.token_bytes)
            .expect("compiler-produced token bytes should form a valid indexed runtime vocabulary"),
        ));
    }
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
    Vec<Option<TerminalCharacterization>>,
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
        return (
            templates,
            template_dfas_by_terminal,
            vec![None; analyzed_grammar.num_terminals as usize],
            templates_ms,
        );
    }

    let (characterizations, characterization_profile) =
        characterize_terminals_profiled(table, analyzed_grammar);
    let (templates, template_profile) =
        Templates::from_characterizations_profiled(&characterizations);
    let mut composition_parser_characterizations_by_terminal =
        vec![None; analyzed_grammar.num_terminals as usize];
    for (terminal, characterization) in characterizations {
        if let Some(slot) = composition_parser_characterizations_by_terminal
            .get_mut(terminal as usize)
        {
            *slot = Some(characterization);
        }
    }
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
        composition_parser_characterizations_by_terminal,
        elapsed_ms(templates_started_at),
    )
}

#[derive(Clone)]
struct TokenizerDagLane {
    tokenizer: Arc<Tokenizer>,
    initial_state_map: Option<ManyToOneIdMap>,
    partition_local_synthesis_plan: Option<Arc<
        crate::compiler::stages::id_map_and_terminal_dwa::PartitionLocalSynthesisPlan,
    >>,
    prepared_partition_local_tokenizers: Option<Arc<
        crate::compiler::stages::id_map_and_terminal_dwa::PreparedPartitionLocalTokenizers,
    >>,
    synthetic_candidate_terminals: usize,
    synthetic_token_quotient_certified: bool,
    synthetic_observation_states: usize,
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
    flat_trans: Arc<[u32]>,
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
    composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    templates_ms: f64,
    templates_started_ms: f64,
    templates_finished_ms: f64,
}

#[derive(Default)]
struct ParserDagJoinState {
    terminal: Option<TerminalDagResult>,
    templates: Option<TemplatesDagResult>,
    possible_matches_id_map: Option<InternalIdMap>,
    launched: bool,
}

struct CompileDagResult {
    tokenizer: Arc<Tokenizer>,
    synthetic_candidate_terminals: usize,
    synthetic_token_quotient_certified: bool,
    synthetic_observation_states: usize,
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
    composition_parser_characterizations_by_terminal:
        Vec<Option<TerminalCharacterization>>,
    templates_ms: f64,
    classify_ms: f64,
    flat_trans: Arc<[u32]>,
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
    prebuilt_token_mask_caches: Option<(InternalIdMap, crate::runtime::TokenMaskCachePrebuild)>,
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
    templates: &Templates,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    final_id_map: Option<&std::sync::OnceLock<InternalIdMap>>,
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

    let parser_dwas = if direct_parts.is_none() {
        if let Some(final_id_map) = final_id_map.and_then(std::sync::OnceLock::get) {
            if parser_dwas.len() == 2 {
                let mut parser_dwas = parser_dwas.into_iter();
                let left = parser_dwas.next().expect("two parser families have a left member");
                let right = parser_dwas.next().expect("two parser families have a right member");
                let (left, right) = rayon::join(
                    || left.remap_into_existing_common(final_id_map),
                    || right.remap_into_existing_common(final_id_map),
                );
                vec![left, right]
            } else {
                parser_dwas
                    .into_iter()
                    .map(|parser| parser.remap_into_existing_common(final_id_map))
                    .collect()
            }
        } else {
            parser_dwas
        }
    } else {
        parser_dwas
    };

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
                glrmask_parser_dwa::__private::merge::merge_mapped_parser_dwas(
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
        glrmask_parser_dwa::__private::merge::merge_mapped_parser_dwas_with_top_accept(
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

fn common_terminal_family_id_map(terminal_dwas: &TerminalDwaFamilies) -> InternalIdMap {
    // Ordinary parser-family merge keeps the deeper L2P family as the primary
    // coordinate and folds the immediate L1 top-accept overlay into it. Mirror
    // that left-to-right ordering so map-only construction gets identical class
    // numbering, not merely the same equivalence partition.
    let maps = [
        terminal_dwas.l2p.as_ref().map(MappedArtifact::id_map),
        terminal_dwas.l1.as_ref().map(MappedArtifact::id_map),
        terminal_dwas.special.as_ref().map(MappedArtifact::id_map),
    ];
    let mut maps = maps.into_iter().flatten();
    let mut common = maps
        .next()
        .expect("terminal families are non-empty when parser construction starts")
        .clone();
    for next in maps {
        common = crate::compiler::stages::mapped_artifact::common_internal_id_map(&[
            &common,
            next,
        ]);
    }
    common
}

fn same_internal_id_map_numbering(left: &InternalIdMap, right: &InternalIdMap) -> bool {
    left.tokenizer_states.original_to_internal == right.tokenizer_states.original_to_internal
        && left.vocab_tokens.original_to_internal == right.vocab_tokens.original_to_internal
}

fn compacted_possible_matches_id_map(
    result: &cpm::ConstraintPossibleMatchesComputation,
) -> InternalIdMap {
    let mut mapped = result.mapped_possible_matches.clone();
    if compact_possible_matches_before_reconcile_enabled() {
        let _ = mapped.compact_dimensions_fast();
    }
    mapped.id_map().clone()
}


fn maybe_dump_terminal_dwa_experiment(
    families: &TerminalDwaFamilies,
    terminal_display_names: &[String],
) {
    let Ok(path) = std::env::var("GLRMASK_EXPERIMENT_DUMP_TERMINAL_DWA") else {
        return;
    };
    use std::io::Write;
    let copies = [&families.l1, &families.l2p, &families.special]
        .into_iter()
        .filter_map(|family| family.as_ref())
        .map(|family| MappedArtifact::new(family.artifact().clone(), family.id_map().clone()))
        .collect::<Vec<_>>();
    if copies.is_empty() {
        return;
    }
    let reconciled = MappedArtifact::reconcile_vec(copies);
    let (automata, id_map) = reconciled.into_parts();
    let mut union = crate::automata::weighted_u32::nwa::NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let mut starts = Vec::new();
    for automaton in automata {
        let nwa = match automaton {
            TerminalAutomaton::Dwa(dwa) => dwa.to_nwa(),
            TerminalAutomaton::TokenDeterministicNwa(nwa)
            | TerminalAutomaton::EpsilonNwa(nwa) => nwa,
        };
        let body = union.append_with_body(&nwa);
        starts.extend(body.start_states);
    }
    union.set_start_states(starts);
    let dwa = crate::automata::weighted_u32::determinize::determinize(&union).expect("terminal-DWA experiment dump union must determinize");

    fn write_u32(out: &mut Vec<u8>, value: u32) { out.extend_from_slice(&value.to_le_bytes()); }
    fn write_u64(out: &mut Vec<u8>, value: u64) { out.extend_from_slice(&value.to_le_bytes()); }
    fn write_vec(out: &mut Vec<u8>, values: &[u32]) {
        write_u32(out, values.len() as u32);
        for &value in values { write_u32(out, value); }
    }
    fn write_vec_vec(out: &mut Vec<u8>, values: &[Vec<u32>]) {
        write_u32(out, values.len() as u32);
        for value in values { write_vec(out, value); }
    }
    fn write_map(out: &mut Vec<u8>, map: &ManyToOneIdMap) {
        write_vec(out, &map.original_to_internal);
        write_vec_vec(out, &map.internal_to_originals);
        write_vec(out, &map.representative_original_ids);
    }

    let encoded_dwa = bincode::serialize(&dwa).expect("serialize terminal-DWA experiment dump");
    let mut out = Vec::with_capacity(encoded_dwa.len() + 1024);
    out.extend_from_slice(b"GLRMTD1\0");
    write_u32(&mut out, terminal_display_names.len() as u32);
    for name in terminal_display_names {
        write_u32(&mut out, name.len() as u32);
        out.extend_from_slice(name.as_bytes());
    }
    write_map(&mut out, &id_map.tokenizer_states);
    write_map(&mut out, &id_map.vocab_tokens);
    write_u64(&mut out, encoded_dwa.len() as u64);
    out.extend_from_slice(&encoded_dwa);
    let mut file = std::fs::File::create(&path).expect("create terminal-DWA experiment dump");
    file.write_all(&out).expect("write terminal-DWA experiment dump");
    eprintln!(
        "[glrmask/experiment][terminal_dwa_dump] path={} states={} transitions={} tsids={} tokens={} bytes={}",
        path,
        dwa.num_states(),
        dwa.num_transitions(),
        id_map.num_tsids(),
        id_map.num_internal_tokens(),
        out.len(),
    );
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
                state.possible_matches_id_map.take(),
            ))
        }
    };

    let Some((terminal, templates, possible_matches_id_map)) = ready else {
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
            flat_trans,
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
            composition_parser_characterizations_by_terminal,
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

        let (templates, prebuilt_parser_dwa, prebuilt_token_mask_caches) =
            if dwa_pm_mode.does_terminal_reconcile() {
                (Some(templates), None, None)
            } else {
                let parser_dwa_started_at = Instant::now();
                let parser_dwa_started_ms = elapsed_ms(compile_started_at.clone());
                let early_cache_enabled = env_flag_enabled_by_default("GLRMASK_EARLY_TOKEN_CACHE_PREBUILD");
                let parser_final_id_map = std::sync::OnceLock::new();
                let parser_final_coordinate_merge_max_states = std::env::var(
                    "GLRMASK_PARSER_FINAL_COORDINATE_MERGE_MAX_TOKENIZER_STATES",
                )
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
                .unwrap_or(512);
                let parser_final_coordinate_merge = env_flag_enabled_by_default("GLRMASK_PARSER_FINAL_COORDINATE_MERGE")
                    && early_cache_enabled
                    && possible_matches_id_map.is_some()
                    && !dwa_pm_mode.does_parser_compact()
                    && tokenizer.tokenizer.num_states() <= parser_final_coordinate_merge_max_states;
                let parser_final_id_map_target =
                    parser_final_coordinate_merge.then_some(&parser_final_id_map);
                let (parser_dwa, prebuilt_token_mask_caches) = rayon::join(
                    || {
                        build_and_merge_parser_dwa_families(
                            &terminal_dwas,
                            &table,
                            &analysis.analyzed_grammar,
                            ignore_terminal,
                            &templates,
                            &tokenizer.tokenizer,
                            vocab,
                            parser_final_id_map_target,
                        )
                    },
                    || {
                        if !early_cache_enabled {
                            return None;
                        }
                        let possible_matches_id_map = possible_matches_id_map.as_ref()?;
                        let parser_id_map = common_terminal_family_id_map(&terminal_dwas);
                        let final_id_map =
                            crate::compiler::stages::mapped_artifact::common_internal_id_map(&[
                                &parser_id_map,
                                possible_matches_id_map,
                            ]);
                        if parser_final_coordinate_merge {
                            parser_final_id_map
                                .set(final_id_map.clone())
                                .expect("parser final ID map is published once");
                        }
                        let internal_to_tokens =
                            final_id_map.vocab_tokens.internal_to_originals_vecs();
                        let mask_words = final_id_map
                            .vocab_tokens
                            .original_to_internal
                            .len()
                            .div_ceil(32);
                        let caches = crate::runtime::TokenMaskCachePrebuild::build(
                            &final_id_map.vocab_tokens.original_to_internal,
                            &internal_to_tokens,
                            mask_words,
                        );
                        Some((final_id_map, caches))
                    },
                );
                let parser_dwa_ms = elapsed_ms(parser_dwa_started_at);
                let parser_dwa_finished_ms = elapsed_ms(compile_started_at);
                (
                    Some(templates),
                    Some((
                        parser_dwa,
                        parser_dwa_ms,
                        parser_dwa_started_ms,
                        parser_dwa_finished_ms,
                    )),
                    prebuilt_token_mask_caches,
                )
            };

        *result.lock().expect("compile DAG result poisoned") = Some(CompileDagResult {
            tokenizer: tokenizer.tokenizer,
            synthetic_candidate_terminals: tokenizer.synthetic_candidate_terminals,
            synthetic_token_quotient_certified: tokenizer.synthetic_token_quotient_certified,
            synthetic_observation_states: tokenizer.synthetic_observation_states,
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
            composition_parser_characterizations_by_terminal,
            templates_ms,
            classify_ms,
            flat_trans,
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
            prebuilt_token_mask_caches,
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
                flat_trans: flat_global.flat_trans,
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
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!(
                "[glrmask/profile][compile_dag_classify_start] tokenizer_states={} terminals={} disallowed={}",
                tokenizer.tokenizer.num_states(),
                analysis.analyzed_grammar.num_terminals,
                token_path_disallowed_follows.len(),
            );
        }
        prewarm_shared_classify_cache(
            &tokenizer.tokenizer,
            analysis.analyzed_grammar.num_terminals,
            &shared_classify_cache,
        );
        let classify_ms = elapsed_ms(classify_started_at);
        if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
            eprintln!(
                "[glrmask/profile][compile_dag_classify_end] ms={:.3}",
                classify_ms,
            );
        }
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
        None,
    )
}

fn compile_prepared_with_profile_and_table_construction(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
    lexer_adaptive_override: Option<bool>,
    protected_shift_terminals: Option<Arc<Vec<u32>>>,
) -> (Constraint, CompilePhaseProfile) {
    // Synthetic lexer planning may certify a smaller exact tokenizer against
    // the vocabulary. Install the cross-crate certifier before planning on
    // every static compilation path, including profiled and explicit table
    // construction entry points.
    crate::automata::lexer::compile::install_vocabulary_exact_state_certifier(
        crate::compiler::stages::id_map_and_terminal_dwa::synthetic_state_map::certify_vocabulary_exact_state_candidates,
    );
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
                    .unwrap_or_else(lexer_adaptive_enabled),
                global_max_token_len: vocab
                    .entries_map()
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
                        let select_pair = |vocabulary_token_quotient: bool| {
                            prepare_structural_tokenizer_pair(
                                prepared_grammar_ref,
                                plan,
                                vocab,
                                lexer_adaptive_override,
                                vocabulary_token_quotient,
                            )
                            .and_then(|(synthesized, full, certified)| {
                                structural_state_reduction_is_profitable(
                                    full.num_states(),
                                    synthesized.num_states() as usize,
                                )
                                .then_some((synthesized, full, certified))
                            })
                        };

                        if let Some((synthesized, full, certified)) = select_pair(true) {
                            // The compile coordinate is a complete-vocabulary-token
                            // quotient. Constructing the same pair first under the
                            // stricter raw-byte proof only to reject and rebuild it
                            // doubles the dominant tokenizer planning work for the
                            // slow cohort. Runtime commits still use the exact full
                            // tokenizer below.
                            (synthesized, Some(full), Some(certified), 0.0, true)
                        } else {
                            (
                                build_ordinary_compile_tokenizer(
                                    prepared_grammar_ref,
                                    lexer_adaptive_override,
                                ),
                                None,
                                None,
                                0.0,
                                false,
                            )
                        }
                    } else {
                        let tokenizer = build_ordinary_compile_tokenizer(
                            prepared_grammar_ref,
                            lexer_adaptive_override,
                        );
                        (tokenizer, None, None, 0.0, false)
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
                    mut deferred_runtime_tokenizer,
                    mut full_to_synthesized_state_map,
                    synthetic_certification_ms,
                    use_full_tokenizer_for_token_quotient,
                ) = global_tokenizer_result;
                let mut initial_state_map = None;
                // The certified vocabulary-token quotient is the compile
                // coordinate by default. Keep the full-coordinate route as a
                // diagnostic kill switch while still constructing the exact
                // tokenizer for runtime commits.
                let direct_token_quotient_compile = use_full_tokenizer_for_token_quotient
                    && env_flag_enabled_by_default("GLRMASK_DIRECT_TOKEN_QUOTIENT_COMPILE");
                if use_full_tokenizer_for_token_quotient
                    && !direct_token_quotient_compile
                    && let (Some(deferred), Some(certified)) = (
                        deferred_runtime_tokenizer.take(),
                        full_to_synthesized_state_map.take(),
                    )
                {
                    let synthesized_states = tokenizer.num_states() as usize;
                    let mut quotient_id_by_synthesized = vec![u32::MAX; synthesized_states];
                    let mut quotient_states = 0u32;
                    let mut full_to_quotient = certified.full_to_synthesized;
                    for state in &mut full_to_quotient {
                        let slot = quotient_id_by_synthesized
                            .get_mut(*state as usize)
                            .expect("certified synthesized state is in range");
                        if *slot == u32::MAX {
                            *slot = quotient_states;
                            quotient_states += 1;
                        }
                        *state = *slot;
                    }
                    initial_state_map = Some(
                        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                            full_to_quotient,
                            quotient_states,
                        ),
                    );
                    tokenizer = deferred.finish();
                    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                        eprintln!(
                            "[glrmask/profile][tokenizer] token_quotient_compile_coordinate=full full_states={} quotient_states={}",
                            tokenizer.num_states(),
                            initial_state_map
                                .as_ref()
                                .map_or(0, ManyToOneIdMap::num_internal_ids),
                        );
                    }
                }
                let tokenizer_construct_ms = elapsed_ms(tok_started);
                let isolate_started = Instant::now();
                if deferred_runtime_tokenizer.is_none() {
                    tokenizer.isolate_start_state_and_drain_nullable_terminals();
                }
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    if use_full_tokenizer_for_token_quotient {
                        eprintln!(
                            "[glrmask/profile][tokenizer] token_quotient_compile_coordinate={} compile_states={} runtime_deferred={}",
                            if direct_token_quotient_compile {
                                "synthesized"
                            } else {
                                "full"
                            },
                            tokenizer.num_states(),
                            deferred_runtime_tokenizer.is_some(),
                        );
                    }
                    eprintln!(
                        "[glrmask/profile][tokenizer] construction_vs_isolation construct_ms={:.3} isolate_ms={:.3} total_ms={:.3}",
                        tokenizer_construct_ms,
                        elapsed_ms(isolate_started),
                        elapsed_ms(tok_started),
                    );
                }
                let compile_tokenizer_states = tokenizer.num_states() as usize;
                let compile_tokenizer_transitions = tokenizer.transition_count();
                let synthetic_token_quotient_certified = use_full_tokenizer_for_token_quotient
                    && (direct_token_quotient_compile || initial_state_map.is_some());
                let synthetic_observation_states = initial_state_map.as_ref().map_or(
                    compile_tokenizer_states,
                    |state_map| state_map.num_internal_ids() as usize,
                );
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
                    initial_state_map,
                    partition_local_synthesis_plan,
                    prepared_partition_local_tokenizers,
                    synthetic_candidate_terminals: synthetic_tokenizer_plan_ref
                        .map_or(0, |plan| plan.changed_terminal_count),
                    synthetic_token_quotient_certified,
                    synthetic_observation_states,
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
                        if env_flag_enabled_by_default("GLRMASK_EARLY_TOKEN_CACHE_PREBUILD") {
                            parser_state_ref
                                .lock()
                                .expect("parser DAG join state poisoned")
                                .possible_matches_id_map =
                                Some(compacted_possible_matches_id_map(&result));
                        }
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
                let flat_global_initial_state_map = tokenizer_lane.initial_state_map.clone();
                let compile_started_for_terminal = compile_started_for_tokenizer.clone();
                scope.spawn(move |scope| {
                    let flat_global_started_ms = elapsed_ms(compile_started_for_terminal.clone());
                    let flat_trans_started_at = Instant::now();
                    if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
                        eprintln!(
                            "[glrmask/profile][compile_dag_flat_start] tokenizer_states={}",
                            flat_global_tokenizer.num_states(),
                        );
                    }
                    let flat_trans: Arc<[u32]> = Arc::from(
                        crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(
                            &flat_global_tokenizer,
                        ),
                    );
                    let flat_trans_ms = elapsed_ms(flat_trans_started_at);
                    if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
                        eprintln!(
                            "[glrmask/profile][compile_dag_flat_end] ms={:.3}",
                            flat_trans_ms,
                        );
                    }
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
                            if env_flag_enabled_by_default("GLRMASK_EARLY_TOKEN_CACHE_PREBUILD") {
                                parser_state_ref
                                    .lock()
                                    .expect("parser DAG join state poisoned")
                                    .possible_matches_id_map =
                                    Some(compacted_possible_matches_id_map(&result));
                            }
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
                    if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
                        eprintln!("[glrmask/profile][compile_dag_global_max_start]");
                    }
                    let global_max_length_state_map =
                        crate::compiler::stages::id_map_and_terminal_dwa::build_global_max_length_state_map_with_initial(
                            &flat_global_tokenizer,
                            vocab,
                            &flat_trans,
                            flat_global_initial_state_map.as_ref(),
                        );
                    let global_max_length_ms = elapsed_ms(global_max_length_started_at);
                    if std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
                        eprintln!(
                            "[glrmask/profile][compile_dag_global_max_end] ms={:.3}",
                            global_max_length_ms,
                        );
                    }
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
                let analyzed_grammar = Arc::new(match protected_shift_terminals.as_deref() {
                    Some(protected) => AnalyzedGrammar::from_grammar_def_with_protected_shift_terminals(
                        prepared_grammar_ref,
                        protected.iter().copied(),
                    ),
                    None => AnalyzedGrammar::from_grammar_def(prepared_grammar_ref),
                });
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
                        let (
                            templates,
                            template_dfas_by_terminal,
                            composition_parser_characterizations_by_terminal,
                            templates_ms,
                        ) = build_templates_for_compile(
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
                            composition_parser_characterizations_by_terminal,
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
            synthetic_token_quotient_certified,
            synthetic_observation_states,
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
            composition_parser_characterizations_by_terminal,
            templates_ms,
            classify_ms,
            flat_trans,
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
            prebuilt_token_mask_caches,
        } = compile_dag_result
            .into_inner()
            .expect("compile DAG result slot poisoned")
            .expect("compile DAG did not produce a result");
        maybe_dump_terminal_dwa_experiment(
            &terminal_dwas,
            &analyzed_grammar.terminal_display_names,
        );
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
        profile.synthetic_token_quotient_certified = synthetic_token_quotient_certified;
        profile.synthetic_observation_states = synthetic_observation_states;
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
                .as_ref()
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

                if dwa_pm_mode.does_pre_parser_compact() {
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
                        None,
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
                        None,
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
                    None,
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
        let precomputed_parser_pm_id_map = prebuilt_token_mask_caches
            .as_ref()
            .map(|(id_map, _)| id_map);
        let mut parser_pm_pair = if !dwa_pm_mode.does_parser_compact()
            && let Some(common_id_map) = precomputed_parser_pm_id_map
        {
            let parser_dwa = parser_dwa.remap_into_existing_common(common_id_map);
            let possible_matches = possible_matches.remap_into_existing_common(common_id_map);
            MappedArtifact::new(
                (parser_dwa.into_artifact(), possible_matches.into_artifact()),
                common_id_map.clone(),
            )
        } else {
            MappedArtifact::from((parser_dwa, possible_matches))
        };
        shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
        if dwa_pm_mode.does_parser_compact() {
            let compact_started_at = Instant::now();
            parser_pm_pair.compact_dimensions();
            profile.compact_ms += elapsed_ms(compact_started_at);
        }
        let ((parser_dwa_artifact, possible_matches_artifact), internal_ids) =
            parser_pm_pair.into_parts();
        let prebuilt_token_mask_caches = prebuilt_token_mask_caches.and_then(
            |(prebuilt_id_map, caches)| {
                same_internal_id_map_numbering(&prebuilt_id_map, &internal_ids).then_some(caches)
            },
        );
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
        let parser_dwa = parser_dwa.share_exact_transition_rows_owned();
        // Large parser DWAs are much cheaper to serialize and load from the
        // immutable packed runtime representation. Small/medium schema DWAs are
        // deliberately left materialized: their first save/load is already only
        // a few milliseconds, so even small amounts of extra finalization work
        // would be a bad trade. The gap between the two populations is large in
        // practice (hundreds of states versus tens of thousands).
        // Keep the compact runtime representation for the upper ordinary-
        // schema tail as well as giant parser DWAs. The p99-ish population is
        // already expensive enough to repack on first save, while the p50/p90
        // population remains below this cutoff and pays no extra finalization.
        // This boundary is deliberately well above their observed ~136/~275
        // states but below the ~454-state p99 representative.
        const PACKED_RUNTIME_DWA_STATE_THRESHOLD: usize = 384;
        let build_packed_parser_dwa =
            parser_dwa.states().len() >= PACKED_RUNTIME_DWA_STATE_THRESHOLD;
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
        let token_bytes = vocab.entries_arc();
        let special_token_terminals = collect_special_token_terminals(&prepared_grammar);
        let ignore_expr = prepared_grammar
            .ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let reuse_compile_flat_transitions = runtime_tokenizer.is_none()
            && flat_trans.len() == tokenizer.num_states() as usize * 256
            && env_flag_enabled_by_default("GLRMASK_REUSE_COMPILE_FLAT_TRANSITIONS");
        let prebuilt_tokenizer_fast_transitions = if reuse_compile_flat_transitions {
            crate::runtime::FastTokenizerTransitions::Flat(flat_trans)
        } else {
            crate::runtime::FastTokenizerTransitions::default()
        };
        let composition_parser_templates_by_terminal = templates
            .take()
            .map(|templates| {
                let mut by_terminal =
                    vec![None; analyzed_grammar.num_terminals as usize];
                for (terminal, dfa) in templates.by_terminal {
                    if let Some(slot) = by_terminal.get_mut(terminal as usize) {
                        *slot = Some(dfa);
                    }
                }
                by_terminal
            })
            .unwrap_or_default();
        let composition_grammar_summary = std::env::var_os(
            "GLRMASK_DISABLE_COMPOSITION_GRAMMAR_SUMMARY",
        )
        .is_none()
        .then(|| composition_grammar_summary_from_analysis(&analyzed_grammar));
        let mut tokenizer = runtime_tokenizer.unwrap_or(tokenizer);
        if !crate::automata::lexer::tokenizer::artifact_serde::compact_large_runtime(&mut tokenizer) {
            crate::automata::lexer::tokenizer::artifact_serde::compact_large_fast_runtime(&mut tokenizer);
        }
        let mut constraint = Constraint {
            runtime_backend: crate::runtime::ConstraintRuntimeBackend::Static,
            static_dynamic_overlay: None,
            late_grammar_slots: Vec::new(),
            late_bind_vocab: std::sync::OnceLock::from(vocab.clone()),
            scoped_ignore_only_tokens: Vec::new(),
            scoped_ignore_prefix_fusions: Vec::new(),
            parser_dwa,
            packed_parser_dwa: None,
            parser_start_final_override: None,
            parser_top_accept,
            parser_top_accept_parts,
            direct_regular_l1_complete_by_terminal: direct_l1_complete_by_terminal,
            packed_non_dwa_weights: None,
            direct_regular_wide_frontier_acceptance: Vec::new(),
            direct_regular_dynamic_hot_frontiers: Vec::new(),
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
            deferred_internal_tsid_to_states: Default::default(),
            composition_reset_tokens_by_terminal: Vec::new(),
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_parser_templates_by_terminal,
            composition_parser_characterizations_by_terminal,
            composition_grammar_summary,
            terminal_live_states: Vec::new(),
            // Unless optional runtime full-adaptive product states were selected,
            // `runtime_tokenizer_state_map` is a ManyToOne partition: every raw
            // runtime tokenizer state has exactly one internal TSID. Runtime
            // lookup already falls back to `state_to_internal_tsid`; sentinel
            // `[u32::MAX]` prevents generic finalization from allocating 1.4M
            // temporary SmallVec rows and rebuilding an equivalent CSR relation.
            state_internal_tsid_offsets: if runtime_source_state_offset.is_none() {
                vec![u32::MAX]
            } else {
                Vec::new()
            },
            state_internal_tsids: Vec::new(),
            runtime_source_state_offset,
            runtime_product_source_offsets,
            runtime_product_source_states,
            runtime_product_exact_source_states,
            runtime_product_state_by_source_subset: Default::default(),
            original_token_to_internal: internal_ids.vocab_tokens.original_to_internal.clone(),
            packed_original_token_to_internal: None,
            deferred_original_token_to_internal: std::sync::OnceLock::new(),
            internal_token_to_tokens: internal_ids.vocab_tokens.internal_to_originals_vecs(),
            deferred_internal_token_to_tokens: std::sync::OnceLock::new(),
            template_dfas_by_terminal,
            fast_template_dfas_by_terminal: Vec::new(),
            token_bytes,
            packed_token_bytes: None,
            internal_token_bytes,
            token_bytes_dense: Vec::new(),
            internal_token_buf_masks: Vec::new(),
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks: Default::default(),
            quad_word_group_buf_masks: Default::default(),
            super_word_group_buf_masks: Default::default(),
            mega_word_group_buf_masks: Default::default(),
            giga_word_group_buf_masks: Default::default(),
            word_group_sparse_masks: Vec::new(),
            word_group_prefix_buf_masks: Default::default(),
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
            packed_dwa_token_dense_masks: Default::default(),
            weight_token_buf_masks: rustc_hash::FxHashMap::default(),
            weight_token_sparse_buf_masks: rustc_hash::FxHashMap::default(),
            direct_sparse_weight_token_sets: rustc_hash::FxHashSet::default(),
            seed_terminal_dense: rustc_hash::FxHashMap::default(),
            seed_terminal_dense_fallback: Default::default(),
            seed_universe_dense: std::sync::Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Default::default(),
            parser_runtime_caches_prebuilt: false,
            indexed_dag_dense_transitions: Vec::new(),
            indexed_dag_dense_finals: Vec::new(),
            tokenizer_fast_transitions: prebuilt_tokenizer_fast_transitions,
            heavy_token_dense_masks: Vec::new(),
            heavy_token_indices: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            backed_internal_token_buf_flat: None,
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: crate::runtime::mask_mapping::FinalMaskMapping::default(),
            parser_state_domain_labels: Vec::new(),
            ignore_expr,
            serialized_artifact_cache: None,
            deferred_terminal_exprs_blob: None,
            deferred_terminal_exprs: Default::default(),
            deferred_composition_metadata_blob: None,
            composition_link_metadata_materialized: true,
            deferred_table_rules_blob: None,
            deferred_table_rules: Default::default(),
        };
        if build_packed_parser_dwa {
            let packed_started_at = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION")
                .is_some()
                .then(Instant::now);
            let packed = crate::automata::weighted::dwa::PackedRuntimeDwa::from_dwa(
                &constraint.parser_dwa,
            )
            .expect("direct packed-runtime DWA construction should succeed");
            if let Some(started_at) = packed_started_at {
                eprintln!(
                    "[glrmask/profile][packed_runtime_dwa_build] states={} ms={:.3}",
                    constraint.parser_dwa.states().len(),
                    elapsed_ms(started_at),
                );
            }
            // Keep the packed runtime DWA because masking/loading use it
            // directly, but do not pre-encode the serialization wire here.
            // Fresh-save timings must include serialization work rather than
            // shifting it into compile finalization.
            constraint.packed_parser_dwa = Some(std::sync::Arc::new(packed));
        }
        if let Some(caches) = prebuilt_token_mask_caches
            && caches.matches_constraint(&constraint)
        {
            caches.install(&mut constraint);
        }
        let mut constraint = finalize_constraint(constraint);
        crate::runtime::compact_large_non_dwa_weight_runtime(&mut constraint);
        // Keep compiler-produced packed runtime storage in its owned form.
        // Converting it to a backed artifact representation here would perform
        // serialization work during compilation and make first-save benchmarks
        // dishonest. Loaded constraints use a zero-copy backed view of the same
        // logical packed pools; only save() is allowed to create wire bytes.
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
) -> crate::Result<DynamicConstraint> {
    compile_dynamic_owned_impl(grammar, vocab, default_table_construction, true)
}

pub(crate) fn compile_dynamic_owned_unfinalized_with_table_construction(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> crate::Result<DynamicConstraint> {
    compile_dynamic_owned_impl(grammar, vocab, default_table_construction, false)
}

fn compile_dynamic_owned_impl(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
    finalize_runtime: bool,
) -> crate::Result<DynamicConstraint> {
    let start_nullable = grammar.start_is_nullable();
    let profile = compile_profile_enabled();
    let total_started_at = profile.then(Instant::now);
    let prepare_started_at = profile.then(Instant::now);
    // A direct-regular frontend result already contains the complete parser
    // language. Generic CFG normalization cannot improve that automaton and is
    // unnecessary for the dynamic backend, which consumes the retained
    // automaton directly. Keep the terminal definitions untouched for lexer
    // construction.
    let force_cfg_runtime = std::env::var_os("GLRMASK_DYNAMIC_FORCE_CFG_RUNTIME").is_some();
    if profile {
        eprintln!(
            "[glrmask/profile][dynamic_path] input_direct_regular={} force_cfg_runtime={}",
            grammar.direct_regular_automaton.is_some(),
            force_cfg_runtime,
        );
    }
    let mut prepared_grammar = if grammar.direct_regular_automaton.is_some() && !force_cfg_runtime {
        grammar
    } else {
        let mut grammar = grammar;
        if force_cfg_runtime {
            grammar.direct_regular_automaton = None;
        }
        prepare_grammar(grammar)
    };
    let prepare_ms = prepare_started_at.map_or(0.0, elapsed_ms);
    let prepared_has_giant_repeat = prepared_grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .map(factor_regex_expr)
        .any(|expression| expression_contains_large_bounded_repeat(&expression));
    run_with_compile_thread_pool(|| -> crate::Result<DynamicConstraint> {
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK_QUOTIENT").is_some()
            && !prepared_has_giant_repeat
        {
            let quotient_started_at = Instant::now();
            let plan = plan_synthetic_tokenizer(&prepared_grammar, vocab);
            let planned_ms = elapsed_ms(quotient_started_at);
            let pair_started_at = Instant::now();
            let pair = plan.as_ref().and_then(|plan| {
                prepare_structural_tokenizer_pair(
                    &prepared_grammar,
                    plan,
                    vocab,
                    Some(false),
                    true,
                )
            });
            let pair_ms = elapsed_ms(pair_started_at);
            if let Some((synthesized, full, certified)) = pair {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_quotient_probe] selected=true full_states={} quotient_states={} map_states={} plan_ms={:.3} pair_ms={:.3} total_ms={:.3}",
                    full.num_states(),
                    synthesized.num_states(),
                    certified.full_to_synthesized.len(),
                    planned_ms,
                    pair_ms,
                    elapsed_ms(quotient_started_at),
                );
                let requested_states = std::env::var(
                    "GLRMASK_PROFILE_DYNAMIC_MASK_QUOTIENT_STATES",
                )
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|state| state.trim().parse::<u32>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
                let mut requested_quotients = std::collections::BTreeSet::new();
                for &full_state in &requested_states {
                    let Some(&quotient_state) =
                        certified.full_to_synthesized.get(full_state as usize)
                    else {
                        continue;
                    };
                    requested_quotients.insert(quotient_state);
                    eprintln!(
                        "[glrmask/profile][dynamic_mask_quotient_state] full={} quotient={} transitions={} matched={} futures={} loop_bytes={}",
                        full_state,
                        quotient_state,
                        synthesized.transitions_from(quotient_state).count(),
                        synthesized.matched_terminals_iter(quotient_state).count(),
                        synthesized.possible_future_terminals_iter(quotient_state).count(),
                        synthesized.self_loop_bytes(quotient_state).len(),
                    );
                }
                if !requested_states.is_empty() {
                    eprintln!(
                        "[glrmask/profile][dynamic_mask_quotient_requested] full_states={} unique_quotient_states={} quotient_states={:?}",
                        requested_states.len(),
                        requested_quotients.len(),
                        requested_quotients,
                    );
                }
            } else {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_quotient_probe] selected=false planned={} plan_ms={:.3} pair_ms={:.3} total_ms={:.3}",
                    plan.is_some(),
                    planned_ms,
                    pair_ms,
                    elapsed_ms(quotient_started_at),
                );
            }
        }
        let analysis_started_at = profile.then(Instant::now);
        // Move a complete direct automaton out of the grammar instead of
        // cloning its 20k-state graph into AnalyzedGrammar and cloning it again
        // into the runtime artifact. Generic grammars still use full analysis.
        let direct_regular_automaton = prepared_grammar.direct_regular_automaton.take();
        let analyzed_grammar = if direct_regular_automaton.is_none() {
            let analyzed = AnalyzedGrammar::from_grammar_def(&prepared_grammar);
            if let Err(message) = analyzed.check_table_build_normal_form() {
                panic!("[glrmask] grammar precondition violations:\n{}", message);
            }
            Some(analyzed)
        } else {
            None
        };
        let num_terminals = prepared_grammar.num_terminals();
        let terminal_display_names = (0..num_terminals)
            .map(|terminal| prepared_grammar.terminal_display_name(terminal))
            .collect::<Vec<_>>();
        let direct_state_count = direct_regular_automaton
            .as_ref()
            .map(|automaton| automaton.states.len());
        let analysis_ms = analysis_started_at.map_or(0.0, elapsed_ms);

        let (tokenizer_result, ((table, table_ms), (dynamic_mask_vocab, dynamic_vocab_ms))) = rayon::join(
            || -> crate::Result<((Tokenizer, Option<(Tokenizer, Vec<u32>)>), f64)> {
                let started_at = Instant::now();
                let force_quotient_over_virtual = std::env::var_os(
                    "GLRMASK_EXPERIMENT_FORCE_DYNAMIC_MASK_TOKEN_QUOTIENT",
                )
                .is_some();
                let quotient_enabled = std::env::var_os("GLRMASK_DYNAMIC_MASK_TOKEN_QUOTIENT")
                    .is_some()
                    && (!prepared_has_giant_repeat || force_quotient_over_virtual);
                // Experiment only: force the certified full->mask structural
                // quotient even for grammars that would normally select a
                // virtual residual full lexer. Commit still uses the full
                // tokenizer produced by the structural pair; masking uses the
                // certified synthesized tokenizer.
                let virtual_tokenizer = if force_quotient_over_virtual {
                    None
                } else {
                    build_dynamic_virtual_tokenizer(&prepared_grammar)?
                };
                let quotient_pair = (virtual_tokenizer.is_none() && quotient_enabled)
                    .then(|| plan_synthetic_tokenizer(&prepared_grammar, vocab))
                    .flatten()
                    .and_then(|plan| {
                        prepare_structural_tokenizer_pair(
                            &prepared_grammar,
                            &plan,
                            vocab,
                            Some(false),
                            true,
                        )
                    });
                let has_virtual_runtime = virtual_tokenizer.is_some();
                let (mut tokenizer, mask_tokenizer_quotient) = if let Some((
                    synthesized,
                    full,
                    certified,
                )) = quotient_pair
                {
                    (
                        full.finish(),
                        Some((synthesized, certified.full_to_synthesized)),
                    )
                } else {
                    let mut tokenizer = match virtual_tokenizer {
                        Some(tokenizer) => tokenizer,
                        None => build_dynamic_tokenizer(&prepared_grammar)?,
                    };
                    tokenizer.isolate_start_state_and_drain_nullable_terminals();
                    (tokenizer, None)
                };
                // A virtual tokenizer's exact state space is intentionally
                // larger than its materialized DFA state array. The ordinary
                // subset-construction helper is defined only over that
                // materialized domain, so running it here would discard (or
                // mis-handle) the arithmetic residual component. Keep the
                // symbolic runtime authoritative and determinize only fully
                // materialized tokenizers.
                if !has_virtual_runtime && tokenizer.has_epsilon_transitions() {
                    let source_states = tokenizer.num_states();
                    let source_transitions = tokenizer.transition_count();
                    let source_state_limit = std::env::var(
                        "GLRMASK_DYNAMIC_LEXER_MAX_SOURCE_STATES",
                    )
                        .ok()
                        .and_then(|value| value.trim().parse::<u32>().ok())
                        .filter(|&value| value > 0)
                        .unwrap_or(512);
                    if source_states <= source_state_limit {
                        let transition_limit = source_transitions.saturating_mul(6).max(1);
                        let state_limit = std::env::var("GLRMASK_DYNAMIC_LEXER_MAX_STATES")
                            .ok()
                            .and_then(|value| value.trim().parse::<usize>().ok())
                            .filter(|&value| value > 0)
                            .unwrap_or(8_192);
                        if let Some(determinized) =
                            tokenizer.try_full_determinization(state_limit, transition_limit)
                        {
                            tokenizer = determinized.tokenizer;
                        }
                    }
                    if profile {
                        eprintln!(
                            "[glrmask/profile][dynamic_lexer_determinization] source_states={} source_transitions={} source_state_limit={} attempted={} final_states={} final_transitions={}",
                            source_states,
                            source_transitions,
                            source_state_limit,
                            source_states <= source_state_limit,
                            tokenizer.num_states(),
                            tokenizer.transition_count(),
                        );
                    }
                }
                Ok(((tokenizer, mask_tokenizer_quotient), elapsed_ms(started_at)))
            },
            || rayon::join(
                || {
                    let started_at = Instant::now();
                    let table = if let Some(state_count) = direct_state_count {
                    GLRTable::direct_regular_runtime_stub(
                        state_count.saturating_add(1) as u32,
                        num_terminals,
                    )
                } else {
                    GLRTable::build_with_default_construction(
                        analyzed_grammar.as_ref().expect("generic grammar was analyzed"),
                        default_table_construction,
                    )
                };
                    (table, elapsed_ms(started_at))
                },
                || {
                    let started_at = Instant::now();
                    let dynamic_vocab = if finalize_runtime {
                        crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab)
                    } else {
                        crate::runtime::DynamicMaskVocab::default()
                    };
                    (dynamic_vocab, elapsed_ms(started_at))
                },
            ),
        );
        let ((tokenizer, mask_tokenizer_quotient), tokenizer_ms) = tokenizer_result?;

        let finalize_started_at = profile.then(Instant::now);
        // Build unfinalized so a mask-only finite-token quotient can be
        // attached before dynamic runtime caches/projections are constructed.
        // The exact full tokenizer above remains authoritative for commit.
        let mut constraint = DynamicConstraint::from_parts_with_dynamic_vocab_unfinalized(
            table,
            terminal_display_names,
            tokenizer,
            direct_regular_automaton,
            prepared_grammar.ignore_terminal,
            collect_special_token_terminals(&prepared_grammar),
            vocab,
            dynamic_mask_vocab,
        );
        if let Some((mask_tokenizer, full_to_mask_state)) = mask_tokenizer_quotient {
            constraint
                .inner
                .dynamic_mask_vocab
                .set_mask_tokenizer_quotient(mask_tokenizer, full_to_mask_state);
        }
        if finalize_runtime {
            constraint.inner.rebuild_dynamic_runtime_caches();
        }
        constraint
            .inner
            .table
            .set_embedded_start_nullable(start_nullable);
        constraint.set_composition_grammar(prepared_grammar);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][dynamic_compile] finalize_runtime={} prepare_ms={:.3} analysis_ms={:.3} tokenizer_ms={:.3} table_ms={:.3} dynamic_vocab_ms={:.3} finalize_ms={:.3} parallel_core_wall_ms={:.3} total_ms={:.3}",
                finalize_runtime,
                prepare_ms,
                analysis_ms,
                tokenizer_ms,
                table_ms,
                dynamic_vocab_ms,
                finalize_started_at.map_or(0.0, elapsed_ms),
                tokenizer_ms.max(table_ms.max(dynamic_vocab_ms)),
                elapsed_ms(total_started_at),
            );
        }
        Ok(constraint)
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

pub(crate) fn compile_owned_with_table_construction_and_protected_shift_terminal_names(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
    protected_shift_terminal_names: Vec<String>,
) -> Constraint {
    let start_nullable = grammar.start_is_nullable();
    let prepared_grammar = prepare_grammar(grammar);
    let protected_shift_terminals = protected_shift_terminal_names
        .iter()
        .map(|name| {
            prepared_grammar
                .terminal_names
                .iter()
                .find_map(|(&terminal, candidate)| (candidate == name).then_some(terminal))
                .unwrap_or_else(|| {
                    panic!(
                        "protected shift terminal {name:?} disappeared during grammar preparation"
                    )
                })
        })
        .collect::<Vec<_>>();
    let (mut constraint, profile) = compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        default_table_construction,
        None,
        Some(Arc::new(protected_shift_terminals)),
    );
    constraint.table.set_embedded_start_nullable(start_nullable);
    if compile_profile_summary_enabled() || compile_top_profile_enabled() {
        emit_compile_profile_summary(None, None, &profile);
    }
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
        None,
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
        None,
    );
    constraint
        .table
        .set_embedded_start_nullable(start_nullable);
    profile.prepare_ms = prepare_ms;
    profile.total_ms = elapsed_ms(total_started_at);
    (constraint, profile)
}
