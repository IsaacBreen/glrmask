use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use once_cell::sync::Lazy;

use crate::Vocab;
use crate::automata::lexer::compile::{
    build_regex,
    build_regex_with_profile_labels,
    factor_regex_expr,
};
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::constraint_possible_matches as cpm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{GLRTable, GlrTableConstruction};
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::dwa_build_graph::DwaBuildTopology;
use crate::compiler::stages::id_map_and_terminal_dwa::{
    TerminalDwaLeaves,
    build_terminal_dwa_leaves_with_precomputed_global_max_length,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    SharedClassifyCache,
    classify_terminal_path_lengths,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::{
    compute_ever_allowed_follows,
    compute_terminal_coloring,
    ignore_transparent_disallowed_follows,
};
use crate::compiler::stages::mapped_artifact::{
    MappedArtifact,
    WeightRefs,
    count_interned_ranges_for_weights,
};
use crate::compiler::stages::parser_dwa::{
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    finalize_parser_dwa_from_nwa,
};
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals_profiled;
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    split_commit_template_dfas,
};
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::{GrammarDef, Terminal};
use crate::runtime::Constraint;

enum PipelineTerminalDwaBuild {
    Merged(MappedArtifact<TerminalAutomaton>, TerminalDwaPhaseProfile),
    Leaves(TerminalDwaLeaves),
}

/// The production entry points always use `LegacyGlobalTerminal`. Other
/// variants are code-level layouts for tests and future source changes.
#[derive(Debug, Clone, Copy)]
enum PipelineDwaBuildLayout {
    LegacyGlobalTerminal,
    Graph(DwaBuildTopology),
}

impl PipelineDwaBuildLayout {
    fn graph_topology(self) -> Option<DwaBuildTopology> {
        match self {
            Self::LegacyGlobalTerminal => None,
            Self::Graph(topology) => Some(topology),
        }
    }
}

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

fn commit_template_dfas_enabled() -> bool {
    env_flag_enabled("GLRMASK_ENABLE_COMMIT_TEMPLATE_DFAS")
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
            DwaPossibleMatchesMode::TerminalReconcile
        }
    }
}

pub(crate) fn compile_profile_summary_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
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

    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        return std::thread::available_parallelism()
            .ok()
            .map(|parallelism| parallelism.get().min(10))
            .filter(|&value| value > 1);
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
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
    pub(crate) possible_matches_collect_ms: f64,
    pub(crate) possible_matches_materialize_ms: f64,
    pub(crate) shared_id_reconcile_ms: f64,
    pub(crate) possible_matches_pipeline_ms: f64,
    pub(crate) terminal_dwa_interned_ranges_before_pm_reconcile: usize,
    pub(crate) possible_matches_interned_ranges_before_pm_reconcile: usize,
    pub(crate) terminal_pm_joint_interned_ranges_before_reconcile: usize,
    pub(crate) terminal_pm_joint_interned_ranges: usize,
    pub(crate) internal_token_bytes_ms: f64,
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
    if !compile_profile_summary_enabled() {
        return;
    }

    let source = source_kind.unwrap_or("grammar");
    let import_fragment = import_ms
        .map(|ms| format!(" import_ms={ms:.3}"))
        .unwrap_or_default();

    eprintln!(
        "[glrmask/profile][compile] source={}{} prepare_ms={:.3} tokenizer_build_ms={:.3} tokenizer_final_states={} tokenizer_final_transitions={} analyze_grammar_ms={:.3} glr_table_ms={:.3} terminal_coloring_ms={:.3} disallowed_follows_ms={:.3} analysis_wall_ms={:.3} classify_ms={:.3} id_map_ms={:.3} terminal_dwa_ms={:.3} split_terminal_dwa_total_ms={:.3} global_merge_ms={:.3} templates_ms={:.3} compact_ms={:.3} possible_matches_collect_ms={:.3} possible_matches_materialize_ms={:.3} shared_id_reconcile_ms={:.3} possible_matches_pipeline_ms={:.3} terminal_dwa_interned_ranges_before_pm_reconcile={} possible_matches_interned_ranges_before_pm_reconcile={} terminal_pm_joint_interned_ranges_before_reconcile={} terminal_pm_joint_interned_ranges={} internal_token_bytes_ms={:.3} parser_dwa_ms={:.3} parser_dwa_interned_ranges={} possible_matches_interned_ranges={} parser_pm_joint_interned_ranges={} finalize_ms={:.3} compile_ms={:.3} total_ms={:.3}",
        source,
        import_fragment,
        profile.prepare_ms,
        profile.tokenizer_build_ms,
        profile.tokenizer_final_states,
        profile.tokenizer_final_transitions,
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
        profile.possible_matches_collect_ms,
        profile.possible_matches_materialize_ms,
        profile.shared_id_reconcile_ms,
        profile.possible_matches_pipeline_ms,
        profile.terminal_dwa_interned_ranges_before_pm_reconcile,
        profile.possible_matches_interned_ranges_before_pm_reconcile,
        profile.terminal_pm_joint_interned_ranges_before_reconcile,
        profile.terminal_pm_joint_interned_ranges,
        profile.internal_token_bytes_ms,
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
    let num_terminals = grammar.num_terminals as usize;
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
    if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some() {
        eprintln!(
            "[glrmask/profile][tokenizer] terminals={}",
            grammar.terminals.len()
        );
        for (index, expr) in exprs.iter().enumerate() {
            let started_at = Instant::now();
            let regex = build_regex(std::slice::from_ref(expr));
            let elapsed = elapsed_ms(started_at);
            let name = grammar.terminal_display_name(index as u32);
            eprintln!(
                "[glrmask/profile][tokenizer] terminal id={} name={:?} final_states={} final_transitions={} alone_ms={:.3}",
                index,
                name,
                regex.num_states(),
                regex.num_transitions(),
                elapsed
            );
        }
    }
    build_tokenizer_from_exprs(&exprs, Some(&terminal_labels))
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

fn terminal_expr(terminal: &Terminal) -> Expr {
    match terminal {
        Terminal::Literal { bytes, .. } => Expr::U8Seq(bytes.clone()),
        Terminal::Pattern { pattern, utf8, .. } => parse_regex(pattern, *utf8),
        Terminal::Expr { expr, .. } => expr.clone(),
    }
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

fn compile_prepared_with_profile(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
) -> (Constraint, CompilePhaseProfile) {
    compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        GlrTableConstruction::ExperimentalCoreMerged,
    )
}

fn compile_prepared_with_profile_and_table_construction(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> (Constraint, CompilePhaseProfile) {
    compile_prepared_with_profile_and_table_construction_and_dwa_build_layout(
        prepared_grammar,
        vocab,
        default_table_construction,
        PipelineDwaBuildLayout::LegacyGlobalTerminal,
    )
}

fn compile_prepared_with_profile_and_table_construction_and_dwa_build_layout(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
    dwa_build_layout: PipelineDwaBuildLayout,
) -> (Constraint, CompilePhaseProfile) {
    let interner_cleanup = crate::ds::weight::defer_weight_interner_cleanup();
    let result = run_with_compile_thread_pool(|| {
        let compile_started_at = Instant::now();
        let mut profile = CompilePhaseProfile::default();

        let analysis_started_at = Instant::now();
        let (
            (tokenizer, tokenizer_build_ms),
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
                let mut tokenizer = build_tokenizer(&prepared_grammar);
                let tokenizer_construct_ms = elapsed_ms(tok_started);
                let isolate_started = Instant::now();
                tokenizer.isolate_start_state_and_drain_nullable_terminals();
                if std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some() {
                    eprintln!(
                        "[glrmask/profile][tokenizer] construction_vs_isolation construct_ms={:.3} isolate_ms={:.3} total_ms={:.3}",
                        tokenizer_construct_ms,
                        elapsed_ms(isolate_started),
                        elapsed_ms(tok_started),
                    );
                }
                (tokenizer, elapsed_ms(tok_started))
            },
            || {
                let analyze_grammar_started_at = Instant::now();
                let analyzed_grammar = AnalyzedGrammar::from_grammar_def(&prepared_grammar);
                let analyze_grammar_ms = elapsed_ms(analyze_grammar_started_at);

                if let Err(message) = analyzed_grammar.check_table_build_normal_form() {
                    panic!("[glrmask] grammar precondition violations:\n{}", message);
                }

                let table_started_at = Instant::now();
                let table = GLRTable::build_with_default_construction(
                    &analyzed_grammar,
                    default_table_construction,
                );
                let glr_table_ms = elapsed_ms(table_started_at);
                if std::env::var_os("GLRMASK_STOP_AFTER_GLR_TABLE").is_some() {
                    panic!("[glrmask] stopped after GLR table build by GLRMASK_STOP_AFTER_GLR_TABLE");
                }

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
        profile.tokenizer_final_states = tokenizer.num_states() as usize;
        profile.tokenizer_final_transitions = tokenizer.transition_count();
        profile.analyze_grammar_ms = analyze_grammar_ms;
        profile.glr_table_ms = glr_table_ms;
        profile.terminal_coloring_ms = terminal_coloring_ms;
        profile.disallowed_follows_ms = disallowed_follows_ms;
        profile.analysis_wall_ms = elapsed_ms(analysis_started_at);

        let token_path_disallowed_follows = ignore_transparent_disallowed_follows(
            &disallowed_follows,
            prepared_grammar.ignore_terminal,
        );
        let disallowed_follows_for_classification = &token_path_disallowed_follows;
        let shared_classify_cache = SharedClassifyCache::new();
        let classify_started_at = Instant::now();
        let _terminal_path_lengths = classify_terminal_path_lengths(
            &tokenizer,
            vocab,
            disallowed_follows_for_classification,
            analyzed_grammar.num_terminals,
            Some(&shared_classify_cache),
        );
        profile.classify_ms = elapsed_ms(classify_started_at);

        let flat_trans_started_at = Instant::now();
        let flat_trans: Arc<[u32]> = Arc::from(
            crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(
                &tokenizer,
            ),
        );
        let flat_trans_ms = elapsed_ms(flat_trans_started_at);

        let global_max_length_started_at = Instant::now();
        let global_max_length_state_map =
            crate::compiler::stages::id_map_and_terminal_dwa::build_global_max_length_state_map(
                &tokenizer,
                vocab,
                &flat_trans,
            );
        let global_max_length_ms = elapsed_ms(global_max_length_started_at);

        let dwa_build_topology = dwa_build_layout.graph_topology();
        let use_dwa_build_graph = dwa_build_topology.is_some();
        let (
            (terminal_build, cpm_result),
            (templates, template_dfas_by_terminal, templates_ms),
        ) = rayon::join(
            || {
                rayon::join(
                    || {
                        if use_dwa_build_graph {
                            PipelineTerminalDwaBuild::Leaves(
                                build_terminal_dwa_leaves_with_precomputed_global_max_length(
                                    &tokenizer,
                                    vocab,
                                    &terminal_coloring,
                                    true,
                                    prepared_grammar.ignore_terminal,
                                    &analyzed_grammar,
                                    &disallowed_follows,
                                    Arc::clone(&flat_trans),
                                    &global_max_length_state_map,
                                    Some(&shared_classify_cache),
                                ),
                            )
                        } else {
                            let (terminal_dwa, terminal_phase_profile) =
                                crate::compiler::stages::id_map_and_terminal_dwa::build_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                                    &tokenizer,
                                    vocab,
                                    &terminal_coloring,
                                    true,
                                    prepared_grammar.ignore_terminal,
                                    &analyzed_grammar,
                                    &disallowed_follows,
                                    Arc::clone(&flat_trans),
                                    &global_max_length_state_map,
                                    Some(&shared_classify_cache),
                                );
                            PipelineTerminalDwaBuild::Merged(terminal_dwa, terminal_phase_profile)
                        }
                    },
                    || cpm::compute_constraint_possible_matches_for_vocab(
                        &tokenizer,
                        vocab,
                        cpm::ConstraintPossibleMatchesConfig,
                    ),
                )
            },
            || {
                let templates_started_at = Instant::now();
                let (characterizations, characterization_profile) =
                    characterize_terminals_profiled(&table, &analyzed_grammar);
                let (mut templates, template_profile) =
                    Templates::from_characterizations_profiled(&characterizations);
                if let Some(ignore_terminal) = prepared_grammar.ignore_terminal {
                    templates.install_identity_template(ignore_terminal);
                }
                let mut template_dfas_by_terminal =
                    vec![None; analyzed_grammar.num_terminals as usize];
                let commit_template_dfas_enabled = commit_template_dfas_enabled();
                let mut commit_template_dfas_built = 0usize;
                if commit_template_dfas_enabled {
                    for (&terminal, dfa) in &templates.by_terminal {
                        if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
                            let commit_dfa =
                                specialize_template_dfa_defaults_for_commit_split_input(dfa);
                            let split_commit_dfas = split_commit_template_dfas(&commit_dfa);
                            *slot = Some(Arc::new(split_commit_dfas));
                            commit_template_dfas_built += 1;
                        }
                    }
                }
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][templates] terminals={} action_signature_classes={} action_quotient_hits={} max_action_signature_multiplicity={} characterization_signature_ms={:.3} characterization_ms={:.3} characterization_fanout_ms={:.3} characterization_validation_ms={:.3} characterization_total_ms={:.3} characterization_quotient_disabled={} unique_characterizations={} compiled_characterizations={} template_quotient_hits={} max_characterization_multiplicity={} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} template_fanout_ms={:.3} template_validation_ms={:.3} template_total_ms={:.3} template_wall_ms={:.3} template_minimize_skipped={} avg_nfa_states={:.2} avg_nfa_transitions={:.2} avg_premin_dfa_states={:.2} avg_premin_dfa_transitions={:.2} avg_dfa_states={:.2} avg_dfa_transitions={:.2} max_dfa_states={} max_dfa_transitions={} commit_template_dfas_enabled={} commit_template_dfas_built={}",
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
                    );
                }
                (
                    templates,
                    template_dfas_by_terminal,
                    elapsed_ms(templates_started_at),
                )
            },
        );
        let mut terminal_phase_profile = match &terminal_build {
            PipelineTerminalDwaBuild::Merged(_, profile) => *profile,
            PipelineTerminalDwaBuild::Leaves(leaves) => leaves.profile,
        };
        terminal_phase_profile.terminal_dwa_ms += flat_trans_ms;
        terminal_phase_profile.id_map_ms += global_max_length_ms;
        profile.templates_ms = templates_ms;
        profile.id_map_ms = terminal_phase_profile.id_map_ms;
        profile.terminal_dwa_ms = terminal_phase_profile.terminal_dwa_ms;
        profile.compact_ms = terminal_phase_profile.compact_ms;
        profile.split_terminal_dwa_total_ms = terminal_phase_profile.split_terminal_dwa_total_ms;
        profile.global_merge_ms = terminal_phase_profile.global_merge_ms;

        let (
            mut parser_dwa,
            mut possible_matches,
            internal_ids,
            parser_dwa_ms,
            shared_id_reconcile_ms,
            cpm_profile,
            terminal_dwa_interned_ranges_before_pm_reconcile,
            possible_matches_interned_ranges_before_pm_reconcile,
            terminal_pm_joint_interned_ranges_before_reconcile,
            terminal_pm_joint_interned_ranges,
        ) = if let Some(dwa_build_topology) = dwa_build_topology {
            let PipelineTerminalDwaBuild::Leaves(leaves) = terminal_build else {
                unreachable!("DWA build graph requested but terminal stage returned a global DWA");
            };
            let mut possible_matches = cpm_result.mapped_possible_matches;
            let cpm_profile = cpm_result.profile;
            let dwa_pm_mode = dwa_possible_matches_mode();
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

            let parser_points_started_at = Instant::now();
            let parser_nwa = dwa_build_topology.build_parser_point(leaves, &analyzed_grammar, &templates);
            let parser_points_ms = elapsed_ms(parser_points_started_at);

            // The planned path intentionally does not materialize one global
            // terminal DWA. These profiling fields therefore have no terminal
            // DWA range count; possible-match counts remain meaningful.
            let terminal_dwa_interned_ranges_before_pm_reconcile = 0;
            let possible_matches_interned_ranges_before_pm_reconcile =
                interned_range_count_for_artifact(possible_matches.artifact_mut());
            let terminal_pm_joint_interned_ranges_before_reconcile = 0;

            // Reconcile at the parser-NWA point. This is the earliest common
            // representation for arbitrary terminal/parser merge trees and is
            // also before parser DEFAULT_LABEL/fallback normalization.
            let shared_id_reconcile_started_at = Instant::now();
            let mut parser_nwa_pm_pair = MappedArtifact::from((parser_nwa, possible_matches));
            shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
            if dwa_pm_mode.does_terminal_compact() {
                let compact_started_at = Instant::now();
                parser_nwa_pm_pair.compact_dimensions();
                profile.compact_ms += elapsed_ms(compact_started_at);
            }
            let ((parser_nwa, possible_matches_artifact), mut internal_ids) =
                parser_nwa_pm_pair.into_parts();
            let mut possible_matches =
                MappedArtifact::new(possible_matches_artifact, internal_ids.clone());
            let terminal_pm_joint_interned_ranges = 0;

            let parser_dwa_started_at = Instant::now();
            let parser_dwa_artifact = finalize_parser_dwa_from_nwa(&table, &parser_nwa);
            let parser_dwa_ms = parser_points_ms + elapsed_ms(parser_dwa_started_at);
            let mut parser_dwa = MappedArtifact::new(parser_dwa_artifact, internal_ids.clone());

            if dwa_pm_mode.does_parser_compact() {
                let shared_id_reconcile_started_at = Instant::now();
                let mut parser_pm_pair = MappedArtifact::from((parser_dwa, possible_matches));
                shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
                let compact_started_at = Instant::now();
                parser_pm_pair.compact_dimensions();
                profile.compact_ms += elapsed_ms(compact_started_at);
                let ((parser_dwa_artifact, possible_matches_artifact), compacted_ids) =
                    parser_pm_pair.into_parts();
                parser_dwa = MappedArtifact::new(parser_dwa_artifact, compacted_ids.clone());
                possible_matches = MappedArtifact::new(possible_matches_artifact, compacted_ids.clone());
                internal_ids = compacted_ids;
            }

            (
                parser_dwa,
                possible_matches,
                internal_ids,
                parser_dwa_ms,
                shared_id_reconcile_ms,
                cpm_profile,
                terminal_dwa_interned_ranges_before_pm_reconcile,
                possible_matches_interned_ranges_before_pm_reconcile,
                terminal_pm_joint_interned_ranges_before_reconcile,
                terminal_pm_joint_interned_ranges,
            )
        } else {
            let PipelineTerminalDwaBuild::Merged(mut terminal_dwa, _terminal_profile) = terminal_build else {
                unreachable!("legacy terminal path received branch leaves");
            };
            let mut possible_matches = cpm_result.mapped_possible_matches;
            let cpm_profile = cpm_result.profile;
            let dwa_pm_mode = dwa_possible_matches_mode();

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
            let terminal_dwa_interned_ranges_before_pm_reconcile =
                interned_range_count_for_artifact(terminal_dwa.artifact_mut());
            let possible_matches_interned_ranges_before_pm_reconcile =
                interned_range_count_for_artifact(possible_matches.artifact_mut());
            let terminal_pm_joint_interned_ranges_before_reconcile =
                joint_interned_range_count_for_artifacts(terminal_dwa.artifact_mut(), possible_matches.artifact_mut());
            let mut internal_ids = terminal_dwa.id_map().clone();
            let (mut parser_dwa, parser_dwa_ms) = if dwa_pm_mode.does_terminal_compact() {
                let shared_id_reconcile_started_at = Instant::now();
                let mut terminal_pm_pair = MappedArtifact::from((terminal_dwa, possible_matches));
                shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
                if dwa_pm_mode.does_parser_compact() {
                    let compact_plan_started_at = Instant::now();
                    let terminal_compaction_plan = terminal_pm_pair.plan_dimensions_compaction(true, true);
                    profile.compact_ms += elapsed_ms(compact_plan_started_at);
                    let ((terminal_dwa_artifact, possible_matches_artifact), compacted_ids) =
                        terminal_pm_pair.into_parts();
                    terminal_dwa = MappedArtifact::new(terminal_dwa_artifact, compacted_ids.clone());
                    possible_matches = MappedArtifact::new(possible_matches_artifact, compacted_ids.clone());
                    let terminal_apply_started_at = Instant::now();
                    terminal_dwa.apply_compaction_plan(&terminal_compaction_plan);
                    profile.compact_ms += elapsed_ms(terminal_apply_started_at);
                    internal_ids = terminal_dwa.id_map().clone();

                    let ((parser_dwa, parser_dwa_ms), possible_matches_compact_ms) = rayon::join(
                        || {
                            let parser_dwa_started_at = Instant::now();
                            let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                                &table,
                                &analyzed_grammar,
                                terminal_dwa.artifact(),
                                templates,
                                vocab,
                                &internal_ids,
                            );
                            (parser_dwa, elapsed_ms(parser_dwa_started_at))
                        },
                        || {
                            let possible_matches_apply_started_at = Instant::now();
                            possible_matches.apply_compaction_plan(&terminal_compaction_plan);
                            elapsed_ms(possible_matches_apply_started_at)
                        },
                    );
                    if possible_matches_compact_ms > parser_dwa_ms {
                        profile.compact_ms += possible_matches_compact_ms - parser_dwa_ms;
                    }
                    (MappedArtifact::new(parser_dwa, internal_ids.clone()), parser_dwa_ms)
                } else {
                    let pre_compact_ids = terminal_pm_pair.id_map().clone();
                    let ((terminal_compaction_plan, terminal_compaction_plan_ms), (parser_dwa, parser_dwa_ms)) =
                        rayon::join(
                            || {
                                let compact_started_at = Instant::now();
                                let plan = terminal_pm_pair.plan_dimensions_compaction(true, true);
                                (plan, elapsed_ms(compact_started_at))
                            },
                            || {
                                let parser_dwa_started_at = Instant::now();
                                let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                                    &table,
                                    &analyzed_grammar,
                                    &terminal_pm_pair.artifact().0,
                                    templates,
                                    vocab,
                                    &pre_compact_ids,
                                );
                                (parser_dwa, elapsed_ms(parser_dwa_started_at))
                            },
                        );

                    let compact_apply_started_at = Instant::now();
                    terminal_pm_pair.apply_compaction_plan(&terminal_compaction_plan);
                    let mut parser_dwa = MappedArtifact::new(parser_dwa, pre_compact_ids);
                    parser_dwa.apply_compaction_plan(&terminal_compaction_plan);
                    profile.compact_ms += terminal_compaction_plan_ms + elapsed_ms(compact_apply_started_at);

                    let ((terminal_dwa_artifact, possible_matches_artifact), compacted_ids) =
                        terminal_pm_pair.into_parts();
                    terminal_dwa = MappedArtifact::new(terminal_dwa_artifact, compacted_ids.clone());
                    possible_matches = MappedArtifact::new(possible_matches_artifact, compacted_ids.clone());
                    internal_ids = compacted_ids.clone();
                    (parser_dwa, parser_dwa_ms)
                }
            } else {
                if dwa_pm_mode.does_terminal_reconcile() {
                    let shared_id_reconcile_started_at = Instant::now();
                    internal_ids = terminal_dwa.reconcile_with(&mut possible_matches);
                    shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
                }
                let parser_dwa_started_at = Instant::now();
                let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                    &table,
                    &analyzed_grammar,
                    terminal_dwa.artifact(),
                    templates,
                    vocab,
                    &internal_ids,
                );
                let parser_dwa_ms = elapsed_ms(parser_dwa_started_at);
                (MappedArtifact::new(parser_dwa, internal_ids.clone()), parser_dwa_ms)
            };

            let terminal_pm_joint_interned_ranges =
                joint_interned_range_count_for_artifacts(terminal_dwa.artifact_mut(), possible_matches.artifact_mut());

            if dwa_pm_mode.does_terminal_reconcile() {
                if dwa_pm_mode.does_parser_compact() {
                    let shared_id_reconcile_started_at = Instant::now();
                    let mut parser_pm_pair = MappedArtifact::from((parser_dwa, possible_matches));
                    shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
                    let compact_started_at = Instant::now();
                    parser_pm_pair.compact_dimensions();
                    profile.compact_ms += elapsed_ms(compact_started_at);
                    let ((parser_dwa_artifact, possible_matches_artifact), compacted_ids) =
                        parser_pm_pair.into_parts();
                    parser_dwa = MappedArtifact::new(parser_dwa_artifact, compacted_ids.clone());
                    possible_matches = MappedArtifact::new(possible_matches_artifact, compacted_ids.clone());
                    internal_ids = compacted_ids;
                }
            } else {
                let shared_id_reconcile_started_at = Instant::now();
                let mut parser_pm_pair = MappedArtifact::from((parser_dwa, possible_matches));
                shared_id_reconcile_ms += elapsed_ms(shared_id_reconcile_started_at);
                if dwa_pm_mode.does_parser_compact() {
                    let compact_started_at = Instant::now();
                    parser_pm_pair.compact_dimensions();
                    profile.compact_ms += elapsed_ms(compact_started_at);
                }
                let ((parser_dwa_artifact, possible_matches_artifact), reconciled_ids) =
                    parser_pm_pair.into_parts();
                parser_dwa = MappedArtifact::new(parser_dwa_artifact, reconciled_ids.clone());
                possible_matches = MappedArtifact::new(possible_matches_artifact, reconciled_ids.clone());
                internal_ids = reconciled_ids;
            }

        (
            parser_dwa,
            possible_matches,
            internal_ids,
            parser_dwa_ms,
            shared_id_reconcile_ms,
            cpm_profile,
            terminal_dwa_interned_ranges_before_pm_reconcile,
            possible_matches_interned_ranges_before_pm_reconcile,
            terminal_pm_joint_interned_ranges_before_reconcile,
            terminal_pm_joint_interned_ranges,
        )
        };

        let parser_dwa_interned_ranges = parser_dwa.artifact().stats().interned_ranges;
        let (possible_matches_interned_ranges, parser_pm_joint_interned_ranges) = {
            let (parser_dwa_artifact, _) = parser_dwa.parts_mut();
            let (possible_matches_artifact, _) = possible_matches.parts_mut();
            (
                interned_range_count_for_artifact(possible_matches_artifact),
                joint_interned_range_count_for_artifacts(parser_dwa_artifact, possible_matches_artifact),
            )
        };
        let parser_dwa = parser_dwa.into_artifact();

        let internal_token_bytes_started_at = Instant::now();
        let internal_token_bytes = cpm::build_internal_token_bytes_from_groups(
            vocab,
            &internal_ids.vocab_tokens.internal_to_originals,
        );
        let internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

        profile.parser_dwa_ms = parser_dwa_ms;
        profile.possible_matches_collect_ms = cpm_profile.possible_matches_collect_ms;
        profile.possible_matches_materialize_ms = cpm_profile.possible_match_vocab_ms;
        profile.shared_id_reconcile_ms = shared_id_reconcile_ms;
        profile.possible_matches_pipeline_ms =
            cpm_profile.possible_matches_collect_ms + cpm_profile.possible_match_vocab_ms + shared_id_reconcile_ms;
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
        let constraint = finalize_constraint(Constraint {
            parser_dwa,
            table,
            terminal_display_names: analyzed_grammar.terminal_display_names.clone(),
            tokenizer,
            ignore_terminal: prepared_grammar.ignore_terminal,
            dynamic_mask_vocab: Default::default(),
            possible_matches: possible_matches.into_artifact(),
            state_to_internal_tsid: internal_ids.tokenizer_states.original_to_internal.clone(),
            internal_tsid_to_states: internal_ids.tokenizer_states.internal_to_originals_vecs(),
            original_token_to_internal: internal_ids.vocab_tokens.original_to_internal.clone(),
            internal_token_to_tokens: internal_ids.vocab_tokens.internal_to_originals_vecs(),
            template_dfas_by_terminal,
            eos_token_id: vocab.eos_token_id,
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
            byte_group_sparse_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: rustc_hash::FxHashMap::default(),
            weight_token_buf_masks: rustc_hash::FxHashMap::default(),
            weight_token_sparse_buf_masks: rustc_hash::FxHashMap::default(),
            seed_terminal_dense: rustc_hash::FxHashMap::default(),
            seed_universe_dense: std::sync::Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Vec::new(),
            tokenizer_fast_transitions: Vec::new(),
            heavy_token_dense_masks: Vec::new(),
            heavy_token_indices: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: crate::runtime::FinalMaskMapping::default(),
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
    compile_prepared_with_profile(prepared_grammar, vocab).0
}

pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    compile_owned_with_table_construction(
        grammar,
        vocab,
        GlrTableConstruction::ExperimentalCoreMerged,
    )
}

pub(crate) fn compile_owned_with_table_construction(
    grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> Constraint {
    if compile_profile_summary_enabled() {
        let (constraint, profile) =
            compile_owned_profiled_with_table_construction(grammar, vocab, default_table_construction);
        emit_compile_profile_summary(None, None, &profile);
        return constraint;
    }

    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    compile_prepared_with_table_construction(prepared_grammar, vocab, default_table_construction)
}

pub(crate) fn compile_prepared_with_table_construction(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> Constraint {
    compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        default_table_construction,
    )
    .0
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
    let total_started_at = Instant::now();
    let prepare_started_at = Instant::now();
    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    let prepare_ms = elapsed_ms(prepare_started_at);

    let (constraint, mut profile) = compile_prepared_with_profile_and_table_construction(
        prepared_grammar,
        vocab,
        default_table_construction,
    );
    profile.prepare_ms = prepare_ms;
    profile.total_ms = elapsed_ms(total_started_at);
    (constraint, profile)
}

#[cfg(test)]
mod dwa_build_graph_tests {
    use super::*;
    use crate::import::ebnf::parse_ebnf;

    fn witness_vocab() -> Vocab {
        Vocab::new(
            vec![
                (0, b"[".to_vec()),
                (1, b"x".to_vec()),
                (2, b"]".to_vec()),
                (3, b"abc".to_vec()),
                (4, b"a".to_vec()),
                (5, b"bc".to_vec()),
                (6, b"[x]".to_vec()),
                (7, b"z".to_vec()),
                (8, b"[z]".to_vec()),
            ],
            None,
        )
    }

    fn compile_witness(layout: PipelineDwaBuildLayout) -> Constraint {
        let grammar = parse_ebnf(r#"start ::= "abc" | "[" "x" "]""#).unwrap();
        let prepared = prepare_grammar_transforms_only(grammar);
        compile_prepared_with_profile_and_table_construction_and_dwa_build_layout(
            prepared,
            &witness_vocab(),
            GlrTableConstruction::ExperimentalCoreMerged,
            layout,
        )
        .0
    }

    fn accepts_tokens(constraint: &Constraint, tokens: &[u32]) -> bool {
        let mut state = constraint.start();
        tokens.iter().all(|&token| state.commit_token(token).is_ok()) && state.is_finished()
    }

    fn accepts_bytes(constraint: &Constraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_finished()
    }

    fn mask_after(constraint: &Constraint, tokens: &[u32]) -> Option<Vec<u32>> {
        let mut state = constraint.start();
        for &token in tokens {
            state.commit_token(token).ok()?;
        }
        Some(state.mask())
    }

    #[test]
    fn static_dwa_build_topologies_preserve_terminal_and_parser_semantics() {
        let legacy = compile_witness(PipelineDwaBuildLayout::LegacyGlobalTerminal);
        let prefixes: &[&[u32]] = &[&[], &[0], &[0, 1], &[4], &[4, 5], &[6]];
        let layouts = [
            (
                "global_terminal",
                PipelineDwaBuildLayout::Graph(DwaBuildTopology::GlobalTerminal),
            ),
            (
                "partition_parser",
                PipelineDwaBuildLayout::Graph(DwaBuildTopology::PerPartitionParser),
            ),
            (
                "branch_parser",
                PipelineDwaBuildLayout::Graph(DwaBuildTopology::PerBranchParser),
            ),
            (
                "left_deep_terminal",
                PipelineDwaBuildLayout::Graph(DwaBuildTopology::LeftDeepTerminal),
            ),
        ];

        for (name, layout) in layouts {
            let constraint = compile_witness(layout);
            for &prefix in prefixes {
                assert_eq!(
                    mask_after(&constraint, prefix),
                    mask_after(&legacy, prefix),
                    "{name}: mask diverged after {prefix:?}",
                );
            }
            assert!(accepts_tokens(&constraint, &[3]), "{name}: abc token");
            assert!(accepts_tokens(&constraint, &[4, 5]), "{name}: split abc");
            assert!(accepts_tokens(&constraint, &[0, 1, 2]), "{name}: split bracket form");
            assert!(accepts_tokens(&constraint, &[6]), "{name}: cross-terminal token");
            assert!(!accepts_tokens(&constraint, &[7]), "{name}: z must reject");
            assert!(!accepts_tokens(&constraint, &[8]), "{name}: [z] must reject");
            assert!(accepts_bytes(&constraint, b"abc"), "{name}: byte abc");
            assert!(accepts_bytes(&constraint, b"[x]"), "{name}: byte bracket form");
            assert!(!accepts_bytes(&constraint, b"abd"), "{name}: byte abd must reject");
            assert!(!accepts_bytes(&constraint, b"[z]"), "{name}: byte [z] must reject");
        }
    }
}
