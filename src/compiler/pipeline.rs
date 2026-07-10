use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
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
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::constraint_possible_matches as cpm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{GLRTable, GlrTableConstruction};
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    SharedClassifyCache,
    classify_terminal_path_lengths,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::{
    compute_ever_allowed_follows,
    compute_terminal_coloring,
    ignore_transparent_disallowed_follows,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    TerminalColoring,
    TerminalDwaFamilies,
    TerminalDwaPhaseProfile,
};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::mapped_artifact::{
    MappedArtifact,
    WeightRefs,
    count_interned_ranges_for_weights,
};
use crate::compiler::stages::parser_dwa::{
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    try_build_immediate_parser_dwa,
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
use crate::runtime::{Constraint, DynamicMaskVocab};

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

fn build_templates_for_compile(
    table: &GLRTable,
    analyzed_grammar: &AnalyzedGrammar,
    ignore_terminal: Option<u32>,
) -> (
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
) {
    let templates_started_at = Instant::now();
    let (characterizations, characterization_profile) =
        characterize_terminals_profiled(table, analyzed_grammar);
    let (mut templates, template_profile) =
        Templates::from_characterizations_profiled(&characterizations);
    if let Some(ignore_terminal) = ignore_terminal {
        templates.install_identity_template(ignore_terminal);
    }
    let mut template_dfas_by_terminal = vec![None; analyzed_grammar.num_terminals as usize];
    let commit_template_dfas_enabled = commit_template_dfas_enabled();
    let mut commit_template_dfas_built = 0usize;
    if commit_template_dfas_enabled {
        for (&terminal, dfa) in &templates.by_terminal {
            if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
                let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
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
}

#[derive(Clone)]
struct TokenizerDagLane {
    tokenizer: Arc<Tokenizer>,
    tokenizer_build_ms: f64,
    tokenizer_ready_ms: f64,
}

struct FlatGlobalDagLane {
    flat_trans: Arc<[u32]>,
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
    prebuilt_parser_dwa: Option<(MappedArtifact<DWA>, f64, f64, f64)>,
}

fn build_parser_dwa_for_terminal_family(
    family_name: &str,
    family: Option<&MappedArtifact<TerminalAutomaton>>,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    templates: Templates,
    vocab: &Vocab,
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

fn build_and_merge_parser_dwa_families(
    terminal_dwas: &TerminalDwaFamilies,
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    templates: Templates,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> MappedArtifact<DWA> {
    let l1_templates = templates.clone();
    let (l1_parser, l2p_parser) = rayon::join(
        || {
            build_parser_dwa_for_terminal_family(
                "l1",
                terminal_dwas.l1.as_ref(),
                table,
                grammar,
                l1_templates,
                vocab,
            )
        },
        || {
            build_parser_dwa_for_terminal_family(
                "l2p",
                terminal_dwas.l2p.as_ref(),
                table,
                grammar,
                templates,
                vocab,
            )
        },
    );
    let parser_dwas: Vec<MappedArtifact<DWA>> = l1_parser.into_iter().chain(l2p_parser).collect();
    crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_mapped_parser_dwas(
        parser_dwas,
        tokenizer.num_states() as usize,
        vocab.max_token_id(),
    )
}

#[derive(Clone, Copy)]
struct TerminalFamilyLayout {
    has_l1: bool,
    has_l2p: bool,
}

fn reconcile_terminal_dwa_families(
    families: TerminalDwaFamilies,
) -> (MappedArtifact<Vec<TerminalAutomaton>>, TerminalFamilyLayout) {
    let layout = TerminalFamilyLayout {
        has_l1: families.l1.is_some(),
        has_l2p: families.l2p.is_some(),
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
    assert!(
        pieces.next().is_none(),
        "unexpected extra terminal family after reconciliation"
    );
    TerminalDwaFamilies { l1, l2p }
}

fn terminal_family_interned_range_count(families: &TerminalDwaFamilies) -> usize {
    let mut weights = Vec::new();
    if let Some(l1) = &families.l1 {
        weights.extend(l1.artifact().weight_refs());
    }
    if let Some(l2p) = &families.l2p {
        weights.extend(l2p.artifact().weight_refs());
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
            terminal_coloring_ms,
            terminal_dwas,
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

        let (templates, prebuilt_parser_dwa) = if dwa_pm_mode.does_terminal_reconcile() {
            (Some(templates), None)
        } else {
            let parser_dwa_started_at = Instant::now();
            let parser_dwa_started_ms = elapsed_ms(compile_started_at.clone());
            let parser_dwa = build_and_merge_parser_dwa_families(
                &terminal_dwas,
                &table,
                &analysis.analyzed_grammar,
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
        let (terminal_dwas, terminal_phase_profile) =
            crate::compiler::stages::id_map_and_terminal_dwa::build_terminal_dwa_families_with_precomputed_global_max_length(
                &tokenizer.tokenizer,
                vocab,
                &terminal_coloring,
                use_terminal_coloring,
                prepared_grammar.ignore_terminal,
                &analysis.analyzed_grammar,
                &analysis.disallowed_follows,
                Arc::clone(&flat_global.flat_trans),
                &flat_global.global_max_length_state_map,
                Some(&classify.shared_classify_cache),
            );
        let terminal_dwa_finished_ms = elapsed_ms(compile_started_at.clone());

        parser_state
            .lock()
            .expect("parser DAG join state poisoned")
            .terminal = Some(TerminalDagResult {
                tokenizer,
                analysis,
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
        let _terminal_path_lengths = classify_terminal_path_lengths(
            &tokenizer.tokenizer,
            vocab,
            &token_path_disallowed_follows,
            analysis.analyzed_grammar.num_terminals,
            Some(&shared_classify_cache),
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
    )
}

fn compile_prepared_with_profile_and_table_construction(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
    default_table_construction: GlrTableConstruction,
) -> (Constraint, CompilePhaseProfile) {
    let interner_cleanup = crate::ds::weight::defer_weight_interner_cleanup();
    let result = run_with_compile_thread_pool(|| {
        let compile_started_at = Instant::now();
        let mut profile = CompilePhaseProfile::default();

        let analysis_started_at = Instant::now();
        let dwa_pm_mode = dwa_possible_matches_mode();
        let use_terminal_coloring = terminal_coloring_enabled();
        let terminal_state = Mutex::new(TerminalDagJoinState::default());
        let parser_state = Mutex::new(ParserDagJoinState::default());
        let compile_dag_result = Mutex::new(None);
        let cpm_result = Mutex::new(None);

        rayon::scope(|scope| {
            let terminal_state_ref = &terminal_state;
            let parser_state_ref = &parser_state;
            let compile_dag_result_ref = &compile_dag_result;
            let cpm_result_ref = &cpm_result;
            let prepared_grammar_ref = &prepared_grammar;
            let analysis_started_for_tokenizer = analysis_started_at.clone();
            let compile_started_for_tokenizer = compile_started_at.clone();

            scope.spawn(move |scope| {
                let tok_started = Instant::now();
                let mut tokenizer = build_tokenizer(prepared_grammar_ref);
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
                let tokenizer_lane = TokenizerDagLane {
                    tokenizer: Arc::new(tokenizer),
                    tokenizer_build_ms: elapsed_ms(tok_started),
                    tokenizer_ready_ms: elapsed_ms(analysis_started_for_tokenizer),
                };

                let possible_matches_tokenizer = Arc::clone(&tokenizer_lane.tokenizer);
                let compile_started_for_cpm = compile_started_for_tokenizer.clone();
                scope.spawn(move |_| {
                    let possible_matches_started_ms = elapsed_ms(compile_started_for_cpm.clone());
                    let result = cpm::compute_constraint_possible_matches_for_vocab(
                        &possible_matches_tokenizer,
                        vocab,
                        cpm::ConstraintPossibleMatchesConfig,
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
                        let compile_started_for_coloring = compile_started_for_glr.clone();
                        scope.spawn(move |scope| {
                            let terminal_coloring_started_at = Instant::now();
                            let terminal_coloring = compute_terminal_coloring(&coloring_table);
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
                let disallowed_follows = Arc::new(compute_disallowed_follows(&analyzed_grammar));
                let analysis_lane = AnalysisDagLane {
                    analyzed_grammar,
                    analyze_grammar_ms,
                    disallowed_follows,
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

        let (cpm_result, possible_matches_started_ms, possible_matches_finished_ms) = cpm_result
            .into_inner()
            .expect("possible-matches result slot poisoned")
            .expect("possible-matches task did not complete");
        let CompileDagResult {
            tokenizer,
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
            prebuilt_parser_dwa,
        } = compile_dag_result
            .into_inner()
            .expect("compile DAG result slot poisoned")
            .expect("compile DAG did not produce a result");
        let tokenizer = Arc::try_unwrap(tokenizer)
            .unwrap_or_else(|_| panic!("tokenizer references outlived compile DAG"));
        let analyzed_grammar = Arc::try_unwrap(analyzed_grammar)
            .unwrap_or_else(|_| panic!("analyzed grammar references outlived compile DAG"));
        let table = Arc::try_unwrap(table)
            .unwrap_or_else(|_| panic!("GLR table references outlived compile DAG"));

        profile.tokenizer_build_ms = tokenizer_build_ms;
        profile.tokenizer_final_states = tokenizer.num_states() as usize;
        profile.tokenizer_final_transitions = tokenizer.transition_count();
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
        let terminal_dwa_interned_ranges_before_pm_reconcile =
            terminal_family_interned_range_count(&terminal_dwas);
        let possible_matches_interned_ranges_before_pm_reconcile =
            interned_range_count_for_artifact(possible_matches.artifact_mut());
        let terminal_pm_joint_interned_ranges_before_reconcile =
            terminal_family_joint_interned_range_count(
                &terminal_dwas,
                possible_matches.artifact(),
            );

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
                    retained_templates,
                    &tokenizer,
                    vocab,
                )
            };
            (parser_dwa, elapsed_ms(parser_dwa_started_at))
        };
        if compile_profile_enabled() {
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

        let terminal_pm_joint_interned_ranges = terminal_family_joint_interned_range_count(
            &terminal_dwas,
            possible_matches.artifact(),
        );

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
        parser_dwa = MappedArtifact::new(parser_dwa_artifact, internal_ids.clone());
        possible_matches =
            MappedArtifact::new(possible_matches_artifact, internal_ids.clone());

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
            dynamic_mask_vocab: DynamicMaskVocab::from_compiler_artifacts(
                runtime_dynamic_vocab.trie,
                runtime_dynamic_vocab.token_aliases,
            ),
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
            direct_sparse_weight_token_sets: rustc_hash::FxHashSet::default(),
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
