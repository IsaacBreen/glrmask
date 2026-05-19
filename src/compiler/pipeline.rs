use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use once_cell::sync::Lazy;

use crate::Vocab;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::compiler::constraint_possible_matches as cpm;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    SharedClassifyCache,
    classify_terminal_path_lengths,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::{
    compute_ever_allowed_follows,
    compute_terminal_coloring,
};
use crate::compiler::stages::mapped_artifact::{
    MappedArtifact,
    WeightRefs,
    count_interned_ranges_for_weights,
};
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals_profiled;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::grammar::flat::{GrammarDef, Terminal};
use crate::runtime::Constraint;

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
            // Parser-side PM compaction remains available via `GLRMASK_DWA_PM_MODE=both`
            // and the parser compact modes, but it is not the default because large
            // schemas can pay several extra compile seconds for small artifact-size wins.
            DwaPossibleMatchesMode::TerminalReconcileAndCompact
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
        "[glrmask/profile][compile] source={}{} prepare_ms={:.3} tokenizer_build_ms={:.3} analyze_grammar_ms={:.3} glr_table_ms={:.3} terminal_coloring_ms={:.3} disallowed_follows_ms={:.3} analysis_wall_ms={:.3} classify_ms={:.3} id_map_ms={:.3} terminal_dwa_ms={:.3} templates_ms={:.3} compact_ms={:.3} possible_matches_collect_ms={:.3} possible_matches_materialize_ms={:.3} shared_id_reconcile_ms={:.3} possible_matches_pipeline_ms={:.3} terminal_dwa_interned_ranges_before_pm_reconcile={} possible_matches_interned_ranges_before_pm_reconcile={} terminal_pm_joint_interned_ranges_before_reconcile={} terminal_pm_joint_interned_ranges={} internal_token_bytes_ms={:.3} parser_dwa_ms={:.3} parser_dwa_interned_ranges={} possible_matches_interned_ranges={} parser_pm_joint_interned_ranges={} finalize_ms={:.3} compile_ms={:.3} total_ms={:.3}",
        source,
        import_fragment,
        profile.prepare_ms,
        profile.tokenizer_build_ms,
        profile.analyze_grammar_ms,
        profile.glr_table_ms,
        profile.terminal_coloring_ms,
        profile.disallowed_follows_ms,
        profile.analysis_wall_ms,
        profile.classify_ms,
        profile.id_map_ms,
        profile.terminal_dwa_ms,
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

pub(crate) fn build_tokenizer(grammar: &GrammarDef) -> Tokenizer {
    let exprs: Vec<Expr> = grammar.terminals.iter().map(terminal_expr).collect();
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
                "[glrmask/profile][tokenizer] terminal id={} name={:?} states={} alone_ms={:.3}",
                index,
                name,
                regex.num_states(),
                elapsed
            );
        }
    }
    build_tokenizer_from_exprs(&exprs)
}

pub(crate) fn build_tokenizer_from_exprs(exprs: &[Expr]) -> Tokenizer {
    let regex = build_regex(exprs);

    Tokenizer {
        dfa: regex.dfa,
        num_terminals: exprs.len() as u32,
        exprs: Some(std::sync::Arc::from(exprs.to_vec())),
    }
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
    run_with_compile_thread_pool(|| {
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
                tokenizer.isolate_start_state_and_drain_nullable_terminals();
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

        let disallowed_follows_for_classification = &disallowed_follows;
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

        let (((mut terminal_dwa, mut terminal_phase_profile), cpm_result), (templates, templates_ms)) = rayon::join(
            || {
                rayon::join(
                    || {
                        crate::compiler::stages::id_map_and_terminal_dwa::build_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                            &tokenizer,
                            vocab,
                            &terminal_coloring,
                            true,
                            prepared_grammar.ignore_terminal,
                            &analyzed_grammar,
                            disallowed_follows_for_classification,
                            Arc::clone(&flat_trans),
                            &global_max_length_state_map,
                            Some(&shared_classify_cache),
                        )
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
                let (templates, template_profile) =
                    Templates::from_characterizations_profiled(&characterizations);
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][templates] terminals={} action_signature_classes={} action_quotient_hits={} max_action_signature_multiplicity={} characterization_signature_ms={:.3} characterization_ms={:.3} characterization_fanout_ms={:.3} characterization_validation_ms={:.3} characterization_total_ms={:.3} characterization_quotient_disabled={} unique_characterizations={} compiled_characterizations={} template_quotient_hits={} max_characterization_multiplicity={} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} template_fanout_ms={:.3} template_validation_ms={:.3} template_total_ms={:.3} template_wall_ms={:.3} template_minimize_skipped={} avg_nfa_states={:.2} avg_nfa_transitions={:.2} avg_premin_dfa_states={:.2} avg_premin_dfa_transitions={:.2} avg_dfa_states={:.2} avg_dfa_transitions={:.2} max_dfa_states={} max_dfa_transitions={}",
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
                    );
                }
                (templates, elapsed_ms(templates_started_at))
            },
        );
        terminal_phase_profile.terminal_dwa_ms += flat_trans_ms;
        terminal_phase_profile.id_map_ms += global_max_length_ms;
        profile.templates_ms = templates_ms;
        profile.id_map_ms = terminal_phase_profile.id_map_ms;
        profile.terminal_dwa_ms = terminal_phase_profile.terminal_dwa_ms;
        profile.compact_ms = terminal_phase_profile.compact_ms;

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
            possible_matches: possible_matches.into_artifact(),
            state_to_internal_tsid: internal_ids.tokenizer_states.original_to_internal.clone(),
            internal_tsid_to_states: internal_ids.tokenizer_states.internal_to_originals_vecs(),
            original_token_to_internal: internal_ids.vocab_tokens.original_to_internal.clone(),
            internal_token_to_tokens: internal_ids.vocab_tokens.internal_to_originals_vecs(),
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
            seed_state_dense: Vec::new(),
            seed_state_by_dense_hash: rustc_hash::FxHashMap::default(),
            seed_state_buf_masks: Vec::new(),
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
    })
}

pub(crate) fn compile_prepared(prepared_grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    compile_prepared_with_profile(prepared_grammar, vocab).0
}

pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    if compile_profile_summary_enabled() {
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(None, None, &profile);
        return constraint;
    }

    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    compile_prepared(prepared_grammar, vocab)
}

pub(crate) fn compile_owned_profiled(
    grammar: GrammarDef,
    vocab: &Vocab,
) -> (Constraint, CompilePhaseProfile) {
    let total_started_at = Instant::now();
    let prepare_started_at = Instant::now();
    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    let prepare_ms = elapsed_ms(prepare_started_at);

    let (constraint, mut profile) = compile_prepared_with_profile(prepared_grammar, vocab);
    profile.prepare_ms = prepare_ms;
    profile.total_ms = elapsed_ms(total_started_at);
    (constraint, profile)
}
