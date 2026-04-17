use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use once_cell::sync::Lazy;

use crate::Vocab;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    SharedClassifyCache,
    classify_terminal_path_lengths,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::{
    compute_ever_allowed_follows,
    compute_terminal_coloring,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring;
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::compile_dfa::emit_template_profile_summary;
use crate::compiler::stages::terminal_dwa_compat::build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring;
use crate::ds::bitset::BitSet;
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

pub(crate) fn compile_profile_summary_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

pub(crate) fn compile_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE") || compile_profile_summary_enabled()
}

fn debug_verbose_enabled() -> bool {
    env_flag_enabled("GLRMASK_DEBUG_VERBOSE")
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
    pub(crate) permute_possible_matches_ms: f64,
    pub(crate) internal_token_bytes_ms: f64,
    pub(crate) parser_dwa_ms: f64,
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
        "[glrmask/profile][compile] source={}{} prepare_ms={:.3} tokenizer_build_ms={:.3} analyze_grammar_ms={:.3} glr_table_ms={:.3} terminal_coloring_ms={:.3} disallowed_follows_ms={:.3} analysis_wall_ms={:.3} classify_ms={:.3} id_map_ms={:.3} terminal_dwa_ms={:.3} templates_ms={:.3} compact_ms={:.3} permute_possible_matches_ms={:.3} internal_token_bytes_ms={:.3} parser_dwa_ms={:.3} finalize_ms={:.3} compile_ms={:.3} total_ms={:.3}",
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
        profile.permute_possible_matches_ms,
        profile.internal_token_bytes_ms,
        profile.parser_dwa_ms,
        profile.finalize_ms,
        profile.compile_ms,
        profile.total_ms,
    );
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
    if debug_verbose_enabled() {
        for (index, _) in grammar.terminals.iter().enumerate() {
            eprintln!(
                "[glrmask/debug][tokenizer_terminal] expr={} name={}",
                index,
                grammar.terminal_display_name(index as u32),
            );
        }
    }

    let exprs: Vec<Expr> = grammar.terminals.iter().map(terminal_expr).collect();
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

fn build_internal_token_bytes(vocab: &Vocab, internal_ids: &InternalIdMap) -> BTreeMap<u32, Vec<u8>> {
    internal_ids
        .vocab_tokens
        .representative_original_ids
        .iter()
        .enumerate()
        .filter_map(|(internal_token_id, &representative)| {
            let bytes = vocab.entries.get(&representative)?.clone();
            Some((internal_token_id as u32, bytes))
        })
        .collect()
}

fn finalize_constraint(mut constraint: Constraint) -> Constraint {
    constraint.rebuild_runtime_caches();
    constraint
}

fn warn_problematic_byte_terminals(tokenizer: &Tokenizer, vocab: &Vocab) {
    const SCORE_THRESHOLD: u64 = 100_000;

    let start = tokenizer.start_state();
    let mut problematic = [false; 256];
    let mut any_problematic = false;
    for b in 0u16..=255 {
        if let Some(next_state) = tokenizer.step(start, b as u8) {
            if !tokenizer.matched_terminals(next_state).is_empty()
                && tokenizer.step(next_state, b as u8).is_none()
            {
                problematic[b as usize] = true;
                any_problematic = true;
            }
        }
    }
    if !any_problematic {
        return;
    }

    let mut byte_count = [0u64; 256];
    for bytes in vocab.entries.values() {
        for &b in bytes.iter() {
            byte_count[b as usize] += 1;
        }
    }

    let score: u64 = (0..256)
        .filter(|&b| problematic[b])
        .map(|b| byte_count[b])
        .sum();
    if score < SCORE_THRESHOLD {
        return;
    }

    let problematic_bytes: Vec<u8> = (0..=255u8).filter(|&b| problematic[b as usize]).collect();
    let display = format_byte_ranges(&problematic_bytes);
    eprintln!(
        "[glrmask/warn] grammar has length-1 terminal matches on high-frequency bytes \
         (score={score}, threshold={SCORE_THRESHOLD}): {display}"
    );
}

fn format_byte_ranges(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    fn is_rangeable(b: u8) -> bool {
        b.is_ascii_alphanumeric()
    }
    fn display_byte(b: u8) -> String {
        if b.is_ascii_graphic() || b == b' ' {
            format!("{}", b as char)
        } else {
            format!("0x{b:02x}")
        }
    }

    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = bytes[i];
        if is_rangeable(start) {
            let mut end = start;
            while i + 1 < bytes.len()
                && bytes[i + 1] == end + 1
                && is_rangeable(bytes[i + 1])
            {
                i += 1;
                end = bytes[i];
            }
            if end > start + 1 {
                parts.push(format!("{}-{}", display_byte(start), display_byte(end)));
            } else if end == start + 1 {
                parts.push(display_byte(start));
                parts.push(display_byte(end));
            } else {
                parts.push(display_byte(start));
            }
        } else {
            parts.push(display_byte(start));
        }
        i += 1;
    }
    parts.join(", ")
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
                terminal_coloring_enabled,
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

                if let Err(message) = analyzed_grammar.debug_check_grammar_preconditions() {
                    panic!("[glrmask] grammar precondition violations:\n{}", message);
                }

                let table_started_at = Instant::now();
                let table = GLRTable::build(&analyzed_grammar);
                let glr_table_ms = elapsed_ms(table_started_at);

                let terminal_coloring_started_at = Instant::now();
                let terminal_coloring_enabled = !env_flag_enabled("GLRMASK_DISABLE_TERMINAL_COLORING");
                let terminal_coloring = if terminal_coloring_enabled {
                    compute_terminal_coloring(&table)
                } else {
                    TerminalColoring::identity(table.num_terminals as usize)
                };
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
                    terminal_coloring_enabled,
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

        let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
            .map(|v| {
                let n = v.trim().to_ascii_lowercase();
                !matches!(n.as_str(), "" | "0" | "false" | "no" | "off")
            })
            .unwrap_or(false);
        if debug_profile {
            eprintln!(
                "[glrmask/debug][analysis_overlap] wall_ms={:.3} analyze_ms={:.3} glr_ms={:.3} disallowed_ms={:.3}",
                elapsed_ms(analysis_started_at),
                analyze_grammar_ms,
                glr_table_ms,
                disallowed_follows_ms,
            );
        }

        if env_flag_enabled("GLRMASK_WARN_PROBLEMATIC_BYTE_TERMINALS") {
            let warn_started_at = Instant::now();
            warn_problematic_byte_terminals(&tokenizer, vocab);
            if debug_profile {
                eprintln!(
                    "[glrmask/debug][warn_byte_terminals] ms={:.3}",
                    elapsed_ms(warn_started_at),
                );
            }
        }

        if compile_profile_enabled() {
            let num_groups = analyzed_grammar.num_terminals as usize;
            let mut universally_disallowed = 0usize;
            for gid in 0..num_groups {
                let is_disallowed_by_all = (0..num_groups).all(|other| {
                    disallowed_follows
                        .get(&(other as u32))
                        .is_some_and(|bs| bs.contains(gid))
                });
                if is_disallowed_by_all {
                    universally_disallowed += 1;
                }
            }
            let total_disallowed: usize = disallowed_follows.values().map(|bs| bs.count_ones()).sum();
            let total_possible = num_groups * num_groups;
            let groups_with_disallowed = disallowed_follows.len();
            eprintln!(
                "[glrmask/profile][disallowed_follows] num_groups={} groups_with_disallowed={} total_disallowed_pairs={}/{} ({:.1}%) universally_disallowed_groups={}",
                num_groups,
                groups_with_disallowed,
                total_disallowed,
                total_possible,
                total_disallowed as f64 / total_possible as f64 * 100.0,
                universally_disallowed,
            );
        }

        let adjusted_disallowed_for_classification = if let Some(ign) = prepared_grammar.ignore_terminal {
            let mut adj = disallowed_follows.clone();
            adj.remove(&ign);
            for bits in adj.values_mut() {
                if (ign as usize) < bits.len() {
                    bits.clear(ign as usize);
                }
            }
            adj.retain(|_, bits| !bits.is_zero());
            adj
        } else {
            disallowed_follows.clone()
        };
        let shared_classify_cache = SharedClassifyCache::new();
        let classify_started_at = Instant::now();
        let terminal_path_lengths = classify_terminal_path_lengths(
            &tokenizer,
            vocab,
            &adjusted_disallowed_for_classification,
            analyzed_grammar.num_terminals,
            Some(&shared_classify_cache),
        );
        profile.classify_ms = elapsed_ms(classify_started_at);
        if compile_profile_enabled() {
            use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalPathLength;
            let n0 = terminal_path_lengths
                .iter()
                .filter(|l| **l == TerminalPathLength::Zero)
                .count();
            let n1 = terminal_path_lengths
                .iter()
                .filter(|l| **l == TerminalPathLength::One)
                .count();
            let n2 = terminal_path_lengths
                .iter()
                .filter(|l| **l == TerminalPathLength::TwoPlus)
                .count();
            eprintln!(
                "[glrmask/profile][terminal_path_lengths] total={} length0={} length1={} length2plus={}",
                terminal_path_lengths.len(),
                n0,
                n1,
                n2,
            );
        }

        let all_l1 = {
            use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalPathLength;
            terminal_path_lengths
                .iter()
                .all(|l| matches!(l, TerminalPathLength::Zero | TerminalPathLength::One))
        };

        enum IdMapBuildResult {
            Ready {
                global: InternalIdMap,
                phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile,
            },
            SplitComplete {
                global: InternalIdMap,
                terminal_dwa: crate::automata::weighted::dwa::DWA,
                phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile,
            },
        }

        let ((id_map_build_result, _id_map_wall_ms), (templates, templates_ms)) = rayon::join(
            || {
                let id_map_started_at = Instant::now();
                let result = if let Ok(load_path) = std::env::var("GLRMASK_ORACLE_LOAD") {
                    let data: serde_json::Value = serde_json::from_str(
                        &std::fs::read_to_string(&load_path).expect("failed to read oracle file"),
                    )
                    .expect("failed to parse oracle JSON");
                    let state_map: Vec<u32> = data["state_map"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect();
                    let token_map: Vec<u32> = data["token_map"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect();
                    let num_state_classes = data["num_state_classes"].as_u64().unwrap() as u32;
                    let num_token_classes = data["num_token_classes"].as_u64().unwrap() as u32;
                    let state_reps: Vec<u32> = data["state_representatives"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect();
                    let token_reps: Vec<u32> = data["token_representatives"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect();
                    eprintln!(
                        "[glrmask/oracle] loaded from {load_path}: {num_state_classes} state classes, {num_token_classes} token classes"
                    );
                    IdMapBuildResult::Ready {
                        global: InternalIdMap {
                            tokenizer_states: crate::compiler::stages::equiv_types::ManyToOneIdMap::from_original_to_internal_with_representatives(
                                state_map,
                                num_state_classes,
                                state_reps,
                            ),
                            vocab_tokens: crate::compiler::stages::equiv_types::ManyToOneIdMap::from_original_to_internal_with_representatives(
                                token_map,
                                num_token_classes,
                                token_reps,
                            ),
                        },
                        phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile {
                            id_map_ms: elapsed_ms(id_map_started_at),
                            terminal_dwa_ms: 0.0,
                            compact_ms: 0.0,
                        },
                    }
                } else if all_l1 && std::env::var("GLRMASK_L1_IDMAP").map_or(false, |v| v == "1") {
                    IdMapBuildResult::Ready {
                        global: crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences_l1_fast(&tokenizer, vocab),
                        phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile {
                            id_map_ms: elapsed_ms(id_map_started_at),
                            terminal_dwa_ms: 0.0,
                            compact_ms: 0.0,
                        },
                    }
                } else if std::env::var("GLRMASK_NO_PARTITION").map_or(false, |v| v == "1") {
                    IdMapBuildResult::Ready {
                        global: crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences(&tokenizer, vocab, &disallowed_follows, prepared_grammar.ignore_terminal, None),
                        phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile {
                            id_map_ms: elapsed_ms(id_map_started_at),
                            terminal_dwa_ms: 0.0,
                            compact_ms: 0.0,
                        },
                    }
                } else {
                    let (id_map, dwa, phase_profile) = crate::compiler::stages::id_map_and_terminal_dwa::build_id_map_and_terminal_dwa(
                        &tokenizer,
                        vocab,
                        &terminal_coloring,
                        terminal_coloring_enabled,
                        prepared_grammar.ignore_terminal,
                        &analyzed_grammar,
                        &adjusted_disallowed_for_classification,
                        Some(&shared_classify_cache),
                    );
                    IdMapBuildResult::SplitComplete {
                        global: id_map,
                        terminal_dwa: dwa,
                        phase_profile,
                    }
                };
                (result, elapsed_ms(id_map_started_at))
            },
            || {
                let templates_started_at = Instant::now();
                if compile_profile_enabled() {
                    let characterize_started_at = Instant::now();
                    let characterizations = characterize_terminals(&table, &analyzed_grammar);
                    let characterize_ms = elapsed_ms(characterize_started_at);
                    let (templates, template_profile) =
                        Templates::from_characterizations_profiled(&characterizations);
                    emit_template_profile_summary(characterize_ms, &template_profile);
                    (templates, elapsed_ms(templates_started_at))
                } else {
                    let characterizations = characterize_terminals(&table, &analyzed_grammar);
                    let templates = Templates::from_characterizations(&characterizations);
                    (templates, elapsed_ms(templates_started_at))
                }
            },
        );
        let (mut internal_ids, prebuilt_terminal_dwa, mut terminal_phase_profile) = match id_map_build_result {
            IdMapBuildResult::Ready {
                global,
                phase_profile,
            } => (global, None, phase_profile),
            IdMapBuildResult::SplitComplete {
                global,
                terminal_dwa,
                phase_profile,
            } => (global, Some(terminal_dwa), phase_profile),
        };
        profile.templates_ms = templates_ms;
        let token_bytes = vocab.entries.clone();

        let (mut terminal_dwa, already_compacted) = if let Some(dwa) = prebuilt_terminal_dwa {
            (dwa, true)
        } else {
            let terminal_dwa_started_at = Instant::now();
            let (dwa, _pm) = build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring(
                &analyzed_grammar,
                &tokenizer,
                vocab,
                &internal_ids,
                &terminal_coloring,
                terminal_coloring_enabled,
                prepared_grammar.ignore_terminal,
                Some(&adjusted_disallowed_for_classification),
            );
            terminal_phase_profile.terminal_dwa_ms += elapsed_ms(terminal_dwa_started_at);
            (dwa, false)
        };

        if !already_compacted {
            let compact_started_at = Instant::now();
            let compact_report = crate::compiler::stages::compact::compact_from_env(
                &mut terminal_dwa,
                &mut internal_ids,
                "GLRMASK_COMPACT_FINAL",
                crate::compiler::stages::compact::CompactMode::Full,
                compile_profile_summary_enabled(),
            );
            let compact_ms = elapsed_ms(compact_started_at);
            if let Some(stats) = compact_report.profile_stats {
                eprintln!(
                    "[glrmask/profile][compact] tsids={}=>{} tokens={}=>{} weight_ranges={}=>{} token_ranges={}=>{} total_ranges={}=>{}",
                    stats.tsids_before,
                    stats.tsids_after,
                    stats.tokens_before,
                    stats.tokens_after,
                    stats.weight_ranges_before,
                    stats.weight_ranges_after,
                    stats.token_ranges_before,
                    stats.token_ranges_after,
                    stats.total_ranges_before(),
                    stats.total_ranges_after(),
                );
            }
            terminal_phase_profile.compact_ms += compact_ms;
        }
        profile.id_map_ms = terminal_phase_profile.id_map_ms;
        profile.terminal_dwa_ms = terminal_phase_profile.terminal_dwa_ms;
        profile.compact_ms = terminal_phase_profile.compact_ms;

        if let Ok(dump_path) = std::env::var("GLRMASK_ORACLE_DUMP") {
            let mut canonical_state_reps = vec![u32::MAX; internal_ids.num_tsids() as usize];
            for (orig, &class) in internal_ids
                .tokenizer_states
                .original_to_internal
                .iter()
                .enumerate()
            {
                let orig = orig as u32;
                if orig < canonical_state_reps[class as usize] {
                    canonical_state_reps[class as usize] = orig;
                }
            }
            let mut canonical_token_reps = vec![u32::MAX; internal_ids.num_internal_tokens() as usize];
            for (orig, &class) in internal_ids.vocab_tokens.original_to_internal.iter().enumerate() {
                let orig = orig as u32;
                if orig < canonical_token_reps[class as usize] {
                    canonical_token_reps[class as usize] = orig;
                }
            }
            let oracle_data = serde_json::json!({
                "state_map": internal_ids.tokenizer_states.original_to_internal,
                "token_map": internal_ids.vocab_tokens.original_to_internal,
                "num_state_classes": internal_ids.num_tsids(),
                "num_token_classes": internal_ids.num_internal_tokens(),
                "state_representatives": canonical_state_reps,
                "token_representatives": canonical_token_reps,
            });
            std::fs::write(&dump_path, serde_json::to_string(&oracle_data).unwrap())
                .expect("failed to write oracle dump");
            eprintln!("[glrmask/oracle] dumped post-compact mappings to {dump_path}");
        }

        let internal_token_bytes_started_at = Instant::now();
        let internal_token_bytes = build_internal_token_bytes(vocab, &internal_ids);
        let internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

        let ((parser_dwa, parser_dwa_ms), (possible_matches, permute_possible_matches_ms)) =
            rayon::join(
                || {
                    let parser_dwa_started_at = Instant::now();
                    let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                        &table,
                        &analyzed_grammar,
                        &terminal_dwa,
                        templates,
                    );
                    (parser_dwa, elapsed_ms(parser_dwa_started_at))
                },
                || {
                    let pm_started_at = Instant::now();
                    let token_entries: Vec<(usize, Vec<u8>)> = internal_token_bytes
                        .iter()
                        .map(|(&id, bytes)| (id as usize, bytes.clone()))
                        .collect();
                    let trie_build_started_at = Instant::now();
                    let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(token_entries);
                    let trie_build_ms = elapsed_ms(trie_build_started_at);
                    let collect_started_at = Instant::now();
                    let (pm_by_tsid, dense_profile) = crate::compiler::possible_matches::collect_possible_matches_by_internal_tsid_dense(
                        &tokenizer,
                        &trie.root,
                        &internal_ids.tokenizer_states,
                        internal_ids.vocab_tokens.num_internal_ids(),
                    );
                    let collect_ms = elapsed_ms(collect_started_at);
                    crate::compiler::possible_matches::emit_possible_matches_profile_summary(
                        "internal_tsid",
                        internal_token_bytes.len(),
                        internal_ids.tokenizer_states.num_internal_ids(),
                        trie_build_ms,
                        collect_ms,
                        &dense_profile,
                    );
                    (pm_by_tsid, elapsed_ms(pm_started_at))
                },
            );
        profile.parser_dwa_ms = parser_dwa_ms;
        profile.permute_possible_matches_ms = permute_possible_matches_ms;
        profile.internal_token_bytes_ms = internal_token_bytes_ms;

        let finalize_started_at = Instant::now();
        let constraint = finalize_constraint(Constraint {
            parser_dwa,
            table,
            tokenizer,
            ignore_terminal: prepared_grammar.ignore_terminal,
            possible_matches,
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
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: rustc_hash::FxHashMap::default(),
            seed_terminal_dense: rustc_hash::FxHashMap::default(),
            seed_universe_dense: Box::new([]),
            dwa_fast_transitions: Vec::new(),
            heavy_token_dense_masks: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            n_heavy_tokens: 0,
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
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
    if compile_profile_enabled() || env_flag_enabled("GLRMASK_PROFILE_PHASES") {
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        if compile_profile_summary_enabled() {
            emit_compile_profile_summary(None, None, &profile);
        } else {
            eprintln!(
                "[glrmask/profile][phases] prepare={:.1} analysis_wall={:.1} classify={:.1} id_map={:.1} terminal_dwa={:.1} templates={:.1} compact={:.1} possible_matches={:.1} internal_token_bytes={:.1} parser_dwa={:.1} finalize={:.1} compile={:.1} total={:.1}",
                profile.prepare_ms,
                profile.analysis_wall_ms,
                profile.classify_ms,
                profile.id_map_ms,
                profile.terminal_dwa_ms,
                profile.templates_ms,
                profile.compact_ms,
                profile.permute_possible_matches_ms,
                profile.internal_token_bytes_ms,
                profile.parser_dwa_ms,
                profile.finalize_ms,
                profile.compile_ms,
                profile.total_ms,
            );
        }
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