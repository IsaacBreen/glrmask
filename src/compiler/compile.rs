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
use crate::compiler::grammar::model::{GrammarDef, Terminal};
#[cfg(test)]
use crate::compiler::grammar::transforms::prepare_grammar_for_compile;
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::terminal_dwa_compat::build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::compile_dfa::emit_template_profile_summary;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::compute_terminal_coloring;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::compute_ever_allowed_follows;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{classify_terminal_path_lengths, SharedClassifyCache};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring;
use crate::ds::bitset::BitSet;
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

    let exprs: Vec<Expr> = grammar
        .terminals
        .iter()
        .map(terminal_expr)
        .collect();
    build_tokenizer_from_exprs(&exprs)
}

pub(crate) fn build_tokenizer_from_exprs(exprs: &[Expr]) -> Tokenizer {
    let regex = build_regex(exprs);

    Tokenizer {
        dfa: regex.dfa,
        num_terminals: exprs.len() as u32,
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

/// Warn if the grammar has length-1 terminal matches on bytes that appear
/// frequently in the vocabulary.  Such terminals create many product-DFA
/// states and can slow compilation significantly.
fn warn_problematic_byte_terminals(tokenizer: &Tokenizer, vocab: &Vocab) {
    const SCORE_THRESHOLD: u64 = 100_000;

    // Quick check: identify problematic bytes first (O(256) DFA lookups).
    // If none exist, skip the expensive byte counting entirely.
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

    // Count byte frequencies directly from vocab token bytes.
    // This counts each byte occurrence in each token (including shared prefixes),
    // which slightly overcounts vs. trie segment counting, but is sufficient for
    // the warning heuristic and avoids building a full VocabPrefixTree (~50-80ms).
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

/// Format a list of bytes into compact ranges.  Alphanumeric runs like
/// `a,b,c,d` are compressed into `a-d`, while punctuation bytes are always
/// listed individually.
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
            // Try to extend a consecutive alphanumeric range.
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

        // Build tokenizer concurrently with grammar analysis + GLR table + disallowed follows.
        // Tokenizer build (~70ms) runs in parallel with analysis (~46ms), saving ~46ms.
        let analysis_started_at = Instant::now();
        let (
            (tokenizer, tokenizer_build_ms),
            (analyzed_grammar, analyze_grammar_ms, table, glr_table_ms,
             terminal_coloring, terminal_coloring_enabled, terminal_coloring_ms,
             disallowed_follows, disallowed_follows_ms),
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

                #[cfg(debug_assertions)]
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

                (analyzed_grammar, analyze_grammar_ms, table, glr_table_ms,
                 terminal_coloring, terminal_coloring_enabled, terminal_coloring_ms,
                 disallowed_follows, disallowed_follows_ms)
            },
        );

        profile.tokenizer_build_ms = tokenizer_build_ms;
        profile.analyze_grammar_ms = analyze_grammar_ms;
        profile.glr_table_ms = glr_table_ms;
        profile.terminal_coloring_ms = terminal_coloring_ms;
        profile.disallowed_follows_ms = disallowed_follows_ms;
        profile.analysis_wall_ms = elapsed_ms(analysis_started_at);

        let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
            .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
            .unwrap_or(false);
        if debug_profile {
            eprintln!(
                "[glrmask/debug][analysis_overlap] wall_ms={:.3} analyze_ms={:.3} glr_ms={:.3} disallowed_ms={:.3}",
                elapsed_ms(analysis_started_at), analyze_grammar_ms, glr_table_ms, disallowed_follows_ms,
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
                    disallowed_follows.get(&(other as u32)).map_or(false, |bs| bs.contains(gid))
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
                num_groups, groups_with_disallowed, total_disallowed, total_possible,
                total_disallowed as f64 / total_possible as f64 * 100.0,
                universally_disallowed,
            );
        }

        // Classify terminals by path length for future optimization.
        // Adjust disallowed follows for the ignore terminal: the ignore
        // terminal can follow and be followed by any terminal.
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
            let n0 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::Zero).count();
            let n1 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::One).count();
            let n2 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::TwoPlus).count();
            eprintln!(
                "[glrmask/profile][terminal_path_lengths] total={} length0={} length1={} length2plus={}",
                terminal_path_lengths.len(), n0, n1, n2,
            );
        }

        let all_l1 = {
            use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalPathLength;
            terminal_path_lengths.iter().all(|l| matches!(l, TerminalPathLength::Zero | TerminalPathLength::One))
        };

        enum IdMapBuildResult {
            Ready {
                global: InternalIdMap,
                phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile,
            },
            /// Id_map + terminal DWA already built (and compacted).
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
                    // L1 fast path: direct fingerprint-based equivalence,
                    // no partitioning needed. Opt-in via env var for now.
                    IdMapBuildResult::Ready {
                        global: crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences_l1_fast(&tokenizer, vocab),
                        phase_profile: crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaPhaseProfile {
                            id_map_ms: elapsed_ms(id_map_started_at),
                            terminal_dwa_ms: 0.0,
                            compact_ms: 0.0,
                        },
                    }
                } else if std::env::var("GLRMASK_NO_PARTITION").map_or(false, |v| v == "1") {
                    // Force non-partitioned path for benchmarking.
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
                        &tokenizer, vocab, &terminal_coloring, terminal_coloring_enabled,
                        prepared_grammar.ignore_terminal, &analyzed_grammar,
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
            IdMapBuildResult::Ready { global, phase_profile } => (global, None, phase_profile),
            IdMapBuildResult::SplitComplete {
                global,
                terminal_dwa,
                phase_profile,
            } => (global, Some(terminal_dwa), phase_profile),
        };
        profile.templates_ms = templates_ms;
        let token_bytes = vocab.entries.clone();

        let (mut terminal_dwa, already_compacted) = if let Some(dwa) = prebuilt_terminal_dwa {
            (dwa, true) // SplitComplete: already compacted by merge::f4
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

        // Oracle dump: save post-compact mappings for two-pass experiment
        if let Ok(dump_path) = std::env::var("GLRMASK_ORACLE_DUMP") {
            let mut canonical_state_reps = vec![u32::MAX; internal_ids.num_tsids() as usize];
            for (orig, &class) in internal_ids.tokenizer_states.original_to_internal.iter().enumerate() {
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

        // Build internal_token_bytes (needed for possible_matches and Constraint).
        let internal_token_bytes_started_at = Instant::now();
        let internal_token_bytes = build_internal_token_bytes(vocab, &internal_ids);
        let internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

        // Build possible_matches from scratch in post-compact space, and
        // parser_dwa in parallel.
        let (
            (parser_dwa, parser_dwa_ms),
            (possible_matches, permute_possible_matches_ms),
        ) = rayon::join(
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

fn compile_prepared(prepared_grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    compile_prepared_with_profile(prepared_grammar, vocab).0
}

#[cfg(test)]
pub(crate) fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    let (prepared_grammar, _tokenizer) = prepare_grammar_for_compile(grammar);
    compile_prepared(prepared_grammar, vocab)
}

pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    if compile_profile_enabled() || env_flag_enabled("GLRMASK_PROFILE_PHASES") {
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        if compile_profile_summary_enabled() {
            emit_compile_profile_summary(None, None, &profile);
        } else {
            // Lightweight phases-only summary (no sub-phase profiling overhead)
            eprintln!(
                "[glrmask/profile][phases] prepare={:.1} analysis_wall={:.1} classify={:.1} id_map={:.1} terminal_dwa={:.1} templates={:.1} compact={:.1} possible_matches={:.1} internal_token_bytes={:.1} parser_dwa={:.1} finalize={:.1} compile={:.1} total={:.1}",
                profile.prepare_ms, profile.analysis_wall_ms, profile.classify_ms,
                profile.id_map_ms, profile.terminal_dwa_ms, profile.templates_ms,
                profile.compact_ms, profile.permute_possible_matches_ms,
                profile.internal_token_bytes_ms, profile.parser_dwa_ms,
                profile.finalize_ms, profile.compile_ms, profile.total_ms,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::tests::*;
    use crate::compiler::grammar::model::{NonterminalID, Rule, Symbol, Terminal};
    use crate::compiler::grammar::transforms::{
        compact_unused_terminals,
        expand_nullable_terminals,
        inline_single_use_nonterminals,
        prepare_owned_grammar_for_compile,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences;
    use crate::import::json_schema::json_schema_to_grammar;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    fn mask_has_token(mask: &[u32], token: u32) -> bool {
        let word = token as usize / 32;
        let bit = token as usize % 32;
        word < mask.len() && (mask[word] & (1u32 << bit)) != 0
    }

    fn kb814_normalized_schema_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kb814_normalized_schema.json")
    }

    fn kb814_prepared_terminals_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kb814_prepared_terminals.json")
    }

    fn gpt2_vocab_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../grammars2024/benchmarking/gpt2_vocab.json")
    }

    fn gpt2_token_str_to_bytes(token_str: &str) -> Vec<u8> {
        token_str.chars().map(unicode_char_to_byte).collect()
    }

    fn unicode_char_to_byte(ch: char) -> u8 {
        if let Some(byte) = printable_byte(ch) {
            return byte;
        }

        let codepoint = ch as u32;
        let offset = codepoint
            .checked_sub(256)
            .expect("unsupported GPT-2 vocab char");
        for byte in 0u16..=255 {
            if printable_byte(char::from_u32(byte as u32).unwrap()).is_none() {
                let candidate_offset = non_printable_rank(byte as u8);
                if candidate_offset == offset as usize {
                    return byte as u8;
                }
            }
        }
        panic!("unable to decode GPT-2 vocab char: {ch:?}");
    }

    fn printable_byte(ch: char) -> Option<u8> {
        let codepoint = ch as u32;
        if (33..=126).contains(&codepoint)
            || (161..=172).contains(&codepoint)
            || (174..=255).contains(&codepoint)
        {
            Some(codepoint as u8)
        } else {
            None
        }
    }

    fn non_printable_rank(target: u8) -> usize {
        let mut rank = 0usize;
        for byte in 0u16..target as u16 {
            let byte = byte as u8;
            if printable_byte(char::from_u32(byte as u32).unwrap()).is_none() {
                rank += 1;
            }
        }
        rank
    }

    fn load_gpt2_vocab() -> Vocab {
        let vocab_path = gpt2_vocab_path();
        let vocab_json = fs::read_to_string(&vocab_path)
            .unwrap_or_else(|err| panic!("failed to read GPT-2 vocab at {}: {err}", vocab_path.display()));
        let vocab_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&vocab_json).expect("parse GPT-2 vocab json");
        let entries = vocab_map
            .into_iter()
            .map(|(token_str, token_id)| {
                let token_id = token_id.as_u64().expect("token id must be integer") as u32;
                (token_id, gpt2_token_str_to_bytes(&token_str))
            })
            .collect();
        Vocab::new(entries, None)
    }

    #[test]
    fn test_compile_simple_ab() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(!constraint.possible_matches_for_state(0).is_empty());
    }

    #[test]
    fn test_possible_matches_union_covers_all_tokenizer_reachable_tokens() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"ba".to_vec()),
                (4, b"x".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);

        for tokenizer_state in 0..constraint.tokenizer.num_states() {
            let mut expected = std::collections::BTreeSet::new();
            for (token_id, token_bytes) in &vocab.entries {
                let exec = constraint
                    .tokenizer
                    .execute_from_state(token_bytes, tokenizer_state);
                if !exec.matches.is_empty() {
                    expected.insert(*token_id);
                }
            }

            let actual: std::collections::BTreeSet<u32> = constraint
                .possible_matches_for_state(tokenizer_state)
                .values()
                .flat_map(|token_ids| token_ids.iter())
                .collect();

            // After dimension compaction (token merging), possible_matches may
            // over-approximate: merged classes expand to all original tokens even
            // if only the representative matched.  The soundness requirement is
            // that every truly reachable token appears (expected ⊆ actual).
            assert!(
                expected.is_subset(&actual),
                "possible_matches union should cover all tokenizer-reachable tokens for state {} \
                 (expected {:?} ⊆ actual {:?})",
                tokenizer_state,
                expected,
                actual,
            );
        }
    }

    #[test]
    fn test_compile_choice() {
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
    }

    #[test]
    fn test_compile_two_nt() {
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(constraint.table.num_states > 0);
    }

    #[test]
    fn test_compile_duplicate_token_bytes_expand_back_to_all_original_tokens() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );

        let mask = compile(&gdef, &vocab).start().mask();
        assert!(mask_has_token(&mask, 10));
        assert!(mask_has_token(&mask, 20));
        assert!(!mask_has_token(&mask, 30));
    }

    #[test]
    fn test_compile_duplicate_token_bytes_collapse_in_internal_possible_matches() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let tokenizer_state = constraint.tokenizer.initial_state();
        let internal_token = constraint.internal_token_for_original(10);
        assert_eq!(internal_token, constraint.internal_token_for_original(20));

        let internal_matches: std::collections::BTreeSet<u32> = constraint
            .possible_matches_for_state_internal(tokenizer_state)
            .into_iter()
            .flat_map(|m| m.into_values())
            .flat_map(|token_ids| token_ids.into_iter())
            .collect();
        assert_eq!(internal_matches, std::collections::BTreeSet::from([internal_token]));

        let original_matches: std::collections::BTreeSet<u32> = constraint
            .possible_matches_for_state(tokenizer_state)
            .values()
            .flat_map(|token_ids| token_ids.iter())
            .collect();
        assert_eq!(original_matches, std::collections::BTreeSet::from([10, 20]));
    }

    #[test]
    fn test_build_tokenizer_projects_hidden_exclusion_groups() {
        let grammar = GrammarDef {
            rules: vec![],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Exclude {
                        expr: Box::new(Expr::U8Class(crate::ds::u8set::U8Set::from_range(0, 255))),
                        exclude: Box::new(Expr::U8Seq(b"a".to_vec())),
                    },
                },
            ],
            ..Default::default()
        };

        let tokenizer = build_tokenizer(&grammar);

        assert_eq!(tokenizer.matched_terminals(tokenizer.run(b"a")), std::collections::BTreeSet::from([0]));
        assert_eq!(tokenizer.matched_terminals(tokenizer.run(b"b")), std::collections::BTreeSet::from([1]));
    }

    #[test]
    fn test_end_to_end_simple_ab() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_choice() {
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed");

        state
            .commit_token(0).unwrap();
        assert!(
            state.is_finished(),
            "parse should accept after 'a'"
        );
    }

    #[test]
    fn test_end_to_end_two_nt() {
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_nested_nt() {
        let gdef = nested_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(!state.is_finished(), "not accepting after 'a'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_three_terminals() {
        let gdef = three_terminal_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        state.commit_token(0).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    #[test]
    fn test_end_to_end_nested_two_rhs() {
        let gdef = nested_two_rhs_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        state.commit_token(0).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    #[test]
    fn test_commit_preserves_longer_terminal_continuation_after_shorter_match() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"ab".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");

        state.commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "the shorter literal 'a' should not complete a grammar expecting 'ab'"
        );

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token 'b' should remain allowed as a continuation of the longer literal 'ab'"
        );

        state.commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after committing 'ab' byte by byte");
    }

    // ── Nullable terminal expansion tests ───────────────────────────────────

    #[test]
    fn test_expand_nullable_terminals_no_nullables() {
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::new();
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);
        assert_eq!(rules.len(), gdef.rules.len());
        assert_eq!(rules[0].rhs, gdef.rules[0].rhs);
    }

    #[test]
    fn test_expand_nullable_terminals_single_nullable() {
        // Grammar: S → t0 t1, where t0 is nullable.
        // Expected: fresh NT2, S → NT2 t1, NT2 → ε, NT2 → t0
        let gdef = simple_ab_grammar(); // S → T0 T1, nonterminals: {0}
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        // 1 rewritten original rule + 2 fresh-NT rules = 3 total.
        assert_eq!(rules.len(), 3);

        // The fresh NT id should be grammar.num_nonterminals() = 1.
        let fresh_nt = gdef.num_nonterminals();

        // S → NT_fresh t1
        assert_eq!(rules[0].lhs, 0);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Terminal(1)]
        );

        // NT_fresh → ε and NT_fresh → t0
        let fresh_rules: Vec<&Rule> =
            rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);
        let rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            fresh_rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(rhs_set.contains(&vec![])); // ε
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0)])); // t0
    }

    #[test]
    fn test_expand_nullable_terminals_both_nullable() {
        // Grammar: S → t0 t1, where both are nullable.
        // Expected: fresh NT1 for t0, fresh NT2 for t1.
        // S → NT1 NT2, NT1 → ε | t0, NT2 → ε | t1
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::from([0u32, 1u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        // 1 rewritten rule + 2*2 fresh-NT rules = 5 total.
        assert_eq!(rules.len(), 5);

        let nt0 = gdef.num_nonterminals();     // fresh NT for t0
        let nt1 = gdef.num_nonterminals() + 1; // fresh NT for t1

        // S → NT0 NT1
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(nt0), Symbol::Nonterminal(nt1)]
        );
    }

    #[test]
    fn test_expand_nullable_terminals_nonterminal_untouched() {
        // Grammar: S → A t1, A → t0. If t0 is nullable:
        //   - Fresh NT for t0.
        //   - S → A t1 unchanged (A is a nonterminal, not touched).
        //   - A → NT_fresh (rewritten from A → t0).
        let gdef = two_nt_grammar(); // S → N1 T1, N1 → T0
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals(); // = 2

        // S → N1 T1 — N1 is a nonterminal, not rewritten.
        let s_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(s_rules.len(), 1);
        assert_eq!(
            s_rules[0].rhs,
            vec![Symbol::Nonterminal(1), Symbol::Terminal(1)]
        );

        // N1 → NT_fresh (was N1 → T0, T0 is nullable so replaced).
        let n1_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 1).collect();
        assert_eq!(n1_rules.len(), 1);
        assert_eq!(n1_rules[0].rhs, vec![Symbol::Nonterminal(fresh_nt)]);

        // Fresh NT → ε and Fresh NT → T0.
        let fresh_rules: Vec<&Rule> =
            rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);
    }

    #[test]
    fn test_expand_nullable_terminals_multiple_occurrences() {
        // Grammar: S → t0 t0, where t0 is nullable.
        // Both occurrences should be replaced by the SAME fresh NT.
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals(); // = 1
        // S → NT NT (same fresh NT for both positions) + 2 fresh-NT rules = 3.
        assert_eq!(rules.len(), 3);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Nonterminal(fresh_nt)]
        );
    }

    #[test]
    fn test_drain_nullable_terminals_from_tokenizer() {
        // Build a tokenizer with a nullable terminal (regex `a*` matches empty string).
        let exprs = vec![
            crate::automata::regex::Expr::Repeat {    // nullable: matches ""
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),                  // not nullable
        ];
        let mut tok = build_tokenizer_from_exprs(&exprs);

        // Before drain: terminal 0 should match at start state.
        assert!(
            tok.matched_terminals(tok.start_state()).contains(&0),
            "terminal 0 should be a start-state finalizer before drain"
        );

        let nullable = tok.isolate_start_state_and_drain_nullable_terminals();
        assert_eq!(nullable, std::collections::BTreeSet::from([0u32]));

        // After drain: terminal 0 should NOT match at start state.
        assert!(
            !tok.matched_terminals(tok.start_state()).contains(&0),
            "terminal 0 should be removed from start-state finalizers after drain"
        );
    }

    #[test]
    fn test_compile_with_nullable_terminal() {
        // S → opt_a b, where opt_a is `a*` (nullable).
        // The grammar should accept both "ab" and "b".
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Pattern {
                    id: 0,
                    pattern: "a*".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"aa".to_vec()),
            ],
            None,
        );
        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);

        // "b" alone should be accepted (opt_a consumed nothing).
        let state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 1), "'b' should be allowed initially (opt_a is nullable)");
    }

    #[test]
    fn test_compact_unused_terminals_remaps_rules_and_terminal_ids() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "used terminals should be renumbered densely when a dead terminal is removed from the middle"
        );
        assert_eq!(grammar.terminals.len(), 2);
        assert_eq!(grammar.terminals[0].id(), 0);
        assert_eq!(grammar.terminals[1].id(), 1);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        assert_eq!(grammar.ignore_terminal, None);
    }

    #[test]
    fn test_compact_unused_terminals_preserves_ignore_terminal_and_remaps_it() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "used terminals should still be renumbered densely when an ignore terminal is retained"
        );
        assert_eq!(grammar.terminals.len(), 3);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), " +");
        assert_eq!(grammar.terminals[2].name(), "b");
        assert_eq!(grammar.ignore_terminal, Some(1));
    }

    #[test]
    fn test_compact_unused_terminals_merges_identical_terminals() {
        // Terminals 0 and 2 are identical ("a"), terminal 1 is different ("b").
        // After compacting, terminals 0 and 2 should map to the same new ID.
        let mut grammar = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(2)] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() },
                Terminal::Literal { id: 2, bytes: b"a".to_vec() },
            ],
            nonterminal_names: BTreeMap::new(),
            terminal_names: BTreeMap::new(),
            ignore_terminal: None,
        };
        compact_unused_terminals(&mut grammar);
        assert_eq!(grammar.terminals.len(), 2, "identical terminals should be merged");
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        // Rule 1: T0 → merged "a" (id 0), T1 → "b" (id 1)
        assert_eq!(grammar.rules[0].rhs, vec![Symbol::Terminal(0), Symbol::Terminal(1)]);
        // Rule 2: T2 → merged "a" (id 0)
        assert_eq!(grammar.rules[1].rhs, vec![Symbol::Terminal(0)]);
    }

    #[test]
    fn test_compile_drops_unused_terminals_before_final_tokenizer_build() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Pattern {
                    id: 1,
                    pattern: "x*".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"x".to_vec()),
            ],
            None,
        );

        let (normalized, tokenizer) = prepare_grammar_for_compile(&gdef);
        let constraint = compile_owned(gdef, &vocab);

        assert_eq!(
            tokenizer.num_terminals,
            2,
            "the final tokenizer should be built only from the live compacted terminals"
        );
        assert_eq!(normalized.terminals.len(), 2);
        assert_eq!(
            normalized
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()],
            "the dead middle terminal should be absent from the normalized grammar"
        );
        assert_eq!(
            normalized.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "rules should be remapped to the compacted terminal IDs"
        );

        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should still be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");
        assert!(!mask_has_token(&mask, 2), "dead terminal token 'x' should not leak into the mask");

        state.commit_token(0).unwrap();
        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should not be allowed after committing 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should remain the live continuation after remapping");
        assert!(!mask_has_token(&mask, 2), "dead terminal token 'x' should remain absent after remapping");
    }

    #[test]
    fn test_compile_treats_ignore_terminal_as_epsilon_and_preserves_it_through_compaction() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b" ".to_vec()),
                (2, b"b".to_vec()),
                (3, b" a".to_vec()),
                (4, b" b".to_vec()),
            ],
            None,
        );

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&gdef);
        let constraint = compile_owned(gdef, &vocab);

        assert_eq!(constraint.ignore_terminal, Some(1));
        assert_eq!(normalized.terminals.len(), 3);
        assert_eq!(normalized.ignore_terminal, Some(1));
        assert_eq!(
            normalized
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), " +".to_string(), "b".to_string()],
            "the dead terminal should be removed while the ignore terminal is preserved"
        );
        assert_eq!(
            normalized.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "live grammar terminals should be remapped around the retained ignore terminal"
        );

        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'b' should not be allowed before 'a'");
        assert!(mask_has_token(&mask, 3), "token ' a' should be allowed via ignore+terminal composition");
        assert!(!mask_has_token(&mask, 4), "token ' b' should not be allowed before 'a'");

        state.commit_token(3).unwrap();
        assert!(!state.is_finished(), "consuming ignored space plus 'a' should still leave trailing 'b'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should no longer be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should still be allowed between grammar terminals");
        assert!(mask_has_token(&mask, 2), "token 'b' should be allowed after 'a'");
        assert!(!mask_has_token(&mask, 3), "token ' a' should not be allowed once the grammar expects 'b'");
        assert!(mask_has_token(&mask, 4), "token ' b' should be allowed via ignore+terminal composition after 'a'");

        state.commit_token(4).unwrap();
        assert!(state.is_finished(), "consuming ignored space plus 'b' should finish the grammar");
    }

    #[test]
    fn test_prepare_grammar_for_compile_retains_and_remaps_names() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            nonterminal_names: std::collections::BTreeMap::from([(0, "start".to_string())]),
            terminal_names: std::collections::BTreeMap::from([
                (0, "A".to_string()),
                (1, "DEAD".to_string()),
                (2, "B".to_string()),
            ]),
            ignore_terminal: None,
        };

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&grammar);

        assert_eq!(normalized.nonterminal_names.get(&0).map(String::as_str), Some("start"));
        assert_eq!(normalized.terminal_names.get(&0).map(String::as_str), Some("A"));
        assert_eq!(normalized.terminal_names.get(&1).map(String::as_str), Some("B"));
        assert!(!normalized.terminal_names.values().any(|name| name == "DEAD"));
    }

    #[test]
    fn test_inline_single_use_nonterminals_compacts_repetition_tail_chain() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(1), Symbol::Terminal(1)],
            },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(4),
                    Symbol::Terminal(1),
                ],
            },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(7)],
            },
            Rule {
                lhs: 6,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 7,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(9)],
            },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
            },
            Rule {
                lhs: 9,
                rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 10,
                rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(2), Symbol::Nonterminal(8), Symbol::Terminal(4)],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "json_kv".to_string()),
            (2, "json_value".to_string()),
            (3, "json_object".to_string()),
            (10, "json_array".to_string()),
        ]);

        let protected: std::collections::BTreeSet<NonterminalID> = names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(!rules.iter().any(|rule| matches!(rule.lhs, 6 | 7)));
        assert!(rules.contains(&Rule {
            lhs: 5,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(1)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 9,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(9)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
        }));
    }

    #[test]
    fn test_inline_single_use_nonterminals_keeps_multi_symbol_helper_with_multiple_occurrences() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(1)],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "root".to_string()),
        ]);
        let protected: std::collections::BTreeSet<NonterminalID> = names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(rules.iter().any(|rule| rule.lhs == 2));
        assert!(rules.contains(&Rule {
            lhs: 1,
            rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 2,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
        }));
    }

    #[test]
    #[ignore = "fixture generation for kb814 tokenizer/equivalence benchmarking"]
    fn test_write_kb814_prepared_terminals_fixture() {
        let schema_path = kb814_normalized_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let grammar = json_schema_to_grammar(&schema_json).expect("kb814 schema should import");
        let (prepared_grammar, _tokenizer) = prepare_owned_grammar_for_compile(grammar);

        let terminals_path = kb814_prepared_terminals_path();
        let payload = serde_json::to_vec(&prepared_grammar.terminals)
            .expect("serialize prepared terminals");
        fs::write(&terminals_path, payload)
            .unwrap_or_else(|err| panic!("failed to write {}: {err}", terminals_path.display()));

        eprintln!(
            "[kb814] wrote_prepared_terminals path={} terminals={}",
            terminals_path.display(),
            prepared_grammar.terminals.len(),
        );
    }

    #[test]
    #[ignore = "kb814 tokenizer/equivalence timing benchmark"]
    fn test_kb814_prepared_terminals_gpt2_timings() {
        let terminals_path = kb814_prepared_terminals_path();
        let terminals_json = fs::read_to_string(&terminals_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", terminals_path.display()));
        let terminals: Vec<Terminal> = serde_json::from_str(&terminals_json)
            .expect("parse prepared terminals json");
        let vocab = load_gpt2_vocab();
        let grammar = GrammarDef {
            terminals,
            ..Default::default()
        };

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);
        eprintln!(
            "[kb814] terminals_file={} vocab_file={} terminals={} tokenizer_states={} build_tokenizer_ms={:.3}",
            terminals_path.display(),
            gpt2_vocab_path().display(),
            grammar.terminals.len(),
            tokenizer.num_states(),
            tokenizer_ms,
        );

        unsafe {
            std::env::set_var("GLRMASK_PROFILE_COMPILE", "1");
        }
        let equivalence_started_at = Instant::now();
        let id_map = analyze_equivalences(
            &tokenizer,
            &vocab,
            &std::collections::BTreeMap::new(),
            None,
            None,
        );
        let equivalence_ms = elapsed_ms(equivalence_started_at);
        eprintln!(
            "[kb814] tokenizer_state_classes={} vocab_classes={} equivalence_ms={:.3}",
            id_map.tokenizer_states.internal_to_originals.len(),
            id_map.vocab_tokens.internal_to_originals.len(),
            equivalence_ms,
        );
    }

    /// Regression test for o76439: structural import with nested closed objects
    /// must accept cross-token terminal matches (e.g., ` {"` after `,`).
    #[test]
    fn test_o76439_gpt2_vocab_false_negative() {
        let vocab = load_gpt2_vocab();

        // Actual o76439 schema
        let schema = r#"{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "ignoreSevertiesAtOrBelow": {
                    "type": "string",
                    "enum": ["negligible", "Negligible", "low", "Low",
                             "medium", "Medium", "high", "High"]
                },
                "vulnerabilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cveId": {"type": "string", "minLength": 1, "maxLength": 512},
                            "rationale": {"type": "string", "minLength": 1, "maxLength": 512}
                        },
                        "required": ["cveId", "rationale"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["ignoreSevertiesAtOrBelow"],
            "additionalProperties": false
        }"#;

        let c = crate::Constraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = c.start();

        // Commit the prefix (token positions 0..49 from the mismatch report)
        let prefix = b"{\"ignoreSevertiesAtOrBelow\": \"Medium\", \"vulnerabilities\": [{\"cveId\": \"CVE-2022-1234\", \"rationale\": \"This vulnerability is not applicable to our system.\"},";
        state.commit_bytes(prefix).expect("prefix should commit");

        let mut state_clone = state.clone();
        state_clone
            .commit_bytes(b" {\"")
            .expect("token bytes ` {\"` should commit after the array-item separator");

        let target_bytes = b" {\"";
        let target_token_id = vocab
            .entries
            .iter()
            .find(|(_, bytes)| bytes.as_slice() == target_bytes)
            .map(|(&id, _)| id)
            .expect("GPT-2 vocab must contain ` {\"`");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, target_token_id),
            "token {} (` {{\"`) must be in the mask — false negative regression (o76439)",
            target_token_id
        );
    }

}
