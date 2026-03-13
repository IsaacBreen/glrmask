#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::compile::build_regex;
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::debug::{AutomataDebug, CompileDebug, TerminalDebug};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Terminal};
use crate::compiler::grammar_def::{Rule, Symbol, TerminalID};
use crate::compiler::parser_dwa::build_parser_dwa_from_terminal_dwa;
use crate::compiler::possible_matches::build_possible_matches_by_state;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::parser_dwa::{
    ParserDwaBuildReport,
    build_parser_dwa_from_terminal_dwa_with_report,
};
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::Templates;
use crate::compiler::terminal_dwa::{TerminalDwaBuildReport, build_terminal_dwa, build_terminal_dwa_with_report};
use crate::compiler::stages::terminal_dwa::compute_ever_allowed_follows;
use crate::ds::bitset::BitSet;
use crate::runtime::Constraint;

/// Convert ever-allowed follow sets into the `BTreeMap<u32, BitSet>` disallowed-follows
/// format expected by the equivalence analysis.
fn compute_disallowed_follows(grammar: &AnalyzedGrammar) -> BTreeMap<u32, BitSet> {
    let ever_allowed = compute_ever_allowed_follows(grammar);
    let num_terminals = grammar.num_terminals as usize;
    let mut result = BTreeMap::new();
    for (tid, allowed) in ever_allowed.iter().enumerate() {
        let mut disallowed = BitSet::new(num_terminals);
        let allowed_set: std::collections::BTreeSet<u32> = allowed.iter().copied().collect();
        for other in 0..num_terminals {
            if !allowed_set.contains(&(other as u32)) {
                disallowed.set(other);
            }
        }
        if !disallowed.is_zero() {
            result.insert(tid as u32, disallowed);
        }
    }
    result
}

// ── Tokenizer construction ──────────────────────────────────────────────────

/// Build a [`Tokenizer`] from a [`GrammarDef`].
///
/// Each terminal is compiled through the NFA→DFA pipeline via [`build_regex`].
/// The group index matches the terminal ID (guaranteed by construction).
pub(crate) fn build_tokenizer(grammar: &GrammarDef) -> Tokenizer {
    let exprs: Vec<Expr> = grammar
        .terminals
        .iter()
        .map(|terminal| match terminal {
            Terminal::Literal { bytes, .. } => Expr::U8Seq(bytes.clone()),
            Terminal::Pattern { pattern, utf8, .. } => parse_regex(pattern, *utf8),
            Terminal::Expr { expr, .. } => expr.clone(),
        })
        .collect();
    let regex = build_regex(&exprs);
    Tokenizer {
        dfa: regex.dfa,
        num_terminals: grammar.num_terminals(),
    }
}

/// Build a [`Tokenizer`] from a slice of regex expressions.
///
/// Each expression's index becomes its `TerminalID`.
pub(crate) fn build_tokenizer_from_exprs(exprs: &[Expr]) -> Tokenizer {
    let num = exprs.len() as u32;
    let regex = build_regex(exprs);
    Tokenizer {
        dfa: regex.dfa,
        num_terminals: num,
    }
}

fn decode_literal_pattern(pattern: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(pattern.len());
    let bytes = pattern.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' && index + 1 < bytes.len() {
            index += 1;
            out.push(match bytes[index] {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                other => other,
            });
        } else {
            out.push(bytes[index]);
        }
        index += 1;
    }
    out
}

fn build_internal_token_bytes(vocab: &Vocab, id_map: &InternalIdMap) -> BTreeMap<u32, Vec<u8>> {
    id_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
        .filter_map(|(internal_token_id, original_ids)| {
            let representative = *original_ids.first()?;
            let bytes = vocab.entries.get(&representative)?.clone();
            Some((internal_token_id as u32, bytes))
        })
        .collect()
}

use crate::compiler::grammar::transforms::{expand_nullable_terminals, compact_unused_terminals, inline_single_use_nonterminals, compact_bounded_repeat_ladders, prepare_grammar_for_compile};


fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

fn compile_summary_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn log_compile_profile(enabled: bool, phase: &str, started_at: std::time::Instant) {
    if enabled {
        eprintln!(
            "[glrmask/profile][compile] {phase}_ms={:.3}",
            started_at.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn reduction_ratio(original: usize, reduced: usize) -> f64 {
    if reduced == 0 {
        0.0
    } else {
        original as f64 / reduced as f64
    }
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn log_compile_summary(
    normalized: &GrammarDef,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    prepare_grammar_time: std::time::Duration,
    analyze_grammar_time: std::time::Duration,
    build_glr_table_time: std::time::Duration,
    build_internal_id_map_time: std::time::Duration,
    collect_token_bytes_time: std::time::Duration,
    terminal_build: &TerminalDwaBuildReport,
    parser_build: &ParserDwaBuildReport,
    total_time: std::time::Duration,
) {
    eprintln!(
        "[glrmask/profile][summary] compile_ms={:.3} prepare_ms={:.3} analyze_ms={:.3} glr_table_ms={:.3} id_map_ms={:.3} token_bytes_ms={:.3} terminal_ms={:.3} parser_ms={:.3}",
        ms(total_time),
        ms(prepare_grammar_time),
        ms(analyze_grammar_time),
        ms(build_glr_table_time),
        ms(build_internal_id_map_time),
        ms(collect_token_bytes_time),
        ms(terminal_build.total_time),
        ms(parser_build.total_time),
    );
    eprintln!(
        "[glrmask/profile][summary] state_eq orig={} classes={} ratio={:.2}x | vocab_eq orig={} classes={} ratio={:.2}x",
        tokenizer.num_states(),
        id_map.tokenizer_states.num_internal_ids(),
        reduction_ratio(tokenizer.num_states() as usize, id_map.tokenizer_states.num_internal_ids() as usize),
        vocab.entries.len(),
        id_map.vocab_tokens.num_internal_ids(),
        reduction_ratio(vocab.entries.len(), id_map.vocab_tokens.num_internal_ids() as usize),
    );
    eprintln!(
        "[glrmask/profile][summary] terminal_nwa states={} start={} final={} eps={} labeled={} total={}",
        terminal_build.terminal_nwa.states,
        terminal_build.terminal_nwa.start_states,
        terminal_build.terminal_nwa.final_states,
        terminal_build.terminal_nwa.epsilon_edges,
        terminal_build.terminal_nwa.labeled_edges,
        terminal_build.terminal_nwa.transitions,
    );
    eprintln!(
        "[glrmask/profile][summary] terminal_dwa det states={} final={} trans={} det_ms={:.3} | min states={} final={} trans={} min_ms={:.3} collapse={} prune_ms={:.3}",
        terminal_build.terminal_dwa.states,
        terminal_build.terminal_dwa.final_states,
        terminal_build.terminal_dwa.transitions,
        ms(terminal_build.determinize_time),
        terminal_build.terminal_minimized_dwa.states,
        terminal_build.terminal_minimized_dwa.final_states,
        terminal_build.terminal_minimized_dwa.transitions,
        ms(terminal_build.minimize_time),
        if terminal_build.collapse_always_allowed_applied { "applied" } else { "skipped" },
        ms(terminal_build.prune_disallowed_follows_time),
    );
    eprintln!(
        "[glrmask/profile][summary] templates characterize_ms={:.3} terminals={} shifts={} reduces={} escapes={} rereduces={} | build_ms={:.3} templates={} total_states={} total_trans={} max_states={} max_trans={}",
        ms(parser_build.characterize_terminals_time),
        parser_build.characterizations.terminals,
        parser_build.characterizations.shifts,
        parser_build.characterizations.reduces,
        parser_build.characterizations.nt_escapes,
        parser_build.characterizations.nt_rereduces,
        ms(parser_build.build_templates_time),
        parser_build.templates.templates,
        parser_build.templates.total_states,
        parser_build.templates.total_transitions,
        parser_build.templates.max_states,
        parser_build.templates.max_transitions,
    );
    eprintln!(
        "[glrmask/profile][summary] bundles states={} total={} unique={} unique_targets={} cache_hits={} group_ms={:.3} build_ms={:.3}",
        terminal_build.terminal_minimized_dwa.states,
        parser_build.bundles.total_bundles,
        parser_build.bundles.unique_bundles,
        parser_build.bundles.unique_bundle_targets,
        parser_build.bundles.bundle_cache_hits,
        ms(parser_build.bundles.group_targets_time),
        ms(parser_build.bundles.build_bundle_time),
    );
    eprintln!(
        "[glrmask/profile][summary] parser_nwa compose_ms={:.3} pre states={} start={} final={} eps={} labeled={} total={} | post_resolve states={} final={} eps={} labeled={} total={} neg={}→{} default={}→{} resolve_ms={:.3}",
        ms(parser_build.compose_state_time),
        parser_build.parser_nwa_before_resolve.states,
        parser_build.parser_nwa_before_resolve.start_states,
        parser_build.parser_nwa_before_resolve.final_states,
        parser_build.parser_nwa_before_resolve.epsilon_edges,
        parser_build.parser_nwa_before_resolve.labeled_edges,
        parser_build.parser_nwa_before_resolve.transitions,
        parser_build.parser_nwa_after_resolve.states,
        parser_build.parser_nwa_after_resolve.final_states,
        parser_build.parser_nwa_after_resolve.epsilon_edges,
        parser_build.parser_nwa_after_resolve.labeled_edges,
        parser_build.parser_nwa_after_resolve.transitions,
        parser_build.negative_edges_before_resolve,
        parser_build.negative_edges_after_resolve,
        parser_build.default_edges_before_resolve,
        parser_build.default_edges_after_resolve,
        ms(parser_build.resolve_negative_codes_time),
    );
    eprintln!(
        "[glrmask/profile][summary] parser_dwa pre_min states={} final={} trans={} | post_min states={} final={} trans={} detmin_ms={:.3} subtract_final_ms={:.3} rules={} terminals={} tokenizer_states={} vocab_entries={}",
        parser_build.parser_dwa_pre_minimize.states,
        parser_build.parser_dwa_pre_minimize.final_states,
        parser_build.parser_dwa_pre_minimize.transitions,
        parser_build.parser_dwa_minimized.states,
        parser_build.parser_dwa_minimized.final_states,
        parser_build.parser_dwa_minimized.transitions,
        ms(parser_build.determinize_minimize_time),
        ms(parser_build.subtract_final_weights_time),
        normalized.rules.len(),
        normalized.terminals.len(),
        tokenizer.num_states(),
        vocab.entries.len(),
    );
}

pub fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    let profile_enabled = compile_profile_enabled();
    let summary_enabled = compile_summary_enabled();
    let compile_started_at = std::time::Instant::now();

    let phase_started_at = std::time::Instant::now();
    let (normalized, tokenizer) = prepare_grammar_for_compile(grammar);
    let prepare_grammar_time = phase_started_at.elapsed();
    log_compile_profile(profile_enabled, "prepare_grammar", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    let glr_grammar = AnalyzedGrammar::from_grammar_def(&normalized);
    let analyze_grammar_time = phase_started_at.elapsed();
    log_compile_profile(profile_enabled, "analyze_grammar", phase_started_at);

    // Debug check: verify grammar preconditions before expensive pipeline stages.
    // Violations here indicate the grammar (or its normalization) has shapes that
    // will cause panics or incorrect results later in the pipeline.
    #[cfg(debug_assertions)]
    if let Err(msg) = glr_grammar.debug_check_grammar_preconditions() {
        panic!("[glrmask] grammar precondition violations:\n{}", msg);
    }

    let phase_started_at = std::time::Instant::now();
    let table = GLRTable::build(&glr_grammar);
    let build_glr_table_time = phase_started_at.elapsed();
    log_compile_profile(profile_enabled, "build_glr_table", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    let disallowed_follows = compute_disallowed_follows(&glr_grammar);
    let mut id_map = InternalIdMap::build(&tokenizer, vocab, &disallowed_follows, normalized.ignore_terminal);
    let build_internal_id_map_time = phase_started_at.elapsed();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_internal_id_map_ms={:.3} states={}→{}_tsids ({:.1}x) vocab={}→{}_classes ({:.1}x)",
            ms(build_internal_id_map_time),
            tokenizer.num_states(),
            id_map.num_tsids(),
            tokenizer.num_states() as f64 / id_map.num_tsids().max(1) as f64,
            vocab.entries.len(),
            id_map.num_internal_tokens(),
            vocab.entries.len() as f64 / id_map.num_internal_tokens().max(1) as f64,
        );
    }

    let phase_started_at = std::time::Instant::now();
    let token_bytes = vocab.entries.clone();
    let collect_token_bytes_time = phase_started_at.elapsed();
    log_compile_profile(profile_enabled, "collect_token_bytes", phase_started_at);

    let (mut terminal_dwa, terminal_build) = build_terminal_dwa_with_report(
        &glr_grammar,
        &tokenizer,
        vocab,
        &id_map,
        normalized.ignore_terminal,
    );
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_terminal_dwa_ms={:.3}",
            ms(terminal_build.total_time),
        );
    }

    // Compact: merge equivalent IDs and reorder for range adjacency
    let phase_started_at = std::time::Instant::now();
    let compact_report = crate::compiler::stages::compact::compact_dwa_dimensions(
        &mut terminal_dwa,
        &mut id_map,
    );
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] compact_terminal_ms={:.3} tsids={}→{} tokens={}→{} ranges={}→{}",
            ms(phase_started_at.elapsed()),
            compact_report.old_num_tsids,
            compact_report.new_num_tsids,
            compact_report.old_num_tokens,
            compact_report.new_num_tokens,
            compact_report.old_ranges,
            compact_report.new_ranges,
        );
    }

    // Build internal_token_bytes and possible_matches with post-compact id_map
    let phase_started_at = std::time::Instant::now();
    let internal_token_bytes = build_internal_token_bytes(vocab, &id_map);
    let possible_matches = build_possible_matches_by_state(&tokenizer, &internal_token_bytes);
    log_compile_profile(profile_enabled, "build_possible_matches", phase_started_at);

    let (parser_dwa, parser_build) = build_parser_dwa_from_terminal_dwa_with_report(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
    );
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_parser_dwa_ms={:.3}",
            ms(parser_build.total_time),
        );
    }

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] total_ms={:.3} rules={} terminals={} vocab_entries={} tokenizer_states={} internal_tsids={} glr_table_states={} terminal_dwa_states={} parser_dwa_states={}",
            compile_started_at.elapsed().as_secs_f64() * 1000.0,
            normalized.rules.len(),
            normalized.terminals.len(),
            vocab.entries.len(),
            tokenizer.num_states(),
            id_map.num_tsids(),
            table.num_states,
            terminal_dwa.num_states(),
            parser_dwa.num_states(),
        );
    }

    if summary_enabled {
        log_compile_summary(
            &normalized,
            &tokenizer,
            vocab,
            &id_map,
            prepare_grammar_time,
            analyze_grammar_time,
            build_glr_table_time,
            build_internal_id_map_time,
            collect_token_bytes_time,
            &terminal_build,
            &parser_build,
            compile_started_at.elapsed(),
        );
    }

    let mut constraint = Constraint {
        parser_dwa,
        table,
        tokenizer,
        ignore_terminal: normalized.ignore_terminal,
        possible_matches,
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals.clone(),
        original_token_to_internal: id_map.vocab_tokens.original_to_internal.clone(),
        internal_token_to_tokens: id_map.vocab_tokens.internal_to_originals.clone(),
        eos_token_id: vocab.eos_token_id,
        token_bytes,
        internal_token_bytes,
        internal_token_buf_masks: Vec::new(),
        internal_token_dense_words: 0,
        weight_token_dense_masks: rustc_hash::FxHashMap::default(),
    };
    constraint.build_buf_masks();
    constraint.build_dense_token_masks();
    constraint
}

pub(crate) fn compile_with_debug(grammar: &GrammarDef, vocab: &Vocab) -> (Constraint, CompileDebug) {
    let (normalized, tokenizer) = prepare_grammar_for_compile(grammar);

    let glr_grammar = AnalyzedGrammar::from_grammar_def(&normalized);

    #[cfg(debug_assertions)]
    if let Err(msg) = glr_grammar.debug_check_grammar_preconditions() {
        panic!("[glrmask] grammar precondition violations:\n{}", msg);
    }

    let table = GLRTable::build(&glr_grammar);
    let disallowed_follows = compute_disallowed_follows(&glr_grammar);
    let id_map = InternalIdMap::build(&tokenizer, vocab, &disallowed_follows, normalized.ignore_terminal);

    let internal_token_bytes = build_internal_token_bytes(vocab, &id_map);
    let possible_matches_by_state = build_possible_matches_by_state(&tokenizer, &internal_token_bytes);

    let characterizations = characterize_terminals(&table, &glr_grammar);
    let templates = Templates::from_characterizations(&characterizations);
    let terminal_dwa = build_terminal_dwa(
        &glr_grammar,
        &tokenizer,
        vocab,
        &id_map,
        normalized.ignore_terminal,
    );
    let parser_dwa = build_parser_dwa_from_terminal_dwa(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
    );

    let vocab_entries: Vec<(u32, Vec<u8>)> = vocab.entries.iter().map(|(token_id, bytes)| (*token_id, bytes.clone())).collect();
    let token_bytes = vocab.entries.clone();
    let mut constraint = Constraint {
        parser_dwa: parser_dwa.clone(),
        table: table.clone(),
        tokenizer: tokenizer.clone(),
        ignore_terminal: normalized.ignore_terminal,
        possible_matches: possible_matches_by_state.clone(),
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals.clone(),
        original_token_to_internal: id_map.vocab_tokens.original_to_internal.clone(),
        internal_token_to_tokens: id_map.vocab_tokens.internal_to_originals.clone(),
        eos_token_id: vocab.eos_token_id,
        token_bytes: token_bytes.clone(),
        internal_token_bytes,
        internal_token_buf_masks: Vec::new(),
        internal_token_dense_words: 0,
        weight_token_dense_masks: rustc_hash::FxHashMap::default(),
    };
    constraint.build_buf_masks();
    constraint.build_dense_token_masks();

    let debug = CompileDebug::from_parts(
        grammar.clone(),
        normalized.clone(),
        glr_grammar.clone(),
        table.clone(),
        AutomataDebug {
            characterizations,
            terminal_dwa: terminal_dwa.clone(),
            terminal_debug: TerminalDebug {
                nwa_after_build: NWA::new(0, 0),
                nwa_after_collapse: NWA::new(0, 0),
            },
            templates,
            parser_nwa_before_resolve: NWA::new(0, 0),
            parser_nwa_after_resolve: NWA::new(0, 0),
            parser_dwa_pre_minimize: parser_dwa.clone(),
            parser_dwa: parser_dwa.clone(),
            id_map: id_map.clone(),
        },
        vocab_entries,
        vocab.eos_token_id,
    );

    (constraint, debug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::tests::*;
    use crate::compiler::grammar::model::{Rule, Symbol, Terminal};

    fn mask_has_token(mask: &[u32], token: u32) -> bool {
        let word = token as usize / 32;
        let bit = token as usize % 32;
        word < mask.len() && (mask[word] & (1u32 << bit)) != 0
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
            .values()
            .flat_map(|token_ids| token_ids.iter())
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
    fn test_end_to_end_simple_ab() {
        
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0);
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1);
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
            .commit_token(0);
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
            .commit_token(0);
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1);
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
            .commit_token(0);
        assert!(!state.is_finished(), "not accepting after 'a'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1);
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

        state.commit_token(0);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2);
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

        state.commit_token(0);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1);

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2);
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

        state.commit_token(0);
        assert!(
            !state.is_finished(),
            "the shorter literal 'a' should not complete a grammar expecting 'ab'"
        );

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token 'b' should remain allowed as a continuation of the longer literal 'ab'"
        );

        state.commit_token(1);
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

        let nullable = tok.drain_nullable_terminals();
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
        let mut state = constraint.start();
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

        let (constraint, debug) = compile_with_debug(&gdef, &vocab);

        assert_eq!(
            constraint.tokenizer.num_terminals,
            2,
            "the final tokenizer should be built only from the live compacted terminals"
        );
        assert_eq!(debug.normalized_grammar_def.terminals.len(), 2);
        assert_eq!(
            debug
                .normalized_grammar_def
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()],
            "the dead middle terminal should be absent from the normalized grammar"
        );
        assert_eq!(
            debug.normalized_grammar_def.rules,
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

        state.commit_token(0);
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

        let (constraint, debug) = compile_with_debug(&gdef, &vocab);

        assert_eq!(constraint.ignore_terminal, Some(1));
        assert_eq!(debug.normalized_grammar_def.terminals.len(), 3);
        assert_eq!(debug.normalized_grammar_def.ignore_terminal, Some(1));
        assert_eq!(
            debug
                .normalized_grammar_def
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), " +".to_string(), "b".to_string()],
            "the dead terminal should be removed while the ignore terminal is preserved"
        );
        assert_eq!(
            debug.normalized_grammar_def.rules,
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

        state.commit_token(3);
        assert!(!state.is_finished(), "consuming ignored space plus 'a' should still leave trailing 'b'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should no longer be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should still be allowed between grammar terminals");
        assert!(mask_has_token(&mask, 2), "token 'b' should be allowed after 'a'");
        assert!(!mask_has_token(&mask, 3), "token ' a' should not be allowed once the grammar expects 'b'");
        assert!(mask_has_token(&mask, 4), "token ' b' should be allowed via ignore+terminal composition after 'a'");

        state.commit_token(4);
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

    fn derivable_chunk_counts(
        rules: &[Rule],
        nonterminal: NonterminalID,
        chunk_nt: NonterminalID,
        memo: &mut std::collections::BTreeMap<NonterminalID, std::collections::BTreeSet<usize>>,
    ) -> std::collections::BTreeSet<usize> {
        if let Some(existing) = memo.get(&nonterminal) {
            return existing.clone();
        }
        if nonterminal == chunk_nt {
            return std::collections::BTreeSet::from([1]);
        }

        let mut result = std::collections::BTreeSet::new();
        for rule in rules.iter().filter(|rule| rule.lhs == nonterminal) {
            let mut totals = std::collections::BTreeSet::from([0usize]);
            for symbol in &rule.rhs {
                let counts = match symbol {
                    Symbol::Terminal(_) => std::collections::BTreeSet::from([0usize]),
                    Symbol::Nonterminal(id) => derivable_chunk_counts(rules, *id, chunk_nt, memo),
                };
                let mut next_totals = std::collections::BTreeSet::new();
                for left in &totals {
                    for right in &counts {
                        next_totals.insert(left + right);
                    }
                }
                totals = next_totals;
            }
            result.extend(totals);
        }

        memo.insert(nonterminal, result.clone());
        result
    }

    #[test]
    fn test_compact_bounded_repeat_ladders_rewrites_linear_family() {
        let chunk_nt = 20;
        let family = [30, 31, 32, 33, 34, 35, 36, 37];
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(family[0])],
            },
            Rule {
                lhs: chunk_nt,
                rhs: vec![Symbol::Terminal(0)],
            },
        ];
        for (index, lhs) in family.iter().copied().enumerate() {
            rules.push(Rule {
                lhs,
                rhs: vec![Symbol::Nonterminal(chunk_nt)],
            });
            let long_rhs = if index + 1 == family.len() {
                vec![Symbol::Nonterminal(chunk_nt), Symbol::Nonterminal(chunk_nt)]
            } else {
                vec![Symbol::Nonterminal(chunk_nt), Symbol::Nonterminal(family[index + 1])]
            };
            rules.push(Rule { lhs, rhs: long_rhs });
        }

        let original_family_rule_count = rules.iter().filter(|rule| family.contains(&rule.lhs)).count();
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "root".to_string()),
        ]);

        compact_bounded_repeat_ladders(&mut rules, 0, &names);

        let rewritten_family_rule_count = rules.iter().filter(|rule| family.contains(&rule.lhs)).count();
        assert!(rewritten_family_rule_count < original_family_rule_count);
        assert_eq!(rules.iter().filter(|rule| rule.lhs == family[0]).count(), 3);

        let mut memo = std::collections::BTreeMap::new();
        let counts = derivable_chunk_counts(&rules, family[0], chunk_nt, &mut memo);
        assert_eq!(counts, (1..=family.len() + 1).collect::<std::collections::BTreeSet<_>>());
    }

}
