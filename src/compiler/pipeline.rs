use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
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
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals;
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

fn set_dense_bit(words: &mut [u64], token_id: u32) {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    if let Some(slot) = words.get_mut(word) {
        *slot |= 1u64 << bit;
    }
}
fn remap_parser_dwa_to_constraint_vocab(
    parser_dwa: &mut DWA,
    old_internal_to_constraint: &[Vec<u32>],
) {
    let mut token_set_cache = FxHashMap::default();

    for state in parser_dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = cpm::remap_weight_to_constraint_vocab(
                final_weight,
                old_internal_to_constraint,
                &mut token_set_cache,
            );
        }

        for (_, weight) in state.transitions.values_mut() {
            *weight = cpm::remap_weight_to_constraint_vocab(
                weight,
                old_internal_to_constraint,
                &mut token_set_cache,
            );
        }
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

                if let Err(message) = analyzed_grammar.debug_check_grammar_preconditions() {
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
        let _terminal_path_lengths = classify_terminal_path_lengths(
            &tokenizer,
            vocab,
            &adjusted_disallowed_for_classification,
            analyzed_grammar.num_terminals,
            Some(&shared_classify_cache),
        );
        profile.classify_ms = elapsed_ms(classify_started_at);

        let (((internal_ids, terminal_dwa, terminal_phase_profile, global_max_length_state_map), _id_map_wall_ms), (templates, templates_ms)) = rayon::join(
            || {
                let id_map_started_at = Instant::now();
                let result = crate::compiler::stages::id_map_and_terminal_dwa::build_id_map_and_terminal_dwa(
                    &tokenizer,
                    vocab,
                    &terminal_coloring,
                    true,
                    prepared_grammar.ignore_terminal,
                    &analyzed_grammar,
                    &adjusted_disallowed_for_classification,
                    Some(&shared_classify_cache),
                );
                (result, elapsed_ms(id_map_started_at))
            },
            || {
                let templates_started_at = Instant::now();
                let characterizations = characterize_terminals(&table, &analyzed_grammar);
                let templates = Templates::from_characterizations(&characterizations);
                (templates, elapsed_ms(templates_started_at))
            },
        );
        let global_max_length_state_map_ref = Some(&global_max_length_state_map);
        profile.templates_ms = templates_ms;
        profile.id_map_ms = terminal_phase_profile.id_map_ms;
        profile.terminal_dwa_ms = terminal_phase_profile.terminal_dwa_ms;
        profile.compact_ms = terminal_phase_profile.compact_ms;

        let token_bytes = vocab.entries.clone();

        // Compute constraint possible matches and build parser DWA
        // concurrently. Parser DWA does not depend on possible matches
        // or constraint-vocab; only the later remap step does.
        let (cpm_result, (mut parser_dwa, parser_dwa_ms)) = rayon::join(
            || {
                cpm::compute_constraint_possible_matches(
                    &tokenizer,
                    &token_bytes,
                    &internal_ids,
                    cpm::ConstraintPossibleMatchesConfig {
                        initial_state_map: global_max_length_state_map_ref,
                    },
                )
            },
            || {
                let parser_dwa_started_at = Instant::now();
                let parser_dwa =
                    build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                        &table,
                        &analyzed_grammar,
                        &terminal_dwa,
                        templates,
                        vocab,
                        &internal_ids,
                    );
                (parser_dwa, elapsed_ms(parser_dwa_started_at))
            },
        );

        let constraint_vocab = cpm_result.constraint_vocab;
        let possible_matches = cpm_result.possible_matches;
        let cpm_profile = cpm_result.profile;

        let remap_parser_dwa_started_at = Instant::now();
        let remap_parser_dwa_ms = if cpm::constraint_vocab_is_identity(&constraint_vocab) {
            0.0
        } else {
            remap_parser_dwa_to_constraint_vocab(
                &mut parser_dwa,
                &constraint_vocab.old_internal_to_constraint,
            );
            elapsed_ms(remap_parser_dwa_started_at)
        };

        let internal_token_bytes_started_at = Instant::now();
        let internal_token_bytes = cpm::build_internal_token_bytes_from_groups(
            vocab,
            &constraint_vocab.internal_to_originals,
        );
        let internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

        profile.parser_dwa_ms = parser_dwa_ms;
        profile.permute_possible_matches_ms =
            cpm_profile.possible_matches_collect_ms + cpm_profile.constraint_vocab_ms + remap_parser_dwa_ms;
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
            original_token_to_internal: constraint_vocab.original_to_internal,
            internal_token_to_tokens: constraint_vocab.internal_to_originals,
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
            seed_state_dense: Vec::new(),
            seed_universe_dense: Box::new([]),
            dwa_fast_transitions: Vec::new(),
            heavy_token_dense_masks: Vec::new(),
            heavy_token_indices: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::compiler::stages::equiv_types::ManyToOneIdMap;
    use crate::ds::weight::Weight;
    use crate::runtime::Constraint;

    fn bitmap(tokens: &[u32], token_slots: u32) -> Box<[u64]> {
        let mut words = vec![0u64; cpm::dense_word_count(token_slots)];
        for &token in tokens {
            set_dense_bit(&mut words, token);
        }
        words.into_boxed_slice()
    }

    fn brute_force_seed_state_signatures(
        tokenizer: &Tokenizer,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) -> FxHashMap<u32, Vec<u32>> {
        let mut signatures: FxHashMap<u32, Vec<u32>> = token_bytes
            .keys()
            .map(|&token_id| (token_id, Vec::new()))
            .collect();

        let terminal_states: Vec<bool> = (0..tokenizer.num_states())
            .map(|state| tokenizer.matched_terminals_iter(state).next().is_some())
            .collect();

        for tokenizer_state in 0..tokenizer.num_states() {
            if terminal_states[tokenizer_state as usize] {
                continue;
            }
            for (&token_id, bytes) in token_bytes {
                let exec = tokenizer.execute_from_state(bytes, tokenizer_state);
                if exec.end_state.is_none() && exec.matches.is_empty() {
                    continue;
                }

                signatures
                    .get_mut(&token_id)
                    .expect("token should have a seed signature slot")
                    .push(tokenizer_state);
            }
        }

        signatures
    }

    #[test]
    fn constraint_vocab_refines_parser_token_class_by_possible_matches() {
        let parser_vocab = ManyToOneIdMap::from_original_to_internal_with_representatives(
            vec![0, 0, 1],
            2,
            vec![0, 2],
        );

        let token_bytes = BTreeMap::from([
            (0u32, b"a".to_vec()),
            (1u32, b"b".to_vec()),
            (2u32, b"c".to_vec()),
        ]);

        let mut terminals = BTreeMap::new();
        terminals.insert(10u32, bitmap(&[0], 3));
        terminals.insert(11u32, bitmap(&[1], 3));

        let raw_possible_matches = BTreeMap::from([(5u32, terminals)]);

        let tokens_with_same_bytes = cpm::build_tokens_with_same_bytes(&token_bytes);
        let signatures = cpm::build_possible_match_signatures(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let signature_ids = cpm::intern_signature_ids(signatures);
        let seed_state_signatures: FxHashMap<u32, Vec<u32>> = token_bytes
            .keys()
            .map(|&token_id| (token_id, Vec::new()))
            .collect();
        let seed_state_signature_ids = cpm::intern_signature_ids(seed_state_signatures);
        let constraint_vocab = cpm::build_constraint_vocab_map(
            &parser_vocab,
            &token_bytes,
            &signature_ids,
            &seed_state_signature_ids,
        );

        let tok0 = constraint_vocab.original_to_internal[0];
        let tok1 = constraint_vocab.original_to_internal[1];
        let tok2 = constraint_vocab.original_to_internal[2];

        assert_ne!(tok0, tok1);
        assert_eq!(constraint_vocab.old_internal_to_constraint[0].len(), 2);
        assert_eq!(constraint_vocab.old_internal_to_constraint[1], vec![tok2]);

        let remapped_pm = cpm::remap_possible_matches_to_constraint_vocab(
            raw_possible_matches,
            &constraint_vocab.original_to_internal,
            constraint_vocab.internal_to_originals.len() as u32,
            &tokens_with_same_bytes,
        );

        let terminal_10 = remapped_pm[&10].tokens_for_tsid(5);
        let terminal_11 = remapped_pm[&11].tokens_for_tsid(5);

        assert!(terminal_10.contains(tok0));
        assert!(!terminal_10.contains(tok1));
        assert!(!terminal_11.contains(tok0));
        assert!(terminal_11.contains(tok1));
    }

    #[test]
    fn parser_weight_remap_expands_old_parser_token_to_all_constraint_splits() {
        let parser_vocab = ManyToOneIdMap::from_original_to_internal_with_representatives(
            vec![0, 0, 1],
            2,
            vec![0, 2],
        );

        let token_bytes = BTreeMap::from([
            (0u32, b"a".to_vec()),
            (1u32, b"b".to_vec()),
            (2u32, b"c".to_vec()),
        ]);

        let mut terminals = BTreeMap::new();
        terminals.insert(10u32, bitmap(&[0], 3));
        terminals.insert(11u32, bitmap(&[1], 3));

        let raw_possible_matches = BTreeMap::from([(5u32, terminals)]);

        let tokens_with_same_bytes = cpm::build_tokens_with_same_bytes(&token_bytes);
        let signatures = cpm::build_possible_match_signatures(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let signature_ids = cpm::intern_signature_ids(signatures);
        let seed_state_signatures: FxHashMap<u32, Vec<u32>> = token_bytes
            .keys()
            .map(|&token_id| (token_id, Vec::new()))
            .collect();
        let seed_state_signature_ids = cpm::intern_signature_ids(seed_state_signatures);
        let constraint_vocab = cpm::build_constraint_vocab_map(
            &parser_vocab,
            &token_bytes,
            &signature_ids,
            &seed_state_signature_ids,
        );

        let tok0 = constraint_vocab.original_to_internal[0];
        let tok1 = constraint_vocab.original_to_internal[1];
        let tok2 = constraint_vocab.original_to_internal[2];

        let old_weight = Weight::from_uniform(123..=123, RangeSetBlaze::from_iter([0u32..=0u32]));

        let mut cache = FxHashMap::default();
        let new_weight = cpm::remap_weight_to_constraint_vocab(
            &old_weight,
            &constraint_vocab.old_internal_to_constraint,
            &mut cache,
        );

        let new_set = new_weight.0.get(123).expect("tsid weight should remain");

        assert!(new_set.contains(tok0));
        assert!(new_set.contains(tok1));
        assert!(!new_set.contains(tok2));
    }

    #[test]
    fn seed_state_signatures_match_bruteforce_across_shared_prefixes() {
        let vocab = Vocab::new(
            vec![
                (0, b"ab".to_vec()),
                (1, b"ac".to_vec()),
            ],
            None,
        );
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                nt start ::= "ab" | "ac";
            "#,
            &vocab,
        )
        .unwrap();
        let token_bytes = BTreeMap::from([
            (0u32, b"ab".to_vec()),
            (1u32, b"ac".to_vec()),
        ]);
        let tokens_with_same_bytes = cpm::build_tokens_with_same_bytes(&token_bytes);
        let actual = cpm::build_seed_state_signatures_trie(
            &constraint.tokenizer,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let expected = brute_force_seed_state_signatures(&constraint.tokenizer, &token_bytes);

        assert_eq!(actual, expected);
    }

    #[test]
    fn seed_state_signatures_match_bruteforce_on_o43234_glrm_mre() {
        let vocab = Vocab::new(
            vec![
                (1, b"\"".to_vec()),
                (11, b",".to_vec()),
                (25, b":".to_vec()),
                (220, b" ".to_vec()),
                (313, b"--".to_vec()),
            ],
            None,
        );
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                internal t JSON_STRING_CHAR ::= /[^\x00-\x1f\x7f"\\]/;
                t JSON_STRING_BODY ::= JSON_STRING_CHAR* "\"";
                nt json_string ::= "\"" JSON_STRING_BODY;
                t JSON_INTEGER ::= /-?(0|[1-9][0-9]*)/;
                nt start ::= "{" ("\"" "a\"" ": " json_string) (", \"" "b\"" ": " JSON_INTEGER) "}";
            "#,
            &vocab,
        )
        .unwrap();
        let token_bytes = BTreeMap::from([
            (1u32, b"\"".to_vec()),
            (11u32, b",".to_vec()),
            (25u32, b":".to_vec()),
            (220u32, b" ".to_vec()),
            (313u32, b"--".to_vec()),
        ]);
        let tokens_with_same_bytes = cpm::build_tokens_with_same_bytes(&token_bytes);

        let actual = cpm::build_seed_state_signatures_trie(
            &constraint.tokenizer,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let expected = brute_force_seed_state_signatures(&constraint.tokenizer, &token_bytes);

        assert_eq!(actual, expected);
    }

    fn with_env_var<T>(name: &'static str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        use std::ffi::OsString;
        use std::sync::OnceLock;

        static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();

        struct EnvRestoreGuard {
            name: &'static str,
            old_value: Option<OsString>,
        }

        impl Drop for EnvRestoreGuard {
            fn drop(&mut self) {
                match self.old_value.take() {
                    Some(old_value) => unsafe { std::env::set_var(self.name, old_value); },
                    None => unsafe { std::env::remove_var(self.name); },
                }
            }
        }

        let lock = ENV_LOCK.get_or_init(|| std::sync::Mutex::new(()));
        let _lock_guard = lock.lock().expect("env lock poisoned");
        let _restore_guard = EnvRestoreGuard {
            name,
            old_value: std::env::var_os(name),
        };

        match value {
            Some(value) => unsafe { std::env::set_var(name, value); },
            None => unsafe { std::env::remove_var(name); },
        }

        f()
    }

    #[test]
    fn brute_force_possible_matches_pipeline_smoke() {
        with_env_var("GLRMASK_PM_BRUTE_FORCE", Some("1"), || {
            let vocab = Vocab::new(
                vec![
                    (0, b"a".to_vec()),
                    (1, b"b".to_vec()),
                    (2, b"ab".to_vec()),
                ],
                None,
            );
            let constraint = Constraint::from_glrm_grammar(
                r#"
                    start start;
                    nt start ::= "a" | "b" | "ab";
                "#,
                &vocab,
            )
            .unwrap();

            let initial_state = constraint.tokenizer.initial_state();
            let possible_matches = constraint.possible_matches_for_state(initial_state);
            assert!(!possible_matches.is_empty());
            assert!(
                possible_matches.values().any(|tokens| !tokens.is_empty()),
                "brute-force pipeline should produce non-empty possible matches"
            );
            assert_eq!(constraint.original_token_to_internal.len(), 3);
        });
    }
}