use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;

use once_cell::sync::Lazy;
use range_set_blaze::{RangeMapBlaze, RangeSetBlaze};
use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::regex::parse_regex;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{GLRTable, emit_glr_table_debug_dump};
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    SharedClassifyCache,
    classify_terminal_path_lengths,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::{
    compute_ever_allowed_follows,
    compute_terminal_coloring,
};
use crate::compiler::stages::id_map_and_terminal_dwa::maybe_print_terminal_mappings;
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    debug_terminal_mapping_enabled, TerminalColoring,
};
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::compile_dfa::{emit_template_profile_summary, emit_templates_debug_dump};
use crate::compiler::stages::terminal_dwa_compat::build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring;
use crate::ds::bitset::BitSet;
use crate::ds::weight::{Weight, clear_weight_op_profile, emit_weight_op_profile_summary, finalize_weight_map, shared_rangeset};
use crate::grammar::flat::{GrammarDef, Terminal, TerminalID};
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

fn debug_compile_stages_enabled() -> bool {
    env_flag_enabled("GLRMASK_DEBUG_COMPILE_STAGES")
}

fn strict_one_flag_enabled(name: &str) -> bool {
    std::env::var(name).map_or(false, |value| value == "1")
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

type DensePossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>;
type PossibleMatchSignature = Vec<(u32, TerminalID)>;
type SeedStateSignature = Vec<u32>;
type SignatureClassId = u32;

#[derive(Debug)]
struct ConstraintVocabMap {
    original_to_internal: Vec<u32>,
    internal_to_originals: Vec<Vec<u32>>,
    old_internal_to_constraint: Vec<Vec<u32>>,
}

fn build_internal_token_bytes_from_groups(
    vocab: &Vocab,
    internal_to_originals: &[Vec<u32>],
) -> BTreeMap<u32, Vec<u8>> {
    internal_to_originals
        .iter()
        .enumerate()
        .filter_map(|(internal_token_id, originals)| {
            let bytes = originals
                .iter()
                .find_map(|original| vocab.entries.get(original))?
                .clone();

            Some((internal_token_id as u32, bytes))
        })
        .collect()
}

fn dense_word_count(token_slots: u32) -> usize {
    (token_slots as usize + 63) / 64
}

fn max_original_token_slot(token_bytes: &BTreeMap<u32, Vec<u8>>) -> u32 {
    token_bytes
        .keys()
        .next_back()
        .map(|token_id| token_id.saturating_add(1))
        .unwrap_or(0)
}

fn set_dense_bit(words: &mut [u64], token_id: u32) {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    if let Some(slot) = words.get_mut(word) {
        *slot |= 1u64 << bit;
    }
}

fn dense_bit_is_set(words: &[u64], token_id: u32) -> bool {
    let word = token_id as usize / 64;
    let bit = token_id % 64;

    words
        .get(word)
        .map(|word| ((*word >> bit) & 1) != 0)
        .unwrap_or(false)
}

fn for_each_dense_bit(words: &[u64], mut f: impl FnMut(u32)) {
    for (word_idx, &word) in words.iter().enumerate() {
        let mut bits = word;

        while bits != 0 {
            let bit = bits.trailing_zeros();
            let token_id = word_idx as u32 * 64 + bit;
            f(token_id);
            bits &= bits - 1;
        }
    }
}

fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
    let Some((&first, rest)) = ids.split_first() else {
        return RangeSetBlaze::new();
    };

    let mut ranges = Vec::new();
    let mut start = first;
    let mut end = first;

    for &id in rest {
        if id == end + 1 {
            end = id;
        } else {
            ranges.push(start..=end);
            start = id;
            end = id;
        }
    }

    ranges.push(start..=end);
    RangeSetBlaze::from_iter(ranges)
}

fn build_tokens_with_same_bytes(token_bytes: &BTreeMap<u32, Vec<u8>>) -> FxHashMap<u32, Arc<[u32]>> {
    let mut by_bytes: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
    for (&token_id, bytes) in token_bytes {
        by_bytes.entry(bytes.clone()).or_default().push(token_id);
    }

    let mut tokens_with_same_bytes = FxHashMap::default();
    for (_, mut token_ids) in by_bytes {
        token_ids.sort_unstable();
        token_ids.dedup();
        let shared: Arc<[u32]> = Arc::from(token_ids.clone());
        for &token_id in &token_ids {
            tokens_with_same_bytes.insert(token_id, Arc::clone(&shared));
        }
    }

    tokens_with_same_bytes
}

fn build_possible_match_signatures(
    raw_possible_matches: &DensePossibleMatchesByState,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, PossibleMatchSignature> {
    let mut signatures: FxHashMap<u32, PossibleMatchSignature> = token_bytes
        .keys()
        .map(|&token_id| (token_id, Vec::new()))
        .collect();

    for (&original_tokenizer_state, terminals) in raw_possible_matches {
        for (&terminal_id, bitmap) in terminals {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(equivalent_tokens) = tokens_with_same_bytes.get(&original_token_id) else {
                    return;
                };

                for &equivalent_token_id in equivalent_tokens.iter() {
                    if let Some(signature) = signatures.get_mut(&equivalent_token_id) {
                        signature.push((original_tokenizer_state, terminal_id));
                    }
                }
            });
        }
    }

    for signature in signatures.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    signatures
}

fn build_seed_state_signatures_from_possible_matches(
    raw_possible_matches: &DensePossibleMatchesByState,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> FxHashMap<u32, SeedStateSignature> {
    let mut signatures: FxHashMap<u32, SeedStateSignature> = token_bytes
        .keys()
        .map(|&token_id| (token_id, Vec::new()))
        .collect();

    for (&original_tokenizer_state, terminals) in raw_possible_matches {
        for bitmap in terminals.values() {
            for_each_dense_bit(bitmap, |original_token_id| {
                let Some(equivalent_tokens) = tokens_with_same_bytes.get(&original_token_id) else {
                    return;
                };

                for &equivalent_token_id in equivalent_tokens.iter() {
                    if let Some(signature) = signatures.get_mut(&equivalent_token_id) {
                        signature.push(original_tokenizer_state);
                    }
                }
            });
        }
    }

    for signature in signatures.values_mut() {
        signature.sort_unstable();
        signature.dedup();
    }

    signatures
}

fn intern_signature_ids<T>(signatures: FxHashMap<u32, T>) -> FxHashMap<u32, SignatureClassId>
where
    T: Eq + Hash,
{
    let mut signature_to_id: FxHashMap<T, SignatureClassId> = FxHashMap::default();
    let mut token_to_id = FxHashMap::default();
    let mut next_id: SignatureClassId = 0;

    for (token_id, signature) in signatures {
        let signature_id = *signature_to_id.entry(signature).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        token_to_id.insert(token_id, signature_id);
    }

    token_to_id
}

fn build_constraint_vocab_map(
    parser_vocab: &ManyToOneIdMap,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    possible_match_signature_ids: &FxHashMap<u32, SignatureClassId>,
    seed_state_signature_ids: &FxHashMap<u32, SignatureClassId>,
) -> ConstraintVocabMap {
    let max_original_slot = token_bytes
        .keys()
        .next_back()
        .map(|token_id| *token_id as usize + 1)
        .unwrap_or(0);

    let mut original_to_internal = vec![
        u32::MAX;
        parser_vocab
            .original_to_internal
            .len()
            .max(max_original_slot)
    ];

    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut old_internal_to_constraint =
        vec![Vec::<u32>::new(); parser_vocab.internal_to_originals.len()];

    // parser_vocab may intentionally leave larger-vocab entries unmapped
    // when they never participate in parser-DWA behavior for this grammar.
    // Keep those originals at u32::MAX; downstream remap paths already skip
    // unmapped tokens.

    for (old_internal_id, originals) in parser_vocab.internal_to_originals.iter().enumerate() {
        let mut groups: BTreeMap<(SignatureClassId, SignatureClassId), Vec<u32>> = BTreeMap::new();

        for &original_token_id in originals {
            if !token_bytes.contains_key(&original_token_id) {
                continue;
            }

            let forward = parser_vocab
                .original_to_internal
                .get(original_token_id as usize)
                .copied()
                .unwrap_or(u32::MAX);

            debug_assert_eq!(
                forward,
                old_internal_id as u32,
                "inconsistent parser vocab map for original token {original_token_id}"
            );

            let signature = possible_match_signature_ids
                .get(&original_token_id)
                .cloned()
                .unwrap_or_default();
            let seed_signature = seed_state_signature_ids
                .get(&original_token_id)
                .cloned()
                .unwrap_or_default();

            groups
                .entry((signature, seed_signature))
                .or_default()
                .push(original_token_id);
        }

        for (_, mut originals) in groups {
            originals.sort_unstable();
            originals.dedup();

            let new_internal_id = internal_to_originals.len() as u32;

            for &original_token_id in &originals {
                if original_token_id as usize >= original_to_internal.len() {
                    original_to_internal.resize(original_token_id as usize + 1, u32::MAX);
                }

                original_to_internal[original_token_id as usize] = new_internal_id;
            }

            old_internal_to_constraint[old_internal_id].push(new_internal_id);
            internal_to_originals.push(originals);
        }
    }

    ConstraintVocabMap {
        original_to_internal,
        internal_to_originals,
        old_internal_to_constraint,
    }
}

fn remap_possible_matches_to_constraint_vocab(
    raw_possible_matches: DensePossibleMatchesByState,
    original_to_constraint_internal: &[u32],
    constraint_token_count: u32,
    tokens_with_same_bytes: &FxHashMap<u32, Arc<[u32]>>,
) -> DensePossibleMatchesByState {
    let num_words = dense_word_count(constraint_token_count);

    raw_possible_matches
        .into_iter()
        .map(|(original_tokenizer_state, terminals)| {
            let remapped_terminals = terminals
                .into_iter()
                .filter_map(|(terminal_id, original_bitmap)| {
                    let mut remapped = vec![0u64; num_words];
                    let mut any = false;

                    for_each_dense_bit(&original_bitmap, |original_token_id| {
                        let Some(equivalent_tokens) = tokens_with_same_bytes.get(&original_token_id) else {
                            return;
                        };

                        for &equivalent_token_id in equivalent_tokens.iter() {
                            let Some(&constraint_internal_id) =
                                original_to_constraint_internal.get(equivalent_token_id as usize)
                            else {
                                continue;
                            };

                            if constraint_internal_id == u32::MAX {
                                continue;
                            }

                            set_dense_bit(&mut remapped, constraint_internal_id);
                            any = true;
                        }
                    });

                    if any {
                        Some((terminal_id, remapped.into_boxed_slice()))
                    } else {
                        None
                    }
                })
                .collect();

            (original_tokenizer_state, remapped_terminals)
        })
        .collect()
}

fn remap_token_set_to_constraint_vocab(
    old_tokens: &RangeSetBlaze<u32>,
    old_internal_to_constraint: &[Vec<u32>],
) -> RangeSetBlaze<u32> {
    let mut new_ids = Vec::new();

    for old_internal_token in old_tokens.iter() {
        debug_assert!(
            (old_internal_token as usize) < old_internal_to_constraint.len(),
            "parser-DWA weight references old internal token id {old_internal_token}, but old_internal_to_constraint has only {} entries",
            old_internal_to_constraint.len()
        );

        if let Some(mapped_ids) = old_internal_to_constraint.get(old_internal_token as usize) {
            new_ids.extend_from_slice(mapped_ids);
        }
    }

    new_ids.sort_unstable();
    new_ids.dedup();
    range_set_from_sorted_ids(&new_ids)
}

fn remap_arc_token_set_to_constraint_vocab(
    token_set: &Arc<RangeSetBlaze<u32>>,
    old_internal_to_constraint: &[Vec<u32>],
    token_set_cache: &mut FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
) -> Arc<RangeSetBlaze<u32>> {
    let cache_key = Arc::as_ptr(token_set) as usize;

    if let Some(cached) = token_set_cache.get(&cache_key) {
        return Arc::clone(cached);
    }

    let remapped = shared_rangeset(remap_token_set_to_constraint_vocab(
        token_set,
        old_internal_to_constraint,
    ));

    token_set_cache.insert(cache_key, Arc::clone(&remapped));
    remapped
}

fn remap_weight_to_constraint_vocab(
    weight: &Weight,
    old_internal_to_constraint: &[Vec<u32>],
    token_set_cache: &mut FxHashMap<usize, Arc<RangeSetBlaze<u32>>>,
) -> Weight {
    if weight.is_full() {
        return Weight::all();
    }

    let mut remapped = RangeMapBlaze::new();

    for (start, end, token_set) in weight.compact_entries().unwrap_or_default() {
        let remapped_token_set = remap_arc_token_set_to_constraint_vocab(
            &token_set,
            old_internal_to_constraint,
            token_set_cache,
        );

        if !remapped_token_set.is_empty() {
            remapped.extend_simple(std::iter::once((start..=end, remapped_token_set)));
        }
    }

    finalize_weight_map(remapped)
}

fn remap_parser_dwa_to_constraint_vocab(
    parser_dwa: &mut DWA,
    old_internal_to_constraint: &[Vec<u32>],
) {
    let mut token_set_cache: FxHashMap<usize, Arc<RangeSetBlaze<u32>>> = FxHashMap::default();

    for state in parser_dwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap_weight_to_constraint_vocab(
                final_weight,
                old_internal_to_constraint,
                &mut token_set_cache,
            );
        }

        for (_, weight) in state.transitions.values_mut() {
            *weight = remap_weight_to_constraint_vocab(
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
        clear_weight_op_profile();
        let compile_started_at = Instant::now();
        let mut profile = CompilePhaseProfile::default();
        let debug_compile_stages = debug_compile_stages_enabled();

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
        if debug_compile_stages {
            eprintln!(
                "[glrmask/debug][compile_stage] analysis_done wall_ms={:.3} analyze_ms={:.3} glr_ms={:.3} coloring_ms={:.3} disallowed_ms={:.3}",
                profile.analysis_wall_ms,
                analyze_grammar_ms,
                glr_table_ms,
                terminal_coloring_ms,
                disallowed_follows_ms,
            );
        }

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

        if debug_terminal_mapping_enabled() {
            maybe_print_terminal_mappings(&analyzed_grammar);
        }

        if strict_one_flag_enabled("GLRMASK_DEBUG_DUMP_GLR_TABLE") {
            emit_glr_table_debug_dump(&table);
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
        if debug_compile_stages {
            eprintln!(
                "[glrmask/debug][compile_stage] classify_done ms={:.3}",
                profile.classify_ms,
            );
        }
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
        if strict_one_flag_enabled("GLRMASK_DEBUG_DUMP_TEMPLATES") {
            emit_templates_debug_dump(&templates);
        }
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
        if debug_compile_stages {
            eprintln!(
                "[glrmask/debug][compile_stage] id_map_templates_done id_map_ms={:.3} terminal_dwa_ms={:.3} compact_ms={:.3} templates_ms={:.3}",
                terminal_phase_profile.id_map_ms,
                terminal_phase_profile.terminal_dwa_ms,
                terminal_phase_profile.compact_ms,
                profile.templates_ms,
            );
        }
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
        if debug_compile_stages {
            eprintln!(
                "[glrmask/debug][compile_stage] terminal_dwa_done id_map_ms={:.3} terminal_dwa_ms={:.3} compact_ms={:.3}",
                profile.id_map_ms,
                profile.terminal_dwa_ms,
                profile.compact_ms,
            );
        }

        if let Ok(dump_path) = std::env::var("GLRMASK_ORACLE_DUMP") {
            let mut canonical_state_reps = vec![u32::MAX; internal_ids.num_tsids() as usize];
            for (orig, &class) in internal_ids
                .tokenizer_states
                .original_to_internal
                .iter()
                .enumerate()
            {
                if class == u32::MAX {
                    continue;
                }
                let orig = orig as u32;
                if orig < canonical_state_reps[class as usize] {
                    canonical_state_reps[class as usize] = orig;
                }
            }
            let mut canonical_token_reps = vec![u32::MAX; internal_ids.num_internal_tokens() as usize];
            for (orig, &class) in internal_ids.vocab_tokens.original_to_internal.iter().enumerate() {
                if class == u32::MAX {
                    continue;
                }
                let orig = orig as u32;
                if orig < canonical_token_reps[class as usize] {
                    canonical_token_reps[class as usize] = orig;
                }
            }
            let oracle_data = serde_json::json!({
                "state_map": internal_ids.tokenizer_states.original_to_internal,
                "parser_token_map": internal_ids.vocab_tokens.original_to_internal,
                "num_state_classes": internal_ids.num_tsids(),
                "parser_num_token_classes": internal_ids.num_internal_tokens(),
                "state_representatives": canonical_state_reps,
                "parser_token_representatives": canonical_token_reps,
            });
            std::fs::write(&dump_path, serde_json::to_string(&oracle_data).unwrap())
                .expect("failed to write oracle dump");
            eprintln!("[glrmask/oracle] dumped post-compact mappings to {dump_path}");
        }

        let ((mut parser_dwa, parser_dwa_ms), (raw_possible_matches, possible_matches_collect_ms)) =
            rayon::join(
                || {
                    let parser_dwa_started_at = Instant::now();
                    let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                        &table,
                        &analyzed_grammar,
                        &terminal_dwa,
                        templates,
                        vocab,
                        &internal_ids,
                    );
                    (parser_dwa, elapsed_ms(parser_dwa_started_at))
                },
                || {
                    let pm_started_at = Instant::now();
                    // IMPORTANT: Constraint possible_matches must be computed from
                    // ORIGINAL vocab token bytes. Do not use parser-DWA
                    // internal_token_bytes here. The parser-DWA token equivalence
                    // relation is not valid for possible_matches.
                    let token_entries: Vec<(usize, Vec<u8>)> = token_bytes
                        .iter()
                        .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                        .collect();
                    let trie_build_started_at = Instant::now();
                    let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(token_entries);
                    let trie_build_ms = elapsed_ms(trie_build_started_at);
                    let collect_started_at = Instant::now();
                    // This is a dense original-token-id universe size, not a count
                    // of parser-DWA internal tokens. Sparse original token ids require
                    // max_original_token_id + 1 slots.
                    let original_token_slots = max_original_token_slot(&token_bytes);
                    let (pm_by_tsid, dense_profile) = crate::compiler::possible_matches::collect_possible_matches_by_original_tsid_dense(
                        &tokenizer,
                        &trie.root,
                        original_token_slots,
                    );
                    let collect_ms = elapsed_ms(collect_started_at);
                    crate::compiler::possible_matches::emit_possible_matches_profile_summary(
                        "constraint_original_tokens",
                        token_bytes.len(),
                        tokenizer.num_states(),
                        trie_build_ms,
                        collect_ms,
                        &dense_profile,
                    );
                    (pm_by_tsid, elapsed_ms(pm_started_at))
                },
            );

        let constraint_vocab_started_at = Instant::now();
        let tokens_with_same_bytes_started_at = Instant::now();
        let tokens_with_same_bytes = build_tokens_with_same_bytes(&token_bytes);
        let tokens_with_same_bytes_ms = elapsed_ms(tokens_with_same_bytes_started_at);
        if compile_profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_vocab_step] step=same_bytes ms={:.3}",
                tokens_with_same_bytes_ms,
            );
        }

        let possible_match_signatures_started_at = Instant::now();
        let possible_match_signatures = build_possible_match_signatures(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let possible_match_signatures_ms = elapsed_ms(possible_match_signatures_started_at);
        if compile_profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_vocab_step] step=possible_match_signatures ms={:.3}",
                possible_match_signatures_ms,
            );
        }
        let possible_match_signature_ids = intern_signature_ids(possible_match_signatures);

        let seed_state_signatures_started_at = Instant::now();
        let seed_state_signatures = build_seed_state_signatures_from_possible_matches(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let seed_state_signatures_ms = elapsed_ms(seed_state_signatures_started_at);
        if compile_profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_vocab_step] step=seed_state_signatures ms={:.3}",
                seed_state_signatures_ms,
            );
        }
        let seed_state_signature_ids = intern_signature_ids(seed_state_signatures);

        let constraint_vocab_map_started_at = Instant::now();
        let constraint_vocab = build_constraint_vocab_map(
            &internal_ids.vocab_tokens,
            &token_bytes,
            &possible_match_signature_ids,
            &seed_state_signature_ids,
        );
        let constraint_vocab_map_ms = elapsed_ms(constraint_vocab_map_started_at);
        if compile_profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_vocab_step] step=build_map ms={:.3}",
                constraint_vocab_map_ms,
            );
        }
        let constraint_token_count = constraint_vocab.internal_to_originals.len() as u32;

        let remap_possible_matches_started_at = Instant::now();
        let possible_matches = remap_possible_matches_to_constraint_vocab(
            raw_possible_matches,
            &constraint_vocab.original_to_internal,
            constraint_token_count,
            &tokens_with_same_bytes,
        );
        let remap_possible_matches_ms = elapsed_ms(remap_possible_matches_started_at);
        if compile_profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_vocab_step] step=remap_possible_matches ms={:.3}",
                remap_possible_matches_ms,
            );
        }

        let remap_parser_dwa_started_at = Instant::now();
        remap_parser_dwa_to_constraint_vocab(
            &mut parser_dwa,
            &constraint_vocab.old_internal_to_constraint,
        );
        let remap_parser_dwa_ms = elapsed_ms(remap_parser_dwa_started_at);
        if compile_profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_vocab_step] step=remap_parser_dwa ms={:.3}",
                remap_parser_dwa_ms,
            );
        }
        let constraint_vocab_ms = elapsed_ms(constraint_vocab_started_at);

        if compile_profile_summary_enabled() {
            let split_parser_tokens = constraint_vocab
                .old_internal_to_constraint
                .iter()
                .filter(|mapped| mapped.len() > 1)
                .count();

            eprintln!(
                "[glrmask/profile][constraint_vocab] parser_tokens={} constraint_tokens={} split_parser_tokens={} reconcile_ms={:.3} same_bytes_ms={:.3} possible_match_signatures_ms={:.3} seed_state_signatures_ms={:.3} build_map_ms={:.3} remap_possible_matches_ms={:.3} remap_parser_dwa_ms={:.3}",
                internal_ids.vocab_tokens.num_internal_ids(),
                constraint_token_count,
                split_parser_tokens,
                constraint_vocab_ms,
                tokens_with_same_bytes_ms,
                possible_match_signatures_ms,
                seed_state_signatures_ms,
                constraint_vocab_map_ms,
                remap_possible_matches_ms,
                remap_parser_dwa_ms,
            );
        }

        let internal_token_bytes_started_at = Instant::now();
        let internal_token_bytes = build_internal_token_bytes_from_groups(
            vocab,
            &constraint_vocab.internal_to_originals,
        );
        let internal_token_bytes_ms = elapsed_ms(internal_token_bytes_started_at);

        profile.parser_dwa_ms = parser_dwa_ms;
        profile.permute_possible_matches_ms = possible_matches_collect_ms + constraint_vocab_ms;
        profile.internal_token_bytes_ms = internal_token_bytes_ms;
        if debug_compile_stages {
            eprintln!(
                "[glrmask/debug][compile_stage] parser_possible_matches_done parser_dwa_ms={:.3} possible_matches_ms={:.3} internal_token_bytes_ms={:.3}",
                profile.parser_dwa_ms,
                profile.permute_possible_matches_ms,
                profile.internal_token_bytes_ms,
            );
        }

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
        if debug_compile_stages {
            eprintln!(
                "[glrmask/debug][compile_stage] finalize_done finalize_ms={:.3} compile_ms={:.3}",
                profile.finalize_ms,
                profile.compile_ms,
            );
        }

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
    emit_weight_op_profile_summary();
    (constraint, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Constraint;

    fn bitmap(tokens: &[u32], token_slots: u32) -> Box<[u64]> {
        let mut words = vec![0u64; dense_word_count(token_slots)];
        for &token in tokens {
            set_dense_bit(&mut words, token);
        }
        words.into_boxed_slice()
    }

    fn brute_force_seed_state_signatures(
        tokenizer: &Tokenizer,
        token_bytes: &BTreeMap<u32, Vec<u8>>,
    ) -> FxHashMap<u32, SeedStateSignature> {
        let mut signatures: FxHashMap<u32, SeedStateSignature> = token_bytes
            .keys()
            .map(|&token_id| (token_id, Vec::new()))
            .collect();

        for tokenizer_state in 0..tokenizer.num_states() {
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

        let tokens_with_same_bytes = build_tokens_with_same_bytes(&token_bytes);
        let signatures = build_possible_match_signatures(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let signature_ids = intern_signature_ids(signatures);
        let seed_state_signatures: FxHashMap<u32, SeedStateSignature> = token_bytes
            .keys()
            .map(|&token_id| (token_id, Vec::new()))
            .collect();
        let seed_state_signature_ids = intern_signature_ids(seed_state_signatures);
        let constraint_vocab = build_constraint_vocab_map(
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

        let remapped_pm = remap_possible_matches_to_constraint_vocab(
            raw_possible_matches,
            &constraint_vocab.original_to_internal,
            constraint_vocab.internal_to_originals.len() as u32,
            &tokens_with_same_bytes,
        );

        let terminal_10 = &remapped_pm[&5][&10];
        let terminal_11 = &remapped_pm[&5][&11];

        assert!(dense_bit_is_set(terminal_10, tok0));
        assert!(!dense_bit_is_set(terminal_10, tok1));
        assert!(!dense_bit_is_set(terminal_11, tok0));
        assert!(dense_bit_is_set(terminal_11, tok1));
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

        let tokens_with_same_bytes = build_tokens_with_same_bytes(&token_bytes);
        let signatures = build_possible_match_signatures(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let signature_ids = intern_signature_ids(signatures);
        let seed_state_signatures: FxHashMap<u32, SeedStateSignature> = token_bytes
            .keys()
            .map(|&token_id| (token_id, Vec::new()))
            .collect();
        let seed_state_signature_ids = intern_signature_ids(seed_state_signatures);
        let constraint_vocab = build_constraint_vocab_map(
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
        let new_weight = remap_weight_to_constraint_vocab(
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
        let tokens_with_same_bytes = build_tokens_with_same_bytes(&token_bytes);
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            token_bytes
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let (raw_possible_matches, _) = crate::compiler::possible_matches::collect_possible_matches_by_original_tsid_dense(
            &constraint.tokenizer,
            &trie.root,
            max_original_token_slot(&token_bytes),
        );

        let actual = build_seed_state_signatures_from_possible_matches(
            &raw_possible_matches,
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
        let tokens_with_same_bytes = build_tokens_with_same_bytes(&token_bytes);
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            token_bytes
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let (raw_possible_matches, _) = crate::compiler::possible_matches::collect_possible_matches_by_original_tsid_dense(
            &constraint.tokenizer,
            &trie.root,
            max_original_token_slot(&token_bytes),
        );

        let actual = build_seed_state_signatures_from_possible_matches(
            &raw_possible_matches,
            &token_bytes,
            &tokens_with_same_bytes,
        );
        let expected = brute_force_seed_state_signatures(&constraint.tokenizer, &token_bytes);

        assert_eq!(actual, expected);
    }
}