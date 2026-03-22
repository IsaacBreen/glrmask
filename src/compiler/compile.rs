#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, HashMap, VecDeque};

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
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates_report,
};
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::compiler::stages::templates::Templates;
use crate::compiler::terminal_dwa::{TerminalDwaBuildReport, build_terminal_dwa, build_terminal_dwa_with_possible_matches_report, build_terminal_dwa_with_report};
use crate::compiler::stages::terminal_dwa::compute_ever_allowed_follows;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;
use crate::runtime::Constraint;

const DWA_SAMPLE_TOKEN_REPR_LIMIT: usize = 48;
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DWA_TERM: &str = "\x1b[38;5;45m";
const ANSI_DWA_TOKEN: &str = "\x1b[38;5;114m";
const ANSI_DWA_LEN: &str = "\x1b[38;5;220m";

/// Convert ever-allowed follow sets into the `BTreeMap<u32, BitSet>` disallowed-follows
/// format expected by the equivalence analysis.
pub(crate) fn compute_disallowed_follows(grammar: &AnalyzedGrammar) -> BTreeMap<u32, BitSet> {
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
        .representative_original_ids
        .iter()
        .enumerate()
        .filter_map(|(internal_token_id, &representative)| {
            let bytes = vocab.entries.get(&representative)?.clone();
            Some((internal_token_id as u32, bytes))
        })
        .collect()
}

fn build_internal_token_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {
    vocab
        .entries
        .iter()
        .filter_map(|(&original_token_id, bytes)| {
            id_map
                .vocab_tokens
                .internal_id_for_original(original_token_id)
                .map(|internal_token_id| (internal_token_id, bytes.clone()))
        })
        .collect()
}

use crate::compiler::grammar::transforms::{expand_nullable_terminals, compact_unused_terminals, inline_single_use_nonterminals, prepare_grammar_for_compile, prepare_owned_grammar_for_compile};


pub(crate) fn compile_profile_enabled() -> bool {
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

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|value| value.parse().ok())
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|value| value.parse().ok())
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    for _ in 0..max_chars {
        let Some(ch) = chars.next() else {
            return text.to_string();
        };
        out.push(ch);
    }
    if chars.next().is_some() {
        out.push_str("...");
    }
    out
}

fn colorize(text: impl AsRef<str>, ansi: &str) -> String {
    format!("{ansi}{}{ANSI_RESET}", text.as_ref())
}

fn escape_single_quoted(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '\'' => escaped.push_str("\\'"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            other => escaped.push(other),
        }
    }
    escaped
}

fn single_quoted(text: &str) -> String {
    format!("'{}'", escape_single_quoted(text))
}

fn token_repr(bytes: &[u8]) -> String {
    let escaped = escape_single_quoted(&String::from_utf8_lossy(bytes));
    format!(
        "'{}'",
        colorize(truncate_chars(&escaped, DWA_SAMPLE_TOKEN_REPR_LIMIT), ANSI_DWA_TOKEN),
    )
}

/// Try to extract a fixed byte sequence from an Expr tree.
/// Returns Some(bytes) if the expr always matches exactly one fixed string,
/// None otherwise (e.g., contains character classes, repetitions, choices).
fn expr_to_fixed_bytes(expr: &Expr) -> Option<Vec<u8>> {
    match expr {
        Expr::U8Seq(bytes) => Some(bytes.clone()),
        Expr::Seq(parts) => {
            let mut result = Vec::new();
            for part in parts {
                result.extend(expr_to_fixed_bytes(part)?);
            }
            Some(result)
        }
        Expr::Exclude { expr, exclude } => {
            let base = expr_to_fixed_bytes(expr)?;
            match expr_to_fixed_bytes(exclude) {
                Some(blocked) if blocked == base => None,
                _ => Some(base),
            }
        }
        _ => None,
    }
}

/// Given a terminal DWA path (as a list of terminal IDs), find vocab tokens whose
/// bytes match the concatenated terminal byte string. Shows tokens that are a prefix
/// of the path bytes or whose bytes the path bytes are a prefix of.
fn sample_path_matching_tokens(
    grammar: &GrammarDef,
    labels: &[i32],
    vocab: &Vocab,
    max_tokens: usize,
) -> String {
    // Build the concatenated byte string for this path from leading literal terminals.
    // Handles Literal, Pattern (simple regex), and Expr (terminal rules) terminals.
    let mut path_bytes = Vec::new();
    for &label in labels {
        let tid = label as u32;
        match grammar.terminals.iter().find(|t| t.id() == tid) {
            Some(Terminal::Literal { bytes, .. }) => path_bytes.extend_from_slice(bytes),
            Some(Terminal::Pattern { pattern, utf8, .. }) => {
                match parse_regex(pattern, *utf8) {
                    Expr::U8Seq(bytes) => path_bytes.extend_from_slice(&bytes),
                    _ => break,
                }
            }
            Some(Terminal::Expr { expr, .. }) => {
                match expr_to_fixed_bytes(expr) {
                    Some(bytes) => path_bytes.extend_from_slice(&bytes),
                    None => break,
                }
            }
            _ => break,
        }
    }

    if path_bytes.is_empty() {
        return "tokens=?".to_string();
    }

    // Find vocab tokens whose bytes overlap with path_bytes:
    // - token bytes are a prefix of path_bytes (token covers start of path), OR
    // - path_bytes is a prefix of token bytes (token extends beyond path)
    let mut candidates: Vec<(Vec<u8>, u32)> = Vec::new();
    for (&token_id, token_bytes) in &vocab.entries {
        if token_bytes.is_empty() {
            continue;
        }
        let is_prefix_of_path = path_bytes.starts_with(token_bytes);
        let path_is_prefix_of_token = token_bytes.starts_with(&path_bytes);
        if is_prefix_of_path || path_is_prefix_of_token {
            candidates.push((token_bytes.clone(), token_id));
        }
    }

    // Sort: exact match first, then by length (shortest first), then by bytes
    candidates.sort_by(|(a_bytes, a_id), (b_bytes, b_id)| {
        let a_exact = a_bytes.as_slice() == path_bytes.as_slice();
        let b_exact = b_bytes.as_slice() == path_bytes.as_slice();
        b_exact.cmp(&a_exact)
            .then_with(|| a_bytes.len().cmp(&b_bytes.len()))
            .then_with(|| a_bytes.cmp(b_bytes))
            .then_with(|| a_id.cmp(b_id))
    });

    let samples: Vec<String> = candidates
        .into_iter()
        .take(max_tokens)
        .map(|(bytes, _)| token_repr(&bytes))
        .collect();

    if samples.is_empty() {
        "tokens=[]".to_string()
    } else {
        format!("tokens=[{}]", samples.join(", "))
    }
}

fn sample_weight_tokens(
    weight: &Weight,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    max_tokens: usize,
) -> String {
    if weight.is_empty() {
        return "tokens=[]".to_string();
    }
    if weight.is_full() {
        return "tokens=ALL".to_string();
    }

    // Collect internal token IDs from the weight, capping at 256
    let mut internal_ids = Vec::new();
    for range in weight.token_union().ranges() {
        let mut token_id = *range.start();
        while token_id <= *range.end() && internal_ids.len() < 256 {
            internal_ids.push(token_id);
            if token_id == u32::MAX {
                break;
            }
            token_id = token_id.saturating_add(1);
        }
        if internal_ids.len() >= 256 {
            break;
        }
    }

    // Expand internal IDs to original token IDs and collect (bytes, original_id) pairs.
    // Deduplicate by bytes so we don't show the same string multiple times.
    let mut seen_bytes = std::collections::HashSet::new();
    let mut candidates: Vec<(Vec<u8>, u32)> = Vec::new();
    for internal_id in internal_ids {
        let originals = match id_map.vocab_tokens.internal_to_originals.get(internal_id as usize) {
            Some(ids) => ids,
            None => continue,
        };
        for original_id in originals.iter() {
            if let Some(bytes) = vocab.entries.get(&original_id) {
                if seen_bytes.insert(bytes.clone()) {
                    candidates.push((bytes.clone(), original_id));
                }
            }
            // Cap total candidates to avoid excessive work
            if candidates.len() >= 512 {
                break;
            }
        }
        if candidates.len() >= 512 {
            break;
        }
    }

    // Sort shortest first (most readable), break ties by bytes then by id
    candidates.sort_by(|(left_bytes, left_id), (right_bytes, right_id)| {
        left_bytes
            .len()
            .cmp(&right_bytes.len())
            .then_with(|| left_bytes.cmp(right_bytes))
            .then_with(|| left_id.cmp(right_id))
    });

    let total = candidates.len();
    let samples: Vec<String> = candidates
        .into_iter()
        .take(max_tokens)
        .map(|(bytes, _)| token_repr(&bytes))
        .collect();

    if samples.is_empty() {
        "tokens=[]".to_string()
    } else if total > max_tokens {
        let remaining = total - max_tokens;
        format!("tokens=[{}, ...and {} others]", samples.join(", "), remaining)
    } else {
        format!("tokens=[{}]", samples.join(", "))
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct ValidPathConfig {
    state: u32,
    weight: Weight,
}

struct ValidPathNode {
    accepting: bool,
    transitions: Vec<usize>,
}

enum MaxValidPathLen {
    Finite(usize),
    Infinite,
}

fn terminal_label_name(grammar: &GrammarDef, label: i32) -> String {
    assert!(label >= 0, "terminal DWA emitted unexpected negative label {label}");
    let terminal = label as u32;
    let rendered = grammar
        .terminals
        .iter()
        .find(|candidate| candidate.id() == terminal)
        .map(|terminal_def| match terminal_def {
            Terminal::Literal { bytes, .. } => single_quoted(&String::from_utf8_lossy(bytes)),
            Terminal::Expr { expr, .. } => {
                if let Some(bytes) = expr_to_fixed_bytes(expr) {
                    single_quoted(&String::from_utf8_lossy(&bytes))
                } else {
                    grammar.terminal_display_name(terminal)
                }
            }
            _ => grammar.terminal_display_name(terminal),
        })
        .unwrap_or_else(|| grammar.terminal_display_name(terminal));
    colorize(rendered, ANSI_DWA_TERM)
}

fn terminal_dwa_max_valid_path_len(terminal_dwa: &DWA) -> Option<MaxValidPathLen> {
    if terminal_dwa.states.is_empty() {
        return None;
    }

    let mut node_ids = HashMap::<ValidPathConfig, usize>::new();
    let mut node_configs = Vec::<ValidPathConfig>::new();
    let mut nodes = Vec::<ValidPathNode>::new();
    let mut queue = VecDeque::<usize>::new();

    let start = ValidPathConfig {
        state: terminal_dwa.start_state,
        weight: Weight::all(),
    };
    node_ids.insert(start.clone(), 0);
    node_configs.push(start);
    nodes.push(ValidPathNode {
        accepting: false,
        transitions: Vec::new(),
    });
    queue.push_back(0);

    while let Some(node_id) = queue.pop_front() {
        let config = node_configs[node_id].clone();
        let state = &terminal_dwa.states[config.state as usize];

        nodes[node_id].accepting = state
            .final_weight
            .as_ref()
            .map(|final_weight| !config.weight.intersection(final_weight).is_empty())
            .unwrap_or(false);

        let mut next_ids = Vec::new();
        for (next_state, edge_weight) in state.transitions.values() {
            let next_weight = config.weight.intersection(edge_weight);
            if next_weight.is_empty() {
                continue;
            }
            let next_config = ValidPathConfig {
                state: *next_state,
                weight: next_weight,
            };
            let next_id = if let Some(&existing) = node_ids.get(&next_config) {
                existing
            } else {
                let new_id = nodes.len();
                node_ids.insert(next_config.clone(), new_id);
                node_configs.push(next_config);
                nodes.push(ValidPathNode {
                    accepting: false,
                    transitions: Vec::new(),
                });
                queue.push_back(new_id);
                new_id
            };
            next_ids.push(next_id);
        }
        next_ids.sort_unstable();
        next_ids.dedup();
        nodes[node_id].transitions = next_ids;
    }

    let mut reverse = vec![Vec::<usize>::new(); nodes.len()];
    for (node_id, node) in nodes.iter().enumerate() {
        for &next_id in &node.transitions {
            reverse[next_id].push(node_id);
        }
    }

    let mut productive = vec![false; nodes.len()];
    let mut productive_queue = VecDeque::new();
    for (node_id, node) in nodes.iter().enumerate() {
        if node.accepting {
            productive[node_id] = true;
            productive_queue.push_back(node_id);
        }
    }
    while let Some(node_id) = productive_queue.pop_front() {
        for &prev_id in &reverse[node_id] {
            if !productive[prev_id] {
                productive[prev_id] = true;
                productive_queue.push_back(prev_id);
            }
        }
    }

    if !productive[0] {
        return None;
    }

    let mut reachable = vec![false; nodes.len()];
    let mut reachable_queue = VecDeque::from([0usize]);
    reachable[0] = true;
    while let Some(node_id) = reachable_queue.pop_front() {
        for &next_id in &nodes[node_id].transitions {
            if productive[next_id] && !reachable[next_id] {
                reachable[next_id] = true;
                reachable_queue.push_back(next_id);
            }
        }
    }

    let reachable_count = reachable
        .iter()
        .zip(productive.iter())
        .filter(|(is_reachable, is_productive)| **is_reachable && **is_productive)
        .count();

    let mut indegree = vec![0usize; nodes.len()];
    for (node_id, node) in nodes.iter().enumerate() {
        if !(reachable[node_id] && productive[node_id]) {
            continue;
        }
        for &next_id in &node.transitions {
            if reachable[next_id] && productive[next_id] {
                indegree[next_id] += 1;
            }
        }
    }

    let mut topo_queue = VecDeque::new();
    for node_id in 0..nodes.len() {
        if reachable[node_id] && productive[node_id] && indegree[node_id] == 0 {
            topo_queue.push_back(node_id);
        }
    }

    let mut topo_order = Vec::with_capacity(reachable_count);
    while let Some(node_id) = topo_queue.pop_front() {
        topo_order.push(node_id);
        for &next_id in &nodes[node_id].transitions {
            if reachable[next_id] && productive[next_id] {
                indegree[next_id] -= 1;
                if indegree[next_id] == 0 {
                    topo_queue.push_back(next_id);
                }
            }
        }
    }

    if topo_order.len() != reachable_count {
        return Some(MaxValidPathLen::Infinite);
    }

    let mut longest = vec![None::<usize>; nodes.len()];
    for &node_id in topo_order.iter().rev() {
        let mut best = nodes[node_id].accepting.then_some(0);
        for &next_id in &nodes[node_id].transitions {
            if !(reachable[next_id] && productive[next_id]) {
                continue;
            }
            if let Some(next_best) = longest[next_id] {
                let candidate = next_best + 1;
                best = Some(best.map_or(candidate, |current| current.max(candidate)));
            }
        }
        longest[node_id] = best;
    }

    longest[0].map(MaxValidPathLen::Finite)
}

fn log_terminal_dwa_sample_paths(
    grammar: &GrammarDef,
    terminal_dwa: &DWA,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) {
    use rand::Rng;

    let Some(num_sample_paths) = env_usize("DWA_SAMPLE_PATHS") else {
        return;
    };

    if terminal_dwa.states.is_empty() {
        eprintln!("[glrmask/dwa_sample] terminal DWA is empty");
        return;
    }

    let sample_long = env_flag_enabled("DWA_SAMPLE_LONG");
    let min_len = env_usize("DWA_SAMPLE_MIN_LEN");
    let max_tokens = env_usize("DWA_SAMPLE_MAX_TOKENS").unwrap_or(3);
    let target_samples = if sample_long {
        num_sample_paths.saturating_mul(20).max(num_sample_paths)
    } else {
        num_sample_paths
    };
    let max_attempts = env_usize("DWA_SAMPLE_MAX_ATTEMPTS")
        .unwrap_or_else(|| target_samples.saturating_mul(50).max(target_samples));
    let max_steps = env_usize("DWA_SAMPLE_MAX_STEPS")
        .unwrap_or(if sample_long { 2048 } else { 512 });
    let end_prob = env_f64("DWA_SAMPLE_END_PROB")
        .map(|prob| prob.clamp(0.0, 1.0))
        .unwrap_or(if sample_long { 0.1 } else { 0.3 });

    let max_valid_len = terminal_dwa_max_valid_path_len(terminal_dwa);

    let mut rng = rand::thread_rng();
    let mut collected: Vec<(String, usize, Vec<i32>, Weight)> = Vec::new();
    let mut attempts = 0usize;

    while collected.len() < target_samples && attempts < max_attempts {
        let mut state = terminal_dwa.start_state;
        let mut path_labels: Vec<i32> = Vec::new();
        let mut weight = Weight::all();
        let mut steps = 0usize;
        let mut accepted = false;

        loop {
            let end_weight = terminal_dwa.states[state as usize]
                .final_weight
                .as_ref()
                .map(|final_weight| weight.intersection(final_weight))
                .filter(|weight| !weight.is_empty());

            let mut choices = Vec::new();
            for (&label, (next_state, edge_weight)) in &terminal_dwa.states[state as usize].transitions {
                let next_weight = weight.intersection(edge_weight);
                if !next_weight.is_empty() {
                    choices.push((label, *next_state, next_weight));
                }
            }

            if let Some(ref ew) = end_weight {
                if (choices.is_empty() || rng.gen_bool(end_prob))
                    && min_len.map_or(true, |min| path_labels.len() >= min)
                {
                    let display = path_labels
                        .iter()
                        .map(|&l| terminal_label_name(grammar, l))
                        .collect::<Vec<_>>()
                        .join(" -> ");
                    collected.push((display, path_labels.len(), path_labels.clone(), ew.clone()));
                    accepted = true;
                    break;
                }
            }

            if choices.is_empty() || steps >= max_steps {
                break;
            }

            let idx = rng.gen_range(0..choices.len());
            let (label, next_state, next_weight) = choices.swap_remove(idx);
            path_labels.push(label);
            weight = next_weight;
            state = next_state;
            steps += 1;
        }

        attempts += 1;
        if !accepted {
            continue;
        }
    }

    let mut deduped = BTreeMap::new();
    for (path, len, labels, end_weight) in collected {
        deduped.entry(path).or_insert((len, labels, end_weight));
    }
    let mut collected: Vec<(String, usize, Vec<i32>, Weight)> = deduped
        .into_iter()
        .map(|(path, (len, labels, end_weight))| (path, len, labels, end_weight))
        .collect();

    if sample_long {
        collected.sort_by(|left, right| right.1.cmp(&left.1));
    }
    if collected.len() > num_sample_paths {
        collected.truncate(num_sample_paths);
    }

    let max_valid_len = match max_valid_len {
        Some(MaxValidPathLen::Finite(len)) => format!("max_valid_len={}", colorize(len.to_string(), ANSI_DWA_LEN)),
        Some(MaxValidPathLen::Infinite) => format!("max_valid_len={}", colorize("infinite", ANSI_DWA_LEN)),
        None => format!("max_valid_len={}", colorize("none", ANSI_DWA_LEN)),
    };
    eprintln!(
        "[glrmask/dwa_sample] terminal DWA sample paths (n={}, attempts={}, {}):",
        collected.len(),
        attempts,
        max_valid_len,
    );
    for (idx, (path, _len, _labels, end_weight)) in collected.iter().enumerate() {
        eprintln!(
            "[glrmask/dwa_sample]   Path {}: {} ({})",
            idx,
            path,
            sample_weight_tokens(end_weight, vocab, id_map, max_tokens),
        );
    }
    if collected.is_empty() {
        eprintln!("[glrmask/dwa_sample]   (no non-empty terminal DWA paths collected)");
    }
}

fn parser_dwa_label_name(label: i32) -> String {
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    if label == DEFAULT_LABEL {
        "*".to_string()
    } else {
        label.to_string()
    }
}

fn log_parser_dwa_sample_paths(
    parser_dwa: &DWA,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) {
    use rand::Rng;

    let Some(num_sample_paths) = env_usize("PARSER_DWA_SAMPLE_PATHS") else {
        return;
    };

    if parser_dwa.states.is_empty() {
        eprintln!("[glrmask/parser_dwa_sample] parser DWA is empty");
        return;
    }

    let sample_long = env_flag_enabled("PARSER_DWA_SAMPLE_LONG");
    let max_tokens = env_usize("PARSER_DWA_SAMPLE_MAX_TOKENS").unwrap_or(3);
    let target_samples = if sample_long {
        num_sample_paths.saturating_mul(20).max(num_sample_paths)
    } else {
        num_sample_paths
    };
    let max_attempts = target_samples.saturating_mul(50).max(target_samples);
    let max_steps = if sample_long { 2048 } else { 512 };
    let end_prob = if sample_long { 0.1 } else { 0.3 };

    let mut rng = rand::thread_rng();
    let mut collected: Vec<(String, usize, Weight)> = Vec::new();
    let mut attempts = 0usize;

    while collected.len() < target_samples && attempts < max_attempts {
        let mut state = parser_dwa.start_state;
        let mut path_labels: Vec<i32> = Vec::new();
        let mut weight = Weight::all();
        let mut steps = 0usize;

        loop {
            let end_weight = parser_dwa.states[state as usize]
                .final_weight
                .as_ref()
                .map(|fw| weight.intersection(fw))
                .filter(|w| !w.is_empty());

            let mut choices = Vec::new();
            for (&label, (next_state, edge_weight)) in &parser_dwa.states[state as usize].transitions {
                let next_weight = weight.intersection(edge_weight);
                if !next_weight.is_empty() {
                    choices.push((label, *next_state, next_weight));
                }
            }

            if let Some(ref ew) = end_weight {
                if choices.is_empty() || rng.gen_bool(end_prob) {
                    let display = path_labels
                        .iter()
                        .map(|&l| parser_dwa_label_name(l))
                        .collect::<Vec<_>>()
                        .join(" -> ");
                    collected.push((display, path_labels.len(), ew.clone()));
                    break;
                }
            }

            if choices.is_empty() || steps >= max_steps {
                break;
            }

            let idx = rng.gen_range(0..choices.len());
            let (label, next_state, next_weight) = choices.swap_remove(idx);
            path_labels.push(label);
            weight = next_weight;
            state = next_state;
            steps += 1;
        }

        attempts += 1;
    }

    let mut deduped = BTreeMap::new();
    for (path, len, end_weight) in collected {
        deduped.entry(path).or_insert((len, end_weight));
    }
    let mut collected: Vec<(String, usize, Weight)> = deduped
        .into_iter()
        .map(|(path, (len, ew))| (path, len, ew))
        .collect();

    if sample_long {
        collected.sort_by(|a, b| b.1.cmp(&a.1));
    }
    if collected.len() > num_sample_paths {
        collected.truncate(num_sample_paths);
    }

    eprintln!(
        "[glrmask/parser_dwa_sample] parser DWA sample paths (n={}, attempts={}):",
        collected.len(),
        attempts,
    );
    for (idx, (path, _len, end_weight)) in collected.iter().enumerate() {
        eprintln!(
            "[glrmask/parser_dwa_sample]   Path {}: {} ({})",
            idx,
            path,
            sample_weight_tokens(end_weight, vocab, id_map, max_tokens),
        );
    }
    if collected.is_empty() {
        eprintln!("[glrmask/parser_dwa_sample]   (no paths collected)");
    }
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
        "[glrmask/profile][summary] terminal_dwa det states={} final={} trans={} det_ms={:.3} | min states={} final={} trans={} min_ms={:.3} collapse={} subtract_disallowed_ms={:.3}",
        terminal_build.terminal_dwa.states,
        terminal_build.terminal_dwa.final_states,
        terminal_build.terminal_dwa.transitions,
        ms(terminal_build.determinize_time),
        terminal_build.terminal_minimized_dwa.states,
        terminal_build.terminal_minimized_dwa.final_states,
        terminal_build.terminal_minimized_dwa.transitions,
        ms(terminal_build.minimize_time),
        if terminal_build.collapse_always_allowed_applied { "applied" } else { "skipped" },
        ms(terminal_build.subtract_disallowed_time),
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

fn compile_prepared(
    normalized: GrammarDef,
    tokenizer: Tokenizer,
    vocab: &Vocab,
    profile_enabled: bool,
    summary_enabled: bool,
    compile_started_at: std::time::Instant,
    prepare_grammar_time: std::time::Duration,
) -> Constraint {
    let normalized = normalized;
    let tokenizer = tokenizer;

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
    let mut id_map = if env_flag_enabled("GLRMASK_SKIP_EQ_ANALYSIS") {
        eprintln!("[glrmask] GLRMASK_SKIP_EQ_ANALYSIS: using identity id_map (no equivalence merging)");
        InternalIdMap::build_identity(&tokenizer, vocab)
    } else {
        InternalIdMap::build(&tokenizer, vocab, &disallowed_follows, normalized.ignore_terminal)
    };
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

    #[cfg(feature = "rayon")]
    let ((mut terminal_dwa, mut possible_matches, terminal_build), (characterizations, templates)) = rayon::join(
        || build_terminal_dwa_with_possible_matches_report(&glr_grammar, &tokenizer, vocab, &id_map, normalized.ignore_terminal),
        || {
            let characterizations = characterize_terminals(&table, &glr_grammar);
            let templates = Templates::from_characterizations(&characterizations);
            (characterizations, templates)
        },
    );
    #[cfg(not(feature = "rayon"))]
    let ((mut terminal_dwa, mut possible_matches, terminal_build), (characterizations, templates)) = {
        let td = build_terminal_dwa_with_possible_matches_report(&glr_grammar, &tokenizer, vocab, &id_map, normalized.ignore_terminal);
        let characterizations = characterize_terminals(&table, &glr_grammar);
        let templates = Templates::from_characterizations(&characterizations);
        (td, (characterizations, templates))
    };
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_terminal_dwa_ms={:.3}",
            ms(terminal_build.total_time),
        );
    }

    let phase_started_at = std::time::Instant::now();
    let compact_report = crate::compiler::stages::compact::compact_dwa_dimensions(
        &mut terminal_dwa,
        &mut id_map,
    );
    crate::compiler::possible_matches::permute_possible_matches_in_place(
        &mut possible_matches,
        &compact_report.token_perm,
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
        eprintln!(
            "[glrmask/profile][compile] compact_terminal_breakdown outer={}→{} token_ranges={}→{}",
            compact_report.old_outer_ranges,
            compact_report.new_outer_ranges,
            compact_report.old_token_ranges,
            compact_report.new_token_ranges,
        );
    }

    let phase_started_at = std::time::Instant::now();
    let internal_token_bytes = build_internal_token_bytes(vocab, &id_map);
    log_compile_profile(profile_enabled, "build_possible_matches", phase_started_at);

    log_terminal_dwa_sample_paths(&normalized, &terminal_dwa, vocab, &id_map);

    let (parser_dwa, parser_build) = build_parser_dwa_from_terminal_dwa_with_precomputed_templates_report(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
        characterizations,
        templates,
    );
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_parser_dwa_ms={:.3}",
            ms(parser_build.total_time),
        );
    }

    log_parser_dwa_sample_paths(&parser_dwa, vocab, &id_map);

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] total_ms={:.3} rules={} terminals={} vocab_entries={} tokenizer_states={} internal_tsids={} glr_table_states={} terminal_{} parser_{}",
            compile_started_at.elapsed().as_secs_f64() * 1000.0,
            normalized.rules.len(),
            normalized.terminals.len(),
            vocab.entries.len(),
            tokenizer.num_states(),
            id_map.num_tsids(),
            table.num_states,
            terminal_build.terminal_minimized_dwa,
            parser_build.parser_dwa_minimized,
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

    let phase_started_at = std::time::Instant::now();
    let mut constraint = Constraint {
        parser_dwa,
        table,
        tokenizer,
        ignore_terminal: normalized.ignore_terminal,
        possible_matches,
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals_vecs(),
        original_token_to_internal: id_map.vocab_tokens.original_to_internal.clone(),
        internal_token_to_tokens: id_map.vocab_tokens.internal_to_originals_vecs(),
        eos_token_id: vocab.eos_token_id,
        token_bytes,
        internal_token_bytes,
        token_bytes_dense: Vec::new(),
        internal_token_buf_masks: Vec::new(),
        internal_token_dense_words: 0,
        weight_token_dense_masks: rustc_hash::FxHashMap::default(),
        seed_terminal_dense: rustc_hash::FxHashMap::default(),
        seed_universe_dense: Box::new([]),
        dwa_fast_transitions: Vec::new(),
    };
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_constraint_struct_ms={:.3}",
            ms(phase_started_at.elapsed()),
        );
    }
    let phase_started_at = std::time::Instant::now();
    constraint.build_buf_masks();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_buf_masks_ms={:.3}",
            ms(phase_started_at.elapsed()),
        );
    }
    let phase_started_at = std::time::Instant::now();
    constraint.build_dense_token_bytes();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_dense_token_bytes_ms={:.3}",
            ms(phase_started_at.elapsed()),
        );
    }
    let phase_started_at = std::time::Instant::now();
    constraint.build_dense_token_masks();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][compile] build_dense_token_masks_ms={:.3}",
            ms(phase_started_at.elapsed()),
        );
    }
    constraint.build_fast_transitions();
    constraint.build_seed_dense_masks();
    constraint
}

pub fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    let profile_enabled = compile_profile_enabled();
    let summary_enabled = compile_summary_enabled();
    let compile_started_at = std::time::Instant::now();

    let phase_started_at = std::time::Instant::now();
    let (normalized, tokenizer) = prepare_grammar_for_compile(grammar);
    let prepare_grammar_time = phase_started_at.elapsed();
    log_compile_profile(profile_enabled, "prepare_grammar", phase_started_at);

    compile_prepared(
        normalized,
        tokenizer,
        vocab,
        profile_enabled,
        summary_enabled,
        compile_started_at,
        prepare_grammar_time,
    )
}

pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    let profile_enabled = compile_profile_enabled();
    let summary_enabled = compile_summary_enabled();
    let compile_started_at = std::time::Instant::now();

    let phase_started_at = std::time::Instant::now();
    let (normalized, tokenizer) = prepare_owned_grammar_for_compile(grammar);
    let prepare_grammar_time = phase_started_at.elapsed();
    log_compile_profile(profile_enabled, "prepare_grammar", phase_started_at);

    compile_prepared(
        normalized,
        tokenizer,
        vocab,
        profile_enabled,
        summary_enabled,
        compile_started_at,
        prepare_grammar_time,
    )
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
    let internal_token_entries = build_internal_token_entries(vocab, &id_map);
    let possible_matches_by_state =
        crate::compiler::possible_matches::build_possible_matches_from_owned_token_entries(
            &tokenizer,
            internal_token_entries,
        );

    let characterizations = characterize_terminals(&table, &glr_grammar);
    let templates = Templates::from_characterizations(&characterizations);
    let terminal_dwa = build_terminal_dwa(
        &glr_grammar,
        &tokenizer,
        vocab,
        &id_map,
        normalized.ignore_terminal,
    );

    log_terminal_dwa_sample_paths(&normalized, &terminal_dwa, vocab, &id_map);

    let parser_dwa = build_parser_dwa_from_terminal_dwa(
        &table,
        &glr_grammar,
        &tokenizer,
        &terminal_dwa,
    );

    log_parser_dwa_sample_paths(&parser_dwa, vocab, &id_map);

    let vocab_entries: Vec<(u32, Vec<u8>)> = vocab.entries.iter().map(|(token_id, bytes)| (*token_id, bytes.clone())).collect();
    let token_bytes = vocab.entries.clone();
    let mut constraint = Constraint {
        parser_dwa: parser_dwa.clone(),
        table: table.clone(),
        tokenizer: tokenizer.clone(),
        ignore_terminal: normalized.ignore_terminal,
        possible_matches: possible_matches_by_state.clone(),
        state_to_internal_tsid: id_map.tokenizer_states.original_to_internal.clone(),
        internal_tsid_to_states: id_map.tokenizer_states.internal_to_originals_vecs(),
        original_token_to_internal: id_map.vocab_tokens.original_to_internal.clone(),
        internal_token_to_tokens: id_map.vocab_tokens.internal_to_originals_vecs(),
        eos_token_id: vocab.eos_token_id,
        token_bytes: token_bytes.clone(),
        internal_token_bytes,
        token_bytes_dense: Vec::new(),
        internal_token_buf_masks: Vec::new(),
        internal_token_dense_words: 0,
        weight_token_dense_masks: rustc_hash::FxHashMap::default(),
        seed_terminal_dense: rustc_hash::FxHashMap::default(),
        seed_universe_dense: Box::new([]),
        dwa_fast_transitions: Vec::new(),
    };
    constraint.build_buf_masks();
    constraint.build_dense_token_bytes();
    constraint.build_dense_token_masks();
    constraint.build_fast_transitions();
    constraint.build_seed_dense_masks();

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
            .into_iter()
            .flat_map(|m| m.values())
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

}
