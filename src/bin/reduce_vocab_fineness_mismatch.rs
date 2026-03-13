#![allow(unused)]

#[path = "../automata/mod.rs"]
pub(crate) mod automata;
#[path = "../compiler/mod.rs"]
pub(crate) mod compiler;
#[path = "../ds/mod.rs"]
pub(crate) mod ds;
#[path = "../error.rs"]
mod error;
#[path = "../grammar/mod.rs"]
pub(crate) mod grammar;
#[path = "../import/mod.rs"]
pub(crate) mod import;
#[path = "../runtime/mod.rs"]
pub(crate) mod runtime;
#[path = "../vocab.rs"]
mod vocab;

pub use compiler::debug::{AutomataDebug, CompileDebug, TerminalDebug};
pub use error::{GlrMaskError, Result};
pub use runtime::{Constraint, ConstraintState};
pub use vocab::Vocab;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use automata::regex::Expr;
use compiler::glr::analysis::AnalyzedGrammar;
use compiler::stages::equivalence_analysis::compat::Sep1Tokenizer;
use compiler::stages::equivalence_analysis::reference;
use compiler::stages::equivalence_analysis::state::fast as state_fast;
use compiler::stages::equivalence_analysis::vocab::fast as vocab_fast;
use ds::bitset::BitSet;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
struct TokenizerFixture {
    exprs: Vec<Expr>,
    disallowed_follows: BTreeMap<u32, Vec<u32>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ReducedMismatchFixture {
    left_token_id: Option<u32>,
    right_token_id: Option<u32>,
    left_token: Vec<u8>,
    right_token: Vec<u8>,
    fixture: TokenizerFixture,
}

#[derive(Clone)]
struct WitnessPair {
    left_id: u32,
    right_id: u32,
    left_token: Vec<u8>,
    right_token: Vec<u8>,
}

#[derive(Clone)]
struct Cli {
    lark_path: PathBuf,
    vocab_path: PathBuf,
    output_path: PathBuf,
    seed: u64,
    iterations: usize,
    left_id: Option<u32>,
    right_id: Option<u32>,
}

fn default_lark_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/github_hard_o56012_split_quotes.lark")
}

fn default_vocab_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../constraint-framework-analysis/.cache/vocab_cache/vocab.json")
}

fn default_output_path() -> PathBuf {
    PathBuf::from("/tmp/glrmask_reduced_vocab_mismatch_fixture.json")
}

fn parse_cli() -> Result<Cli> {
    let mut cli = Cli {
        lark_path: default_lark_path(),
        vocab_path: default_vocab_path(),
        output_path: default_output_path(),
        seed: 0,
        iterations: 10_000,
        left_id: None,
        right_id: None,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lark" => cli.lark_path = PathBuf::from(expect_arg(&mut args, "--lark")?),
            "--vocab" => cli.vocab_path = PathBuf::from(expect_arg(&mut args, "--vocab")?),
            "--output" => cli.output_path = PathBuf::from(expect_arg(&mut args, "--output")?),
            "--seed" => cli.seed = parse_u64(&expect_arg(&mut args, "--seed")?, "--seed")?,
            "--iterations" => {
                cli.iterations = parse_usize(&expect_arg(&mut args, "--iterations")?, "--iterations")?
            }
            "--left-id" => {
                cli.left_id = Some(parse_u32(&expect_arg(&mut args, "--left-id")?, "--left-id")?)
            }
            "--right-id" => {
                cli.right_id = Some(parse_u32(&expect_arg(&mut args, "--right-id")?, "--right-id")?)
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(GlrMaskError::Compilation(format!("unrecognized argument: {other}")));
            }
        }
    }

    if cli.left_id.is_some() ^ cli.right_id.is_some() {
        return Err(GlrMaskError::Compilation(
            "--left-id and --right-id must be provided together".to_string(),
        ));
    }

    Ok(cli)
}

fn print_help() {
    eprintln!(
        "reduce_vocab_fineness_mismatch\n\n\
         Randomly shrinks tokenizer terminal definitions while preserving a fast/reference\n\
         vocab-partition mismatch for a chosen token pair.\n\n\
         Options:\n\
           --lark PATH         Input Lark grammar (default: current o56012 fixture)\n\
           --vocab PATH        GPT-2 vocab json (default: CFA cache)\n\
           --left-id N         Left token ID to preserve\n\
           --right-id N        Right token ID to preserve\n\
           --iterations N      Random reduction attempts (default: 10000)\n\
           --seed N            RNG seed (default: 0)\n\
           --output PATH       Output fixture json\n"
    );
}

fn expect_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| GlrMaskError::Compilation(format!("missing value for {flag}")))
}

fn parse_u32(value: &str, flag: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|err| GlrMaskError::Compilation(format!("invalid value for {flag}: {err}")))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|err| GlrMaskError::Compilation(format!("invalid value for {flag}: {err}")))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|err| GlrMaskError::Compilation(format!("invalid value for {flag}: {err}")))
}

fn build_gpt2_unicode_to_byte_map() -> BTreeMap<char, u8> {
    let mut byte_values: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    byte_values.extend(0xA1u32..=0xACu32);
    byte_values.extend(0xAEu32..=0xFFu32);

    let mut unicode_values = byte_values.clone();
    let mut extra = 0u32;
    for byte in 0u32..=255u32 {
        if !byte_values.contains(&byte) {
            byte_values.push(byte);
            unicode_values.push(256 + extra);
            extra += 1;
        }
    }

    let mut unicode_to_byte = BTreeMap::new();
    for (byte, codepoint) in byte_values.into_iter().zip(unicode_values.into_iter()) {
        let ch = char::from_u32(codepoint).expect("valid GPT-2 codepoint");
        unicode_to_byte.insert(ch, byte as u8);
    }
    unicode_to_byte
}

fn load_cached_gpt2_vocab(path: &Path) -> Result<Vocab> {
    let raw = fs::read_to_string(path)
        .map_err(|err| GlrMaskError::Compilation(format!("failed to read {}: {err}", path.display())))?;
    let vocab: BTreeMap<String, u32> = serde_json::from_str(&raw)
        .map_err(|err| GlrMaskError::Compilation(format!("failed to parse {}: {err}", path.display())))?;
    let unicode_to_byte = build_gpt2_unicode_to_byte_map();
    let entries: Vec<(u32, Vec<u8>)> = vocab
        .into_iter()
        .map(|(token_str, token_id)| {
            let token_bytes: Vec<u8> = token_str
                .chars()
                .map(|ch| unicode_to_byte[&ch])
                .collect();
            (token_id, token_bytes)
        })
        .collect();
    Ok(Vocab::new(entries, None))
}

fn dense_vocab_tokens(vocab: &Vocab) -> Vec<Vec<u8>> {
    let max_id = vocab.max_token_id() as usize;
    let mut tokens = vec![Vec::new(); max_id + 1];
    for (&token_id, token_bytes) in &vocab.entries {
        tokens[token_id as usize] = token_bytes.clone();
    }
    tokens
}

fn terminal_to_expr(terminal: &grammar::flat::Terminal) -> Expr {
    match terminal {
        grammar::flat::Terminal::Literal { bytes, .. } => automata::regex::bytes(bytes),
        grammar::flat::Terminal::Pattern { pattern, utf8, .. } => {
            automata::lexer::regex::parse_regex(pattern, *utf8)
        }
        grammar::flat::Terminal::Expr { expr, .. } => expr.clone(),
    }
}

fn fixture_from_lark(path: &Path) -> Result<TokenizerFixture> {
    let lark = fs::read_to_string(path)
        .map_err(|err| GlrMaskError::Compilation(format!("failed to read {}: {err}", path.display())))?;
    let grammar = import::lark::parse_lark(&lark)?;
    let (normalized, _) = compiler::grammar::transforms::prepare_grammar_for_compile(&grammar);
    let analyzed = AnalyzedGrammar::from_grammar_def(&normalized);
    let disallowed = compiler::compile::compute_disallowed_follows(&analyzed);
    let exprs = normalized.terminals.iter().map(terminal_to_expr).collect();
    let disallowed_follows = bitset_map_to_vec_map(&disallowed);
    Ok(TokenizerFixture {
        exprs,
        disallowed_follows,
    })
}

fn bitset_map_to_vec_map(input: &BTreeMap<u32, BitSet>) -> BTreeMap<u32, Vec<u32>> {
    input
        .iter()
        .map(|(&gid, bits)| (gid, bits.iter().map(|bit| bit as u32).collect()))
        .collect()
}

fn vec_map_to_bitset_map(input: &BTreeMap<u32, Vec<u32>>, num_groups: usize) -> BTreeMap<u32, BitSet> {
    input
        .iter()
        .map(|(&gid, targets)| {
            let mut bits = BitSet::new(num_groups);
            for &target in targets {
                bits.set(target as usize);
            }
            (gid, bits)
        })
        .collect()
}

fn build_tokenizer_fixture(fixture: &TokenizerFixture) -> (Sep1Tokenizer, BTreeMap<u32, BitSet>) {
    let tokenizer = compiler::compile::build_tokenizer_from_exprs(&fixture.exprs);
    let sep1 = Sep1Tokenizer::new(&tokenizer);
    let disallowed = vec_map_to_bitset_map(&fixture.disallowed_follows, fixture.exprs.len());
    (sep1, disallowed)
}

fn reduced_states_for_tokens(tokens: &[Vec<u8>], sep1: &Sep1Tokenizer) -> Vec<usize> {
    let all_states: Vec<usize> = (0..sep1.dfa().states.len()).collect();
    let mapping = state_fast::find_state_equivalence_classes(sep1, tokens, &all_states);
    let mut reps = BTreeSet::new();
    for rep in mapping {
        reps.insert(rep);
    }
    reps.into_iter().collect()
}

fn pair_reproduces_mismatch(fixture: &TokenizerFixture, left: &[u8], right: &[u8]) -> bool {
    let (sep1, disallowed) = build_tokenizer_fixture(fixture);
    let pair_tokens = vec![left.to_vec(), right.to_vec()];
    let reduced_states = reduced_states_for_tokens(&pair_tokens, &sep1);
    let fast = vocab_fast::find_vocab_equivalence_classes_with_follow(
        &sep1,
        &pair_tokens,
        &reduced_states,
        &disallowed,
    );
    let reference = reference::find_equivalence_classes(
        &sep1,
        &pair_tokens,
        &reduced_states,
        &disallowed,
        None,
    );
    fast == BTreeSet::from([vec![0, 1]])
        && reference.vocab_classes == BTreeSet::from([vec![0], vec![1]])
}

fn find_simplest_witness_pair(fixture: &TokenizerFixture, vocab: &Vocab) -> Option<WitnessPair> {
    let (sep1, disallowed) = build_tokenizer_fixture(fixture);
    let full_tokens = dense_vocab_tokens(vocab);
    let reduced_states = reduced_states_for_tokens(&full_tokens, &sep1);
    let fast = vocab_fast::find_vocab_equivalence_classes_with_follow(
        &sep1,
        &full_tokens,
        &reduced_states,
        &disallowed,
    );
    let reference = reference::find_equivalence_classes(
        &sep1,
        &full_tokens,
        &reduced_states,
        &disallowed,
        None,
    );

    let mut reference_class_by_token = vec![usize::MAX; full_tokens.len()];
    for (class_index, class) in reference.vocab_classes.iter().enumerate() {
        for &token_index in class {
            reference_class_by_token[token_index] = class_index;
        }
    }

    let mut best: Option<(usize, usize, usize, usize)> = None;
    for class in &fast {
        for left_pos in 0..class.len() {
            for right_pos in (left_pos + 1)..class.len() {
                let left = class[left_pos];
                let right = class[right_pos];
                if reference_class_by_token[left] == reference_class_by_token[right] {
                    continue;
                }
                let left_len = full_tokens[left].len();
                let right_len = full_tokens[right].len();
                let score = (
                    left_len + right_len,
                    left_len.max(right_len),
                    left.min(right),
                    left.max(right),
                );
                if best.as_ref().is_none_or(|best_score| score < *best_score) {
                    best = Some(score);
                }
            }
        }
    }

    best.map(|(_, _, left, right)| WitnessPair {
        left_id: left as u32,
        right_id: right as u32,
        left_token: full_tokens[left].clone(),
        right_token: full_tokens[right].clone(),
    })
}

fn fixture_score(fixture: &TokenizerFixture) -> (usize, usize) {
    (fixture.exprs.len(), fixture.exprs.iter().map(expr_cost).sum())
}

fn expr_cost(expr: &Expr) -> usize {
    match expr {
        Expr::U8Seq(bytes) => bytes.len().max(1),
        Expr::U8Class(_) => 1,
        Expr::Seq(parts) | Expr::Choice(parts) => 1 + parts.iter().map(expr_cost).sum::<usize>(),
        Expr::Repeat { expr, .. } => 1 + expr_cost(expr),
        Expr::Shared(expr) => 1 + expr_cost(expr),
        Expr::Epsilon => 1,
    }
}

fn remove_terminals(fixture: &TokenizerFixture, keep: &[usize]) -> TokenizerFixture {
    let old_to_new: BTreeMap<usize, usize> = keep
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();
    let exprs = keep.iter().map(|&old_idx| fixture.exprs[old_idx].clone()).collect();
    let mut disallowed_follows = BTreeMap::new();

    for (&old_source, &new_source) in &old_to_new {
        let Some(targets) = fixture.disallowed_follows.get(&(old_source as u32)) else {
            continue;
        };
        let remapped: Vec<u32> = targets
            .iter()
            .filter_map(|&old_target| old_to_new.get(&(old_target as usize)).copied())
            .map(|target| target as u32)
            .collect();
        if !remapped.is_empty() {
            disallowed_follows.insert(new_source as u32, remapped);
        }
    }

    TokenizerFixture {
        exprs,
        disallowed_follows,
    }
}

fn greedy_single_terminal_prune(
    mut fixture: TokenizerFixture,
    left: &[u8],
    right: &[u8],
) -> TokenizerFixture {
    loop {
        let mut improved = false;
        for remove_idx in 0..fixture.exprs.len() {
            let keep: Vec<usize> = (0..fixture.exprs.len())
                .filter(|&idx| idx != remove_idx)
                .collect();
            if keep.is_empty() {
                continue;
            }
            let candidate = remove_terminals(&fixture, &keep);
            if pair_reproduces_mismatch(&candidate, left, right) {
                fixture = candidate;
                improved = true;
                break;
            }
        }
        if !improved {
            return fixture;
        }
    }
}

fn random_reduce(
    initial_fixture: TokenizerFixture,
    left: &[u8],
    right: &[u8],
    seed: u64,
    iterations: usize,
) -> TokenizerFixture {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best = greedy_single_terminal_prune(initial_fixture, left, right);
    let mut best_score = fixture_score(&best);

    for iteration in 0..iterations {
        if best.exprs.len() <= 1 {
            break;
        }
        let max_remove = best.exprs.len().saturating_sub(1).min(8).max(1);
        let remove_count = rng.gen_range(1..=max_remove);
        let mut indices: Vec<usize> = (0..best.exprs.len()).collect();
        indices.shuffle(&mut rng);
        let removed: BTreeSet<usize> = indices.into_iter().take(remove_count).collect();
        let keep: Vec<usize> = (0..best.exprs.len())
            .filter(|idx| !removed.contains(idx))
            .collect();
        if keep.is_empty() {
            continue;
        }

        let candidate = remove_terminals(&best, &keep);
        if !pair_reproduces_mismatch(&candidate, left, right) {
            continue;
        }

        let candidate = greedy_single_terminal_prune(candidate, left, right);
        let candidate_score = fixture_score(&candidate);
        if candidate_score < best_score {
            best = candidate;
            best_score = candidate_score;
            eprintln!(
                "[reduce] iteration={} improved terminals={} expr_cost={}",
                iteration,
                best_score.0,
                best_score.1,
            );
        }
    }

    best
}

fn main() -> Result<()> {
    let cli = parse_cli()?;
    let vocab = load_cached_gpt2_vocab(&cli.vocab_path)?;
    let fixture = fixture_from_lark(&cli.lark_path)?;

    let witness = match (cli.left_id, cli.right_id) {
        (Some(left_id), Some(right_id)) => {
            let left_token = vocab
                .entries
                .get(&left_id)
                .cloned()
                .ok_or_else(|| GlrMaskError::Compilation(format!("left token {left_id} not found in vocab")))?;
            let right_token = vocab
                .entries
                .get(&right_id)
                .cloned()
                .ok_or_else(|| GlrMaskError::Compilation(format!("right token {right_id} not found in vocab")))?;
            WitnessPair {
                left_id,
                right_id,
                left_token,
                right_token,
            }
        }
        _ => match find_simplest_witness_pair(&fixture, &vocab) {
            Some(witness) => witness,
            None => {
                eprintln!("no fast/reference fineness mismatch found in the supplied grammar/vocab");
                return Ok(());
            }
        },
    };

    eprintln!(
        "[reduce] witness left_id={} {:?} right_id={} {:?}",
        witness.left_id,
        String::from_utf8_lossy(&witness.left_token),
        witness.right_id,
        String::from_utf8_lossy(&witness.right_token),
    );

    if !pair_reproduces_mismatch(&fixture, &witness.left_token, &witness.right_token) {
        eprintln!(
            "the chosen token pair does not reproduce the mismatch on its own; \n\
             provide an explicit pair that does, or extend the reducer to keep a larger witness class"
        );
        return Ok(());
    }

    let reduced_fixture = random_reduce(
        fixture,
        &witness.left_token,
        &witness.right_token,
        cli.seed,
        cli.iterations,
    );

    let output = ReducedMismatchFixture {
        left_token_id: Some(witness.left_id),
        right_token_id: Some(witness.right_id),
        left_token: witness.left_token,
        right_token: witness.right_token,
        fixture: reduced_fixture.clone(),
    };
    let serialized = serde_json::to_string_pretty(&output)
        .map_err(|err| GlrMaskError::Serialization(format!("failed to serialize output fixture: {err}")))?;
    fs::write(&cli.output_path, serialized).map_err(|err| {
        GlrMaskError::Serialization(format!("failed to write {}: {err}", cli.output_path.display()))
    })?;

    let score = fixture_score(&reduced_fixture);
    eprintln!(
        "[reduce] wrote {} terminals={} expr_cost={} output={}",
        reduced_fixture.exprs.len(),
        score.0,
        score.1,
        cli.output_path.display(),
    );
    Ok(())
}