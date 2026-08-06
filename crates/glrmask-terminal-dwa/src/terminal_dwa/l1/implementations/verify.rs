//! Exact quotient-level semantic comparison.
//!
//! Compare only unique `(left TSID, right TSID)` and
//! `(left token class, right token class)` coordinate pairs. Each artifact is
//! first converted to an exact interned terminal-signature matrix over its own
//! compact coordinates. This proves equality of every raw
//! `(state, original token, terminal)` tuple without expanding those tuples.

use rustc_hash::FxHashMap;

use super::{BuildInput, Implementation, LocalIdMapTerminalDwa};
use crate::automata::lexer::Lexer;

#[derive(Default)]
struct Signatures {
    ids: FxHashMap<(u32, u32), u32>,
    nodes: Vec<(u32, u32)>,
}

impl Signatures {
    fn push(&mut self, previous: u32, terminal: u32) -> u32 {
        if let Some(&id) = self.ids.get(&(previous, terminal)) {
            return id;
        }
        let id = self.nodes.len() as u32 + 1;
        self.ids.insert((previous, terminal), id);
        self.nodes.push((previous, terminal));
        id
    }

    fn expand(&self, mut id: u32) -> Vec<u32> {
        let mut terminals = Vec::new();
        while id != 0 {
            let (previous, terminal) = self.nodes[id as usize - 1];
            terminals.push(terminal);
            id = previous;
        }
        terminals.reverse();
        terminals
    }
}

struct Matrix {
    cells: Vec<u32>,
    tokens: usize,
}

fn matrix(
    artifact: &LocalIdMapTerminalDwa,
    num_terminals: u32,
    signatures: &mut Signatures,
) -> Matrix {
    let tsids = artifact.id_map.num_tsids() as usize;
    let tokens = artifact.id_map.num_internal_tokens() as usize;
    let mut cells = vec![0u32; tsids * tokens];

    for terminal in 0..num_terminals {
        let weight = artifact.dwa.eval_word(&[terminal as i32]);
        if weight.is_empty() {
            continue;
        }
        let mut next = FxHashMap::<u32, u32>::default();
        let mut update = |cell: &mut u32| {
            let previous = *cell;
            *cell = *next
                .entry(previous)
                .or_insert_with(|| signatures.push(previous, terminal));
        };
        if weight.is_full() {
            cells.iter_mut().for_each(&mut update);
            continue;
        }
        for (range, allowed) in weight.raw_range_values() {
            for tsid in *range.start() as usize..=*range.end() as usize {
                let row = &mut cells[tsid * tokens..(tsid + 1) * tokens];
                for token in allowed.iter() {
                    update(&mut row[token as usize]);
                }
            }
        }
    }
    Matrix { cells, tokens }
}

fn original_to_internal(artifact: &LocalIdMapTerminalDwa, original_count: usize) -> Vec<u32> {
    let mut result = vec![u32::MAX; original_count];
    if let Some(originals) = artifact.id_map.deferred_vocab_singleton_original_ids.as_ref() {
        for (internal, &original) in originals.iter().enumerate() {
            if original as usize >= result.len() {
                result.resize(original as usize + 1, u32::MAX);
            }
            result[original as usize] = internal as u32;
        }
    } else {
        let source = &artifact.id_map.vocab_tokens.original_to_internal;
        let len = source.len().min(result.len());
        result[..len].copy_from_slice(&source[..len]);
    }
    result
}

pub(super) fn assert_equivalent(
    input: BuildInput<'_>,
    actual_name: Implementation,
    actual: Option<&LocalIdMapTerminalDwa>,
    expected_name: Implementation,
    expected: Option<&LocalIdMapTerminalDwa>,
) {
    let (actual, expected) = match (actual, expected) {
        (None, None) => return,
        (Some(actual), Some(expected)) => (actual, expected),
        _ => panic!(
            "L1 implementation mismatch in partition {}: {:?} returned {}, {:?} returned {}",
            input.partition_label,
            actual_name,
            if actual.is_some() { "a DWA" } else { "None" },
            expected_name,
            if expected.is_some() { "a DWA" } else { "None" },
        ),
    };

    let mut signatures = Signatures::default();
    let actual_matrix = matrix(actual, input.grammar.num_terminals, &mut signatures);
    let expected_matrix = matrix(expected, input.grammar.num_terminals, &mut signatures);

    let mut state_pairs = FxHashMap::<(u32, u32), u32>::default();
    for raw in 0..input.tokenizer.num_states() {
        let pair = (
            actual.id_map.tokenizer_states.original_to_internal[raw as usize],
            expected.id_map.tokenizer_states.original_to_internal[raw as usize],
        );
        state_pairs.entry(pair).or_insert(raw);
    }

    let original_count = input.vocab.max_token_id() as usize + 1;
    let actual_tokens = original_to_internal(actual, original_count);
    let expected_tokens = original_to_internal(expected, original_count);
    let mut token_pairs = FxHashMap::<(u32, u32), u32>::default();
    for (original, _) in input.vocab.iter() {
        let pair = (actual_tokens[original as usize], expected_tokens[original as usize]);
        assert!(
            pair.0 != u32::MAX && pair.1 != u32::MAX,
            "L1 implementation omitted original token {original} in partition {}",
            input.partition_label,
        );
        token_pairs.entry(pair).or_insert(original);
    }

    for (&(actual_tsid, expected_tsid), &raw_state) in &state_pairs {
        let actual_row = &actual_matrix.cells
            [actual_tsid as usize * actual_matrix.tokens..(actual_tsid as usize + 1) * actual_matrix.tokens];
        let expected_row = &expected_matrix.cells[expected_tsid as usize * expected_matrix.tokens
            ..(expected_tsid as usize + 1) * expected_matrix.tokens];
        for (&(actual_token, expected_token), &original_token) in &token_pairs {
            let actual_signature = actual_row[actual_token as usize];
            let expected_signature = expected_row[expected_token as usize];
            assert_eq!(
                actual_signature,
                expected_signature,
                "L1 mismatch partition={} raw_state={} original_token={} actual={:?} expected={:?} actual_terminals={:?} expected_terminals={:?}",
                input.partition_label,
                raw_state,
                original_token,
                actual_name,
                expected_name,
                signatures.expand(actual_signature),
                signatures.expand(expected_signature),
            );
        }
    }

    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_verify] partition={} state_pairs={} token_pairs={} actual_cells={} expected_cells={} signatures={}",
            input.partition_label,
            state_pairs.len(),
            token_pairs.len(),
            actual_matrix.cells.len(),
            expected_matrix.cells.len(),
            signatures.nodes.len() + 1,
        );
    }
}
