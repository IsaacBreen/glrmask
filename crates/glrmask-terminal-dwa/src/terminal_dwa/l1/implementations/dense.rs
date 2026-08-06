//! Cache-sized dense batches of start configurations over one flattened trie.

use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::terminal_dwa::l1::implementations::support::{DEAD, FlatNode, Scanner, flatten, vocab};

fn walk(
    node: usize,
    configs: &[u32],
    trie: &[FlatNode],
    scanner: &mut Scanner<'_>,
    rows: &mut [Vec<u32>],
    frontier_cells: &mut usize,
) {
    *frontier_cells += configs.len();
    if let Some(token) = trie[node].token {
        for (row, &config) in rows.iter_mut().zip(configs) {
            row[token] = scanner.signature(config);
        }
    }
    let mut next = vec![DEAD; configs.len()];
    for (edge, child) in &trie[node].edges {
        let mut live = false;
        for (target, &config) in next.iter_mut().zip(configs) {
            *target = scanner.step_bytes(config, edge);
            live |= *target != DEAD;
        }
        if live {
            walk(*child, &next, trie, scanner, rows, frontier_cells);
        }
    }
}

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let (aliases, tree) = vocab(input);
    let trie = flatten(&tree);
    let scan = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut starts = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for state in 0..input.tokenizer.num_states() {
        starts.entry(scanner.start(state)).or_default().push(state);
    }
    let starts = starts.into_iter().collect::<Vec<_>>();
    let chunk = std::env::var("GLRMASK_L1_DENSE_CHUNK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128usize)
        .max(1);
    let mut state_class = vec![0; input.tokenizer.num_states() as usize];
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut unique_rows = Vec::<Vec<u32>>::new();
    let mut frontier_cells = 0usize;
    for batch in starts.chunks(chunk) {
        let configs = batch.iter().map(|(config, _)| *config).collect::<Vec<_>>();
        let mut rows = vec![vec![0; aliases.len()]; batch.len()];
        walk(0, &configs, &trie, &mut scanner, &mut rows, &mut frontier_cells);
        for ((_, raw_states), row) in batch.iter().zip(rows) {
            let class = if let Some(&class) = row_ids.get(&row) {
                class
            } else {
                let class = unique_rows.len() as u32;
                row_ids.insert(row.clone(), class);
                unique_rows.push(row);
                class
            };
            for &state in raw_states {
                state_class[state as usize] = class;
            }
        }
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish(
        input,
        &aliases,
        &scanner.signatures,
        state_class,
        unique_rows,
        scan_ms,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_dense] partition={} states={} tokens={} trie_nodes={} configs={} signatures={} chunk={} frontier_cells={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            aliases.len(),
            trie.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            chunk,
            frontier_cells,
            finished.state_classes,
            finished.token_classes,
            scan_ms,
            finished.compact_ms,
            finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}
