//! Weighted all-start execution with a compact `u64` origin mask.
//!
//! A frontier entry is `(lexer configuration, origins)`. Equal targets merge by
//! OR-ing their origin masks after every trie edge, exactly matching the proposed
//! weighted-NWA execution without constructing or determinizing that NWA.

use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::terminal_dwa::l1::implementations::support::{DEAD, FlatNode, Scanner, flatten, vocab};

fn walk(
    node: usize,
    frontier: &[(u32, u64)],
    trie: &[FlatNode],
    scanner: &mut Scanner<'_>,
    rows: &mut [Vec<u32>],
    group_visits: &mut usize,
    origin_visits: &mut usize,
    merges: &mut usize,
) {
    *group_visits += frontier.len();
    *origin_visits += frontier.iter().map(|(_, origins)| origins.count_ones() as usize).sum::<usize>();
    if let Some(token) = trie[node].token {
        for &(config, mut origins) in frontier {
            let signature = scanner.signature(config);
            while origins != 0 {
                let origin = origins.trailing_zeros() as usize;
                rows[origin][token] = signature;
                origins &= origins - 1;
            }
        }
    }
    for (edge, child) in &trie[node].edges {
        let mut next = Vec::<(u32, u64)>::with_capacity(frontier.len());
        for &(config, origins) in frontier {
            let target = scanner.step_bytes(config, edge);
            if target == DEAD {
                continue;
            }
            if let Some((_, existing)) = next.iter_mut().find(|(candidate, _)| *candidate == target) {
                *existing |= origins;
                *merges += 1;
            } else {
                next.push((target, origins));
            }
        }
        if !next.is_empty() {
            walk(
                *child,
                &next,
                trie,
                scanner,
                rows,
                group_visits,
                origin_visits,
                merges,
            );
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
    let chunk = std::env::var("GLRMASK_L1_FRONTIER_CHUNK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64usize)
        .clamp(1, 64);
    let mut state_class = vec![0; input.tokenizer.num_states() as usize];
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut unique_rows = Vec::<Vec<u32>>::new();
    let mut group_visits = 0usize;
    let mut origin_visits = 0usize;
    let mut merges = 0usize;
    for batch in starts.chunks(chunk) {
        let frontier = batch
            .iter()
            .enumerate()
            .map(|(origin, (config, _))| (*config, 1u64 << origin))
            .collect::<Vec<_>>();
        let mut rows = vec![vec![0; aliases.len()]; batch.len()];
        walk(
            0,
            &frontier,
            &trie,
            &mut scanner,
            &mut rows,
            &mut group_visits,
            &mut origin_visits,
            &mut merges,
        );
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
            "[glrmask/profile][l1_frontier] partition={} states={} tokens={} trie_nodes={} configs={} signatures={} chunk={} group_visits={} origin_visits={} merges={} convergence={:.3} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            aliases.len(),
            trie.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            chunk,
            group_visits,
            origin_visits,
            merges,
            if origin_visits == 0 { 1.0 } else { group_visits as f64 / origin_visits as f64 },
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
