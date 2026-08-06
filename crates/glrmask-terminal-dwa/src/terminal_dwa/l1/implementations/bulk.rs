//! Bulk adaptive determinization over all starts.
//!
//! Every `(trie node, lexer configuration)` pair is evaluated once. Converged
//! starts therefore share the whole remaining subtree, not merely its first-byte
//! entry. Per-node canonical profile IDs are the implicit weighted frontier.

use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::terminal_dwa::l1::implementations::support::{DEAD, FlatNode, Scanner, flatten, vocab};

#[derive(Default)]
struct Profiles {
    by_config: FxHashMap<u32, u32>,
    ids: FxHashMap<Vec<u32>, u32>,
    variants: Vec<Vec<u32>>,
}

fn profile(
    node: usize,
    config: u32,
    trie: &[FlatNode],
    profiles: &mut [Profiles],
    scanner: &mut Scanner<'_>,
    pair_visits: &mut usize,
    cache_hits: &mut usize,
) -> u32 {
    if config == DEAD {
        return 0;
    }
    if let Some(&id) = profiles[node].by_config.get(&config) {
        *cache_hits += 1;
        return id;
    }
    *pair_visits += 1;
    let mut key = Vec::with_capacity(trie[node].edges.len() + 1);
    key.push(trie[node].token.map_or(0, |_| scanner.signature(config)));
    for (edge, child) in &trie[node].edges {
        let target = scanner.step_bytes(config, edge);
        key.push(profile(*child, target, trie, profiles, scanner, pair_visits, cache_hits));
    }
    let id = if key.iter().all(|&part| part == 0) {
        0
    } else if let Some(&id) = profiles[node].ids.get(&key) {
        id
    } else {
        let id = profiles[node].variants.len() as u32 + 1;
        profiles[node].ids.insert(key.clone(), id);
        profiles[node].variants.push(key);
        id
    };
    profiles[node].by_config.insert(config, id);
    id
}

fn expand(
    node: usize,
    id: u32,
    trie: &[FlatNode],
    profiles: &[Profiles],
    row: &mut [u32],
) {
    if id == 0 {
        return;
    }
    let variant = &profiles[node].variants[id as usize - 1];
    if let Some(token) = trie[node].token {
        row[token] = variant[0];
    }
    for ((_, child), &child_id) in trie[node].edges.iter().zip(&variant[1..]) {
        expand(*child, child_id, trie, profiles, row);
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
    let mut profiles = (0..trie.len()).map(|_| Profiles::default()).collect::<Vec<_>>();
    let mut state_class = vec![0; input.tokenizer.num_states() as usize];
    let mut root_to_class = FxHashMap::<u32, u32>::default();
    let mut root_profiles = Vec::new();
    let mut pair_visits = 0usize;
    let mut cache_hits = 0usize;
    for (start, states) in starts {
        let root = profile(
            0,
            start,
            &trie,
            &mut profiles,
            &mut scanner,
            &mut pair_visits,
            &mut cache_hits,
        );
        let class = *root_to_class.entry(root).or_insert_with(|| {
            let class = root_profiles.len() as u32;
            root_profiles.push(root);
            class
        });
        for state in states {
            state_class[state as usize] = class;
        }
    }
    let mut rows = Vec::with_capacity(root_profiles.len());
    for root in root_profiles {
        let mut row = vec![0; aliases.len()];
        expand(0, root, &trie, &profiles, &mut row);
        rows.push(row);
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
    let profile_variants = profiles.iter().map(|node| node.variants.len()).sum::<usize>();
    let finished = common::finish(
        input,
        &aliases,
        &scanner.signatures,
        state_class,
        rows,
        scan_ms,
        || total.elapsed().as_secs_f64() * 1000.0,
    )?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_bulk] partition={} states={} tokens={} trie_nodes={} configs={} signatures={} pair_visits={} cache_hits={} profile_variants={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            aliases.len(),
            trie.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            pair_visits,
            cache_hits,
            profile_variants,
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
