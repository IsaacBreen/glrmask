//! Concise optimized reference: share token prefixes and memoize subtree walks.

use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::terminal_dwa::l1::implementations::support::{DEAD, Scanner, vocab};

fn collect(
    scanner: &mut Scanner<'_>,
    node: &VocabPrefixTreeNode,
    config: u32,
    out: &mut Vec<(usize, u32)>,
) {
    let signature = scanner.signature(config);
    if node.has_token() && signature != 0 {
        out.push((node.token_id(), signature));
    }
    for (edge, child) in node.iter_children() {
        let target = scanner.step_bytes(config, edge);
        if target != DEAD {
            collect(scanner, child, target, out);
        }
    }
}

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total = Instant::now();
    let (aliases, trie) = vocab(input);
    let scan = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut starts = std::collections::BTreeMap::<u32, Vec<u32>>::new();
    for state in 0..input.tokenizer.num_states() {
        starts.entry(scanner.start(state)).or_default().push(state);
    }
    let children = trie.root.iter_children().collect::<Vec<_>>();
    let mut state_class = vec![0; input.tokenizer.num_states() as usize];
    let mut class_profiles = Vec::<Vec<u32>>::new();
    let mut class_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut profiles = vec![Arc::<[(usize, u32)]>::from([])];
    let mut profile_ids = FxHashMap::<Arc<[(usize, u32)]>, u32>::default();
    let mut first_byte_cache = FxHashMap::<(usize, u32), u32>::default();
    let mut cache_hits = 0usize;
    for (start, states) in starts {
        let mut fingerprint = Vec::with_capacity(children.len() + 1);
        fingerprint.push(if trie.root.has_token() { scanner.signature(start) } else { 0 });
        for &(edge, child) in &children {
            let target = scanner.step_bytes(start, edge);
            if target == DEAD {
                fingerprint.push(0);
                continue;
            }
            let key = (child as *const VocabPrefixTreeNode as usize, target);
            let profile = if let Some(&profile) = first_byte_cache.get(&key) {
                cache_hits += 1;
                profile
            } else {
                let mut values = Vec::new();
                collect(&mut scanner, child, target, &mut values);
                let values: Arc<[(usize, u32)]> = Arc::from(values);
                let profile = if values.is_empty() {
                    0
                } else if let Some(&profile) = profile_ids.get(&values) {
                    profile
                } else {
                    let profile = profiles.len() as u32;
                    profile_ids.insert(Arc::clone(&values), profile);
                    profiles.push(values);
                    profile
                };
                first_byte_cache.insert(key, profile);
                profile
            };
            fingerprint.push(profile);
        }
        let next = class_profiles.len() as u32;
        let class = *class_ids.entry(fingerprint.clone()).or_insert_with(|| {
            class_profiles.push(fingerprint);
            next
        });
        for state in states {
            state_class[state as usize] = class;
        }
    }
    let mut rows = Vec::with_capacity(class_profiles.len());
    for fingerprint in &class_profiles {
        let mut row = vec![0; aliases.len()];
        if trie.root.has_token() {
            row[trie.root.token_id()] = fingerprint[0];
        }
        for &profile in &fingerprint[1..] {
            for &(token, signature) in profiles[profile as usize].iter() {
                row[token] = signature;
            }
        }
        rows.push(row);
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
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
            "[glrmask/profile][l1_trie] partition={} states={} tokens={} configs={} signatures={} cache_entries={} cache_hits={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            aliases.len(),
            scanner.configs.len(),
            scanner.signatures.len(),
            first_byte_cache.len(),
            cache_hits,
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
