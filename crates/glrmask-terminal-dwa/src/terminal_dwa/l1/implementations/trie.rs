//! Concise optimized reference: share token prefixes and memoize scanner steps.

use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use super::{BuildInput, LocalIdMapTerminalDwa, common};
use crate::automata::lexer::Lexer;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

const UNKNOWN: u32 = u32::MAX - 1;
const DEAD: u32 = u32::MAX;

struct Scanner<'a> {
    input: BuildInput<'a>,
    configs: Vec<Box<[u32]>>,
    ids: FxHashMap<Vec<u32>, u32>,
    transitions: Vec<[u32; 256]>,
    signatures: Vec<Vec<u32>>,
    signature_ids: FxHashMap<Vec<u32>, u32>,
    config_signature: Vec<u32>,
}

impl<'a> Scanner<'a> {
    fn new(input: BuildInput<'a>) -> Self {
        Self {
            input,
            configs: Vec::new(),
            ids: FxHashMap::default(),
            transitions: Vec::new(),
            signatures: vec![Vec::new()],
            signature_ids: FxHashMap::from_iter([(Vec::new(), 0)]),
            config_signature: Vec::new(),
        }
    }

    fn intern(&mut self, mut states: Vec<u32>) -> u32 {
        if states.is_empty() { return DEAD; }
        states.sort_unstable(); states.dedup();
        if let Some(&id) = self.ids.get(&states) { return id; }
        let mut signature = states.iter().flat_map(|&state|
            super::super::collect_active_terminal_signature(
                self.input.tokenizer, state, self.input.active_terminals,
            )
        ).collect::<Vec<_>>();
        signature.sort_unstable(); signature.dedup();
        let next_signature = self.signatures.len() as u32;
        let signature_id = *self.signature_ids.entry(signature.clone()).or_insert_with(|| {
            self.signatures.push(signature); next_signature
        });
        let id = self.configs.len() as u32;
        self.ids.insert(states.clone(), id);
        self.configs.push(states.into_boxed_slice());
        self.transitions.push([UNKNOWN; 256]);
        self.config_signature.push(signature_id);
        id
    }

    fn start(&mut self, state: u32) -> u32 {
        self.intern(self.input.tokenizer.execute_from_state_end_only(&[], state).to_vec())
    }

    fn step(&mut self, config: u32, byte: u8) -> u32 {
        if config == DEAD { return DEAD; }
        let cached = self.transitions[config as usize][byte as usize];
        if cached != UNKNOWN { return cached; }
        let target = self.intern(self.input.tokenizer.step_all(&self.configs[config as usize], byte).to_vec());
        self.transitions[config as usize][byte as usize] = target;
        target
    }

    fn step_bytes(&mut self, mut config: u32, bytes: &[u8]) -> u32 {
        for &byte in bytes { config = self.step(config, byte); if config == DEAD { break; } }
        config
    }

    fn signature(&self, config: u32) -> u32 {
        if config == DEAD { 0 } else { self.config_signature[config as usize] }
    }
}

fn collect(scanner: &mut Scanner<'_>, node: &VocabPrefixTreeNode, config: u32, out: &mut Vec<(usize, u32)>) {
    let signature = scanner.signature(config);
    if node.has_token() && signature != 0 { out.push((node.token_id(), signature)); }
    for (edge, child) in node.iter_children() {
        let target = scanner.step_bytes(config, edge);
        if target != DEAD { collect(scanner, child, target, out); }
    }
}

fn vocab(input: BuildInput<'_>) -> (Vec<Vec<u32>>, VocabPrefixTree) {
    let mut entries = input.vocab.iter().map(|(id, bytes)| (bytes.to_vec(), id)).collect::<Vec<_>>();
    entries.sort_unstable();
    let mut bytes = Vec::<Vec<u8>>::new();
    let mut aliases = Vec::<Vec<u32>>::new();
    for (token, id) in entries {
        if bytes.last() == Some(&token) { aliases.last_mut().unwrap().push(id); }
        else { bytes.push(token); aliases.push(vec![id]); }
    }
    let refs = bytes.iter().enumerate().map(|(id, bytes)| (id, bytes.as_slice())).collect::<Vec<_>>();
    (aliases, VocabPrefixTree::build_presorted(&refs))
}

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() { return None; }
    let total = Instant::now();
    let (aliases, trie) = vocab(input);
    let scan = Instant::now();
    let mut scanner = Scanner::new(input);
    let mut starts = FxHashMap::<u32, Vec<u32>>::default();
    for state in 0..input.tokenizer.num_states() { starts.entry(scanner.start(state)).or_default().push(state); }
    let mut state_class = vec![0; input.tokenizer.num_states() as usize];
    let mut rows = Vec::<Vec<u32>>::new();
    let mut row_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut first_byte_cache = FxHashMap::<(usize, u32), Arc<[(usize, u32)]>>::default();
    let mut cache_hits = 0usize;
    for (start, states) in starts {
        let mut row = vec![0; aliases.len()];
        if trie.root.has_token() { row[trie.root.token_id()] = scanner.signature(start); }
        for (edge, child) in trie.root.iter_children() {
            let target = scanner.step_bytes(start, edge);
            if target == DEAD { continue; }
            let key = (child as *const VocabPrefixTreeNode as usize, target);
            let profile = if let Some(profile) = first_byte_cache.get(&key) {
                cache_hits += 1; Arc::clone(profile)
            } else {
                let mut profile = Vec::new(); collect(&mut scanner, child, target, &mut profile);
                let profile: Arc<[(usize, u32)]> = Arc::from(profile);
                first_byte_cache.insert(key, Arc::clone(&profile)); profile
            };
            for &(token, signature) in profile.iter() { row[token] = signature; }
        }
        let next = rows.len() as u32;
        let class = *row_ids.entry(row.clone()).or_insert_with(|| { rows.push(row); next });
        for state in states { state_class[state as usize] = class; }
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
    let finished = common::finish(input, &aliases, &scanner.signatures, state_class, rows, scan_ms,
        || total.elapsed().as_secs_f64() * 1000.0)?;
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_trie] partition={} states={} tokens={} configs={} signatures={} cache_entries={} cache_hits={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label, input.tokenizer.num_states(), aliases.len(), scanner.configs.len(),
            scanner.signatures.len(), first_byte_cache.len(), cache_hits, finished.state_classes,
            finished.token_classes, scan_ms, finished.compact_ms, finished.build_ms,
            total.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(finished.artifact)
}
