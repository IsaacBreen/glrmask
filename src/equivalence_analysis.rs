use crate::finite_automata::Regex;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::mem;

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

/// A fast, non-cryptographic hasher mixer (similar to FxHash).
const K: u64 = 0x517cc1b727220a95;

#[inline(always)]
fn hash_mix(h: &mut u64, data: u64) {
    *h = h.rotate_left(5) ^ data;
    *h = h.wrapping_mul(K);
}

/// Thread-local reusable context to eliminate allocations.
struct WorkerContext {
    // Maps position -> computed hash. Flat vector for O(1) access.
    memo: Vec<u64>,

    // Tracks visited status using generation counters to avoid O(N) clearing.
    visited_gen: Vec<u32>,
    current_gen: u32,

    // Reusable buffers
    discovery_stack: Vec<usize>,
    sorted_nodes: Vec<usize>,
    edge_buffer: Vec<(usize, u64)>,

    // Cached execution results to avoid recomputation across passes
    node_end_states: Vec<Option<usize>>,
    node_edges: Vec<Vec<(usize, usize)>>,
}

impl WorkerContext {
    fn new(capacity: usize) -> Self {
        // +1 because positions range from 0 to text.len() inclusive
        let cap = capacity + 1;
        Self {
            memo: vec![0; cap],
            visited_gen: vec![0; cap],
            current_gen: 0,
            discovery_stack: Vec::with_capacity(128),
            sorted_nodes: Vec::with_capacity(128),
            edge_buffer: Vec::with_capacity(16),
            node_end_states: vec![None; cap],
            node_edges: (0..cap).map(|_| Vec::with_capacity(4)).collect(),
        }
    }

    fn reset_for_text(&mut self, text_len: usize) {
        let required = text_len + 1;
        if required > self.visited_gen.len() {
            self.visited_gen.resize(required, 0);
            self.memo.resize(required, 0);
            self.node_end_states.resize(required, None);
            self
                .node_edges
                .resize_with(required, || Vec::with_capacity(4));
        }

        // Clear state for the range that will be used in this run.
        self.node_end_states[..required].fill(None);
        for edges in self.node_edges[..required].iter_mut() {
            edges.clear();
        }

        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.visited_gen.fill(0);
            self.current_gen = 1;
        }
    }
}

fn compute_hash(
    ctx: &mut WorkerContext,
    regex: &Regex,
    text: &[u8],
    start_state_id: usize,
) -> u64 {
    // === 1. Discover all reachable positions (DFS) ===
    // Note: We use the context's generation to track visited nodes for this specific run.
    // If compute_hash is called multiple times per text, we must increment gen each time
    // (handled by caller or here). Here we assume the context is prepared.

    ctx.discovery_stack.clear();
    ctx.sorted_nodes.clear();

    // Reset generation for this specific hash computation
    ctx.current_gen = ctx.current_gen.wrapping_add(1);
    if ctx.current_gen == 0 {
        ctx.visited_gen.fill(0);
        ctx.current_gen = 1;
    }
    let gen = ctx.current_gen;

    // Start DFS
    ctx.discovery_stack.push(0);
    // Safe because reset_for_text ensures capacity
    unsafe { *ctx.visited_gen.get_unchecked_mut(0) = gen; }

    while let Some(pos) = ctx.discovery_stack.pop() {
        if pos > text.len() {
            continue;
        }

        ctx.sorted_nodes.push(pos);

        let state = if pos == 0 {
            start_state_id
        } else {
            regex.dfa.start_state
        };
        // We assume execute_from_state_nonzero is a given API we cannot optimize further
        let res = regex.execute_from_state_nonzero(&text[pos..], state);
        ctx.node_end_states[pos] = res.end_state;

        let edges = &mut ctx.node_edges[pos];
        for m in &res.matches {
            edges.push((m.group_id, pos + m.position));
        }

        for &(_, next_pos) in edges.iter() {
            if next_pos <= text.len() {
                unsafe {
                    let v = ctx.visited_gen.get_unchecked_mut(next_pos);
                    if *v != gen {
                        *v = gen;
                        ctx.discovery_stack.push(next_pos);
                    }
                }
            }
        }
    }

    // === 2. Sort positions descending (Topological / Bottom-up) ===
    ctx.sorted_nodes.sort_unstable_by(|a, b| b.cmp(a));

    // === 3. Compute structural hashes ===
    // We swap sorted_nodes out to iterate it while mutating ctx fields
    let nodes = mem::take(&mut ctx.sorted_nodes);

    for &pos in &nodes {
        let res_end_state = ctx.node_end_states[pos];
        let edges = &mut ctx.node_edges[pos];

        ctx.edge_buffer.clear();

        // Collect edges
        for &(group_id, target_pos) in edges.iter() {
            let target_hash = if target_pos <= text.len() {
                unsafe { *ctx.memo.get_unchecked(target_pos) }
            } else {
                0
            };
            ctx.edge_buffer.push((group_id, target_hash));
        }

        // Sort for determinism
        ctx.edge_buffer.sort_unstable();

        let mut h = 0;

        // Mix Future Group IDs
        if let Some(id) = res_end_state {
            let futures = &regex.dfa.states[id].possible_future_group_ids;
            hash_mix(&mut h, futures.len() as u64); // Mix length to distinguish
            for &gid in futures {
                hash_mix(&mut h, gid as u64);
            }
        } else {
            hash_mix(&mut h, u64::MAX);
        }

        // Mix Edges
        hash_mix(&mut h, ctx.edge_buffer.len() as u64);
        for (gid, child_h) in &ctx.edge_buffer {
            hash_mix(&mut h, *gid as u64);
            hash_mix(&mut h, *child_h);
        }

        unsafe { *ctx.memo.get_unchecked_mut(pos) = h; }
    }

    // Restore sorted_nodes buffer for reuse
    ctx.sorted_nodes = nodes;

    unsafe { *ctx.memo.get_unchecked(0) }
}

pub fn find_equivalence_classes(regex: &Regex, strings: &[Vec<u8>], starts: &[usize]) -> EquivalenceResult {
    if strings.is_empty() {
        return BTreeSet::new();
    }

    // Allocate thread-local storage once
    let max_len = strings.iter().map(|s| s.len()).max().unwrap_or(0);

    // Compute signatures in parallel
    // Returns: Vec<(Signature, OriginalIndex)>
    let mut signatures: Vec<(u64, usize)> = strings
        .par_iter()
        .enumerate()
        .map_init(
            || WorkerContext::new(max_len),
            |ctx, (idx, s)| {
                ctx.reset_for_text(s.len());

                let mut combined_hash = 0;
                for &start_state in starts {
                    let h = compute_hash(ctx, regex, s, start_state);
                    hash_mix(&mut combined_hash, h);
                }
                (combined_hash, idx)
            },
        )
        .collect();

    // Group by signature
    // Sorting puts identical hashes together. Secondary sort by index ensures deterministic output.
    signatures.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut result = BTreeSet::new();
    if signatures.is_empty() {
        return result;
    }

    // Collect groups
    let mut chunk_start = 0;
    for i in 1..=signatures.len() {
        if i == signatures.len() || signatures[i].0 != signatures[chunk_start].0 {
            let group: Vec<usize> = signatures[chunk_start..i]
                .iter()
                .map(|(_, idx)| *idx)
                .collect();

            result.insert(group);
            chunk_start = i;
        }
    }

    result
}