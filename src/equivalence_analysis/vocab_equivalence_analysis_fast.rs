//! Fast vocabulary equivalence analysis via iterative DFA signature refinement.
// Do NOT add caching shortcuts that skip states/tokens. Full correctness mandatory.

use crate::dfa_u8::Tokenizer;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::hash::{BuildHasher, Hasher};

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;
type EdgeList = SmallVec<[(usize, usize); 4]>;
type GroupList = SmallVec<[usize; 4]>;
type FinalizerList = SmallVec<[Finalizer; 4]>;
const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE_STATE: u32 = u32::MAX;
const NONE_POS: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Finalizer { gid: usize, non_greedy: bool }
#[derive(Clone, Copy, PartialEq)]
enum FutureMode { Terminate, Continue }

struct PrecomputedDfa {
    start_state: usize, transitions: Vec<[u32; 256]>, finalizers: Vec<FinalizerList>,
    future_modes: Vec<FutureMode>, has_transitions: Vec<bool>, num_groups: usize,
    completion_hash: Vec<u64>, none_completion_hash: u64,
}

struct Pos0Scratch {
    current_states: Vec<usize>, done: Vec<bool>, active_indices: Vec<usize>,
    end_states: Vec<Option<usize>>, match_positions: Vec<u32>, match_gen: Vec<u32>,
    cur_gen: u32, touched_groups: Vec<GroupList>, touched_states: Vec<usize>,
    base_offsets: Vec<usize>, seen_target: Vec<bool>, all_targets: Vec<usize>,
}

struct SuffixScratch {
    match_positions: Vec<u32>, visited: Vec<bool>, queue: Vec<usize>,
    order: Vec<usize>, nodes: Vec<Option<(u64, EdgeList)>>,
}

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));
#[inline] fn new_hasher() -> AHasher { HASH_RANDOM_STATE.build_hasher() }
#[inline] fn hash_group_list(list: &[usize]) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1); h.write_u64(list.len() as u64);
    for &v in list { h.write_u64(v as u64); }
    h.finish()
}

fn precompute_dfa(regex: &Tokenizer) -> PrecomputedDfa {
    let dfa = regex.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");
    let mut max_gid: Option<usize> = None;
    for s in &dfa.states {
        if let Some(m) = s.finalizers.iter().max() {
            max_gid = Some(max_gid.map_or(m, |c| c.max(m)));
        }
        if let Some(m) = s.possible_future_group_ids.iter().max() {
            max_gid = Some(max_gid.map_or(*m, |c| c.max(*m)));
        }
    }
    if let Some(m) = dfa.non_greedy_finalizers.iter().max() {
        max_gid = Some(max_gid.map_or(*m, |c| c.max(*m)));
    }
    let ng = max_gid.map(|m| m + 1).unwrap_or(0);
    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers: Vec<FinalizerList> = Vec::with_capacity(dfa.states.len());
    let mut possible_future: Vec<GroupList> = Vec::with_capacity(dfa.states.len());
    let mut has_transitions = Vec::with_capacity(dfa.states.len());
    for s in &dfa.states {
        let mut t = [NONE_STATE; 256];
        for (b, &tgt) in s.transitions.iter() { t[b as usize] = tgt as u32; }
        transitions.push(t);
        finalizers.push(s.finalizers.iter()
            .map(|gid| Finalizer { gid, non_greedy: false }).collect());
        possible_future.push(s.possible_future_group_ids.iter().copied().collect());
        has_transitions.push(!s.transitions.is_empty());
    }
    let mut ng_flags = vec![false; ng];
    for &gid in &dfa.non_greedy_finalizers { if gid < ng { ng_flags[gid] = true; } }
    for fs in &mut finalizers {
        for f in fs.iter_mut() { f.non_greedy = ng_flags.get(f.gid).copied().unwrap_or(false); }
    }
    let future_modes: Vec<FutureMode> = possible_future.iter()
        .map(|f| if f.is_empty() { FutureMode::Terminate } else { FutureMode::Continue }).collect();
    let none_ch = { let mut h = new_hasher(); h.write_u8(0); h.finish() };
    let ch: Vec<u64> = possible_future.iter().map(|v| hash_group_list(v)).collect();
    PrecomputedDfa {
        start_state: dfa.start_state, transitions, finalizers, future_modes,
        has_transitions, num_groups: ng, completion_hash: ch, none_completion_hash: none_ch,
    }
}

impl Pos0Scratch {
    fn new(ns: usize, ng: usize) -> Self {
        Pos0Scratch {
            current_states: vec![0; ns], done: vec![false; ns],
            active_indices: Vec::new(), end_states: vec![None; ns],
            match_positions: vec![0u32; ns.saturating_mul(ng)],
            match_gen: vec![0u32; ns.saturating_mul(ng)], cur_gen: 1,
            touched_groups: vec![GroupList::new(); ns], touched_states: Vec::new(),
            base_offsets: (0..ns).map(|i| i.saturating_mul(ng)).collect(),
            seen_target: Vec::new(), all_targets: Vec::new(),
        }
    }
    fn reset(&mut self, initial_states: &[usize], ng: usize) {
        let n = initial_states.len();
        if n > self.current_states.len() {
            self.current_states.resize(n, 0); self.done.resize(n, false);
            self.end_states.resize(n, None);
            self.match_positions.resize(n.saturating_mul(ng), 0);
            self.match_gen.resize(n.saturating_mul(ng), 0);
            self.touched_groups.resize(n, GroupList::new());
            self.base_offsets = (0..n).map(|i| i * ng).collect();
        }
        self.current_states[..n].clone_from_slice(initial_states);
        self.done.fill(false); self.active_indices.clear(); self.end_states[..n].fill(None);
        self.cur_gen = self.cur_gen.wrapping_add(1);
        if self.cur_gen == 0 { self.match_gen.fill(0); self.cur_gen = 1; }
        for &si in &self.touched_states {
            if si < self.touched_groups.len() { self.touched_groups[si].clear(); }
        }
        self.touched_states.clear();
    }
}

fn compute_pos0(
    pre: &PrecomputedDfa, sc: &mut Pos0Scratch, slice: &[u8], init: &[usize],
) {
    let (ns, ng, len) = (init.len(), pre.num_groups, slice.len());
    sc.reset(init, ng);
    let at = &mut sc.all_targets;
    let st = &mut sc.seen_target;
    for &p in at.iter() { if p < st.len() { st[p] = false; } }
    at.clear();
    if st.len() < len + 1 { st.resize(len + 1, false); }
    let cs = &mut sc.current_states;
    let done = &mut sc.done;
    let ai = &mut sc.active_indices;
    let mp = &mut sc.match_positions;
    let mg = &mut sc.match_gen;
    let cg = sc.cur_gen;
    let tg = &mut sc.touched_groups;
    let ts = &mut sc.touched_states;
    let bo = &sc.base_offsets;
    ai.clear();
    let has_bytes = !slice.is_empty();
    let fb = if has_bytes { slice[0] } else { 0 };
    for (i, &state) in init.iter().enumerate() {
        let base = bo[i];
        for f in &pre.finalizers[state] {
            if f.gid < ng && mg[base + f.gid] != cg {
                mg[base + f.gid] = cg; mp[base + f.gid] = 0;
                let g = &mut tg[i];
                if g.is_empty() { ts.push(i); }
                g.push(f.gid);
            }
        }
        if !pre.has_transitions[state] { done[i] = true; continue; }
        if has_bytes && pre.transitions[state][fb as usize] == NONE_STATE {
            done[i] = true; continue;
        }
        ai.push(i);
    }
    if has_bytes && !ai.is_empty() {
        let mut al = ai.len();
        for (pos, &byte) in slice.iter().enumerate() {
            let position = (pos + 1) as u32;
            let mut nl = 0usize;
            unsafe {
                for idx in 0..al {
                    let i = *ai.get_unchecked(idx);
                    let base = *bo.get_unchecked(i);
                    let next = *pre.transitions
                        .get_unchecked(*cs.get_unchecked(i))
                        .get_unchecked(byte as usize);
                    if next != NONE_STATE {
                        let next = next as usize;
                        *cs.get_unchecked_mut(i) = next;
                        for f in pre.finalizers.get_unchecked(next) {
                            if f.gid < ng {
                                let ix = base + f.gid;
                                let sg = mg.get_unchecked_mut(ix);
                                let wn = *sg != cg;
                                if !f.non_greedy || wn {
                                    *sg = cg;
                                    *mp.get_unchecked_mut(ix) = position;
                                }
                                if wn {
                                    let g = tg.get_unchecked_mut(i);
                                    if g.is_empty() { ts.push(i); }
                                    g.push(f.gid);
                                }
                            }
                        }
                        if *pre.future_modes.get_unchecked(next) == FutureMode::Terminate {
                            *done.get_unchecked_mut(i) = true;
                        }
                    } else {
                        *done.get_unchecked_mut(i) = true;
                    }
                    if !*done.get_unchecked(i) {
                        *ai.get_unchecked_mut(nl) = i; nl += 1;
                    }
                }
            }
            al = nl;
            if al == 0 { break; }
        }
    }
    for i in 0..ns {
        sc.end_states[i] = if done[i] || !pre.has_transitions[cs[i]] { None }
            else { Some(cs[i]) };
        if ng > 0 {
            let base = bo[i];
            for &gid in &tg[i] {
                let pv = mp[base + gid];
                if pv > 0 {
                    let p = pv as usize;
                    if p <= len && !st[p] { st[p] = true; at.push(p); }
                }
            }
        }
    }
}

impl SuffixScratch {
    fn new(ng: usize) -> Self {
        SuffixScratch {
            match_positions: vec![NONE_POS; ng], visited: Vec::new(),
            queue: Vec::new(), order: Vec::new(), nodes: Vec::new(),
        }
    }
    fn ensure_capacity(&mut self, len: usize) {
        let n = len + 1;
        for &p in &self.queue {
            if p < self.visited.len() { self.visited[p] = false; }
            if p < self.nodes.len() { self.nodes[p] = None; }
        }
        if self.visited.len() < n { self.visited.resize(n, false); }
        if self.nodes.len() < n { self.nodes.resize(n, None); }
        self.queue.clear(); self.order.clear();
    }
}

fn execute_suffix(
    pre: &PrecomputedDfa, slice: &[u8], base_pos: usize, mpos: &mut [u32],
) -> (Option<usize>, EdgeList) {
    let ng = pre.num_groups;
    mpos[..ng].fill(NONE_POS);
    let mut touched = GroupList::new();
    let mut cur = pre.start_state;
    let mut done = !pre.has_transitions[cur];
    for f in &pre.finalizers[cur] {
        if f.gid < ng && mpos[f.gid] == NONE_POS { mpos[f.gid] = 0; touched.push(f.gid); }
    }
    for (idx, &byte) in slice.iter().enumerate() {
        if done { break; }
        let ns = pre.transitions[cur][byte as usize];
        if ns == NONE_STATE { done = true; break; }
        cur = ns as usize;
        let pos = (idx + 1) as u32;
        for f in &pre.finalizers[cur] {
            if f.gid < ng {
                let wn = mpos[f.gid] == NONE_POS;
                if !f.non_greedy || wn { mpos[f.gid] = pos; }
                if wn { touched.push(f.gid); }
            }
        }
        if pre.future_modes[cur] == FutureMode::Terminate { done = true; }
    }
    let end = if done || !pre.has_transitions[cur] { None } else { Some(cur) };
    touched.sort_unstable();
    let edges: EdgeList = touched.iter()
        .filter_map(|&g| (mpos[g] != NONE_POS && mpos[g] != 0)
            .then(|| (g, base_pos + mpos[g] as usize)))
        .collect();
    (end, edges)
}

fn compute_suffix_hashes(
    pre: &PrecomputedDfa, slice: &[u8], targets: &[usize],
    cache: &mut Vec<Option<u64>>, sc: &mut SuffixScratch,
) {
    sc.ensure_capacity(slice.len());
    for &pos in targets {
        if pos <= slice.len() && sc.nodes[pos].is_none() && !sc.visited[pos] {
            sc.visited[pos] = true; sc.queue.push(pos);
        }
    }
    if sc.queue.is_empty() { return; }
    let mut cursor = 0;
    while cursor < sc.queue.len() {
        let pos = sc.queue[cursor]; cursor += 1;
        let (es, edges) = execute_suffix(pre, &slice[pos..], pos, &mut sc.match_positions);
        for &(_, t) in &edges {
            if t <= slice.len() && sc.nodes[t].is_none() && !sc.visited[t] {
                sc.visited[t] = true; sc.queue.push(t);
            }
        }
        let ch = es.map(|id| pre.completion_hash[id]).unwrap_or(pre.none_completion_hash);
        sc.nodes[pos] = Some((ch, edges)); sc.order.push(pos);
    }
    sc.order.sort_unstable_by(|a, b| b.cmp(a));
    for &pos in &sc.order {
        if cache[pos].is_some() { continue; }
        if let Some((ch, ref edges)) = sc.nodes[pos] {
            let mut h = new_hasher(); h.write_u64(ch);
            for &(gid, t) in edges.iter() {
                h.write_u64(gid as u64); h.write_u64(cache[t].unwrap_or(0));
            }
            cache[pos] = Some(h.finish());
        }
    }
    sc.order.clear();
}

fn compute_chunk_signature(
    pre: &PrecomputedDfa, token: &[u8], states: &[usize],
    p0: &mut Pos0Scratch, sf: &mut SuffixScratch, cache: &mut Vec<Option<u64>>,
) -> u64 {
    compute_pos0(pre, p0, token, states);
    if !p0.all_targets.is_empty() {
        compute_suffix_hashes(pre, token, &p0.all_targets, cache, sf);
    }
    let mut sig: u64 = HASH_SEED3;
    for i in 0..states.len() {
        let ch = p0.end_states[i]
            .map(|id| pre.completion_hash[id]).unwrap_or(pre.none_completion_hash);
        let ss = if pre.num_groups > 0 && !p0.touched_groups[i].is_empty() {
            let g = &mut p0.touched_groups[i];
            if g.len() > 1 { g.sort_unstable(); }
            let base = p0.base_offsets[i];
            let mut h = new_hasher(); h.write_u64(ch);
            for &gid in g.iter() {
                let pv = p0.match_positions[base + gid];
                if pv > 0 { h.write_u64(gid as u64); h.write_u64(cache[pv as usize].unwrap_or(0)); }
            }
            h.finish()
        } else { ch };
        sig = sig.wrapping_mul(HASH_SEED1).wrapping_add(ss);
    }
    sig
}

pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer, strings: &[S], initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes. `suffix_group_mask`, `ever_allowed_by_group`,
/// and `group_to_class` are accepted for API compatibility but unused internally.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer, strings: &[S], initial_states: &[usize],
    _: Option<&[bool]>, _: Option<&[Vec<bool>]>, _: Option<&[usize]>,
) -> VocabEquivalenceResult {
    let pre = precompute_dfa(regex);
    let (nt, ns) = (strings.len(), initial_states.len());
    if ns == 0 || nt == 0 { return BTreeSet::from_iter(vec![(0..nt).collect()]); }
    let ng = pre.num_groups;
    let bs = if ns < 200 { ns } else { 200 };
    let mut active: Vec<usize> = (0..nt).collect();
    let mut partition = vec![0usize; nt];
    let mut next_id = 1usize;
    for batch_start in (0..ns).step_by(bs) {
        if active.is_empty() { break; }
        let batch = &initial_states[batch_start..(batch_start + bs).min(ns)];
        let sigs: Vec<(usize, u64)> = active.par_iter()
            .map_init(
                || (Pos0Scratch::new(batch.len(), ng), SuffixScratch::new(ng), vec![None; 256]),
                |st, &ti| {
                    let (p0, sf, sc) = st;
                    let tok = strings[ti].as_ref();
                    if sc.len() <= tok.len() { sc.resize(tok.len() + 1, None); }
                    sc.iter_mut().for_each(|x| *x = None);
                    (ti, compute_chunk_signature(&pre, tok, batch, p0, sf, sc))
                },
            ).collect();
        let mut refine: HashMap<(usize, u64), Vec<usize>> =
            HashMap::with_capacity(sigs.len() / 2);
        for (ti, sig) in sigs { refine.entry((partition[ti], sig)).or_default().push(ti); }
        let mut by_old: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
        for ((oc, _), toks) in refine { by_old.entry(oc).or_default().push(toks); }
        let mut new_active = Vec::with_capacity(active.len());
        for (oc, subs) in by_old {
            let mut first = true;
            for toks in subs {
                let cid = if first { first = false; oc }
                    else { let id = next_id; next_id += 1; id };
                for &ti in &toks { partition[ti] = cid; }
                if toks.len() > 1 { new_active.extend(toks); }
            }
        }
        active = new_active;
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::with_capacity(next_id);
    for (ti, &cid) in partition.iter().enumerate() { groups.entry(cid).or_default().push(ti); }
    groups.into_values().collect()
}
