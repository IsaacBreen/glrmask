#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NWAState as NWAStateType};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::grammar::model::Symbol;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::PossibleMatchesComputer;

/// NWA state identifier (index into `NWA.states`).
type NwaState = u32;
/// Tokenizer state identifier.
type TokenizerState = u32;
use crate::compiler::compile::compute_disallowed_follows;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::equivalence_analysis::reference::{
    build_disallowed_follow_dfa, normalize_disallowed_follows,
};
use crate::compiler::stages::profile_stats::{
    WeightedDwaStats,
    WeightedNwaStats,
    collect_weighted_dwa_stats,
    collect_weighted_nwa_stats,
};
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;

#[derive(Debug, Clone, Default)]
pub(crate) struct TerminalDwaBuildReport {
    pub build_vocab_trie_time: std::time::Duration,
    pub build_nwa_from_trie_time: std::time::Duration,
    pub collapse_always_allowed_time: std::time::Duration,
    pub collapse_always_allowed_applied: bool,
    pub subtract_disallowed_time: std::time::Duration,
    pub determinize_time: std::time::Duration,
    pub minimize_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub vocab_entries: usize,
    pub internal_tsids: usize,
    pub terminal_nwa: WeightedNwaStats,
    pub terminal_dwa: WeightedDwaStats,
    pub terminal_minimized_dwa: WeightedDwaStats,
}

pub(crate) fn compute_ever_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
    let mut ever_allowed = vec![BTreeSet::new(); grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }
            ever_allowed[*terminal as usize]
                .extend(occurrence_follow_set(grammar, rule.lhs, &rule.rhs, index));
        }
    }

    ever_allowed
        .into_iter()
        .map(|set| set.into_iter().collect())
        .collect()
}

fn compute_always_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
    let mut always_allowed = vec![None::<BTreeSet<TerminalID>>; grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }

            let follows = occurrence_follow_set(grammar, rule.lhs, &rule.rhs, index);
            match &mut always_allowed[*terminal as usize] {
                None => always_allowed[*terminal as usize] = Some(follows),
                Some(existing) => existing.retain(|follow| follows.contains(follow)),
            }
        }
    }

    always_allowed
        .into_iter()
        .map(|set| set.unwrap_or_default().into_iter().collect())
        .collect()
}

fn occurrence_follow_set(
    grammar: &AnalyzedGrammar,
    lhs: u32,
    rhs: &[Symbol],
    index: usize,
) -> BTreeSet<TerminalID> {
    let mut follows = BTreeSet::new();
    let mut suffix_nullable = true;

    for symbol in &rhs[index + 1..] {
        match symbol {
            Symbol::Terminal(terminal) => {
                if *terminal < grammar.num_terminals {
                    follows.insert(*terminal);
                }
                suffix_nullable = false;
                break;
            }
            Symbol::Nonterminal(nonterminal) => {
                if let Some(first) = grammar.first.get(*nonterminal as usize) {
                    follows.extend(first.iter().copied().filter(|terminal| *terminal != EOF));
                }
                if !grammar.nullable.contains(nonterminal) {
                    suffix_nullable = false;
                    break;
                }
            }
        }
    }

    if suffix_nullable {
        if let Some(follow) = grammar.follow.get(lhs as usize) {
            follows.extend(follow.iter().copied().filter(|terminal| *terminal != EOF));
        }
    }

    follows
}

/// Compute a structural hash of an NWAState without string allocation.
/// Uses the same logical content as PartialEq — transitions, epsilons, final_weight —
/// but feeds it directly into a Hasher for O(1) memory overhead.
fn structural_hash_nwa_state(state: &NWAStateType) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Hash final_weight
    state.final_weight.is_some().hash(&mut hasher);
    if let Some(w) = &state.final_weight {
        hash_weight(w, &mut hasher);
    }

    // Hash transitions (BTreeMap iterates in sorted key order)
    state.transitions.len().hash(&mut hasher);
    for (label, targets) in &state.transitions {
        label.hash(&mut hasher);
        targets.len().hash(&mut hasher);
        for (target, weight) in targets {
            target.hash(&mut hasher);
            hash_weight(weight, &mut hasher);
        }
    }

    // Hash epsilons
    state.epsilons.len().hash(&mut hasher);
    for (target, weight) in &state.epsilons {
        target.hash(&mut hasher);
        hash_weight(weight, &mut hasher);
    }

    hasher.finish()
}

/// Feed Weight contents into a Hasher by iterating its range-value structure.
fn hash_weight(weight: &Weight, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    if weight.is_full() {
        0xFFFF_FFFFu32.hash(hasher);
        return;
    }
    for (range, tokens) in weight.0.range_values() {
        range.start().hash(hasher);
        range.end().hash(hasher);
        for r in tokens.ranges() {
            r.start().hash(hasher);
            r.end().hash(hasher);
        }
    }
}

/// Deduplicate exact-duplicate root states in the terminal NWA.
///
/// Root states are states `start_state+1 .. start_state+1+num_roots` that were
/// created one-per-tokenizer-state-class. Many of these end up structurally
/// identical after trie-based construction and pruning. This function:
/// 1. Groups roots by a cheap structural hash, then confirms equality with PartialEq.
/// 2. For each group of duplicates, picks one canonical representative.
/// 3. Rewrites the start state's epsilon edges to point to the representative,
///    unioning the epsilon weights of merged roots.
/// 4. Calls `prune_unreachable_states` to compact the result.
fn deduplicate_roots(
    nwa: &mut NWA,
    start_state: u32,
    num_roots: usize,
    profile_enabled: bool,
) {
    let root_start = start_state as usize + 1;
    let root_end = root_start + num_roots;
    if root_end > nwa.states.len() {
        return;
    }

    // Group roots by structural hash, then confirm equality within each bucket.
    let mut hash_buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    for root_idx in 0..num_roots {
        let state_id = (root_start + root_idx) as u32;
        let h = structural_hash_nwa_state(&nwa.states[state_id as usize]);
        hash_buckets.entry(h).or_default().push(state_id);
    }

    // For each bucket, find the canonical representative via exact equality.
    let mut remap: Vec<u32> = (0..nwa.states.len() as u32).collect();
    let mut dedup_count = 0usize;

    for (_hash, bucket) in &hash_buckets {
        if bucket.len() < 2 {
            continue;
        }
        // Within the bucket, group by exact equality.
        // canonical_reps[i] = (representative_id, already matched)
        let mut canonical_reps: Vec<u32> = Vec::new();
        for &state_id in bucket {
            let mut found = false;
            for &rep in &canonical_reps {
                if nwa.states[state_id as usize] == nwa.states[rep as usize] {
                    remap[state_id as usize] = rep;
                    dedup_count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                canonical_reps.push(state_id);
            }
        }
    }

    if dedup_count == 0 {
        if profile_enabled {
            eprintln!(
                "[glrmask/profile][terminal_dwa] deduplicate_roots dedup=0 roots={}",
                num_roots,
            );
        }
        return;
    }

    // Rewrite start state epsilon edges: merge weights for roots that map to the
    // same canonical representative.
    let start = &nwa.states[start_state as usize];
    let mut merged_epsilons: BTreeMap<u32, Weight> = BTreeMap::new();
    for (target, weight) in &start.epsilons {
        let canonical = remap[*target as usize];
        merged_epsilons
            .entry(canonical)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }
    nwa.states[start_state as usize].epsilons = merged_epsilons
        .into_iter()
        .map(|(target, weight)| (target, weight))
        .collect();

    // Prune now-unreachable duplicate root states.
    prune_unreachable_states(nwa);

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] deduplicate_roots dedup={} roots={} remaining_states={}",
            dedup_count,
            num_roots,
            nwa.states.len(),
        );
    }
}

fn prune_unreachable_states(nwa: &mut NWA) -> bool {
    if nwa.states.is_empty() {
        return false;
    }

    let mut reachable = vec![false; nwa.states.len()];
    let mut queue = VecDeque::new();

    for &start in &nwa.start_states {
        if let Some(flag) = reachable.get_mut(start as usize) {
            if !*flag {
                *flag = true;
                queue.push_back(start);
            }
        }
    }

    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states[state_id as usize];
        for (target, _) in &state.epsilons {
            if let Some(flag) = reachable.get_mut(*target as usize) {
                if !*flag {
                    *flag = true;
                    queue.push_back(*target);
                }
            }
        }
        for (target, _) in state.transitions.values().flatten() {
            if let Some(flag) = reachable.get_mut(*target as usize) {
                if !*flag {
                    *flag = true;
                    queue.push_back(*target);
                }
            }
        }
    }

    if reachable.iter().all(|flag| *flag) {
        return false;
    }

    let mut remap = vec![u32::MAX; nwa.states.len()];
    let mut new_states = Vec::with_capacity(reachable.iter().filter(|flag| **flag).count());

    for (old_id, state) in nwa.states.iter().enumerate() {
        if reachable[old_id] {
            remap[old_id] = new_states.len() as u32;
            new_states.push(state.clone());
        }
    }

    for state in &mut new_states {
        state.epsilons.retain(|(target, _)| reachable[*target as usize]);
        for (target, _) in &mut state.epsilons {
            *target = remap[*target as usize];
        }

        for targets in state.transitions.values_mut() {
            targets.retain(|(target, _)| reachable[*target as usize]);
            for (target, _) in targets.iter_mut() {
                *target = remap[*target as usize];
            }
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }

    nwa.start_states = nwa
        .start_states
        .iter()
        .copied()
        .filter(|state_id| reachable[*state_id as usize])
        .map(|state_id| remap[state_id as usize])
        .collect();
    nwa.states = new_states;
    true
}

fn topological_order(nwa: &NWA) -> Vec<usize> {
    let mut in_degree = vec![0u32; nwa.states.len()];
    for state in &nwa.states {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }

    let mut queue = VecDeque::new();
    for (state_id, degree) in in_degree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }

    let mut order = Vec::with_capacity(nwa.states.len());
    while let Some(state_id) = queue.pop_front() {
        order.push(state_id);
        let state = &nwa.states[state_id];
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] -= 1;
            if in_degree[*dst as usize] == 0 {
                queue.push_back(*dst as usize);
            }
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] -= 1;
                if in_degree[*dst as usize] == 0 {
                    queue.push_back(*dst as usize);
                }
            }
        }
    }

    order
}

fn collapse_always_allowed(
    nwa: &mut NWA,
    always_allowed_by_label: &[Vec<TerminalID>],
    terminals_count: usize,
) -> bool {
    if always_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let topo_order = topological_order(nwa);
    if topo_order.is_empty() {
        return false;
    }

    let mut incoming: Vec<HashSet<TerminalID>> = vec![HashSet::new(); nwa.states.len()];
    let mut domain: Vec<Weight> = vec![Weight::empty(); nwa.states.len()];
    let mut queue = VecDeque::new();
    let mut in_queue = vec![false; nwa.states.len()];

    for &start in &nwa.start_states {
        domain[start as usize] = Weight::all();
        queue.push_back(start);
        in_queue[start as usize] = true;
    }

    while let Some(state_id) = queue.pop_front() {
        in_queue[state_id as usize] = false;
        let state_domain = domain[state_id as usize].clone();
        if state_domain.is_empty() {
            continue;
        }

        let state = &nwa.states[state_id as usize];
        let incoming_labels = incoming[state_id as usize].clone();

        for (dst, _) in &state.epsilons {
            let next_domain = domain[*dst as usize].union(&state_domain);
            let domain_changed = !next_domain.is_subset(&domain[*dst as usize]);
            if domain_changed {
                domain[*dst as usize] = next_domain;
            }

            let labels_before = incoming[*dst as usize].len();
            incoming[*dst as usize].extend(incoming_labels.iter().copied());
            let labels_changed = incoming[*dst as usize].len() != labels_before;

            if (domain_changed || labels_changed) && !in_queue[*dst as usize] {
                in_queue[*dst as usize] = true;
                queue.push_back(*dst);
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }

            for (dst, weight) in targets {
                let contrib = state_domain.intersection(weight);
                let next_domain = domain[*dst as usize].union(&contrib);
                let domain_changed = !next_domain.is_subset(&domain[*dst as usize]);
                if domain_changed {
                    domain[*dst as usize] = next_domain;
                }

                let labels_changed = incoming[*dst as usize].insert(label as TerminalID);
                if (domain_changed || labels_changed) && !in_queue[*dst as usize] {
                    in_queue[*dst as usize] = true;
                    queue.push_back(*dst);
                }
            }
        }
    }

    let mut allowed_by_state: Vec<HashSet<TerminalID>> = vec![HashSet::new(); nwa.states.len()];
    for state_id in 0..nwa.states.len() {
        let Some(&first_label) = incoming[state_id].iter().next() else {
            continue;
        };
        let Some(first_follows) = always_allowed_by_label.get(first_label as usize) else {
            continue;
        };

        let mut allowed: HashSet<TerminalID> = first_follows.iter().copied().collect();
        for &label in incoming[state_id].iter().skip(1) {
            let Some(follows) = always_allowed_by_label.get(label as usize) else {
                continue;
            };
            allowed.retain(|terminal| follows.contains(terminal));
            if allowed.is_empty() {
                break;
            }
        }
        allowed_by_state[state_id] = allowed;
    }

    let mut final_weights: Vec<Option<Weight>> =
        nwa.states.iter().map(|state| state.final_weight.clone()).collect();
    let mut changed = false;

    for &state_id in topo_order.iter().rev() {
        let allowed = &allowed_by_state[state_id];
        if allowed.len() != 1 {
            continue;
        }
        let only_allowed = *allowed.iter().next().expect("singleton set checked above");

        let domain_state = &domain[state_id];
        if domain_state.is_empty() {
            continue;
        }

        let state = &mut nwa.states[state_id];
        let mut state_final_weight = final_weights[state_id].clone();
        let mut labels_to_remove = Vec::new();

        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }
            if label as TerminalID != only_allowed {
                continue;
            }

            let mut new_targets = Vec::new();
            for (dst, weight) in targets.iter() {
                let Some(dst_final_weight) = final_weights[*dst as usize].as_ref() else {
                    new_targets.push((*dst, weight.clone()));
                    continue;
                };

                let reach = domain_state.intersection(weight);
                if !reach.is_empty() && reach.is_subset(dst_final_weight) {
                    let contrib = dst_final_weight.intersection(weight);
                    if !contrib.is_empty() {
                        state_final_weight = Some(match state_final_weight.take() {
                            Some(existing) => existing.union(&contrib),
                            None => contrib,
                        });
                    }
                    changed = true;
                    continue;
                }

                new_targets.push((*dst, weight.clone()));
            }

            if new_targets.is_empty() {
                labels_to_remove.push(label);
            } else {
                *targets = new_targets;
            }
        }

        for label in labels_to_remove {
            state.transitions.remove(&label);
        }

        state.final_weight = state_final_weight.clone();
        final_weights[state_id] = state_final_weight;
    }

    if prune_unreachable_states(nwa) {
        changed = true;
    }

    changed
}

fn should_collapse_always_allowed(vocab: &Vocab) -> bool {
    // This collapse is a compile-time optimization only. On very large LLM
    // vocabularies it can become pathologically expensive because it performs
    // repeated `Weight` set algebra over large token-ID domains. Keep it for
    // small/medium vocabs where it helps, and skip it for huge vocabularies.
    vocab.entries.len() <= 8_192
}

/// Subtract a disallowed-follow DFA from a weighted NWA.
///
/// Computes the product NWA × DFA where the result keeps only paths that the
/// NWA accepts but the DFA does NOT accept. The DFA detects sequences
/// containing a disallowed terminal follow pair; subtracting it removes
/// exactly those paths from the NWA.
///
/// Terminal labels (≥ 0) advance the DFA. Other labels and epsilons leave the
/// DFA state unchanged.
fn subtract_disallowed_dfa(nwa: &NWA, right: &crate::automata::unweighted::dfa::DFA) -> NWA {
    // Product state: (nwa_state, Option<dfa_state>).
    // None means the DFA fell to the implicit non-accepting sink.
    type ProdState = (u32, Option<u32>);

    let right_start = (!right.states.is_empty()).then_some(right.start_state);

    let mut result = NWA {
        states: Vec::new(),
        start_states: Vec::new(),
    };
    let mut state_ids: HashMap<ProdState, u32> = HashMap::new();
    let mut worklist: VecDeque<ProdState> = VecDeque::new();

    let mut get_or_create = |result: &mut NWA,
                              state_ids: &mut HashMap<ProdState, u32>,
                              worklist: &mut VecDeque<ProdState>,
                              ps: ProdState|
     -> u32 {
        if let Some(&id) = state_ids.get(&ps) {
            id
        } else {
            let id = result.add_state();
            state_ids.insert(ps, id);
            worklist.push_back(ps);
            id
        }
    };

    // Seed start states
    for &nwa_start in &nwa.start_states {
        let ps = (nwa_start, right_start);
        let id = get_or_create(&mut result, &mut state_ids, &mut worklist, ps);
        result.start_states.push(id);
    }

    while let Some((nwa_sid, dfa_sid)) = worklist.pop_front() {
        let result_sid = state_ids[&(nwa_sid, dfa_sid)];
        let nwa_state = &nwa.states[nwa_sid as usize];
        let dfa_accepting = dfa_sid
            .map(|s| right.states[s as usize].is_accepting)
            .unwrap_or(false);

        // Final weight: keep iff DFA is not accepting (no disallowed pair detected)
        if !dfa_accepting {
            if let Some(fw) = &nwa_state.final_weight {
                result.set_final_weight(result_sid, fw.clone());
            }
        }

        // Epsilon transitions: DFA state unchanged
        for (nwa_dst, weight) in &nwa_state.epsilons {
            let ps = (*nwa_dst, dfa_sid);
            let dst_id = get_or_create(&mut result, &mut state_ids, &mut worklist, ps);
            result.add_epsilon(result_sid, dst_id, weight.clone());
        }

        // Labeled transitions
        for (&label, targets) in &nwa_state.transitions {
            // Advance DFA only on terminal labels (≥ 0)
            let next_dfa = if label >= 0 {
                // Some(s) → look up transition; None stays None (sink)
                dfa_sid.and_then(|s| {
                    right.states[s as usize].transitions.get(&label).copied()
                })
            } else {
                dfa_sid // non-terminal labels don't advance DFA
            };

            for (nwa_dst, weight) in targets {
                let ps = (*nwa_dst, next_dfa);
                let dst_id = get_or_create(&mut result, &mut state_ids, &mut worklist, ps);
                result.add_transition(result_sid, label, dst_id, weight.clone());
            }
        }
    }

    result
}

fn token_weight_all_tsids(num_tsids: u32, internal_token_id: u32) -> Weight {
    if num_tsids == 0 {
        return Weight::empty();
    }
    Weight::from_uniform(
        0..=num_tsids - 1,
        RangeSetBlaze::from_iter([internal_token_id..=internal_token_id]),
    )
}

fn token_set_weight_all_tsids(num_tsids: u32, token_ids: &RangeSetBlaze<usize>) -> Weight {
    if num_tsids == 0 || token_ids.is_empty() {
        return Weight::empty();
    }
    Weight::from_uniform(0..=num_tsids - 1, RangeSetBlaze::from_iter(
        token_ids
            .ranges()
            .map(|r| (*r.start() as u32)..=(*r.end() as u32)),
    ))
}

fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([0..=max_token_id]),
    )
}

#[inline]
fn u8set_is_subset(a: &[u64; 4], b: &[u64; 4]) -> bool {
    (a[0] & !b[0]) == 0 && (a[1] & !b[1]) == 0 && (a[2] & !b[2]) == 0 && (a[3] & !b[3]) == 0
}

fn build_self_loop_bytes(tokenizer: &Tokenizer) -> Vec<[u64; 4]> {
    tokenizer
        .dfa
        .states()
        .iter()
        .enumerate()
        .map(|(state_id, state)| {
            let mut bytes = [0u64; 4];
            for (byte, &target) in state.transitions.iter() {
                if target == state_id as u32 {
                    bytes[byte as usize >> 6] |= 1u64 << (byte & 63);
                }
            }
            bytes
        })
        .collect()
}

struct TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    tokenizer: &'tok Tokenizer,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    self_loop_bytes: Vec<[u64; 4]>,
    leaf_token_ids_buffer: Vec<Vec<Vec<u32>>>,
    reachable_weight_cache: HashMap<usize, Weight>,
    pruned_weight_cache: HashMap<(usize, u32, TerminalID), Weight>,
    leaf_weight_cache: HashMap<Vec<u32>, Weight>,
    transition_buffer: BTreeMap<(u32, i32, u32), Weight>,
    epsilon_buffer: BTreeMap<(u32, u32), Weight>,
    profile_enabled: bool,
    profile_trie_calls: usize,
    profile_assoc_clones: usize,
    profile_tokenizer_execs: usize,
    profile_exec_ms: std::time::Duration,
    profile_weight_ms: std::time::Duration,
    profile_weight_compute_ms: std::time::Duration,
    profile_weight_compute_calls: usize,
    profile_match_ms: std::time::Duration,
    profile_assoc_clone_ms: std::time::Duration,
    profile_leaf_ms: std::time::Duration,
    profile_merge_ms: std::time::Duration,
    profile_pending_ms: std::time::Duration,
    profile_flush_ms: std::time::Duration,
}

impl<'tok, 'pm, 'nwa> TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {
        let source_idx = source as usize;
        if source_idx >= self.leaf_token_ids_buffer.len() {
            self.leaf_token_ids_buffer.resize_with(source_idx + 1, Vec::new);
        }
        let labels = &mut self.leaf_token_ids_buffer[source_idx];
        let label_idx = label as usize;
        if label_idx >= labels.len() {
            labels.resize_with(label_idx + 1, Vec::new);
        }
        labels[label_idx].push(internal_token_id);
    }

    fn buffer_leaf_token_id_set(&mut self, source: u32, label: TerminalID, token_ids: &RangeSetBlaze<usize>) {
        let source_idx = source as usize;
        if source_idx >= self.leaf_token_ids_buffer.len() {
            self.leaf_token_ids_buffer.resize_with(source_idx + 1, Vec::new);
        }
        let labels = &mut self.leaf_token_ids_buffer[source_idx];
        let label_idx = label as usize;
        if label_idx >= labels.len() {
            labels.resize_with(label_idx + 1, Vec::new);
        }
        for token_id in token_ids.iter() {
            let internal_token_id = token_id as u32;
            labels[label_idx].push(internal_token_id);
        }
    }

    fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {
        let cache_key = token_ids as *const RangeSetBlaze<usize> as usize;
        if let Some(weight) = self.reachable_weight_cache.get(&cache_key) {
            return weight.clone();
        }

        let t = std::time::Instant::now();
        let weight = self.token_set_weight_fast(token_ids);
        self.profile_weight_compute_ms += t.elapsed();
        self.profile_weight_compute_calls += 1;
        self.reachable_weight_cache.insert(cache_key, weight.clone());
        weight
    }

    /// Build a weight covering all tsids for the given set of internal token IDs.
    fn token_set_weight_fast(&self, internal_token_ids: &RangeSetBlaze<usize>) -> Weight {
        if self.num_tsids == 0 || internal_token_ids.is_empty() {
            return Weight::empty();
        }
        let tokens: RangeSetBlaze<u32> = internal_token_ids
            .ranges()
            .map(|r| (*r.start() as u32)..=(*r.end() as u32))
            .collect();
        Weight::from_uniform(0..=self.num_tsids - 1, tokens)
    }

    fn cached_leaf_weight(&mut self, token_ids: Vec<u32>) -> Weight {
        if let Some(weight) = self.leaf_weight_cache.get(&token_ids) {
            return weight.clone();
        }

        let tokens = RangeSetBlaze::from_iter(token_ids.iter().copied().map(|id| id..=id));
        let weight = Weight::from_uniform(0..=self.num_tsids - 1, tokens);
        self.leaf_weight_cache.insert(token_ids, weight.clone());
        weight
    }

    fn add_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        internal_token_id: u32,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = token_weight_all_tsids(self.num_tsids, internal_token_id);
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id(source, label, internal_token_id);
        }
    }

    fn add_leaf_token_set_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        token_ids: &RangeSetBlaze<usize>,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = token_set_weight_all_tsids(self.num_tsids, token_ids);
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id_set(source, label, token_ids);
        }
    }

    fn can_skip_self_loop_subtree(
        &self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: TokenizerState,
    ) -> bool {
        u8set_is_subset(node.subtree_bytes(), &self.self_loop_bytes[tokenizer_state as usize])
    }

    fn emit_self_loop_leaf_only_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &BTreeMap<TokenizerState, Vec<NwaState>>,
    ) {
        let mut accessible = node.reachable_token_ids().clone();
        if node.has_token() {
            accessible.remove(node.token_id() as usize);
        }
        for (&tokenizer_state, source_nodes) in assoc_by_state {
            let accessible_terminals = self.tokenizer.tokens_accessible_from_state(tokenizer_state);
            for terminal_id in accessible_terminals {
                self.add_leaf_token_set_from_sources(
                    source_nodes,
                    terminal_id,
                    &accessible,
                );
            }
        }
    }

    fn add_match_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        target: u32,
        weight: &Weight,
    ) {
        for &source in sources {
            if self.ignore_terminal == Some(label) {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            } else {
                self.transition_buffer
                    .entry((source, label as i32, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
    }

    fn flush_transition_buffer(&mut self) {
        let t0 = std::time::Instant::now();
        let mut leaf_entries = 0usize;
        let mut leaf_cache_misses = 0usize;
        for (from, labels_vec) in std::mem::take(&mut self.leaf_token_ids_buffer)
            .into_iter()
            .enumerate()
        {
            for (label_idx, mut token_ids) in labels_vec.into_iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                leaf_entries += 1;
                token_ids.sort_unstable();
                token_ids.dedup();
                if !self.leaf_weight_cache.contains_key(&token_ids) {
                    leaf_cache_misses += 1;
                }
                let weight = self.cached_leaf_weight(token_ids);
                self.transition_buffer
                    .entry((from as u32, label_idx as i32, self.leaf_state))
                    .and_modify(|existing| *existing = existing.union(&weight))
                    .or_insert(weight);
            }
        }
        let leaf_ms = t0.elapsed();

        let t1 = std::time::Instant::now();
        for ((from, target), weight) in std::mem::take(&mut self.epsilon_buffer) {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered epsilon source state must exist");
            state.epsilons.push((target, weight));
        }
        let eps_ms = t1.elapsed();
        let t2 = std::time::Instant::now();
        for ((from, label, target), weight) in std::mem::take(&mut self.transition_buffer) {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered transition source state must exist");
            state.transitions.entry(label).or_default().push((target, weight));
        }
        let trans_ms = t2.elapsed();
        if self.profile_enabled {
            eprintln!(
                "[glrmask/profile][terminal_dwa] flush leaf_ms={:.3} eps_ms={:.3} trans_ms={:.3} leaf_entries={} leaf_cache_misses={} transition_buffer_size={}",
                leaf_ms.as_secs_f64() * 1000.0,
                eps_ms.as_secs_f64() * 1000.0,
                trans_ms.as_secs_f64() * 1000.0,
                leaf_entries,
                leaf_cache_misses,
                0, // already drained
            );
        }
    }

    fn build_from_trie(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &BTreeMap<TokenizerState, Vec<NwaState>>,
    ) {
        self.profile_trie_calls += 1;
        let mut recursive_assoc = BTreeMap::<TokenizerState, Vec<NwaState>>::new();
        let mut self_loop_leaf_only_assoc = BTreeMap::<TokenizerState, Vec<NwaState>>::new();
        for (&tokenizer_state, source_nodes) in assoc_by_state {
            if self.can_skip_self_loop_subtree(node, tokenizer_state) {
                merge_assoc(&mut self_loop_leaf_only_assoc, tokenizer_state, source_nodes);
            } else {
                merge_assoc(&mut recursive_assoc, tokenizer_state, source_nodes);
            }
        }

        if !self_loop_leaf_only_assoc.is_empty() {
            self.emit_self_loop_leaf_only_subtree(node, &self_loop_leaf_only_assoc);
        }

        if recursive_assoc.is_empty() {
            return;
        }

        for (segment_bytes, child_node) in node.iter_children() {
            // Token IDs in the trie are already internal (equivalence class) IDs.
            let internal_child_token_id = child_node.token_id() as u32;

            let mut next_level_assoc = BTreeMap::<TokenizerState, Vec<NwaState>>::new();
            let mut pending = BTreeMap::<usize, BTreeMap<TokenizerState, Vec<NwaState>>>::new();
            let clone_started = std::time::Instant::now();
            pending.insert(0, recursive_assoc.clone());
            self.profile_assoc_clone_ms += clone_started.elapsed();
            self.profile_assoc_clones += 1;

            while let Some((pos, states_at_pos)) = pending.pop_first() {
                if pos == segment_bytes.len() {
                    let t = std::time::Instant::now();
                    for (tokenizer_state, nwa_states) in states_at_pos {
                        merge_assoc(&mut next_level_assoc, tokenizer_state, &nwa_states);
                    }
                    self.profile_merge_ms += t.elapsed();
                    continue;
                }

                for (tokenizer_state, source_nodes) in states_at_pos {
                    let exec_started = std::time::Instant::now();
                    let exec = self
                        .tokenizer
                        .execute_from_state(&segment_bytes[pos..], tokenizer_state);
                    self.profile_exec_ms += exec_started.elapsed();
                    self.profile_tokenizer_execs += 1;
                    let exec_end_state = exec.end_state;
                    let mut possible_matches_at_end = None;

                    if let Some(end_state) = exec_end_state {
                        let t = std::time::Instant::now();
                        if child_node.has_token() {
                            for terminal_id in self.tokenizer.tokens_accessible_from_state(end_state) {
                                self.add_leaf_token_from_sources(
                                    &source_nodes,
                                    terminal_id,
                                    internal_child_token_id,
                                );
                            }
                        }
                        self.profile_leaf_ms += t.elapsed();

                        let t = std::time::Instant::now();
                        merge_assoc(&mut next_level_assoc, end_state, &source_nodes);
                        self.profile_merge_ms += t.elapsed();
                    }

                    for matched in exec.matches {
                        let next_pos = pos + matched.width;

                        if next_pos == segment_bytes.len() && child_node.has_token() {
                            self.add_leaf_token_from_sources(
                                &source_nodes,
                                matched.id,
                                internal_child_token_id,
                            );
                        }

                        let weight_started = std::time::Instant::now();
                        let continuation_weight = if next_pos == segment_bytes.len()
                            && child_node.has_token()
                        {
                            let cache_key = (
                                child_node as *const VocabPrefixTreeNode as usize,
                                exec_end_state.unwrap_or(u32::MAX),
                                matched.id,
                            );
                            if let Some(weight) = self.pruned_weight_cache.get(&cache_key) {
                                weight.clone()
                            } else {
                                let mut remaining = child_node.reachable_token_ids().clone();
                                remaining.remove(internal_child_token_id as usize);
                                if let Some(end_state) = exec_end_state {
                                    let matches_at_end = possible_matches_at_end.get_or_insert_with(|| {
                                        self.possible_matches
                                            .possible_matches_for_node(child_node, end_state)
                                    });
                                    if let Some(pm) = matches_at_end.get(&matched.id) {
                                        subtract_possible_matches(&mut remaining, pm);
                                    }
                                }
                                if remaining.is_empty() {
                                    self.profile_weight_ms += weight_started.elapsed();
                                    continue;
                                }
                                let t = std::time::Instant::now();
                                let weight = self.token_set_weight_fast(&remaining);
                                self.profile_weight_compute_ms += t.elapsed();
                                self.profile_weight_compute_calls += 1;
                                self.pruned_weight_cache.insert(cache_key, weight.clone());
                                weight
                            }
                        } else {
                            self.cached_reachable_weight(child_node.reachable_token_ids())
                        };
                        self.profile_weight_ms += weight_started.elapsed();
                        if continuation_weight.is_empty() {
                            continue;
                        }

                        let t = std::time::Instant::now();
                        let continuation_assoc = pending.entry(next_pos).or_default();
                        let destination = continuation_state(
                            continuation_assoc,
                            self.tokenizer.initial_state_id(),
                            self.nwa,
                        );
                        self.profile_pending_ms += t.elapsed();

                        let match_started = std::time::Instant::now();
                        self.add_match_from_sources(
                            &source_nodes,
                            matched.id,
                            destination,
                            &continuation_weight,
                        );
                        self.profile_match_ms += match_started.elapsed();
                    }
                }
            }

            if !next_level_assoc.is_empty() {
                self.build_from_trie(child_node, &next_level_assoc);
            }
        }
    }
}

fn subtract_possible_matches(
    continuation_tokens: &mut RangeSetBlaze<usize>,
    possible_matches: &RangeSetBlaze<u32>,
) {
    for token_id in possible_matches.iter() {
        continuation_tokens.remove(token_id as usize);
    }
}

fn merge_assoc(into: &mut BTreeMap<TokenizerState, Vec<NwaState>>, state: TokenizerState, nodes: &[NwaState]) {
    match into.entry(state) {
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry.get_mut().extend(nodes.iter().copied());
        }
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(nodes.to_vec());
        }
    }
}

fn continuation_state(
    pending: &mut BTreeMap<TokenizerState, Vec<NwaState>>,
    tokenizer_state: TokenizerState,
    nwa: &mut NWA,
) -> NwaState {
    if let Some(existing) = pending
        .get(&tokenizer_state)
        .and_then(|nwa_states| nwa_states.first())
        .copied()
    {
        return existing;
    }

    let state = nwa.add_state();
    match pending.entry(tokenizer_state) {
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry.get_mut().push(state);
        }
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(vec![state]);
        }
    }
    state
}

fn terminal_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

fn log_terminal_profile(enabled: bool, phase: &str, started_at: std::time::Instant) {
    if enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] {phase}_ms={:.3}",
            started_at.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn representative_original_ids(map: &crate::compiler::stages::equivalence_analysis::ManyToOneIdMap) -> Vec<u32> {
    map.original_to_internal
        .iter()
        .enumerate()
        .map(|(original_id, _)| {
            map.representative_original_id_for_original(original_id as u32)
                .unwrap_or(original_id as u32)
        })
        .collect()
}

fn internal_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {
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

pub(crate) fn build_terminal_dwa(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    build_terminal_dwa_with_report(grammar, tokenizer, vocab, id_map, ignore_terminal).0
}

pub(crate) fn build_terminal_dwa_with_report(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> (DWA, TerminalDwaBuildReport) {
    let profile_enabled = terminal_profile_enabled();
    let total_started_at = std::time::Instant::now();
    let mut report = TerminalDwaBuildReport {
        vocab_entries: vocab.entries.len(),
        internal_tsids: id_map.num_tsids() as usize,
        ..TerminalDwaBuildReport::default()
    };

    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let phase_started_at = std::time::Instant::now();
    let internal_vocab = internal_vocab_entries(vocab, id_map);
    let vocab_tree = VocabPrefixTree::build(
        &internal_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect::<Vec<_>>(),
    );
    let self_loop_bytes = build_self_loop_bytes(tokenizer);
    let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
    report.build_vocab_trie_time = phase_started_at.elapsed();
    log_terminal_profile(profile_enabled, "build_vocab_trie", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    let mut assoc_by_state = BTreeMap::<TokenizerState, Vec<NwaState>>::new();
    for internal_tsid in 0..id_map.num_tsids() {
        let root = nwa.add_state();
        nwa.add_epsilon(
            start_state,
            root,
            all_token_weight(internal_tsid, id_map.max_internal_token_id()),
        );

        let representative_state = id_map
            .tokenizer_states
            .representative_original_id_for_internal(internal_tsid)
            .expect("internal tokenizer state class must have a representative original state");
        merge_assoc(&mut assoc_by_state, representative_state, &[root]);
    }

    let mut builder = TerminalNwaBuilder {
        tokenizer,
        possible_matches: &mut possible_matches,
        nwa: &mut nwa,
        num_tsids: id_map.num_tsids(),
        leaf_state,
        ignore_terminal,
        self_loop_bytes,
        leaf_token_ids_buffer: Vec::new(),
        reachable_weight_cache: HashMap::new(),
        pruned_weight_cache: HashMap::new(),
        leaf_weight_cache: HashMap::new(),
        transition_buffer: BTreeMap::new(),
        epsilon_buffer: BTreeMap::new(),
        profile_enabled,
        profile_trie_calls: 0,
        profile_assoc_clones: 0,
        profile_tokenizer_execs: 0,
        profile_exec_ms: std::time::Duration::ZERO,
        profile_weight_ms: std::time::Duration::ZERO,
        profile_weight_compute_ms: std::time::Duration::ZERO,
        profile_weight_compute_calls: 0,
        profile_match_ms: std::time::Duration::ZERO,
        profile_assoc_clone_ms: std::time::Duration::ZERO,
        profile_leaf_ms: std::time::Duration::ZERO,
        profile_merge_ms: std::time::Duration::ZERO,
        profile_pending_ms: std::time::Duration::ZERO,
        profile_flush_ms: std::time::Duration::ZERO,
    };
    builder.build_from_trie(&vocab_tree.root, &assoc_by_state);
    let flush_t = std::time::Instant::now();
    builder.flush_transition_buffer();
    builder.profile_flush_ms = flush_t.elapsed();
    report.build_nwa_from_trie_time = phase_started_at.elapsed();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] build_nwa_from_trie_ms={:.3} trie_calls={} assoc_clones={} tokenizer_execs={} exec_ms={:.3} weight_ms={:.3} weight_compute_ms={:.3} weight_compute_calls={} match_ms={:.3} assoc_clone_ms={:.3} leaf_ms={:.3} merge_ms={:.3} pending_ms={:.3} flush_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
            builder.profile_trie_calls,
            builder.profile_assoc_clones,
            builder.profile_tokenizer_execs,
            builder.profile_exec_ms.as_secs_f64() * 1000.0,
            builder.profile_weight_ms.as_secs_f64() * 1000.0,
            builder.profile_weight_compute_ms.as_secs_f64() * 1000.0,
            builder.profile_weight_compute_calls,
            builder.profile_match_ms.as_secs_f64() * 1000.0,
            builder.profile_assoc_clone_ms.as_secs_f64() * 1000.0,
            builder.profile_leaf_ms.as_secs_f64() * 1000.0,
            builder.profile_merge_ms.as_secs_f64() * 1000.0,
            builder.profile_pending_ms.as_secs_f64() * 1000.0,
            builder.profile_flush_ms.as_secs_f64() * 1000.0,
        );
    }

    if should_collapse_always_allowed(vocab) {
        let phase_started_at = std::time::Instant::now();
        let always_allowed_by_label = compute_always_allowed_follows(grammar);
        let _ = collapse_always_allowed(&mut nwa, &always_allowed_by_label, grammar.num_terminals as usize);
        report.collapse_always_allowed_applied = true;
        report.collapse_always_allowed_time = phase_started_at.elapsed();
        log_terminal_profile(profile_enabled, "collapse_always_allowed", phase_started_at);
    } else if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] collapse_always_allowed_skipped=1 vocab_entries={} threshold=8192",
            vocab.entries.len(),
        );
    }

    let phase_started_at = std::time::Instant::now();
    {
        let disallowed_follows = compute_disallowed_follows(grammar);
        let normalized = normalize_disallowed_follows(
            grammar.num_terminals as usize,
            &disallowed_follows,
        );
        if normalized.iter().any(|bits| !bits.is_zero()) {
            let disallowed_dfa = build_disallowed_follow_dfa(&normalized);
            nwa = subtract_disallowed_dfa(&nwa, &disallowed_dfa);
        }
    }
    report.subtract_disallowed_time = phase_started_at.elapsed();
    log_terminal_profile(profile_enabled, "subtract_disallowed", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    deduplicate_roots(&mut nwa, start_state, id_map.num_tsids() as usize, profile_enabled);
    log_terminal_profile(profile_enabled, "deduplicate_roots", phase_started_at);

    report.terminal_nwa = collect_weighted_nwa_stats(&nwa);

    let phase_started_at = std::time::Instant::now();
    let determinized = determinize(&nwa)
        .expect("terminal NWA determinization failed despite acyclic token trie construction");
    report.determinize_time = phase_started_at.elapsed();
    report.terminal_dwa = collect_weighted_dwa_stats(&determinized);
    log_terminal_profile(profile_enabled, "determinize", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    let dwa = minimize(&determinized);
    report.minimize_time = phase_started_at.elapsed();
    report.terminal_minimized_dwa = collect_weighted_dwa_stats(&dwa);
    log_terminal_profile(profile_enabled, "minimize", phase_started_at);
    report.total_time = total_started_at.elapsed();

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] total_ms={:.3} vocab_entries={} internal_tsids={} {} {}",
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            vocab.entries.len(),
            id_map.num_tsids(),
            report.terminal_nwa,
            report.terminal_minimized_dwa,
        );
    }

    (dwa, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};
    use crate::compiler::grammar::model::tests::simple_ab_grammar;
    use crate::compiler::stages::equivalence_analysis::ManyToOneIdMap;
    use std::collections::BTreeSet;

    fn expand_original_tokens(weight: &Weight, id_map: &InternalIdMap) -> BTreeSet<u32> {
        let mut original_tokens = BTreeSet::new();
        for internal_token_id in weight.token_union().iter() {
            if let Some(original_ids) = id_map.vocab_tokens.internal_to_originals.get(internal_token_id as usize) {
                original_tokens.extend(original_ids.iter().copied());
            } else {
                original_tokens.insert(internal_token_id);
            }
        }
        original_tokens
    }

    fn build_literal_terminal_dwa(
        rules: Vec<Rule>,
        literals: &[&[u8]],
        vocab_entries: Vec<(u32, &[u8])>,
    ) -> (DWA, InternalIdMap) {
        let grammar = GrammarDef {
            rules,
            start: 0,
            terminals: literals
                .iter()
                .enumerate()
                .map(|(id, bytes)| Terminal::Literal {
                    id: id as u32,
                    bytes: bytes.to_vec(),
                })
                .collect(),
            ..Default::default()
        };
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vocab_entries
                .into_iter()
                .map(|(id, bytes)| (id, bytes.to_vec()))
                .collect(),
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab, &std::collections::BTreeMap::new(), None);
        (build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map, None), id_map)
    }

    #[test]
    fn test_terminal_dwa_accepts_single_and_multi_terminal_paths() {
        let grammar = simple_ab_grammar();
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"ab".to_vec()), (2, b"b".to_vec())],
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab, &std::collections::BTreeMap::new(), None);

        let terminal_dwa = build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map, None);

        let a_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&a_weight, &id_map);
        assert!(
            original_tokens.contains(&0),
            "terminal DWA should still accept the explicit single-terminal token 'a'"
        );
        assert!(
            original_tokens.contains(&1),
            "always-allowed collapse should make the multi-terminal token 'ab' available on the 'a' terminal word"
        );
        assert!(
            terminal_dwa.eval_word(&[0, 1]).is_empty(),
            "after collapse, the explicit multi-terminal word 'ab' should no longer be required"
        );
    }

    #[test]
    fn test_terminal_dwa_treats_ignore_terminal_as_epsilon() {
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
                Terminal::Pattern {
                    id: 1,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(1),
            ..Default::default()
        };
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vec![
                (0, b" ".to_vec()),
                (1, b"a".to_vec()),
                (2, b" a".to_vec()),
                (3, b"b".to_vec()),
            ],
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab, &std::collections::BTreeMap::new(), None);

        let terminal_dwa = build_terminal_dwa(
            &glr_grammar,
            &tokenizer,
            &vocab,
            &id_map,
            grammar.ignore_terminal,
        );

        let empty_weight = terminal_dwa.eval_word(&[]);
        let empty_original_tokens = expand_original_tokens(&empty_weight, &id_map);
        assert!(
            empty_original_tokens.contains(&0),
            "ignore-only tokens should appear in the terminal DWA start-state final weight"
        );

        let a_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&a_weight, &id_map);
        assert!(
            original_tokens.contains(&1),
            "plain non-ignore terminal tokens should still be accepted"
        );
        assert!(
            original_tokens.contains(&2),
            "tokens with ignored prefixes should also be accepted on the same terminal word"
        );
    }

    #[test]
    fn test_terminal_dwa_supports_distinct_bytes_for_same_internal_token() {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };

        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(vec![(10, b"a".to_vec()), (20, b"b".to_vec())], None);
        let mut id_map = InternalIdMap::build_identity(&tokenizer, &vocab);
        let mut original_to_internal = vec![u32::MAX; 21];
        original_to_internal[10] = 0;
        original_to_internal[20] = 0;
        id_map.vocab_tokens = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals: vec![vec![10, 20]],
        };

        let terminal_dwa = build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map, None);

        let a_weight = terminal_dwa.eval_word(&[0]);
        let a_original_tokens = expand_original_tokens(&a_weight, &id_map);
        assert!(
            a_original_tokens.contains(&10) && a_original_tokens.contains(&20),
            "merged internal token should stay reachable on terminal 'a'"
        );

        let b_weight = terminal_dwa.eval_word(&[1]);
        let b_original_tokens = expand_original_tokens(&b_weight, &id_map);
        assert!(
            b_original_tokens.contains(&10) && b_original_tokens.contains(&20),
            "merged internal token should stay reachable on terminal 'b'"
        );
    }

    #[test]
    fn test_terminal_dwa_collapses_always_allowed_chain_to_first_terminal() {
        let (terminal_dwa, id_map) = build_literal_terminal_dwa(
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2)],
            }],
            &[b"a", b"b", b"c"],
            vec![(0, b"a"), (1, b"ab"), (2, b"abc")],
        );

        let first_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&first_weight, &id_map);
        assert!(original_tokens.contains(&0), "single-terminal token should still be accepted");
        assert!(original_tokens.contains(&1), "always-allowed suffix 'b' should collapse into the 'a' state");
        assert!(original_tokens.contains(&2), "always-allowed chain 'b' then 'c' should collapse all the way into the 'a' state");
    }

    #[test]
    fn test_terminal_dwa_does_not_collapse_non_always_follow() {
        let (terminal_dwa, id_map) = build_literal_terminal_dwa(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
                },
            ],
            &[b"a", b"b", b"c", b"d"],
            vec![(0, b"a"), (1, b"ab"), (2, b"abc"), (3, b"ad")],
        );

        let first_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&first_weight, &id_map);
        assert!(original_tokens.contains(&0), "the explicit 'a' token should still be accepted");
        assert!(!original_tokens.contains(&1), "'b' is only ever allowed after 'a', not always allowed, so 'ab' must not collapse");
        assert!(!original_tokens.contains(&2), "the 'abc' chain must not collapse when the first follow is not always allowed");
    }

    /// Grammar: start: A B; A: "a"+; B: "b"+ with vocab = ["a"]
    /// The terminal DWA should have exactly one transition from start (on terminal A).
    #[test]
    fn test_terminal_dwa_expr_terminals_a_plus_b_plus_single_token() {
        use crate::automata::lexer::ast::Expr;

        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 1,
                        max: None,
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min: 1,
                        max: None,
                    },
                },
            ],
            ..Default::default()
        };

        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec())],
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab, &std::collections::BTreeMap::new(), None);
        let terminal_dwa = build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map, None);

        let start_state = &terminal_dwa.states[terminal_dwa.start_state as usize];
        assert_eq!(
            start_state.transitions.len(),
            1,
            "terminal DWA start state should have exactly 1 transition (for terminal A) \
             but has {} transitions",
            start_state.transitions.len(),
        );
    }

    /// Regression test: a Kleene-star terminal (min=0) at the top level used
    /// to create an NFA loop-back edge to state 0 (the global initial state),
    /// making ALL terminals appear in `possible_future_group_ids` for any DFA
    /// state reachable from that loop.  This caused unrelated literals (like
    /// `"a"`) to get DWA transitions for tokens that can't byte-match them.
    #[test]
    fn test_kleene_star_no_spurious_terminal_transition() {
        // X: "b"* is a Kleene-star terminal.  With vocab=["b"], only X
        // should get a transition from the DWA start state.  Literal "a"
        // (byte 0x61) should NOT get a transition because "b" != "a".
        let lark = r#"
X: "b"*
start: "a" X
"#;
        let grammar = crate::import::lark::parse_lark(lark).unwrap();
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(vec![(0, b"b".to_vec())], None);
        let id_map = InternalIdMap::build(
            &tokenizer, &vocab, &std::collections::BTreeMap::new(), None,
        );
        let terminal_dwa = build_terminal_dwa(
            &glr_grammar, &tokenizer, &vocab, &id_map, None,
        );
        let start = &terminal_dwa.states[terminal_dwa.start_state as usize];

        let a_id = grammar.terminals.iter().find_map(|t| match t {
            Terminal::Literal { id, bytes } if bytes == b"a" => Some(*id as i32),
            _ => None,
        }).expect("literal 'a' terminal must exist");

        assert!(
            !start.transitions.contains_key(&a_id),
            "literal 'a' should NOT have a DWA transition with vocab=[\"b\"], \
             but it does — the Kleene-star NFA loop-back is polluting \
             possible_future_group_ids",
        );
        assert_eq!(
            start.transitions.len(), 1,
            "only terminal X should get a transition from start",
        );
    }
}
