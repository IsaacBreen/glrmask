use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerMatch};
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NWAState as NWAStateType};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::Symbol;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesByState,
    PossibleMatchesComputer,
    PossibleMatchesProfile,
    collect_possible_matches_by_internal_tsid,
};
use crate::compiler::stages::equivalence_analysis::disallowed_follows::{
    build_disallowed_follow_dfa, normalize_disallowed_follows,
};

/// NWA state identifier (index into `NWA.states`).
type NwaState = u32;
/// Tokenizer state identifier.
type TokenizerState = u32;
type ColorId = u32;
type LeafTokenIds = SmallVec<[u32; 8]>;
type FutureTerminalColorGroups = SmallVec<[(ColorId, SmallVec<[TerminalID; 4]>); 8]>;
use crate::compiler::compile::compute_disallowed_follows;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;

#[derive(Debug, Clone)]
pub(crate) struct TerminalColoring {
    terminal_to_color: Vec<ColorId>,
    num_colors: usize,
}

impl TerminalColoring {
    pub(crate) fn identity(num_terminals: usize) -> Self {
        Self {
            terminal_to_color: (0..num_terminals as ColorId).collect(),
            num_colors: num_terminals,
        }
    }

    #[inline]
    fn color_for(&self, terminal_id: TerminalID) -> ColorId {
        self.terminal_to_color
            .get(terminal_id as usize)
            .copied()
            .unwrap_or(terminal_id)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TerminalDwaBuildProfile {
    future_terminal_additions: u64,
    match_transition_additions: u64,
}

fn terminal_dwa_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_TERMINAL_DWA").is_some()
}

fn debug_profile_enabled() -> bool {
    std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn root_combined_signature(
    tokenizer: &Tokenizer,
    representative_state: u32,
    internal_tsid: u32,
    terminal_coloring: &TerminalColoring,
    ignore_terminal: Option<TerminalID>,
    possible_matches_by_state: &PossibleMatchesByState,
) -> u64 {
    let mut future_groups = BTreeMap::<ColorId, SmallVec<[TerminalID; 4]>>::new();
    for terminal_id in tokenizer.possible_future_terminals_iter(representative_state) {
        if Some(terminal_id) == ignore_terminal {
            continue;
        }
        future_groups
            .entry(terminal_coloring.color_for(terminal_id))
            .or_default()
            .push(terminal_id);
    }

    let mut future_hasher = std::collections::hash_map::DefaultHasher::new();
    for (color, terminals) in future_groups {
        color.hash(&mut future_hasher);
        terminals.len().hash(&mut future_hasher);
        for terminal_id in terminals {
            terminal_id.hash(&mut future_hasher);
        }
    }
    let future_sig = future_hasher.finish();

    let mut possible_matches_hasher = std::collections::hash_map::DefaultHasher::new();
    if let Some(matches_by_terminal) = possible_matches_by_state.get(&internal_tsid) {
        for (terminal_id, token_ids) in matches_by_terminal {
            terminal_id.hash(&mut possible_matches_hasher);
            for range in token_ids.ranges() {
                range.start().hash(&mut possible_matches_hasher);
                range.end().hash(&mut possible_matches_hasher);
            }
        }
    }
    let possible_matches_sig = possible_matches_hasher.finish();

    let mut combined_hasher = std::collections::hash_map::DefaultHasher::new();
    representative_state.hash(&mut combined_hasher);
    future_sig.hash(&mut combined_hasher);
    possible_matches_sig.hash(&mut combined_hasher);
    combined_hasher.finish()
}

pub(crate) fn compute_terminal_coloring(table: &GLRTable) -> TerminalColoring {
    let num_terminals = table.num_terminals as usize;
    if num_terminals <= 1 {
        return TerminalColoring::identity(num_terminals);
    }

    let mut adjacency = vec![BitSet::new(num_terminals); num_terminals];
    for row in &table.action {
        let terminals: Vec<usize> = row
            .keys()
            .copied()
            .filter(|&terminal| (terminal as usize) < num_terminals)
            .map(|terminal| terminal as usize)
            .collect();
        for left_idx in 0..terminals.len() {
            let left = terminals[left_idx];
            for &right in &terminals[left_idx + 1..] {
                adjacency[left].set(right);
                adjacency[right].set(left);
            }
        }
    }

    let degrees: Vec<usize> = adjacency.iter().map(BitSet::count_ones).collect();
    let mut terminal_to_color = vec![ColorId::MAX; num_terminals];
    let mut neighbor_colors = vec![BitSet::new(num_terminals); num_terminals];
    let mut num_colors = 0usize;

    for _ in 0..num_terminals {
        let next_terminal = (0..num_terminals)
            .filter(|&terminal| terminal_to_color[terminal] == ColorId::MAX)
            .max_by(|&left, &right| {
                neighbor_colors[left]
                    .count_ones()
                    .cmp(&neighbor_colors[right].count_ones())
                    .then_with(|| degrees[left].cmp(&degrees[right]))
                    .then_with(|| right.cmp(&left))
            })
            .expect("there should always be an uncolored terminal");

        let mut color = 0usize;
        while neighbor_colors[next_terminal].contains(color) {
            color += 1;
        }
        terminal_to_color[next_terminal] = color as ColorId;
        num_colors = num_colors.max(color + 1);

        for neighbor in adjacency[next_terminal].iter_ones() {
            if terminal_to_color[neighbor] == ColorId::MAX {
                neighbor_colors[neighbor].set(color);
            }
        }
    }

    TerminalColoring {
        terminal_to_color,
        num_colors,
    }
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

fn canonicalize_acyclic_nwa(nwa: &mut NWA) {
    if nwa.states.len() <= 1 {
        return;
    }

    prune_unreachable_states(nwa);
    let topo_order = topological_order(nwa);
    if topo_order.len() != nwa.states.len() {
        return;
    }

    let old_states = nwa.states.len();
    let mut remap = vec![u32::MAX; old_states];
    let mut canonical_states: Vec<NWAStateType> = Vec::with_capacity(old_states);
    let mut hash_buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut merged = 0usize;

    for old_state_id in topo_order.into_iter().rev() {
        let old_state = &nwa.states[old_state_id];

        let mut epsilons: BTreeMap<u32, Weight> = BTreeMap::new();
        for (target, weight) in &old_state.epsilons {
            let canonical_target = remap[*target as usize];
            epsilons
                .entry(canonical_target)
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }

        let mut transitions = BTreeMap::new();
        for (&label, targets) in &old_state.transitions {
            let mut canonical_targets: BTreeMap<u32, Weight> = BTreeMap::new();
            for (target, weight) in targets {
                let canonical_target = remap[*target as usize];
                canonical_targets
                    .entry(canonical_target)
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
            if !canonical_targets.is_empty() {
                transitions.insert(label, canonical_targets.into_iter().collect());
            }
        }

        let canonical_state = NWAStateType {
            final_weight: old_state.final_weight.clone(),
            transitions,
            epsilons: epsilons.into_iter().collect(),
        };

        let state_hash = structural_hash_nwa_state(&canonical_state);
        let mut canonical_id = None;
        if let Some(candidates) = hash_buckets.get(&state_hash) {
            for &candidate in candidates {
                if canonical_states[candidate as usize] == canonical_state {
                    canonical_id = Some(candidate);
                    merged += 1;
                    break;
                }
            }
        }

        let canonical_id = canonical_id.unwrap_or_else(|| {
            let new_id = canonical_states.len() as u32;
            canonical_states.push(canonical_state);
            hash_buckets.entry(state_hash).or_default().push(new_id);
            new_id
        });
        remap[old_state_id] = canonical_id;
    }

    if merged == 0 {
        return;
    }

    let mut start_states = Vec::with_capacity(nwa.start_states.len());
    let mut seen_start_states = HashSet::new();
    for &start_state in &nwa.start_states {
        let canonical_start = remap[start_state as usize];
        if seen_start_states.insert(canonical_start) {
            start_states.push(canonical_start);
        }
    }

    nwa.states = canonical_states;
    nwa.start_states = start_states;
}

/// Remove NWA states where `retain[state_id]` is false, remapping all
/// remaining state references. When `drop_empty_weights` is true, also
/// drop edges whose weight is empty. Returns true if any states were removed.
fn retain_nwa_states(nwa: &mut NWA, retain: &[bool], drop_empty_weights: bool) -> bool {
    if retain.iter().all(|&f| f) {
        return false;
    }

    let mut remap = vec![u32::MAX; nwa.states.len()];
    let mut new_states = Vec::with_capacity(retain.iter().filter(|&&f| f).count());

    for (old_id, state) in nwa.states.iter().enumerate() {
        if retain[old_id] {
            remap[old_id] = new_states.len() as u32;
            new_states.push(state.clone());
        }
    }

    for state in &mut new_states {
        state.epsilons.retain(|(target, weight)| {
            retain[*target as usize] && (!drop_empty_weights || !weight.is_empty())
        });
        for (target, _) in &mut state.epsilons {
            *target = remap[*target as usize];
        }

        for targets in state.transitions.values_mut() {
            targets.retain(|(target, weight)| {
                retain[*target as usize] && (!drop_empty_weights || !weight.is_empty())
            });
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
        .filter(|state_id| retain[*state_id as usize])
        .map(|state_id| remap[state_id as usize])
        .collect();
    nwa.states = new_states;
    true
}

fn compute_forward_reachable(nwa: &NWA) -> Vec<bool> {
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

    reachable
}

fn prune_unreachable_states(nwa: &mut NWA) -> bool {
    if nwa.states.is_empty() {
        return false;
    }
    let reachable = compute_forward_reachable(nwa);
    retain_nwa_states(nwa, &reachable, false)
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

fn compute_coreachable_nwa(nwa: &NWA) -> Vec<bool> {
    if nwa.states.is_empty() {
        return Vec::new();
    }

    let mut reverse_edges: Vec<Vec<usize>> = vec![Vec::new(); nwa.states.len()];
    for (state_id, state) in nwa.states.iter().enumerate() {
        for (dst, weight) in &state.epsilons {
            if !weight.is_empty() {
                reverse_edges[*dst as usize].push(state_id);
            }
        }
        for targets in state.transitions.values() {
            for (dst, weight) in targets {
                if !weight.is_empty() {
                    reverse_edges[*dst as usize].push(state_id);
                }
            }
        }
    }

    let mut coreachable = vec![false; nwa.states.len()];
    let mut queue = VecDeque::new();
    for (state_id, state) in nwa.states.iter().enumerate() {
        if state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()) {
            coreachable[state_id] = true;
            queue.push_back(state_id);
        }
    }

    while let Some(state_id) = queue.pop_front() {
        for &pred in &reverse_edges[state_id] {
            if !coreachable[pred] {
                coreachable[pred] = true;
                queue.push_back(pred);
            }
        }
    }

    coreachable
}

fn prune_non_coreachable_states(nwa: &mut NWA) -> bool {
    if nwa.states.is_empty() {
        return false;
    }
    let coreachable = compute_coreachable_nwa(nwa);
    retain_nwa_states(nwa, &coreachable, true)
}

fn propagate_collapse_context(
    nwa: &NWA,
    terminals_count: usize,
) -> (Vec<HashSet<TerminalID>>, Vec<Weight>) {
    let mut incoming = vec![HashSet::new(); nwa.states.len()];
    let mut domain = vec![Weight::empty(); nwa.states.len()];
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

    (incoming, domain)
}

fn allowed_labels_by_state(
    incoming: &[HashSet<TerminalID>],
    always_allowed_by_label: &[Vec<TerminalID>],
) -> Vec<HashSet<TerminalID>> {
    let mut allowed_by_state = vec![HashSet::new(); incoming.len()];

    for (state_id, labels) in incoming.iter().enumerate() {
        let Some(&first_label) = labels.iter().next() else {
            continue;
        };
        let Some(first_follows) = always_allowed_by_label.get(first_label as usize) else {
            continue;
        };

        let mut allowed: HashSet<TerminalID> = first_follows.iter().copied().collect();
        for &label in labels.iter().skip(1) {
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

    allowed_by_state
}

fn collapse_single_allowed_transitions(
    nwa: &mut NWA,
    topo_order: &[usize],
    domain: &[Weight],
    allowed_by_state: &[HashSet<TerminalID>],
    terminals_count: usize,
) -> bool {
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

    changed
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

    let (incoming, domain) = propagate_collapse_context(nwa, terminals_count);
    let allowed_by_state = allowed_labels_by_state(&incoming, always_allowed_by_label);
    let mut changed =
        collapse_single_allowed_transitions(nwa, &topo_order, &domain, &allowed_by_state, terminals_count);

    if prune_unreachable_states(nwa) {
        changed = true;
    }

    changed
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

    let get_or_create = |result: &mut NWA,
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

fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([0..=max_token_id]),
    )
}

#[derive(Clone)]
struct NodesByTokenizerState {
    entries: FxHashMap<TokenizerState, Vec<NwaState>>,
}

impl NodesByTokenizerState {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn merge(&mut self, state: TokenizerState, nodes: &[NwaState]) {
        self.entries.entry(state).or_default().extend_from_slice(nodes);
    }

    fn first(&self, state: TokenizerState) -> Option<NwaState> {
        self.entries.get(&state).and_then(|nodes| nodes.first().copied())
    }

    fn push_one(&mut self, state: TokenizerState, node: NwaState) {
        self.entries.entry(state).or_default().push(node);
    }

    fn iter(&self) -> impl Iterator<Item = (TokenizerState, &[NwaState])> {
        self.entries
            .iter()
            .map(|(&state, nodes)| (state, nodes.as_slice()))
    }
}

impl IntoIterator for NodesByTokenizerState {
    type Item = (TokenizerState, Vec<NwaState>);
    type IntoIter = <FxHashMap<TokenizerState, Vec<NwaState>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

struct TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    tokenizer: &'tok Tokenizer,
    terminal_coloring: TerminalColoring,
    possible_future_terminals: FxHashMap<TokenizerState, Vec<TerminalID>>,
    future_terminal_color_groups: FxHashMap<TokenizerState, FutureTerminalColorGroups>,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    use_terminal_coloring: bool,
    terminal_path_lengths: Option<Vec<TerminalPathLength>>,
    self_loop_bytes: FxHashMap<TokenizerState, U8Set>,
    leaf_token_ids_buffer: Vec<Vec<LeafTokenIds>>,
    future_leaf_buffer: FxHashMap<(u32, TokenizerState, ColorId), BufferedLeafTransition>,
    reachable_weight_cache: HashMap<usize, Weight>,
    pruned_weight_cache: HashMap<(usize, u32, TerminalID), Weight>,
    leaf_weight_cache: HashMap<LeafTokenIds, Weight>,
    transition_buffer: FxHashMap<(u32, i32, u32), Weight>,
    epsilon_buffer: FxHashMap<(u32, u32), Weight>,
    profile: TerminalDwaBuildProfile,
    /// Lazily-built flat DFA transition table. Indexed by state ID.
    /// Each entry maps byte → next_state (u32::MAX = no transition).
    /// Replaces binary-search CharTransitions::get with O(1) array index.
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
}

#[derive(Default)]
struct BufferedLeafTransition {
    token_ids: LeafTokenIds,
    weight: Option<Weight>,
}

impl<'tok, 'pm, 'nwa> TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    /// O(1) DFA step using lazily-built flat transition table.
    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            let dfa_state = &self.tokenizer.dfa.states()[state_idx];
            let mut flat = Box::new([u32::MAX; 256]);
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            self.flat_transitions[state_idx] = Some(flat);
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn leaf_token_ids_for(&mut self, source: u32, label: TerminalID) -> &mut LeafTokenIds {
        let source_idx = source as usize;
        if source_idx >= self.leaf_token_ids_buffer.len() {
            self.leaf_token_ids_buffer.resize_with(source_idx + 1, Vec::new);
        }

        let labels = &mut self.leaf_token_ids_buffer[source_idx];
        let label_idx = label as usize;
        if label_idx >= labels.len() {
            labels.resize_with(label_idx + 1, SmallVec::new);
        }

        &mut labels[label_idx]
    }

    fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {
        self.leaf_token_ids_for(source, label).push(internal_token_id);
    }

    fn possible_future_terminals_for_state(&mut self, tokenizer_state: TokenizerState) -> Vec<TerminalID> {
        self.possible_future_terminals
            .entry(tokenizer_state)
            .or_insert_with(|| {
                self.tokenizer
                    .possible_future_terminals_iter(tokenizer_state)
                    .collect()
            })
            .clone()
    }

    fn future_terminal_color_groups_for_state(
        &mut self,
        tokenizer_state: TokenizerState,
    ) -> FutureTerminalColorGroups {
        self.future_terminal_color_groups
            .entry(tokenizer_state)
            .or_insert_with(|| {
                let mut groups = BTreeMap::<ColorId, SmallVec<[TerminalID; 4]>>::new();
                for terminal_id in self.tokenizer.possible_future_terminals_iter(tokenizer_state) {
                    if Some(terminal_id) == self.ignore_terminal {
                        continue;
                    }
                    groups
                        .entry(self.terminal_coloring.color_for(terminal_id))
                        .or_default()
                        .push(terminal_id);
                }
                groups.into_iter().collect()
            })
            .clone()
    }

    fn buffer_future_leaf_token_id(
        &mut self,
        source: u32,
        internal_tsid: TokenizerState,
        color: ColorId,
        internal_token_id: u32,
    ) {
        self.profile.future_terminal_additions += 1;
        self.future_leaf_buffer
            .entry((source, internal_tsid, color))
            .or_default()
            .token_ids
            .push(internal_token_id);
    }

    fn add_future_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        internal_token_id: u32,
    ) {
        if !self.use_terminal_coloring {
            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for terminal_id in future_terminals {
                self.add_leaf_token_from_sources(sources, terminal_id, internal_token_id);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal {
            if self
                .possible_future_terminals_for_state(tokenizer_state)
                .contains(&ignore_terminal)
            {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_leaf_token_from_sources(sources, ignore_terminal, internal_token_id);
            }
        }

        let color_groups = self.future_terminal_color_groups_for_state(tokenizer_state);
        for (color, terminals) in color_groups {
            if terminals.is_empty() {
                continue;
            }
            for &source in sources {
                self.buffer_future_leaf_token_id(source, tokenizer_state, color, internal_token_id);
            }
        }
    }

    fn add_future_weighted_match_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        weight: &Weight,
    ) {
        if !self.use_terminal_coloring {
            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for terminal_id in future_terminals {
                self.add_match_from_sources(sources, terminal_id, self.leaf_state, weight);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal {
            if self
                .possible_future_terminals_for_state(tokenizer_state)
                .contains(&ignore_terminal)
            {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_match_from_sources(sources, ignore_terminal, self.leaf_state, weight);
            }
        }

        let color_groups = self.future_terminal_color_groups_for_state(tokenizer_state);
        for (color, terminals) in color_groups {
            if terminals.is_empty() || weight.is_empty() {
                continue;
            }
            for &source in sources {
                self.profile.future_terminal_additions += 1;
                let entry = self.future_leaf_buffer
                    .entry((source, tokenizer_state, color))
                    .or_default();
                if let Some(existing) = &mut entry.weight {
                    *existing = existing.union(weight);
                } else {
                    entry.weight = Some(weight.clone());
                }
            }
        }
    }

    fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {
        let cache_key = token_ids as *const RangeSetBlaze<usize> as usize;
        if let Some(weight) = self.reachable_weight_cache.get(&cache_key) {
            return weight.clone();
        }

        let weight = self.token_set_weight_fast(token_ids);
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

    fn cached_leaf_weight(&mut self, mut token_ids: LeafTokenIds) -> Weight {
        token_ids.sort_unstable();
        token_ids.dedup();

        if let Some(weight) = self.leaf_weight_cache.get(&token_ids) {
            return weight.clone();
        }

        let tokens = RangeSetBlaze::from_iter(token_ids.iter().copied().map(|id| id..=id));
        let weight = Weight::from_uniform(0..=self.num_tsids - 1, tokens);
        self.leaf_weight_cache.insert(token_ids, weight.clone());
        weight
    }

    fn continuation_weight_for_match(
        &mut self,
        child_node: &VocabPrefixTreeNode,
        leaf_token_id: u32,
        terminal_id: TerminalID,
        end_state: Option<u32>,
        completes_segment: bool,
    ) -> Option<Weight> {
        if !(completes_segment && child_node.has_token()) {
            return Some(self.cached_reachable_weight(child_node.reachable_token_ids()));
        }

        let cache_key = (
            child_node as *const VocabPrefixTreeNode as usize,
            end_state.unwrap_or(u32::MAX),
            terminal_id,
        );
        if let Some(weight) = self.pruned_weight_cache.get(&cache_key) {
            return Some(weight.clone());
        }

        let mut remaining = child_node.reachable_token_ids().clone();
        remaining.remove(leaf_token_id as usize);

        if let Some(end_state) = end_state {
            let possible_matches = self
                .possible_matches
                .possible_matches_for_node(child_node, end_state);
            if let Some(matches_for_terminal) = possible_matches.get(&terminal_id) {
                subtract_possible_matches(&mut remaining, matches_for_terminal);
            }
        }

        if remaining.is_empty() {
            return None;
        }

        let weight = self.token_set_weight_fast(&remaining);
        self.pruned_weight_cache.insert(cache_key, weight.clone());
        Some(weight)
    }

    fn add_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        internal_token_id: u32,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = if self.num_tsids == 0 {
                Weight::empty()
            } else {
                Weight::from_uniform(
                    0..=self.num_tsids - 1,
                    RangeSetBlaze::from_iter([internal_token_id..=internal_token_id]),
                )
            };
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id(source, label, internal_token_id);
        }
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: TokenizerState,
    ) -> bool {
        let self_loop_bytes = self.self_loop_bytes.entry(tokenizer_state).or_insert_with(|| {
            let state = &self.tokenizer.dfa.states()[tokenizer_state as usize];
            let mut bytes = U8Set::empty();
            for (byte, &target) in state.transitions.iter() {
                if target == tokenizer_state {
                    bytes.insert(byte);
                }
            }
            bytes
        });
        U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
    }

    fn emit_self_loop_leaf_only_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        let mut accessible = node.reachable_token_ids().clone();
        if node.has_token() {
            accessible.remove(node.token_id() as usize);
        }
        if accessible.is_empty() {
            return;
        }
        let accessible_weight = self.token_set_weight_fast(&accessible);
        for (internal_tsid, source_nodes) in assoc_by_state.iter() {
            self.add_future_weighted_match_from_sources(
                source_nodes,
                internal_tsid,
                &accessible_weight,
            );
        }
    }

    fn add_match_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        target: u32,
        weight: &Weight,
    ) {
        if self.ignore_terminal == Some(label) {
            for &source in sources {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        } else {
            for &source in sources {
                self.transition_buffer
                    .entry((source, label as i32, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
    }

    fn flush_transition_buffer(&mut self) {
        let mut leaf_transition_buckets: Vec<FxHashMap<i32, BufferedLeafTransition>> =
            (0..self.nwa.states.len()).map(|_| FxHashMap::default()).collect();

        for (from, labels_vec) in std::mem::take(&mut self.leaf_token_ids_buffer)
            .into_iter()
            .enumerate()
        {
            for (label_idx, token_ids) in labels_vec.into_iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                leaf_transition_buckets[from]
                    .entry(label_idx as i32)
                    .or_default()
                    .token_ids
                    .extend(token_ids);
            }
        }

        for ((source, tokenizer_state, color), buffered) in
            std::mem::take(&mut self.future_leaf_buffer)
        {
            if buffered.token_ids.is_empty() && buffered.weight.as_ref().map_or(true, |w| w.is_empty()) {
                continue;
            }
            let color_groups = self.future_terminal_color_groups_for_state(tokenizer_state);
            let terminals = color_groups
                .iter()
                .find_map(|(group_color, terminals)| (*group_color == color).then_some(terminals.to_vec()))
                .unwrap_or_default();
            for terminal_id in terminals {
                let entry = leaf_transition_buckets[source as usize]
                    .entry(terminal_id as i32)
                    .or_default();
                if !buffered.token_ids.is_empty() {
                    entry.token_ids.extend_from_slice(&buffered.token_ids);
                }
                if let Some(w) = &buffered.weight {
                    if let Some(existing) = &mut entry.weight {
                        *existing = existing.union(w);
                    } else {
                        entry.weight = Some(w.clone());
                    }
                }
            }
        }

        let mut epsilon_entries: Vec<_> = std::mem::take(&mut self.epsilon_buffer).into_iter().collect();
        epsilon_entries.sort_unstable_by_key(|((from, target), _)| (*from, *target));
        for ((from, target), weight) in epsilon_entries {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered epsilon source state must exist");
            state.epsilons.push((target, weight));
        }

        let mut transition_entries: Vec<_> = std::mem::take(&mut self.transition_buffer).into_iter().collect();
        transition_entries.sort_unstable_by_key(|((from, label, target), _)| (*from, *label, *target));
        for ((from, label, target), weight) in transition_entries {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered transition source state must exist");
            state.transitions.entry(label).or_default().push((target, weight));
        }

        for (from, bucket) in leaf_transition_buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            let mut entries: Vec<(i32, BufferedLeafTransition)> = bucket.into_iter().collect();
            entries.sort_unstable_by_key(|(label, _)| *label);

            let mut finalized_entries = Vec::with_capacity(entries.len());
            for (label, mut entry) in entries {
                let mut weight = entry.weight.take().unwrap_or_else(Weight::empty);
                if !entry.token_ids.is_empty() {
                    let token_weight = self.cached_leaf_weight(entry.token_ids);
                    weight = if weight.is_empty() {
                        token_weight
                    } else {
                        weight.union(&token_weight)
                    };
                }
                if !weight.is_empty() {
                    finalized_entries.push((label, weight));
                }
            }

            let state = self
                .nwa
                .states
                .get_mut(from)
                .expect("buffered leaf transition source state must exist");
            for (label, weight) in finalized_entries {
                state.transitions.entry(label).or_default().push((self.leaf_state, weight));
            }
        }
    }

    /// Fast NWA construction for L1-only grammars (all terminals have path
    /// length ≤ 1).  Replaces the trie walk with a simple flat loop over
    /// internal vocab × state class representatives.
    fn build_l1_fast(
        &mut self,
        internal_vocab: &[(u32, Vec<u8>)],
        roots_by_tokenizer_state: &NodesByTokenizerState,
        id_map: &InternalIdMap,
    ) {
        for &(internal_token_id, ref bytes) in internal_vocab {
            for (_tsid_idx, representative_state) in
                id_map.tokenizer_states.iter_representative_ids().enumerate()
            {
                let source_nodes =
                    match roots_by_tokenizer_state.entries.get(&representative_state) {
                        Some(nodes) => nodes.as_slice(),
                        None => continue,
                    };
                if source_nodes.is_empty() {
                    continue;
                }

                // Scan bytes through the tokenizer DFA.
                let mut scan_state = representative_state;
                let mut scan_alive = true;
                let mut match_map: FxHashMap<TerminalID, (usize, u32)> = FxHashMap::default();
                for (index, &byte) in bytes.iter().enumerate() {
                    if let Some(next) = self.tokenizer.step(scan_state, byte) {
                        scan_state = next;
                        for terminal in self.tokenizer.matched_terminals_iter(scan_state) {
                            match_map.insert(terminal, (index + 1, scan_state));
                        }
                    } else {
                        scan_alive = false;
                        break;
                    }
                }
                let end_state = if scan_alive { Some(scan_state) } else { None };

                // Record terminal matches at the token endpoint.
                for (&terminal, &(width, _end_st)) in &match_map {
                    if width == bytes.len() {
                        self.profile.match_transition_additions += source_nodes.len() as u64;
                        self.add_leaf_token_from_sources(
                            source_nodes,
                            terminal,
                            internal_token_id,
                        );
                    }
                }

                // Future leaf: token fully consumed, tokenizer still alive.
                if let Some(end_state) = end_state {
                    self.add_future_leaf_token_from_sources(
                        source_nodes,
                        end_state,
                        internal_token_id,
                    );
                }
            }
        }
    }

    fn build_from_trie(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        let mut recursive_nodes = NodesByTokenizerState::new();
        let mut self_loop_only_nodes = NodesByTokenizerState::new();
        for (tokenizer_state, source_nodes) in assoc_by_state.iter() {
            if self.can_skip_self_loop_subtree(node, tokenizer_state) {
                self_loop_only_nodes.merge(tokenizer_state, source_nodes);
            } else {
                recursive_nodes.merge(tokenizer_state, source_nodes);
            }
        }

        if !self_loop_only_nodes.is_empty() {
            self.emit_self_loop_leaf_only_subtree(node, &self_loop_only_nodes);
        }

        if recursive_nodes.is_empty() {
            return;
        }

        for (segment_bytes, child_node) in node.iter_children() {
            let next_level_nodes = self.process_child_segment(
                segment_bytes,
                child_node,
                &recursive_nodes,
            );
            if !next_level_nodes.is_empty() {
                self.build_from_trie(child_node, &next_level_nodes);
            }
        }
    }

    fn process_child_segment(
        &mut self,
        segment_bytes: &[u8],
        child_node: &VocabPrefixTreeNode,
        initial_nodes: &NodesByTokenizerState,
    ) -> NodesByTokenizerState {
        // Token IDs in the trie are already internal (equivalence class) IDs.
        let leaf_token_id = child_node.token_id() as u32;
        let mut next_level_nodes = NodesByTokenizerState::new();
        let mut pending_by_offset = BTreeMap::<usize, NodesByTokenizerState>::new();
        pending_by_offset.insert(0, initial_nodes.clone());

        // Reusable buffers for DFA execution (avoids per-call allocation)
        let mut match_map_buf = FxHashMap::<TerminalID, (usize, u32)>::default();
        let mut matches_buf: Vec<TokenizerMatch> = Vec::new();

        while let Some((offset, nodes_at_offset)) = pending_by_offset.pop_first() {
            if offset == segment_bytes.len() {
                for (tokenizer_state, nwa_states) in nodes_at_offset {
                    next_level_nodes.merge(tokenizer_state, &nwa_states);
                }
                continue;
            }

            for (tokenizer_state, source_nodes) in nodes_at_offset {
                // Inline DFA scanning with flat transition table for O(1) per-byte stepping
                match_map_buf.clear();
                let mut scan_state = tokenizer_state;
                let mut scan_alive = true;
                for (index, &byte) in segment_bytes[offset..].iter().enumerate() {
                    if let Some(next) = self.fast_step(scan_state, byte) {
                        scan_state = next;
                        // Record longest match per terminal
                        for terminal in self.tokenizer.matched_terminals_iter(scan_state) {
                            match_map_buf.insert(terminal, (index + 1, scan_state));
                        }
                    } else {
                        scan_alive = false;
                        break;
                    }
                }
                let end_state = if scan_alive { Some(scan_state) } else { None };

                // Collect matches into reusable buffer
                matches_buf.clear();
                for (&id, &(width, end_st)) in match_map_buf.iter() {
                    matches_buf.push(TokenizerMatch { id, width, end_state: end_st });
                }

                if let Some(end_state) = end_state {
                    if child_node.has_token() {
                        self.add_future_leaf_token_from_sources(
                            &source_nodes,
                            end_state,
                            leaf_token_id,
                        );
                    }

                    next_level_nodes.merge(end_state, &source_nodes);
                }

                for matched in &matches_buf {
                    let next_offset = offset + matched.width;

                    if next_offset == segment_bytes.len() && child_node.has_token() {
                        self.profile.match_transition_additions += source_nodes.len() as u64;
                        self.add_leaf_token_from_sources(
                            &source_nodes,
                            matched.id,
                            leaf_token_id,
                        );
                    }

                    // L1 terminals never appear in multi-terminal paths, so
                    // skip continuation processing (no second terminal will
                    // follow within the same token).
                    if let Some(ref lengths) = self.terminal_path_lengths {
                        if let Some(&TerminalPathLength::One) = lengths.get(matched.id as usize) {
                            continue;
                        }
                    }

                    let Some(continuation_weight) = self.continuation_weight_for_match(
                        child_node,
                        leaf_token_id,
                        matched.id,
                        end_state,
                        next_offset == segment_bytes.len(),
                    ) else {
                        continue;
                    };
                    if continuation_weight.is_empty() {
                        continue;
                    }

                    let continuation_nodes = pending_by_offset
                        .entry(next_offset)
                        .or_insert_with(NodesByTokenizerState::new);
                    let destination = ensure_continuation_state(
                        continuation_nodes,
                        self.tokenizer.initial_state_id(),
                        self.nwa,
                    );

                    self.profile.match_transition_additions += source_nodes.len() as u64;
                    self.add_match_from_sources(
                        &source_nodes,
                        matched.id,
                        destination,
                        &continuation_weight,
                    );
                }
            }
        }

        next_level_nodes
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

fn ensure_continuation_state(
    pending: &mut NodesByTokenizerState,
    tokenizer_state: TokenizerState,
    nwa: &mut NWA,
) -> NwaState {
    if let Some(existing) = pending.first(tokenizer_state) {
        return existing;
    }

    let state = nwa.add_state();
    pending.push_one(tokenizer_state, state);
    state
}

fn internal_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {
    id_map
        .vocab_tokens
        .iter_representative_ids()
        .enumerate()
        .filter_map(|(internal_token_id, representative)| {
            vocab
                .entries
                .get(&representative)
                .map(|bytes| (internal_token_id as u32, bytes.clone()))
        })
        .collect()
}

fn seed_root_nodes(
    nwa: &mut NWA,
    start_state: u32,
    tokenizer: &Tokenizer,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    ignore_terminal: Option<TerminalID>,
    possible_matches_by_state: &PossibleMatchesByState,
) -> NodesByTokenizerState {
    let mut roots_by_tokenizer_state = NodesByTokenizerState::new();
    let mut roots_by_signature = HashMap::<u64, NwaState>::new();
    let mut start_weights_by_root = HashMap::<NwaState, Weight>::new();

    for (internal_tsid, representative_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        let combined_sig = root_combined_signature(
            tokenizer,
            representative_state,
            internal_tsid as u32,
            terminal_coloring,
            ignore_terminal,
            possible_matches_by_state,
        );

        let root = *roots_by_signature
            .entry(combined_sig)
            .or_insert_with(|| nwa.add_state());
        let start_weight = all_token_weight(internal_tsid as u32, id_map.max_internal_token_id());
        start_weights_by_root
            .entry(root)
            .and_modify(|existing| *existing = existing.union(&start_weight))
            .or_insert(start_weight);

        roots_by_tokenizer_state.merge(representative_state, &[root]);
    }

    let mut start_weight_entries: Vec<(NwaState, Weight)> = start_weights_by_root.into_iter().collect();
    start_weight_entries.sort_unstable_by_key(|(root, _)| *root);
    for (root, weight) in start_weight_entries {
        nwa.add_epsilon(start_state, root, weight);
    }

    roots_by_tokenizer_state
}

fn apply_disallowed_follow_constraints(nwa: &mut NWA, grammar: &AnalyzedGrammar) {
    let disallowed_follows = compute_disallowed_follows(grammar);
    let normalized = normalize_disallowed_follows(grammar.num_terminals as usize, &disallowed_follows);
    if normalized.iter().all(|bits| bits.is_zero()) {
        return;
    }

    let disallowed_dfa = build_disallowed_follow_dfa(&normalized);
    *nwa = subtract_disallowed_dfa(nwa, &disallowed_dfa);
}

/// Classifies a token's bytes by character type for vocab partitioning.
/// Returns 0 (pure non-alnum), 1 (mixed), or 2 (pure alnum).
pub(crate) fn classify_vocab_char_type(bytes: &[u8]) -> u8 {
    let mut has_alnum = false;
    let mut has_non_alnum = false;
    for &b in bytes {
        if b.is_ascii_alphanumeric() {
            has_alnum = true;
        } else {
            has_non_alnum = true;
        }
        if has_alnum && has_non_alnum {
            return 1; // Mixed
        }
    }
    if has_alnum { 2 } else { 0 } // PureAlnum or PureNonAlnum
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalPathLength {
    /// Terminal's first-byte bitset is disjoint from vocab byte bitset – ignorable.
    Zero,
    /// Single-step paths only – fast special case for id_map/DWA.
    One,
    /// Multi-terminal token paths possible – full treatment required.
    TwoPlus,
}

/// Classifies each terminal by the longest token-path length it can participate in.
///
/// - **Length 0**: No vocab byte from any tokenizer state can lead towards
///   matching this terminal.  The terminal is ignorable.
/// - **Length 1**: The terminal is matchable but never co-occurs with another
///   terminal inside a single vocab token.
/// - **Length 2+**: There exists a pair (t1, t2) of terminals with an allowed
///   follow relationship whose last/first bytes overlap the vocab byte bitset,
///   so a single token could span both.
pub(crate) fn classify_terminal_path_lengths(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
) -> Vec<TerminalPathLength> {
    let nt = num_terminals as usize;

    // 1. Vocab byte bitset: all bytes appearing in any vocab token.
    let mut vocab_bytes = U8Set::empty();
    for bytes in vocab.entries.values() {
        for &b in bytes {
            vocab_bytes.insert(b);
        }
    }

    // 2. Byte bitsets per terminal.
    let num_states = tokenizer.num_states() as usize;
    let initial = tokenizer.start_state();
    let dfa_states = tokenizer.dfa.states();

    // reachable_bytes[t]: bytes from ANY state that lead towards matching
    // terminal t (finalized or in possible_future).  Used for L0 check.
    let mut reachable_bytes = vec![U8Set::empty(); nt];
    for state in dfa_states {
        for (byte, target) in state.transitions.iter() {
            let target = *target;
            let finalizers = tokenizer.dfa.finalizers(target);
            let futures = tokenizer.dfa.possible_future_group_ids(target);
            for t in finalizers.iter().chain(futures.iter()) {
                if t < nt {
                    reachable_bytes[t].insert(byte);
                }
            }
        }
    }

    // first_bytes[t]: bytes from the INITIAL state leading towards terminal t.
    // After a terminal match the tokenizer resets to initial, so this is the
    // relevant set for "can t2 start after t1?".
    let mut first_bytes = vec![U8Set::empty(); nt];
    for (byte, target) in dfa_states[initial as usize].transitions.iter() {
        let target = *target;
        let finalizers = tokenizer.dfa.finalizers(target);
        let futures = tokenizer.dfa.possible_future_group_ids(target);
        for t in finalizers.iter().chain(futures.iter()) {
            if t < nt {
                first_bytes[t].insert(byte);
            }
        }
    }

    // last_bytes[t]: bytes on transitions arriving at states that finalize t.
    let mut incoming_bytes = vec![U8Set::empty(); num_states];
    for state in dfa_states {
        for (byte, target) in state.transitions.iter() {
            incoming_bytes[*target as usize].insert(byte);
        }
    }

    let mut last_bytes = vec![U8Set::empty(); nt];
    for (state_idx, _) in dfa_states.iter().enumerate() {
        let finalizers = tokenizer.dfa.finalizers(state_idx as u32);
        for t in finalizers.iter() {
            if t < nt {
                last_bytes[t] = last_bytes[t].union(&incoming_bytes[state_idx]);
            }
        }
    }

    // 3. Mark terminals that may participate in paths of length ≥ 2.
    let mut is_two_plus = BitSet::new(nt);

    for t1 in 0..nt {
        if last_bytes[t1].is_disjoint(&vocab_bytes) {
            continue;
        }
        let disallowed = disallowed_follows.get(&(t1 as u32));
        for t2 in 0..nt {
            if first_bytes[t2].is_disjoint(&vocab_bytes) {
                continue;
            }
            if let Some(d) = disallowed {
                if d.contains(t2) {
                    continue;
                }
            }
            is_two_plus.set(t1);
            is_two_plus.set(t2);
        }
    }

    // 4. Final classification.
    let mut result = vec![TerminalPathLength::Zero; nt];
    for t in 0..nt {
        if reachable_bytes[t].is_disjoint(&vocab_bytes) {
            result[t] = TerminalPathLength::Zero;
        } else if is_two_plus.contains(t) {
            result[t] = TerminalPathLength::TwoPlus;
        } else {
            result[t] = TerminalPathLength::One;
        }
    }

    result
}

/// Splits internal vocab entries into 3 partitions by character type:
/// [0] = pure non-alnum, [1] = mixed, [2] = pure alnum.
fn partition_internal_vocab(
    entries: Vec<(u32, Vec<u8>)>,
) -> [Vec<(usize, Vec<u8>)>; 3] {
    let mut partitions: [Vec<(usize, Vec<u8>)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (token_id, bytes) in entries {
        let idx = classify_vocab_char_type(&bytes) as usize;
        partitions[idx].push((token_id as usize, bytes));
    }
    partitions
}

/// Merges partition NWAs that share template states (start, leaf, root nodes).
///
/// Template states (indices `0..template_state_count`) are shared across all
/// partition NWAs.  Non-template states from each partition are renumbered to
/// occupy distinct ranges in the merged NWA.
fn merge_partition_nwas(
    template_state_count: u32,
    partition_nwas: Vec<NWA>,
) -> NWA {
    if partition_nwas.len() == 1 {
        return partition_nwas.into_iter().next().unwrap();
    }

    // Renumber offsets: partition i's non-template state `s` (where s >= T)
    // becomes `s + offsets[i]` in the merged NWA.
    let mut offsets = Vec::with_capacity(partition_nwas.len());
    let mut cumulative = 0u32;
    for nwa in &partition_nwas {
        offsets.push(cumulative);
        let extra = nwa.num_states() - template_state_count;
        cumulative += extra;
    }

    let total_states = template_state_count + cumulative;
    let renumber = |state: u32, p: usize| -> u32 {
        if state < template_state_count {
            state
        } else {
            state + offsets[p]
        }
    };

    let mut merged = NWA {
        states: Vec::with_capacity(total_states as usize),
        start_states: partition_nwas[0].start_states.clone(),
    };

    // Template states: union transitions from all partitions with
    // deduplication — identical (target, weight) entries from different
    // partitions (e.g. start→root epsilons) are merged rather than
    // tripled, keeping the NWA compact.
    for s in 0..template_state_count as usize {
        let mut state = NWAStateType::default();
        state.final_weight = partition_nwas[0].states[s].final_weight.clone();

        let mut eps_map: BTreeMap<u32, Weight> = BTreeMap::new();
        let mut trans_map: BTreeMap<i32, BTreeMap<u32, Weight>> = BTreeMap::new();

        for (p, nwa) in partition_nwas.iter().enumerate() {
            let src = &nwa.states[s];
            for (&label, targets) in &src.transitions {
                let m = trans_map.entry(label).or_default();
                for &(target, ref weight) in targets {
                    let t = renumber(target, p);
                    m.entry(t)
                        .and_modify(|w| *w = w.union(weight))
                        .or_insert_with(|| weight.clone());
                }
            }
            for &(target, ref weight) in &src.epsilons {
                let t = renumber(target, p);
                eps_map
                    .entry(t)
                    .and_modify(|w| *w = w.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }

        state.epsilons = eps_map.into_iter().collect();
        for (label, targets) in trans_map {
            state
                .transitions
                .insert(label, targets.into_iter().collect());
        }

        merged.states.push(state);
    }

    // Non-template states from each partition, renumbered.
    for (p, nwa) in partition_nwas.iter().enumerate() {
        for s in template_state_count as usize..nwa.num_states() as usize {
            let src = &nwa.states[s];
            let mut state = NWAStateType::default();
            state.final_weight = src.final_weight.clone();

            for (&label, targets) in &src.transitions {
                let v = state.transitions.entry(label).or_default();
                for &(target, ref weight) in targets {
                    v.push((renumber(target, p), weight.clone()));
                }
            }
            for &(target, ref weight) in &src.epsilons {
                state
                    .epsilons
                    .push((renumber(target, p), weight.clone()));
            }

            merged.states.push(state);
        }
    }

    merged
}

#[derive(Debug, Default)]
struct PartitionTerminalNwaBuild {
    nwa: Option<NWA>,
    possible_matches_by_state: PossibleMatchesByState,
    build_profile: TerminalDwaBuildProfile,
    possible_matches_profile: PossibleMatchesProfile,
    internal_vocab_len: usize,
    roots_count: usize,
}

fn build_partition_terminal_nwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    partition_index: usize,
) -> PartitionTerminalNwaBuild {
    let partition_total_start = std::time::Instant::now();
    let internal_vocab = internal_vocab_entries(vocab, id_map);
    if internal_vocab.is_empty() {
        return PartitionTerminalNwaBuild::default();
    }

    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let tree_start = std::time::Instant::now();
    let full_tree = VocabPrefixTree::build_owned(
        internal_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect(),
    );
    let tree_ms = tree_start.elapsed().as_secs_f64() * 1000.0;

    let pm_start = std::time::Instant::now();
    let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
    let possible_matches_by_state = collect_possible_matches_by_internal_tsid(
        tokenizer,
        &full_tree.root,
        &mut possible_matches,
        &id_map.tokenizer_states,
    );
    let pm_ms = pm_start.elapsed().as_secs_f64() * 1000.0;

    let seed_start = std::time::Instant::now();
    let roots_by_tokenizer_state = seed_root_nodes(
        &mut nwa,
        start_state,
        tokenizer,
        id_map,
        terminal_coloring,
        ignore_terminal,
        &possible_matches_by_state,
    );
    let seed_ms = seed_start.elapsed().as_secs_f64() * 1000.0;

    let terminal_path_lengths = classify_terminal_path_lengths(
        tokenizer,
        vocab,
        disallowed_follows,
        num_terminals,
    );
    let all_l1 = terminal_path_lengths.iter().all(|l| matches!(l, TerminalPathLength::Zero | TerminalPathLength::One));

    if debug_profile_enabled() {
        let n0 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::Zero).count();
        let n1 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::One).count();
        let n2 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::TwoPlus).count();
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build[{}] all_l1={} internal_vocab_len={} num_tsids={} l0={} l1={} l2p={} tree_ms={:.1} pm_ms={:.1} seed_ms={:.1}",
            partition_index, all_l1, internal_vocab.len(), id_map.num_tsids(), n0, n1, n2, tree_ms, pm_ms, seed_ms,
        );
    }

    let num_tsids = id_map.num_tsids();
    let num_tokenizer_states = tokenizer.num_states() as usize;
    let mut builder = TerminalNwaBuilder {
        tokenizer,
        terminal_coloring: terminal_coloring.clone(),
        possible_future_terminals: FxHashMap::default(),
        future_terminal_color_groups: FxHashMap::default(),
        possible_matches: &mut possible_matches,
        nwa: &mut nwa,
        num_tsids,
        leaf_state,
        ignore_terminal,
        use_terminal_coloring,
        terminal_path_lengths: Some(terminal_path_lengths.clone()),
        self_loop_bytes: FxHashMap::default(),
        leaf_token_ids_buffer: Vec::new(),
        future_leaf_buffer: FxHashMap::default(),
        reachable_weight_cache: HashMap::new(),
        pruned_weight_cache: HashMap::new(),
        leaf_weight_cache: HashMap::new(),
        transition_buffer: FxHashMap::default(),
        epsilon_buffer: FxHashMap::default(),
        profile: TerminalDwaBuildProfile::default(),
        flat_transitions: vec![None; num_tokenizer_states],
    };
    let nwa_build_start = std::time::Instant::now();
    if all_l1 && (internal_vocab.len() as u64) * (id_map.num_tsids() as u64) < 100_000 {
        builder.build_l1_fast(&internal_vocab, &roots_by_tokenizer_state, id_map);
    } else {
        builder.build_from_trie(&full_tree.root, &roots_by_tokenizer_state);
    }
    builder.flush_transition_buffer();
    let nwa_build_ms = nwa_build_start.elapsed().as_secs_f64() * 1000.0;
    let build_profile = builder.profile;
    drop(builder);
    let possible_matches_profile = possible_matches.profile();

    if debug_profile_enabled() {
        let partition_total_ms = partition_total_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build[{}] nwa_build_ms={:.1} nwa_states={} nwa_transitions={} partition_total_ms={:.1}",
            partition_index, nwa_build_ms, nwa.num_states(), nwa.num_transitions(), partition_total_ms,
        );
    }

    PartitionTerminalNwaBuild {
        nwa: Some(nwa),
        possible_matches_by_state,
        build_profile,
        possible_matches_profile,
        internal_vocab_len: internal_vocab.len(),
        roots_count: roots_by_tokenizer_state.entries.len(),
    }
}

fn build_local_to_global_tsid_map(
    local_id_map: &InternalIdMap,
    global_id_map: &InternalIdMap,
) -> Vec<Vec<u32>> {
    let mut local_to_global = vec![BTreeSet::new(); local_id_map.num_tsids() as usize];

    for (state_id, &local_tsid) in local_id_map
        .tokenizer_states
        .original_to_internal
        .iter()
        .enumerate()
    {
        let global_tsid = global_id_map.tokenizer_states.original_to_internal[state_id];
        local_to_global[local_tsid as usize].insert(global_tsid);
    }

    local_to_global
        .into_iter()
        .map(|bucket| bucket.into_iter().collect())
        .collect()
}

fn offset_token_set(tokens: &RangeSetBlaze<u32>, offset: u32) -> RangeSetBlaze<u32> {
    if offset == 0 {
        return tokens.clone();
    }

    RangeSetBlaze::from_iter(
        tokens
            .ranges()
            .map(|range| (*range.start() + offset)..=(*range.end() + offset)),
    )
}

fn remap_weight_to_global(
    weight: &Weight,
    local_to_global_tsids: &[Vec<u32>],
    token_offset: u32,
    token_cache: &mut HashMap<usize, RangeSetBlaze<u32>>,
) -> Weight {
    if weight.is_empty() || weight.is_full() {
        return weight.clone();
    }

    let Some(entries) = weight.compact_entries() else {
        return weight.clone();
    };

    let mut tokens_by_global_tsid = BTreeMap::<u32, RangeSetBlaze<u32>>::new();
    for (start, end, tokens) in entries {
        let token_key = std::sync::Arc::as_ptr(&tokens) as usize;
        let mapped_tokens = token_cache
            .entry(token_key)
            .or_insert_with(|| offset_token_set(tokens.as_ref(), token_offset))
            .clone();

        for local_tsid in start..=end {
            let Some(global_tsids) = local_to_global_tsids.get(local_tsid as usize) else {
                continue;
            };
            for &global_tsid in global_tsids {
                tokens_by_global_tsid
                    .entry(global_tsid)
                    .and_modify(|existing| *existing |= mapped_tokens.clone())
                    .or_insert_with(|| mapped_tokens.clone());
            }
        }
    }

    if tokens_by_global_tsid.is_empty() {
        return Weight::empty();
    }

    // Build WeightMap directly with interned/shared token sets.
    // This is O(n) instead of the O(n²) from_compact_ranges path,
    // and deduplicates identical token sets via the global interner.
    use crate::ds::weight::{finalize_weight_map, shared_rangeset};
    use range_set_blaze::RangeMapBlaze;

    let mut map = RangeMapBlaze::<u32, std::sync::Arc<RangeSetBlaze<u32>>>::new();
    let mut iter = tokens_by_global_tsid.into_iter();
    let (mut run_start, first_tokens) = iter.next().unwrap();
    let mut run_end = run_start;
    let mut run_shared = shared_rangeset(first_tokens);

    for (global_tsid, tokens) in iter {
        let next_shared = shared_rangeset(tokens);
        if global_tsid == run_end + 1
            && (std::sync::Arc::ptr_eq(&run_shared, &next_shared)
                || run_shared.as_ref() == next_shared.as_ref())
        {
            run_end = global_tsid;
        } else {
            map.extend_simple(std::iter::once((
                run_start..=run_end,
                std::sync::Arc::clone(&run_shared),
            )));
            run_start = global_tsid;
            run_end = global_tsid;
            run_shared = next_shared;
        }
    }
    map.extend_simple(std::iter::once((run_start..=run_end, run_shared)));

    finalize_weight_map(map)
}

fn remap_partition_nwa_to_global(
    nwa: &mut NWA,
    local_to_global_tsids: &[Vec<u32>],
    token_offset: u32,
) {
    let mut token_cache = HashMap::<usize, RangeSetBlaze<u32>>::new();
    let mut weight_cache = HashMap::<usize, Weight>::new();

    let remap = |weight: &Weight,
                 token_cache: &mut HashMap<usize, RangeSetBlaze<u32>>,
                 weight_cache: &mut HashMap<usize, Weight>|
     -> Weight {
        let weight_ptr = std::sync::Arc::as_ptr(&weight.0) as usize;
        if let Some(cached) = weight_cache.get(&weight_ptr) {
            return cached.clone();
        }
        let remapped = remap_weight_to_global(
            weight,
            local_to_global_tsids,
            token_offset,
            token_cache,
        );
        weight_cache.insert(weight_ptr, remapped.clone());
        remapped
    };

    for state in &mut nwa.states {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = remap(final_weight, &mut token_cache, &mut weight_cache);
            if final_weight.is_empty() {
                state.final_weight = None;
            }
        }

        for targets in state.transitions.values_mut() {
            for (_, weight) in targets.iter_mut() {
                *weight = remap(weight, &mut token_cache, &mut weight_cache);
            }
            targets.retain(|(_, weight)| !weight.is_empty());
        }
        state.transitions.retain(|_, targets| !targets.is_empty());

        for (_, weight) in state.epsilons.iter_mut() {
            *weight = remap(weight, &mut token_cache, &mut weight_cache);
        }
        state.epsilons.retain(|(_, weight)| !weight.is_empty());
    }
}

fn merge_partition_possible_matches_into_global(
    merged: &mut PossibleMatchesByState,
    partition_matches: &PossibleMatchesByState,
    local_to_global_tsids: &[Vec<u32>],
    token_offset: u32,
) {
    for (&local_tsid, matches_by_terminal) in partition_matches {
        let Some(global_tsids) = local_to_global_tsids.get(local_tsid as usize) else {
            continue;
        };

        for &global_tsid in global_tsids {
            let global_matches = merged.entry(global_tsid).or_default();
            for (&terminal_id, token_ids) in matches_by_terminal {
                let mapped_tokens = offset_token_set(token_ids, token_offset);
                global_matches
                    .entry(terminal_id)
                    .and_modify(|existing| *existing |= mapped_tokens.clone())
                    .or_insert(mapped_tokens);
            }
        }
    }
}

pub(crate) fn build_terminal_dwa_from_partition_id_maps_with_possible_matches_and_coloring(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    full_vocab: &Vocab,
    partition_vocabs: &[Vocab],
    partition_id_maps: &[InternalIdMap],
    global_id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> (DWA, PossibleMatchesByState) {
    debug_assert_eq!(partition_vocabs.len(), partition_id_maps.len());

    let debug_profile = debug_profile_enabled();
    let profile_enabled = terminal_dwa_profile_enabled();
    let total_started_at = std::time::Instant::now();
    let raw_build_started_at = std::time::Instant::now();

    let num_terminals = grammar.num_terminals;

    let builds: Vec<PartitionTerminalNwaBuild> = match partition_vocabs.len() {
        0 => Vec::new(),
        1 => vec![build_partition_terminal_nwa(
            tokenizer,
            &partition_vocabs[0],
            &partition_id_maps[0],
            terminal_coloring,
            use_terminal_coloring,
            ignore_terminal,
            disallowed_follows,
            num_terminals,
            0,
        )],
        2 => {
            let (left, right) = rayon::join(
                || build_partition_terminal_nwa(
                    tokenizer,
                    &partition_vocabs[0],
                    &partition_id_maps[0],
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    disallowed_follows,
                    num_terminals,
                    0,
                ),
                || build_partition_terminal_nwa(
                    tokenizer,
                    &partition_vocabs[1],
                    &partition_id_maps[1],
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    disallowed_follows,
                    num_terminals,
                    1,
                ),
            );
            vec![left, right]
        }
        _ => {
            let (left, (middle, right)) = rayon::join(
                || build_partition_terminal_nwa(
                    tokenizer,
                    &partition_vocabs[0],
                    &partition_id_maps[0],
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    disallowed_follows,
                    num_terminals,
                    0,
                ),
                || rayon::join(
                    || build_partition_terminal_nwa(
                        tokenizer,
                        &partition_vocabs[1],
                        &partition_id_maps[1],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        disallowed_follows,
                        num_terminals,
                        1,
                    ),
                    || build_partition_terminal_nwa(
                        tokenizer,
                        &partition_vocabs[2],
                        &partition_id_maps[2],
                        terminal_coloring,
                        use_terminal_coloring,
                        ignore_terminal,
                        disallowed_follows,
                        num_terminals,
                        2,
                    ),
                ),
            );
            let mut results = vec![left, middle, right];
            for index in 3..partition_vocabs.len() {
                results.push(build_partition_terminal_nwa(
                    tokenizer,
                    &partition_vocabs[index],
                    &partition_id_maps[index],
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    disallowed_follows,
                    num_terminals,
                    index,
                ));
            }
            results
        }
    };

    let mut aggregate_profile = TerminalDwaBuildProfile::default();
    let mut merged_nwa = NWA::new(global_id_map.num_tsids(), global_id_map.max_internal_token_id());
    let mut merged_body = merged_nwa.body();
    let mut merged_possible_matches = PossibleMatchesByState::new();
    let mut aggregate_possible_matches_profile = PossibleMatchesProfile::default();
    let mut aggregate_internal_vocab_len = 0usize;
    let mut aggregate_roots = 0usize;
    let mut token_offset = 0u32;

    for (build, partition_id_map) in builds.into_iter().zip(partition_id_maps.iter()) {
        aggregate_profile.future_terminal_additions += build.build_profile.future_terminal_additions;
        aggregate_profile.match_transition_additions += build.build_profile.match_transition_additions;
        aggregate_possible_matches_profile.cache_hits += build.possible_matches_profile.cache_hits;
        aggregate_possible_matches_profile.cache_misses += build.possible_matches_profile.cache_misses;
        aggregate_possible_matches_profile.reachable_cache_hits += build.possible_matches_profile.reachable_cache_hits;
        aggregate_possible_matches_profile.reachable_cache_misses += build.possible_matches_profile.reachable_cache_misses;
        aggregate_possible_matches_profile.child_segments_visited += build.possible_matches_profile.child_segments_visited;
        aggregate_possible_matches_profile.byte_steps += build.possible_matches_profile.byte_steps;
        aggregate_possible_matches_profile.blocked_segments += build.possible_matches_profile.blocked_segments;
        aggregate_possible_matches_profile.recursive_descents += build.possible_matches_profile.recursive_descents;
        aggregate_possible_matches_profile.terminal_insertions += build.possible_matches_profile.terminal_insertions;
        aggregate_possible_matches_profile.cache_entries += build.possible_matches_profile.cache_entries;
        aggregate_possible_matches_profile.reachable_cache_entries += build.possible_matches_profile.reachable_cache_entries;
        aggregate_internal_vocab_len += build.internal_vocab_len;
        aggregate_roots += build.roots_count;

        if let Some(mut partition_nwa) = build.nwa {
            let local_to_global_tsids = build_local_to_global_tsid_map(partition_id_map, global_id_map);
            remap_partition_nwa_to_global(&mut partition_nwa, &local_to_global_tsids, token_offset);
            merge_partition_possible_matches_into_global(
                &mut merged_possible_matches,
                &build.possible_matches_by_state,
                &local_to_global_tsids,
                token_offset,
            );
            merged_body = merged_nwa.union_in_place(&partition_nwa, &merged_body);
        }

        token_offset = partition_id_map
            .num_internal_tokens()
            .checked_add(token_offset)
            .expect("global token offset overflow while merging partition terminal DWAs");
    }
    merged_nwa.start_states = merged_body.start_states;

    let raw_build_ms = raw_build_started_at.elapsed().as_secs_f64() * 1000.0;

    if merged_nwa.states.is_empty() {
        return (DWA::new(global_id_map.num_tsids(), global_id_map.max_internal_token_id()), merged_possible_matches);
    }

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] partitioned_merge internal_tokenizer_states={} vocab_entries={} roots={} possible_matches_states={} raw_build_ms={:.3}",
            global_id_map.num_tsids(),
            aggregate_internal_vocab_len,
            aggregate_roots,
            merged_possible_matches.len(),
            raw_build_ms,
        );
    }

    let always_allowed_started_at = std::time::Instant::now();
    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;

    let collapse_started_at = std::time::Instant::now();
    let _ = collapse_always_allowed(&mut merged_nwa, &always_allowed_by_label, grammar.num_terminals as usize);
    let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;

    let disallowed_started_at = std::time::Instant::now();
    apply_disallowed_follow_constraints(&mut merged_nwa, grammar);
    let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;

    let coreachable_prune_started_at = std::time::Instant::now();
    prune_non_coreachable_states(&mut merged_nwa);
    let coreachable_prune_ms = coreachable_prune_started_at.elapsed().as_secs_f64() * 1000.0;

    let canonicalize_started_at = std::time::Instant::now();
    canonicalize_acyclic_nwa(&mut merged_nwa);
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let nwa_states = merged_nwa.num_states();
    let nwa_transitions = merged_nwa.num_transitions();

    let determinize_started_at = std::time::Instant::now();
    let determinized = determinize(&merged_nwa)
        .expect("terminal NWA determinization failed despite acyclic token trie construction");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let determinized_states = determinized.num_states();
    let determinized_transitions = determinized.num_transitions();

    let minimize_started_at = std::time::Instant::now();
    let dwa = minimize(&determinized);
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] colors={} future_terminal_additions={} match_transition_additions={}",
            terminal_coloring.num_colors,
            aggregate_profile.future_terminal_additions,
            aggregate_profile.match_transition_additions,
        );
    }

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] tokenizer_states={} internal_tokenizer_states={} vocab_entries={} roots={} possible_matches_states={} possible_matches_cache_entries={} reachable_cache_entries={} nwa_states={} nwa_transitions={} determinized_states={} determinized_transitions={} minimized_states={}",
            tokenizer.num_states(),
            global_id_map.num_tsids(),
            aggregate_internal_vocab_len,
            aggregate_roots,
            merged_possible_matches.len(),
            aggregate_possible_matches_profile.cache_entries,
            aggregate_possible_matches_profile.reachable_cache_entries,
            nwa_states,
            nwa_transitions,
            determinized_states,
            determinized_transitions,
            dwa.num_states(),
        );
        eprintln!(
            "[glrmask/debug][terminal_dwa] raw_build_ms={:.3} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} coreachable_prune_ms={:.3} canonicalize_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            raw_build_ms,
            always_allowed_ms,
            collapse_ms,
            disallowed_ms,
            coreachable_prune_ms,
            canonicalize_ms,
            determinize_ms,
            minimize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        eprintln!(
            "[glrmask/debug][terminal_dwa] possible_matches cache_hits={} cache_misses={} reachable_hits={} reachable_misses={} child_segments={} byte_steps={} blocked_segments={} recursive_descents={} terminal_insertions={}",
            aggregate_possible_matches_profile.cache_hits,
            aggregate_possible_matches_profile.cache_misses,
            aggregate_possible_matches_profile.reachable_cache_hits,
            aggregate_possible_matches_profile.reachable_cache_misses,
            aggregate_possible_matches_profile.child_segments_visited,
            aggregate_possible_matches_profile.byte_steps,
            aggregate_possible_matches_profile.blocked_segments,
            aggregate_possible_matches_profile.recursive_descents,
            aggregate_possible_matches_profile.terminal_insertions,
        );
        emit_terminal_dwa_token_map(&dwa, full_vocab, global_id_map);
    }

    (dwa, merged_possible_matches)
}

pub(crate) fn build_terminal_dwa(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    let terminal_coloring = TerminalColoring::identity(grammar.num_terminals as usize);
    build_terminal_dwa_with_possible_matches_and_coloring(
        grammar,
        tokenizer,
        vocab,
        id_map,
        &terminal_coloring,
        false,
        ignore_terminal,
        None,
    )
    .0
}

pub(crate) fn build_terminal_dwa_with_possible_matches_and_coloring(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
) -> (DWA, PossibleMatchesByState) {
    let debug_profile = debug_profile_enabled();
    let total_started_at = std::time::Instant::now();
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let setup_started_at = std::time::Instant::now();
    let mut internal_vocab = internal_vocab_entries(vocab, id_map);
    let internal_vocab_len = internal_vocab.len();

    // Build the full vocab tree (needed for possible_matches computation).
    let full_tree = VocabPrefixTree::build_owned(
        internal_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect(),
    );

    // Check if all terminals are L1 (or L0) for fast path.
    let empty_disallowed = BTreeMap::new();
    let effective_disallowed = disallowed_follows.unwrap_or(&empty_disallowed);
    let terminal_path_lengths = classify_terminal_path_lengths(
        tokenizer,
        vocab,
        effective_disallowed,
        grammar.num_terminals,
    );
    let all_l1 = terminal_path_lengths.iter().all(|l| matches!(l, TerminalPathLength::Zero | TerminalPathLength::One));

    if debug_profile_enabled() {
        let n0 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::Zero).count();
        let n1 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::One).count();
        let n2 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::TwoPlus).count();
        eprintln!(
            "[glrmask/debug][terminal_dwa] non_partition_build all_l1={} internal_vocab_len={} l0={} l1={} l2p={}",
            all_l1, internal_vocab.len(), n0, n1, n2,
        );
    }

    // Partition the vocab into 3 sets by character type and build per-partition trees.
    // Skip partitioning on the L1 fast path (trie trees are not needed).
    let partition_sizes: [usize; 3];
    let partition_trees: Vec<VocabPrefixTree>;
    if all_l1 {
        partition_sizes = [0, 0, 0];
        partition_trees = Vec::new();
    } else {
        let partitions = partition_internal_vocab(std::mem::take(&mut internal_vocab));
        partition_sizes = [
            partitions[0].len(),
            partitions[1].len(),
            partitions[2].len(),
        ];
        partition_trees = partitions
            .into_iter()
            .map(|entries| VocabPrefixTree::build_owned(entries))
            .collect();
    }

    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let profile_enabled = terminal_dwa_profile_enabled();
    let mut possible_matches = PossibleMatchesComputer::new(tokenizer);

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] start grammar_rules={} grammar_terminals={} grammar_nonterminals={} tokenizer_states={} internal_tokenizer_states={} vocab_entries={} partitions=[{},{},{}] setup_ms={:.3}",
            grammar.rules.len(),
            grammar.num_terminals,
            grammar.num_nonterminals,
            tokenizer.num_states(),
            id_map.num_tsids(),
            internal_vocab_len,
            partition_sizes[0],
            partition_sizes[1],
            partition_sizes[2],
            setup_ms,
        );
    }

    let possible_matches_started_at = std::time::Instant::now();
    let possible_matches_by_state = collect_possible_matches_by_internal_tsid(
        tokenizer,
        &full_tree.root,
        &mut possible_matches,
        &id_map.tokenizer_states,
    );
    let possible_matches_ms = possible_matches_started_at.elapsed().as_secs_f64() * 1000.0;
    let possible_matches_profile = possible_matches.profile();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=possible_matches states={} cache_entries={} reachable_cache_entries={} ms={:.3}",
            possible_matches_by_state.len(),
            possible_matches_profile.cache_entries,
            possible_matches_profile.reachable_cache_entries,
            possible_matches_ms,
        );
    }

    let seed_started_at = std::time::Instant::now();
    let roots_by_tokenizer_state = seed_root_nodes(
        &mut nwa,
        start_state,
        tokenizer,
        id_map,
        terminal_coloring,
        ignore_terminal,
        &possible_matches_by_state,
    );
    let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let template_state_count = nwa.num_states();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=seed roots={} template_states={} ms={:.3}",
            roots_by_tokenizer_state.entries.len(),
            template_state_count,
            seed_ms,
        );
    }

    // Build NWA transitions.  When all terminals are L1 (no multi-terminal
    // token paths), use a fast flat loop instead of the trie walk.
    let build_trie_started_at = std::time::Instant::now();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=nwa_build all_l1={} internal_vocab_len={}",
            all_l1,
            if all_l1 { internal_vocab.len() } else { 0 },
        );
    }

    let num_tsids = id_map.num_tsids();
    let num_tokenizer_states = tokenizer.num_states() as usize;

    let profile = if all_l1 {
        // L1 fast path: direct token→terminal→leaf NWA construction.
        let mut pm = PossibleMatchesComputer::new(tokenizer);
        let mut builder = TerminalNwaBuilder {
            tokenizer,
            terminal_coloring: terminal_coloring.clone(),
            possible_future_terminals: FxHashMap::default(),
            future_terminal_color_groups: FxHashMap::default(),
            possible_matches: &mut pm,
            nwa: &mut nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            use_terminal_coloring,
            terminal_path_lengths: Some(terminal_path_lengths.clone()),
            self_loop_bytes: FxHashMap::default(),
            leaf_token_ids_buffer: Vec::new(),
            future_leaf_buffer: FxHashMap::default(),
            reachable_weight_cache: HashMap::new(),
            pruned_weight_cache: HashMap::new(),
            leaf_weight_cache: HashMap::new(),
            transition_buffer: FxHashMap::default(),
            epsilon_buffer: FxHashMap::default(),
            profile: TerminalDwaBuildProfile::default(),
            flat_transitions: vec![None; num_tokenizer_states],
        };
        builder.build_l1_fast(&internal_vocab, &roots_by_tokenizer_state, id_map);
        builder.flush_transition_buffer();
        let prof = builder.profile;
        drop(builder);
        prof
    } else {
        // Normal path: partition into 3 char-type groups and trie-walk.
        let template_nwa = &nwa;

        let build_one = |tree: &VocabPrefixTree| -> (NWA, TerminalDwaBuildProfile) {
            let mut part_nwa = template_nwa.clone();
            let mut pm = PossibleMatchesComputer::new(tokenizer);
            let mut builder = TerminalNwaBuilder {
                tokenizer,
                terminal_coloring: terminal_coloring.clone(),
                possible_future_terminals: FxHashMap::default(),
                future_terminal_color_groups: FxHashMap::default(),
                possible_matches: &mut pm,
                nwa: &mut part_nwa,
                num_tsids,
                leaf_state,
                ignore_terminal,
                use_terminal_coloring,
                terminal_path_lengths: Some(terminal_path_lengths.clone()),
                self_loop_bytes: FxHashMap::default(),
                leaf_token_ids_buffer: Vec::new(),
                future_leaf_buffer: FxHashMap::default(),
                reachable_weight_cache: HashMap::new(),
                pruned_weight_cache: HashMap::new(),
                leaf_weight_cache: HashMap::new(),
                transition_buffer: FxHashMap::default(),
                epsilon_buffer: FxHashMap::default(),
                profile: TerminalDwaBuildProfile::default(),
                flat_transitions: vec![None; num_tokenizer_states],
            };
            builder.build_from_trie(&tree.root, &roots_by_tokenizer_state);
            builder.flush_transition_buffer();
            let prof = builder.profile;
            drop(builder);
            (part_nwa, prof)
        };

        let ((nwa_a, prof_a), ((nwa_b, prof_b), (nwa_c, prof_c))) = rayon::join(
            || build_one(&partition_trees[0]),
            || rayon::join(
                || build_one(&partition_trees[1]),
                || build_one(&partition_trees[2]),
            ),
        );

        nwa = merge_partition_nwas(
            template_state_count,
            vec![nwa_a, nwa_b, nwa_c],
        );
        TerminalDwaBuildProfile {
            future_terminal_additions: prof_a.future_terminal_additions
                + prof_b.future_terminal_additions
                + prof_c.future_terminal_additions,
            match_transition_additions: prof_a.match_transition_additions
                + prof_b.match_transition_additions
                + prof_c.match_transition_additions,
        }
    };
    let build_trie_ms = build_trie_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=build_trie nwa_states={} nwa_transitions={} ms={:.3}",
            nwa.num_states(),
            nwa.num_transitions(),
            build_trie_ms,
        );
    }

    let always_allowed_started_at = std::time::Instant::now();
    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;

    let collapse_started_at = std::time::Instant::now();
    let _ = collapse_always_allowed(&mut nwa, &always_allowed_by_label, grammar.num_terminals as usize);
    let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;

    let disallowed_started_at = std::time::Instant::now();
    apply_disallowed_follow_constraints(&mut nwa, grammar);
    let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;

    // Prune non-co-reachable states (dead ends), then canonicalize
    // (which deduplicates structurally identical states including roots).
    let coreachable_prune_started_at = std::time::Instant::now();
    prune_non_coreachable_states(&mut nwa);
    let coreachable_prune_ms = coreachable_prune_started_at.elapsed().as_secs_f64() * 1000.0;

    let canonicalize_started_at = std::time::Instant::now();
    canonicalize_acyclic_nwa(&mut nwa);
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] after_cleanup nwa_states={} nwa_transitions={}",
            nwa.num_states(),
            nwa.num_transitions(),
        );
    }

    let nwa_states = nwa.num_states();
    let nwa_transitions = nwa.num_transitions();

    let determinize_started_at = std::time::Instant::now();
    let determinized = determinize(&nwa)
        .expect("terminal NWA determinization failed despite acyclic token trie construction");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let determinized_states = determinized.num_states();
    let determinized_transitions = determinized.num_transitions();

    let minimize_started_at = std::time::Instant::now();
    let dwa = minimize(&determinized);
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] colors={} future_terminal_additions={} match_transition_additions={}",
            terminal_coloring.num_colors,
            profile.future_terminal_additions,
            profile.match_transition_additions,
        );
    }

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] tokenizer_states={} internal_tokenizer_states={} vocab_entries={} roots={} possible_matches_states={} possible_matches_cache_entries={} reachable_cache_entries={} nwa_states={} nwa_transitions={} determinized_states={} determinized_transitions={} minimized_states={}",
            tokenizer.num_states(),
            id_map.num_tsids(),
            internal_vocab_len,
            roots_by_tokenizer_state.entries.len(),
            possible_matches_by_state.len(),
            possible_matches_profile.cache_entries,
            possible_matches_profile.reachable_cache_entries,
            nwa_states,
            nwa_transitions,
            determinized_states,
            determinized_transitions,
            dwa.num_states(),
        );
        eprintln!(
            "[glrmask/debug][terminal_dwa] setup_ms={:.3} seed_ms={:.3} build_trie_ms={:.3} possible_matches_ms={:.3} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} coreachable_prune_ms={:.3} canonicalize_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            setup_ms,
            seed_ms,
            build_trie_ms,
            possible_matches_ms,
            always_allowed_ms,
            collapse_ms,
            disallowed_ms,
            coreachable_prune_ms,
            canonicalize_ms,
            determinize_ms,
            minimize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        eprintln!(
            "[glrmask/debug][terminal_dwa] possible_matches cache_hits={} cache_misses={} reachable_hits={} reachable_misses={} child_segments={} byte_steps={} blocked_segments={} recursive_descents={} terminal_insertions={}",
            possible_matches_profile.cache_hits,
            possible_matches_profile.cache_misses,
            possible_matches_profile.reachable_cache_hits,
            possible_matches_profile.reachable_cache_misses,
            possible_matches_profile.child_segments_visited,
            possible_matches_profile.byte_steps,
            possible_matches_profile.blocked_segments,
            possible_matches_profile.recursive_descents,
            possible_matches_profile.terminal_insertions,
        );
    }

    if debug_profile {
        emit_terminal_dwa_token_map(&dwa, vocab, id_map);
        emit_terminal_dwa_debug_dump(&dwa);
    }

    (dwa, possible_matches_by_state)
}

fn emit_terminal_dwa_token_map(dwa: &DWA, vocab: &Vocab, id_map: &InternalIdMap) {
    let internal_vocab = internal_vocab_entries(vocab, id_map);
    let internal_bytes: std::collections::BTreeMap<u32, &[u8]> =
        internal_vocab.iter().map(|(id, bytes)| (*id, bytes.as_slice())).collect();
    let mut referenced_tokens = std::collections::BTreeSet::new();
    for state in &dwa.states {
        for (_, (_, weight)) in &state.transitions {
            for tid in weight.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
        if let Some(fw) = &state.final_weight {
            for tid in fw.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
    }
    for tid in &referenced_tokens {
        if let Some(bytes) = internal_bytes.get(tid) {
            let originals = id_map.vocab_tokens.internal_to_originals.get(*tid as usize)
                .map(|v| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .unwrap_or_else(|| "?".into());
            eprintln!(
                "[glrmask/debug][terminal_dwa][token_map] internal={} originals=[{}] bytes={:?}",
                tid, originals, String::from_utf8_lossy(bytes)
            );
        }
    }
}

fn emit_terminal_dwa_debug_dump(dwa: &DWA) {
    let num_states = dwa.num_states() as usize;
    let start_state = dwa.start_state as usize;
    let mut incoming_counts = vec![0usize; num_states];
    let mut outgoing_counts = vec![0usize; num_states];
    let mut final_states = 0usize;
    let mut self_loops = 0usize;
    let mut transitions_to_start = 0usize;
    let mut transitions_from_start = 0usize;
    let mut transitions_from_start_to_start = 0usize;

    for (from, state) in dwa.states.iter().enumerate() {
        outgoing_counts[from] = state.transitions.len();
        if state.final_weight.is_some() {
            final_states += 1;
        }
        for (_, (to, _)) in &state.transitions {
            let to = *to as usize;
            if let Some(slot) = incoming_counts.get_mut(to) {
                *slot += 1;
            }
            if to == from {
                self_loops += 1;
            }
            if to == start_state {
                transitions_to_start += 1;
                if from == start_state {
                    transitions_from_start_to_start += 1;
                }
            }
        }
        if from == start_state {
            transitions_from_start = state.transitions.len();
        }
    }

    eprintln!(
        "[glrmask/debug][terminal_dwa][summary] states={} transitions={} start_state={} final_states={} transitions_from_start={} transitions_to_start={} transitions_to_start_from_non_start={} start_to_start={} self_loops={}",
        num_states,
        dwa.num_transitions(),
        dwa.start_state,
        final_states,
        transitions_from_start,
        transitions_to_start,
        transitions_to_start.saturating_sub(transitions_from_start_to_start),
        transitions_from_start_to_start,
        self_loops,
    );

    for (state_id, state) in dwa.states.iter().enumerate() {
        let incoming = incoming_counts[state_id];
        let outgoing = outgoing_counts[state_id];
        let to_start = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == start_state)
            .count();
        let self_loop_count = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == state_id)
            .count();
        let final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| format!("{weight}"))
            .unwrap_or_else(|| "none".to_string());
        let start_mark = if state_id == start_state {
            " [START]"
        } else {
            ""
        };

        eprintln!(
            "[glrmask/debug][terminal_dwa][state] id={}{} incoming={} outgoing={} to_start={} self_loops={} final={}",
            state_id,
            start_mark,
            incoming,
            outgoing,
            to_start,
            self_loop_count,
            final_weight,
        );

        for (label, (target, weight)) in &state.transitions {
            eprintln!("    {label} -> State {target}");
            eprintln!("      weight: {weight}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};
    use crate::compiler::grammar::model::tests::simple_ab_grammar;
    use crate::compiler::stages::equivalence_analysis::{ManyToOneIdMap, refine_partition_id_maps};
    use std::collections::{BTreeMap, BTreeSet};

    fn expand_original_tokens(weight: &Weight, id_map: &InternalIdMap) -> BTreeSet<u32> {
        let mut original_tokens = BTreeSet::new();
        for internal_token_id in weight.token_union().iter() {
            if let Some(original_ids) = id_map.vocab_tokens.original_ids_for_internal(internal_token_id) {
                original_tokens.extend(original_ids.iter());
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
    fn test_terminal_dwa_uses_representative_bytes_for_internal_token() {
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
            internal_to_originals: vec![vec![10u32, 20u32]],
            representative_original_ids: vec![10],
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
            !b_original_tokens.contains(&10) && !b_original_tokens.contains(&20),
            "non-representative bytes should not create additional terminal paths"
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

    #[test]
    fn test_partition_local_terminal_dwa_matches_global_build() {
        let grammar = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(2)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(2), Symbol::Terminal(1)] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"!".to_vec() },
                Terminal::Literal { id: 2, bytes: b"a!".to_vec() },
            ],
            ..Default::default()
        };
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"!".to_vec()),
                (2, b"a!".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"!!".to_vec()),
                (5, b"!a".to_vec()),
            ],
            None,
        );

        let mut partition_entries: [Vec<(u32, Vec<u8>)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (&token_id, bytes) in &vocab.entries {
            partition_entries[classify_vocab_char_type(bytes) as usize].push((token_id, bytes.clone()));
        }
        let partition_vocabs: Vec<Vocab> = partition_entries
            .into_iter()
            .map(|entries| Vocab::new(entries, None))
            .collect();
        let partition_maps: Vec<InternalIdMap> = partition_vocabs
            .iter()
            .map(|sub_vocab| InternalIdMap::build(&tokenizer, sub_vocab, &BTreeMap::new(), None))
            .collect();
        let global_id_map = refine_partition_id_maps(
            &partition_maps,
            tokenizer.num_states() as usize,
            vocab.max_token_id(),
        );
        let terminal_coloring = TerminalColoring::identity(analyzed.num_terminals as usize);

        let (global_dwa, global_matches) = build_terminal_dwa_with_possible_matches_and_coloring(
            &analyzed,
            &tokenizer,
            &vocab,
            &global_id_map,
            &terminal_coloring,
            false,
            None,
            None,
        );
        let (partitioned_dwa, partitioned_matches) =
            build_terminal_dwa_from_partition_id_maps_with_possible_matches_and_coloring(
                &analyzed,
                &tokenizer,
                &vocab,
                &partition_vocabs,
                &partition_maps,
                &global_id_map,
                &terminal_coloring,
                false,
                None,
                &BTreeMap::new(),
            );

        assert_eq!(partitioned_matches, global_matches);

        let sequences = [
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![2, 1],
            vec![1, 0],
        ];
        for sequence in sequences {
            let global_tokens = expand_original_tokens(&global_dwa.eval_word(&sequence), &global_id_map);
            let partitioned_tokens = expand_original_tokens(&partitioned_dwa.eval_word(&sequence), &global_id_map);
            assert_eq!(
                partitioned_tokens,
                global_tokens,
                "partitioned terminal DWA disagreed with global build for sequence {:?}",
                sequence,
            );
        }
    }
}
