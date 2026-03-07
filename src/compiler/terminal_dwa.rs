//! Terminal DWA construction.
//!
//! The paper architecture has a single terminal-side compilation object: the
//! Terminal DWA. This module now reflects that cardinality directly.
//!
//! The current implementation is still reduced relative to `sep1`, but it now
//! walks actual vocabulary tokens through the tokenizer and builds terminal-path
//! structure instead of projecting everything from `possible_matches`.

use std::collections::BTreeSet;

use crate::Vocab;
use crate::automata::weighted::nwa::Nwa;
use crate::automata::weighted::weight::Weight;
use crate::compiler::glr::grammar::{EOF, GlrGrammar};
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::grammar_def::Symbol;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::ds::RangeSet;

/// Reduced terminal-side compilation artifact.
#[derive(Debug, Clone)]
pub struct TerminalDwa {
    pub nwa: Nwa,
    pub tsid_roots: Vec<u32>,
    /// Non-greedy terminals at each tokenizer state; reserved for future use by
    /// the suffix-pruning optimisation.
    #[allow(dead_code)]
    pub non_greedy_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalId>>,
    /// Terminals still reachable on a non-empty continuation at each tokenizer
    /// state; reserved for the suffix-pruning optimisation.
    #[allow(dead_code)]
    pub possible_future_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalId>>,
}


/// Build a terminal DWA NWA by iterating over tokens directly.
///
/// For each (TSID, token), runs the tokenizer and discovers all terminal
/// match chains. Continuation NWA states are keyed by the hash of the
/// remaining byte suffix (equivalent to trie-node sharing) to avoid
/// determinization-explosion from over-shared states.
fn build_terminal_dwa_nwa(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    used_terminals: &BTreeSet<TerminalId>,
) -> TerminalDwa {
    use rustc_hash::FxHashMap;
    use std::hash::{Hash, Hasher};

    let num_tsids = vocab_pre.num_tsids;
    let max_token = vocab_pre.max_token;
    let mut nwa = Nwa::new(num_tsids, max_token);

    // Leaf state.
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all(nwa.max_position(), num_tsids));

    // Root states: one per TSID.
    let mut tsid_roots: Vec<u32> = Vec::with_capacity(num_tsids as usize);
    for _tsid in 0..num_tsids {
        let root = nwa.add_state();
        tsid_roots.push(root);
        nwa.start_states.push(root);
    }

    // Continuation NWA states keyed by the hash of the remaining byte suffix.
    // Two tokens with identical remaining bytes share the same state.
    let mut cont_states: FxHashMap<u64, u32> = FxHashMap::default();
    let hash_suffix = |bytes: &[u8]| -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        h.finish()
    };

    // Pending transitions: (src_nwa, terminal, dst_nwa) → Vec<(tsid, token_id)>.
    // FxHashMap for faster hashing of small integer tuples.
    let mut pending: FxHashMap<(u32, i32, u32), Vec<(u32, u32)>> = FxHashMap::default();

    let max_token_len = vocab.entries.iter().map(|(_, b)| b.len()).max().unwrap_or(0);

    // Pre-compute which terminals are reachable from each tokenizer state.
    // This lets us skip entire TSIDs whose start state can never produce a
    // used terminal match.
    let reachable_terminals = tokenizer.compute_reachable_terminals();

    // Fast O(1) lookup for used terminals (replaces BTreeSet::contains in hot loop).
    let max_terminal_id = used_terminals.iter().copied().max().unwrap_or(0) as usize;
    let mut used_terminal_flags: Vec<bool> = vec![false; max_terminal_id + 1];
    for &t in used_terminals {
        used_terminal_flags[t as usize] = true;
    }

    // Pre-compute which DFA states have at least one used-terminal finalizer.
    // This lets execute_all_matches_cb_filtered skip callbacks entirely for
    // states that only match unused terminals (~99.7% of callbacks for
    // object_simple schema).
    let state_has_used: Vec<bool> = (0..tokenizer.num_states())
        .map(|s| {
            tokenizer.dfa.finalizers(s)
                .iter()
                .any(|&gid| gid < used_terminal_flags.len() && used_terminal_flags[gid])
        })
        .collect();

    eprintln!("[terminal_dwa] START: {} TSIDs, {} tokens, max_token_len={}",
        num_tsids, vocab.entries.len(), max_token_len);

    let t_start = std::time::Instant::now();
    let mut total_depth0_calls: u64 = 0;
    let mut total_depth0_skipped: u64 = 0;
    let mut total_cache_hits: u64 = 0;
    let mut total_matches: u64 = 0;
    let mut total_used_matches: u64 = 0;
    let mut total_callbacks: u64 = 0;
    let mut skipped_tsids: u32 = 0;

    // Cache for continuation execute_all_matches results (depth > 0).
    // Key: hash of remaining byte suffix. Value: matches from initial_state.
    let mut cont_cache: FxHashMap<u64, Vec<(usize, BTreeSet<TerminalId>)>> = FxHashMap::default();

    // TSID-first loop with continuation caching.
    for (tsid_idx, &tok_start_state) in vocab_pre.tsid_to_state.iter().enumerate() {
        let tsid = tsid_idx as u32;
        let src_root = tsid_roots[tsid_idx];

        // Skip TSIDs whose tokenizer start state can never reach any used terminal.
        let state_reachable = &reachable_terminals[tok_start_state as usize];
        if !state_reachable.iter().any(|t| used_terminals.contains(t)) {
            skipped_tsids += 1;
            continue;
        }

        let t_tsid = std::time::Instant::now();
        for &(token_id, ref token_bytes) in &vocab.entries {
            if token_bytes.is_empty() {
                continue;
            }

            // First-byte pre-filter: skip tokens whose first byte immediately
            // leads to the DEAD state from this TSID's tokenizer start state.
            // This avoids the overhead of setting up the DFA walk callback for
            // tokens that can never produce any terminal match.
            if tokenizer.dfa.get_transition(tok_start_state, token_bytes[0])
                == crate::automata::dfa::DEAD
            {
                total_depth0_skipped += 1;
                continue;
            }

            // Stack: (byte_offset, start_state_for_execute, src_nwa_state, depth)
            let mut stack: Vec<(usize, u32, u32, usize)> = Vec::with_capacity(4);
            stack.push((0, tok_start_state, src_root, 0));

            // Dedup: prevent re-processing the same (offset, depth).
            let mut visited: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
            while let Some((byte_offset, start_state, src_state, depth)) = stack.pop() {
                if depth > max_token_len || byte_offset >= token_bytes.len() {
                    continue;
                }

                let remaining = &token_bytes[byte_offset..];

                if depth == 0 {
                    // Zero-allocation depth-0: use filtered callback-based execute.
                    // Only fires callback at DFA states with used-terminal finalizers.
                    total_depth0_calls += 1;
                    tokenizer.execute_all_matches_cb_filtered(remaining, start_state, &state_has_used, |match_offset, finalizers| {
                        let abs_offset = byte_offset + match_offset;
                        let is_leaf = abs_offset == token_bytes.len();
                        total_callbacks += 1;
                        total_matches += finalizers.len() as u64;
                        for &gid in finalizers {
                            let terminal = gid as TerminalId;
                            if gid >= used_terminal_flags.len() || !used_terminal_flags[gid] {
                                continue;
                            }
                            total_used_matches += 1;
                            if is_leaf {
                                pending
                                    .entry((src_state, terminal as i32, leaf_state))
                                    .or_default()
                                    .push((tsid, token_id));
                            } else {
                                let suffix = &token_bytes[abs_offset..];
                                let sh = hash_suffix(suffix);
                                let cont = *cont_states
                                    .entry(sh)
                                    .or_insert_with(|| nwa.add_state());
                                pending
                                    .entry((src_state, terminal as i32, cont))
                                    .or_default()
                                    .push((tsid, token_id));
                                if visited.insert((abs_offset, depth + 1)) {
                                    stack.push((
                                        abs_offset,
                                        tokenizer.initial_state(),
                                        cont,
                                        depth + 1,
                                    ));
                                }
                            }
                        }
                    });
                } else {
                    // Depth > 0: use cached results (all start from initial_state).
                    let sh = hash_suffix(remaining);
                    if !cont_cache.contains_key(&sh) {
                        let result = tokenizer.execute_all_matches(remaining, tokenizer.initial_state());
                        cont_cache.insert(sh, result.matches);
                    } else {
                        total_cache_hits += 1;
                    }
                    let matches = cont_cache.get(&hash_suffix(remaining)).unwrap();

                    for &(match_offset, ref matched_terminals) in matches.iter() {
                        let abs_offset = byte_offset + match_offset;
                        let is_leaf = abs_offset == token_bytes.len();

                        total_matches += matched_terminals.len() as u64;
                        for &terminal in matched_terminals {
                            if (terminal as usize) >= used_terminal_flags.len() || !used_terminal_flags[terminal as usize] {
                                continue;
                            }
                            if is_leaf {
                                pending
                                    .entry((src_state, terminal as i32, leaf_state))
                                    .or_default()
                                    .push((tsid, token_id));
                            } else {
                                let suffix = &token_bytes[abs_offset..];
                                let sh = hash_suffix(suffix);
                                let cont = *cont_states
                                    .entry(sh)
                                    .or_insert_with(|| nwa.add_state());
                                pending
                                    .entry((src_state, terminal as i32, cont))
                                    .or_default()
                                    .push((tsid, token_id));
                                if visited.insert((abs_offset, depth + 1)) {
                                    stack.push((
                                        abs_offset,
                                        tokenizer.initial_state(),
                                        cont,
                                        depth + 1,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        if tsid_idx % 10 == 0 || tsid_idx == num_tsids as usize - 1 {
            eprintln!("[terminal_dwa] TSID {}/{}: {:.3}s (pending={}, cont_states={}, cache_hits={})",
                tsid_idx + 1, num_tsids, t_tsid.elapsed().as_secs_f64(), pending.len(), cont_states.len(), total_cache_hits);
        }
    }

    let t_iter = t_start.elapsed();
    eprintln!("[terminal_dwa] token iteration: {:.3}s (depth0={}, skipped_1st_byte={}, callbacks={}, matches={}, used_matches={}, cache_hits={}, pending={}, cont_states={}, skipped_tsids={}/{})",
        t_iter.as_secs_f64(), total_depth0_calls, total_depth0_skipped, total_callbacks, total_matches, total_used_matches, total_cache_hits, pending.len(), cont_states.len(), skipped_tsids, num_tsids);

    let t2 = std::time::Instant::now();
    // Convert pending into NWA transitions with proper weights.
    // Group by (src, label) → Vec<(dst, Weight)> to avoid O(N²) linear scans.
    let pending_count = pending.len();
    let mut by_src_label: FxHashMap<(u32, i32), Vec<(u32, Weight)>> = FxHashMap::default();
    let mut weight_progress = 0usize;

    for ((src, label, dst), mut pairs) in pending {
        weight_progress += 1;
        if weight_progress % 500_000 == 0 || weight_progress == pending_count {
            eprintln!("[terminal_dwa] weight progress: {}/{} ({:.1}s)",
                weight_progress, pending_count, t2.elapsed().as_secs_f64());
        }
        // Sort by (tsid, token_id) and dedup, then batch-build RangeSets.
        pairs.sort_unstable();
        pairs.dedup();

        let mut entries: Vec<(u32, u32, RangeSet)> = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let tsid = pairs[i].0;
            let start = i;
            while i < pairs.len() && pairs[i].0 == tsid {
                i += 1;
            }
            // Build RangeSet from sorted token_ids without per-insert cloning.
            let rs = RangeSet::from_ranges(
                pairs[start..i].iter().map(|&(_, tid)| (tid, tid))
            );
            entries.push((tsid, tsid, rs));
        }
        let weight = Weight::from_entries(entries, num_tsids);

        by_src_label.entry((src, label)).or_default().push((dst, weight));
    }

    // Bulk-insert transitions into NWA (avoids linear scanning per insertion).
    for ((src, label), targets) in by_src_label {
        let nwa_targets = nwa.states[src as usize].transitions.entry(label).or_default();
        nwa_targets.extend(targets);
    }

    let t_weights = t2.elapsed();
    eprintln!("[terminal_dwa] weight construction: {:.3}s (nwa_states={})",
        t_weights.as_secs_f64(), nwa.num_states());

    TerminalDwa {
        nwa,
        tsid_roots,
        non_greedy_terminals_by_tokenizer_state: (0..tokenizer.num_states())
            .map(|state| tokenizer.matched_non_greedy_terminals(state))
            .collect(),
        possible_future_terminals_by_tokenizer_state: (0..tokenizer.num_states())
            .map(|state| tokenizer.possible_future_terminals(state))
            .collect(),
    }
}

fn compute_ever_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    let mut ever_allowed: Vec<BTreeSet<TerminalId>> =
        vec![BTreeSet::new(); grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }

            let suffix = &rule.rhs[index + 1..];
            let mut allowed = grammar.first_of_seq(suffix);
            allowed.remove(&EOF);
            if suffix.is_empty() || grammar.seq_is_nullable(suffix) {
                allowed.extend(
                    grammar.follow[rule.lhs as usize]
                        .iter()
                        .copied()
                        .filter(|follow| *follow != EOF && *follow < grammar.num_terminals),
                );
            }
            ever_allowed[*terminal as usize].extend(
                allowed
                    .into_iter()
                    .filter(|follow| *follow < grammar.num_terminals),
            );
        }
    }

    ever_allowed
        .into_iter()
        .map(|allowed| allowed.into_iter().collect())
        .collect()
}

fn compute_always_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    let mut always_allowed: Vec<Option<BTreeSet<TerminalId>>> =
        vec![None; grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }

            let suffix = &rule.rhs[index + 1..];
            let mut allowed = grammar.first_of_seq(suffix);
            allowed.remove(&EOF);
            if suffix.is_empty() || grammar.seq_is_nullable(suffix) {
                allowed.extend(
                    grammar.follow[rule.lhs as usize]
                        .iter()
                        .copied()
                        .filter(|follow| *follow != EOF && *follow < grammar.num_terminals),
                );
            }
            let allowed: BTreeSet<TerminalId> = allowed
                .into_iter()
                .filter(|follow| *follow < grammar.num_terminals)
                .collect();
            match &mut always_allowed[*terminal as usize] {
                None => always_allowed[*terminal as usize] = Some(allowed),
                Some(existing) => existing.retain(|follow| allowed.contains(follow)),
            }
        }
    }

    always_allowed
        .into_iter()
        .map(|allowed| allowed.unwrap_or_default().into_iter().collect())
        .collect()
}

fn collapse_always_allowed(
    nwa: &mut Nwa,
    always_allowed_by_label: &[Vec<TerminalId>],
    terminals_count: usize,
) -> bool {
    if always_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let num_states = nwa.states.len();

    // ---------------------------------------------------------------
    // Phase 1: BFS to compute reachability + incoming labels per state.
    // No Weight operations needed — just set operations on labels.
    // ---------------------------------------------------------------
    let mut reachable = vec![false; num_states];
    let mut incoming: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); num_states];
    let all_labels: Vec<i32> = (0..terminals_count as i32).collect();

    let mut queue = std::collections::VecDeque::new();
    for &start in &nwa.start_states {
        if !reachable[start as usize] {
            reachable[start as usize] = true;
            incoming[start as usize].extend(all_labels.iter().copied());
            queue.push_back(start);
        }
    }

    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states[state_id as usize];
        let incoming_labels: Vec<i32> = incoming[state_id as usize].iter().copied().collect();

        for (dest, _) in &state.epsilons {
            let dest_idx = *dest as usize;
            let was_reachable = reachable[dest_idx];
            reachable[dest_idx] = true;
            let old_len = incoming[dest_idx].len();
            incoming[dest_idx].extend(incoming_labels.iter().copied());
            if !was_reachable || incoming[dest_idx].len() != old_len {
                queue.push_back(*dest);
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            for (dest, _) in targets {
                let dest_idx = *dest as usize;
                let was_reachable = reachable[dest_idx];
                reachable[dest_idx] = true;
                let added = incoming[dest_idx].insert(label);
                if !was_reachable || added {
                    queue.push_back(*dest);
                }
            }
        }
    }

    // States with incoming epsilon transitions get all labels (conservative).
    for state_id in 0..num_states {
        if !nwa.states[state_id].epsilons.is_empty() {
            incoming[state_id].extend(all_labels.iter().copied());
        }
    }

    // ---------------------------------------------------------------
    // Phase 2: Compute allowed-by-state from incoming labels.
    // ---------------------------------------------------------------
    let mut allowed_by_state: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); num_states];
    for state_id in 0..num_states {
        if !reachable[state_id] {
            continue;
        }
        let mut labels = incoming[state_id].iter();
        let Some(&first_label) = labels.next() else {
            continue;
        };
        if first_label < 0 || first_label as usize >= always_allowed_by_label.len() {
            continue;
        }
        let mut allowed: BTreeSet<i32> = always_allowed_by_label[first_label as usize]
            .iter()
            .map(|label| *label as i32)
            .collect();
        for &label in labels {
            if label < 0 || label as usize >= always_allowed_by_label.len() {
                continue;
            }
            let follows: BTreeSet<i32> = always_allowed_by_label[label as usize]
                .iter()
                .map(|follow| *follow as i32)
                .collect();
            allowed.retain(|candidate| follows.contains(candidate));
            if allowed.is_empty() {
                break;
            }
        }
        allowed_by_state[state_id] = allowed;
    }

    // ---------------------------------------------------------------
    // Phase 3: Collapse transitions.
    // Conservative check: weight ⊆ final_weight (no domain intersection).
    // This is stricter than the full check but avoids all Weight propagation.
    // ---------------------------------------------------------------
    let final_weights: Vec<Option<Weight>> = nwa.states.iter().map(|state| state.final_weight.clone()).collect();
    let mut changed = false;
    for state_id in 0..num_states {
        let allowed = &allowed_by_state[state_id];
        if allowed.is_empty() || !reachable[state_id] {
            continue;
        }

        let mut labels_to_remove = Vec::new();
        let state = &mut nwa.states[state_id];
        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 || label as usize >= terminals_count || !allowed.contains(&label) {
                continue;
            }

            let mut retained = Vec::new();
            for (dest, weight) in targets.iter() {
                let Some(final_weight) = final_weights[*dest as usize].as_ref() else {
                    retained.push((*dest, weight.clone()));
                    continue;
                };
                // Conservative: check weight ⊆ final_weight directly.
                // Since domain ∩ weight ⊆ weight, if weight ⊆ final_weight then
                // domain ∩ weight ⊆ final_weight certainly holds.
                if weight.is_subset(final_weight) {
                    let collapsed = final_weight.intersection(weight);
                    let updated = state
                        .final_weight
                        .clone()
                        .unwrap_or_else(|| Weight::empty(nwa.num_tsids))
                        .union(&collapsed);
                    state.final_weight = if updated.is_empty() { None } else { Some(updated) };
                    changed = true;
                } else {
                    retained.push((*dest, weight.clone()));
                }
            }

            if retained.is_empty() {
                labels_to_remove.push(label);
            } else {
                *targets = retained;
            }
        }
        for label in labels_to_remove {
            state.transitions.remove(&label);
        }
    }

    changed
}

fn prune_disallowed_follows(
    nwa: &mut Nwa,
    ever_allowed_by_label: &[Vec<TerminalId>],
    terminals_count: usize,
) -> bool {
    if ever_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let words_needed = (terminals_count + 63) / 64;
    let new_bitset = || vec![0u64; words_needed];
    let set_bit = |bs: &mut [u64], idx: usize| {
        if idx < terminals_count {
            bs[idx / 64] |= 1u64 << (idx % 64);
        }
    };
    let test_bit = |bs: &[u64], idx: usize| -> bool {
        idx < terminals_count && (bs[idx / 64] & (1u64 << (idx % 64))) != 0
    };
    let is_empty = |bs: &[u64]| -> bool { bs.iter().all(|&word| word == 0) };
    let union_into = |dst: &mut [u64], src: &[u64]| {
        for (lhs, rhs) in dst.iter_mut().zip(src) {
            *lhs |= *rhs;
        }
    };
    let intersect_into = |dst: &mut [u64], src: &[u64]| {
        for (lhs, rhs) in dst.iter_mut().zip(src) {
            *lhs &= *rhs;
        }
    };

    let mut all_terminals = new_bitset();
    for idx in 0..terminals_count {
        set_bit(&mut all_terminals, idx);
    }
    let disallowed_after: Vec<Vec<u64>> = (0..terminals_count)
        .map(|idx| {
            if idx >= ever_allowed_by_label.len() {
                return new_bitset();
            }
            let mut bitset = all_terminals.clone();
            for &allowed in &ever_allowed_by_label[idx] {
                let allowed = allowed as usize;
                if allowed < terminals_count {
                    bitset[allowed / 64] &= !(1u64 << (allowed % 64));
                }
            }
            bitset
        })
        .collect();

    let mut in_degree = vec![0u32; nwa.states.len()];
    for state in &nwa.states {
        for (dest, _) in &state.epsilons {
            in_degree[*dest as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dest, _) in targets {
                in_degree[*dest as usize] += 1;
            }
        }
    }

    let mut topo_queue = std::collections::VecDeque::new();
    for (sid, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            topo_queue.push_back(sid as u32);
        }
    }
    let mut topo_order = Vec::with_capacity(nwa.states.len());
    let mut disallowed_union: Vec<Option<Vec<u64>>> = vec![None; nwa.states.len()];
    for &start in &nwa.start_states {
        disallowed_union[start as usize] = Some(new_bitset());
    }

    while let Some(sid) = topo_queue.pop_front() {
        topo_order.push(sid);
        let src_disallowed = disallowed_union[sid as usize]
            .clone()
            .unwrap_or_else(new_bitset);
        let state = &nwa.states[sid as usize];

        for (dest, _) in &state.epsilons {
            let dest_set = disallowed_union[*dest as usize].get_or_insert_with(new_bitset);
            union_into(dest_set, &src_disallowed);
        }
        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            let label_disallowed = &disallowed_after[label as usize];
            for (dest, _) in targets {
                let dest_set = disallowed_union[*dest as usize].get_or_insert_with(new_bitset);
                union_into(dest_set, label_disallowed);
            }
        }

        for (dest, _) in &state.epsilons {
            in_degree[*dest as usize] -= 1;
            if in_degree[*dest as usize] == 0 {
                topo_queue.push_back(*dest);
            }
        }
        for targets in state.transitions.values() {
            for (dest, _) in targets {
                in_degree[*dest as usize] -= 1;
                if in_degree[*dest as usize] == 0 {
                    topo_queue.push_back(*dest);
                }
            }
        }
    }

    let mut disallowed_intersection: Vec<Option<Vec<u64>>> = vec![None; nwa.states.len()];
    for &start in &nwa.start_states {
        disallowed_intersection[start as usize] = Some(new_bitset());
    }
    for &sid in &topo_order {
        let src_disallowed = disallowed_intersection[sid as usize]
            .clone()
            .unwrap_or_else(new_bitset);
        let state = &nwa.states[sid as usize];

        for (dest, _) in &state.epsilons {
            match &mut disallowed_intersection[*dest as usize] {
                None => disallowed_intersection[*dest as usize] = Some(src_disallowed.clone()),
                Some(existing) => intersect_into(existing, &src_disallowed),
            }
        }
        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            let label_disallowed = &disallowed_after[label as usize];
            for (dest, _) in targets {
                match &mut disallowed_intersection[*dest as usize] {
                    None => disallowed_intersection[*dest as usize] = Some(label_disallowed.clone()),
                    Some(existing) => intersect_into(existing, label_disallowed),
                }
            }
        }
    }

    let mut changed = false;
    for sid in 0..nwa.states.len() {
        let Some(state_disallowed) = &disallowed_intersection[sid] else {
            continue;
        };
        if is_empty(state_disallowed) {
            continue;
        }

        let labels_to_remove: Vec<i32> = nwa.states[sid]
            .transitions
            .keys()
            .copied()
            .filter(|label| *label >= 0 && (*label as usize) < terminals_count && test_bit(state_disallowed, *label as usize))
            .collect();
        if labels_to_remove.is_empty() {
            continue;
        }
        changed = true;
        for label in labels_to_remove {
            nwa.states[sid].transitions.remove(&label);
        }
    }

    changed
}

/// Build the singular terminal-side compilation object from an actual
/// tokenizer/vocabulary walk.
pub(crate) fn build_terminal_dwa(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
) -> TerminalDwa {
    let (dwa, _) = build_terminal_dwa_impl(tokenizer, vocab, vocab_pre, grammar, used_terminals, false);
    dwa
}

/// Build the terminal DWA, returning [`TerminalDebug`] alongside.
pub(crate) fn build_terminal_dwa_with_debug(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
) -> (TerminalDwa, crate::compiler::debug::TerminalDebug) {
    let (dwa, dbg) = build_terminal_dwa_impl(tokenizer, vocab, vocab_pre, grammar, used_terminals, true);
    (dwa, dbg.expect("debug=true must produce Some"))
}

fn build_terminal_dwa_impl(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
    used_terminals: &BTreeSet<TerminalId>,
    capture_debug: bool,
) -> (TerminalDwa, Option<crate::compiler::debug::TerminalDebug>) {
    let mut terminal_dwa = build_terminal_dwa_nwa(tokenizer, vocab, vocab_pre, used_terminals);

    let nwa_after_build = if capture_debug { Some(terminal_dwa.nwa.clone()) } else { None };

    let t_post = std::time::Instant::now();
    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let t_collapse = std::time::Instant::now();
    let _ = collapse_always_allowed(
        &mut terminal_dwa.nwa,
        &always_allowed_by_label,
        grammar.num_terminals as usize,
    );
    let t_collapse_done = t_collapse.elapsed();

    let nwa_after_collapse = if capture_debug { Some(terminal_dwa.nwa.clone()) } else { None };

    let ever_allowed_by_label = compute_ever_allowed_follows(grammar);
    let t_prune = std::time::Instant::now();
    let _ = prune_disallowed_follows(
        &mut terminal_dwa.nwa,
        &ever_allowed_by_label,
        grammar.num_terminals as usize,
    );
    let t_prune_done = t_prune.elapsed();
    eprintln!("[terminal_dwa] follow-path: {:.3}s (collapse={:.3}s, prune={:.3}s)",
        t_post.elapsed().as_secs_f64(), t_collapse_done.as_secs_f64(), t_prune_done.as_secs_f64());

    let debug = if capture_debug {
        Some(crate::compiler::debug::TerminalDebug {
            nwa_after_build: nwa_after_build.unwrap(),
            nwa_after_collapse: nwa_after_collapse.unwrap(),
        })
    } else {
        None
    };

    (terminal_dwa, debug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::bytes;
    use crate::compiler::grammar_def::tests::simple_ab_grammar;
    use crate::compiler::glr::grammar::GlrGrammar;
    use crate::compiler::tokenizer_dfa::TokenizerDfa;
    use crate::compiler::vocab_pre::VocabPreprocessing;

    #[test]
    fn test_build_terminal_dwa_collapses_always_allowed_follow_path() {
        let grammar = simple_ab_grammar();
        let glr_grammar = GlrGrammar::from_grammar_def(&grammar);
        let tokenizer = TokenizerDfa::from_grammar_def(&grammar);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let vocab_pre = VocabPreprocessing::compute(&tokenizer, &vocab, None);

        let all_terminals: BTreeSet<TerminalId> = (0..glr_grammar.num_terminals).collect();
        let terminal_dwa = build_terminal_dwa(&tokenizer, &vocab, &vocab_pre, &glr_grammar, &all_terminals);
        let initial_tsid = vocab_pre.state_to_tsid[tokenizer.initial_state() as usize] as usize;
        let root = terminal_dwa.tsid_roots[initial_tsid];
        let a_targets = &terminal_dwa.nwa.states[root as usize].transitions[&0];
        assert!(!a_targets.is_empty());

        let mut combined_a = Weight::empty(vocab_pre.num_tsids);
        for (_, weight) in a_targets {
            combined_a = combined_a.union(weight);
        }
        assert_eq!(combined_a.tokens_for_tsid(initial_tsid as u32), RangeSet::from_range(0, 1));

        for (dest, weight) in a_targets {
            let state = &terminal_dwa.nwa.states[*dest as usize];
            assert!(state.final_weight.is_some());
            assert!(!state.transitions.contains_key(&1));
            if !state.transitions.is_empty() {
                assert_eq!(weight.tokens_for_tsid(initial_tsid as u32), RangeSet::from_range(1, 1));
            }
        }
    }

    #[test]
    fn test_terminal_dwa_carries_tokenizer_greedy_metadata() {
        let grammar = simple_ab_grammar();
        let glr_grammar = GlrGrammar::from_grammar_def(&grammar);
        let tokenizer = TokenizerDfa::from_expr_groups(&[
            crate::automata::regex::ExprGroup {
                expr: bytes(b"a"),
                is_non_greedy: true,
            },
            crate::automata::regex::ExprGroup {
                expr: bytes(b"ab"),
                is_non_greedy: false,
            },
        ]);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let vocab_pre = VocabPreprocessing::compute(&tokenizer, &vocab, None);

        let all_terminals: BTreeSet<TerminalId> = (0..glr_grammar.num_terminals).collect();
        let terminal_dwa = build_terminal_dwa(&tokenizer, &vocab, &vocab_pre, &glr_grammar, &all_terminals);
        let state_after_a = tokenizer.run(b"a") as usize;

        assert!(terminal_dwa.non_greedy_terminals_by_tokenizer_state[state_after_a].contains(&0));
        assert!(terminal_dwa.possible_future_terminals_by_tokenizer_state[state_after_a].contains(&1));
    }
}
