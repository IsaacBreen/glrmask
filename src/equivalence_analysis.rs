use crate::finite_automata::{ExecutionResult, GroupID, Match, Regex};
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

fn should_terminate_early(
    possible_future_group_ids: &BTreeSet<GroupID>,
    non_greedy_finalizers: &BTreeSet<GroupID>,
    matched_groups: &BTreeSet<GroupID>,
) -> bool {
    possible_future_group_ids
        .iter()
        .all(|group_id| non_greedy_finalizers.contains(group_id) && matched_groups.contains(group_id))
}

/// Computes a deterministic hash representing the parsing structure of the string.
fn compute_signature(regex: &Regex, slice: &[u8], start_state: usize) -> u64 {
    // 1. Forward Pass: Build a graph of valid state transitions (nodes and edges).
    // We Map: Position -> (EndStateData, List of Outgoing Edges)
    let mut graph = HashMap::new();
    let mut queue = VecDeque::from([0]);
    let mut visited: HashSet<usize> = HashSet::from([0]);

    while let Some(pos) = queue.pop_front() {
        if pos > slice.len() {
            continue;
        }

        let exec_start = if pos == 0 {
            start_state
        } else {
            regex.dfa.start_state
        };
        let result = {
            let text = &slice[pos..];

            // --- INLINED execute_from_state_nonzero START ---
            // Inlined from Regex::init_to_state
            let mut current_state = exec_start;
            let mut done = regex.dfa.states[exec_start].transitions.is_empty();
            let mut matches: BTreeMap<GroupID, usize> = regex.dfa.states[exec_start]
                .finalizers
                .iter()
                .map(|group_id| (group_id, 0))
                .collect();
            let mut position = 0;

            // Inlined from RegexState::execute
            if done {
                position += text.len();
            } else {
                let dfa = &regex.dfa;
                let mut local_position = 0;
                while local_position < text.len() {
                    let state_data = &dfa.states[current_state];
                    let next_u8 = text[local_position];
                    if let Some(&next_state) = state_data.transitions.get(next_u8) {
                        current_state = next_state;
                        local_position += 1;
                        for group_id in &dfa.states[current_state].finalizers {
                            if dfa.non_greedy_finalizers.contains(&group_id) {
                                matches
                                    .entry(group_id)
                                    .or_insert(position + local_position);
                            } else {
                                matches.insert(group_id, position + local_position);
                            }
                        }

                        let matched: BTreeSet<GroupID> = matches.keys().cloned().collect();
                        let should_terminate = should_terminate_early(
                            &dfa.states[current_state].possible_future_group_ids,
                            &dfa.non_greedy_finalizers,
                            &matched,
                        );

                        if should_terminate {
                            position += text.len();
                            done = true;
                            break;
                        }
                    } else {
                        position += text.len();
                        done = true;
                        break;
                    }
                }
                if !done {
                    position += text.len();
                    if dfa.states[current_state].transitions.is_empty() {
                        done = true;
                    }
                }
            }

            let result_matches: Vec<_> = matches
                .iter()
                .map(|(&id, &width)| Match {
                    group_id: id,
                    position: width,
                })
                .filter(|token| token.position != 0)
                .collect();

            let result_end_state = if done { None } else { Some(current_state) };

            ExecutionResult {
                matches: result_matches,
                end_state: result_end_state,
            }
            // --- INLINED execute_from_state_nonzero END ---
        };

        let mut edges = Vec::with_capacity(result.matches.len());
        for m in result.matches {
            let target = pos + m.position;
            edges.push((m.group_id, target));

            if visited.insert(target) {
                queue.push_back(target);
            }
        }

        // Sort edges by Group ID so the hash is consistent regardless of execution order
        edges.sort_unstable_by_key(|e| e.0);

        let completion = result
            .end_state
            .map(|id| &regex.dfa.states[id].possible_future_group_ids);
        graph.insert(pos, (completion, edges));
    }

    // 2. Backward Pass: Calculate hashes from the end of the string back to the start.
    // We sort positions descending to ensure we hash a target node before the node pointing to it.
    let mut positions: Vec<_> = graph.keys().copied().collect();
    positions.sort_unstable_by(|a, b| b.cmp(a));

    let mut node_hashes = HashMap::with_capacity(graph.len());

    for pos in positions {
        let (completion, edges) = &graph[&pos];
        let mut hasher = DefaultHasher::new();

        // Hash the local state (completion data)
        completion.hash(&mut hasher);

        // Hash the structural connections (outgoing edges + hash of target nodes)
        for (group_id, target) in edges {
            let target_hash = node_hashes
                .get(target)
                .expect("Target must be processed before Source");
            (group_id, target_hash).hash(&mut hasher);
        }

        node_hashes.insert(pos, hasher.finish());
    }

    // The signature is the hash of the root node (position 0)
    node_hashes.get(&0).copied().unwrap_or(0)
}

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    // Compute a unique signature for every string in parallel
    let signatures: Vec<u64> = strings
        .par_iter()
        .map(|s| {
            let mut hasher = DefaultHasher::new();
            // Combine signatures for all requested start states into one final hash
            for &state in initial_states {
                compute_signature(regex, s, state).hash(&mut hasher);
            }
            hasher.finish()
        })
        .collect();

    // Group string indices by their computed signature
    let mut groups = HashMap::new();
    for (index, sig) in signatures.into_iter().enumerate() {
        groups.entry(sig).or_insert_with(Vec::new).push(index);
    }

    groups.into_values().collect()
}
