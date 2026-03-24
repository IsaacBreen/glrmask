//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use rustc_hash::FxHashMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as unweighted_minimize;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_fast;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::Weight;

impl Templates {
    /// Assemble a weighted NWA for one bundle of (terminal, weight) entries.
    ///
    /// Pipeline: group-by-weight → unweighted DFA union → product determinize → minimize → NWA.
    ///
    /// The **determinize** step (product construction) is essential for correctness
    /// and performance — it fuses all group DFAs into a single DWA that the
    /// downstream parser composition relies on. Do not remove it.
    pub(crate) fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if terminal_weights.len() == 1 {
            let (&terminal, weight) = terminal_weights.iter().next().expect("single-entry bundle");
            if weight.is_empty() {
                let mut nwa = NWA::new(0, 0);
                let s = nwa.add_state();
                nwa.start_states.push(s);
                return nwa;
            }
            if let Some(template) = self.by_terminal.get(&terminal) {
                // The template DFA is already deterministic and minimal.
                // Construct weighted NWA directly — skip NWA→determinize→DWA→NWA.
                let mut nwa = NWA::new(0, 0);
                for _ in &template.states {
                    nwa.add_state();
                }
                nwa.start_states.push(template.start_state);
                for (state_id, state) in template.states.iter().enumerate() {
                    if state.is_accepting {
                        nwa.set_final_weight(state_id as u32, weight.clone());
                    }
                    for (&label, &target) in &state.transitions {
                        nwa.add_transition(state_id as u32, label, target, weight.clone());
                    }
                }
                return nwa;
            }
            let mut nwa = NWA::new(0, 0);
            let s = nwa.add_state();
            nwa.start_states.push(s);
            return nwa;
        }

        // Group entries by weight so we can merge templates that share weights
        // using fast unweighted DFA operations.
        let mut weight_groups: HashMap<&Weight, Vec<TerminalID>> = HashMap::new();
        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() {
                continue;
            }
            if self.by_terminal.contains_key(&terminal) {
                weight_groups.entry(weight).or_default().push(terminal);
            }
        }

        let num_groups = weight_groups.len();

        // Build a merged unweighted DFA for each weight group.
        let mut group_dfas: Vec<(&Weight, UnweightedDfa)> = Vec::with_capacity(num_groups);
        for (weight, terminals) in &weight_groups {
            if terminals.len() == 1 {
                // Single terminal in group — use template DFA directly.
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((weight, template.clone()));
                }
            } else {
                // Multiple terminals sharing a weight — union their DFAs via NFA.
                let merged = union_unweighted_dfas(
                    terminals.iter().filter_map(|t| self.by_terminal.get(t)),
                );
                group_dfas.push((weight, merged));
            }
        }

        // Specialized weighted determinize: product of unweighted group DFAs.
        let bundle_dwa = determinize_bundle_groups(&group_dfas);

        // Determinization is essential here; minimization is only worthwhile
        // for multi-group bundles, so keep it as a cheap best-effort pass.
        let minimized = if num_groups > 1 {
            minimize_fast(&bundle_dwa)
        } else {
            bundle_dwa
        };

        dwa_to_nwa(&minimized)
    }
}

/// Specialized weighted determinize for bundles.
///
/// Instead of running generic NWA determinize (which clones weights for every
/// transition), this builds the product of unweighted group DFAs and computes
/// weights from precomputed effective-weight tables. O(states × labels × groups)
/// with no Weight intersection operations in the inner loop.
fn determinize_bundle_groups(groups: &[(&Weight, UnweightedDfa)]) -> DWA {
    use crate::automata::weighted_u32::dwa::{DWA, DWAState};

    let n = groups.len();
    if n == 0 {
        return DWA::new(0, 0);
    }

    const DEAD: u32 = u32::MAX;

    // Precompute normalized weights.
    // w_i_norm = w_i ∪ complement(union_all)
    let union_all = Weight::union_all(groups.iter().map(|(w, _)| *w));
    let complement_all = union_all.complement();
    let norms: Vec<Weight> = groups
        .iter()
        .map(|(w, _)| (*w).union(&complement_all))
        .collect();

    // Cache of effective weights per alive-set.
    // Key: sorted list of alive group indices. Value: effective weight per group.
    let mut alive_cache: FxHashMap<Vec<usize>, Vec<Weight>> = FxHashMap::default();

    let compute_effective = |alive: &[usize], norms: &[Weight]| -> Vec<Weight> {
        let alive_union = Weight::union_all(alive.iter().map(|&i| &norms[i]));
        let complement_alive = alive_union.complement();
        (0..n)
            .map(|i| {
                if alive.contains(&i) {
                    norms[i].union(&complement_alive)
                } else {
                    Weight::empty()
                }
            })
            .collect()
    };

    // Product state: [group0_state, group1_state, ...] with DEAD for inactive.
    let start_key: Vec<u32> = groups.iter().map(|(_, dfa)| dfa.start_state).collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key.clone());

    let mut is_start = true;

    // Reusable buffers.
    let mut all_labels: BTreeSet<i32> = BTreeSet::new();
    let mut next_state: Vec<u32> = vec![DEAD; n];

    while let Some(product_state) = worklist.pop_front() {
        let dwa_state = state_map[&product_state];

        // Determine alive groups and get effective weights.
        let alive: Vec<usize> = (0..n)
            .filter(|&i| product_state[i] != DEAD)
            .collect();

        let effective: &Vec<Weight> = if is_start {
            // Start state: use raw group weights (before normalization).
            // We store temporarily and won't cache this.
            // This is only hit once so perf doesn't matter.
            &*alive_cache.entry(vec![usize::MAX]).or_insert_with(|| {
                groups.iter().map(|(w, _)| (*w).clone()).collect()
            })
        } else {
            alive_cache
                .entry(alive.clone())
                .or_insert_with(|| compute_effective(&alive, &norms))
        };

        // Final weight.
        let mut final_w = Weight::empty();
        for &i in &alive {
            if groups[i].1.states[product_state[i] as usize].is_accepting {
                final_w = final_w.union(&effective[i]);
            }
        }
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }

        // Collect all labels from alive groups.
        all_labels.clear();
        for &i in &alive {
            for &label in groups[i].1.states[product_state[i] as usize]
                .transitions
                .keys()
            {
                all_labels.insert(label);
            }
        }

        // Compute transitions.
        for &label in &all_labels {
            // Compute next product state.
            for i in 0..n {
                next_state[i] = if product_state[i] == DEAD {
                    DEAD
                } else if let Some(&target) = groups[i]
                    .1
                    .states[product_state[i] as usize]
                    .transitions
                    .get(&label)
                {
                    target
                } else {
                    DEAD
                };
            }

            // Edge weight = union of effective weights for groups active in target.
            let mut edge_w = Weight::empty();
            for i in 0..n {
                if next_state[i] != DEAD {
                    edge_w = edge_w.union(&effective[i]);
                }
            }
            if edge_w.is_empty() {
                continue;
            }

            let to_dwa = if let Some(&existing) = state_map.get(&*next_state) {
                existing
            } else {
                let key = next_state.clone();
                let new_id = dwa.add_state();
                state_map.insert(key.clone(), new_id);
                worklist.push_back(key);
                new_id
            };

            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
        }

        is_start = false;
    }

    dwa
}

/// Union multiple unweighted DFAs into one DFA via NFA union + determinize + minimize.
fn union_unweighted_dfas<'a>(dfas: impl Iterator<Item = &'a UnweightedDfa>) -> UnweightedDfa {
    let mut nfa = UnweightedNfa::new_empty();
    let shared_start = nfa.add_state();
    nfa.start_states.push(shared_start);

    for dfa in dfas {
        if dfa.states.is_empty() {
            continue;
        }
        let offset = nfa.states.len() as u32;
        for _ in &dfa.states {
            nfa.add_state();
        }
        // Epsilon from shared start to this DFA's start.
        nfa.add_epsilon(shared_start, offset + dfa.start_state);
        for (state_id, state) in dfa.states.iter().enumerate() {
            let from = offset + state_id as u32;
            if state.is_accepting {
                nfa.set_accepting(from);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(from, label, offset + target);
            }
        }
    }

    let det = unweighted_determinize(&nfa);
    unweighted_minimize(&det)
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    for _ in &dwa.states {
        nwa.add_state();
    }

    nwa.start_states.push(dwa.start_state);
    for (state_id, state) in dwa.states.iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.set_final_weight(state_id as u32, final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.add_transition(state_id as u32, label, *target, weight.clone());
        }
    }

    nwa
}
