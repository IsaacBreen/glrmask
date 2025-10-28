use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::EdgeKey;
use crate::trie3_opt::passes::OptimizationPass;

/// Aggressive, state-precise factoring across all pops using token atoms and per-signature intermediates.
///
/// For each node u:
///   - For every state s, build the family of token sets { tokens_s[p][v] } where tokens_s[p][v]
///     is the union of all tokens that send s to destination v under pop p.
///   - Build token atoms across the entire family for s so that tokens in the same atom are
///     indistinguishable with respect to membership in every tokens_s[p][v].
///   - For each atom A and state s, compute its signature: for every pop p, the set of reachable
///     destinations D_s,A(p) = { v | tokens_s[p][v] ∩ A ≠ ∅ }.
///   - Group atoms for each (s) by identical signatures; union their tokens.
///     * If for a signature, every pop has at most one destination, keep direct edges:
///        u --(p, tokens_union)-> that single destination, sids contain the states for which the group applies.
///     * Otherwise, introduce/reuse a single intermediate I_sig:
///        u --(0, tokens_union)-> I_sig with sids being the states; and
///        I_sig --(p, tokens_union_over_all_states_for_sig)-> v with sids = all_states for each (p, v) in the signature.
///
/// This reduces per-state fanout at u to at most the number of distinct signatures (often 1), and
/// it operates across all pops (including 0), which is critical for reducing root_state_fanout.
pub struct FactorStateFanoutPass;

impl OptimizationPass for FactorStateFanoutPass {
    fn name(&self) -> &'static str {
        "FactorStateFanout"
    }

    fn run(&self, trie: &mut MiniTrie, _ctx: &mut OptimizationContext) {
        // Cap of token atoms per state to avoid pathological explosions.
        const MAX_ATOMS_PER_STATE: usize = 4096;

        #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        struct Sig {
            // Canonical signature: sorted by pop, and each entry has a sorted dest list.
            entries: Vec<(isize, Vec<NodeId>)>,
        }

        // Helper: build atoms from a family of sets with a hard cap; fallback to the universe.
        fn build_atoms(family: &[SortedSet], universe: &SortedSet, cap: usize) -> Vec<SortedSet> {
            if universe.is_empty() {
                return Vec::new();
            }
            let mut blocks = vec![universe.clone()];
            let mut aborted = false;
            for f in family {
                if f.is_empty() {
                    continue;
                }
                let mut next = Vec::with_capacity(blocks.len().saturating_mul(2));
                let mut any_split = false;
                for b in &blocks {
                    if !b.intersects(f) {
                        next.push(b.clone());
                        continue;
                    }
                    let inter = b.intersect(f);
                    if !inter.is_empty() {
                        next.push(inter);
                    }
                    let diff = b.difference(f);
                    if !diff.is_empty() {
                        next.push(diff);
                    }
                    any_split = true;
                }
                if cap > 0 && next.len() > cap {
                    aborted = true;
                    break;
                }
                if any_split {
                    blocks = next;
                }
            }
            if aborted {
                vec![universe.clone()]
            } else {
                blocks
            }
        }

        let node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in node_ids {
            let node = if let Some(n) = trie.get_node(node_id) { n } else { continue };
            if node.children().is_empty() {
                continue;
            }

            // tokens_s: state -> pop -> dest -> tokens
            let mut tokens_s: HashMap<usize, BTreeMap<isize, BTreeMap<NodeId, SortedSet>>> =
                HashMap::new();
            for (ek, dm) in node.children() {
                for (dst, sids) in dm {
                    for s in sids.iter() {
                        tokens_s
                            .entry(s)
                            .or_default()
                            .entry(ek.pop)
                            .or_default()
                            .entry(*dst)
                            .or_default()
                            .union_inplace(&ek.tokens);
                    }
                }
            }

            if tokens_s.is_empty() {
                continue;
            }

            // Aggregators for rebuilding u's (node_id) children:
            let mut new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> = BTreeMap::new();

            // For intermediate reuse per signature at this node:
            let mut mid_for_sig: BTreeMap<Sig, NodeId> = BTreeMap::new();

            // For MI->dest edges tokens aggregation: sig -> pop -> dest -> tokens
            let mut sig_pop_dest_to_tokens: BTreeMap<Sig, BTreeMap<isize, BTreeMap<NodeId, SortedSet>>> =
                BTreeMap::new();

            // For u->MI edges aggregation per signature across states:
            // sig -> (state -> tokens). We'll invert to tokens -> states.
            let mut sig_to_state_tokens: BTreeMap<Sig, HashMap<usize, SortedSet>> = BTreeMap::new();

            // For direct edges (no intermediate needed): accumulate per (pop, dest):
            // pop -> dest -> (states -> tokens). We'll invert to tokens -> states.
            let mut direct_edges: BTreeMap<isize, BTreeMap<NodeId, HashMap<SortedSet, SortedSet>>> =
                BTreeMap::new();

            for (s, by_pop) in &tokens_s {
                // Build family and union for this state across all pops/dests.
                let mut family: Vec<SortedSet> = Vec::new();
                let mut universe = SortedSet::new();
                for (_p, dmap) in by_pop {
                    for (_dst, toks) in dmap {
                        if !toks.is_empty() {
                            family.push(toks.clone());
                            universe.union_inplace(toks);
                        }
                    }
                }
                if universe.is_empty() {
                    continue;
                }
                let atoms = build_atoms(&family, &universe, MAX_ATOMS_PER_STATE);

                // For each atom, compute signature and route either directly or via MI.
                for atom in atoms {
                    // Compute per-pop dest sets for this state and atom.
                    let mut per_pop_dests: BTreeMap<isize, Vec<NodeId>> = BTreeMap::new();
                    for (p, dmap) in by_pop {
                        let mut dests: Vec<NodeId> = Vec::new();
                        for (dst, toks) in dmap {
                            if toks.intersects(&atom) {
                                dests.push(*dst);
                            }
                        }
                        if !dests.is_empty() {
                            dests.sort_unstable();
                            dests.dedup();
                            per_pop_dests.insert(*p, dests);
                        }
                    }
                    if per_pop_dests.is_empty() {
                        continue;
                    }

                    let sig = Sig {
                        entries: per_pop_dests
                            .iter()
                            .map(|(p, v)| (*p, v.clone()))
                            .collect(),
                    };

                    // Decide whether we can keep direct edges (every pop has a single dest)
                    // or must route through an intermediate (any pop has multiple dests).
                    let all_single = per_pop_dests.values().all(|v| v.len() == 1);

                    if all_single {
                        // Keep direct edges: for each pop, add (pop, tokens)-> single dest with sids={s}.
                        for (p, vlist) in per_pop_dests {
                            let v = vlist[0];
                            // Accumulate by (pop, dest) grouping states by identical token sets.
                            let entry = direct_edges.entry(p).or_default().entry(v).or_default();
                            entry
                                .entry(atom.clone())
                                .or_insert_with(SortedSet::new)
                                .insert(*s);
                        }
                    } else {
                        // Route via a per-signature intermediate.
                        // Accumulate u->MI per (signature, state) tokens.
                        sig_to_state_tokens
                            .entry(sig.clone())
                            .or_default()
                            .entry(*s)
                            .or_insert_with(SortedSet::new)
                            .union_inplace(&atom);

                        // Accumulate MI->dest tokens per (signature, pop, dest).
                        let pop_map = sig_pop_dest_to_tokens
                            .entry(sig.clone())
                            .or_default();
                        for (p, vlist) in per_pop_dests {
                            let dm = pop_map.entry(p).or_default();
                            for v in vlist {
                                dm.entry(v)
                                    .or_insert_with(SortedSet::new)
                                    .union_inplace(&atom);
                            }
                        }
                    }
                }
            }

            // Emit direct edges (collapsed by tokens -> states).
            for (p, by_dest) in direct_edges {
                for (dst, tokens_to_states) in by_dest {
                    // Invert tokens_to_states: currently key = tokens, val = states (as a SortedSet of state IDs).
                    // It is already tokens -> states, so we can emit directly.
                    for (toks, states) in tokens_to_states {
                        if toks.is_empty() || states.is_empty() {
                            continue;
                        }
                        let key = EdgeKey::new(p, toks);
                        new_children
                            .entry(key)
                            .or_default()
                            .entry(dst)
                            .or_default()
                            .union_inplace(&states);
                    }
                }
            }

            // Emit intermediates and their edges.
            let all_states = SortedSet::from_iter(0..=_ctx.max_state_id);
            for (sig, state_to_tokens) in sig_to_state_tokens {
                // Create/reuse intermediate for this signature at this source node.
                let mid_id = *mid_for_sig
                    .entry(sig.clone())
                    .or_insert_with(|| trie.add_node(false));

                // Invert state->tokens to tokens->states to reduce number of u->MI edges.
                let mut tokens_to_states: BTreeMap<SortedSet, SortedSet> = BTreeMap::new();
                for (s, toks) in state_to_tokens {
                    if toks.is_empty() {
                        continue;
                    }
                    tokens_to_states
                        .entry(toks)
                        .or_insert_with(SortedSet::new)
                        .insert(s);
                }

                // u -> MI edges (pop=0), grouped by tokens.
                for (toks, states) in tokens_to_states {
                    if toks.is_empty() || states.is_empty() {
                        continue;
                    }
                    let key = EdgeKey::new(0, toks);
                    new_children
                        .entry(key)
                        .or_default()
                        .entry(mid_id)
                        .or_default()
                        .union_inplace(&states);
                }

                // MI -> dest edges (state-agnostic) aggregated by (pop, dest) with tokens union across all states/atoms.
                if let Some(by_pop_dest) = sig_pop_dest_to_tokens.get(&sig) {
                    for (p, vmap) in by_pop_dest {
                        for (v, toks) in vmap {
                            if !toks.is_empty() {
                                trie.add_edge(mid_id, EdgeKey::new(*p, toks.clone()), *v, all_states.clone());
                            }
                        }
                    }
                }
            }

            // Finalize this node's children.
            trie.set_children(node_id, new_children);
        }
    }
}

