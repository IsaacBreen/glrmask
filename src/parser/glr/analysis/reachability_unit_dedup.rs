fn remove_unreachable_rules(rules: &[Rule], start: NonterminalID) -> Vec<Rule> {
    // Build index: lhs → rule indices for O(1) lookup per NT.
    let mut rules_by_lhs = BTreeMap::<NonterminalID, Vec<usize>>::new();
    for (i, rule) in rules.iter().enumerate() {
        rules_by_lhs.entry(rule.lhs).or_default().push(i);
    }

    let mut reachable = BTreeSet::new();
    let mut worklist = vec![start];
    while let Some(nt) = worklist.pop() {
        if !reachable.insert(nt) {
            continue;
        }
        if let Some(indexes) = rules_by_lhs.get(&nt) {
            for &idx in indexes {
                for sym in &rules[idx].rhs {
                    if let Symbol::Nonterminal(n) = sym {
                        if !reachable.contains(n) {
                            worklist.push(*n);
                        }
                    }
                }
            }
        }
    }
    rules
        .iter()
        .filter(|r| reachable.contains(&r.lhs))
        .cloned()
        .collect()
}

fn build_rhs_by_lhs(rules: &[Rule]) -> BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>> {
    let mut rhs_by_lhs = BTreeMap::<NonterminalID, BTreeSet<Vec<Symbol>>>::new();
    for rule in rules {
        rhs_by_lhs
            .entry(rule.lhs)
            .or_default()
            .insert(rule.rhs.clone());
    }
    rhs_by_lhs
}

fn compute_expandable_single_productions(
    rhs_by_lhs: &BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>>,
) -> (BTreeMap<NonterminalID, Vec<Symbol>>, BTreeSet<NonterminalID>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum VisitState {
        Visiting,
        Expandable,
        NotExpandable,
    }

    fn visit(
        nt: NonterminalID,
        unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
        state: &mut BTreeMap<NonterminalID, VisitState>,
    ) -> bool {
        if let Some(existing) = state.get(&nt).copied() {
            return match existing {
                VisitState::Visiting => false,
                VisitState::Expandable => true,
                VisitState::NotExpandable => false,
            };
        }

        let Some(rhs) = unique_rhs_by_lhs.get(&nt) else {
            return false;
        };

        state.insert(nt, VisitState::Visiting);
        let expandable = rhs.iter().all(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(child) => {
                if unique_rhs_by_lhs.contains_key(child) {
                    visit(*child, unique_rhs_by_lhs, state)
                } else {
                    true
                }
            }
        });
        state.insert(
            nt,
            if expandable {
                VisitState::Expandable
            } else {
                VisitState::NotExpandable
            },
        );
        expandable
    }

    let unique_rhs_by_lhs: BTreeMap<NonterminalID, Vec<Symbol>> = rhs_by_lhs
        .iter()
        .filter_map(|(&nt, rhss)| {
            if rhss.len() == 1 {
                rhss.iter().next().cloned().map(|rhs| (nt, rhs))
            } else {
                None
            }
        })
        .collect();

    let mut state = BTreeMap::<NonterminalID, VisitState>::new();
    let mut expandable = BTreeSet::new();
    for &nt in unique_rhs_by_lhs.keys() {
        if visit(nt, &unique_rhs_by_lhs, &mut state) {
            expandable.insert(nt);
        }
    }

    (unique_rhs_by_lhs, expandable)
}

fn flatten_rhs_symbols(
    rhs: &[Symbol],
    unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
    expandable_single_productions: &BTreeSet<NonterminalID>,
    flatten_cache: &mut HashMap<NonterminalID, Option<Vec<Symbol>>>,
) -> Vec<Symbol> {
    const MAX_FLATTENED_RHS_LEN: usize = 4096;

    fn flatten_symbol(
        symbol: &Symbol,
        out: &mut Vec<Symbol>,
        unique_rhs_by_lhs: &BTreeMap<NonterminalID, Vec<Symbol>>,
        expandable_single_productions: &BTreeSet<NonterminalID>,
        flatten_cache: &mut HashMap<NonterminalID, Option<Vec<Symbol>>>,
    ) {
        match symbol {
            Symbol::Terminal(_) => out.push(symbol.clone()),
            Symbol::Nonterminal(nt)
                if expandable_single_productions.contains(nt) =>
            {
                if let Some(cached) = flatten_cache.get(nt) {
                    match cached {
                        Some(flattened) if out.len() + flattened.len() <= MAX_FLATTENED_RHS_LEN => {
                            out.extend(flattened.iter().cloned());
                        }
                        _ => out.push(symbol.clone()),
                    }
                    return;
                }

                if let Some(expanded_rhs) = unique_rhs_by_lhs.get(nt) {
                    let mut flattened_nt = Vec::new();
                    for expanded_symbol in expanded_rhs {
                        flatten_symbol(
                            expanded_symbol,
                            &mut flattened_nt,
                            unique_rhs_by_lhs,
                            expandable_single_productions,
                            flatten_cache,
                        );
                        if flattened_nt.len() > MAX_FLATTENED_RHS_LEN {
                            flatten_cache.insert(*nt, None);
                            out.push(symbol.clone());
                            return;
                        }
                    }

                    flatten_cache.insert(*nt, Some(flattened_nt.clone()));
                    out.extend(flattened_nt);
                } else {
                    out.push(symbol.clone());
                }
            }
            Symbol::Nonterminal(_) => out.push(symbol.clone()),
        }
    }

    let mut flattened = Vec::new();
    for symbol in rhs {
        flatten_symbol(
            symbol,
            &mut flattened,
            unique_rhs_by_lhs,
            expandable_single_productions,
            flatten_cache,
        );
        if flattened.len() > MAX_FLATTENED_RHS_LEN {
            return rhs.to_vec();
        }
    }
    flattened
}

/// Deduplicate rules, preserving order of first occurrence.
enum RuleDedupKey<'a> {
    Borrowed(NonterminalID, &'a [Symbol]),
    Owned(NonterminalID, Vec<Symbol>),
}

impl RuleDedupKey<'_> {
    fn lhs(&self) -> NonterminalID {
        match self {
            RuleDedupKey::Borrowed(lhs, _) | RuleDedupKey::Owned(lhs, _) => *lhs,
        }
    }

    fn rhs(&self) -> &[Symbol] {
        match self {
            RuleDedupKey::Borrowed(_, rhs) => rhs,
            RuleDedupKey::Owned(_, rhs) => rhs,
        }
    }
}

impl PartialEq for RuleDedupKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.lhs() == other.lhs() && self.rhs() == other.rhs()
    }
}

impl Eq for RuleDedupKey<'_> {}

impl Hash for RuleDedupKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lhs().hash(state);
        self.rhs().hash(state);
    }
}

fn dedup_rules(rules: &mut Vec<Rule>) {
    let rhs_by_lhs = build_rhs_by_lhs(rules);
    let (unique_rhs_by_lhs, expandable_single_productions) =
        compute_expandable_single_productions(&rhs_by_lhs);
    let mut keep = Vec::with_capacity(rules.len());
    {
        let mut seen = HashSet::with_capacity(rules.len());
        let mut flatten_cache = HashMap::<NonterminalID, Option<Vec<Symbol>>>::new();
        for rule in rules.iter() {
            let can_flatten = rule.rhs.iter().any(|symbol| {
                matches!(symbol, Symbol::Nonterminal(nt) if expandable_single_productions.contains(nt))
            });
            let key = if can_flatten {
                RuleDedupKey::Owned(
                    rule.lhs,
                    flatten_rhs_symbols(
                        &rule.rhs,
                        &unique_rhs_by_lhs,
                        &expandable_single_productions,
                        &mut flatten_cache,
                    ),
                )
            } else {
                RuleDedupKey::Borrowed(rule.lhs, &rule.rhs)
            };
            keep.push(seen.insert(key));
        }
    }

    let mut keep_iter = keep.into_iter();
    rules.retain(|_| keep_iter.next().unwrap_or(false));
}

fn is_reflexive_unit_rule(rule: &Rule) -> bool {
    matches!(rule.rhs.as_slice(), [Symbol::Nonterminal(nonterminal)] if *nonterminal == rule.lhs)
}

pub(crate) fn merge_identical_nonterminals(
    rules: &[Rule],
    start: NonterminalID,
) -> Vec<Rule> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Build rhs_by_lhs: the set of productions for each nonterminal.
    let rhs_by_lhs = build_rhs_by_lhs(rules);

    if rhs_by_lhs.len() <= 1 {
        return rules.to_vec();
    }

    let nts: Vec<NonterminalID> = rhs_by_lhs.keys().copied().collect();

    // Fast O(1) lookup from NT ID → index, replacing BTreeMap (which is
    // O(log n) and dominates the hot loop when there are millions of
    // lookups across refinement iterations).
    let max_nt_id = *nts.last().unwrap() as usize;
    let mut nt_to_idx_fast = vec![u32::MAX; max_nt_id + 1];
    for (i, &nt) in nts.iter().enumerate() {
        nt_to_idx_fast[nt as usize] = i as u32;
    }

    // Pre-index production sets for O(1) access by NT index.
    let rhs_by_idx: Vec<&BTreeSet<Vec<Symbol>>> =
        nts.iter().map(|nt| &rhs_by_lhs[nt]).collect();
    let (unique_rhs_by_lhs, expandable_single_productions) =
        compute_expandable_single_productions(&rhs_by_lhs);
    let mut flatten_cache = HashMap::<NonterminalID, Option<Vec<Symbol>>>::new();
    let flattened_rhs_by_idx: Vec<Vec<Vec<Symbol>>> = nts
        .iter()
        .map(|nt| {
            rhs_by_lhs[nt]
                .iter()
                .map(|rhs| {
                    flatten_rhs_symbols(
                        rhs,
                        &unique_rhs_by_lhs,
                        &expandable_single_productions,
                        &mut flatten_cache,
                    )
                })
                .collect()
        })
        .collect();

    // ── Partition refinement (top-down) ──────────────────────────────────
    //
    // Instead of the classical bottom-up "find-one-merge, re-scan" loop
    // (which cascades for O(chain_length) iterations), we use partition
    // refinement: start with the *coarsest* partition consistent with
    // terminal structure, then iteratively refine by incorporating the
    // partition classes of referenced nonterminals.  This discovers all
    // transitively-isomorphic nonterminals in O(refinement_depth) passes,
    // with O(n) work per pass.
    //
    // Each refinement hashes each NT's normalised production set (with
    // self-refs → sentinel, other NT refs → current class ID), then
    // normalises the resulting class IDs by order of first appearance so
    // that the convergence check is stable.

    const SELF_SENTINEL: u64 = u64::MAX;
    const GENERIC_NT: u64 = u64::MAX - 1;

    // Hash a single NT's production set given current class assignments.
    // Uses commutative accumulation (wrapping_add of rotated hashes) to
    // avoid Vec allocation and sorting.
    let hash_nt = |nt_idx: usize, class_of: &[u64]| -> u64 {
        let nt = nts[nt_idx];
        let mut sig: u64 = 0;
        for rhs in &flattened_rhs_by_idx[nt_idx] {
            let mut h = DefaultHasher::new();
            for s in rhs {
                match s {
                    Symbol::Terminal(t) => {
                        0u8.hash(&mut h);
                        t.hash(&mut h);
                    }
                    Symbol::Nonterminal(n) if *n == nt => {
                        1u8.hash(&mut h);
                        SELF_SENTINEL.hash(&mut h);
                    }
                    Symbol::Nonterminal(n) => {
                        let ni = *n as usize;
                        if ni <= max_nt_id && nt_to_idx_fast[ni] != u32::MAX {
                            1u8.hash(&mut h);
                            class_of[nt_to_idx_fast[ni] as usize].hash(&mut h);
                        } else {
                            2u8.hash(&mut h);
                            n.hash(&mut h);
                        }
                    }
                }
            }
            // Commutative combine: wrapping_add of scrambled prod hash.
            let ph = h.finish();
            sig = sig.wrapping_add(ph.wrapping_mul(0x9E3779B97F4A7C15));
        }
        sig
    };

    // Normalise returns (normalised_vec, n_distinct_classes).
    let normalise_counted = |raw: &[u64]| -> (Vec<u64>, usize) {
        let mut map = HashMap::<u64, u64>::with_capacity(raw.len());
        let mut nc: u64 = 0;
        let v: Vec<u64> = raw
            .iter()
            .map(|&v| {
                *map.entry(v).or_insert_with(|| {
                    let c = nc;
                    nc += 1;
                    c
                })
            })
            .collect();
        (v, nc as usize)
    };

    // Quick pre-check: hash with REAL NT IDs (no GENERIC_NT). If every
    // NT already has a unique signature, there are no isomorphisms and
    // we can skip the expensive partition refinement entirely. This
    // handles the common "confirmatory call" case in O(n).
    {
        let real_classes: Vec<u64> = (0..nts.len())
            .map(|i| {
                // Use the NT's own index as its class (finest partition).
                let nt = nts[i];
                let mut sig: u64 = 0;
                for rhs in &flattened_rhs_by_idx[i] {
                    let mut h = DefaultHasher::new();
                    for s in rhs {
                        match s {
                            Symbol::Terminal(t) => {
                                0u8.hash(&mut h);
                                t.hash(&mut h);
                            }
                            Symbol::Nonterminal(n) if *n == nt => {
                                1u8.hash(&mut h);
                                SELF_SENTINEL.hash(&mut h);
                            }
                            Symbol::Nonterminal(n) => {
                                2u8.hash(&mut h);
                                n.hash(&mut h);
                            }
                        }
                    }
                    let ph = h.finish();
                    sig = sig.wrapping_add(ph.wrapping_mul(0x9E3779B97F4A7C15));
                }
                sig
            })
            .collect();
        let mut seen = HashSet::with_capacity(real_classes.len());
        if real_classes.iter().all(|h| seen.insert(*h)) {
            // Every NT has a unique raw-ID signature → no merges possible.
            return rules.to_vec();
        }
    }

    // Initial partition: all NT refs → GENERIC_NT.
    let initial_classes: Vec<u64> = {
        let init = vec![GENERIC_NT; nts.len()];
        (0..nts.len()).map(|i| hash_nt(i, &init)).collect()
    };
    let (mut class_of, n_classes) = normalise_counted(&initial_classes);

    // If every NT is already in its own class, no merges are possible.
    if n_classes == nts.len() {
        return rules.to_vec();
    }

    // ── Compute processing order: ascending "depth" in the reference DAG ──
    //
    // NTs at depth 0 have no references to other NTs (leaves). Depth d means
    // the longest path to a leaf is d hops. Processing in ascending depth
    // with in-place updates (Gauss-Seidel) propagates chain information in
    // ONE pass instead of one-per-chain-link. For purely acyclic chains of
    // depth D, this reduces D iterations to ~1.
    //
    // We compute exact depths via SCC condensation (Kosaraju's) + DAG depth,
    // which handles cycles correctly (NTs in the same SCC share a depth) and
    // works for arbitrarily deep chains.
    let processing_order: Vec<usize> = {
        let n = nts.len();

        // Build adjacency: for each NT index, which other NT indices does it reference?
        let mut refs_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &nt) in nts.iter().enumerate() {
            for rhs in rhs_by_idx[i].iter() {
                for s in rhs {
                    if let Symbol::Nonterminal(r) = s {
                        if *r != nt {
                            let ri = *r as usize;
                            if ri <= max_nt_id && nt_to_idx_fast[ri] != u32::MAX {
                                refs_of[i].push(nt_to_idx_fast[ri] as usize);
                            }
                        }
                    }
                }
            }
        }

        // ── Kosaraju's SCC algorithm (O(V+E), iterative) ──

        // Step 1: iterative DFS on original graph → finish order.
        let mut visited = vec![false; n];
        let mut finish_order = Vec::with_capacity(n);
        for start in 0..n {
            if visited[start] { continue; }
            let mut stk: Vec<(usize, usize)> = vec![(start, 0)];
            visited[start] = true;
            while let Some((v, ni)) = stk.last_mut() {
                if *ni < refs_of[*v].len() {
                    let w = refs_of[*v][*ni];
                    *ni += 1;
                    if !visited[w] {
                        visited[w] = true;
                        stk.push((w, 0));
                    }
                } else {
                    finish_order.push(*v);
                    stk.pop();
                }
            }
        }

        // Step 2: reverse graph + DFS in reverse finish order → SCC IDs.
        // Kosaraju's numbers SCCs in topological order (sources first in the
        // original dependent→dependency edge direction).
        let mut rev_refs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, neighbors) in refs_of.iter().enumerate() {
            for &j in neighbors {
                rev_refs[j].push(i);
            }
        }
        let mut scc_id = vec![0u32; n];
        let mut next_scc = 0u32;
        visited.fill(false);
        for &start in finish_order.iter().rev() {
            if visited[start] { continue; }
            let mut stk = vec![start];
            visited[start] = true;
            while let Some(v) = stk.pop() {
                scc_id[v] = next_scc;
                for &w in &rev_refs[v] {
                    if !visited[w] {
                        visited[w] = true;
                        stk.push(w);
                    }
                }
            }
            next_scc += 1;
        }
        let num_sccs = next_scc as usize;

        // Step 3: condensed DAG depth (longest path to a sink = leaf depth 0).
        // Build inter-SCC edges, deduplicate, then compute depth in reverse
        // topological order (SCCs numbered source-first, so iterate high→low).
        let mut scc_edges: Vec<Vec<u32>> = vec![Vec::new(); num_sccs];
        for (i, neighbors) in refs_of.iter().enumerate() {
            for &j in neighbors {
                if scc_id[i] != scc_id[j] {
                    scc_edges[scc_id[i] as usize].push(scc_id[j]);
                }
            }
        }
        for edges in &mut scc_edges {
            edges.sort_unstable();
            edges.dedup();
        }
        let mut scc_depth = vec![0u32; num_sccs];
        for s in (0..num_sccs).rev() {
            for &dst in &scc_edges[s] {
                scc_depth[s] = scc_depth[s].max(scc_depth[dst as usize] + 1);
            }
        }

        // Map SCC depth back to NTs, sort by ascending depth (leaves first).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| scc_depth[scc_id[i] as usize]);
        order
    };

    // Refine until stable, using depth-ordered in-place (Gauss-Seidel) updates.
    let mut iters = 0u32;
    loop {
        iters += 1;
        let prev_class_of = class_of.clone();
        // In-place update: process NTs in depth order so that deeper NTs
        // (which depend on shallower ones) see already-updated classes.
        let mut raw = vec![0u64; nts.len()];
        for &i in &processing_order {
            raw[i] = hash_nt(i, &class_of);
            // Update class_of in-place for Gauss-Seidel propagation.
            class_of[i] = raw[i];
        }
        let (new_class_of, nc) = normalise_counted(&raw);
        class_of = new_class_of;
        if class_of == prev_class_of {
            break;
        }
        // Early termination: every NT is in its own class → no merges.
        if nc == nts.len() {
            return rules.to_vec();
        }
    }

    // ── Build merge map from final partition ─────────────────────────────
    // Within each equivalence class, pick a representative (prefer start,
    // otherwise lowest NT ID — which is naturally first since `nts` is
    // sorted from a BTreeMap).

    let mut class_to_rep: BTreeMap<u64, NonterminalID> = BTreeMap::new();
    for (idx, &nt) in nts.iter().enumerate() {
        let class = class_of[idx];
        let rep = class_to_rep.entry(class).or_insert(nt);
        if nt == start {
            *rep = start;
        }
    }

    let mut merge_map: BTreeMap<NonterminalID, NonterminalID> = BTreeMap::new();
    for (idx, &nt) in nts.iter().enumerate() {
        let rep = class_to_rep[&class_of[idx]];
        if nt != rep {
            merge_map.insert(nt, rep);
        }
    }

    if merge_map.is_empty() {
        return rules.to_vec();
    }

    // Apply merge map to produce the deduplicated rule set.
    let apply = |nt: NonterminalID| -> NonterminalID {
        *merge_map.get(&nt).unwrap_or(&nt)
    };

    let mut result = Vec::new();
    let mut seen = HashSet::with_capacity(rules.len());
    for rule in rules {
        let lhs = apply(rule.lhs);
        let rhs: Vec<Symbol> = rule
            .rhs
            .iter()
            .map(|symbol| match symbol {
                Symbol::Terminal(terminal) => Symbol::Terminal(*terminal),
                Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(apply(*nonterminal)),
            })
            .collect();
        let merged = Rule { lhs, rhs };
        if is_reflexive_unit_rule(&merged) {
            continue;
        }
        if seen.insert(merged.clone()) {
            result.push(merged);
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Grammar Normalization Pipeline
// ─────────────────────────────────────────────────────────────────────────────
//
// Transforms a grammar so that it satisfies the preconditions required by the
// terminal-characterization stage:
//
//   1. No nullable nonterminals — every nonterminal derives at least one
//      terminal symbol.
//   2. No right recursion — neither direct (A → α A) nor indirect
//      (A →* α B, B →* β A).
//   3. No indirect left recursion — only direct left recursion (A → A α) is
//      permitted (safe for GLR).
//
// The normalization loop repeatedly inlines null productions, eliminates
// right recursion, and exposes hidden left recursion until the grammar stops
// changing, then runs the final epsilon-elimination and unreachable pruning.

