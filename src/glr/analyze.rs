use crate::glr::automaton::{
    compute_first_sets_for_nonterminals,
    compute_follow_sets_for_nonterminals,
    compute_nonterminal_nullability,
    compute_nullable_nonterminals,
    Nullability,
};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

/// Find all non-terminals that are part of a cycle in the given graph using Tarjan's SCC algorithm.
/// Returns a set of non-terminals that can reach themselves (directly or indirectly).
fn find_cyclic_nodes(graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>) -> BTreeSet<NonTerminal> {
    // Build index mapping
    let all_nts: Vec<_> = graph.keys().cloned().collect();
    let nt_to_idx: HashMap<&NonTerminal, usize> = all_nts.iter().enumerate().map(|(i, nt)| (nt, i)).collect();
    let n = all_nts.len();
    
    // Build adjacency list with indices
    let adj: Vec<Vec<usize>> = all_nts.iter().map(|nt| {
        graph.get(nt).map(|targets| {
            targets.iter().filter_map(|t| nt_to_idx.get(t).copied()).collect()
        }).unwrap_or_default()
    }).collect();
    
    // Tarjan's SCC algorithm
    let mut ids = vec![-1i32; n];
    let mut low = vec![0i32; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut id_counter = 0i32;
    let mut cyclic = BTreeSet::new();
    
    fn dfs(
        at: usize,
        adj: &[Vec<usize>],
        ids: &mut [i32],
        low: &mut [i32],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        id_counter: &mut i32,
        cyclic: &mut BTreeSet<usize>,
    ) {
        stack.push(at);
        on_stack[at] = true;
        ids[at] = *id_counter;
        low[at] = *id_counter;
        *id_counter += 1;
        
        for &to in &adj[at] {
            if ids[to] == -1 {
                dfs(to, adj, ids, low, on_stack, stack, id_counter, cyclic);
                low[at] = low[at].min(low[to]);
            } else if on_stack[to] {
                low[at] = low[at].min(ids[to]);
            }
        }
        
        if ids[at] == low[at] {
            let mut scc = Vec::new();
            while let Some(node) = stack.pop() {
                on_stack[node] = false;
                scc.push(node);
                if node == at { break; }
            }
            // An SCC with >1 node means cycles, or a single node with self-loop
            if scc.len() > 1 {
                for &node in &scc {
                    cyclic.insert(node);
                }
            } else if scc.len() == 1 {
                let node = scc[0];
                if adj[node].contains(&node) {
                    cyclic.insert(node);
                }
            }
        }
    }
    
    for i in 0..n {
        if ids[i] == -1 {
            dfs(i, &adj, &mut ids, &mut low, &mut on_stack, &mut stack, &mut id_counter, &mut cyclic);
        }
    }
    
    // Convert indices back to NonTerminals
    cyclic.into_iter().map(|i| all_nts[i].clone()).collect()
}

/// Checks for non-terminals used in rule RHS but never defined in LHS.
pub fn check_for_undefined_non_terminals(productions: &[Production]) -> Vec<String> {
    let mut lhs_nonterms = BTreeSet::new();
    let mut rhs_nonterms = BTreeSet::new();

    for prod in productions {
        lhs_nonterms.insert(prod.lhs.clone());
        rhs_nonterms.extend(prod.rhs.iter().filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt.clone()),
            _ => None,
        }));
    }

    let missing: Vec<_> = rhs_nonterms
        .difference(&lhs_nonterms)
        .map(|nt| nt.0.clone())
        .collect();

    if missing.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "Non-terminal(s) used in rule RHS but never defined in LHS: {:?}",
            missing
        )]
    }
}

/// Checks for length-1 recursion: A ::= (nullable)* B (nullable)* where B can reach A.
pub fn check_for_length_1_recursion(productions: &[Production]) -> Vec<String> {
    let nullable = compute_nullable_nonterminals(productions);
    
    // Build unit graph: A -> B if A has production with exactly one non-nullable symbol B
    let mut graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for prod in productions {
        let non_nullable: Vec<_> = prod.rhs.iter()
            .filter(|s| !matches!(s, Symbol::NonTerminal(nt) if nullable.contains(nt)))
            .collect();
        if non_nullable.len() == 1 {
            if let Symbol::NonTerminal(nt) = non_nullable[0] {
                graph.entry(prod.lhs.clone()).or_default().insert(nt.clone());
            }
        }
    }
    
    // Find all cycles using DFS, returning the path
    let mut cycles = Vec::new();
    let mut visited = BTreeSet::new();
    
    for start_nt in graph.keys() {
        if visited.contains(start_nt) { continue; }
        let mut path = vec![start_nt.clone()];
        let mut in_path = BTreeSet::from([start_nt.clone()]);
        
        fn dfs(current: &NonTerminal, graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>,
               path: &mut Vec<NonTerminal>, in_path: &mut BTreeSet<NonTerminal>,
               visited: &mut BTreeSet<NonTerminal>, cycles: &mut Vec<Vec<NonTerminal>>) {
            if let Some(neighbors) = graph.get(current) {
                for neighbor in neighbors {
                    if in_path.contains(neighbor) {
                        // Found cycle - extract it
                        let start = path.iter().position(|n| n == neighbor).unwrap();
                        let mut cycle = path[start..].to_vec();
                        cycle.push(neighbor.clone());
                        cycles.push(cycle);
                    } else if !visited.contains(neighbor) {
                        path.push(neighbor.clone());
                        in_path.insert(neighbor.clone());
                        dfs(neighbor, graph, path, in_path, visited, cycles);
                        path.pop();
                        in_path.remove(neighbor);
                    }
                }
            }
            visited.insert(current.clone());
        }
        
        dfs(start_nt, &graph, &mut path, &mut in_path, &mut visited, &mut cycles);
    }
    
    // Deduplicate and format cycles
    let mut seen_cycles = BTreeSet::new();
    cycles.into_iter()
        .filter_map(|cycle| {
            // Normalize cycle by rotating to smallest element
            let min_idx = cycle[..cycle.len()-1].iter().enumerate()
                .min_by_key(|(_, n)| n.0.as_str()).map(|(i, _)| i)?;
            let mut normalized: Vec<_> = cycle[min_idx..cycle.len()-1].to_vec();
            normalized.extend(cycle[..min_idx].to_vec());
            normalized.push(normalized[0].clone());
            
            let key: String = normalized.iter().map(|n| n.0.as_str()).collect::<Vec<_>>().join("->");
            if seen_cycles.insert(key.clone()) {
                let path_str = normalized.iter().map(|n| n.0.as_str()).collect::<Vec<_>>().join(" -> ");
                let rtype = if normalized.len() == 2 { "Direct" } else { "Indirect" };
                Some(format!("{} length-1 recursion cycle detected: {}", rtype, path_str))
            } else { None }
        })
        .collect()
}

/// Checks for left-nullable left recursion (e.g., A ::= B A ..., where B is nullable).
pub fn check_for_left_nullable_left_recursion(productions: &[Production]) -> Vec<String> {
    let nullable = compute_nullable_nonterminals(productions);
    productions.iter()
        .filter_map(|prod| {
            let pos = prod.rhs.iter().position(|s| !matches!(s, Symbol::NonTerminal(n) if nullable.contains(n)))?;
            if matches!(&prod.rhs.get(pos), Some(Symbol::NonTerminal(nt)) if nt == &prod.lhs) && pos > 0 {
                Some(format!("Left-nullable left recursion detected in rule '{}'", prod.lhs.0))
            } else { None }
        })
        .collect()
}

/// Compute transitive closure of a graph in-place.
fn transitive_closure(graph: &mut BTreeMap<NonTerminal, BTreeSet<NonTerminal>>) {
    loop {
        let mut changed = false;
        let keys: Vec<_> = graph.keys().cloned().collect();
        for nt in keys {
            let reachable = graph.get(&nt).cloned().unwrap_or_default();
            for reach_nt in reachable {
                if let Some(further) = graph.get(&reach_nt).cloned() {
                    let entry = graph.entry(nt.clone()).or_default();
                    for f in further {
                        if entry.insert(f) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed { break; }
    }
}

/// Checks for indirect hidden left recursion: A -> B α where B is nullable and α can derive to A.
/// Uses Tarjan's SCC algorithm for cycle detection instead of full transitive closure.
/// 
/// Hidden left recursion specifically requires a NULLABLE prefix before the recursive part.
/// Direct left recursion (A -> A α) is NOT hidden left recursion and is fine for LR parsers.
pub fn check_for_indirect_hidden_left_recursion(productions: &[Production]) -> Vec<String> {
    let nullable = compute_nullable_nonterminals(productions);
    
    crate::debug!(5, "HLR check: {} nullable nonterminals", nullable.len());
    for nt in &nullable {
        crate::debug!(6, "  Nullable: {}", nt.0);
    }
    
    // Build left-reachability graph: A -> B if A has production starting with (nullable)* B
    // We track left-reachability through nullable prefixes
    let mut graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
    for prod in productions {
        let mut saw_nullable = false;
        for sym in &prod.rhs {
            if let Symbol::NonTerminal(nt) = sym {
                if saw_nullable {
                    // This nonterminal is after a nullable prefix - potential HLR target
                    graph.entry(prod.lhs.clone()).or_default().insert(nt.clone());
                }
                if nullable.contains(nt) {
                    saw_nullable = true;
                    // Also add edge for nullable nonterminal itself (for indirect HLR)
                    graph.entry(prod.lhs.clone()).or_default().insert(nt.clone());
                } else {
                    break;
                }
            } else { 
                break; 
            }
        }
    }
    
    // Find all non-terminals that are part of a cycle
    let cyclic = find_cyclic_nodes(&graph);
    
    crate::debug!(5, "HLR check: {} cyclic nonterminals in left-reachability graph", cyclic.len());
    
    // Check: for each production A -> (nullable)+ B ... , is B in a cycle that includes A?
    // Note: we only report HLR if there's at least one nullable nonterminal before the recursive part
    let mut errors = Vec::new();
    for prod in productions {
        let lhs = &prod.lhs;
        
        // Find the position after all leading nullable nonterminals
        let mut pos = 0;
        for (i, sym) in prod.rhs.iter().enumerate() {
            match sym {
                Symbol::NonTerminal(nt) if nullable.contains(nt) => pos = i + 1,
                _ => break,
            }
        }
        
        // Only report HLR if there was a nullable prefix (pos > 0)
        // Direct left recursion (pos = 0) is fine for LR parsers
        if pos == 0 {
            continue;
        }
        
        // Check if any nonterminal in the suffix (after nullable prefix) is cyclic
        for sym in &prod.rhs[pos..] {
            if let Symbol::NonTerminal(nt) = sym {
                // Check if lhs is in a cycle AND nt is in the same cycle (both cyclic)
                if cyclic.contains(lhs) && (nt == lhs || cyclic.contains(nt)) {
                    crate::debug!(6, "  HLR detected: {} -> ... {} ... (pos={}, nullable prefix skipped)", lhs.0, nt.0, pos);
                    errors.push(format!("Hidden left recursion: {} via {}", lhs.0, nt.0));
                    break;
                }
            }
        }
    }
    errors
}

/// Eliminates hidden left recursion by inlining nullable prefixes.
/// 
/// Hidden left recursion occurs when A -> B α where B is nullable and α can derive to A.
/// We eliminate it by expanding productions to add variants where nullable prefixes are skipped.
/// This converts hidden left recursion into direct left recursion (which is fine for LR parsing).
///
/// Algorithm:
/// 1. Find nullable nonterminals
/// 2. For each production A -> B1 B2 ... Bk α where B1...Bk are nullable:
///    - If there's hidden left recursion (α can derive to A), add A -> α
///    - This exposes the left recursion directly
/// 3. Remove productions that only serve to hide left recursion
/// 4. Repeat until no more hidden left recursion exists
pub fn eliminate_hidden_left_recursion(
    productions: &mut Vec<Production>,
) {
    const MAX_ITERATIONS: usize = 20;
    
    for iteration in 0..MAX_ITERATIONS {
        let nullable = compute_nullable_nonterminals(productions);
        
        // Build left-reachability graph: A -> B if A has production starting with (nullable)* B
        let mut graph: BTreeMap<NonTerminal, BTreeSet<NonTerminal>> = BTreeMap::new();
        for prod in productions.iter() {
            for sym in &prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    graph.entry(prod.lhs.clone()).or_default().insert(nt.clone());
                    if !nullable.contains(nt) { break; }
                } else { break; }
            }
        }
        
        // Find all non-terminals that are part of a cycle
        let cyclic = find_cyclic_nodes(&graph);
        
        if cyclic.is_empty() {
            crate::debug!(4, "Hidden left recursion eliminated after {} iterations", iteration + 1);
            break;
        }
        
        crate::debug!(5, "Iteration {}: {} cyclic nonterminals", iteration + 1, cyclic.len());
        for nt in &cyclic {
            crate::debug!(6, "  Cyclic: {}", nt.0);
        }
        
        // Find productions with hidden left recursion (nullable prefix + cyclic suffix)
        let mut new_productions: Vec<Production> = Vec::new();
        let mut modified = false;
        
        for prod in productions.iter() {
            // Always keep the original production
            new_productions.push(prod.clone());
            
            let lhs = &prod.lhs;
            
            // Only process productions where lhs is cyclic
            if !cyclic.contains(lhs) {
                continue;
            }
            
            // Check if this production has a nullable prefix
            // Find the nullable prefix length
            let mut nullable_prefix_len = 0;
            for sym in &prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    if nullable.contains(nt) {
                        nullable_prefix_len += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            
            // Debug: show all productions for cyclic nonterminals
            crate::debug!(6, "  {} -> {} (nullable_prefix_len={})", 
                lhs.0,
                prod.rhs.iter().map(|s| match s {
                    Symbol::Terminal(t) => format!("{:?}", t),
                    Symbol::NonTerminal(nt) => nt.0.clone(),
                }).collect::<Vec<_>>().join(" "),
                nullable_prefix_len);
            
            // If there's no nullable prefix, nothing to do for hidden left recursion
            if nullable_prefix_len == 0 {
                continue;
            }
            
            let suffix = &prod.rhs[nullable_prefix_len..];
            
            // Check if suffix starts with a cyclic nonterminal (including lhs)
            let first_suffix_nt = suffix.iter().find_map(|s| {
                if let Symbol::NonTerminal(nt) = s { Some(nt) } else { None }
            });
            
            let suffix_is_cyclic = first_suffix_nt.map_or(false, |nt| {
                nt == lhs || cyclic.contains(nt)
            });
            
            crate::debug!(6, "    suffix={:?}, first_nt={:?}, cyclic={}", 
                suffix.iter().map(|s| match s {
                    Symbol::Terminal(t) => format!("{:?}", t),
                    Symbol::NonTerminal(nt) => nt.0.clone(),
                }).collect::<Vec<_>>(),
                first_suffix_nt.map(|nt| &nt.0),
                suffix_is_cyclic);
            
            if !suffix_is_cyclic {
                continue;
            }
            
            crate::debug!(5, "HLR candidate: {} -> {}", 
                lhs.0,
                prod.rhs.iter().map(|s| match s {
                    Symbol::Terminal(t) => format!("{:?}", t),
                    Symbol::NonTerminal(nt) => nt.0.clone(),
                }).collect::<Vec<_>>().join(" "));
            
            // Generate variants by skipping nullable prefix elements
            for skip in 1..=nullable_prefix_len {
                let new_rhs: Vec<Symbol> = prod.rhs[skip..].to_vec();
                if !new_rhs.is_empty() {
                    let new_prod = Production {
                        lhs: lhs.clone(),
                        rhs: new_rhs,
                    };
                    // Only add if not already present
                    if !new_productions.contains(&new_prod) {
                        crate::debug!(5, "HLR elimination: {} -> {} (skip {})", 
                            lhs.0,
                            new_prod.rhs.iter().map(|s| match s {
                                Symbol::Terminal(t) => format!("{:?}", t),
                                Symbol::NonTerminal(nt) => nt.0.clone(),
                            }).collect::<Vec<_>>().join(" "),
                            skip);
                        new_productions.push(new_prod);
                        modified = true;
                    }
                }
            }
        }
        
        if !modified {
            crate::debug!(4, "No more hidden left recursion to eliminate after {} iterations", iteration + 1);
            break;
        }
        
        // Deduplicate productions
        new_productions.sort_by(|a, b| {
            (&a.lhs.0, &a.rhs).cmp(&(&b.lhs.0, &b.rhs))
        });
        new_productions.dedup();
        
        *productions = new_productions;
        
        if iteration == MAX_ITERATIONS - 1 {
            crate::log_warn!("Hidden left recursion elimination did not converge after {} iterations", MAX_ITERATIONS);
        }
    }
}

/// Checks for any remaining right recursion (direct or indirect) in the grammar.
/// Uses Tarjan's SCC algorithm (O(V+E)) instead of transitive closure (O(n³)).
pub fn check_for_right_recursion(productions: &[Production]) -> Vec<String> {
    let nullable = compute_nullable_nonterminals(productions);
    let graph = build_right_reachability_graph(productions, &nullable);
    let cyclic = find_cyclic_nodes(&graph);
    
    cyclic.iter()
        .map(|nt| format!("Right recursion: {}", nt.0))
        .collect()
}

/// Computes the set of productive non-terminals (those that can derive a terminal string).
fn compute_productive_non_terminals(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut productive_nts = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for prod in productions {
            if productive_nts.contains(&prod.lhs) {
                continue;
            }

            let rhs_is_productive = prod.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::NonTerminal(nt) => productive_nts.contains(nt),
            });

            if rhs_is_productive && productive_nts.insert(prod.lhs.clone()) {
                changed = true;
            }
        }
    }
    productive_nts
}

/// Checks for non-terminals that cannot derive any terminal string.
pub fn check_for_non_productive_non_terminals(productions: &[Production]) -> Vec<String> {
    let all_nonterminals: BTreeSet<NonTerminal> =
        productions.iter().map(|p| p.lhs.clone()).collect();
    let productive_nts = compute_productive_non_terminals(productions);

    let non_productive: Vec<_> = all_nonterminals
        .difference(&productive_nts)
        .map(|nt| nt.0.clone())
        .collect();

    if non_productive.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "Non-terminal(s) are non-productive (cannot derive a terminal string): {:?}",
            non_productive
        )]
    }
}

/// Validates the grammar for common issues, collecting all errors.
///
/// Checks for:
/// 1. Undefined non-terminals.
/// 2. Non-productive non-terminals.
/// 3. Length-1 recursion (direct or indirect).
/// 4. Left-nullable left recursion.
pub fn validate(productions: &[Production]) -> Result<(), String> {
    let mut errors = Vec::new();

    errors.extend(check_for_undefined_non_terminals(productions));
    errors.extend(check_for_non_productive_non_terminals(productions));
    errors.extend(check_for_length_1_recursion(productions));
    errors.extend(check_for_left_nullable_left_recursion(productions));

    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Grammar validation failed with {} error(s):\n- {}",
            errors.len(),
            errors.join("\n- ")
        ))
    }
}

/// Removes productions that use non-terminals on their RHS which are never defined on the LHS
/// of any *remaining* production. This process is repeated until no more productions can be removed.
pub fn remove_productions_with_undefined_nonterminals(
    initial_productions: &[Production],
    exempt: &[usize],
) -> Vec<Production> {
    let mut current: Vec<(usize, Production)> =
        initial_productions.iter().cloned().enumerate().collect();

    loop {
        let defined_lhs: BTreeSet<NonTerminal> = current
            .iter()
            .map(|(_, prod)| prod.lhs.clone())
            .collect();

        let mut removed = Vec::new();
        let mut kept = Vec::new();

        for (i, prod) in current {
            let keep = exempt.contains(&i)
                || prod.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => true,
                    Symbol::NonTerminal(nt) => defined_lhs.contains(nt),
                });
            if keep {
                kept.push((i, prod));
            } else {
                removed.push((i, prod));
            }
        }

        if removed.is_empty() {
            current = kept;
            break;
        }

        crate::debug!(5, "Removing {} productions with undefined non-terminals.", removed.len());

        let all_rhs_nonterminals: BTreeSet<NonTerminal> = removed
            .iter()
            .flat_map(|(_, prod)| {
                prod.rhs.iter().filter_map(|symbol| match symbol {
                    Symbol::NonTerminal(nt) => Some(nt.clone()),
                    _ => None,
                })
            })
            .collect();

        crate::debug!(4, "Missing non-terminals ({}) in productions:", all_rhs_nonterminals.len());
        for nt in all_rhs_nonterminals.difference(&defined_lhs) {
            crate::debug!(6, "  {}", nt.0);
        }

        crate::debug!(7, "Removed productions:");
        for (_, prod) in &removed {
            crate::debug!(7, "  {}", prod);
        }

        current = kept;
    }

    current.into_iter().map(|(_, prod)| prod).collect()
}

/// DEPRECATED: This function is incomplete and known to be broken. It's not currently used.
/// Keeping it here for reference in case we need dead-production elimination in the future.
#[allow(dead_code)]
pub fn drop_dead(productions: &[Production]) -> Vec<Production> {
    let mut nt_reachables: BTreeMap<&NonTerminal, BTreeSet<&NonTerminal>> = BTreeMap::new();

    for prod in productions {
        let rhs_nonterms: BTreeSet<_> = prod
            .rhs
            .iter()
            .filter_map(|symbol| {
                if let Symbol::NonTerminal(nt) = symbol {
                    Some(nt)
                } else {
                    None
                }
            })
            .collect();
        nt_reachables.insert(&prod.lhs, rhs_nonterms);
    }

    loop {
        let mut changed = false;
        for (nt, reachables) in nt_reachables.clone() {
            let old_len = nt_reachables[nt].len();
            for reachable in reachables {
                if let Some(reachable_reachables) = nt_reachables.get(reachable).cloned() {
                    nt_reachables.get_mut(nt).unwrap().extend(reachable_reachables);
                }
            }
            if nt_reachables[nt].len() != old_len {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let start_prod = &productions[0];
    let mut reachable_from_start = BTreeSet::new();
    for symbol in &start_prod.rhs {
        if let Symbol::NonTerminal(nt) = symbol {
            reachable_from_start.insert(nt);
            if let Some(nt_reachables) = nt_reachables.get(nt).cloned() {
                reachable_from_start.extend(nt_reachables);
            }
        }
    }

    let new_productions: Vec<_> = productions
        .iter()
        .filter(|prod| reachable_from_start.contains(&prod.lhs) || *prod == start_prod)
        .cloned()
        .collect();

    crate::debug!(4, "Dropped {} productions", productions.len() - new_productions.len());

    new_productions
}

/// Computes the set of non-terminals that can derive a string containing at least one of the interesting_symbols.
fn compute_can_derive_interesting(
    productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> BTreeSet<NonTerminal> {
    let mut can_derive_interesting = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for production in productions {
            if can_derive_interesting.contains(&production.lhs) {
                continue;
            }

            let rhs_can_lead = production.rhs.iter().any(|symbol| match symbol {
                Symbol::Terminal(_) => interesting_symbols.contains(symbol),
                Symbol::NonTerminal(nt) => {
                    interesting_symbols.contains(symbol) || can_derive_interesting.contains(nt)
                }
            });

            if rhs_can_lead && can_derive_interesting.insert(production.lhs.clone()) {
                changed = true;
            }
        }
    }
    can_derive_interesting
}

/// Computes the set of non-terminals that are reachable by derivation from any non-terminal in interesting_symbols.
/// If interesting_symbols contains no non-terminals, this returns an empty set.
fn compute_reachable_from_interesting_nts(
    productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> BTreeSet<NonTerminal> {
    let seed_interesting_nts: BTreeSet<NonTerminal> = interesting_symbols
        .iter()
        .filter_map(|s| match s {
            Symbol::NonTerminal(nt) => Some(nt.clone()),
            _ => None,
        })
        .collect();

    if seed_interesting_nts.is_empty() {
        return BTreeSet::new();
    }

    let mut reachable_set = seed_interesting_nts.clone();
    let mut worklist: VecDeque<NonTerminal> = seed_interesting_nts.into_iter().collect();

    while let Some(nt_lhs_from_worklist) = worklist.pop_front() {
        for production in productions.iter().filter(|p| p.lhs == nt_lhs_from_worklist) {
            for symbol_in_rhs in &production.rhs {
                if let Symbol::NonTerminal(nt_in_rhs) = symbol_in_rhs {
                    if reachable_set.insert(nt_in_rhs.clone()) {
                        worklist.push_back(nt_in_rhs.clone());
                    }
                }
            }
        }
    }
    reachable_set
}

/// Filters productions to keep only those relevant to deriving specified "interesting" symbols.
pub fn filter_productions_by_reachability(
    initial_productions: &[Production],
    interesting_symbols: &BTreeSet<Symbol>,
) -> Vec<Production> {
    if interesting_symbols.is_empty() {
        crate::debug!(4, "filter_productions_by_reachability: interesting_symbols is empty, returning no productions.");
        return Vec::new();
    }

    let can_derive_set =
        compute_can_derive_interesting(initial_productions, interesting_symbols);
    crate::debug!(6, "filter_productions_by_reachability: CanDeriveInteresting set: {:?}", can_derive_set.iter().map(|nt| &nt.0).collect::<Vec<_>>());

    let mut kept_productions = Vec::new();
    for production in initial_productions {
        let lhs_can_derive_interesting = can_derive_set.contains(&production.lhs);

        let rhs_can_derive_interesting_for_this_rule =
            production.rhs.iter().any(|symbol_in_rhs| match symbol_in_rhs {
                Symbol::Terminal(_) => interesting_symbols.contains(symbol_in_rhs),
                Symbol::NonTerminal(nt_in_rhs) => {
                    interesting_symbols.contains(symbol_in_rhs)
                        || can_derive_set.contains(nt_in_rhs)
                }
            });

        if lhs_can_derive_interesting && rhs_can_derive_interesting_for_this_rule {
            kept_productions.push(production.clone());
        } else {
            crate::debug!(6, "Filtering out production: {} (LHS can derive interesting: {}, RHS of this rule can derive interesting: {})", production, lhs_can_derive_interesting, rhs_can_derive_interesting_for_this_rule);
        }
    }

    kept_productions
}

pub fn minimize_grammar(initial_productions: &[Production]) -> Vec<Production> {
    todo!()
}

/// Helper function to find the last symbol in a rule's RHS that is not a nullable non-terminal.
fn find_last_non_nullable_symbol<'a>(
    rhs: &'a [Symbol],
    nullable_set: &BTreeSet<NonTerminal>,
) -> Option<(usize, &'a Symbol)> {
    for (i, symbol) in rhs.iter().enumerate().rev() {
        let is_nullable = match symbol {
            Symbol::NonTerminal(nt) => nullable_set.contains(nt),
            Symbol::Terminal(_) => false,
        };
        if !is_nullable {
            return Some((i, symbol));
        }
    }
    None
}

pub fn compute_terminal_follow_sets(
    productions: &[Production],
) -> BTreeMap<Terminal, BTreeSet<Terminal>> {
    let nullable_nonterminals = compute_nullable_nonterminals(productions);
    let first_sets =
        compute_first_sets_for_nonterminals(productions, &nullable_nonterminals);
    let nonterminal_follow_sets =
        compute_follow_sets_for_nonterminals(productions, &first_sets, &nullable_nonterminals);

    let mut terminal_follows: BTreeMap<Terminal, BTreeSet<Terminal>> = BTreeMap::new();

    for production in productions {
        let lhs = &production.lhs;
        let rhs = &production.rhs;

        for (i, symbol) in rhs.iter().enumerate() {
            if let Symbol::Terminal(t) = symbol {
                let mut all_following_are_nullable = true;

                for next_symbol in &rhs[i + 1..] {
                    match next_symbol {
                        Symbol::Terminal(next_t) => {
                            terminal_follows
                                .entry(t.clone())
                                .or_default()
                                .insert(next_t.clone());
                            all_following_are_nullable = false;
                            break;
                        }
                        Symbol::NonTerminal(next_nt) => {
                            if let Some(first_set_for_next_nt) = first_sets.get(next_nt) {
                                terminal_follows
                                    .entry(t.clone())
                                    .or_default()
                                    .extend(first_set_for_next_nt.iter().cloned());
                            }
                            if !nullable_nonterminals.contains(next_nt) {
                                all_following_are_nullable = false;
                                break;
                            }
                        }
                    }
                }

                if all_following_are_nullable {
                    if let Some(follow_set_for_lhs) = nonterminal_follow_sets.get(lhs) {
                        terminal_follows
                            .entry(t.clone())
                            .or_default()
                            .extend(follow_set_for_lhs.iter().filter_map(|opt_t| opt_t.clone()));
                    }
                }
            }
        }
    }

    terminal_follows
}

/// Creates a closure that generates unique non-terminal names, suitable for `resolve_right_recursion`.
pub fn create_unique_name_generator(
    all_nonterminals: &BTreeSet<NonTerminal>,
) -> impl FnMut(&str) -> String {
    let mut existing_names: BTreeSet<String> =
        all_nonterminals.iter().map(|nt| nt.0.clone()).collect();

    move |base_name: &str| {
        let mut new_name = format!("{base_name}_rr");
        let mut counter = 1;

        while existing_names.contains(&new_name) {
            counter += 1;
            new_name = format!("{base_name}_rr_{counter}");
        }

        existing_names.insert(new_name.clone());
        new_name
    }
}

/// Resolves indirect right recursion by inlining to convert it to direct.
pub fn resolve_indirect_right_recursion(
    productions: &mut Vec<Production>,
    _new_name_generator: &mut impl FnMut(&str) -> String,
) {
    for iteration in 0..100 {
        let nullable = compute_nullable_nonterminals(productions);
        let graph = build_right_reachability_graph(productions, &nullable);
        
        // Find a non-self-loop cycle using DFS
        let cycle = find_cycle_excluding_self_loops(&graph);
        
        if let Some(cycle) = cycle {
            crate::debug!(6, "Found indirect right recursion: {:?}", cycle.iter().map(|nt| &nt.0).collect::<Vec<_>>());
            inline_right_end(productions, &cycle[0], &cycle[1], &nullable);
        } else {
            crate::debug!(6, "Indirect right recursion resolved after {} iterations", iteration);
            break;
        }
    }
}

/// Build graph: A -> B if A has production ending with B (considering nullable suffix)
fn build_right_reachability_graph(
    productions: &[Production],
    nullable: &BTreeSet<NonTerminal>,
) -> BTreeMap<NonTerminal, BTreeSet<NonTerminal>> {
    let mut graph = BTreeMap::new();
    for prod in productions {
        for i in (0..prod.rhs.len()).rev() {
            if let Symbol::NonTerminal(nt) = &prod.rhs[i] {
                let suffix_nullable = prod.rhs[i + 1..].iter().all(|s| matches!(s, Symbol::NonTerminal(n) if nullable.contains(n)));
                if suffix_nullable {
                    graph.entry(prod.lhs.clone()).or_insert_with(BTreeSet::new).insert(nt.clone());
                }
                if !nullable.contains(nt) { break; }
            } else { break; }
        }
    }
    graph
}

/// Find a cycle in graph that is NOT a self-loop (length > 1)
fn find_cycle_excluding_self_loops(graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>) -> Option<Vec<NonTerminal>> {
    let mut visited = BTreeSet::new();
    let mut in_stack = BTreeSet::new();
    let mut path = Vec::new();
    
    fn dfs(
        node: &NonTerminal,
        graph: &BTreeMap<NonTerminal, BTreeSet<NonTerminal>>,
        visited: &mut BTreeSet<NonTerminal>,
        in_stack: &mut BTreeSet<NonTerminal>,
        path: &mut Vec<NonTerminal>,
    ) -> Option<Vec<NonTerminal>> {
        visited.insert(node.clone());
        in_stack.insert(node.clone());
        path.push(node.clone());
        
        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if neighbor == node { continue; }  // skip self-loops
                if in_stack.contains(neighbor) {
                    // Found cycle back to neighbor already in current path
                    let start = path.iter().position(|n| n == neighbor).unwrap();
                    let cycle: Vec<_> = path[start..].to_vec();
                    if cycle.len() > 1 { return Some(cycle); }
                } else if !visited.contains(neighbor) {
                    if let Some(cycle) = dfs(neighbor, graph, visited, in_stack, path) {
                        return Some(cycle);
                    }
                }
            }
        }
        path.pop();
        in_stack.remove(node);
        None
    }
    
    for node in graph.keys() {
        if !visited.contains(node) {
            if let Some(cycle) = dfs(node, graph, &mut visited, &mut in_stack, &mut path) {
                return Some(cycle);
            }
        }
    }
    None
}

/// Inline productions of `to_nt` into `from_nt` where `from_nt -> ... to_nt (nullable)*`
fn inline_right_end(productions: &mut Vec<Production>, from_nt: &NonTerminal, to_nt: &NonTerminal, nullable: &BTreeSet<NonTerminal>) {
    let to_prods: Vec<_> = productions.iter().filter(|p| &p.lhs == to_nt).cloned().collect();
    if to_prods.is_empty() { return; }
    
    let mut new_prods = Vec::new();
    for prod in productions.iter() {
        if &prod.lhs != from_nt {
            new_prods.push(prod.clone());
            continue;
        }
        // Find rightmost position of to_nt with nullable suffix
        let pos = (0..prod.rhs.len()).rev().find(|&i| {
            matches!(&prod.rhs[i], Symbol::NonTerminal(nt) if nt == to_nt) &&
            prod.rhs[i + 1..].iter().all(|s| matches!(s, Symbol::NonTerminal(n) if nullable.contains(n)))
        });
        if let Some(pos) = pos {
            for to_prod in &to_prods {
                let mut rhs = prod.rhs[..pos].to_vec();
                rhs.extend(to_prod.rhs.clone());
                rhs.extend(prod.rhs[pos + 1..].to_vec());
                // Skip trivial self-loops
                if !(rhs.len() == 1 && matches!(&rhs[0], Symbol::NonTerminal(nt) if nt == from_nt)) {
                    new_prods.push(Production { lhs: from_nt.clone(), rhs });
                }
            }
        } else {
            new_prods.push(prod.clone());
        }
    }
    *productions = new_prods;
}

/// Checks if production ends with its own LHS (direct right recursion)
fn is_direct_right_recursive(prod: &Production) -> bool {
    matches!(prod.rhs.last(), Some(Symbol::NonTerminal(nt)) if nt == &prod.lhs)
}

pub fn resolve_direct_right_recursion(
    productions: &mut Vec<Production>,
    mut new_name_generator: impl FnMut(&str) -> String,
) {
    // Right recursion elimination WITHOUT epsilon productions.
    //
    // For a right-recursive non-terminal A with:
    //   A → α₁ A | α₂ A | ... (recursive productions where αᵢ is prefix)
    //   A → β₁ | β₂ | ...     (non-recursive base cases)
    //
    // Traditional transformation creates epsilon:
    //   A → A' β₁ | A' β₂ | ...
    //   A' → α₁ A' | α₂ A' | ... | ε
    //
    // Epsilon-free transformation:
    //   A → β₁ | β₂ | ... | A' β₁ | A' β₂ | ...  (base cases + recursive cases)
    //   A' → α₁ | α₂ | ... | A' α₁ | A' α₂ | ... (one-or-more αs via left recursion)
    //
    // This avoids epsilon productions by ensuring A' always produces at least one prefix.
    
    let prods_by_lhs: BTreeMap<_, Vec<_>> = productions.iter().cloned()
        .fold(BTreeMap::new(), |mut m, p| { m.entry(p.lhs.clone()).or_default().push(p); m });

    let recursive_nts: BTreeSet<_> = prods_by_lhs.iter()
        .filter(|(_, prods)| prods.iter().any(is_direct_right_recursive))
        .map(|(nt, _)| nt.clone())
        .collect();

    let mut new_productions = Vec::new();
    let mut processed = BTreeSet::new();

    for prod in productions.iter() {
        if !recursive_nts.contains(&prod.lhs) {
            new_productions.push(prod.clone());
            continue;
        }
        if processed.contains(&prod.lhs) { continue; }
        processed.insert(prod.lhs.clone());

        let prods_for_nt = &prods_by_lhs[&prod.lhs];
        let (recursive, non_recursive): (Vec<_>, Vec<_>) = prods_for_nt.iter().cloned()
            .partition(is_direct_right_recursive);

        // Must have both recursive and non-recursive productions to transform
        if non_recursive.is_empty() || recursive.is_empty() {
            // Edge case: only recursive or only non-recursive
            // Keep as-is (though this is unusual)
            for rule in prods_for_nt {
                new_productions.push(rule.clone());
            }
            continue;
        }

        let new_nt = NonTerminal(new_name_generator(&prod.lhs.0));
        crate::debug!(7, "Right-recursion {} -> {} (epsilon-free)", prod.lhs.0, new_nt.0);

        // A → β  (non-recursive base cases stay as-is)
        for rule in &non_recursive {
            new_productions.push(rule.clone());
        }
        
        // A → A' β  (recursive case: prefix(es) followed by base case)
        for rule in &non_recursive {
            let mut rhs = vec![Symbol::NonTerminal(new_nt.clone())];
            rhs.extend(rule.rhs.clone());
            new_productions.push(Production { lhs: prod.lhs.clone(), rhs });
        }
        
        // A' → α  (base case for A': just the prefix, no recursion)
        for rule in &recursive {
            // Extract prefix: everything except the final self-reference
            let prefix = rule.rhs[..rule.rhs.len() - 1].to_vec();
            new_productions.push(Production { lhs: new_nt.clone(), rhs: prefix });
        }
        
        // A' → A' α  (recursive case for A': left-recursive accumulation of prefixes)
        for rule in &recursive {
            let prefix = rule.rhs[..rule.rhs.len() - 1].to_vec();
            let mut rhs = vec![Symbol::NonTerminal(new_nt.clone())];
            rhs.extend(prefix);
            new_productions.push(Production { lhs: new_nt.clone(), rhs });
        }
    }

    *productions = new_productions;
}

/// Left-factor the grammar to share common prefixes.
/// 
/// For productions with the same LHS that share a common prefix:
///   A → α β₁ | α β₂ | α β₃
/// Transform to:
///   A → α A'
///   A' → β₁ | β₂ | β₃
/// 
/// This can reduce the number of parser states by sharing the prefix parsing.
/// 
/// IMPORTANT: This transformation is only applied when it doesn't create epsilon
/// productions. If one alternative IS the common prefix (empty suffix), we skip
/// factoring that group.
/// 
/// Note: This is a single-pass left factoring. Multiple passes may be needed
/// for deeply nested common prefixes.
/// 
/// DEPRECATED: Use `crate::glr::minimizer::left_factor_grammar` instead.
/// This version has a bug where it doesn't preserve the start production at index 0.
#[deprecated(note = "Use crate::glr::minimizer::left_factor_grammar instead")]
pub fn left_factor_grammar(
    productions: &[Production],
    name_generator: &mut impl FnMut(&str) -> String,
) -> Vec<Production> {
    // Group productions by LHS
    let mut prods_by_lhs: BTreeMap<NonTerminal, Vec<&Production>> = BTreeMap::new();
    for p in productions {
        prods_by_lhs.entry(p.lhs.clone()).or_default().push(p);
    }
    
    let mut new_productions = Vec::new();
    
    for (lhs, prods) in &prods_by_lhs {
        if prods.len() < 2 {
            // No factoring possible with single production
            for p in prods {
                new_productions.push((*p).clone());
            }
            continue;
        }
        
        // Find the longest common prefix among all productions for this LHS
        let common_prefix = find_longest_common_prefix(&prods.iter().map(|p| &p.rhs[..]).collect::<Vec<_>>());
        
        if common_prefix.is_empty() {
            // No common prefix, keep as-is
            for p in prods {
                new_productions.push((*p).clone());
            }
            continue;
        }
        
        // Check if ALL productions share this prefix (otherwise it's not a useful factoring)
        let all_share_prefix = prods.iter().all(|p| {
            p.rhs.len() >= common_prefix.len() && p.rhs[..common_prefix.len()] == common_prefix[..]
        });
        
        if !all_share_prefix {
            // Only some productions share the prefix - find groups
            // Group productions by their first symbol
            let mut groups: BTreeMap<Option<Symbol>, Vec<&Production>> = BTreeMap::new();
            for p in prods {
                let key = p.rhs.first().cloned();
                groups.entry(key).or_default().push(p);
            }
            
            // For each group, check if we can factor
            for (first_sym, group) in groups {
                if group.len() >= 2 && first_sym.is_some() {
                    // Find common prefix within this group
                    let group_prefix = find_longest_common_prefix(&group.iter().map(|p| &p.rhs[..]).collect::<Vec<_>>());
                    
                    // Check if factoring would create an epsilon production
                    let would_create_epsilon = group.iter().any(|p| p.rhs.len() == group_prefix.len());
                    
                    if !would_create_epsilon && (group_prefix.len() >= 2 || (group_prefix.len() == 1 && group.len() >= 3)) {
                        // Worth factoring - create new non-terminal
                        let new_nt = NonTerminal(name_generator(&lhs.0));
                        
                        // A → prefix A'
                        let mut factored_rhs = group_prefix.clone();
                        factored_rhs.push(Symbol::NonTerminal(new_nt.clone()));
                        new_productions.push(Production { lhs: lhs.clone(), rhs: factored_rhs });
                        
                        // A' → suffix for each original
                        for p in &group {
                            let suffix = p.rhs[group_prefix.len()..].to_vec();
                            debug_assert!(!suffix.is_empty(), "Left factoring should not create epsilon");
                            new_productions.push(Production { lhs: new_nt.clone(), rhs: suffix });
                        }
                    } else {
                        // Not worth factoring (or would create epsilon)
                        for p in group {
                            new_productions.push((*p).clone());
                        }
                    }
                } else {
                    // Single production or empty first symbol
                    for p in group {
                        new_productions.push((*p).clone());
                    }
                }
            }
        } else {
            // All productions share the prefix - factor them all
            // But first check if factoring would create an epsilon production
            let would_create_epsilon = prods.iter().any(|p| p.rhs.len() == common_prefix.len());
            
            if would_create_epsilon {
                // Skip factoring to avoid epsilon
                for p in prods {
                    new_productions.push((*p).clone());
                }
            } else {
                let new_nt = NonTerminal(name_generator(&lhs.0));
                
                // A → prefix A'
                let mut factored_rhs = common_prefix.clone();
                factored_rhs.push(Symbol::NonTerminal(new_nt.clone()));
                new_productions.push(Production { lhs: lhs.clone(), rhs: factored_rhs });
                
                // A' → suffix for each original
                for p in prods {
                    let suffix = p.rhs[common_prefix.len()..].to_vec();
                    debug_assert!(!suffix.is_empty(), "Left factoring should not create epsilon");
                    new_productions.push(Production { lhs: new_nt.clone(), rhs: suffix });
                }
            }
        }
    }
    
    new_productions
}

/// Find the longest common prefix among a set of sequences.
fn find_longest_common_prefix(sequences: &[&[Symbol]]) -> Vec<Symbol> {
    if sequences.is_empty() {
        return Vec::new();
    }
    if sequences.len() == 1 {
        return sequences[0].to_vec();
    }
    
    let min_len = sequences.iter().map(|s| s.len()).min().unwrap_or(0);
    let mut prefix_len = 0;
    
    for i in 0..min_len {
        let first_sym = &sequences[0][i];
        if sequences.iter().all(|s| &s[i] == first_sym) {
            prefix_len = i + 1;
        } else {
            break;
        }
    }
    
    sequences[0][..prefix_len].to_vec()
}

/// Merge identical non-terminals in the grammar.
/// 
/// If two non-terminals A and B have exactly the same set of productions
/// (modulo renaming A↔B), merge them into one.
/// 
/// This reduces the grammar size and number of parser states.
pub fn merge_identical_nonterminals(
    productions: &[Production],
    start_nt: &NonTerminal,
) -> Vec<Production> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // Group productions by LHS and collect RHS sets
    let mut prods_by_lhs: BTreeMap<NonTerminal, BTreeSet<Vec<Symbol>>> = BTreeMap::new();
    for p in productions {
        prods_by_lhs.entry(p.lhs.clone()).or_default().insert(p.rhs.clone());
    }
    
    // Compute hash for each nonterminal's production set
    let compute_hash = |rhs_set: &BTreeSet<Vec<Symbol>>| -> u64 {
        let mut hasher = DefaultHasher::new();
        for rhs in rhs_set {
            rhs.hash(&mut hasher);
        }
        hasher.finish()
    };
    
    // Group nonterminals by their production set hash
    let mut hash_to_nts: BTreeMap<u64, Vec<NonTerminal>> = BTreeMap::new();
    for (nt, rhs_set) in &prods_by_lhs {
        let h = compute_hash(rhs_set);
        hash_to_nts.entry(h).or_default().push(nt.clone());
    }
    
    // Find pairs of non-terminals with identical production sets (only within same hash bucket)
    let mut merge_map: BTreeMap<NonTerminal, NonTerminal> = BTreeMap::new();
    
    for nts in hash_to_nts.values() {
        if nts.len() < 2 {
            continue;
        }
        for i in 0..nts.len() {
            if merge_map.contains_key(&nts[i]) {
                continue;
            }
            for j in (i + 1)..nts.len() {
                if merge_map.contains_key(&nts[j]) {
                    continue;
                }
                // Check if nts[i] and nts[j] have the same productions
                if prods_by_lhs.get(&nts[i]) == prods_by_lhs.get(&nts[j]) {
                    // Prefer to keep the start NT
                    let (keep, remove) = if nts[j] == *start_nt {
                        (nts[j].clone(), nts[i].clone())
                    } else {
                        (nts[i].clone(), nts[j].clone())
                    };
                    crate::debug!(6, "Merging identical non-terminals: {} -> {}", remove.0, keep.0);
                    merge_map.insert(remove, keep);
                }
            }
        }
    }
    
    if merge_map.is_empty() {
        return productions.to_vec();
    }
    
    // Apply the merge: rename all occurrences of merged NTs
    let apply_merge = |nt: &NonTerminal| -> NonTerminal {
        merge_map.get(nt).cloned().unwrap_or_else(|| nt.clone())
    };
    
    let mut result: Vec<Production> = Vec::new();
    let mut seen: BTreeSet<(NonTerminal, Vec<Symbol>)> = BTreeSet::new();
    
    for p in productions {
        let new_lhs = apply_merge(&p.lhs);
        let new_rhs: Vec<Symbol> = p.rhs.iter().map(|s| match s {
            Symbol::NonTerminal(nt) => Symbol::NonTerminal(apply_merge(nt)),
            Symbol::Terminal(t) => Symbol::Terminal(t.clone()),
        }).collect();
        
        let key = (new_lhs.clone(), new_rhs.clone());
        if seen.insert(key) {
            result.push(Production { lhs: new_lhs, rhs: new_rhs });
        }
    }
    
    result
}

pub fn inline_null_productions(productions: &[Production]) -> Vec<Production> {
    if productions.is_empty() {
        return Vec::new();
    }

    let nullability = compute_nonterminal_nullability(productions);
    
    // We process ALL productions, including the start production.
    // This ensures that if the start production uses a nullable nonterminal, 
    // it gets expanded (e.g. S -> A, A -> epsilon  becomes  S -> A | epsilon).
    // This allows us to subsequently remove ALL epsilon productions safely.
    
    let mut seen = BTreeSet::<Production>::new();
    let mut out = Vec::<Production>::new();

    for (i, prod) in productions.iter().enumerate() {
        // Count nullable symbols for this production
        let nullable_count = prod.rhs.iter().filter(|sym| {
            if let Symbol::NonTerminal(nt) = sym {
                matches!(nullability.get(nt), Some(Nullability::Nullable))
            } else {
                false
            }
        }).count();
        
        // Log productions that would cause exponential blowup
        if nullable_count > 10 {
            crate::debug!(4, "    Warning: Production {} has {} nullable symbols (2^{} = {} variants): {} -> {:?}",
                i, nullable_count, nullable_count, 1usize << nullable_count.min(20), prod.lhs, prod.rhs);
        }
        
        let rhs_variants: Vec<Vec<Symbol>> =
            prod.rhs.iter().fold(vec![vec![]], |acc, sym| {
                let sym_options = match sym {
                    Symbol::Terminal(_) => vec![Some(sym.clone())],
                    Symbol::NonTerminal(nt) => match nullability.get(nt) {
                        Some(Nullability::Null) => vec![None],
                        Some(Nullability::Nullable) => vec![Some(sym.clone()), None],
                        _ => vec![Some(sym.clone())],
                    },
                };

                acc.into_iter()
                    .flat_map(|variant| {
                        sym_options.iter().map(move |opt| {
                            let mut new_variant = variant.clone();
                            if let Some(s) = opt {
                                new_variant.push(s.clone());
                            }
                            new_variant
                        })
                    })
                    .collect()
            });

        for rhs in rhs_variants {
            let new_prod = Production {
                lhs: prod.lhs.clone(),
                rhs,
            };
            if seen.insert(new_prod.clone()) {
                out.push(new_prod);
            }
        }
        
        // Progress logging
        if i > 0 && i % 100 == 0 {
            crate::debug!(6, "    Processed {} / {} productions, {} output productions so far", i, productions.len(), out.len());
        }
    }
    
    crate::debug!(4, "    Null inlining complete: {} input → {} output productions", productions.len(), out.len());

    // Now strict filtering: remove ALL epsilon productions.
    out.into_iter()
        .filter(|p| !p.rhs.is_empty())
        .collect()
}

pub fn inline_unit_productions(productions: &[Production]) -> Vec<Production> {
    todo!()
}

/// Rewrites productions by inserting dummy terminals before their grouped original terminals.
pub fn rewrite_productions_with_dummies(
    original_productions: &[Production],
    dummy_map: &BTreeMap<String, BTreeSet<Terminal>>,
) -> (Vec<Production>, BTreeSet<Terminal>) {
    if dummy_map.is_empty() {
        return (original_productions.to_vec(), BTreeSet::new());
    }

    let mut original_to_dummy: BTreeMap<Terminal, String> = BTreeMap::new();
    for (dummy_name, originals) in dummy_map {
        for original_terminal in originals {
            original_to_dummy.insert(original_terminal.clone(), dummy_name.clone());
        }
    }

    let mut new_productions = Vec::new();
    let mut new_dummy_terminals = BTreeSet::new();

    for prod in original_productions {
        let mut new_rhs = Vec::new();
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                if let Some(dummy_name) = original_to_dummy.get(t) {
                    let dummy_terminal = Terminal::RegexName(dummy_name.clone());
                    new_rhs.push(Symbol::Terminal(dummy_terminal.clone()));
                    new_dummy_terminals.insert(dummy_terminal);
                }
            }
            new_rhs.push(symbol.clone());
        }
        new_productions.push(Production {
            lhs: prod.lhs.clone(),
            rhs: new_rhs,
        });
    }

    (new_productions, new_dummy_terminals)
}