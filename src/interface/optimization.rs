use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::rc::Rc;
use bimap::BiBTreeMap;
use crate::finite_automata::{Expr, QuantifierType};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::{GrammarDefinition, GrammarExpr, ExprNullability, get_expr_nullability};
use crate::types::TerminalID;
use crate::debug;

pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    if std::env::var("DISABLE_GRAMMAR_OPTIMIZATION").is_ok() {
        return;
    }
    let mut optimizer = GrammarOptimizer::new(grammar);
    optimizer.optimize();
}

impl GrammarDefinition {
    pub fn optimize(&mut self) {
        optimize_grammar(self);
    }
}

#[derive(Default, Debug)]
struct OptimizationStats {
    initial_productions: usize,
    final_productions: usize,
    initial_terminals: usize,
    final_terminals: usize,
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Grammar Optimization Stats:")?;
        writeln!(f, "  Productions: {} -> {} (Diff: {})", self.initial_productions, self.final_productions, self.final_productions as isize - self.initial_productions as isize)?;
        writeln!(f, "  Terminals:   {} -> {} (Diff: {})", self.initial_terminals, self.final_terminals, self.final_terminals as isize - self.initial_terminals as isize)?;
        Ok(())
    }
}

struct GrammarOptimizer<'a> {
    grammar: &'a mut GrammarDefinition,
    stats: OptimizationStats,
}

impl<'a> GrammarOptimizer<'a> {
    fn new(grammar: &'a mut GrammarDefinition) -> Self {
        Self {
            grammar,
            stats: OptimizationStats::default(),
        }
    }

    fn count_terminals(&self) -> usize {
        self.grammar.regex_name_to_group_id.len() + 
        self.grammar.literal_to_group_id.len() + 
        self.grammar.external_name_to_group_id.len()
    }

    fn get_ignore_expr(&self) -> Option<Expr> {
        self.grammar.ignore_terminal_id.and_then(|id| {
            self.grammar.group_id_to_expr.get(&id.0).cloned()
        })
    }

    fn optimize(&mut self) {
        self.stats.initial_productions = self.grammar.productions.len();
        self.stats.initial_terminals = self.count_terminals();

        // Try to optimize the grammar by converting regular sub-grammars to regexes
        self.optimize_regular_subgrammars();

        self.stats.final_productions = self.grammar.productions.len();
        self.stats.final_terminals = self.count_terminals();
    }

    /// Optimize by finding and converting regular sub-grammars to regex terminals.
    /// This works for partial optimization - even if the whole grammar isn't regular,
    /// we can still optimize parts of it.
    fn optimize_regular_subgrammars(&mut self) {
        let total_start = std::time::Instant::now();
        debug!(4, "Starting grammar optimization with {} productions", self.grammar.productions.len());
        
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Build initial equations: each NT maps to a RegexTerm
        let mut equations: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        
        let ignore_expr = self.get_ignore_expr();
        let ignore_term = ignore_expr.as_ref().map(|e| {
            Rc::new(RegexTerm::Star(Rc::new(RegexTerm::Concrete(e.clone()))))
        });

        let eq_start = std::time::Instant::now();
        for prod in &self.grammar.productions {
            let Some(term) = self.production_rhs_to_regex_term(&prod.rhs, &nt_names, ignore_term.as_ref()) else {
                // If any production can't be converted, skip optimization for now
                debug!(4, "Grammar optimization failed: production cannot be converted");
                return;
            };
            let term = Rc::new(term);
            equations.entry(prod.lhs.0.clone())
                .and_modify(|existing| {
                    *existing = Rc::new(RegexTerm::Choice(vec![existing.clone(), term.clone()]));
                })
                .or_insert(term);
        }
        debug!(5, "Built {} equations in {:?}", equations.len(), eq_start.elapsed());
        
        // Build dependency graph and compute elimination order (reverse topological)
        let order_start = std::time::Instant::now();
        let elimination_order = compute_elimination_order(&equations, &start_nt);
        debug!(5, "Computed elimination order ({} NTs) in {:?}", elimination_order.len(), order_start.elapsed());
        
        // PHASE 1: Build solution map (solve each NT's equation with Arden's lemma)
        // We DON'T substitute into other equations yet - just solve each NT in isolation
        let mut solutions: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        // Persistent cache for expansion - shared across all expand_with_solutions calls
        let mut expansion_cache: HashMap<*const RegexTerm, Rc<RegexTerm>> = HashMap::new();
        
        let phase1_start = std::time::Instant::now();
        let mut solved_count = 0;
        let mut failed_count = 0;
        let mut solve_time = std::time::Duration::ZERO;
        for (i, nt_to_eliminate) in elimination_order.iter().enumerate() {
            if i > 0 && i % 5000 == 0 {
                debug!(5, "Phase 1 progress: {}/{} NTs, solved={}, failed={}, solve={:?}", 
                    i, elimination_order.len(), solved_count, failed_count, solve_time);
            }
            if nt_to_eliminate == &start_nt {
                continue;
            }
            
            let Some(nt_eq) = equations.remove(nt_to_eliminate) else {
                continue;
            };
            
            // Solve for this NT (apply Arden's lemma if self-recursive)
            // The equation is still in SYMBOLIC form - no expansion yet
            // This is MUCH faster because the original equations are small
            let t1 = std::time::Instant::now();
            let Some(solved) = solve_single_equation_rc(&nt_eq, nt_to_eliminate) else {
                // Non-linear recursion - can't solve with simple Arden's lemma
                equations.insert(nt_to_eliminate.clone(), nt_eq);
                failed_count += 1;
                solve_time += t1.elapsed();
                continue;
            };
            solve_time += t1.elapsed();
            
            // Store the solution (still in symbolic form with NtRefs)
            let solved_rc = Rc::new(solved);
            solutions.insert(nt_to_eliminate.clone(), solved_rc);
            solved_count += 1;
        }
        debug!(5, "Phase 1: solved {} NTs, failed {} in {:?} (solve={:?})", 
            solved_count, failed_count, phase1_start.elapsed(), solve_time);
        
        // PHASE 2: Solve the start NT
        let phase2_start = std::time::Instant::now();
        let Some(start_eq) = equations.remove(&start_nt) else {
            debug!(4, "Grammar optimization failed: no start equation");
            return;
        };
        
        // First solve the start equation symbolically (to handle self-recursion)
        let Some(solved_start_symbolic) = solve_single_equation_rc(&start_eq, &start_nt) else {
            debug!(4, "Grammar optimization failed: could not solve start NT");
            return;
        };
        debug!(5, "Phase 2: solved start NT symbolically in {:?}", phase2_start.elapsed());
        
        // Now expand all NtRefs in the solved start equation
        let expand_start = std::time::Instant::now();
        let solved_start_rc = Rc::new(solved_start_symbolic);
        let expanded_start = expand_with_solutions_cached(&solved_start_rc, &solutions, &mut expansion_cache);
        debug!(5, "Phase 2: expanded start NT in {:?}", expand_start.elapsed());
        
        // Convert the final RegexTerm to Expr
        let convert_start = std::time::Instant::now();
        let Some(final_expr) = regex_term_to_expr(&expanded_start) else {
            debug!(4, "Grammar optimization failed: could not convert to Expr");
            return;
        };
        debug!(5, "Converted to Expr in {:?}", convert_start.elapsed());
        
        // If we have an ignore terminal, wrap the final expression
        let final_expr = if let Some(expr) = ignore_expr {
            let ignore_star = Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore);
            Expr::Seq(vec![ignore_star.clone(), final_expr, ignore_star])
        } else {
            final_expr
        };

        self.replace_grammar_with_single_terminal(final_expr);
        debug!(4, "Grammar optimization complete in {:?}", total_start.elapsed());
    }

    /// Convert a production RHS to a RegexTerm
    fn production_rhs_to_regex_term(&self, rhs: &[Symbol], nt_names: &HashSet<String>, ignore_term: Option<&Rc<RegexTerm>>) -> Option<RegexTerm> {
        if rhs.is_empty() {
            return Some(RegexTerm::Epsilon);
        }

        let terms: Option<Vec<Rc<RegexTerm>>> = rhs.iter().map(|sym| {
            match sym {
                Symbol::Terminal(t) => {
                    let expr = self.get_expr_for_terminal(t)?;
                    Some(Rc::new(RegexTerm::Concrete(expr)))
                }
                Symbol::NonTerminal(nt) => {
                    if nt_names.contains(&nt.0) {
                        Some(Rc::new(RegexTerm::NtRef(nt.0.clone())))
                    } else {
                        None
                    }
                }
            }
        }).collect();

        terms.map(|ts| {
            if let Some(ign) = ignore_term {
                let mut with_ignore = Vec::with_capacity(ts.len() * 2);
                for (i, t) in ts.into_iter().enumerate() {
                    if i > 0 {
                        with_ignore.push(ign.clone());
                    }
                    with_ignore.push(t);
                }
                RegexTerm::make_seq(with_ignore)
            } else {
                RegexTerm::make_seq(ts)
            }
        })
    }

    /// Get the Expr for a terminal
    fn get_expr_for_terminal(&self, t: &Terminal) -> Option<Expr> {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        group_id.and_then(|gid| self.grammar.group_id_to_expr.get(gid).cloned())
    }

    /// Replace the entire grammar with a single terminal
    fn replace_grammar_with_single_terminal(&mut self, expr: Expr) {
        let expr = simplify_expr(expr);
        
        let new_terminal_name = "__optimized_terminal__".to_string();
        let new_group_id = 0;

        self.grammar.regex_name_to_group_id.clear();
        self.grammar.literal_to_group_id.clear();
        self.grammar.group_id_to_expr.clear();

        self.grammar.regex_name_to_group_id.insert(new_terminal_name.clone(), new_group_id);
        self.grammar.group_id_to_expr.insert(new_group_id, expr);

        if self.grammar.ignore_terminal_id.is_some() {
            self.grammar.ignore_terminal_id = None;
        }

        let start_nt = NonTerminal("start'".to_string());
        self.grammar.productions = vec![
            Production {
                lhs: start_nt,
                rhs: vec![Symbol::Terminal(Terminal::RegexName(new_terminal_name))],
            }
        ];
        self.grammar.start_production_id = 0;
    }
}

/// Check if the grammar structure would cause exponential blowup during substitution.
/// This happens when:
/// 1. There are cycles in the dependency graph (mutual recursion)
/// 2. An NT is referenced multiple times (would cause duplication)
fn would_cause_exponential_blowup(equations: &HashMap<String, Rc<RegexTerm>>) -> bool {
    // Build dependency graph: nt -> set of NTs it references (excluding self)
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    
    // Count total references to each NT across all equations
    let mut total_ref_counts: HashMap<String, usize> = HashMap::new();
    
    for (nt, eq) in equations {
        let mut refs = eq.referenced_nts();
        refs.remove(nt); // Self-recursion is OK (handled by Arden's lemma)
        
        // Count references within THIS equation (excluding self)
        for r in eq.count_nt_references() {
            if &r.0 != nt {
                *total_ref_counts.entry(r.0).or_insert(0) += r.1;
            }
        }
        
        deps.insert(nt.clone(), refs);
    }
    
    // Check if any NT is referenced multiple times across ALL equations
    // This would cause duplication during substitution
    for (_nt, count) in &total_ref_counts {
        if *count > 1 {
            return true;
        }
    }
    
    // Check for cycles using DFS
    let mut visited: HashSet<String> = HashSet::new();
    let mut rec_stack: HashSet<String> = HashSet::new();
    
    fn has_cycle(
        nt: &str,
        deps: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(nt) {
            return true; // Found a cycle
        }
        if visited.contains(nt) {
            return false; // Already fully explored
        }
        
        visited.insert(nt.to_string());
        rec_stack.insert(nt.to_string());
        
        if let Some(neighbors) = deps.get(nt) {
            for neighbor in neighbors {
                if has_cycle(neighbor, deps, visited, rec_stack) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(nt);
        false
    }
    
    for nt in equations.keys() {
        if has_cycle(nt, &deps, &mut visited, &mut rec_stack) {
            return true;
        }
    }
    
    false
}

/// Compute an elimination order for the NTs using reverse topological sort.
/// We want to eliminate NTs that are "leaves" first (no dependencies or only self-references),
/// then work our way up to NTs that depend on more things.
fn compute_elimination_order(equations: &HashMap<String, Rc<RegexTerm>>, start_nt: &str) -> Vec<String> {
    // Build dependency graph: nt -> set of NTs it references (excluding self)
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for (nt, eq) in equations {
        let mut refs = eq.referenced_nts();
        refs.remove(nt); // Remove self-reference
        deps.insert(nt.clone(), refs);
    }
    
    // Build reverse dependency graph: nt -> set of NTs that reference it
    let mut rev_deps: HashMap<String, HashSet<String>> = equations.keys()
        .map(|k| (k.clone(), HashSet::new()))
        .collect();
    for (nt, refs) in &deps {
        for r in refs {
            if let Some(set) = rev_deps.get_mut(r) {
                set.insert(nt.clone());
            }
        }
    }
    
    // Count how many unresolved dependencies each NT has
    let mut unresolved_deps: HashMap<String, usize> = deps.iter()
        .map(|(nt, refs)| (nt.clone(), refs.len()))
        .collect();
    
    // Start with NTs that have no dependencies (leaves)
    let mut queue: Vec<String> = unresolved_deps.iter()
        .filter(|(nt, &count)| count == 0 && *nt != start_nt)
        .map(|(nt, _)| nt.clone())
        .collect();
    
    // Sort queue to ensure deterministic ordering (helps with consistent behavior)
    queue.sort();
    
    let mut result = Vec::new();
    let mut resolved: HashSet<String> = HashSet::new();
    
    while let Some(nt) = queue.pop() {
        if resolved.contains(&nt) || &nt == start_nt {
            continue;
        }
        
        resolved.insert(nt.clone());
        result.push(nt.clone());
        
        // For each NT that depends on this one, decrease their unresolved count
        if let Some(dependents) = rev_deps.get(&nt) {
            for dep in dependents {
                if let Some(count) = unresolved_deps.get_mut(dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0 && !resolved.contains(dep) && dep != start_nt {
                        queue.push(dep.clone());
                        queue.sort(); // Keep sorted for determinism
                    }
                }
            }
        }
    }
    
    // Add any remaining NTs (those in cycles) - sort them for determinism
    let mut remaining: Vec<String> = equations.keys()
        .filter(|nt| !resolved.contains(*nt) && *nt != start_nt)
        .cloned()
        .collect();
    remaining.sort();
    // Reverse to process higher-numbered NTs first (typically better for numbered NTs like s0, s1, etc.)
    remaining.reverse();
    result.extend(remaining);
    
    result
}

/// A term that may reference non-terminals (using Rc for sharing)
#[derive(Clone, Debug)]
enum RegexTerm {
    Epsilon,
    Concrete(Expr),
    NtRef(String),
    Seq(Vec<Rc<RegexTerm>>),
    Choice(Vec<Rc<RegexTerm>>),
    Star(Rc<RegexTerm>),
}

impl RegexTerm {
    fn make_seq(terms: Vec<Rc<RegexTerm>>) -> RegexTerm {
        let mut flat = Vec::new();
        for t in terms {
            match t.as_ref() {
                RegexTerm::Epsilon => {}
                RegexTerm::Seq(inner) => flat.extend(inner.iter().cloned()),
                _ => flat.push(t),
            }
        }
        if flat.is_empty() {
            RegexTerm::Epsilon
        } else if flat.len() == 1 {
            // Unwrap the Rc to get the inner term
            match Rc::try_unwrap(flat.into_iter().next().unwrap()) {
                Ok(t) => t,
                Err(rc) => (*rc).clone(),
            }
        } else {
            RegexTerm::Seq(flat)
        }
    }
    
    fn make_choice(terms: Vec<Rc<RegexTerm>>) -> RegexTerm {
        let mut flat = Vec::new();
        for t in terms {
            match t.as_ref() {
                RegexTerm::Choice(inner) => flat.extend(inner.iter().cloned()),
                _ => flat.push(t),
            }
        }
        if flat.is_empty() {
            RegexTerm::Epsilon
        } else if flat.len() == 1 {
            match Rc::try_unwrap(flat.into_iter().next().unwrap()) {
                Ok(t) => t,
                Err(rc) => (*rc).clone(),
            }
        } else {
            RegexTerm::Choice(flat)
        }
    }
    
    fn make_star(inner: Rc<RegexTerm>) -> RegexTerm {
        match inner.as_ref() {
            RegexTerm::Epsilon => RegexTerm::Epsilon,
            RegexTerm::Star(_) => (*inner).clone(), // (a*)* = a*
            _ => RegexTerm::Star(inner),
        }
    }
    
    fn contains_nt(&self, nt: &str) -> bool {
        // Simple version without memoization - used for non-shared trees
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => false,
            RegexTerm::NtRef(n) => n == nt,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                terms.iter().any(|t| t.contains_nt(nt))
            }
            RegexTerm::Star(inner) => inner.contains_nt(nt),
        }
    }
    
    fn referenced_nts(&self) -> HashSet<String> {
        let mut result = HashSet::new();
        self.collect_nts(&mut result);
        result
    }
    
    fn collect_nts(&self, result: &mut HashSet<String>) {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => {}
            RegexTerm::NtRef(n) => { result.insert(n.clone()); }
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                for t in terms {
                    t.collect_nts(result);
                }
            }
            RegexTerm::Star(inner) => inner.collect_nts(result),
        }
    }
    
    /// Count the number of references to each NT in this term
    fn count_nt_references(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        self.collect_nt_counts(&mut counts);
        counts.into_iter().collect()
    }
    
    fn collect_nt_counts(&self, counts: &mut HashMap<String, usize>) {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => {}
            RegexTerm::NtRef(n) => { *counts.entry(n.clone()).or_insert(0) += 1; }
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                for t in terms {
                    t.collect_nt_counts(counts);
                }
            }
            RegexTerm::Star(inner) => inner.collect_nt_counts(counts),
        }
    }
    
    /// Calculate the size of this term (number of nodes in the tree)
    /// This counts each node once, even if referenced via Rc multiple times
    fn size(&self) -> usize {
        let mut visited: HashSet<*const RegexTerm> = HashSet::new();
        self.size_with_visited(&mut visited)
    }
    
    fn size_with_visited(&self, visited: &mut HashSet<*const RegexTerm>) -> usize {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) | RegexTerm::NtRef(_) => 1,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                1 + terms.iter().map(|t| {
                    let ptr = Rc::as_ptr(t);
                    if visited.contains(&ptr) {
                        0 // Already counted this subtree
                    } else {
                        visited.insert(ptr);
                        t.size_with_visited(visited)
                    }
                }).sum::<usize>()
            }
            RegexTerm::Star(inner) => {
                let ptr = Rc::as_ptr(inner);
                if visited.contains(&ptr) {
                    1 // Just count the Star node, inner already counted
                } else {
                    visited.insert(ptr);
                    1 + inner.size_with_visited(visited)
                }
            }
        }
    }
}

/// Solve a single equation: X = equation, where equation may reference X.
/// Uses Arden's lemma: X = αX | β => X = α*β (for right recursion)
/// And: X = Xα | β => X = βα* (for left recursion)
/// 
/// IMPORTANT: This function is designed to work on SMALL, unexpanded equations.
/// The equation should contain NtRefs to the current NT (for self-recursion)
/// and possibly to other NTs (which will be substituted later).
/// 
/// Takes Rc<RegexTerm> to enable memoization for contains_nt checks
fn solve_single_equation_rc(equation: &Rc<RegexTerm>, nt: &str) -> Option<RegexTerm> {
    // Use a simple, non-memoized contains check since equations are small
    fn contains_nt_simple(term: &RegexTerm, nt: &str) -> bool {
        match term {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => false,
            RegexTerm::NtRef(n) => n == nt,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                terms.iter().any(|t| contains_nt_simple(t.as_ref(), nt))
            }
            RegexTerm::Star(inner) => contains_nt_simple(inner.as_ref(), nt),
        }
    }
    
    if !contains_nt_simple(equation.as_ref(), nt) {
        // No self-reference, just return as-is
        return Some((**equation).clone());
    }
    
    // Separate into recursive and non-recursive alternatives
    fn separate_alts(term: &RegexTerm, nt: &str) -> (Vec<RegexTerm>, Vec<RegexTerm>) {
        match term {
            RegexTerm::Choice(alts) => {
                let mut recursive = Vec::new();
                let mut base = Vec::new();
                for alt in alts {
                    if contains_nt_simple(alt.as_ref(), nt) {
                        recursive.push((**alt).clone());
                    } else {
                        base.push((**alt).clone());
                    }
                }
                (recursive, base)
            }
            _ if contains_nt_simple(term, nt) => (vec![term.clone()], vec![]),
            _ => (vec![], vec![term.clone()]),
        }
    }
    
    let (recursive, base) = separate_alts(equation.as_ref(), nt);
    
    if recursive.is_empty() {
        return Some(RegexTerm::make_choice(base.into_iter().map(Rc::new).collect()));
    }
    
    // Extract coefficients from recursive alternatives
    fn extract_coef(term: &RegexTerm, nt: &str) -> Option<(RegexTerm, RegexTerm)> {
        match term {
            RegexTerm::NtRef(n) if n == nt => {
                Some((RegexTerm::Epsilon, RegexTerm::Epsilon))
            }
            RegexTerm::Seq(parts) => {
                // Find the position of the NT reference
                let mut nt_pos = None;
                for (i, p) in parts.iter().enumerate() {
                    if contains_nt_simple(p.as_ref(), nt) {
                        if nt_pos.is_some() {
                            // Multiple occurrences - not linear
                            return None;
                        }
                        nt_pos = Some(i);
                    }
                }
                let pos = nt_pos?;
                
                // Check that the NT-containing element is just NtRef(nt)
                if !matches!(parts[pos].as_ref(), RegexTerm::NtRef(n) if n == nt) {
                    // NT is nested inside something - not simple linear form
                    return None;
                }
                
                let prefix = if pos == 0 {
                    RegexTerm::Epsilon
                } else {
                    RegexTerm::make_seq(parts[..pos].iter().cloned().collect())
                };
                let suffix = if pos == parts.len() - 1 {
                    RegexTerm::Epsilon
                } else {
                    RegexTerm::make_seq(parts[pos+1..].iter().cloned().collect())
                };
                Some((prefix, suffix))
            }
            _ => None,
        }
    }
    
    fn is_epsilon(t: &RegexTerm) -> bool {
        matches!(t, RegexTerm::Epsilon)
    }
    
    let mut right_coefs: Vec<Rc<RegexTerm>> = Vec::new();
    let mut left_coefs: Vec<Rc<RegexTerm>> = Vec::new();
    
    for rec in &recursive {
        if let Some((prefix, suffix)) = extract_coef(rec, nt) {
            match (is_epsilon(&prefix), is_epsilon(&suffix)) {
                (true, true) => {
                    // Just X alone - this means X = X | ... which is weird but valid
                    // The coefficient is epsilon
                }
                (true, _) => {
                    // X suffix => left recursion
                    left_coefs.push(Rc::new(suffix));
                }
                (_, true) => {
                    // prefix X => right recursion
                    right_coefs.push(Rc::new(prefix));
                }
                _ => {
                    // prefix X suffix => not directly solvable with simple Arden
                    return None;
                }
            }
        } else {
            return None;
        }
    }
    
    // Build the solution
    let base_term = if base.is_empty() {
        Rc::new(RegexTerm::Epsilon)
    } else {
        Rc::new(RegexTerm::make_choice(base.into_iter().map(Rc::new).collect()))
    };
    
    // Handle right recursion: X = αX | β => X = α*β
    let mut result = base_term;
    if !right_coefs.is_empty() {
        let right_coef = Rc::new(RegexTerm::make_choice(right_coefs));
        let right_star = Rc::new(RegexTerm::make_star(right_coef));
        result = Rc::new(RegexTerm::make_seq(vec![right_star, result]));
    }
    
    // Handle left recursion: X = Xα | β => X = βα*
    if !left_coefs.is_empty() {
        let left_coef = Rc::new(RegexTerm::make_choice(left_coefs));
        let left_star = Rc::new(RegexTerm::make_star(left_coef));
        result = Rc::new(RegexTerm::make_seq(vec![result, left_star]));
    }
    
    Some(match Rc::try_unwrap(result) {
        Ok(t) => t,
        Err(rc) => (*rc).clone(),
    })
}

/// Separate a term into recursive and non-recursive alternatives (Rc version with memoization)
fn separate_recursive_alts_rc(
    term: &Rc<RegexTerm>, 
    nt: &str,
    contains_cache: &mut HashMap<*const RegexTerm, bool>
) -> (Vec<Rc<RegexTerm>>, Vec<Rc<RegexTerm>>) {
    match term.as_ref() {
        RegexTerm::Choice(alts) => {
            let mut recursive = Vec::new();
            let mut base = Vec::new();
            for alt in alts {
                if contains_nt_memoized(alt, nt, contains_cache) {
                    recursive.push(alt.clone());
                } else {
                    base.push(alt.clone());
                }
            }
            (recursive, base)
        }
        _ if contains_nt_memoized(term, nt, contains_cache) => (vec![term.clone()], vec![]),
        _ => (vec![], vec![term.clone()]),
    }
}

/// Extract (prefix, suffix) from a term of the form "prefix X suffix" (Rc version)
/// Returns None if the term is not linear in X
fn extract_coef_rc(
    term: &Rc<RegexTerm>, 
    nt: &str,
    contains_cache: &mut HashMap<*const RegexTerm, bool>
) -> Option<(RegexTerm, RegexTerm)> {
    match term.as_ref() {
        RegexTerm::NtRef(n) if n == nt => {
            Some((RegexTerm::Epsilon, RegexTerm::Epsilon))
        }
        RegexTerm::Seq(parts) => {
            // Find the position of the NT reference
            let mut nt_pos = None;
            for (i, p) in parts.iter().enumerate() {
                if contains_nt_memoized(p, nt, contains_cache) {
                    if nt_pos.is_some() {
                        // Multiple references - not linear
                        return None;
                    }
                    nt_pos = Some(i);
                }
            }
            
            let pos = nt_pos?;
            
            // Check that the NT reference is simple (just NtRef, not nested)
            if !matches!(parts[pos].as_ref(), RegexTerm::NtRef(n) if n == nt) {
                return None;
            }
            
            let prefix = RegexTerm::make_seq(parts[..pos].to_vec());
            let suffix = RegexTerm::make_seq(parts[pos+1..].to_vec());
            Some((prefix, suffix))
        }
        _ => None,
    }
}

/// Legacy wrapper for solve_single_equation (converts &RegexTerm to Rc)
fn solve_single_equation(equation: &RegexTerm, nt: &str) -> Option<RegexTerm> {
    solve_single_equation_rc(&Rc::new(equation.clone()), nt)
}

fn is_epsilon(term: &RegexTerm) -> bool {
    match term {
        RegexTerm::Epsilon => true,
        RegexTerm::Seq(s) if s.is_empty() => true,
        _ => false,
    }
}

/// Check if term contains a reference to `nt`, using memoization to handle shared subterms
fn contains_nt_memoized(
    term: &Rc<RegexTerm>,
    nt: &str,
    visited: &mut HashMap<*const RegexTerm, bool>
) -> bool {
    let ptr = Rc::as_ptr(term);
    if let Some(&result) = visited.get(&ptr) {
        return result;
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => false,
        RegexTerm::NtRef(n) => n == nt,
        RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
            terms.iter().any(|t| contains_nt_memoized(t, nt, visited))
        }
        RegexTerm::Star(inner) => contains_nt_memoized(inner, nt, visited),
    };
    
    visited.insert(ptr, result);
    result
}

/// Expand a term by substituting all NT references with their solutions
/// Uses memoization and Rc sharing to avoid exponential blowup
fn expand_with_solutions(
    term: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>
) -> Rc<RegexTerm> {
    let mut cache: HashMap<*const RegexTerm, Rc<RegexTerm>> = HashMap::new();
    expand_with_solutions_cached(term, solutions, &mut cache)
}

/// Expand all NtRefs in a term by substituting solutions.
/// Uses iterative deepening with explicit stack to avoid stack overflow.
fn expand_with_solutions_cached(
    root: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>
) -> Rc<RegexTerm> {
    // Use stacker to handle deep recursion by switching to heap allocation when needed
    stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
        expand_with_solutions_cached_impl(root, solutions, cache)
    })
}

fn expand_with_solutions_cached_impl(
    term: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>
) -> Rc<RegexTerm> {
    let ptr = Rc::as_ptr(term);
    
    // For NtRef, resolve to solution and expand that
    if let RegexTerm::NtRef(n) = term.as_ref() {
        if let Some(solution) = solutions.get(n) {
            // Check if solution is already expanded in cache
            let sol_ptr = Rc::as_ptr(solution);
            if let Some(cached) = cache.get(&sol_ptr) {
                return cached.clone();
            }
            // Recursively expand the solution
            return expand_with_solutions_cached(solution, solutions, cache);
        } else {
            return term.clone();
        }
    }
    
    // Check cache for non-NtRef terms
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => term.clone(),
        RegexTerm::NtRef(_) => unreachable!(), // Handled above
        RegexTerm::Seq(parts) => {
            let new_parts: Vec<Rc<RegexTerm>> = parts.iter()
                .map(|p| expand_with_solutions_cached(p, solutions, cache))
                .collect();
            let changed = parts.iter().zip(new_parts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_seq(new_parts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Choice(alts) => {
            let new_alts: Vec<Rc<RegexTerm>> = alts.iter()
                .map(|a| expand_with_solutions_cached(a, solutions, cache))
                .collect();
            let changed = alts.iter().zip(new_alts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_choice(new_alts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Star(inner) => {
            let new_inner = expand_with_solutions_cached(inner, solutions, cache);
            if Rc::ptr_eq(inner, &new_inner) {
                term.clone()
            } else {
                Rc::new(RegexTerm::make_star(new_inner))
            }
        }
    };
    
    cache.insert(ptr, result.clone());
    result
}

/// Substitute all occurrences of `nt` in `term` with `replacement`
/// Returns an Rc to preserve sharing - crucial for avoiding exponential blowup
/// Uses memoization to handle shared subterms efficiently
fn substitute_nt_rc(term: &Rc<RegexTerm>, nt: &str, replacement: &Rc<RegexTerm>) -> Rc<RegexTerm> {
    let mut cache: HashMap<*const RegexTerm, Rc<RegexTerm>> = HashMap::new();
    let mut contains_cache: HashMap<*const RegexTerm, bool> = HashMap::new();
    let result = substitute_nt_rc_cached(term, nt, replacement, &mut cache, &mut contains_cache);
    result
}

fn substitute_nt_rc_cached(
    term: &Rc<RegexTerm>,
    nt: &str,
    replacement: &Rc<RegexTerm>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>,
    contains_cache: &mut HashMap<*const RegexTerm, bool>
) -> Rc<RegexTerm> {
    let ptr = Rc::as_ptr(term);
    
    // Check cache first
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    
    // Check if this term contains the NT at all (with memoization)
    if !contains_nt_memoized(term, nt, contains_cache) {
        cache.insert(ptr, term.clone());
        return term.clone();
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => term.clone(),
        RegexTerm::NtRef(n) if n == nt => replacement.clone(),
        RegexTerm::NtRef(_) => term.clone(),
        RegexTerm::Seq(parts) => {
            let new_parts: Vec<Rc<RegexTerm>> = parts.iter()
                .map(|p| substitute_nt_rc_cached(p, nt, replacement, cache, contains_cache))
                .collect();
            // Only create a new Rc if something actually changed
            let changed = parts.iter().zip(new_parts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_seq(new_parts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Choice(alts) => {
            let new_alts: Vec<Rc<RegexTerm>> = alts.iter()
                .map(|a| substitute_nt_rc_cached(a, nt, replacement, cache, contains_cache))
                .collect();
            // Only create a new Rc if something actually changed
            let changed = alts.iter().zip(new_alts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_choice(new_alts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Star(inner) => {
            let new_inner = substitute_nt_rc_cached(inner, nt, replacement, cache, contains_cache);
            if Rc::ptr_eq(inner, &new_inner) {
                term.clone()
            } else {
                Rc::new(RegexTerm::make_star(new_inner))
            }
        }
    };
    
    cache.insert(ptr, result.clone());
    result
}

/// Wrapper for backward compatibility - converts result to non-Rc
fn substitute_nt(term: &RegexTerm, nt: &str, replacement: &Rc<RegexTerm>) -> RegexTerm {
    let rc_term = Rc::new(term.clone());
    let result = substitute_nt_rc(&rc_term, nt, replacement);
    // Try to unwrap the Rc if we're the only owner, otherwise clone
    Rc::try_unwrap(result).unwrap_or_else(|rc| (*rc).clone())
}

/// Convert a RegexTerm (with no NT references) to an Expr
/// Uses caching via Arc<Expr> to preserve sharing from Rc<RegexTerm>
fn regex_term_to_expr(term: &RegexTerm) -> Option<Expr> {
    use std::sync::Arc;
    let mut cache: HashMap<*const RegexTerm, Arc<Expr>> = HashMap::new();
    regex_term_to_expr_cached(&Rc::new(term.clone()), &mut cache)
        .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
}

fn regex_term_to_expr_cached(
    term: &Rc<RegexTerm>,
    cache: &mut HashMap<*const RegexTerm, std::sync::Arc<Expr>>
) -> Option<std::sync::Arc<Expr>> {
    // Use stacker to handle deep recursion
    stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
        regex_term_to_expr_cached_impl(term, cache)
    })
}

fn regex_term_to_expr_cached_impl(
    term: &Rc<RegexTerm>,
    cache: &mut HashMap<*const RegexTerm, std::sync::Arc<Expr>>
) -> Option<std::sync::Arc<Expr>> {
    use std::sync::Arc;
    
    let ptr = Rc::as_ptr(term);
    if let Some(cached) = cache.get(&ptr) {
        return Some(cached.clone());
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon => Some(Arc::new(Expr::Epsilon)),
        RegexTerm::Concrete(e) => Some(Arc::new(e.clone())),
        RegexTerm::NtRef(_) => None, // Should not have NT refs at this point
        RegexTerm::Seq(parts) => {
            let exprs: Option<Vec<Expr>> = parts.iter()
                .map(|p| regex_term_to_expr_cached(p, cache).map(|arc| {
                    // Use Shared(Arc<Expr>) to preserve sharing
                    if Arc::strong_count(&arc) > 1 {
                        Expr::Shared(arc)
                    } else {
                        Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
                    }
                }))
                .collect();
            exprs.map(|es| Arc::new(make_seq(es)))
        }
        RegexTerm::Choice(alts) => {
            let exprs: Option<Vec<Expr>> = alts.iter()
                .map(|a| regex_term_to_expr_cached(a, cache).map(|arc| {
                    // Use Shared(Arc<Expr>) to preserve sharing
                    if Arc::strong_count(&arc) > 1 {
                        Expr::Shared(arc)
                    } else {
                        Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
                    }
                }))
                .collect();
            exprs.map(|es| Arc::new(make_choice(es)))
        }
        RegexTerm::Star(inner) => {
            regex_term_to_expr_cached(inner, cache).map(|arc| {
                let inner_expr = if Arc::strong_count(&arc) > 1 {
                    Expr::Shared(arc)
                } else {
                    Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
                };
                Arc::new(Expr::Quantifier(Box::new(inner_expr), QuantifierType::ZeroOrMore))
            })
        }
    };
    
    if let Some(ref r) = result {
        cache.insert(ptr, r.clone());
    }
    result
}

/// Make a sequence Expr, handling Epsilon and flattening
fn make_seq(exprs: Vec<Expr>) -> Expr {
    let filtered: Vec<Expr> = exprs.into_iter()
        .flat_map(|e| match e {
            Expr::Epsilon => vec![],
            Expr::Seq(inner) => inner,
            other => vec![other],
        })
        .collect();
    if filtered.is_empty() {
        Expr::Epsilon
    } else if filtered.len() == 1 {
        filtered.into_iter().next().unwrap()
    } else {
        Expr::Seq(filtered)
    }
}

/// Make a choice Expr, handling single elements and flattening
fn make_choice(exprs: Vec<Expr>) -> Expr {
    let filtered: Vec<Expr> = exprs.into_iter()
        .flat_map(|e| match e {
            Expr::Choice(inner) => inner,
            other => vec![other],
        })
        .collect();
    if filtered.is_empty() {
        Expr::Epsilon
    } else if filtered.len() == 1 {
        filtered.into_iter().next().unwrap()
    } else {
        Expr::Choice(filtered)
    }
}

/// Check if an Expr is epsilon (possibly through Shared wrappers)
fn is_epsilon_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::Shared(inner) => is_epsilon_expr(inner),
        _ => false,
    }
}

/// Simplify an expression
fn simplify_expr(expr: Expr) -> Expr {
    let mut cache: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    simplify_expr_cached(expr, &mut cache)
}

fn simplify_expr_cached(expr: Expr, cache: &mut HashMap<*const Expr, Arc<Expr>>) -> Expr {
    // Use stacker to handle deep recursion
    stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
        simplify_expr_cached_impl(expr, cache)
    })
}

fn simplify_expr_cached_impl(expr: Expr, cache: &mut HashMap<*const Expr, Arc<Expr>>) -> Expr {
    match expr {
        Expr::Seq(exprs) => {
            let mut simplified = Vec::new();
            for e in exprs {
                let e = simplify_expr_cached(e, cache);
                match e {
                    Expr::Epsilon => {}
                    Expr::Seq(inner) => simplified.extend(inner),
                    other => simplified.push(other),
                }
            }
            if simplified.is_empty() {
                Expr::Epsilon
            } else if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else {
                Expr::Seq(simplified)
            }
        }
        Expr::Choice(exprs) => {
            let simplified: Vec<Expr> = exprs.into_iter()
                .map(|e| simplify_expr_cached(e, cache))
                .flat_map(|e| match e {
                    Expr::Choice(inner) => inner,
                    other => vec![other],
                })
                .collect();
            if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else if simplified.len() == 2 {
                // Check for Choice([a, Epsilon]) or Choice([Epsilon, a]) -> a?
                let (first, second) = (&simplified[0], &simplified[1]);
                if is_epsilon_expr(second) {
                    Expr::Quantifier(Box::new(first.clone()), QuantifierType::ZeroOrOne)
                } else if is_epsilon_expr(first) {
                    Expr::Quantifier(Box::new(second.clone()), QuantifierType::ZeroOrOne)
                } else {
                    Expr::Choice(simplified)
                }
            } else {
                Expr::Choice(simplified)
            }
        }
        Expr::Quantifier(inner, qtype) => {
            let inner = simplify_expr_cached(*inner, cache);
            match (&inner, &qtype) {
                (Expr::Epsilon, _) => Expr::Epsilon,
                (Expr::Quantifier(inner2, QuantifierType::ZeroOrMore), QuantifierType::ZeroOrMore) |
                (Expr::Quantifier(inner2, QuantifierType::ZeroOrMore), QuantifierType::OneOrMore) |
                (Expr::Quantifier(inner2, QuantifierType::OneOrMore), QuantifierType::ZeroOrMore) => {
                    Expr::Quantifier(inner2.clone(), QuantifierType::ZeroOrMore)
                }
                _ => Expr::Quantifier(Box::new(inner), qtype),
            }
        }
        Expr::Shared(inner) => {
            let ptr = Arc::as_ptr(&inner);
            // Check cache for this shared subtree
            if let Some(cached) = cache.get(&ptr) {
                // Return a Shared pointing to the cached result
                return Expr::Shared(cached.clone());
            }
            // Simplify the inner expression
            let simplified = simplify_expr_cached((*inner).clone(), cache);
            // Cache and return
            let result = Arc::new(simplified);
            cache.insert(ptr, result.clone());
            Expr::Shared(result)
        }
        other => other,
    }
}


#[cfg(test)]
mod tests {
    use indoc::indoc;
    use super::*;
    use crate::datastructures::u8set::U8Set;

    #[test]
    fn test_converts_leaf_nt_to_terminal() {
        let (grammar_exprs, regex_exprs) = (
            vec![
                ("start".to_string(), GrammarExpr::Ref("A".to_string())),
                ("A".to_string(), GrammarExpr::Choice(vec![GrammarExpr::Ref("B".to_string()), GrammarExpr::Ref("C".to_string())])),
            ],
            vec![
                ("B".to_string(), Expr::U8Seq(b"b".to_vec())),
                ("C".to_string(), Expr::U8Seq(b"c".to_vec())),
            ]
        );
        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
        optimize_grammar(&mut grammar);
        println!("{grammar}");

        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    #[test]
    fn test_merge_adjacent_terminals() {
        let (grammar_exprs, regex_exprs) = (
            vec![
                ("start".to_string(), GrammarExpr::Sequence(vec![GrammarExpr::Ref("A".to_string()), GrammarExpr::Ref("B".to_string())])),
            ],
            vec![
                ("A".to_string(), Expr::U8Seq(b"a".to_vec())),
                ("B".to_string(), Expr::U8Seq(b"b".to_vec())),
            ]
        );
        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
        optimize_grammar(&mut grammar);
        println!("{grammar}");

        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    #[test]
    fn test_rolls_up_chain_of_regular_rules() {
        let mut grammar_exprs = vec![("start".to_string(), GrammarExpr::Ref("s0".to_string()))];
        let mut regex_exprs = vec![("C".to_string(), Expr::U8Seq(b"c".to_vec()))];

        let chain_len = 20;

        for i in 0..chain_len {
             let char_val = (b'a' + i as u8) as char;
             let term_name = format!("T{}", i);
             regex_exprs.push((term_name.clone(), Expr::U8Seq(vec![char_val as u8])));

            let next_s = if i < chain_len -1 {
                GrammarExpr::Ref(format!("s{}", i + 1))
            } else {
                GrammarExpr::Ref("C".to_string())
            };

            grammar_exprs.push((
                format!("s{}", i),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Ref(term_name),
                    next_s
                ])
            ));
        }

        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
        optimize_grammar(&mut grammar);
        println!("{grammar}");

        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    #[test]
    fn test_fuzz_regex_to_grammar_optimization() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
            fn range(&mut self, min: usize, max: usize) -> usize {
                if min >= max { return min; }
                (self.next() as usize % (max - min)) + min
            }
            fn bool(&mut self) -> bool { self.next() % 2 == 0 }
        }

        use crate::finite_automata::{Expr, QuantifierType};
        use crate::datastructures::u8set::U8Set;
        use crate::interface::{GrammarExpr, GrammarDefinition};

        fn gen_expr(rng: &mut Rng, depth: usize, term_defs: &mut Vec<(String, Expr)>, term_counter: &mut usize) -> GrammarExpr {
            if depth == 0 || (rng.bool() && rng.bool()) {
                let is_class = rng.bool();
                let expr = if is_class {
                    let b = (rng.next() % 256) as u8;
                    Expr::U8Class(U8Set::from_u8(b))
                } else {
                    let len = rng.range(1, 4);
                    let bytes: Vec<u8> = (0..len).map(|_| (rng.next() % 256) as u8).collect();
                    Expr::U8Seq(bytes)
                };

                let name = format!("T{}", term_counter);
                *term_counter += 1;
                term_defs.push((name.clone(), expr));
                return GrammarExpr::Ref(name);
            }

            match rng.range(0, 3) {
                0 => {
                    let len = rng.range(2, 4);
                    let exprs = (0..len).map(|_| gen_expr(rng, depth - 1, term_defs, term_counter)).collect();
                    GrammarExpr::Sequence(exprs)
                }
                1 => {
                    let len = rng.range(2, 4);
                    let exprs = (0..len).map(|_| gen_expr(rng, depth - 1, term_defs, term_counter)).collect();
                    GrammarExpr::Choice(exprs)
                }
                2 => {
                    let child = gen_expr(rng, depth - 1, term_defs, term_counter);
                    match rng.range(0, 3) {
                        0 => GrammarExpr::Optional(Box::new(child)),
                        1 => GrammarExpr::Repeat(Box::new(child)),
                        _ => {
                             let child_clone = child.clone();
                             GrammarExpr::Sequence(vec![child, GrammarExpr::Repeat(Box::new(child_clone))])
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        let mut rng = Rng(12345);
        for i in 0..20 {
            let mut regex_exprs = Vec::new();
            let mut term_counter = 0;
            let root = gen_expr(&mut rng, 4, &mut regex_exprs, &mut term_counter);

            let grammar_exprs = vec![("start".to_string(), root)];
            let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();

            let initial_count = grammar.terminal_to_group_id().len();

            optimize_grammar(&mut grammar);

            use crate::interface::CompiledGrammar;
            let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
        }
    }

    #[test]
    fn test_diff_grammar_structure() {
        let grammar_exprs = vec![
            ("start".to_string(), GrammarExpr::Ref("Block".to_string())),
            ("Block".to_string(), GrammarExpr::Choice(vec![
                GrammarExpr::Ref("Line1".to_string()),
                GrammarExpr::Ref("Line2".to_string()),
            ])),
            ("Line1".to_string(), GrammarExpr::Sequence(vec![
                 GrammarExpr::Choice(vec![
                     GrammarExpr::Literal(b" ".to_vec()),
                     GrammarExpr::Literal(b"-".to_vec()),
                 ]),
                 GrammarExpr::Literal(b"foo".to_vec()),
                 GrammarExpr::Literal(b"\n".to_vec()),
            ])),
            ("Line2".to_string(), GrammarExpr::Sequence(vec![
                 GrammarExpr::Choice(vec![
                     GrammarExpr::Literal(b" ".to_vec()),
                     GrammarExpr::Literal(b"-".to_vec()),
                 ]),
                 GrammarExpr::Literal(b"bar".to_vec()),
                 GrammarExpr::Literal(b"\n".to_vec()),
            ])),
        ];
        let regex_exprs = vec![];

        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, regex_exprs).unwrap();
        let initial_terminals = grammar.terminal_to_group_id().len();
        println!("Initial terminals: {}", initial_terminals);

        optimize_grammar(&mut grammar);
        println!("{grammar}");

        assert!(grammar.terminal_to_group_id().len() < initial_terminals);
        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    #[test]
    fn test_complex_nesting() {
        let grammar_exprs = vec![
            ("start".to_string(), GrammarExpr::Ref("A".to_string())),
            ("A".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                    GrammarExpr::Literal(b"b".to_vec()),
                ]),
                GrammarExpr::Literal(b"c".to_vec()),
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"d".to_vec()),
                    GrammarExpr::Literal(b"e".to_vec()),
                ]),
            ])),
        ];

        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, vec![]).unwrap();
        let initial = grammar.terminal_to_group_id().len();
        println!("Initial terminals: {}", initial);

        optimize_grammar(&mut grammar);
        println!("{grammar}");

        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    fn build_diff_grammar(num_lines: usize) -> GrammarDefinition {
        use crate::finite_automata::Expr;
        use crate::interface::{GrammarExpr, GrammarDefinition};

        let mut grammar_exprs = Vec::new();
        let mut regex_exprs = Vec::new();

        regex_exprs.push(("HUNK_HEADER".to_string(), Expr::U8Seq(b"@@".to_vec())));
        regex_exprs.push(("PLUS_LINE".to_string(), Expr::U8Seq(b"+".to_vec())));
        regex_exprs.push(("EOF".to_string(), Expr::U8Seq(b"EOF".to_vec())));

        for i in 0..num_lines {
            regex_exprs.push((format!("L{}", i), Expr::U8Seq(format!("line{}", i).as_bytes().to_vec())));
        }

        grammar_exprs.push(("start".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("HUNK_HEADER".to_string()),
                GrammarExpr::Ref("s0".to_string())
            ]))),
            GrammarExpr::Ref("EOF".to_string())
        ])));

        for i in 0..num_lines {
            grammar_exprs.push((
                format!("s{}", i),
                GrammarExpr::Choice(vec![
                    GrammarExpr::Ref(format!("l{}", i)),
                    if i < num_lines - 1 {
                        GrammarExpr::Ref(format!("s{}", i + 1))
                    } else {
                        GrammarExpr::Ref(format!("s{}", num_lines))
                    }
                ])
            ));
        }
        grammar_exprs.push((
            format!("s{}", num_lines),
            GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string())))
        ));

        for i in 0..num_lines {
            let continuation = if i < num_lines - 1 {
                GrammarExpr::Optional(Box::new(GrammarExpr::Choice(vec![
                    GrammarExpr::Ref(format!("l{}", i + 1)),
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string()))),
                        GrammarExpr::Ref("HUNK_HEADER".to_string()),
                        GrammarExpr::Ref(format!("s{}", i + 1))
                    ])
                ])))
            } else {
                GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string()))),
                    GrammarExpr::Ref("HUNK_HEADER".to_string()),
                    GrammarExpr::Ref(format!("s{}", num_lines))
                ])))
            };

            grammar_exprs.push((
                format!("l{}", i),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string()))),
                    GrammarExpr::Ref(format!("L{}", i)),
                    continuation
                ])
            ));
        }

        GrammarDefinition::from_exprs_no_optimize(grammar_exprs, regex_exprs).unwrap()
    }

    #[test]
    fn test_diff_grammar_optimization_correctness() {
        let mut grammar = build_diff_grammar(10);
        
        let start = std::time::Instant::now();
        optimize_grammar(&mut grammar);
        let duration = start.elapsed();

        println!("Optimization took: {:?}", duration);
        println!("Final terminal count: {}", grammar.terminal_to_group_id().len());

        if grammar.terminal_to_group_id().len() != 1 {
            eprintln!("Grammar: {}", grammar);
            panic!("Expected 1 terminal, got {}.", grammar.terminal_to_group_id().len());
        }

        use crate::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }

    #[ignore]
    #[test]
    fn test_diff_grammar_optimization_performance() {
        // This test verifies that grammar optimization scales roughly linearly.
        // Due to system load variability, we run multiple trials and take the best ratio.
        let mut n = 100;
        let mut t_base = 0.0;

        // Find a baseline size that takes at least 10ms (increased for more stable measurements)
        loop {
            let mut grammar = build_diff_grammar(n);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            let duration = start.elapsed();
            t_base = duration.as_secs_f64();

            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n);

            if t_base > 0.010 {
                break;
            }
            n *= 2;

            if n > 10_000 {
                println!("Warning: reached n={} without exceeding time threshold (got {:.4}s). Skipping perf check.", n, t_base);
                return;
            }
        }

        // Run 5 trials and take the best ratio to reduce flakiness from system load
        let n_scaled = n * 3;
        let mut best_ratio = f64::MAX;
        
        for trial in 0..5 {
            // Re-measure baseline for each trial
            let mut grammar = build_diff_grammar(n);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            t_base = start.elapsed().as_secs_f64();
            
            let mut grammar = build_diff_grammar(n_scaled);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            let t_scaled = start.elapsed().as_secs_f64();

            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n_scaled);

            let ratio = t_scaled / t_base;
            println!("Trial {}: Baseline n={} ({:.4}s), Scaled n={} ({:.4}s), Ratio: {:.2}", 
                     trial + 1, n, t_base, n_scaled, t_scaled, ratio);
            
            best_ratio = best_ratio.min(ratio);
        }

        println!("Best ratio: {:.2}", best_ratio);

        // Linear scaling would give ratio ~3.0 for 3x input
        // Allow up to 8.0x for system variability (we're testing O(n) not O(n^2))
        assert!(best_ratio < 8.0, "Performance scaling looks worse than linear (best ratio {:.2})", best_ratio);
    }

    #[test]
    fn test_simple_recursion_optimization() {
        let grammar_exprs = vec![
            ("start".to_string(), GrammarExpr::Ref("S".to_string())),
            ("S".to_string(), GrammarExpr::Choice(vec![
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                    GrammarExpr::Ref("S".to_string()),
                ]),
                GrammarExpr::Literal(b"b".to_vec()),
            ])),
        ];

        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, vec![]).unwrap();
        optimize_grammar(&mut grammar);

        use crate::interface::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }

    #[test]
    fn test_x_semicolon_x() {
        // Build grammar manually to avoid from_ebnf internal optimization
        let grammar_exprs = vec![
            ("program".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("expression_statement".to_string()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Ref("expression_statement".to_string()))),
                GrammarExpr::Ref("EOF".to_string()),
            ])),
            ("expression_statement".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("expression".to_string()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b";".to_vec()))),
            ])),
            ("expression".to_string(), GrammarExpr::Literal(b"x".to_vec())),
        ];
        let regex_exprs = vec![
            ("EOF".to_string(), Expr::U8Seq(b"$".to_vec())),
        ];
        
        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, regex_exprs).unwrap();
        println!("=== BEFORE OPTIMIZATION ===");
        println!("{grammar}");
        optimize_grammar(&mut grammar);
        println!("=== AFTER OPTIMIZATION ===");
        println!("{grammar}");
        assert_eq!(grammar.terminal_to_group_id().len(), 1);

        let x = Expr::U8Seq(b"x".to_vec());
        let semi_opt = Expr::Quantifier(Box::new(Expr::U8Seq(b";".to_vec())), QuantifierType::ZeroOrOne);
        let eof = Expr::U8Seq(b"$".to_vec());
        let stmt = Expr::Seq(vec![x.clone(), semi_opt.clone()]);

        let expected_regex = Expr::Seq(vec![
            x, semi_opt,
            Expr::Quantifier(Box::new(stmt), QuantifierType::ZeroOrOne),
            eof
        ]);

        // Strip Shared wrappers for comparison since the optimization uses Shared for deduplication
        fn strip_shared(expr: &Expr) -> Expr {
            match expr {
                Expr::Shared(inner) => strip_shared(inner),
                Expr::Seq(exprs) => Expr::Seq(exprs.iter().map(strip_shared).collect()),
                Expr::Choice(exprs) => Expr::Choice(exprs.iter().map(strip_shared).collect()),
                Expr::Quantifier(inner, qtype) => Expr::Quantifier(Box::new(strip_shared(inner)), qtype.clone()),
                other => other.clone(),
            }
        }

        let actual = strip_shared(grammar.group_id_to_expr.get(&0).unwrap());
        assert_eq!(actual, expected_regex, "Expected: {}, got {}", expected_regex, actual);
    }
}
