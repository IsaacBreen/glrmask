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
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Build initial equations: each NT maps to a RegexTerm
        let mut equations: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        
        for prod in &self.grammar.productions {
            let Some(term) = self.production_rhs_to_regex_term(&prod.rhs, &nt_names) else {
                // If any production can't be converted, skip optimization for now
                return;
            };
            let term = Rc::new(term);
            equations.entry(prod.lhs.0.clone())
                .and_modify(|existing| {
                    *existing = Rc::new(RegexTerm::Choice(vec![existing.clone(), term.clone()]));
                })
                .or_insert(term);
        }
        
        // Check for mutual recursion between NTs - if there are cycles, bail out
        // as this can cause exponential blowup during substitution
        if has_mutual_recursion(&equations) {
            eprintln!("DEBUG: Found mutual recursion, bailing out");
            return;
        }
        
        // Build dependency graph and compute elimination order (reverse topological)
        let elimination_order = compute_elimination_order(&equations, &start_nt);
        eprintln!("DEBUG: elimination_order = {:?}", elimination_order);
        
        // Gaussian elimination: eliminate NTs one by one
        for nt_to_eliminate in &elimination_order {
            if nt_to_eliminate == &start_nt {
                continue; // Keep the start NT for last
            }
            
            // Get the equation for this NT
            let Some(nt_eq) = equations.remove(nt_to_eliminate) else {
                continue;
            };
            
            // Solve for this NT (apply Arden's lemma if self-recursive)
            let Some(solved) = solve_single_equation(&nt_eq, nt_to_eliminate) else {
                // Can't solve this NT (non-linear recursion) - put it back and skip
                eprintln!("DEBUG: Can't solve NT {} - non-linear recursion", nt_to_eliminate);
                equations.insert(nt_to_eliminate.clone(), nt_eq);
                continue;
            };
            let solved = Rc::new(solved);
            eprintln!("DEBUG: Solved {} = {:?}", nt_to_eliminate, solved);
            
            // Substitute into all remaining equations that reference this NT
            for (_other_nt, other_eq) in equations.iter_mut() {
                if other_eq.contains_nt(nt_to_eliminate) {
                    *other_eq = Rc::new(substitute_nt(other_eq, nt_to_eliminate, &solved));
                }
            }
        }
        
        // Now solve the start NT
        let Some(start_eq) = equations.remove(&start_nt) else {
            return;
        };
        let Some(solved_start) = solve_single_equation(&start_eq, &start_nt) else {
            return;
        };
        
        // Convert the final RegexTerm to Expr
        let Some(final_expr) = regex_term_to_expr(&solved_start) else {
            return;
        };
        
        // Successfully converted - replace the grammar
        self.replace_grammar_with_single_terminal(final_expr);
    }

    /// Convert a production RHS to a RegexTerm
    fn production_rhs_to_regex_term(&self, rhs: &[Symbol], nt_names: &HashSet<String>) -> Option<RegexTerm> {
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

        terms.map(|ts| RegexTerm::make_seq(ts))
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

/// Check if there is mutual recursion between NTs (cycles in the dependency graph).
/// Mutual recursion causes exponential blowup during Gaussian elimination.
fn has_mutual_recursion(equations: &HashMap<String, Rc<RegexTerm>>) -> bool {
    // Build dependency graph: nt -> set of NTs it references (excluding self)
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for (nt, eq) in equations {
        let mut refs = eq.referenced_nts();
        refs.remove(nt); // Self-recursion is OK (handled by Arden's lemma)
        deps.insert(nt.clone(), refs);
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
}

/// Solve a single equation: X = equation, where equation may reference X.
/// Uses Arden's lemma: X = αX | β => X = α*β (for right recursion)
/// And: X = Xα | β => X = βα* (for left recursion)
fn solve_single_equation(equation: &RegexTerm, nt: &str) -> Option<RegexTerm> {
    if !equation.contains_nt(nt) {
        // No self-reference, just return as-is
        return Some(equation.clone());
    }
    
    // Separate into recursive and non-recursive alternatives
    let (recursive, base) = separate_recursive_alts(equation, nt);
    
    if recursive.is_empty() {
        return Some(RegexTerm::make_choice(base));
    }
    
    // Extract coefficients from recursive alternatives
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
        Rc::new(RegexTerm::make_choice(base))
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

/// Separate a term into recursive and non-recursive alternatives
fn separate_recursive_alts(term: &RegexTerm, nt: &str) -> (Vec<Rc<RegexTerm>>, Vec<Rc<RegexTerm>>) {
    match term {
        RegexTerm::Choice(alts) => {
            let mut recursive = Vec::new();
            let mut base = Vec::new();
            for alt in alts {
                if alt.contains_nt(nt) {
                    recursive.push(alt.clone());
                } else {
                    base.push(alt.clone());
                }
            }
            (recursive, base)
        }
        _ if term.contains_nt(nt) => (vec![Rc::new(term.clone())], vec![]),
        _ => (vec![], vec![Rc::new(term.clone())]),
    }
}

/// Extract (prefix, suffix) from a term of the form "prefix X suffix"
/// Returns None if the term is not linear in X
fn extract_coef(term: &RegexTerm, nt: &str) -> Option<(RegexTerm, RegexTerm)> {
    match term {
        RegexTerm::NtRef(n) if n == nt => {
            Some((RegexTerm::Epsilon, RegexTerm::Epsilon))
        }
        RegexTerm::Seq(parts) => {
            // Find the position of the NT reference
            let mut nt_pos = None;
            for (i, p) in parts.iter().enumerate() {
                if p.contains_nt(nt) {
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

fn is_epsilon(term: &RegexTerm) -> bool {
    match term {
        RegexTerm::Epsilon => true,
        RegexTerm::Seq(s) if s.is_empty() => true,
        _ => false,
    }
}

/// Substitute all occurrences of `nt` in `term` with `replacement`
fn substitute_nt(term: &RegexTerm, nt: &str, replacement: &Rc<RegexTerm>) -> RegexTerm {
    if !term.contains_nt(nt) {
        return term.clone();
    }
    
    match term {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => term.clone(),
        RegexTerm::NtRef(n) if n == nt => (**replacement).clone(),
        RegexTerm::NtRef(_) => term.clone(),
        RegexTerm::Seq(parts) => {
            let new_parts: Vec<Rc<RegexTerm>> = parts.iter()
                .map(|p| {
                    if p.contains_nt(nt) {
                        Rc::new(substitute_nt(p, nt, replacement))
                    } else {
                        p.clone()
                    }
                })
                .collect();
            RegexTerm::make_seq(new_parts)
        }
        RegexTerm::Choice(alts) => {
            let new_alts: Vec<Rc<RegexTerm>> = alts.iter()
                .map(|a| {
                    if a.contains_nt(nt) {
                        Rc::new(substitute_nt(a, nt, replacement))
                    } else {
                        a.clone()
                    }
                })
                .collect();
            RegexTerm::make_choice(new_alts)
        }
        RegexTerm::Star(inner) => {
            if inner.contains_nt(nt) {
                RegexTerm::make_star(Rc::new(substitute_nt(inner, nt, replacement)))
            } else {
                term.clone()
            }
        }
    }
}

/// Convert a RegexTerm (with no NT references) to an Expr
fn regex_term_to_expr(term: &RegexTerm) -> Option<Expr> {
    match term {
        RegexTerm::Epsilon => Some(Expr::Epsilon),
        RegexTerm::Concrete(e) => Some(e.clone()),
        RegexTerm::NtRef(_) => None, // Should not have NT refs at this point
        RegexTerm::Seq(parts) => {
            let exprs: Option<Vec<Expr>> = parts.iter()
                .map(|p| regex_term_to_expr(p))
                .collect();
            exprs.map(|es| make_seq(es))
        }
        RegexTerm::Choice(alts) => {
            let exprs: Option<Vec<Expr>> = alts.iter()
                .map(|a| regex_term_to_expr(a))
                .collect();
            exprs.map(|es| make_choice(es))
        }
        RegexTerm::Star(inner) => {
            regex_term_to_expr(inner).map(|e| {
                Expr::Quantifier(Box::new(e), QuantifierType::ZeroOrMore)
            })
        }
    }
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

/// Simplify an expression
fn simplify_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Seq(exprs) => {
            let mut simplified = Vec::new();
            for e in exprs {
                let e = simplify_expr(e);
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
                .map(simplify_expr)
                .flat_map(|e| match e {
                    Expr::Choice(inner) => inner,
                    other => vec![other],
                })
                .collect();
            if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else {
                Expr::Choice(simplified)
            }
        }
        Expr::Quantifier(inner, qtype) => {
            let inner = simplify_expr(*inner);
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
        Expr::Shared(inner) => simplify_expr((*inner).clone()),
        other => other,
    }
}


#[cfg(test)]
mod tests {
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

        GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap()
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

    #[test]
    fn test_diff_grammar_optimization_performance() {
        let mut n = 100;
        let mut t_base = 0.0;

        loop {
            let mut grammar = build_diff_grammar(n);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            let duration = start.elapsed();
            t_base = duration.as_secs_f64();

            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n);

            if t_base > 0.005 {
                break;
            }
            n *= 2;

            if n > 10_000 {
                println!("Warning: reached n={} without exceeding time threshold (got {:.4}s). Skipping perf check.", n, t_base);
                return;
            }
        }

        println!("Baseline: n={}, time={:.4}s", n, t_base);

        let n_scaled = n * 3;
        let mut grammar = build_diff_grammar(n_scaled);
        let start = std::time::Instant::now();
        optimize_grammar(&mut grammar);
        let duration = start.elapsed();
        let t_scaled = duration.as_secs_f64();

        assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n_scaled);

        let ratio = t_scaled / t_base;
        println!("Scaled: n={}, time={:.4}s", n_scaled, t_scaled);
        println!("Ratio T({})/T({}): {:.2}", n_scaled, n, ratio);

        assert!(ratio < 6.0, "Performance scaling looks worse than linear (ratio {:.2})", ratio);
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
}
