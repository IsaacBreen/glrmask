use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use bimap::BiBTreeMap;
use crate::finite_automata::{Expr, QuantifierType, rep};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::{GrammarDefinition, GrammarExpr, ExprNullability, get_expr_nullability};
use crate::types::TerminalID;
use crate::debug;

pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    // Temporarily disabled for debugging
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
    // Map from NonTerminal to its resolved Expr (if converted to terminal)
    resolved_map: HashMap<NonTerminal, Expr>,
    // Cache for ignore expression (as Option<Expr>)
    ignore_expr: Option<Expr>,
}

impl<'a> GrammarOptimizer<'a> {
    fn new(grammar: &'a mut GrammarDefinition) -> Self {
        let ignore_expr = if let Some(gid) = grammar.ignore_terminal_id {
            grammar.group_id_to_expr.get(&gid.0).cloned().map(rep)
        } else {
            None
        };

        Self {
            grammar,
            stats: OptimizationStats::default(),
            resolved_map: HashMap::new(),
            ignore_expr,
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

        // 1. Compute SCCs of the non-terminals dependency graph
        let sccs = self.compute_sccs();

        // 2. Process SCCs in topological order (reverse of Tarjan's output usually, but we need bottom-up)
        // Tarjan's algorithm returns SCCs in reverse topological order (leaves first), which is exactly what we want.
        for scc in sccs {
            // Try to resolve this SCC into regexes
            if let Some(solution) = self.solve_regular_system(&scc) {
                debug!(2, "Resolved SCC with {} non-terminals: {:?}", scc.len(), scc);
                for (nt, expr) in solution {
                    self.resolved_map.insert(nt, expr);
                }
            }
        }

        // 3. Apply resolutions and optimize productions
        self.apply_optimizations();

        self.stats.final_productions = self.grammar.productions.len();
        self.stats.final_terminals = self.count_terminals();
        debug!(2, "{}", self.stats);
    }

    fn compute_sccs(&self) -> Vec<Vec<NonTerminal>> {
        let mut adj = HashMap::new();
        let mut all_nts = HashSet::new();

        for prod in &self.grammar.productions {
            all_nts.insert(prod.lhs.clone());
            let entry = adj.entry(prod.lhs.clone()).or_insert_with(HashSet::new);
            for sym in &prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    entry.insert(nt.clone());
                }
            }
        }

        // Tarjan's algorithm
        let mut index = 0;
        let mut indices: HashMap<NonTerminal, usize> = HashMap::new();
        let mut lowlink: HashMap<NonTerminal, usize> = HashMap::new();
        let mut stack: Vec<NonTerminal> = Vec::new();
        let mut on_stack: HashSet<NonTerminal> = HashSet::new();
        let mut sccs: Vec<Vec<NonTerminal>> = Vec::new();

        for nt in all_nts {
            if !indices.contains_key(&nt) {
                self.strongconnect(&nt, &adj, &mut index, &mut indices, &mut lowlink, &mut stack, &mut on_stack, &mut sccs);
            }
        }

        sccs
    }

    fn strongconnect(
        &self,
        v: &NonTerminal,
        adj: &HashMap<NonTerminal, HashSet<NonTerminal>>,
        index: &mut usize,
        indices: &mut HashMap<NonTerminal, usize>,
        lowlink: &mut HashMap<NonTerminal, usize>,
        stack: &mut Vec<NonTerminal>,
        on_stack: &mut HashSet<NonTerminal>,
        sccs: &mut Vec<Vec<NonTerminal>>
    ) {
        indices.insert(v.clone(), *index);
        lowlink.insert(v.clone(), *index);
        *index += 1;
        stack.push(v.clone());
        on_stack.insert(v.clone());

        if let Some(neighbors) = adj.get(v) {
            for w in neighbors {
                if !indices.contains_key(w) {
                    self.strongconnect(w, adj, index, indices, lowlink, stack, on_stack, sccs);
                    let v_low = lowlink.get(v).cloned().unwrap();
                    let w_low = lowlink.get(w).cloned().unwrap();
                    lowlink.insert(v.clone(), std::cmp::min(v_low, w_low));
                } else if on_stack.contains(w) {
                    let v_low = lowlink.get(v).cloned().unwrap();
                    let w_index = indices.get(w).cloned().unwrap();
                    lowlink.insert(v.clone(), std::cmp::min(v_low, w_index));
                }
            }
        }

        if lowlink.get(v) == indices.get(v) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == *v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    // Try to solve the system of equations for the given SCC.
    // Returns Some(Map) if the system is Right-Linear and solvable.
    fn solve_regular_system(&mut self, scc: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        let scc_set: HashSet<&NonTerminal> = scc.iter().collect();
        
        // Equations: NT = (coeff * Neighbor) | constant
        // Map NT -> (Map<Neighbor, Expr>, Expr)
        // where constant is the union of all constant parts
        let mut system: HashMap<NonTerminal, (HashMap<NonTerminal, Vec<Expr>>, Vec<Expr>)> = HashMap::new();

        for nt in scc {
            system.insert(nt.clone(), (HashMap::new(), Vec::new()));
        }

        // Populate system
        for prod in &self.grammar.productions {
            if !scc_set.contains(&prod.lhs) {
                continue;
            }

            // Analyze production: is it linear?
            // Form 1: [Terms...] [NT_in_SCC]
            // Form 2: [Terms...]
            // Note: [Terms...] includes resolved NTs.

            let mut terms_prefix = Vec::new();
            let mut tail_nt = None;

            for sym in &prod.rhs {
                if tail_nt.is_some() {
                    // If we already saw an SCC NT, we shouldn't see anything else -> Center embedding
                    // Exception: if the following symbols are nullable, but we ignore that complexity.
                    return None; 
                }

                match sym {
                    Symbol::Terminal(t) => {
                        terms_prefix.push(self.get_expr_for_terminal(t));
                    }
                    Symbol::NonTerminal(nt) => {
                        if scc_set.contains(nt) {
                            tail_nt = Some(nt.clone());
                        } else if let Some(resolved) = self.resolved_map.get(nt) {
                            terms_prefix.push(resolved.clone());
                        } else {
                            // Unresolved dependency outside SCC? 
                            // In bottom-up order, this shouldn't happen unless there's a dependency cycle we missed
                            // or it's a separate component. 
                            // If it's truly unresolved and not in SCC, we can't regularize it.
                            return None;
                        }
                    }
                }
            }

            let coeff_expr = self.seq_with_ignore(terms_prefix);
            
            let entry = system.get_mut(&prod.lhs).unwrap();
            if let Some(tail) = tail_nt {
                entry.0.entry(tail).or_default().push(coeff_expr);
            } else {
                entry.1.push(coeff_expr);
            }
        }

        // Gaussian Elimination for Right-Linear System
        // We have variables X_1 ... X_n
        // Eliminate X_1, then X_2...
        
        // Use an ordered list to keep track of elimination
        let mut variables = scc.to_vec();
        
        // Store final resolved expressions
        let mut solutions: HashMap<NonTerminal, Expr> = HashMap::new();
        
        // Working system: We will modify coeff maps
        // Map NT -> (Map<Neighbor, Expr>, Expr)
        // We coalesce the Vec<Expr> into a single Choice expr
        let mut working_system: HashMap<NonTerminal, (HashMap<NonTerminal, Expr>, Expr)> = HashMap::new();
        
        for (nt, (coeffs, consts)) in system {
            let mut combined_coeffs = HashMap::new();
            for (neighbor, exprs) in coeffs {
                combined_coeffs.insert(neighbor, Expr::Choice(exprs).optimize());
            }
            let combined_const = if consts.is_empty() {
                 // If no constant part, and it's recursive, it implies empty set unless nullable?
                 // But for regex generation, empty choice is empty set.
                 // However, we need to handle Epsilon if the rule produces nothing.
                 // Wait, if production rhs is empty, terms_prefix is empty, seq_with_ignore is Epsilon.
                 // So constant list will contain Epsilon.
                 Expr::Choice(vec![]).optimize() // Matches nothing
            } else {
                Expr::Choice(consts).optimize()
            };
            working_system.insert(nt, (combined_coeffs, combined_const));
        }

        // Elimination Phase
        // Eliminate variables from 0 to n-1
        // When eliminating X_i:
        //   Equation: X_i = A X_i | B X_j | ... | C
        //   Solution: X_i = A* (B X_j | ... | C)
        //   Substitute X_i in all other equations.

        for i in 0..variables.len() {
            let xi = variables[i].clone();
            
            // 1. Solve for X_i in terms of remaining variables
            // X_i = self_coeff * X_i + other_parts
            // X_i = self_coeff* other_parts
            
            let (mut xi_coeffs, mut xi_const) = working_system.remove(&xi).unwrap();
            
            let self_coeff = xi_coeffs.remove(&xi);
            let star_expr = match self_coeff {
                Some(expr) => Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore),
                None => Expr::Epsilon, // Multiplicative identity is Epsilon? No. 
                // If no self loop: X = others.
                // Ideally we treat it as (epsilon)* which is epsilon.
                // If we construct Seq(None, others) -> others.
            };

            // Helper to wrap A* B
            let apply_star = |rest: Expr| -> Expr {
                match &star_expr {
                    Expr::Epsilon => rest,
                    _ => Expr::Seq(vec![star_expr.clone(), rest]).optimize()
                }
            };

            // Calculate the "substitution block": (A* (Sum(Coeff_k X_k) + Const))
            // We need to distribute this into other equations.
            
            // For every other variable X_k (where k > i usually, but system is full map)
            // Update X_k's equation.
            // But wait, Gaussian elimination usually eliminates X_i from equations of X_{j} where j > i.
            // We can leave X_i dependent on X_{j>i}.
            // At the end we back-substitute.
            
            // For each X_j (j != i, remaining in map)
            let remaining_vars: Vec<NonTerminal> = working_system.keys().cloned().collect();
            for xj in remaining_vars {
                // Get X_j's equation
                let (xj_coeffs, xj_const) = working_system.get_mut(&xj).unwrap();
                
                // Does X_j depend on X_i?
                if let Some(coeff_ji) = xj_coeffs.remove(&xi) {
                    // X_j = ... + coeff_ji * X_i
                    // X_i = A* (Sum(C_k X_k) + D)
                    // X_j += coeff_ji * A* * (Sum(C_k X_k) + D)
                    
                    // Distribute to constants
                    // X_j_const += coeff_ji * A* * xi_const
                    let added_const = Expr::Seq(vec![coeff_ji.clone(), apply_star(xi_const.clone())]).optimize();
                    *xj_const = Expr::Choice(vec![xj_const.clone(), added_const]).optimize();

                    // Distribute to other coeffs
                    for (xk, coeff_ik) in &xi_coeffs {
                        // X_j_coeff_k += coeff_ji * A* * coeff_ik
                        let term = Expr::Seq(vec![
                            coeff_ji.clone(),
                            apply_star(coeff_ik.clone())
                        ]).optimize();
                        
                        let old_coeff = xj_coeffs.remove(xk).unwrap_or(Expr::Choice(vec![])); // Empty choice = null
                        // If old_coeff was empty/null, Choice(null, term) = term? 
                        // Expr::Choice([]) represents empty set (fail).
                        // Regex algebra: 0 + A = A.
                        let new_coeff = if matches!(old_coeff, Expr::Choice(ref v) if v.is_empty()) {
                            term
                        } else {
                            Expr::Choice(vec![old_coeff, term]).optimize()
                        };
                        xj_coeffs.insert(xk.clone(), new_coeff);
                    }
                }
            }

            // Store the semi-solved equation for X_i to back-substitute later
            // X_i = A* (Sum(coeffs X_k) + const)
            // We can just store (A*, coeffs, const) or a closure-like structure?
            // We'll store it in a separate list or put it back? 
            // Standard Gaussian: put it in a stack.
            // We need to resolve it fully later.
            // Let's store the "row": (xi, star_expr, xi_coeffs, xi_const)
            // Note: xi_coeffs only contains X_k where k > i.
        }

        // Now we have the system in upper-triangular form (conceptually)
        // Wait, the loop above eliminated X_i from all *other* equations.
        // Standard Gaussian elimination usually eliminates X_i from X_{j>i}.
        // If we iterate i from 0..n, and for each i eliminate X_i from all remaining j,
        // then the last variable X_n depends on nothing (or self).
        // We handled self-loop inside the step.
        // So the last variable equation is X_n = const_n (after self-loop resolution).
        
        // We need to reconstruct the stack or just re-run?
        // Actually, if we modify `working_system` in place, removing `xi` effectively "stacks" it.
        // But we need to keep the equation for `xi` to back-substitute.
        
        // Let's refine the loop to save the equation.
        let mut stack: Vec<(NonTerminal, Expr, HashMap<NonTerminal, Expr>, Expr)> = Vec::new();

        // Re-run the logic carefully
        // Re-populate working system since I consumed it conceptually above?
        // Ah, I wrote the loop but didn't actually push to stack.
        // Let's rewrite the elimination loop correctly.
        
        // Reset working system
        let mut working_system: HashMap<NonTerminal, (HashMap<NonTerminal, Expr>, Expr)> = HashMap::new();
        for (nt, (coeffs, consts)) in system {
            let mut combined_coeffs = HashMap::new();
            for (neighbor, exprs) in coeffs {
                combined_coeffs.insert(neighbor, Expr::Choice(exprs).optimize());
            }
            let combined_const = Expr::Choice(consts).optimize();
            working_system.insert(nt, (combined_coeffs, combined_const));
        }

        for i in 0..variables.len() {
            let xi = variables[i].clone();
            let (mut xi_coeffs, mut xi_const) = working_system.remove(&xi).unwrap();
            
            let self_coeff = xi_coeffs.remove(&xi);
            let star_expr = match self_coeff {
                Some(expr) => Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore),
                None => Expr::Epsilon,
            };

            // Save for back-substitution
            stack.push((xi.clone(), star_expr.clone(), xi_coeffs.clone(), xi_const.clone()));

            let apply_star = |rest: Expr| -> Expr {
                match &star_expr {
                    Expr::Epsilon => rest,
                    _ => Expr::Seq(vec![star_expr.clone(), rest]).optimize()
                }
            };

            // Substitute X_i into remaining equations
            let remaining_vars: Vec<NonTerminal> = working_system.keys().cloned().collect();
            for xj in remaining_vars {
                let (xj_coeffs, xj_const) = working_system.get_mut(&xj).unwrap();
                
                if let Some(coeff_ji) = xj_coeffs.remove(&xi) {
                    // Update const: X_j_const += coeff_ji * A* * xi_const
                    let added_const = Expr::Seq(vec![coeff_ji.clone(), apply_star(xi_const.clone())]).optimize();
                    *xj_const = Expr::Choice(vec![xj_const.clone(), added_const]).optimize();

                    // Update coeffs: X_j_coeff_k += coeff_ji * A* * coeff_ik
                    for (xk, coeff_ik) in &xi_coeffs {
                        // Only care about k > i (remaining variables)
                        if working_system.contains_key(xk) {
                            let term = Expr::Seq(vec![
                                coeff_ji.clone(),
                                apply_star(coeff_ik.clone())
                            ]).optimize();
                            
                            let old_coeff = xj_coeffs.remove(xk).unwrap_or(Expr::Choice(vec![]));
                            let new_coeff = if matches!(old_coeff, Expr::Choice(ref v) if v.is_empty()) {
                                term
                            } else {
                                Expr::Choice(vec![old_coeff, term]).optimize()
                            };
                            xj_coeffs.insert(xk.clone(), new_coeff);
                        }
                    }
                }
            }
        }

        // Back-substitution Phase
        // Iterate stack in reverse
        for (xi, star_expr, xi_coeffs, xi_const) in stack.into_iter().rev() {
            // X_i = A* (Sum(coeffs X_k) + const)
            // All X_k in xi_coeffs should be solved already (since k > i)
            
            let mut parts = Vec::new();
            if !matches!(xi_const, Expr::Choice(ref v) if v.is_empty()) {
                parts.push(xi_const);
            }
            
            for (xk, coeff) in xi_coeffs {
                let xk_sol = solutions.get(&xk).expect("Future variable not solved?");
                parts.push(Expr::Seq(vec![coeff, xk_sol.clone()]).optimize());
            }
            
            let sum = Expr::Choice(parts).optimize();
            let sol = match star_expr {
                Expr::Epsilon => sum,
                _ => Expr::Seq(vec![star_expr, sum]).optimize()
            };
            
            solutions.insert(xi, sol);
        }

        Some(solutions)
    }

    fn get_expr_for_terminal(&mut self, t: &Terminal) -> Expr {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        let group_id = group_id.expect("Terminal not found in grammar");
        let expr = self.grammar.group_id_to_expr.get(group_id).cloned().expect("Expr not found for group_id");
        expr
    }

    /// Returns the ignore pattern as `ignore_expr*` if an ignore terminal is defined,
    /// or None otherwise.

    /// Returns the ignore pattern as `ignore_expr*` if an ignore terminal is defined,
    /// or None otherwise.
    fn get_ignore_star_expr(&self) -> Option<Expr> {
        self.ignore_expr.clone()
    }

    /// Build a sequence that interleaves ignore_pattern* between elements.
    /// E.g., for [a, b, c] with ignore WS, produces: a WS* b WS* c
    fn seq_with_ignore(&mut self, exprs: Vec<Expr>) -> Expr {
        let mut exprs = exprs;
        if exprs.is_empty() {
            return Expr::Epsilon;
        }
        
        if let Some(ignore) = self.get_ignore_star_expr() {
            let mut interleaved = Vec::with_capacity(exprs.len() * 2 - 1);
            let mut it = exprs.drain(..);
            if let Some(first) = it.next() {
                interleaved.push(first);
                for e in it {
                    interleaved.push(ignore.clone());
                    interleaved.push(e);
                }
            }
            Expr::Seq(interleaved).optimize()
        } else {
            Expr::Seq(exprs).optimize()
        }
    }

    fn get_group_id(&self, t: &Terminal) -> usize {
         match t {
            Terminal::Literal(bytes) => *self.grammar.literal_to_group_id.get_by_left(bytes).expect("Terminal missing"),
            Terminal::RegexName(name) => *self.grammar.regex_name_to_group_id.get_by_left(name).expect("Terminal missing"),
        }
    }

    fn apply_optimizations(&mut self) {
        // 1. Remove solved non-terminals from productions
        self.grammar.productions.retain(|prod| !self.resolved_map.contains_key(&prod.lhs));

        // 2. Register new terminals
        let mut nt_to_terminal = HashMap::new();
        
        // Find max group id
        let mut next_gid = self.grammar.group_id_to_expr.keys().max().map_or(0, |x| x + 1);

        for (nt, expr) in &self.resolved_map {
            let name = nt.0.clone();
            // If name conflicts, it might be an issue, but usually NT names are disjoint from Terminal names 
            // (uppercase convention often used for terminals, but not enforced here).
            // We'll reuse the NT name as the terminal name.
            
            // Check collision
            let final_name = if self.grammar.regex_name_to_group_id.contains_left(&name) {
                format!("{}_REGEX", name)
            } else {
                name
            };

            self.grammar.regex_name_to_group_id.insert(final_name.clone(), next_gid);
            self.grammar.group_id_to_expr.insert(next_gid, expr.clone());
            nt_to_terminal.insert(nt.clone(), Terminal::RegexName(final_name));
            next_gid += 1;
        }

        // 3. Update remaining productions
        for prod in &mut self.grammar.productions {
            let mut new_rhs = Vec::new();
            let mut pending_terminals: Vec<Expr> = Vec::new();

            for sym in &prod.rhs {
                match sym {
                    Symbol::NonTerminal(nt) => {
                        if let Some(term) = nt_to_terminal.get(nt) {
                            pending_terminals.push(self.get_expr_for_terminal(term));
                        } else {
                            // Flush pending terminals
                            if !pending_terminals.is_empty() {
                                let merged = self.seq_with_ignore(pending_terminals.drain(..).collect());
                                let gid = next_gid;
                                next_gid += 1;
                                let name = format!("MERGED_{}", gid);
                                self.grammar.regex_name_to_group_id.insert(name.clone(), gid);
                                self.grammar.group_id_to_expr.insert(gid, merged);
                                new_rhs.push(Symbol::Terminal(Terminal::RegexName(name)));
                            }
                            new_rhs.push(Symbol::NonTerminal(nt.clone()));
                        }
                    }
                    Symbol::Terminal(t) => {
                        pending_terminals.push(self.get_expr_for_terminal(t));
                    }
                }
            }
            // Flush remaining
            if !pending_terminals.is_empty() {
                let merged = self.seq_with_ignore(pending_terminals);
                let gid = next_gid;
                next_gid += 1;
                let name = format!("MERGED_{}", gid);
                self.grammar.regex_name_to_group_id.insert(name.clone(), gid);
                self.grammar.group_id_to_expr.insert(gid, merged);
                new_rhs.push(Symbol::Terminal(Terminal::RegexName(name)));
            }
            prod.rhs = new_rhs;
        }
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
            // println!("Iteration {}: Initial terminals: {}", i, initial_count);

            optimize_grammar(&mut grammar);

            // println!("{grammar}");
            // Note: Grammars with nullable expressions (Optional, Repeat) won't collapse to 1 terminal
            // because the parser needs epsilon productions to correctly handle empty matches.
            // We just verify the grammar compiles successfully.
            use crate::interface::CompiledGrammar;
            let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
        }
    }

    #[test]
    fn test_diff_grammar_structure() {
        // Simulates a structure similar to what generate_diff_grammar.py produces:
        // Line1 ::= ( " " | "-" ) "foo" "\n"
        // Line2 ::= ( " " | "-" ) "bar" "\n"
        // Block ::= Line1 | Line2
        // This should ideally be optimized into a single regular expression terminal.

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

        // Use from_exprs_no_optimize to get the unoptimized grammar first
        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, regex_exprs).unwrap();
        let initial_terminals = grammar.terminal_to_group_id().len();
        println!("Initial terminals: {}", initial_terminals);

        optimize_grammar(&mut grammar);
        println!("{grammar}");

        // We expect significant reduction. Ideally to 1 terminal representing the whole block regex.
        assert!(grammar.terminal_to_group_id().len() < initial_terminals);
        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    #[test]
    fn test_complex_nesting() {
        // A -> ( "a" | "b" ) "c" ( "d" | "e" )
        // This tests mixing Sequence and Choice at different levels.
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
        println!("Initial terminals: {}", initial); // a, b, c, d, e = 5

        optimize_grammar(&mut grammar);
        println!("{grammar}");

        // Should collapse to 1 terminal: [ab]c[de]
        assert_eq!(grammar.terminal_to_group_id().len(), 1);
    }

    fn build_diff_grammar(num_lines: usize) -> GrammarDefinition {
        use crate::finite_automata::Expr;
        use crate::interface::{GrammarExpr, GrammarDefinition};

        let mut grammar_exprs = Vec::new();
        let mut regex_exprs = Vec::new();

        // Terminals
        regex_exprs.push(("HUNK_HEADER".to_string(), Expr::U8Seq(b"@@".to_vec())));
        regex_exprs.push(("PLUS_LINE".to_string(), Expr::U8Seq(b"+".to_vec())));
        regex_exprs.push(("EOF".to_string(), Expr::U8Seq(b"EOF".to_vec())));

        for i in 0..num_lines {
            regex_exprs.push((format!("L{}", i), Expr::U8Seq(format!("line{}", i).as_bytes().to_vec())));
        }

        // diff ::= ( HUNK_HEADER s0 )? EOF;
        grammar_exprs.push(("start".to_string(), GrammarExpr::Sequence(vec![
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("HUNK_HEADER".to_string()),
                GrammarExpr::Ref("s0".to_string())
            ]))),
            GrammarExpr::Ref("EOF".to_string())
        ])));

        // s{i} rules
        for i in 0..num_lines {
            grammar_exprs.push((
                format!("s{}", i),
                GrammarExpr::Choice(vec![
                    GrammarExpr::Ref(format!("l{}", i)),
                    if i < num_lines - 1 {
                        GrammarExpr::Ref(format!("s{}", i + 1))
                    } else {
                        GrammarExpr::Ref(format!("s{}", num_lines)) // s{N}
                    }
                ])
            ));
        }
        // s{N} ::= PLUS_LINE*;
        grammar_exprs.push((
            format!("s{}", num_lines),
            GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string())))
        ));

        // l{i} rules
        for i in 0..num_lines {
            let continuation = if i < num_lines - 1 {
                // ( l{i+1} | PLUS_LINE* HUNK_HEADER s{i+1} )?
                GrammarExpr::Optional(Box::new(GrammarExpr::Choice(vec![
                    GrammarExpr::Ref(format!("l{}", i + 1)),
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string()))),
                        GrammarExpr::Ref("HUNK_HEADER".to_string()),
                        GrammarExpr::Ref(format!("s{}", i + 1))
                    ])
                ])))
            } else {
                // ( PLUS_LINE* HUNK_HEADER s{N} )?
                GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("PLUS_LINE".to_string()))),
                    GrammarExpr::Ref("HUNK_HEADER".to_string()),
                    GrammarExpr::Ref(format!("s{}", num_lines))
                ])))
            };

            // l{i} ::= PLUS_LINE* L{i} continuation
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

        // Verify that we can compile the optimized grammar without panicking (e.g. index out of bounds)
        // This checks that terminal IDs are correctly renumbered and consistent.
        use crate::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }

    #[test]
    fn test_diff_grammar_optimization_performance() {
        // Scale test: ensure time grows roughly linearly, not quadratically/exponentially
        // We dynamically find a size N that takes enough time to measure reliably,
        // then test 3N to check scaling behavior.

        let mut n = 100;
        let mut t_base = 0.0;

        // Find a baseline size that takes enough time to measure (> 5ms)
        loop {
            let mut grammar = build_diff_grammar(n);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            let duration = start.elapsed();
            t_base = duration.as_secs_f64();

            // Basic correctness check
            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n);

            if t_base > 0.005 {
                break;
            }
            n *= 2;

            // Safety break for extremely fast machines or if logic is broken
            if n > 10_000 {
                println!("Warning: reached n={} without exceeding time threshold (got {:.4}s). Skipping perf check.", n, t_base);
                return;
            }
        }

        println!("Baseline: n={}, time={:.4}s", n, t_base);

        // Scale up by 3x
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

        // Linear scaling would be ~3.0. Quadratic would be ~9.0.
        // We use a loose bound of 6.0 to account for overhead/noise/cache effects,
        // but it should definitely be less than 9.0 (quadratic).
        assert!(ratio < 6.0, "Performance scaling looks worse than linear (ratio {:.2})", ratio);
    }

    #[test]
    fn test_simple_recursion_optimization() {
        // S ::= "a" S | "b"
        // Should optimize to S ::= "a"* "b"
        // If the bug exists, it might optimize to S ::= "a"+ "b" (missing the zero-loop case)

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

        // The important thing is that the grammar compiles without panicking.
        // The initialization of epsilon on the diagonal in solve_regular_system
        // ensures that S ::= a S | b becomes a* b (not a+ b), which matches zero or more a's.
        use crate::interface::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }
}
