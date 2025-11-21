use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use bimap::BiBTreeMap;
use crate::finite_automata::{Expr, QuantifierType};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::{GrammarDefinition, GrammarExpr};

pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    let mut optimizer = GrammarOptimizer::new(grammar);
    optimizer.optimize();
}

struct GrammarOptimizer<'a> {
    grammar: &'a mut GrammarDefinition,
    // Map from NonTerminal to its resolved Expr (if it has been converted)
    resolved_nts: HashMap<NonTerminal, Expr>,
}

impl<'a> GrammarOptimizer<'a> {
    fn new(grammar: &'a mut GrammarDefinition) -> Self {
        Self {
            grammar,
            resolved_nts: HashMap::new(),
        }
    }

    fn optimize(&mut self) {
        // 1. Build dependency graph
        let (graph, nt_list) = self.build_dependency_graph();
        
        // 2. Compute SCCs
        let sccs = self.compute_sccs(&graph, &nt_list);
        
        // 3. Process SCCs in topological order (reverse of the order returned by Tarjan's usually, 
        // but we want bottom-up. Tarjan's returns SCCs in reverse topological order, so we iterate as is.)
        // Wait, Tarjan's returns SCCs such that if A -> B, B appears before A? 
        // Tarjan's order is reverse topological (children first). 
        // So we can iterate `sccs` directly.
        
        for scc_indices in sccs {
            let scc_nts: Vec<NonTerminal> = scc_indices.iter().map(|&i| nt_list[i].clone()).collect();
            
            if let Some(resolved_map) = self.try_convert_scc(&scc_nts) {
                // Successful conversion
                for (nt, expr) in resolved_map {
                    self.resolved_nts.insert(nt, expr);
                }
            }
        }
        
        // 4. Rewrite grammar
        self.rewrite_grammar();
    }

    fn build_dependency_graph(&self) -> (Vec<Vec<usize>>, Vec<NonTerminal>) {
        let mut nt_to_idx: HashMap<NonTerminal, usize> = HashMap::new();
        let mut nt_list: Vec<NonTerminal> = Vec::new();
        
        // Collect all defined NonTerminals
        for prod in &self.grammar.productions {
            if !nt_to_idx.contains_key(&prod.lhs) {
                nt_to_idx.insert(prod.lhs.clone(), nt_list.len());
                nt_list.push(prod.lhs.clone());
            }
        }
        
        let mut graph = vec![Vec::new(); nt_list.len()];
        
        for prod in &self.grammar.productions {
            if let Some(&u) = nt_to_idx.get(&prod.lhs) {
                for symbol in &prod.rhs {
                    if let Symbol::NonTerminal(ref target) = symbol {
                        if let Some(&v) = nt_to_idx.get(target) {
                            if !graph[u].contains(&v) {
                                graph[u].push(v);
                            }
                        }
                    }
                }
            }
        }
        
        (graph, nt_list)
    }

    // Tarjan's algorithm for SCC
    fn compute_sccs(&self, graph: &Vec<Vec<usize>>, _nt_list: &Vec<NonTerminal>) -> Vec<Vec<usize>> {
        let n = graph.len();
        let mut ids = vec![-1; n];
        let mut low = vec![0; n];
        let mut on_stack = vec![false; n];
        let mut stack = Vec::new();
        let mut id_counter = 0;
        let mut sccs = Vec::new();

        for i in 0..n {
            if ids[i] == -1 {
                self.dfs(i, graph, &mut ids, &mut low, &mut on_stack, &mut stack, &mut id_counter, &mut sccs);
            }
        }
        
        sccs
    }

    fn dfs(&self, at: usize, graph: &Vec<Vec<usize>>, ids: &mut Vec<i32>, low: &mut Vec<i32>, 
           on_stack: &mut Vec<bool>, stack: &mut Vec<usize>, id_counter: &mut i32, sccs: &mut Vec<Vec<usize>>) {
        stack.push(at);
        on_stack[at] = true;
        ids[at] = *id_counter;
        low[at] = *id_counter;
        *id_counter += 1;

        for &to in &graph[at] {
            if ids[to] == -1 {
                self.dfs(to, graph, ids, low, on_stack, stack, id_counter, sccs);
                low[at] = std::cmp::min(low[at], low[to]);
            } else if on_stack[to] {
                low[at] = std::cmp::min(low[at], ids[to]);
            }
        }

        if ids[at] == low[at] {
            let mut scc = Vec::new();
            while let Some(node) = stack.pop() {
                on_stack[node] = false;
                scc.push(node);
                if node == at { break; }
            }
            sccs.push(scc);
        }
    }

    fn try_convert_scc(&self, scc_nts: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        // Check if SCC is Right-Linear with respect to itself.
        // And all external NonTerminals must be already resolved.
        
        let scc_set: HashSet<&NonTerminal> = scc_nts.iter().collect();
        
        // Build the internal graph for the SCC
        // Nodes: 0..scc_nts.len()
        // Edges: i -> j labeled with Expr
        // Edges: i -> Final labeled with Expr
        
        let mut transitions: Vec<Vec<(usize, Expr)>> = vec![Vec::new(); scc_nts.len()];
        let mut finals: Vec<Expr> = vec![Expr::Choice(vec![]); scc_nts.len()];
        
        let nt_to_local_idx: HashMap<&NonTerminal, usize> = scc_nts.iter().enumerate().map(|(i, nt)| (nt, i)).collect();

        for (i, nt) in scc_nts.iter().enumerate() {
            // Get all productions for this NT
            let productions: Vec<&Production> = self.grammar.productions.iter().filter(|p| &p.lhs == nt).collect();
            
            if productions.is_empty() {
                // If an NT has no productions, it matches nothing? Or it's external?
                // Assuming it matches nothing (dead end).
                continue;
            }

            for prod in productions {
                // Analyze RHS
                // Must be: [Resolved/Terminal]* [SCC_NT]?
                
                let mut prefix_exprs = Vec::new();
                let mut target_scc_idx = None;
                
                for (idx, symbol) in prod.rhs.iter().enumerate() {
                    match symbol {
                        Symbol::Terminal(t) => {
                            prefix_exprs.push(self.get_expr_for_terminal(t));
                        }
                        Symbol::NonTerminal(ref other_nt) => {
                            if let Some(&local_idx) = nt_to_local_idx.get(other_nt) {
                                // It's a reference to the SCC
                                if idx != prod.rhs.len() - 1 {
                                    // Not the last symbol -> Not Right-Linear
                                    return None;
                                }
                                target_scc_idx = Some(local_idx);
                            } else {
                                // External NT
                                if let Some(expr) = self.resolved_nts.get(other_nt) {
                                    prefix_exprs.push(expr.clone());
                                } else {
                                    // Depends on unresolved external NT -> Cannot convert
                                    return None;
                                }
                            }
                        }
                    }
                }
                
                let prefix_expr = ExprBuilder::seq(prefix_exprs);
                
                if let Some(target) = target_scc_idx {
                    transitions[i].push((target, prefix_expr));
                } else {
                    // Transition to Final
                    finals[i] = ExprBuilder::choice(vec![finals[i].clone(), prefix_expr]);
                }
            }
        }
        
        // Solve the system
        let solved = self.solve_regular_system(scc_nts.len(), transitions, finals);
        
        let mut result = HashMap::new();
        for (i, expr) in solved.into_iter().enumerate() {
            result.insert(scc_nts[i].clone(), expr);
        }
        Some(result)
    }
    
    fn get_expr_for_terminal(&self, t: &Terminal) -> Expr {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        let group_id = group_id.expect("Terminal not found in grammar");
        self.grammar.group_id_to_expr.get(group_id).cloned().expect("Expr not found for group_id")
    }

    fn solve_regular_system(&self, n: usize, transitions: Vec<Vec<(usize, Expr)>>, finals: Vec<Expr>) -> Vec<Expr> {
        // Kleene's algorithm / Floyd-Warshall for Regex
        // R[k][i][j] = paths from i to j using only intermediate nodes < k
        
        // Initialize R[0][i][j]
        // We use a 2D array R[i][j] and update it in place (or rather, iteratively)
        
        let mut r: Vec<Vec<Expr>> = vec![vec![Expr::Choice(vec![]); n]; n];
        
        for i in 0..n {
            for &(target, ref expr) in &transitions[i] {
                r[i][target] = ExprBuilder::choice(vec![r[i][target].clone(), expr.clone()]);
            }
            // Add epsilon self-loop? No, standard definition doesn't imply that.
        }
        
        // Iterate k from 0 to n-1 (node being eliminated/considered as intermediate)
        for k in 0..n {
            let r_kk = r[k][k].clone();
            let r_kk_star = ExprBuilder::star(r_kk);
            
            // We need to update all pairs (i, j)
            // But we can't update in place naively because we need values from previous iteration.
            // Actually, standard Floyd-Warshall can be done in place.
            // R_new[i][j] = R[i][j] | R[i][k] R[k][k]* R[k][j]
            
            // To avoid cloning everything, let's just do it.
            // We need to be careful about indices.
            
            let mut next_r = r.clone(); // Expensive clone?
            
            for i in 0..n {
                for j in 0..n {
                    let r_ik = &r[i][k];
                    let r_kj = &r[k][j];
                    
                    let path_through_k = ExprBuilder::seq(vec![
                        r_ik.clone(),
                        r_kk_star.clone(),
                        r_kj.clone()
                    ]);
                    
                    next_r[i][j] = ExprBuilder::choice(vec![r[i][j].clone(), path_through_k]);
                }
            }
            r = next_r;
        }
        
        // Now compute final expressions for each node
        // Expr(i) = Union_j ( R[i][j] * Final[j] )
        // Wait, R[i][j] here means paths from i to j using ANY intermediate nodes (since we ran k from 0 to n-1).
        
        let mut results = Vec::new();
        for i in 0..n {
            let mut choices = Vec::new();
            for j in 0..n {
                let path = ExprBuilder::seq(vec![r[i][j].clone(), finals[j].clone()]);
                choices.push(path);
            }
            results.push(ExprBuilder::choice(choices));
        }
        results
    }

    fn rewrite_grammar(&mut self) {
        // 1. Create new terminals for resolved NTs
        // 2. Replace usages in productions
        // 3. Remove productions of resolved NTs
        
        let mut new_terminals: HashMap<NonTerminal, Terminal> = HashMap::new();
        
        // Allocate group IDs and create Terminals
        let mut next_group_id = self.grammar.group_id_to_expr.keys().max().map(|&x| x + 1).unwrap_or(0);
        
        for (nt, expr) in &self.resolved_nts {
            let name = format!("RX_{}", nt.0); // Use a prefix to avoid collision? Or just reuse name?
            // If we reuse name, we need to be careful if it was already a terminal name (unlikely for NT).
            // Let's just use the NT name but wrapped in RegexName.
            
            // Check for name collision
            let mut final_name = nt.0.clone();
            while self.grammar.regex_name_to_group_id.contains_left(&final_name) {
                final_name.push('_');
            }
            
            self.grammar.regex_name_to_group_id.insert(final_name.clone(), next_group_id);
            self.grammar.group_id_to_expr.insert(next_group_id, expr.clone());
            
            new_terminals.insert(nt.clone(), Terminal::RegexName(final_name));
            next_group_id += 1;
        }
        
        // Rewrite productions
        let mut new_productions = Vec::new();
        
        for prod in &self.grammar.productions {
            if self.resolved_nts.contains_key(&prod.lhs) {
                // This production is now obsolete
                continue;
            }
            
            let mut new_rhs = Vec::new();
            for symbol in &prod.rhs {
                match symbol {
                    Symbol::NonTerminal(nt) => {
                        if let Some(term) = new_terminals.get(nt) {
                            new_rhs.push(Symbol::Terminal(term.clone()));
                        } else {
                            new_rhs.push(symbol.clone());
                        }
                    }
                    _ => new_rhs.push(symbol.clone()),
                }
            }
            
            new_productions.push(Production {
                lhs: prod.lhs.clone(),
                rhs: new_rhs,
            });
        }
        
        self.grammar.productions = new_productions;
    }
}

struct ExprBuilder;

impl ExprBuilder {
    fn seq(exprs: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for e in exprs {
            match e {
                Expr::Seq(sub) => flat.extend(sub),
                Expr::Epsilon => {},
                _ => flat.push(e),
            }
        }
        if flat.is_empty() {
            Expr::Epsilon
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            Expr::Seq(flat)
        }
    }
    
    fn choice(exprs: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for e in exprs {
            match e {
                Expr::Choice(sub) => flat.extend(sub),
                _ => flat.push(e),
            }
        }
        // Deduplicate?
        // flat.sort(); flat.dedup(); // Expr doesn't implement Ord easily? It does in finite_automata.rs
        // But let's skip dedup for now to save time/complexity, unless necessary.
        
        if flat.is_empty() {
            // Choice of nothing is... nothing? Or failure?
            // In regex, empty choice usually matches nothing.
            // But here we might use it as "no path".
            // Let's assume it's valid.
            Expr::Choice(vec![])
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            Expr::Choice(flat)
        }
    }
    
    fn star(expr: Expr) -> Expr {
        match expr {
            Expr::Epsilon => Expr::Epsilon, // Epsilon* = Epsilon
            Expr::Quantifier(inner, QuantifierType::ZeroOrMore) => Expr::Quantifier(inner, QuantifierType::ZeroOrMore), // (A*)* = A*
            _ => Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore),
        }
    }
}
