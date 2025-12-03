use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use bimap::BiBTreeMap;
use crate::finite_automata::{Expr, QuantifierType};
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
    // Map from NonTerminal to its resolved Expr (if it has been converted to a regex)
    solved_nt: HashMap<NonTerminal, Expr>,
    // Map from NonTerminal to integer ID for graph algorithms
    nt_to_id: HashMap<NonTerminal, usize>,
    id_to_nt: Vec<NonTerminal>,
}

impl<'a> GrammarOptimizer<'a> {
    fn new(grammar: &'a mut GrammarDefinition) -> Self {
        Self {
            grammar,
            stats: OptimizationStats::default(),
            solved_nt: HashMap::new(),
            nt_to_id: HashMap::new(),
            id_to_nt: Vec::new(),
        }
    }

    fn optimize(&mut self) {
        self.stats.initial_productions = self.grammar.productions.len();
        self.stats.initial_terminals = self.count_terminals();
        
        debug!(2, "Starting optimization: {} productions, {} terminals", self.stats.initial_productions, self.stats.initial_terminals);

        // 1. Build Dependency Graph & Identify SCCs
        self.build_nt_mapping();
        debug!(2, "Built NT mapping: {} NTs", self.id_to_nt.len());
        
        let sccs = self.compute_sccs();
        debug!(2, "Computed {} SCCs", sccs.len());

        // 2. Process SCCs bottom-up
        for (i, scc) in sccs.iter().enumerate() {
            debug!(3, "Processing SCC {}: {:?}", i, scc.iter().map(|&id| &self.id_to_nt[id].0).collect::<Vec<_>>());
            self.process_scc(&scc);
        }

        // 3. Rebuild Grammar
        debug!(2, "Rebuilding grammar...");
        self.rebuild_grammar();

        self.stats.final_productions = self.grammar.productions.len();
        self.stats.final_terminals = self.count_terminals();
        debug!(2, "{}", self.stats);
    }

    fn count_terminals(&self) -> usize {
        self.grammar.regex_name_to_group_id.len() + 
        self.grammar.literal_to_group_id.len() + 
        self.grammar.external_name_to_group_id.len()
    }

    fn build_nt_mapping(&mut self) {
        let mut id = 0;
        // Collect all NTs from productions (LHS and RHS)
        let mut seen = HashSet::new();
        
        // Add LHSs first
        for prod in &self.grammar.productions {
            if seen.insert(prod.lhs.clone()) {
                self.nt_to_id.insert(prod.lhs.clone(), id);
                self.id_to_nt.push(prod.lhs.clone());
                id += 1;
            }
        }
        
        // Add RHSs
        for prod in &self.grammar.productions {
            for sym in &prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    if seen.insert(nt.clone()) {
                        self.nt_to_id.insert(nt.clone(), id);
                        self.id_to_nt.push(nt.clone());
                        id += 1;
                    }
                }
            }
        }
    }

    fn compute_sccs(&self) -> Vec<Vec<usize>> {
        let num_nodes = self.id_to_nt.len();
        let mut adj = vec![HashSet::new(); num_nodes];

        for prod in &self.grammar.productions {
            if let Some(&u) = self.nt_to_id.get(&prod.lhs) {
                for sym in &prod.rhs {
                    if let Symbol::NonTerminal(nt) = sym {
                        if let Some(&v) = self.nt_to_id.get(nt) {
                            adj[u].insert(v);
                        }
                    }
                }
            }
        }

        // Tarjan's Algorithm
        let mut visited = vec![false; num_nodes];
        let mut stack = Vec::new();
        let mut on_stack = vec![false; num_nodes];
        let mut ids = vec![-1; num_nodes];
        let mut low = vec![-1; num_nodes];
        let mut id_counter = 0;
        let mut sccs = Vec::new();

        for i in 0..num_nodes {
            if !visited[i] {
                self.dfs(i, &adj, &mut visited, &mut stack, &mut on_stack, &mut ids, &mut low, &mut id_counter, &mut sccs);
            }
        }

        // Tarjan's returns SCCs in reverse topological order (leaves first).
        // This is exactly what we want for bottom-up processing.
        sccs
    }

    fn dfs(&self, at: usize, adj: &Vec<HashSet<usize>>, visited: &mut Vec<bool>, stack: &mut Vec<usize>, on_stack: &mut Vec<bool>, ids: &mut Vec<isize>, low: &mut Vec<isize>, id_counter: &mut isize, sccs: &mut Vec<Vec<usize>>) {
        stack.push(at);
        on_stack[at] = true;
        visited[at] = true;
        ids[at] = *id_counter;
        low[at] = *id_counter;
        *id_counter += 1;

        for &to in &adj[at] {
            if !visited[to] {
                self.dfs(to, adj, visited, stack, on_stack, ids, low, id_counter, sccs);
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

    fn process_scc(&mut self, scc_ids: &[usize]) {
        let scc_nts: HashSet<NonTerminal> = scc_ids.iter().map(|&id| self.id_to_nt[id].clone()).collect();
        
        // Check if SCC is Right-Linear or Left-Linear
        // For now, we only implement Right-Linear solving as it's most common for "tail recursion" optimization
        // and general regex construction.
        
        if let Some(equations) = self.try_build_right_linear_system(&scc_nts) {
            debug!(3, "SCC {:?} is Right-Linear. Solving...", scc_nts);
            let solution = self.solve_system(equations, &scc_nts);
            for (nt, expr) in solution {
                self.solved_nt.insert(nt, expr);
            }
        } else {
            // TODO: Check Left-Linear
            // If not linear, we can't solve the SCC as a regex.
            // But we might have productions that refer to *already solved* NTs (from lower SCCs).
            // We don't need to do anything here; `rebuild_grammar` will handle substitution of solved NTs.
            debug!(3, "SCC {:?} is NOT Right-Linear. Skipping optimization.", scc_nts);
        }
    }

    // Returns map: NT -> (Map<NT, Expr>, Expr) representing NT = Sum(Coeff * Variable) + Constant
    fn try_build_right_linear_system(&self, scc_nts: &HashSet<NonTerminal>) -> Option<HashMap<NonTerminal, (HashMap<NonTerminal, Expr>, Expr)>> {
        let mut system = HashMap::new();

        for nt in scc_nts {
            let mut coeffs: HashMap<NonTerminal, Vec<Expr>> = HashMap::new();
            let mut constants: Vec<Expr> = Vec::new();

            // Find productions for this NT
            let prods: Vec<&Production> = self.grammar.productions.iter().filter(|p| p.lhs == *nt).collect();
            
            if prods.is_empty() {
                // No productions? It's empty set (or external?). Assume empty set.
                // If external, we shouldn't be here? External NTs don't have productions.
                // But `nt_to_id` only includes things in productions.
            }

            for prod in prods {
                // Check structure: Terminals* (Variable)?
                // Where "Terminals" includes Literals, RegexNames, AND Solved NTs.
                
                let mut prefix_exprs = Vec::new();
                let mut variable = None;

                for sym in &prod.rhs {
                    if variable.is_some() {
                        // If we already saw a variable, we can't have anything after it for Right-Linear
                        return None;
                    }

                    match sym {
                        Symbol::Terminal(t) => {
                            prefix_exprs.push(self.get_expr_for_terminal(t));
                        }
                        Symbol::NonTerminal(ref other_nt) => {
                            if scc_nts.contains(other_nt) {
                                variable = Some(other_nt.clone());
                            } else if let Some(solved_expr) = self.solved_nt.get(other_nt) {
                                prefix_exprs.push(solved_expr.clone());
                            } else {
                                // Reference to an unsolved NT outside the SCC?
                                // This implies the SCCs were not processed in order, or it's a "lower" SCC that wasn't solvable.
                                // If it's effectively a terminal (unsolvable), we can treat it as a "symbol" but we can't inline it as Expr.
                                // We can't put a NonTerminal into an Expr.
                                // So we fail to convert this SCC to Expr.
                                return None;
                            }
                        }
                    }
                }

                let prefix = Self::seq(prefix_exprs);

                if let Some(var) = variable {
                    coeffs.entry(var).or_default().push(prefix);
                } else {
                    constants.push(prefix);
                }
            }

            // Combine choices
            let combined_coeffs: HashMap<NonTerminal, Expr> = coeffs.into_iter().map(|(k, v)| (k, Self::choice(v))).collect();
            let combined_constant = Self::choice(constants);

            system.insert(nt.clone(), (combined_coeffs, combined_constant));
        }

        Some(system)
    }

    fn solve_system(&self, mut system: HashMap<NonTerminal, (HashMap<NonTerminal, Expr>, Expr)>, scc_nts: &HashSet<NonTerminal>) -> HashMap<NonTerminal, Expr> {
        // Gaussian Elimination
        // Order variables
        let variables: Vec<NonTerminal> = scc_nts.iter().cloned().collect();
        let n = variables.len();

        // We work with a matrix M where M[i][j] is coeff of X_j in equation for X_i
        // and C[i] is constant for X_i.
        // X_i = Sum(M[i][j] X_j) + C[i]

        // Initialize table
        // We use indices 0..n
        let mut matrix: Vec<Vec<Expr>> = vec![vec![Expr::Choice(vec![]); n]; n]; // Empty choice is "Zero" (Empty Set)
        let mut constants: Vec<Expr> = vec![Expr::Choice(vec![]); n];

        let nt_to_idx: HashMap<NonTerminal, usize> = variables.iter().enumerate().map(|(i, nt)| (nt.clone(), i)).collect();

        for (i, nt) in variables.iter().enumerate() {
            if let Some((coeffs, constant)) = system.remove(nt) {
                constants[i] = constant;
                for (var, coeff) in coeffs {
                    if let Some(&j) = nt_to_idx.get(&var) {
                        matrix[i][j] = coeff;
                    }
                }
            }
        }

        // Forward elimination
        for i in 0..n {
            // Equation i: X_i = M[i][i] X_i + Sum_{j!=i} M[i][j] X_j + C[i]
            // Apply Arden's Lemma: X = A X + B  =>  X = A* B
            // Here A = M[i][i], B = Sum_{j!=i} M[i][j] X_j + C[i]
            
            let a = matrix[i][i].clone();
            let a_star = Self::star(a);

            // Update M[i][j] and C[i]
            // New M[i][j] = A* M[i][j]
            // New C[i] = A* C[i]
            // M[i][i] becomes 0 (effectively, since we solved for X_i in terms of others)
            
            // Actually, we substitute X_i into other equations k != i (or just k > i for triangular form? 
            // Standard Gaussian eliminates X_i from equations k > i.
            
            // First, normalize row i: X_i = A* (Sum_{j>i} M[i][j] X_j + Sum_{j<i} M[i][j] X_j + C[i])
            // Wait, if we do standard Gaussian, we assume X_j for j < i are already eliminated?
            // No, standard Gaussian makes matrix triangular.
            
            // Let's stick to the loop:
            // 1. Eliminate self-dependency M[i][i]
            //    X_i = M[i][i]* ( Sum_{j!=i} M[i][j] X_j + C[i] )
            //    So M[i][j] <- M[i][i]* M[i][j] for j != i
            //    C[i] <- M[i][i]* C[i]
            //    M[i][i] <- Zero
            
            let loop_coeff = matrix[i][i].clone();
            let loop_star = Self::star(loop_coeff);
            
            matrix[i][i] = Expr::Choice(vec![]); // Zero
            
            for j in 0..n {
                if i == j { continue; }
                matrix[i][j] = Self::seq(vec![loop_star.clone(), matrix[i][j].clone()]);
            }
            constants[i] = Self::seq(vec![loop_star, constants[i].clone()]);
            
            // 2. Substitute X_i into X_k for k > i
            for k in (i + 1)..n {
                // X_k = M[k][i] X_i + ...
                // X_k = M[k][i] ( Sum_j M[i][j] X_j + C[i] ) + ...
                // X_k = Sum_j (M[k][i] M[i][j]) X_j + M[k][i] C[i] + ...
                
                let factor = matrix[k][i].clone();
                if Self::is_empty(&factor) { continue; }
                
                matrix[k][i] = Expr::Choice(vec![]); // Zero out X_i dependency
                
                for j in 0..n {
                    let term = Self::seq(vec![factor.clone(), matrix[i][j].clone()]);
                    matrix[k][j] = Self::choice(vec![matrix[k][j].clone(), term]);
                }
                
                let const_term = Self::seq(vec![factor, constants[i].clone()]);
                constants[k] = Self::choice(vec![constants[k].clone(), const_term]);
            }
        }

        // Back substitution
        // Now matrix is upper triangular (M[k][i] = 0 for k > i).
        // Iterate backwards from n-1 to 0.
        // At step i, X_i depends on X_{i+1} ... X_{n-1}.
        // But we already know their values (constants).
        // Wait, step 1 above eliminated self-loops.
        // And step 2 eliminated lower-triangle dependencies.
        // So at row i, X_i depends only on X_j for j > i.
        // Since we go backwards, X_j (j > i) are already fully resolved to constant Exprs.
        
        let mut results: Vec<Expr> = vec![Expr::Choice(vec![]); n];
        
        for i in (0..n).rev() {
            // X_i = Sum_{j>i} M[i][j] X_j + C[i]
            // Since j > i, X_j is already in `results`.
            
            let mut parts = Vec::new();
            for j in (i + 1)..n {
                let term = Self::seq(vec![matrix[i][j].clone(), results[j].clone()]);
                parts.push(term);
            }
            parts.push(constants[i].clone());
            
            results[i] = Self::choice(parts);
        }

        let mut solution = HashMap::new();
        for (i, nt) in variables.into_iter().enumerate() {
            solution.insert(nt, results[i].clone());
        }
        solution
    }

    fn rebuild_grammar(&mut self) {
        // Create new terminals for all solved NTs that are used in the final grammar.
        
        let mut new_productions = Vec::new();

        // 1. Process Unsolved Productions
        // Replace solved NTs with their new Terminals.
        for prod in &self.grammar.productions {
            if self.solved_nt.contains_key(&prod.lhs) {
                // This production is for a solved NT. It is obsolete.
                continue;
            }

            let mut new_rhs = Vec::new();
            for sym in &prod.rhs {
                match sym {
                    Symbol::Terminal(t) => new_rhs.push(Symbol::Terminal(t.clone())),
                    Symbol::NonTerminal(nt) => {
                        // We keep the NonTerminal symbol for now, will replace with Terminal in pass 2
                        new_rhs.push(Symbol::NonTerminal(nt.clone()));
                    }
                }
            }
            new_productions.push(Production {
                lhs: prod.lhs.clone(),
                rhs: new_rhs,
            });
        }

        let start_nt = self.grammar.productions[self.grammar.start_production_id].lhs.clone();
        let start_is_solved = self.solved_nt.contains_key(&start_nt);

        self.grammar.productions = new_productions;

        // 2. Collect needed terminals
        // We need terminals for any solved NT that appears in the RHS of the new productions,
        // OR for the start symbol if it was solved.
        let mut needed_terminals: Vec<(String, Expr)> = Vec::new();
        
        for prod in &self.grammar.productions {
            for sym in &prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    if let Some(expr) = self.solved_nt.get(nt) {
                        needed_terminals.push((nt.0.clone(), expr.clone()));
                    }
                }
            }
        }
        
        if start_is_solved {
             if let Some(expr) = self.solved_nt.get(&start_nt) {
                 needed_terminals.push((start_nt.0.clone(), expr.clone()));
             }
        }

        // 3. Create terminals
        for (name, expr) in needed_terminals {
            Self::ensure_terminal(self.grammar, name, expr);
        }

        // 4. Update productions to use the new terminals
        for prod in &mut self.grammar.productions {
            for sym in &mut prod.rhs {
                if let Symbol::NonTerminal(nt) = sym {
                    if self.solved_nt.contains_key(nt) {
                        *sym = Symbol::Terminal(Terminal::RegexName(nt.0.clone()));
                    }
                }
            }
        }

        // 5. Add start production if needed
        if start_is_solved {
            let new_prod = Production {
                lhs: start_nt.clone(),
                rhs: vec![Symbol::Terminal(Terminal::RegexName(start_nt.0.clone()))],
            };
            self.grammar.productions.push(new_prod);
            self.grammar.start_production_id = self.grammar.productions.len() - 1;
        } else {
            // Update start_production_id
            if let Some(idx) = self.grammar.productions.iter().position(|p| p.lhs == start_nt) {
                self.grammar.start_production_id = idx;
            } else {
                panic!("Start symbol lost during optimization!");
            }
        }
    }

    fn ensure_terminal(grammar: &mut GrammarDefinition, name: String, expr: Expr) -> usize {
        if let Some(&gid) = grammar.regex_name_to_group_id.get_by_left(&name) {
            // Update expr just in case?
            grammar.group_id_to_expr.insert(gid, expr);
            gid
        } else {
            let gid = grammar.group_id_to_expr.keys().max().map(|k| k + 1).unwrap_or(0);
            grammar.regex_name_to_group_id.insert(name, gid);
            grammar.group_id_to_expr.insert(gid, expr);
            gid
        }
    }

    fn get_expr_for_terminal(&self, t: &Terminal) -> Expr {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        let group_id = group_id.expect("Terminal not found in grammar");
        let expr = self.grammar.group_id_to_expr.get(group_id).cloned().expect("Expr not found for group_id");
        expr
    }

    // Expr Helpers
    fn seq(exprs: Vec<Expr>) -> Expr {
        // Flatten nested sequences and remove epsilons
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
            flat.pop().unwrap()
        } else {
            Expr::Seq(flat)
        }
    }

    fn choice(exprs: Vec<Expr>) -> Expr {
        // Flatten nested choices
        let mut flat = Vec::new();
        for e in exprs {
            match e {
                Expr::Choice(sub) => flat.extend(sub),
                _ => flat.push(e),
            }
        }
        // Remove duplicates? (Optional but good for size)
        // Need PartialEq. Expr has it.
        // Also remove Empty Set? We don't have explicit Empty Set, usually empty Choice is empty set.
        // But if we have `Choice([A, EmptySet])` -> `A`.
        // If `flat` is empty, it's Empty Set.
        
        if flat.is_empty() {
            return Expr::Choice(vec![]);
        }
        if flat.len() == 1 {
            return flat.pop().unwrap();
        }
        Expr::Choice(flat)
    }

    fn star(expr: Expr) -> Expr {
        match expr {
            Expr::Epsilon => Expr::Epsilon, // epsilon* = epsilon
            Expr::Choice(ref c) if c.is_empty() => Expr::Epsilon, // empty* = epsilon
            Expr::Quantifier(inner, QuantifierType::ZeroOrMore) => Expr::Quantifier(inner, QuantifierType::ZeroOrMore), // (A*)* = A*
            _ => Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore),
        }
    }
    
    fn is_empty(expr: &Expr) -> bool {
        match expr {
            Expr::Choice(c) => c.is_empty(),
            _ => false,
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
        eprintln!("After from_exprs: {} terminals", grammar.terminal_to_group_id().len());
        optimize_grammar(&mut grammar);
        eprintln!("After optimize_grammar: {} terminals", grammar.terminal_to_group_id().len());
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

        use crate::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }
}
