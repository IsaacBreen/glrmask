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

impl GrammarDefinition {
    pub fn optimize(&mut self) {
        optimize_grammar(self);
    }
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
        
        // 3. Process SCCs in topological order
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
        
        // 5. Cleanup unused terminals
        self.cleanup_terminals();
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
        if let Some(res) = self.try_convert_scc_right_linear(scc_nts) {
            return Some(res);
        }
        self.try_convert_scc_left_linear(scc_nts)
    }

    fn try_convert_scc_right_linear(&self, scc_nts: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        let scc_set: HashSet<&NonTerminal> = scc_nts.iter().collect();
        let mut transitions: Vec<Vec<(usize, Expr)>> = vec![Vec::new(); scc_nts.len()];
        let mut finals: Vec<Expr> = vec![Expr::Choice(vec![]); scc_nts.len()];
        let nt_to_local_idx: HashMap<&NonTerminal, usize> = scc_nts.iter().enumerate().map(|(i, nt)| (nt, i)).collect();

        for (i, nt) in scc_nts.iter().enumerate() {
            let productions: Vec<&Production> = self.grammar.productions.iter().filter(|p| &p.lhs == nt).collect();
            if productions.is_empty() { continue; }

            for prod in productions {
                let mut prefix_exprs = Vec::new();
                let mut target_scc_idx = None;
                
                for (idx, symbol) in prod.rhs.iter().enumerate() {
                    match symbol {
                        Symbol::Terminal(t) => {
                            prefix_exprs.push(self.get_expr_for_terminal(t));
                        }
                        Symbol::NonTerminal(ref other_nt) => {
                            if let Some(&local_idx) = nt_to_local_idx.get(other_nt) {
                                if idx != prod.rhs.len() - 1 { return None; }
                                target_scc_idx = Some(local_idx);
                            } else {
                                if let Some(expr) = self.resolved_nts.get(other_nt) {
                                    prefix_exprs.push(expr.clone());
                                } else {
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
                    finals[i] = ExprBuilder::choice(vec![finals[i].clone(), prefix_expr]);
                }
            }
        }
        
        let solved = self.solve_regular_system(scc_nts.len(), transitions, finals);
        let mut result = HashMap::new();
        for (i, expr) in solved.into_iter().enumerate() {
            result.insert(scc_nts[i].clone(), expr);
        }
        Some(result)
    }

    fn try_convert_scc_left_linear(&self, scc_nts: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        let scc_set: HashSet<&NonTerminal> = scc_nts.iter().collect();
        let mut transitions: Vec<Vec<(usize, Expr)>> = vec![Vec::new(); scc_nts.len()];
        let mut finals: Vec<Expr> = vec![Expr::Choice(vec![]); scc_nts.len()];
        let nt_to_local_idx: HashMap<&NonTerminal, usize> = scc_nts.iter().enumerate().map(|(i, nt)| (nt, i)).collect();

        for (i, nt) in scc_nts.iter().enumerate() {
            let productions: Vec<&Production> = self.grammar.productions.iter().filter(|p| &p.lhs == nt).collect();
            if productions.is_empty() { continue; }

            for prod in productions {
                let mut suffix_exprs = Vec::new();
                let mut target_scc_idx = None;
                
                for (idx, symbol) in prod.rhs.iter().enumerate() {
                    match symbol {
                        Symbol::Terminal(t) => {
                            suffix_exprs.push(self.get_expr_for_terminal(t));
                        }
                        Symbol::NonTerminal(ref other_nt) => {
                            if let Some(&local_idx) = nt_to_local_idx.get(other_nt) {
                                if idx != 0 { return None; } // Must be first symbol for Left-Linear
                                target_scc_idx = Some(local_idx);
                            } else {
                                if let Some(expr) = self.resolved_nts.get(other_nt) {
                                    suffix_exprs.push(expr.clone());
                                } else {
                                    return None;
                                }
                            }
                        }
                    }
                }
                
                let suffix_expr = ExprBuilder::seq(suffix_exprs);
                let reversed_suffix = Self::reverse_expr(&suffix_expr);

                if let Some(target) = target_scc_idx {
                    // A -> B suffix.  Edge B --suffix--> A.
                    // We map this to Right-Linear: A --rev(suffix)--> B.
                    transitions[i].push((target, reversed_suffix));
                } else {
                    // A -> suffix. Edge Start --suffix--> A.
                    // Map to Right-Linear: A --rev(suffix)--> Final.
                    finals[i] = ExprBuilder::choice(vec![finals[i].clone(), reversed_suffix]);
                }
            }
        }
        
        let solved = self.solve_regular_system(scc_nts.len(), transitions, finals);
        let mut result = HashMap::new();
        for (i, expr) in solved.into_iter().enumerate() {
            result.insert(scc_nts[i].clone(), Self::reverse_expr(&expr));
        }
        Some(result)
    }

    fn reverse_expr(expr: &Expr) -> Expr {
        match expr {
            Expr::U8Seq(bytes) => {
                let mut b = bytes.clone();
                b.reverse();
                Expr::U8Seq(b)
            },
            Expr::Seq(exprs) => {
                let mut e = exprs.clone();
                e.reverse();
                let reversed_sub: Vec<Expr> = e.into_iter().map(|x| Self::reverse_expr(&x)).collect();
                Expr::Seq(reversed_sub)
            },
            Expr::Choice(exprs) => {
                let reversed_sub: Vec<Expr> = exprs.iter().map(|x| Self::reverse_expr(x)).collect();
                Expr::Choice(reversed_sub)
            },
            Expr::Quantifier(inner, q) => Expr::Quantifier(Box::new(Self::reverse_expr(inner)), q.clone()),
            Expr::Shared(inner) => Expr::Shared(Arc::new(Self::reverse_expr(inner))),
            _ => expr.clone(),
        }
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
        
        let mut r: Vec<Vec<Expr>> = vec![vec![Expr::Choice(vec![]); n]; n];
        
        for i in 0..n {
            for &(target, ref expr) in &transitions[i] {
                r[i][target] = ExprBuilder::choice(vec![r[i][target].clone(), expr.clone()]);
            }
        }
        
        // Iterate k from 0 to n-1 (node being eliminated/considered as intermediate)
        for k in 0..n {
            let r_kk = r[k][k].clone();
            let r_kk_star = ExprBuilder::star(r_kk);
            
            let mut next_r = r.clone();
            
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
        let mut new_terminals: HashMap<NonTerminal, Terminal> = HashMap::new();
        
        // Allocate group IDs and create Terminals
        let mut next_group_id = self.grammar.group_id_to_expr.keys().max().map(|&x| x + 1).unwrap_or(0);
        
        for (nt, expr) in &self.resolved_nts {
            let mut final_name = nt.0.clone();
            while self.grammar.regex_name_to_group_id.contains_left(&final_name) {
                final_name.push('_');
            }
            
            self.grammar.regex_name_to_group_id.insert(final_name.clone(), next_group_id);
            self.grammar.group_id_to_expr.insert(next_group_id, expr.clone());
            
            new_terminals.insert(nt.clone(), Terminal::RegexName(final_name));
            next_group_id += 1;
        }
        
        // Identify start symbol
        let start_nt = if self.grammar.productions.len() > self.grammar.start_production_id {
            self.grammar.productions[self.grammar.start_production_id].lhs.clone()
        } else {
            // Should not happen in valid grammar
            NonTerminal("".to_string())
        };

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
        
        // If start symbol was resolved, add a production for it
        if self.resolved_nts.contains_key(&start_nt) {
            if let Some(term) = new_terminals.get(&start_nt) {
                new_productions.insert(0, Production {
                    lhs: start_nt.clone(),
                    rhs: vec![Symbol::Terminal(term.clone())],
                });
            }
        }
        
        // Update start_production_id
        // Find the production with start_nt as lhs
        if let Some(idx) = new_productions.iter().position(|p| p.lhs == start_nt) {
            self.grammar.start_production_id = idx;
        }
        
        self.grammar.productions = new_productions;
    }

    fn cleanup_terminals(&mut self) {
        let mut used_groups = HashSet::new();
        
        for prod in &self.grammar.productions {
            for sym in &prod.rhs {
                if let Symbol::Terminal(t) = sym {
                    let gid = self.get_group_id(t);
                    used_groups.insert(gid);
                }
            }
        }
        
        if let Some(gid) = self.grammar.ignore_terminal_id {
             used_groups.insert(gid.0);
        }

        // Filter group_id_to_expr
        self.grammar.group_id_to_expr.retain(|k, _| used_groups.contains(k));
        
        // Filter maps
        let mut new_literal = BiBTreeMap::new();
        for (k, v) in &self.grammar.literal_to_group_id {
            if used_groups.contains(v) {
                new_literal.insert(k.clone(), *v);
            }
        }
        self.grammar.literal_to_group_id = new_literal;
        
        let mut new_regex = BiBTreeMap::new();
        for (k, v) in &self.grammar.regex_name_to_group_id {
            if used_groups.contains(v) {
                new_regex.insert(k.clone(), *v);
            }
        }
        self.grammar.regex_name_to_group_id = new_regex;
        
        let mut new_external = BiBTreeMap::new();
        for (k, v) in &self.grammar.external_name_to_group_id {
             if used_groups.contains(v) {
                new_external.insert(k.clone(), *v);
            }
        }
        self.grammar.external_name_to_group_id = new_external;
    }
    
    fn get_group_id(&self, t: &Terminal) -> usize {
         match t {
            Terminal::Literal(bytes) => *self.grammar.literal_to_group_id.get_by_left(bytes).expect("Terminal missing"),
            Terminal::RegexName(name) => *self.grammar.regex_name_to_group_id.get_by_left(name).expect("Terminal missing"),
        }
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
        
        if flat.is_empty() {
            Expr::Choice(vec![])
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            Expr::Choice(flat)
        }
    }
    
    fn star(expr: Expr) -> Expr {
        match expr {
            Expr::Epsilon => Expr::Epsilon,
            Expr::Quantifier(inner, QuantifierType::ZeroOrMore) => Expr::Quantifier(inner, QuantifierType::ZeroOrMore),
            Expr::Choice(v) if v.is_empty() => Expr::Epsilon,
            _ => Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
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
            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed to collapse grammar on iteration {} (started with {} terminals)", i, initial_count);
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

        let mut grammar = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs).unwrap();
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
}
