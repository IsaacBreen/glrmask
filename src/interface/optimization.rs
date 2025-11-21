use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use bimap::BiBTreeMap;
use crate::finite_automata::{Expr, QuantifierType};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::{GrammarDefinition, GrammarExpr};
use crate::types::TerminalID;

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
    interner: ExprInterner,
}

impl<'a> GrammarOptimizer<'a> {
    fn new(grammar: &'a mut GrammarDefinition) -> Self {
        Self {
            grammar,
            resolved_nts: HashMap::new(),
            interner: ExprInterner::new(),
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

    fn try_convert_scc(&mut self, scc_nts: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        // Skip single-node SCCs with no self-recursion if they only reference other unresolved NTs
        // (they're just pass-through rules). But if they reference terminals or resolved NTs,
        // we should still optimize them to regex.
        if scc_nts.len() == 1 {
            let nt = &scc_nts[0];
            let productions: Vec<Production> = self.grammar.productions.iter().filter(|p| &p.lhs == nt).cloned().collect();
            
            // Check if any production has self-recursion
            let has_self_recursion = productions.iter().any(|prod| {
                prod.rhs.iter().any(|sym| {
                    matches!(sym, Symbol::NonTerminal(ref other) if other == nt)
                })
            });
            
            if !has_self_recursion {
                // Check if all references are to unresolved non-terminals
                let only_unresolved_nt_refs = productions.iter().all(|prod| {
                    prod.rhs.iter().all(|sym| {
                        match sym {
                            Symbol::Terminal(_) => false, // Has terminals, should optimize
                            Symbol::NonTerminal(ref other) => {
                                // If it's resolved or is self, we can optimize
                                !self.resolved_nts.contains_key(other) && other != nt
                            }
                        }
                    })
                });
                
                if only_unresolved_nt_refs {
                    return None;
                }
            }
        }
        
        if let Some(res) = self.try_convert_scc_right_linear(scc_nts) {
            return Some(res);
        }
        self.try_convert_scc_left_linear(scc_nts)
    }

    fn try_convert_scc_right_linear(&mut self, scc_nts: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        let scc_set: HashSet<&NonTerminal> = scc_nts.iter().collect();
        let mut transitions: Vec<Vec<(usize, Expr)>> = vec![Vec::new(); scc_nts.len()];
        let mut finals: Vec<Expr> = vec![self.interner.choice(vec![]); scc_nts.len()];
        let nt_to_local_idx: HashMap<&NonTerminal, usize> = scc_nts.iter().enumerate().map(|(i, nt)| (nt, i)).collect();

        for (i, nt) in scc_nts.iter().enumerate() {
            let productions: Vec<Production> = self.grammar.productions.iter().filter(|p| &p.lhs == nt).cloned().collect();
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
                                if idx != prod.rhs.len() - 1 { 
                                    return None; 
                                }
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
                
                let prefix_expr = self.interner.seq(prefix_exprs);
                if let Some(target) = target_scc_idx {
                    transitions[i].push((target, prefix_expr));
                } else {
                    finals[i] = self.interner.choice(vec![finals[i].clone(), prefix_expr]);
                }
            }
        }
        
        let solved = self.solve_regular_system(scc_nts.len(), &transitions, &finals);
        let mut result = HashMap::new();
        for (i, expr) in solved.into_iter().enumerate() {
            result.insert(scc_nts[i].clone(), Expr::Shared(Arc::new(expr)));
        }
        Some(result)
    }

    fn try_convert_scc_left_linear(&mut self, scc_nts: &[NonTerminal]) -> Option<HashMap<NonTerminal, Expr>> {
        let scc_set: HashSet<&NonTerminal> = scc_nts.iter().collect();
        let mut transitions: Vec<Vec<(usize, Expr)>> = vec![Vec::new(); scc_nts.len()];
        let mut finals: Vec<Expr> = vec![self.interner.choice(vec![]); scc_nts.len()];
        let nt_to_local_idx: HashMap<&NonTerminal, usize> = scc_nts.iter().enumerate().map(|(i, nt)| (nt, i)).collect();

        for (i, nt) in scc_nts.iter().enumerate() {
            let productions: Vec<Production> = self.grammar.productions.iter().filter(|p| &p.lhs == nt).cloned().collect();
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
                
                let suffix_expr = self.interner.seq(suffix_exprs);
                let mut cache = HashMap::new();
                let reversed_suffix = self.reverse_expr(&suffix_expr, &mut cache);

                if let Some(target) = target_scc_idx {
                    // A -> B suffix.  Edge B --suffix--> A.
                    // We map this to Right-Linear: A --rev(suffix)--> B.
                    transitions[i].push((target, reversed_suffix));
                } else {
                    // A -> suffix. Edge Start --suffix--> A.
                    // Map to Right-Linear: A --rev(suffix)--> Final.
                    finals[i] = self.interner.choice(vec![finals[i].clone(), reversed_suffix]);
                }
            }
        }
        
        let solved = self.solve_regular_system(scc_nts.len(), &transitions, &finals);
        let mut result = HashMap::new();
        let mut cache = HashMap::new();
        for (i, expr) in solved.into_iter().enumerate() {
            result.insert(scc_nts[i].clone(), Expr::Shared(Arc::new(self.reverse_expr(&expr, &mut cache))));
        }
        Some(result)
    }

    fn reverse_expr(&mut self, expr: &Expr, cache: &mut HashMap<usize, Expr>) -> Expr {
        match expr {
            Expr::U8Seq(bytes) => {
                let mut b = bytes.clone();
                b.reverse();
                self.interner.intern(Expr::U8Seq(b))
            },
            Expr::Seq(exprs) => {
                let mut e = exprs.clone();
                e.reverse();
                let reversed_sub: Vec<Expr> = e.into_iter().map(|x| self.reverse_expr(&x, cache)).collect();
                self.interner.seq(reversed_sub)
            },
            Expr::Choice(exprs) => {
                let reversed_sub: Vec<Expr> = exprs.iter().map(|x| self.reverse_expr(x, cache)).collect();
                self.interner.choice(reversed_sub)
            },
            Expr::Quantifier(inner, q) => {
                let reversed_inner = self.reverse_expr(inner, cache);
                self.interner.intern(Expr::Quantifier(Box::new(reversed_inner), q.clone()))
            },
            Expr::Shared(inner) => {
                let key = Arc::as_ptr(inner) as usize;
                if let Some(cached) = cache.get(&key) {
                    return cached.clone();
                }
                let reversed = Expr::Shared(Arc::new(self.reverse_expr(inner, cache)));
                cache.insert(key, reversed.clone());
                reversed
            },
            _ => expr.clone(),
        }
    }
    
    fn get_expr_for_terminal(&mut self, t: &Terminal) -> Expr {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        let group_id = group_id.expect("Terminal not found in grammar");
        let expr = self.grammar.group_id_to_expr.get(group_id).cloned().expect("Expr not found for group_id");
        self.interner.intern(expr)
    }

    fn solve_regular_system(&mut self, n: usize, transitions: &Vec<Vec<(usize, Expr)>>, finals: &Vec<Expr>) -> Vec<Expr> {
        // Kleene's algorithm / Floyd-Warshall for Regex
        // R[k][i][j] = paths from i to j using only intermediate nodes < k
        
        let mut r: Vec<Vec<Expr>> = vec![vec![self.interner.choice(vec![]); n]; n];
        
        for i in 0..n {
            // Initialize diagonal with Epsilon to allow zero-length paths (essential for correct loop optimization)
            r[i][i] = self.interner.choice(vec![r[i][i].clone(), Expr::Epsilon]);
            
            for (target, expr) in &transitions[i] {
                r[i][*target] = self.interner.choice(vec![r[i][*target].clone(), expr.clone()]);
            }
        }
        
        // Iterate k from 0 to n-1 (node being eliminated/considered as intermediate)
        for k in 0..n {
            let r_kk = r[k][k].clone();
            let r_kk_star = self.interner.star(r_kk);
            
            let mut next_r = r.clone();
            
            for i in 0..n {
                for j in 0..n {
                    let r_ik = &r[i][k];
                    let r_kj = &r[k][j];
                    
                    let path_through_k = self.interner.seq(vec![
                        r_ik.clone(),
                        r_kk_star.clone(),
                        r_kj.clone()
                    ]);
                    
                    next_r[i][j] = self.interner.choice(vec![r[i][j].clone(), path_through_k]);
                }
            }
            r = next_r;
        }
        
        // Now compute final expressions for each node
        let mut results = Vec::new();
        for i in 0..n {
            let mut choices = Vec::new();
            for j in 0..n {
                let path = self.interner.seq(vec![r[i][j].clone(), finals[j].clone()]);
                choices.push(path);
            }
            results.push(self.interner.choice(choices));
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
        let mut used_groups = BTreeSet::new();
        
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

        // Create mapping from old_gid -> new_gid
        let mut old_to_new = HashMap::new();
        for (new_id, old_id) in used_groups.iter().enumerate() {
            old_to_new.insert(*old_id, new_id);
        }

        // Update group_id_to_expr
        let mut new_group_id_to_expr = BTreeMap::new();
        for (old_id, expr) in &self.grammar.group_id_to_expr {
            if let Some(&new_id) = old_to_new.get(old_id) {
                new_group_id_to_expr.insert(new_id, expr.clone());
            }
        }
        self.grammar.group_id_to_expr = new_group_id_to_expr;
        
        // Update maps
        let mut new_literal = BiBTreeMap::new();
        for (k, v) in &self.grammar.literal_to_group_id {
            if let Some(&new_id) = old_to_new.get(v) {
                new_literal.insert(k.clone(), new_id);
            }
        }
        self.grammar.literal_to_group_id = new_literal;
        
        let mut new_regex = BiBTreeMap::new();
        for (k, v) in &self.grammar.regex_name_to_group_id {
            if let Some(&new_id) = old_to_new.get(v) {
                new_regex.insert(k.clone(), new_id);
            }
        }
        self.grammar.regex_name_to_group_id = new_regex;
        
        let mut new_external = BiBTreeMap::new();
        for (k, v) in &self.grammar.external_name_to_group_id {
             if let Some(&new_id) = old_to_new.get(v) {
                new_external.insert(k.clone(), new_id);
            }
        }
        self.grammar.external_name_to_group_id = new_external;

        // Update ignore_terminal_id
        if let Some(gid) = self.grammar.ignore_terminal_id {
            if let Some(&new_id) = old_to_new.get(&gid.0) {
                self.grammar.ignore_terminal_id = Some(TerminalID(new_id));
            }
        }
    }
    
    fn get_group_id(&self, t: &Terminal) -> usize {
         match t {
            Terminal::Literal(bytes) => *self.grammar.literal_to_group_id.get_by_left(bytes).expect("Terminal missing"),
            Terminal::RegexName(name) => *self.grammar.regex_name_to_group_id.get_by_left(name).expect("Terminal missing"),
        }
    }
}

use std::hash::{Hash, Hasher};

#[derive(Clone, Eq)]
struct InternKey(Expr);

impl PartialEq for InternKey {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Expr::Shared(a), Expr::Shared(b)) => Arc::ptr_eq(a, b),
            (Expr::U8Seq(a), Expr::U8Seq(b)) => a == b,
            (Expr::U8Class(a), Expr::U8Class(b)) => a == b,
            (Expr::Quantifier(a, qa), Expr::Quantifier(b, qb)) => qa == qb && InternKey(*a.clone()) == InternKey(*b.clone()),
            (Expr::Choice(a), Expr::Choice(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| InternKey(x.clone()) == InternKey(y.clone())),
            (Expr::Seq(a), Expr::Seq(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| InternKey(x.clone()) == InternKey(y.clone())),
            (Expr::Epsilon, Expr::Epsilon) => true,
            _ => false,
        }
    }
}

impl Hash for InternKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.0).hash(state);
        match &self.0 {
            Expr::Shared(arc) => Arc::as_ptr(arc).hash(state),
            Expr::U8Seq(bytes) => bytes.hash(state),
            Expr::U8Class(set) => set.hash(state),
            Expr::Quantifier(expr, q) => {
                InternKey(*expr.clone()).hash(state);
                q.hash(state);
            }
            Expr::Choice(exprs) | Expr::Seq(exprs) => {
                for e in exprs {
                    InternKey(e.clone()).hash(state);
                }
            }
            Expr::Epsilon => {}
        }
    }
}

struct ExprInterner {
    cache: HashMap<InternKey, Expr>,
}

impl ExprInterner {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    fn intern(&mut self, expr: Expr) -> Expr {
        if let Expr::Shared(_) = expr {
            return expr;
        }

        let key = InternKey(expr.clone());
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let shared = Expr::Shared(Arc::new(expr));
        self.cache.insert(key, shared.clone());
        shared
    }

    fn seq(&mut self, exprs: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for e in exprs {
            match e {
                Expr::Seq(sub) => flat.extend(sub),
                Expr::Shared(inner) => {
                    match &*inner {
                        Expr::Seq(sub) => flat.extend(sub.iter().cloned()),
                        Expr::Epsilon => {},
                        _ => flat.push(e),
                    }
                }
                Expr::Epsilon => {}
                _ => flat.push(e),
            }
        }
        if flat.is_empty() {
            Expr::Epsilon
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            self.intern(Expr::Seq(flat))
        }
    }

    fn choice(&mut self, exprs: Vec<Expr>) -> Expr {
        let mut flat = Vec::new();
        for e in exprs {
            match e {
                Expr::Choice(sub) => flat.extend(sub),
                Expr::Shared(inner) => {
                    match &*inner {
                        Expr::Choice(sub) => flat.extend(sub.iter().cloned()),
                        _ => flat.push(e),
                    }
                }
                _ => flat.push(e),
            }
        }

        if flat.is_empty() {
            self.intern(Expr::Choice(vec![]))
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            flat.sort();
            flat.dedup();
            if flat.len() == 1 {
                flat.into_iter().next().unwrap()
            } else {
                self.intern(Expr::Choice(flat))
            }
        }
    }

    fn star(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::Epsilon => Expr::Epsilon,
            Expr::Shared(inner) if matches!(&*inner, Expr::Epsilon) => Expr::Epsilon,
            Expr::Quantifier(inner, QuantifierType::ZeroOrMore) => {
                self.intern(Expr::Quantifier(inner, QuantifierType::ZeroOrMore))
            }
            Expr::Shared(inner) => {
                if let Expr::Quantifier(_, QuantifierType::ZeroOrMore) = &*inner {
                    expr.clone()
                } else {
                    self.intern(Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore))
                }
            }
            Expr::Choice(v) if v.is_empty() => Expr::Epsilon,
            _ => self.intern(Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore)),
        }
    }
}
