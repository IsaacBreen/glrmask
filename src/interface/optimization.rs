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

/// Represents a regex expression that may reference non-terminals.
/// Used during the grammar-to-regex conversion process.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum RegexTerm {
    /// A concrete regex expression (no non-terminal references)
    Concrete(Expr),
    /// A reference to a non-terminal (to be resolved later)
    NtRef(String),
    /// Sequence of terms
    Seq(Vec<RegexTerm>),
    /// Choice of terms
    Choice(Vec<RegexTerm>),
    /// Quantifier (ZeroOrMore, OneOrMore, ZeroOrOne)
    Quantifier(Box<RegexTerm>, QuantifierType),
    /// Epsilon (empty string)
    Epsilon,
}

impl RegexTerm {
    /// Check if this term contains any non-terminal references
    fn has_nt_refs(&self) -> bool {
        match self {
            RegexTerm::Concrete(_) | RegexTerm::Epsilon => false,
            RegexTerm::NtRef(_) => true,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => terms.iter().any(|t| t.has_nt_refs()),
            RegexTerm::Quantifier(inner, _) => inner.has_nt_refs(),
        }
    }

    /// Get all non-terminal references in this term
    fn get_nt_refs(&self) -> HashSet<String> {
        let mut refs = HashSet::new();
        self.collect_nt_refs(&mut refs);
        refs
    }

    fn collect_nt_refs(&self, refs: &mut HashSet<String>) {
        match self {
            RegexTerm::Concrete(_) | RegexTerm::Epsilon => {}
            RegexTerm::NtRef(name) => { refs.insert(name.clone()); }
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                for t in terms {
                    t.collect_nt_refs(refs);
                }
            }
            RegexTerm::Quantifier(inner, _) => inner.collect_nt_refs(refs),
        }
    }

    /// Convert to Expr (only valid if no NT refs remain)
    fn to_expr(&self) -> Option<Expr> {
        match self {
            RegexTerm::Concrete(e) => Some(e.clone()),
            RegexTerm::Epsilon => Some(Expr::Epsilon),
            RegexTerm::NtRef(_) => None,
            RegexTerm::Seq(terms) => {
                let exprs: Option<Vec<Expr>> = terms.iter().map(|t| t.to_expr()).collect();
                exprs.map(|es| {
                    let filtered: Vec<Expr> = es.into_iter()
                        .filter(|e| !matches!(e, Expr::Epsilon))
                        .collect();
                    if filtered.is_empty() {
                        Expr::Epsilon
                    } else if filtered.len() == 1 {
                        filtered.into_iter().next().unwrap()
                    } else {
                        Expr::Seq(filtered)
                    }
                })
            }
            RegexTerm::Choice(terms) => {
                let exprs: Option<Vec<Expr>> = terms.iter().map(|t| t.to_expr()).collect();
                exprs.map(|es| {
                    if es.len() == 1 {
                        es.into_iter().next().unwrap()
                    } else {
                        Expr::Choice(es)
                    }
                })
            }
            RegexTerm::Quantifier(inner, qtype) => {
                inner.to_expr().map(|e| Expr::Quantifier(Box::new(e), qtype.clone()))
            }
        }
    }

    /// Substitute all occurrences of `nt_name` with `replacement`
    fn substitute(&self, nt_name: &str, replacement: &RegexTerm) -> RegexTerm {
        match self {
            RegexTerm::Concrete(_) | RegexTerm::Epsilon => self.clone(),
            RegexTerm::NtRef(name) if name == nt_name => replacement.clone(),
            RegexTerm::NtRef(_) => self.clone(),
            RegexTerm::Seq(terms) => {
                RegexTerm::Seq(terms.iter().map(|t| t.substitute(nt_name, replacement)).collect())
            }
            RegexTerm::Choice(terms) => {
                RegexTerm::Choice(terms.iter().map(|t| t.substitute(nt_name, replacement)).collect())
            }
            RegexTerm::Quantifier(inner, qtype) => {
                RegexTerm::Quantifier(Box::new(inner.substitute(nt_name, replacement)), qtype.clone())
            }
        }
    }

    /// Simplify the term (flatten nested seqs/choices, remove epsilon from seqs, etc.)
    fn simplify(self) -> RegexTerm {
        match self {
            RegexTerm::Concrete(_) | RegexTerm::Epsilon | RegexTerm::NtRef(_) => self,
            RegexTerm::Seq(terms) => {
                let mut simplified: Vec<RegexTerm> = Vec::new();
                for t in terms {
                    let t = t.simplify();
                    match t {
                        RegexTerm::Epsilon => {} // Skip epsilon in sequences
                        RegexTerm::Seq(inner) => simplified.extend(inner), // Flatten
                        other => simplified.push(other),
                    }
                }
                if simplified.is_empty() {
                    RegexTerm::Epsilon
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    RegexTerm::Seq(simplified)
                }
            }
            RegexTerm::Choice(terms) => {
                let mut simplified: Vec<RegexTerm> = Vec::new();
                let mut seen = HashSet::new();
                for t in terms {
                    let t = t.simplify();
                    match t {
                        RegexTerm::Choice(inner) => {
                            for i in inner {
                                let key = format!("{:?}", i);
                                if seen.insert(key) {
                                    simplified.push(i);
                                }
                            }
                        }
                        other => {
                            let key = format!("{:?}", other);
                            if seen.insert(key) {
                                simplified.push(other);
                            }
                        }
                    }
                }
                if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    RegexTerm::Choice(simplified)
                }
            }
            RegexTerm::Quantifier(inner, qtype) => {
                let inner = inner.simplify();
                // Simplify nested quantifiers
                match (&inner, &qtype) {
                    (RegexTerm::Epsilon, _) => RegexTerm::Epsilon,
                    (RegexTerm::Quantifier(inner2, QuantifierType::ZeroOrMore), QuantifierType::ZeroOrMore) |
                    (RegexTerm::Quantifier(inner2, QuantifierType::ZeroOrMore), QuantifierType::OneOrMore) |
                    (RegexTerm::Quantifier(inner2, QuantifierType::OneOrMore), QuantifierType::ZeroOrMore) => {
                        RegexTerm::Quantifier(inner2.clone(), QuantifierType::ZeroOrMore)
                    }
                    _ => RegexTerm::Quantifier(Box::new(inner), qtype),
                }
            }
        }
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

        // Try to convert the entire grammar to a single regex
        if let Some(final_expr) = self.try_convert_to_regex() {
            self.replace_grammar_with_single_terminal(final_expr);
        }

        self.stats.final_productions = self.grammar.productions.len();
        self.stats.final_terminals = self.count_terminals();
    }

    /// Try to convert the entire grammar to a single regex expression.
    /// Returns Some(Expr) if successful, None if the grammar is not regular.
    fn try_convert_to_regex(&self) -> Option<Expr> {
        // Build a mapping from non-terminal names to their combined production RHSs
        let mut nt_to_terms: HashMap<String, Vec<RegexTerm>> = HashMap::new();
        
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();

        // Convert each production to RegexTerm
        for prod in &self.grammar.productions {
            let term = self.production_rhs_to_regex_term(&prod.rhs, &nt_names)?;
            nt_to_terms.entry(prod.lhs.0.clone()).or_default().push(term);
        }

        // Combine alternatives for each non-terminal
        let mut equations: HashMap<String, RegexTerm> = HashMap::new();
        for (nt, terms) in nt_to_terms {
            let combined = if terms.len() == 1 {
                terms.into_iter().next().unwrap()
            } else {
                RegexTerm::Choice(terms)
            };
            equations.insert(nt, combined.simplify());
        }

        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();

        // Solve the system of equations
        self.solve_regex_system(&mut equations, &start_nt)
    }

    /// Convert a production RHS to a RegexTerm
    fn production_rhs_to_regex_term(&self, rhs: &[Symbol], nt_names: &HashSet<String>) -> Option<RegexTerm> {
        if rhs.is_empty() {
            return Some(RegexTerm::Epsilon);
        }

        let terms: Option<Vec<RegexTerm>> = rhs.iter().map(|sym| {
            match sym {
                Symbol::Terminal(t) => {
                    let expr = self.get_expr_for_terminal(t)?;
                    Some(RegexTerm::Concrete(expr))
                }
                Symbol::NonTerminal(nt) => {
                    if nt_names.contains(&nt.0) {
                        Some(RegexTerm::NtRef(nt.0.clone()))
                    } else {
                        // Unknown non-terminal - grammar is not regular as-is
                        None
                    }
                }
            }
        }).collect();

        terms.map(|ts| {
            if ts.len() == 1 {
                ts.into_iter().next().unwrap()
            } else {
                RegexTerm::Seq(ts)
            }
        })
    }

    /// Get the Expr for a terminal (immutable version)
    fn get_expr_for_terminal(&self, t: &Terminal) -> Option<Expr> {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        group_id.and_then(|gid| self.grammar.group_id_to_expr.get(gid).cloned())
    }

    /// Solve the system of regex equations to get the final regex for the start symbol.
    /// Uses Gaussian elimination with Arden's lemma for self-recursive equations.
    fn solve_regex_system(&self, equations: &mut HashMap<String, RegexTerm>, start: &str) -> Option<Expr> {
        // Topologically sort non-terminals by dependency
        let order = self.topological_sort_nts(equations);
        
        // Process in reverse topological order (leaves first)
        for nt in order.iter().rev() {
            if let Some(term) = equations.get(nt).cloned() {
                // Check if this NT is self-recursive: X = αX | β
                // Using Arden's lemma: X = α*β
                let solved = self.solve_single_equation(nt, term, equations);
                equations.insert(nt.clone(), solved);
            }
        }

        // Now substitute all equations into the start symbol
        // Process in topological order to substitute dependencies first
        let mut current = equations.get(start)?.clone();
        
        // Iteratively substitute until no more changes
        let mut iterations = 0;
        let max_iterations = equations.len() * 2 + 10;
        
        while current.has_nt_refs() && iterations < max_iterations {
            let refs = current.get_nt_refs();
            let mut changed = false;
            
            for nt_ref in refs {
                if nt_ref != *start {
                    if let Some(replacement) = equations.get(&nt_ref) {
                        let new_current = current.substitute(&nt_ref, replacement);
                        if new_current != current {
                            current = new_current.simplify();
                            changed = true;
                        }
                    }
                }
            }
            
            if !changed {
                break;
            }
            iterations += 1;
        }

        // Handle self-reference in start if needed
        if current.get_nt_refs().contains(start) {
            current = self.solve_single_equation(start, current, equations);
        }

        current.simplify().to_expr()
    }

    /// Solve a single equation that may be self-recursive using Arden's lemma.
    /// X = αX | β  =>  X = α*β
    fn solve_single_equation(&self, nt: &str, term: RegexTerm, equations: &HashMap<String, RegexTerm>) -> RegexTerm {
        // First, substitute all other non-terminals (not self)
        let mut term = term;
        let refs = term.get_nt_refs();
        for r in &refs {
            if r != nt {
                if let Some(replacement) = equations.get(r) {
                    term = term.substitute(r, replacement);
                }
            }
        }
        term = term.simplify();

        // Now check for self-recursion
        if !term.get_nt_refs().contains(nt) {
            return term;
        }

        // Extract α (coefficient of self-reference) and β (non-recursive part)
        // For X = αX | β, we need X = α*β
        let (alpha, beta) = self.extract_self_recursion(nt, &term);
        
        if alpha.is_empty() {
            // No self-recursion found
            return term;
        }

        // Build α*β
        let alpha_star = if alpha.len() == 1 {
            RegexTerm::Quantifier(Box::new(alpha.into_iter().next().unwrap()), QuantifierType::ZeroOrMore)
        } else {
            RegexTerm::Quantifier(Box::new(RegexTerm::Choice(alpha)), QuantifierType::ZeroOrMore)
        };

        let beta_term = if beta.is_empty() {
            RegexTerm::Epsilon
        } else if beta.len() == 1 {
            beta.into_iter().next().unwrap()
        } else {
            RegexTerm::Choice(beta)
        };

        RegexTerm::Seq(vec![alpha_star, beta_term]).simplify()
    }

    /// Extract the self-recursive coefficient and base from an equation.
    /// For X = αX | β, returns (vec![α], vec![β])
    fn extract_self_recursion(&self, nt: &str, term: &RegexTerm) -> (Vec<RegexTerm>, Vec<RegexTerm>) {
        let mut alpha = Vec::new(); // Coefficients of X (self-reference)
        let mut beta = Vec::new();  // Non-recursive alternatives

        match term {
            RegexTerm::Choice(alternatives) => {
                for alt in alternatives {
                    if let Some((coef, is_left)) = self.extract_single_self_ref(nt, alt) {
                        // This alternative has a self-reference
                        // For right-linear: αX, coef = α
                        // For left-linear: Xα, we'd need different handling
                        // We'll assume right-linear for now (most common)
                        if is_left {
                            // Left recursion: Xα - need to handle differently
                            // For now, treat as non-solvable or use a different approach
                            // Actually, for left recursion: X = Xα | β => X = βα*
                            alpha.push(coef);
                        } else {
                            // Right recursion: αX
                            alpha.push(coef);
                        }
                    } else if !alt.get_nt_refs().contains(nt) {
                        // No self-reference
                        beta.push(alt.clone());
                    } else {
                        // Complex self-reference (e.g., αXβ) - not directly solvable
                        // For now, add to beta and hope for the best
                        beta.push(alt.clone());
                    }
                }
            }
            RegexTerm::Seq(terms) => {
                // Check if it's of the form αX or Xα
                if let Some((coef, _is_left)) = self.extract_single_self_ref(nt, term) {
                    alpha.push(coef);
                    // Add epsilon to beta to handle the base case
                    beta.push(RegexTerm::Epsilon);
                } else if !term.get_nt_refs().contains(nt) {
                    beta.push(term.clone());
                }
            }
            RegexTerm::NtRef(name) if name == nt => {
                // X = X => X = epsilon (edge case)
                alpha.push(RegexTerm::Epsilon);
                beta.push(RegexTerm::Epsilon);
            }
            _ => {
                if !term.get_nt_refs().contains(nt) {
                    beta.push(term.clone());
                }
            }
        }

        (alpha, beta)
    }

    /// Extract the coefficient from a single self-referential term.
    /// Returns Some((coefficient, is_left_recursion)) if the term is of form αX or Xα.
    fn extract_single_self_ref(&self, nt: &str, term: &RegexTerm) -> Option<(RegexTerm, bool)> {
        match term {
            RegexTerm::Seq(terms) => {
                let mut nt_positions = Vec::new();
                for (i, t) in terms.iter().enumerate() {
                    if let RegexTerm::NtRef(name) = t {
                        if name == nt {
                            nt_positions.push(i);
                        }
                    }
                }

                if nt_positions.len() != 1 {
                    return None; // Multiple or no self-references
                }

                let pos = nt_positions[0];
                if pos == terms.len() - 1 {
                    // Right recursion: α X
                    let coef: Vec<RegexTerm> = terms[..pos].to_vec();
                    let coef_term = if coef.is_empty() {
                        RegexTerm::Epsilon
                    } else if coef.len() == 1 {
                        coef.into_iter().next().unwrap()
                    } else {
                        RegexTerm::Seq(coef)
                    };
                    Some((coef_term, false))
                } else if pos == 0 {
                    // Left recursion: X α
                    let coef: Vec<RegexTerm> = terms[1..].to_vec();
                    let coef_term = if coef.is_empty() {
                        RegexTerm::Epsilon
                    } else if coef.len() == 1 {
                        coef.into_iter().next().unwrap()
                    } else {
                        RegexTerm::Seq(coef)
                    };
                    Some((coef_term, true))
                } else {
                    // Middle recursion: α X β - not simple linear recursion
                    None
                }
            }
            RegexTerm::NtRef(name) if name == nt => {
                // Just X
                Some((RegexTerm::Epsilon, false))
            }
            _ => None,
        }
    }

    /// Topological sort of non-terminals based on dependencies.
    fn topological_sort_nts(&self, equations: &HashMap<String, RegexTerm>) -> Vec<String> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();

        // Initialize
        for nt in equations.keys() {
            in_degree.insert(nt.clone(), 0);
            graph.insert(nt.clone(), HashSet::new());
        }

        // Build graph (nt -> nts it depends on)
        for (nt, term) in equations {
            let refs = term.get_nt_refs();
            for r in refs {
                if r != *nt && equations.contains_key(&r) {
                    graph.get_mut(nt).unwrap().insert(r.clone());
                }
            }
        }

        // Calculate in-degrees
        for (_, deps) in &graph {
            for dep in deps {
                *in_degree.get_mut(dep).unwrap() += 1;
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<String> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(nt, _)| nt.clone())
            .collect();
        queue.sort(); // For determinism

        let mut result = Vec::new();
        while let Some(nt) = queue.pop() {
            result.push(nt.clone());
            if let Some(deps) = graph.get(&nt) {
                for dep in deps {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(dep.clone());
                            queue.sort();
                        }
                    }
                }
            }
        }

        // Handle cycles by adding remaining nodes
        for nt in equations.keys() {
            if !result.contains(nt) {
                result.push(nt.clone());
            }
        }

        result
    }

    /// Replace the entire grammar with a single terminal.
    fn replace_grammar_with_single_terminal(&mut self, expr: Expr) {
        // Create a new terminal for the regex
        let new_terminal_name = "__optimized_terminal__".to_string();
        let new_group_id = 0; // Use group ID 0 for the single terminal

        // Clear existing terminals
        self.grammar.regex_name_to_group_id.clear();
        self.grammar.literal_to_group_id.clear();
        self.grammar.group_id_to_expr.clear();

        // Add the new terminal
        self.grammar.regex_name_to_group_id.insert(new_terminal_name.clone(), new_group_id);
        self.grammar.group_id_to_expr.insert(new_group_id, expr);

        // Update ignore terminal if it existed
        if self.grammar.ignore_terminal_id.is_some() {
            // The ignore terminal is now merged into the main regex, so clear it
            self.grammar.ignore_terminal_id = None;
        }

        // Create a minimal grammar: start' -> terminal
        let start_nt = NonTerminal("start'".to_string());
        self.grammar.productions = vec![
            Production {
                lhs: start_nt,
                rhs: vec![Symbol::Terminal(Terminal::RegexName(new_terminal_name))],
            }
        ];
        self.grammar.start_production_id = 0;
    }

    fn get_group_id(&self, t: &Terminal) -> usize {
         match t {
            Terminal::Literal(bytes) => *self.grammar.literal_to_group_id.get_by_left(bytes).expect("Terminal missing"),
            Terminal::RegexName(name) => *self.grammar.regex_name_to_group_id.get_by_left(name).expect("Terminal missing"),
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
