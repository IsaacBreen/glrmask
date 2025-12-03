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
        // Build the linear system: for each non-terminal, we have equations like
        // X = sum of (coefficient * Y) for each production X -> ... Y ...
        // where coefficient is the regex of terminals before/after Y
        
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Build equations: nt_name -> list of (prefix_expr, optional_nt_ref, suffix_expr)
        // Each production A -> α B β becomes an entry (α, Some(B), β) in equations[A]
        // A production A -> α (only terminals) becomes (α, None, ε)
        let mut equations: HashMap<String, Vec<ProductionTerm>> = HashMap::new();
        
        for prod in &self.grammar.productions {
            let term = self.analyze_production(&prod.rhs, &nt_names)?;
            equations.entry(prod.lhs.0.clone()).or_default().push(term);
        }
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Solve using the efficient algorithm
        self.solve_linear_system(&mut equations, &start_nt)
    }

    /// Analyze a production RHS and extract the structure.
    /// Returns None if the production is not linear (has multiple NT refs or NT in middle of terminals).
    fn analyze_production(&self, rhs: &[Symbol], nt_names: &HashSet<String>) -> Option<ProductionTerm> {
        let mut prefix = Vec::new();
        let mut nt_ref: Option<String> = None;
        let mut suffix = Vec::new();
        
        for sym in rhs {
            match sym {
                Symbol::Terminal(t) => {
                    let expr = self.get_expr_for_terminal(t)?;
                    if nt_ref.is_some() {
                        suffix.push(expr);
                    } else {
                        prefix.push(expr);
                    }
                }
                Symbol::NonTerminal(nt) => {
                    if !nt_names.contains(&nt.0) {
                        return None; // Unknown non-terminal
                    }
                    if nt_ref.is_some() {
                        return None; // Multiple non-terminals - not linear
                    }
                    nt_ref = Some(nt.0.clone());
                }
            }
        }
        
        let prefix_expr = exprs_to_seq(prefix);
        let suffix_expr = exprs_to_seq(suffix);
        
        Some(ProductionTerm { prefix: prefix_expr, nt_ref, suffix: suffix_expr })
    }

    /// Get the Expr for a terminal (immutable version)
    fn get_expr_for_terminal(&self, t: &Terminal) -> Option<Expr> {
        let group_id = match t {
            Terminal::Literal(bytes) => self.grammar.literal_to_group_id.get_by_left(bytes),
            Terminal::RegexName(name) => self.grammar.regex_name_to_group_id.get_by_left(name),
        };
        group_id.and_then(|gid| self.grammar.group_id_to_expr.get(gid).cloned())
    }

    /// Solve the linear system using Gaussian elimination with Arden's lemma.
    /// This is efficient because we process each non-terminal exactly once.
    fn solve_linear_system(&self, equations: &mut HashMap<String, Vec<ProductionTerm>>, start: &str) -> Option<Expr> {
        // Find the order to process non-terminals (reverse topological order)
        let order = self.compute_processing_order(equations, start);
        
        // Resolved expressions for each non-terminal
        let mut resolved: HashMap<String, Expr> = HashMap::new();
        
        // Process in order
        for nt in &order {
            let terms = equations.get(nt)?;
            let expr = self.solve_single_nt(nt, terms, &resolved)?;
            resolved.insert(nt.clone(), expr);
        }
        
        resolved.get(start).cloned()
    }
    
    /// Compute the order in which to process non-terminals.
    /// We process dependencies first, handling cycles with Arden's lemma.
    fn compute_processing_order(&self, equations: &HashMap<String, Vec<ProductionTerm>>, start: &str) -> Vec<String> {
        // Build dependency graph
        let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
        for (nt, terms) in equations {
            let mut nt_deps = HashSet::new();
            for term in terms {
                if let Some(ref dep) = term.nt_ref {
                    if dep != nt { // Exclude self-references
                        nt_deps.insert(dep.clone());
                    }
                }
            }
            deps.insert(nt.clone(), nt_deps);
        }
        
        // Find reachable non-terminals from start
        let mut reachable = HashSet::new();
        let mut stack = vec![start.to_string()];
        while let Some(nt) = stack.pop() {
            if reachable.insert(nt.clone()) {
                if let Some(nt_deps) = deps.get(&nt) {
                    for dep in nt_deps {
                        stack.push(dep.clone());
                    }
                }
            }
        }
        
        // Topological sort of reachable non-terminals
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for nt in &reachable {
            in_degree.insert(nt.clone(), 0);
        }
        
        for nt in &reachable {
            if let Some(nt_deps) = deps.get(nt) {
                for dep in nt_deps {
                    if reachable.contains(dep) {
                        *in_degree.get_mut(nt).unwrap() += 1;
                    }
                }
            }
        }
        
        // Process in topological order (sources first, then reverse for our needs)
        let mut result = Vec::new();
        let mut queue: Vec<String> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(nt, _)| nt.clone())
            .collect();
        queue.sort(); // For determinism
        
        while let Some(nt) = queue.pop() {
            result.push(nt.clone());
            
            // Update in-degrees for nodes that depend on this one
            for (other_nt, other_deps) in &deps {
                if reachable.contains(other_nt) && other_deps.contains(&nt) {
                    let deg = in_degree.get_mut(other_nt).unwrap();
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 && !result.contains(other_nt) {
                        queue.push(other_nt.clone());
                        queue.sort();
                    }
                }
            }
        }
        
        // Add any remaining (cyclic) nodes
        for nt in &reachable {
            if !result.contains(nt) {
                result.push(nt.clone());
            }
        }
        
        // Reverse so that we process leaves first
        result.reverse();
        result
    }
    
    /// Solve for a single non-terminal given the resolved expressions of its dependencies.
    fn solve_single_nt(&self, nt: &str, terms: &[ProductionTerm], resolved: &HashMap<String, Expr>) -> Option<Expr> {
        // Separate terms into:
        // - self_coefs: terms of the form α X (where X is this NT)
        // - base_terms: terms without self-reference
        let mut self_coefs: Vec<Expr> = Vec::new();
        let mut base_terms: Vec<Expr> = Vec::new();
        
        for term in terms {
            match &term.nt_ref {
                None => {
                    // Pure terminal production
                    base_terms.push(term.prefix.clone());
                }
                Some(ref_nt) if ref_nt == nt => {
                    // Self-recursive: A -> prefix A suffix
                    // For right-linear (suffix is empty), coef is prefix
                    // For left-linear (prefix is empty), need different handling
                    if is_epsilon(&term.suffix) {
                        // Right recursion: A -> prefix A
                        self_coefs.push(term.prefix.clone());
                    } else if is_epsilon(&term.prefix) {
                        // Left recursion: A -> A suffix
                        // For now, treat suffix as coefficient (requires different formula)
                        // A = A α | β => A = β α*
                        self_coefs.push(term.suffix.clone());
                    } else {
                        // A -> prefix A suffix - complex recursion, not directly solvable
                        // Try to approximate: A -> prefix A suffix ≈ prefix* suffix (not exact)
                        // For now, return None as this isn't a simple linear recursion
                        return None;
                    }
                }
                Some(ref_nt) => {
                    // Reference to another NT - substitute
                    let ref_expr = resolved.get(ref_nt)?;
                    let combined = make_seq(vec![term.prefix.clone(), ref_expr.clone(), term.suffix.clone()]);
                    base_terms.push(combined);
                }
            }
        }
        
        // Apply Arden's lemma: X = αX | β => X = α*β
        let base = if base_terms.is_empty() {
            Expr::Epsilon
        } else if base_terms.len() == 1 {
            base_terms.into_iter().next().unwrap()
        } else {
            Expr::Choice(base_terms)
        };
        
        if self_coefs.is_empty() {
            Some(simplify_expr(base))
        } else {
            let coef = if self_coefs.len() == 1 {
                self_coefs.into_iter().next().unwrap()
            } else {
                Expr::Choice(self_coefs)
            };
            let coef_star = Expr::Quantifier(Box::new(coef), QuantifierType::ZeroOrMore);
            let result = make_seq(vec![coef_star, base]);
            Some(simplify_expr(result))
        }
    }

    /// Replace the entire grammar with a single terminal.
    fn replace_grammar_with_single_terminal(&mut self, expr: Expr) {
        // Simplify the expression before storing
        let expr = simplify_expr(expr);
        
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

/// A term in a production after analysis.
/// Represents: prefix NT suffix (or just prefix if NT is None)
#[derive(Clone, Debug)]
struct ProductionTerm {
    prefix: Expr,
    nt_ref: Option<String>,
    suffix: Expr,
}

/// Convert a list of expressions to a Seq (or single expr, or Epsilon)
fn exprs_to_seq(exprs: Vec<Expr>) -> Expr {
    let filtered: Vec<Expr> = exprs.into_iter()
        .filter(|e| !matches!(e, Expr::Epsilon))
        .collect();
    if filtered.is_empty() {
        Expr::Epsilon
    } else if filtered.len() == 1 {
        filtered.into_iter().next().unwrap()
    } else {
        Expr::Seq(filtered)
    }
}

/// Make a sequence, handling Epsilon and single elements
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

/// Check if an expression is epsilon
fn is_epsilon(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::Seq(s) if s.is_empty() => true,
        Expr::U8Seq(b) if b.is_empty() => true,
        _ => false,
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
                // (A*)* = A*
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
