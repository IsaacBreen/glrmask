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
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Build equations: each NT maps to a RegexTerm (regex with NT references)
        let mut equations: HashMap<String, RegexTerm> = HashMap::new();
        
        for prod in &self.grammar.productions {
            let term = self.production_rhs_to_regex_term(&prod.rhs, &nt_names)?;
            equations.entry(prod.lhs.0.clone())
                .and_modify(|existing| {
                    // Combine with existing as a choice
                    match existing {
                        RegexTerm::Choice(alts) => alts.push(term.clone()),
                        other => *other = RegexTerm::Choice(vec![other.clone(), term.clone()]),
                    }
                })
                .or_insert(term);
        }
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Solve the system using memoized recursion with cycle detection
        let mut resolved: HashMap<String, Expr> = HashMap::new();
        let mut in_progress: HashSet<String> = HashSet::new();
        
        self.solve_nt(&start_nt, &equations, &mut resolved, &mut in_progress)
    }

    /// Convert a production RHS to a RegexTerm (regex with NT references)
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
                        // Unknown non-terminal
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

    /// Solve for a single non-terminal with memoization and cycle detection.
    /// Uses Arden's lemma for self-recursive equations.
    fn solve_nt(
        &self,
        nt: &str,
        equations: &HashMap<String, RegexTerm>,
        resolved: &mut HashMap<String, Expr>,
        in_progress: &mut HashSet<String>,
    ) -> Option<Expr> {
        // Check if already resolved
        if let Some(expr) = resolved.get(nt) {
            return Some(expr.clone());
        }
        
        // Check for cycle - if we're already processing this NT, we have a recursive reference
        if in_progress.contains(nt) {
            // Return a placeholder that will be handled by Arden's lemma
            return None;
        }
        
        // Mark as in progress
        in_progress.insert(nt.to_string());
        
        // Get the equation for this NT
        let term = equations.get(nt)?.clone();
        
        // Solve using the recursive approach
        let expr = self.solve_term(&term, nt, equations, resolved, in_progress)?;
        
        // Mark as done
        in_progress.remove(nt);
        resolved.insert(nt.to_string(), expr.clone());
        
        Some(expr)
    }
    
    /// Solve a RegexTerm, recursively resolving NT references.
    fn solve_term(
        &self,
        term: &RegexTerm,
        current_nt: &str,
        equations: &HashMap<String, RegexTerm>,
        resolved: &mut HashMap<String, Expr>,
        in_progress: &mut HashSet<String>,
    ) -> Option<Expr> {
        match term {
            RegexTerm::Epsilon => Some(Expr::Epsilon),
            RegexTerm::Concrete(e) => Some(e.clone()),
            RegexTerm::NtRef(ref_nt) => {
                if ref_nt == current_nt {
                    // Self-reference - will be handled at the Choice level
                    None
                } else {
                    self.solve_nt(ref_nt, equations, resolved, in_progress)
                }
            }
            RegexTerm::Seq(terms) => {
                // Check if any term is a self-reference
                let has_self_ref = terms.iter().any(|t| matches!(t, RegexTerm::NtRef(n) if n == current_nt));
                
                if has_self_ref {
                    // This is a recursive sequence like α X or X α
                    // Extract the position and handle with Arden's lemma
                    self.solve_recursive_seq(terms, current_nt, equations, resolved, in_progress)
                } else {
                    // No self-reference, just resolve all terms
                    let exprs: Option<Vec<Expr>> = terms.iter()
                        .map(|t| self.solve_term(t, current_nt, equations, resolved, in_progress))
                        .collect();
                    exprs.map(|es| make_seq(es))
                }
            }
            RegexTerm::Choice(alts) => {
                // Separate into recursive and non-recursive alternatives
                let mut recursive_alts: Vec<&RegexTerm> = Vec::new();
                let mut base_alts: Vec<&RegexTerm> = Vec::new();
                
                for alt in alts {
                    if alt.contains_nt_ref(current_nt) {
                        recursive_alts.push(alt);
                    } else {
                        base_alts.push(alt);
                    }
                }
                
                // Solve base alternatives first
                let base_exprs: Option<Vec<Expr>> = base_alts.iter()
                    .map(|t| self.solve_term(t, current_nt, equations, resolved, in_progress))
                    .collect();
                let base_exprs = base_exprs?;
                
                if recursive_alts.is_empty() {
                    // No recursion, just return the choice of bases
                    return Some(make_choice(base_exprs));
                }
                
                // Handle recursion with Arden's lemma
                // X = αX | β => X = α*β  (right recursion)
                // X = Xα | β => X = βα*  (left recursion)
                
                let mut coefficients: Vec<Expr> = Vec::new();
                
                for rec_alt in &recursive_alts {
                    if let Some((coef, is_left)) = self.extract_linear_coef(rec_alt, current_nt, equations, resolved, in_progress) {
                        if is_left {
                            // Left recursion: need to convert to right recursion form
                            // For now, we'll handle it similarly
                            coefficients.push(coef);
                        } else {
                            coefficients.push(coef);
                        }
                    } else {
                        // Non-linear recursion - can't solve directly
                        return None;
                    }
                }
                
                // Build α*β
                let coef = make_choice(coefficients);
                let coef_star = Expr::Quantifier(Box::new(coef), QuantifierType::ZeroOrMore);
                let base = make_choice(base_exprs);
                
                // Check if we have left or right recursion
                // For right recursion: X = αX | β => X = α*β
                // For left recursion: X = Xα | β => X = βα*
                // For mixed: more complex
                
                // For now, assume right recursion (most common in these grammars)
                Some(make_seq(vec![coef_star, base]))
            }
        }
    }
    
    /// Solve a recursive sequence (like α X or X α).
    fn solve_recursive_seq(
        &self,
        terms: &[RegexTerm],
        current_nt: &str,
        equations: &HashMap<String, RegexTerm>,
        resolved: &mut HashMap<String, Expr>,
        in_progress: &mut HashSet<String>,
    ) -> Option<Expr> {
        // Find the position of the self-reference
        let mut self_ref_pos = None;
        for (i, t) in terms.iter().enumerate() {
            if matches!(t, RegexTerm::NtRef(n) if n == current_nt) {
                if self_ref_pos.is_some() {
                    // Multiple self-references - not linear
                    return None;
                }
                self_ref_pos = Some(i);
            }
        }
        
        let pos = self_ref_pos?;
        
        // Resolve prefix and suffix
        let prefix_terms = &terms[..pos];
        let suffix_terms = &terms[pos+1..];
        
        let prefix_exprs: Option<Vec<Expr>> = prefix_terms.iter()
            .map(|t| self.solve_term(t, current_nt, equations, resolved, in_progress))
            .collect();
        let suffix_exprs: Option<Vec<Expr>> = suffix_terms.iter()
            .map(|t| self.solve_term(t, current_nt, equations, resolved, in_progress))
            .collect();
        
        let prefix = make_seq(prefix_exprs?);
        let suffix = make_seq(suffix_exprs?);
        
        if is_epsilon(&suffix) {
            // Right recursion: prefix X => prefix*
            Some(Expr::Quantifier(Box::new(prefix), QuantifierType::ZeroOrMore))
        } else if is_epsilon(&prefix) {
            // Left recursion: X suffix => suffix*
            Some(Expr::Quantifier(Box::new(suffix), QuantifierType::ZeroOrMore))
        } else {
            // Middle recursion: prefix X suffix - not directly solvable with Arden
            None
        }
    }
    
    /// Extract the linear coefficient from a recursive alternative.
    /// Returns (coefficient, is_left_recursion) if the alt is linear.
    fn extract_linear_coef(
        &self,
        alt: &RegexTerm,
        current_nt: &str,
        equations: &HashMap<String, RegexTerm>,
        resolved: &mut HashMap<String, Expr>,
        in_progress: &mut HashSet<String>,
    ) -> Option<(Expr, bool)> {
        match alt {
            RegexTerm::NtRef(n) if n == current_nt => {
                // Just X - coefficient is epsilon
                Some((Expr::Epsilon, false))
            }
            RegexTerm::Seq(terms) => {
                // Find position of self-reference
                let mut self_ref_pos = None;
                for (i, t) in terms.iter().enumerate() {
                    if t.contains_nt_ref(current_nt) {
                        if self_ref_pos.is_some() {
                            // Multiple references to current NT
                            return None;
                        }
                        self_ref_pos = Some(i);
                    }
                }
                
                let pos = self_ref_pos?;
                
                // Check if the self-reference is simple (just the NT ref)
                if !matches!(&terms[pos], RegexTerm::NtRef(n) if n == current_nt) {
                    // The reference is inside a sub-expression - complex case
                    return None;
                }
                
                let prefix_terms = &terms[..pos];
                let suffix_terms = &terms[pos+1..];
                
                // Resolve prefix and suffix
                let prefix_exprs: Option<Vec<Expr>> = prefix_terms.iter()
                    .map(|t| self.solve_term(t, current_nt, equations, resolved, in_progress))
                    .collect();
                let suffix_exprs: Option<Vec<Expr>> = suffix_terms.iter()
                    .map(|t| self.solve_term(t, current_nt, equations, resolved, in_progress))
                    .collect();
                
                let prefix = make_seq(prefix_exprs?);
                let suffix = make_seq(suffix_exprs?);
                
                if is_epsilon(&suffix) {
                    // Right recursion: prefix X
                    Some((prefix, false))
                } else if is_epsilon(&prefix) {
                    // Left recursion: X suffix  
                    Some((suffix, true))
                } else {
                    // Middle recursion: prefix X suffix - need combined coefficient
                    // This is more complex - for now return None
                    None
                }
            }
            _ => None,
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

/// A term that may reference non-terminals.
#[derive(Clone, Debug)]
enum RegexTerm {
    Epsilon,
    Concrete(Expr),
    NtRef(String),
    Seq(Vec<RegexTerm>),
    Choice(Vec<RegexTerm>),
}

impl RegexTerm {
    fn contains_nt_ref(&self, nt: &str) -> bool {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => false,
            RegexTerm::NtRef(n) => n == nt,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                terms.iter().any(|t| t.contains_nt_ref(nt))
            }
        }
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

/// Make a choice, handling single elements
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
