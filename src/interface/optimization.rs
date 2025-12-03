use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
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

        if let Some(final_expr) = self.try_convert_to_regex() {
            self.replace_grammar_with_single_terminal(final_expr);
        }

        self.stats.final_productions = self.grammar.productions.len();
        self.stats.final_terminals = self.count_terminals();
    }

    /// Try to convert the entire grammar to a single regex using Gaussian elimination.
    fn try_convert_to_regex(&self) -> Option<Expr> {
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Build initial equations: each NT maps to a RegexTerm
        let mut equations: HashMap<String, RegexTerm> = HashMap::new();
        
        for prod in &self.grammar.productions {
            let term = self.production_rhs_to_regex_term(&prod.rhs, &nt_names)?;
            equations.entry(prod.lhs.0.clone())
                .and_modify(|existing| {
                    *existing = RegexTerm::choice(vec![existing.clone(), term.clone()]);
                })
                .or_insert(term);
        }
        
        // Gaussian elimination: eliminate NTs one by one until only start remains
        let nt_list: Vec<String> = nt_names.iter().cloned().collect();
        
        // Process NTs in reverse order (typically eliminates dependencies first)
        for nt_to_eliminate in nt_list.iter().rev() {
            if nt_to_eliminate == &start_nt {
                continue; // Keep the start NT for last
            }
            
            // Get the equation for this NT
            let Some(nt_eq) = equations.remove(nt_to_eliminate) else {
                continue;
            };
            
            // Solve for this NT (apply Arden's lemma if self-recursive)
            let solved = solve_single_equation(&nt_eq, nt_to_eliminate)?;
            
            // Substitute into all remaining equations
            for (_other_nt, other_eq) in equations.iter_mut() {
                *other_eq = substitute_nt(other_eq, nt_to_eliminate, &solved);
            }
        }
        
        // Now solve the start NT
        let start_eq = equations.remove(&start_nt)?;
        let solved_start = solve_single_equation(&start_eq, &start_nt)?;
        
        // Convert the final RegexTerm to Expr
        regex_term_to_expr(&solved_start)
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
                        None
                    }
                }
            }
        }).collect();

        terms.map(|ts| RegexTerm::seq(ts))
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

/// A term that may reference non-terminals
#[derive(Clone, Debug)]
enum RegexTerm {
    Epsilon,
    Concrete(Expr),
    NtRef(String),
    Seq(Vec<RegexTerm>),
    Choice(Vec<RegexTerm>),
    Star(Box<RegexTerm>),
}

impl RegexTerm {
    fn seq(terms: Vec<RegexTerm>) -> RegexTerm {
        let mut flat = Vec::new();
        for t in terms {
            match t {
                RegexTerm::Epsilon => {}
                RegexTerm::Seq(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.is_empty() {
            RegexTerm::Epsilon
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            RegexTerm::Seq(flat)
        }
    }
    
    fn choice(terms: Vec<RegexTerm>) -> RegexTerm {
        let mut flat = Vec::new();
        for t in terms {
            match t {
                RegexTerm::Choice(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.is_empty() {
            RegexTerm::Epsilon
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            RegexTerm::Choice(flat)
        }
    }
    
    fn star(inner: RegexTerm) -> RegexTerm {
        match inner {
            RegexTerm::Epsilon => RegexTerm::Epsilon,
            RegexTerm::Star(s) => RegexTerm::Star(s), // (a*)* = a*
            other => RegexTerm::Star(Box::new(other)),
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
        return Some(RegexTerm::choice(base));
    }
    
    // Extract coefficients from recursive alternatives
    // For X = αX | Xβ | γ, we need to handle both left and right recursion
    let mut right_coefs: Vec<RegexTerm> = Vec::new();
    let mut left_coefs: Vec<RegexTerm> = Vec::new();
    
    for rec in &recursive {
        if let Some((prefix, suffix)) = extract_coef(rec, nt) {
            match (is_epsilon(&prefix), is_epsilon(&suffix)) {
                (true, _) => {
                    // X suffix => left recursion
                    left_coefs.push(suffix);
                }
                (_, true) => {
                    // prefix X => right recursion
                    right_coefs.push(prefix);
                }
                _ => {
                    // prefix X suffix => not directly solvable
                    return None;
                }
            }
        } else {
            return None;
        }
    }
    
    // Build the solution
    let base_term = if base.is_empty() {
        RegexTerm::Epsilon
    } else {
        RegexTerm::choice(base)
    };
    
    // Handle right recursion: X = αX | β => X = α*β
    let mut result = base_term;
    if !right_coefs.is_empty() {
        let right_coef = RegexTerm::choice(right_coefs);
        let right_star = RegexTerm::star(right_coef);
        result = RegexTerm::seq(vec![right_star, result]);
    }
    
    // Handle left recursion: X = Xα | β => X = βα*
    if !left_coefs.is_empty() {
        let left_coef = RegexTerm::choice(left_coefs);
        let left_star = RegexTerm::star(left_coef);
        result = RegexTerm::seq(vec![result, left_star]);
    }
    
    Some(result)
}

/// Separate a term into recursive and non-recursive alternatives
fn separate_recursive_alts(term: &RegexTerm, nt: &str) -> (Vec<RegexTerm>, Vec<RegexTerm>) {
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
        _ if term.contains_nt(nt) => (vec![term.clone()], vec![]),
        _ => (vec![], vec![term.clone()]),
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
            if !matches!(&parts[pos], RegexTerm::NtRef(n) if n == nt) {
                return None;
            }
            
            let prefix = RegexTerm::seq(parts[..pos].to_vec());
            let suffix = RegexTerm::seq(parts[pos+1..].to_vec());
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
fn substitute_nt(term: &RegexTerm, nt: &str, replacement: &RegexTerm) -> RegexTerm {
    if !term.contains_nt(nt) {
        return term.clone();
    }
    
    match term {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => term.clone(),
        RegexTerm::NtRef(n) if n == nt => replacement.clone(),
        RegexTerm::NtRef(_) => term.clone(),
        RegexTerm::Seq(parts) => {
            let new_parts: Vec<RegexTerm> = parts.iter()
                .map(|p| substitute_nt(p, nt, replacement))
                .collect();
            RegexTerm::seq(new_parts)
        }
        RegexTerm::Choice(alts) => {
            let new_alts: Vec<RegexTerm> = alts.iter()
                .map(|a| substitute_nt(a, nt, replacement))
                .collect();
            RegexTerm::choice(new_alts)
        }
        RegexTerm::Star(inner) => {
            RegexTerm::star(substitute_nt(inner, nt, replacement))
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
                .map(regex_term_to_expr)
                .collect();
            exprs.map(|es| make_seq(es))
        }
        RegexTerm::Choice(alts) => {
            let exprs: Option<Vec<Expr>> = alts.iter()
                .map(regex_term_to_expr)
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
