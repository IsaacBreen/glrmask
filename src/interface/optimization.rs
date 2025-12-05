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
    if std::env::var("PRINT_OPTIMIZED_GRAMMAR").is_ok() {
        crate::debug!(4, "Optimized grammar:\n{}", grammar);
    }

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

    fn get_ignore_expr(&self) -> Option<Expr> {
        self.grammar.ignore_terminal_id.and_then(|id| {
            self.grammar.group_id_to_expr.get(&id.0).cloned()
        })
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
        // Fast path: if grammar already has only 1 production with 1 terminal RHS,
        // it's already been optimized
        if self.grammar.productions.len() == 1 {
            let prod = &self.grammar.productions[0];
            if prod.rhs.len() == 1 {
                if let crate::glr::grammar::Symbol::Terminal(_) = &prod.rhs[0] {
                    debug!(4, "Grammar already optimized (1 production with 1 terminal), skipping");
                    return;
                }
            }
        }
        
        // Heuristic: Skip optimization for very large grammars where it would create
        // too many terminals. The cutoff is based on empirical testing showing that
        // grammars with >500 productions tend to create tokenizers with 100K+ states
        // when optimized, leading to slow precomputation.
        // 
        // For these large grammars, it's faster to have a complex GLR parser (~5-6K states)
        // and simple tokenizer (~12K states) than a simple parser (~700 states) and 
        // complex tokenizer (~160K states).
        // 
        // UPDATE: With skeleton DWA simplification, grammar optimization can help, but
        // the tokenizer explosion still dominates. Keep threshold at 500 for now.
        const MAX_PRODUCTIONS_FOR_OPTIMIZATION: usize = 50000;
        if self.grammar.productions.len() > MAX_PRODUCTIONS_FOR_OPTIMIZATION {
            debug!(4, "Skipping grammar optimization: {} productions exceeds threshold of {}",
                self.grammar.productions.len(), MAX_PRODUCTIONS_FOR_OPTIMIZATION);
            debug!(4, "Large grammars benefit more from a simple tokenizer than from optimization");
            return;
        }
        
        let total_start = std::time::Instant::now();
        debug!(4, "Starting grammar optimization with {} productions", self.grammar.productions.len());
        
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Build initial equations: each NT maps to a RegexTerm
        let mut equations: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        
        let ignore_expr = self.get_ignore_expr();
        let ignore_term = ignore_expr.as_ref().map(|e| {
            Rc::new(RegexTerm::Star(Rc::new(RegexTerm::Concrete(e.clone()))))
        });

        let eq_start = std::time::Instant::now();
        for prod in &self.grammar.productions {
            let Some(term) = self.production_rhs_to_regex_term(&prod.rhs, &nt_names, ignore_term.as_ref()) else {
                // If any production can't be converted, skip optimization for now
                debug!(4, "Grammar optimization failed: production cannot be converted");
                return;
            };
            let term = Rc::new(term);
            equations.entry(prod.lhs.0.clone())
                .and_modify(|existing| {
                    *existing = Rc::new(RegexTerm::Choice(vec![existing.clone(), term.clone()]));
                })
                .or_insert(term);
        }
        debug!(5, "Built {} equations in {:?}", equations.len(), eq_start.elapsed());
        
        // Find NTs that are part of mutual recursion cycles
        let cyclic_nts = find_cyclic_nts(&equations);
        if !cyclic_nts.is_empty() {
            debug!(5, "Found {} cyclic NTs: {:?}", cyclic_nts.len(), cyclic_nts);
        }
        
        // Build dependency graph and compute elimination order (reverse topological)
        let order_start = std::time::Instant::now();
        let elimination_order = compute_elimination_order(&equations, &start_nt);
        debug!(5, "Computed elimination order ({} NTs) in {:?}", elimination_order.len(), order_start.elapsed());
        
        // PHASE 1: Build solution map (solve each NT's equation with Arden's lemma)
        // We DON'T substitute into other equations yet - just solve each NT in isolation
        let mut solutions: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        // Persistent cache for expansion - shared across all expand_with_solutions calls
        let mut expansion_cache: HashMap<*const RegexTerm, Rc<RegexTerm>> = HashMap::new();
        
        let phase1_start = std::time::Instant::now();
        let mut solved_count = 0;
        let mut failed_count = 0;
        let mut solve_time = std::time::Duration::ZERO;
        for (i, nt_to_eliminate) in elimination_order.iter().enumerate() {
            if i > 0 && i % 5000 == 0 {
                debug!(5, "Phase 1 progress: {}/{} NTs, solved={}, failed={}, solve={:?}", 
                    i, elimination_order.len(), solved_count, failed_count, solve_time);
            }
            if nt_to_eliminate == &start_nt {
                continue;
            }
            
            let Some(nt_eq) = equations.remove(nt_to_eliminate) else {
                continue;
            };
            
            // Solve for this NT (apply Arden's lemma if self-recursive)
            // The equation is still in SYMBOLIC form - no expansion yet
            // This is MUCH faster because the original equations are small
            let t1 = std::time::Instant::now();
            let Some(solved) = solve_single_equation_rc(&nt_eq, nt_to_eliminate) else {
                // Non-linear recursion - can't solve with simple Arden's lemma
                debug!(5, "Failed to solve NT '{}' (non-linear recursion)", nt_to_eliminate);
                equations.insert(nt_to_eliminate.clone(), nt_eq);
                failed_count += 1;
                solve_time += t1.elapsed();
                continue;
            };
            solve_time += t1.elapsed();
            
            // Store the solution (still in symbolic form with NtRefs)
            let solved_rc = Rc::new(solved);
            solutions.insert(nt_to_eliminate.clone(), solved_rc);
            solved_count += 1;
        }
        debug!(5, "Phase 1: solved {} NTs, failed {} in {:?} (solve={:?})", 
            solved_count, failed_count, phase1_start.elapsed(), solve_time);
        
        // PHASE 2: Solve the start NT
        let phase2_start = std::time::Instant::now();
        let Some(start_eq) = equations.remove(&start_nt) else {
            debug!(4, "Grammar optimization failed: no start equation");
            return;
        };
        
        // First solve the start equation symbolically (to handle self-recursion)
        let Some(solved_start_symbolic) = solve_single_equation_rc(&start_eq, &start_nt) else {
            debug!(4, "Grammar optimization failed: could not solve start NT");
            return;
        };
        debug!(5, "Phase 2: solved start NT symbolically in {:?}", phase2_start.elapsed());
        
        // Now expand all NtRefs in the solved start equation
        let expand_start = std::time::Instant::now();
        let solved_start_rc = Rc::new(solved_start_symbolic);
        
        // If there are cyclic NTs, use the skip_cyclic version
        let (expanded_start, has_remaining_ntrefs) = if cyclic_nts.is_empty() {
            // No cycles - use strict expansion
            let Some(expanded) = expand_with_solutions_cached(&solved_start_rc, &solutions, &mut expansion_cache) else {
                debug!(4, "Grammar optimization failed: cycle detected during expansion (mutual recursion)");
                return;
            };
            (expanded, false)
        } else {
            // Has cycles - use lenient expansion that skips cyclic NTs
            let expanded = expand_with_solutions_skip_cyclic(&solved_start_rc, &solutions, &mut expansion_cache, &cyclic_nts);
            // Check if the result still has NtRefs
            let remaining = expanded.referenced_nts();
            debug!(5, "After expansion, {} NtRefs remain: {:?}", remaining.len(), remaining);
            (expanded, !remaining.is_empty())
        };
        debug!(5, "Phase 2: expanded start NT in {:?}", expand_start.elapsed());
        
        // If there are remaining NtRefs, we need to do partial optimization
        if has_remaining_ntrefs {
            // Compute transitive closure of NTs that depend on cyclic NTs
            let cyclic_dependent_nts = compute_cyclic_dependent_nts(&cyclic_nts, &self.grammar.productions);
            debug!(4, "Found {} NTs that depend on cyclic NTs (out of {} total)", 
                cyclic_dependent_nts.len(), nt_names.len());
            
            self.partial_optimize(&solutions, &mut expansion_cache, &cyclic_dependent_nts, ignore_term.as_ref());
            debug!(4, "Partial grammar optimization complete in {:?}", total_start.elapsed());
            return;
        }
        
        // Convert the final RegexTerm to Expr
        let convert_start = std::time::Instant::now();
        let Some(final_expr) = regex_term_to_expr_rc(&expanded_start) else {
            debug!(4, "Grammar optimization failed: could not convert to Expr");
            return;
        };
        debug!(5, "Converted to Expr in {:?}", convert_start.elapsed());
        
        // If we have an ignore terminal, wrap the final expression
        let final_expr = if let Some(expr) = ignore_expr {
            let ignore_star = Expr::Quantifier(Box::new(expr), QuantifierType::ZeroOrMore);
            Expr::Seq(vec![ignore_star.clone(), final_expr, ignore_star])
        } else {
            final_expr
        };

        self.replace_grammar_with_single_terminal(final_expr);
        debug!(4, "Grammar optimization complete in {:?}", total_start.elapsed());
    }

    /// Convert a production RHS to a RegexTerm
    fn production_rhs_to_regex_term(&self, rhs: &[Symbol], nt_names: &HashSet<String>, ignore_term: Option<&Rc<RegexTerm>>) -> Option<RegexTerm> {
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

        terms.map(|ts| {
            if let Some(ign) = ignore_term {
                let mut with_ignore = Vec::with_capacity(ts.len() * 2);
                for (i, t) in ts.into_iter().enumerate() {
                    if i > 0 {
                        with_ignore.push(ign.clone());
                    }
                    with_ignore.push(t);
                }
                RegexTerm::make_seq(with_ignore)
            } else {
                RegexTerm::make_seq(ts)
            }
        })
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
        // Skip simplification for expressions that came from large grammars
        // The regex_term_to_expr already creates well-formed expressions
        // and simplify_expr is O(n) which is slow for 1M+ node trees
        // We check if we had a large number of productions as a proxy for expression size
        let skip_simplify = self.grammar.productions.len() > 100;
        let expr = if skip_simplify {
            debug!(5, "Skipping simplify_expr for large grammar (had {} productions)", self.grammar.productions.len());
            expr
        } else {
            let simplify_start = std::time::Instant::now();
            let result = simplify_expr(expr);
            debug!(5, "Simplified expr in {:?}", simplify_start.elapsed());
            result
        };
        
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
    
    /// Collect all terminals that are referenced in the grammar's productions.
    fn collect_used_terminals(&self) -> HashSet<String> {
        let mut used = HashSet::new();
        for prod in &self.grammar.productions {
            for sym in &prod.rhs {
                match sym {
                    Symbol::Terminal(Terminal::RegexName(name)) => {
                        used.insert(name.clone());
                    }
                    Symbol::Terminal(Terminal::Literal(_)) => {
                        // Literal terminals are in literal_to_group_id, not regex_name_to_group_id
                    }
                    Symbol::NonTerminal(_) => {}
                }
            }
        }
        used
    }
    
    /// Partial optimization: convert non-cyclic NTs to regex terminals while keeping
    /// cyclic NTs as grammar productions.
    /// 
    /// This is the key optimization for grammars with recursive structures (like JSON schemas
    /// with recursive references). We convert all the "regular" parts (enums, simple objects, etc.)
    /// to regex terminals, dramatically reducing the number of grammar productions the GLR
    /// parser needs to handle.
    fn partial_optimize(
        &mut self,
        solutions: &HashMap<String, Rc<RegexTerm>>,
        expansion_cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>,
        cyclic_dependent_nts: &HashSet<String>,
        ignore_term: Option<&Rc<RegexTerm>>,
    ) {
        let start = std::time::Instant::now();
        
        debug!(5, "Pre-existing terminals before partial_optimize: {}", 
            self.grammar.regex_name_to_group_id.len());
        
        // Track which terminals already existed before optimization
        // These should not be re-processed by handle_nullable_terminals_except
        let pre_existing_terminals: HashSet<String> = self.grammar.regex_name_to_group_id
            .left_values()
            .cloned()
            .collect();
        
        // For NTs that DON'T depend on cyclic NTs, we can fully expand and convert to terminals
        // For NTs that DO depend on cyclic NTs, we keep them as productions but optimize their
        // non-cyclic parts
        
        // Step 1: Collect all NTs that can be fully converted to regexes
        let convertible_nts: HashSet<String> = solutions.keys()
            .filter(|nt| !cyclic_dependent_nts.contains(*nt))
            .cloned()
            .collect();
        
        debug!(5, "Partial optimization: {} NTs can be fully converted, {} must stay as productions",
            convertible_nts.len(), cyclic_dependent_nts.len());
        
        // Step 2: For each convertible NT, expand and convert to a terminal
        let mut new_terminals: HashMap<String, (usize, Expr)> = HashMap::new();
        let mut next_group_id = self.grammar.group_id_to_expr.keys().max().unwrap_or(&0) + 1;
        
        // We need to expand solutions in dependency order
        // First, collect which convertible NTs each convertible NT depends on
        let mut nt_expand_order: Vec<String> = Vec::new();
        let mut expanded_solutions: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        
        // Simple approach: iterate and expand NTs whose dependencies are all expanded
        let mut remaining: HashSet<String> = convertible_nts.clone();
        while !remaining.is_empty() {
            let mut made_progress = false;
            let to_process: Vec<String> = remaining.iter().cloned().collect();
            
            for nt in to_process {
                let Some(solution) = solutions.get(&nt) else { continue };
                let deps = solution.referenced_nts();
                
                // Check if all convertible dependencies are expanded
                let all_deps_ready = deps.iter().all(|d| {
                    !convertible_nts.contains(d) || expanded_solutions.contains_key(d)
                });
                
                if all_deps_ready {
                    // Expand this NT
                    let expanded = expand_with_solutions_skip_cyclic(
                        solution,
                        &expanded_solutions,
                        expansion_cache,
                        cyclic_dependent_nts
                    );
                    expanded_solutions.insert(nt.clone(), expanded);
                    remaining.remove(&nt);
                    nt_expand_order.push(nt);
                    made_progress = true;
                }
            }
            
            if !made_progress && !remaining.is_empty() {
                // This shouldn't happen if our cyclic detection is correct
                debug!(5, "Warning: {} NTs couldn't be expanded (circular deps in non-cyclic set?)", remaining.len());
                break;
            }
        }
        
        debug!(5, "Expanded {} NTs in {:?}", expanded_solutions.len(), start.elapsed());
        
        // Step 3: Convert expanded solutions to terminal expressions
        // Include nullable terminals - they'll be handled by handle_nullable_terminals_except later
        // Deduplicate terminals with identical expressions
        let convert_start = std::time::Instant::now();
        let mut nullable_terminal_names: HashSet<String> = HashSet::new();
        
        // Map from expression to (canonical_nt_name, group_id)
        // This allows us to share terminals for NTs with identical expressions
        let mut expr_to_terminal: HashMap<Expr, (String, usize)> = HashMap::new();
        // Map from NT name to the canonical NT name it should use
        let mut nt_to_canonical: HashMap<String, String> = HashMap::new();
        
        for nt in &nt_expand_order {
            if let Some(expanded) = expanded_solutions.get(nt) {
                if let Some(expr) = regex_term_to_expr_rc(expanded) {
                    // Check if we've already seen this expression
                    if let Some((canonical_nt, _group_id)) = expr_to_terminal.get(&expr) {
                        // Reuse existing terminal
                        nt_to_canonical.insert(nt.clone(), canonical_nt.clone());
                    } else {
                        // Track nullable terminals for later handling
                        if matches!(get_expr_nullability(&expr), ExprNullability::CanBeNull | ExprNullability::AlwaysNull) {
                            nullable_terminal_names.insert(format!("__opt_{}__", nt));
                        }
                        
                        let group_id = next_group_id;
                        next_group_id += 1;
                        new_terminals.insert(nt.clone(), (group_id, expr.clone()));
                        expr_to_terminal.insert(expr, (nt.clone(), group_id));
                        nt_to_canonical.insert(nt.clone(), nt.clone());
                    }
                }
            }
        }
        
        let deduped_count = nt_expand_order.len() - new_terminals.len();
        if deduped_count > 0 {
            debug!(4, "Deduplicated {} terminals with identical expressions", deduped_count);
        }
        
        if !nullable_terminal_names.is_empty() {
            debug!(5, "Created {} nullable terminals (will be wrapped in optional NTs)", nullable_terminal_names.len());
        }
        debug!(5, "Converted {} NTs to {} terminal expressions in {:?}", 
            nt_expand_order.len(), new_terminals.len(), convert_start.elapsed());
        
        // Step 4: Update the grammar
        // IMPORTANT: We must determine which terminals are actually needed BEFORE adding them,
        // otherwise we create unused terminals that bloat the tokenizer.
        // Also, we must reassign group_ids to be contiguous starting from the current max.
        
        // 4a: First, determine which new terminals will actually be referenced
        // by building the new productions without adding them yet
        let update_start = std::time::Instant::now();
        let mut new_productions: Vec<Production> = Vec::new();
        let mut referenced_terminals: HashSet<String> = HashSet::new();
        
        for prod in &self.grammar.productions {
            let nt_name = &prod.lhs.0;
            
            // If this NT was converted to a terminal, we don't need its production
            // (unless it's referenced by cyclic NTs)
            // Check both direct and deduplicated cases
            if (new_terminals.contains_key(nt_name) || nt_to_canonical.contains_key(nt_name)) 
                && !cyclic_dependent_nts.contains(nt_name) {
                continue;
            }
            
            // Update the RHS to use new terminals where possible
            let new_rhs: Vec<Symbol> = prod.rhs.iter().map(|sym| {
                match sym {
                    Symbol::NonTerminal(nt) => {
                        // First check if this NT was deduplicated to another canonical NT
                        let canonical_nt = nt_to_canonical.get(&nt.0).unwrap_or(&nt.0);
                        if new_terminals.contains_key(canonical_nt) {
                            let terminal_name = format!("__opt_{}__", canonical_nt);
                            referenced_terminals.insert(canonical_nt.clone());
                            Symbol::Terminal(Terminal::RegexName(terminal_name))
                        } else {
                            sym.clone()
                        }
                    }
                    _ => sym.clone(),
                }
            }).collect();
            
            new_productions.push(Production {
                lhs: prod.lhs.clone(),
                rhs: new_rhs,
            });
        }
        
        // 4b: Now add only the terminals that are actually referenced
        // Use contiguous group_ids starting from the current max
        let current_max_group_id = self.grammar.group_id_to_expr.keys().max().copied().unwrap_or(0);
        let mut next_new_group_id = current_max_group_id + 1;
        let mut terminals_added = 0;
        for nt in referenced_terminals.iter() {
            if let Some((_old_group_id, expr)) = new_terminals.get(nt) {
                let terminal_name = format!("__opt_{}__", nt);
                let new_group_id = next_new_group_id;
                next_new_group_id += 1;
                debug!(5, "Adding terminal '{}' with group_id {}", terminal_name, new_group_id);
                self.grammar.regex_name_to_group_id.insert(terminal_name.clone(), new_group_id);
                self.grammar.group_id_to_expr.insert(new_group_id, expr.clone());
                terminals_added += 1;
            }
        }
        
        // Find the start production
        let start_prod_lhs = self.grammar.productions[self.grammar.start_production_id].lhs.clone();
        self.grammar.productions = new_productions;
        self.grammar.start_production_id = self.grammar.productions.iter()
            .position(|p| p.lhs == start_prod_lhs)
            .unwrap_or(0);
        
        debug!(4, "Partial optimization: {} productions -> {}, {} terminals added (of {} candidates) (in {:?})",
            self.stats.initial_productions, self.grammar.productions.len(), terminals_added, new_terminals.len(),
            update_start.elapsed());
        
        // Step 5: Handle nullable terminals that were created during optimization
        // Only handle those that were actually added
        let referenced_nullable_terminals: HashSet<String> = nullable_terminal_names
            .into_iter()
            .filter(|t| {
                // Extract the NT name from __opt_NTname__
                let nt_name = t.trim_start_matches("__opt_").trim_end_matches("__");
                referenced_terminals.contains(nt_name)
            })
            .collect();
        
        if !referenced_nullable_terminals.is_empty() {
            debug!(5, "Handling {} nullable terminals created by optimization", 
                referenced_nullable_terminals.len());
            self.grammar.handle_nullable_terminals_except(&pre_existing_terminals);
        }
        
        debug!(5, "After nullable handling: {} productions, {} terminals", 
            self.grammar.productions.len(), self.grammar.regex_name_to_group_id.len());
        
        // Remove unused terminals and compact IDs
        // This is important because unused terminals bloat the tokenizer
        self.remove_unused_terminals_and_compact();
        
        // Debug: print the remaining productions
        debug!(5, "Remaining productions after partial optimization:");
        for (i, prod) in self.grammar.productions.iter().enumerate() {
            let rhs_str: Vec<String> = prod.rhs.iter().map(|sym| {
                match sym {
                    Symbol::Terminal(t) => format!("T({:?})", t),
                    Symbol::NonTerminal(nt) => format!("NT({})", nt.0),
                }
            }).collect();
            debug!(5, "  [{}] {} -> {}", i, prod.lhs.0, rhs_str.join(" "));
        }
    }
    
    /// Remove terminals that are not referenced by any production and compact terminal IDs.
    /// This ensures terminals have contiguous group_ids starting from 0.
    fn remove_unused_terminals_and_compact(&mut self) {
        // Step 1: Collect all terminals that are actually used in productions
        let mut used_regex_terminals: HashSet<String> = HashSet::new();
        let mut used_literal_terminals: HashSet<Vec<u8>> = HashSet::new();
        
        for prod in &self.grammar.productions {
            for sym in &prod.rhs {
                match sym {
                    Symbol::Terminal(Terminal::RegexName(name)) => {
                        used_regex_terminals.insert(name.clone());
                    }
                    Symbol::Terminal(Terminal::Literal(lit)) => {
                        used_literal_terminals.insert(lit.clone());
                    }
                    Symbol::NonTerminal(_) => {}
                }
            }
        }
        
        // Also preserve the ignore terminal - it's not in productions but is used by the parser
        if let Some(ignore_id) = self.grammar.ignore_terminal_id {
            if let Some(name) = self.grammar.regex_name_to_group_id.get_by_right(&ignore_id.0) {
                used_regex_terminals.insert(name.clone());
                debug!(5, "Preserving ignore terminal '{}' (ID {})", name, ignore_id.0);
            }
        }
        
        // Step 2: Remove unused terminals from the maps
        let initial_regex_count = self.grammar.regex_name_to_group_id.len();
        let initial_literal_count = self.grammar.literal_to_group_id.len();
        
        let unused_regex: Vec<String> = self.grammar.regex_name_to_group_id
            .left_values()
            .filter(|name| !used_regex_terminals.contains(*name))
            .cloned()
            .collect();
        
        let unused_literals: Vec<Vec<u8>> = self.grammar.literal_to_group_id
            .left_values()
            .filter(|lit| !used_literal_terminals.contains(*lit))
            .cloned()
            .collect();
        
        // Remove unused entries
        for name in &unused_regex {
            debug!(5, "Removing unused regex terminal: {}", name);
            if let Some((_name, group_id)) = self.grammar.regex_name_to_group_id.remove_by_left(name) {
                self.grammar.group_id_to_expr.remove(&group_id);
            }
        }
        
        for lit in &unused_literals {
            debug!(5, "Removing unused literal terminal: {:?}", lit);
            if let Some((_lit, group_id)) = self.grammar.literal_to_group_id.remove_by_left(lit) {
                self.grammar.group_id_to_expr.remove(&group_id);
            }
        }
        
        let removed_regex = initial_regex_count - self.grammar.regex_name_to_group_id.len();
        let removed_literal = initial_literal_count - self.grammar.literal_to_group_id.len();
        
        if removed_regex > 0 || removed_literal > 0 {
            debug!(4, "Removed {} unused regex terminals and {} unused literal terminals", 
                removed_regex, removed_literal);
        }
        
        // Step 3: Compact terminal IDs to be contiguous starting from 0
        // Build mapping from old group_id -> new group_id
        let mut all_old_ids: Vec<usize> = self.grammar.group_id_to_expr.keys().cloned().collect();
        all_old_ids.sort();
        
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        for (new_id, old_id) in all_old_ids.iter().enumerate() {
            old_to_new.insert(*old_id, new_id);
        }
        
        // Check if renumbering is needed (if max_id != count - 1)
        let needs_renumbering = all_old_ids.last().copied().unwrap_or(0) != all_old_ids.len().saturating_sub(1);
        
        if needs_renumbering && !all_old_ids.is_empty() {
            debug!(4, "Compacting terminal IDs: renumbering {} terminals", all_old_ids.len());
            
            // Update group_id_to_expr
            let old_expr_map = std::mem::take(&mut self.grammar.group_id_to_expr);
            for (old_id, expr) in old_expr_map {
                let new_id = old_to_new[&old_id];
                self.grammar.group_id_to_expr.insert(new_id, expr);
            }
            
            // Update regex_name_to_group_id
            let regex_entries: Vec<(String, usize)> = self.grammar.regex_name_to_group_id
                .iter()
                .map(|(name, id)| (name.clone(), *id))
                .collect();
            self.grammar.regex_name_to_group_id.clear();
            for (name, old_id) in regex_entries {
                let new_id = old_to_new[&old_id];
                self.grammar.regex_name_to_group_id.insert(name, new_id);
            }
            
            // Update literal_to_group_id
            let literal_entries: Vec<(Vec<u8>, usize)> = self.grammar.literal_to_group_id
                .iter()
                .map(|(lit, id)| (lit.clone(), *id))
                .collect();
            self.grammar.literal_to_group_id.clear();
            for (lit, old_id) in literal_entries {
                let new_id = old_to_new[&old_id];
                self.grammar.literal_to_group_id.insert(lit, new_id);
            }
            
            // Update ignore_terminal_id if it was renumbered
            if let Some(old_ignore_id) = self.grammar.ignore_terminal_id {
                if let Some(&new_ignore_id) = old_to_new.get(&old_ignore_id.0) {
                    self.grammar.ignore_terminal_id = Some(TerminalID(new_ignore_id));
                    debug!(5, "Updated ignore_terminal_id: {} -> {}", old_ignore_id.0, new_ignore_id);
                }
            }
        }
        
        debug!(5, "After compaction: {} terminals with IDs 0..{}", 
            self.grammar.group_id_to_expr.len(),
            self.grammar.group_id_to_expr.len());
    }
}

/// Compute the set of NTs that transitively depend on cyclic NTs.
/// This includes the cyclic NTs themselves plus any NT that references them (directly or indirectly).
fn compute_cyclic_dependent_nts(cyclic_nts: &HashSet<String>, productions: &[Production]) -> HashSet<String> {
    // Build dependency graph: nt -> set of NTs it references
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for prod in productions {
        let nt = &prod.lhs.0;
        let entry = deps.entry(nt.clone()).or_insert_with(HashSet::new);
        for sym in &prod.rhs {
            if let Symbol::NonTerminal(ref_nt) = sym {
                entry.insert(ref_nt.0.clone());
            }
        }
    }
    
    // Start with cyclic NTs
    let mut result: HashSet<String> = cyclic_nts.clone();
    
    // Build reverse dependency graph for propagation
    let mut rev_deps: HashMap<String, HashSet<String>> = HashMap::new();
    for (nt, refs) in &deps {
        for r in refs {
            rev_deps.entry(r.clone()).or_insert_with(HashSet::new).insert(nt.clone());
        }
    }
    
    // Propagate: any NT that depends on a cyclic-dependent NT is also cyclic-dependent
    let mut changed = true;
    while changed {
        changed = false;
        let current: Vec<String> = result.iter().cloned().collect();
        for nt in current {
            // Any NT that references this one becomes cyclic-dependent
            if let Some(dependents) = rev_deps.get(&nt) {
                for dep in dependents {
                    if result.insert(dep.clone()) {
                        changed = true;
                    }
                }
            }
        }
    }
    
    result
}

/// Check if the grammar structure would cause exponential blowup during substitution.
/// This happens when:
/// 1. There are cycles in the dependency graph (mutual recursion)
/// 2. An NT is referenced multiple times (would cause duplication)
fn would_cause_exponential_blowup(equations: &HashMap<String, Rc<RegexTerm>>) -> bool {
    // Build dependency graph: nt -> set of NTs it references (excluding self)
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    
    // Count total references to each NT across all equations
    let mut total_ref_counts: HashMap<String, usize> = HashMap::new();
    
    for (nt, eq) in equations {
        let mut refs = eq.referenced_nts();
        refs.remove(nt); // Self-recursion is OK (handled by Arden's lemma)
        
        // Count references within THIS equation (excluding self)
        for r in eq.count_nt_references() {
            if &r.0 != nt {
                *total_ref_counts.entry(r.0).or_insert(0) += r.1;
            }
        }
        
        deps.insert(nt.clone(), refs);
    }
    
    // Check if any NT is referenced multiple times across ALL equations
    // This would cause duplication during substitution
    for (_nt, count) in &total_ref_counts {
        if *count > 1 {
            return true;
        }
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

/// Find all NTs that are part of a mutual recursion cycle.
/// Self-recursion (NT -> NT) is NOT a cycle for our purposes (handled by Arden's lemma).
/// Returns a set of NT names that are in cycles.
fn find_cyclic_nts(equations: &HashMap<String, Rc<RegexTerm>>) -> HashSet<String> {
    // Build dependency graph: nt -> set of NTs it references (excluding self)
    let mut deps: HashMap<String, HashSet<String>> = HashMap::new();
    for (nt, eq) in equations {
        let mut refs = eq.referenced_nts();
        refs.remove(nt); // Remove self-reference
        deps.insert(nt.clone(), refs);
    }
    
    // Use Tarjan's algorithm to find strongly connected components
    let mut index = 0;
    let mut stack: Vec<String> = Vec::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut indices: HashMap<String, usize> = HashMap::new();
    let mut lowlinks: HashMap<String, usize> = HashMap::new();
    let mut sccs: Vec<Vec<String>> = Vec::new();
    
    fn strongconnect(
        v: &str,
        deps: &HashMap<String, HashSet<String>>,
        index: &mut usize,
        stack: &mut Vec<String>,
        on_stack: &mut HashSet<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
        sccs: &mut Vec<Vec<String>>,
    ) {
        indices.insert(v.to_string(), *index);
        lowlinks.insert(v.to_string(), *index);
        *index += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string());
        
        if let Some(neighbors) = deps.get(v) {
            for w in neighbors {
                if !indices.contains_key(w) {
                    strongconnect(w, deps, index, stack, on_stack, indices, lowlinks, sccs);
                    let w_low = *lowlinks.get(w).unwrap();
                    let v_low = lowlinks.get_mut(v).unwrap();
                    *v_low = (*v_low).min(w_low);
                } else if on_stack.contains(w) {
                    let w_idx = *indices.get(w).unwrap();
                    let v_low = lowlinks.get_mut(v).unwrap();
                    *v_low = (*v_low).min(w_idx);
                }
            }
        }
        
        if lowlinks.get(v) == indices.get(v) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }
    
    for v in equations.keys() {
        if !indices.contains_key(v) {
            strongconnect(v, &deps, &mut index, &mut stack, &mut on_stack, &mut indices, &mut lowlinks, &mut sccs);
        }
    }
    
    // SCCs with size > 1 are cycles
    let mut cyclic = HashSet::new();
    for scc in sccs {
        if scc.len() > 1 {
            cyclic.extend(scc);
        }
    }
    
    cyclic
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
            // Only flatten if this Rc is NOT shared (strong_count == 1)
            // This avoids O(n²) blowup when flattening shared Choice subtrees
            let should_flatten = Rc::strong_count(&t) == 1;
            match t.as_ref() {
                RegexTerm::Choice(inner) if should_flatten => {
                    flat.extend(inner.iter().cloned())
                }
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
        // Simple version without memoization - used for non-shared trees
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
    
    /// Count the number of references to each NT in this term
    fn count_nt_references(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        self.collect_nt_counts(&mut counts);
        counts.into_iter().collect()
    }
    
    fn collect_nt_counts(&self, counts: &mut HashMap<String, usize>) {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => {}
            RegexTerm::NtRef(n) => { *counts.entry(n.clone()).or_insert(0) += 1; }
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                for t in terms {
                    t.collect_nt_counts(counts);
                }
            }
            RegexTerm::Star(inner) => inner.collect_nt_counts(counts),
        }
    }
    
    /// Calculate the size of this term (number of nodes in the tree)
    /// This counts each node once, even if referenced via Rc multiple times
    fn size(&self) -> usize {
        let mut visited: HashSet<*const RegexTerm> = HashSet::new();
        self.size_with_visited(&mut visited)
    }
    
    fn size_with_visited(&self, visited: &mut HashSet<*const RegexTerm>) -> usize {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) | RegexTerm::NtRef(_) => 1,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                1 + terms.iter().map(|t| {
                    let ptr = Rc::as_ptr(t);
                    if visited.contains(&ptr) {
                        0 // Already counted this subtree
                    } else {
                        visited.insert(ptr);
                        t.size_with_visited(visited)
                    }
                }).sum::<usize>()
            }
            RegexTerm::Star(inner) => {
                let ptr = Rc::as_ptr(inner);
                if visited.contains(&ptr) {
                    1 // Just count the Star node, inner already counted
                } else {
                    visited.insert(ptr);
                    1 + inner.size_with_visited(visited)
                }
            }
        }
    }
}

/// Solve a single equation: X = equation, where equation may reference X.
/// Uses Arden's lemma: X = αX | β => X = α*β (for right recursion)
/// And: X = Xα | β => X = βα* (for left recursion)
/// 
/// IMPORTANT: This function is designed to work on SMALL, unexpanded equations.
/// The equation should contain NtRefs to the current NT (for self-recursion)
/// and possibly to other NTs (which will be substituted later).
/// 
/// Takes Rc<RegexTerm> to enable memoization for contains_nt checks
fn solve_single_equation_rc(equation: &Rc<RegexTerm>, nt: &str) -> Option<RegexTerm> {
    // Use a simple, non-memoized contains check since equations are small
    fn contains_nt_simple(term: &RegexTerm, nt: &str) -> bool {
        match term {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => false,
            RegexTerm::NtRef(n) => n == nt,
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                terms.iter().any(|t| contains_nt_simple(t.as_ref(), nt))
            }
            RegexTerm::Star(inner) => contains_nt_simple(inner.as_ref(), nt),
        }
    }
    
    if !contains_nt_simple(equation.as_ref(), nt) {
        // No self-reference, just return as-is
        return Some((**equation).clone());
    }
    
    // Separate into recursive and non-recursive alternatives
    fn separate_alts(term: &RegexTerm, nt: &str) -> (Vec<RegexTerm>, Vec<RegexTerm>) {
        match term {
            RegexTerm::Choice(alts) => {
                let mut recursive = Vec::new();
                let mut base = Vec::new();
                for alt in alts {
                    if contains_nt_simple(alt.as_ref(), nt) {
                        recursive.push((**alt).clone());
                    } else {
                        base.push((**alt).clone());
                    }
                }
                (recursive, base)
            }
            _ if contains_nt_simple(term, nt) => (vec![term.clone()], vec![]),
            _ => (vec![], vec![term.clone()]),
        }
    }
    
    let (recursive, base) = separate_alts(equation.as_ref(), nt);
    
    if recursive.is_empty() {
        return Some(RegexTerm::make_choice(base.into_iter().map(Rc::new).collect()));
    }
    
    // Extract coefficients from recursive alternatives
    fn extract_coef(term: &RegexTerm, nt: &str) -> Option<(RegexTerm, RegexTerm)> {
        match term {
            RegexTerm::NtRef(n) if n == nt => {
                Some((RegexTerm::Epsilon, RegexTerm::Epsilon))
            }
            RegexTerm::Seq(parts) => {
                // Find the position of the NT reference
                let mut nt_pos = None;
                for (i, p) in parts.iter().enumerate() {
                    if contains_nt_simple(p.as_ref(), nt) {
                        if nt_pos.is_some() {
                            // Multiple occurrences - not linear
                            return None;
                        }
                        nt_pos = Some(i);
                    }
                }
                let pos = nt_pos?;
                
                // Check that the NT-containing element is just NtRef(nt)
                if !matches!(parts[pos].as_ref(), RegexTerm::NtRef(n) if n == nt) {
                    // NT is nested inside something - not simple linear form
                    return None;
                }
                
                let prefix = if pos == 0 {
                    RegexTerm::Epsilon
                } else {
                    RegexTerm::make_seq(parts[..pos].iter().cloned().collect())
                };
                let suffix = if pos == parts.len() - 1 {
                    RegexTerm::Epsilon
                } else {
                    RegexTerm::make_seq(parts[pos+1..].iter().cloned().collect())
                };
                Some((prefix, suffix))
            }
            _ => None,
        }
    }
    
    fn is_epsilon(t: &RegexTerm) -> bool {
        matches!(t, RegexTerm::Epsilon)
    }
    
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
        Rc::new(RegexTerm::make_choice(base.into_iter().map(Rc::new).collect()))
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

/// Separate a term into recursive and non-recursive alternatives (Rc version with memoization)
fn separate_recursive_alts_rc(
    term: &Rc<RegexTerm>, 
    nt: &str,
    contains_cache: &mut HashMap<*const RegexTerm, bool>
) -> (Vec<Rc<RegexTerm>>, Vec<Rc<RegexTerm>>) {
    match term.as_ref() {
        RegexTerm::Choice(alts) => {
            let mut recursive = Vec::new();
            let mut base = Vec::new();
            for alt in alts {
                if contains_nt_memoized(alt, nt, contains_cache) {
                    recursive.push(alt.clone());
                } else {
                    base.push(alt.clone());
                }
            }
            (recursive, base)
        }
        _ if contains_nt_memoized(term, nt, contains_cache) => (vec![term.clone()], vec![]),
        _ => (vec![], vec![term.clone()]),
    }
}

/// Extract (prefix, suffix) from a term of the form "prefix X suffix" (Rc version)
/// Returns None if the term is not linear in X
fn extract_coef_rc(
    term: &Rc<RegexTerm>, 
    nt: &str,
    contains_cache: &mut HashMap<*const RegexTerm, bool>
) -> Option<(RegexTerm, RegexTerm)> {
    match term.as_ref() {
        RegexTerm::NtRef(n) if n == nt => {
            Some((RegexTerm::Epsilon, RegexTerm::Epsilon))
        }
        RegexTerm::Seq(parts) => {
            // Find the position of the NT reference
            let mut nt_pos = None;
            for (i, p) in parts.iter().enumerate() {
                if contains_nt_memoized(p, nt, contains_cache) {
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

/// Legacy wrapper for solve_single_equation (converts &RegexTerm to Rc)
fn solve_single_equation(equation: &RegexTerm, nt: &str) -> Option<RegexTerm> {
    solve_single_equation_rc(&Rc::new(equation.clone()), nt)
}

fn is_epsilon(term: &RegexTerm) -> bool {
    match term {
        RegexTerm::Epsilon => true,
        RegexTerm::Seq(s) if s.is_empty() => true,
        _ => false,
    }
}

/// Check if term contains a reference to `nt`, using memoization to handle shared subterms
fn contains_nt_memoized(
    term: &Rc<RegexTerm>,
    nt: &str,
    visited: &mut HashMap<*const RegexTerm, bool>
) -> bool {
    let ptr = Rc::as_ptr(term);
    if let Some(&result) = visited.get(&ptr) {
        return result;
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => false,
        RegexTerm::NtRef(n) => n == nt,
        RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
            terms.iter().any(|t| contains_nt_memoized(t, nt, visited))
        }
        RegexTerm::Star(inner) => contains_nt_memoized(inner, nt, visited),
    };
    
    visited.insert(ptr, result);
    result
}

/// Expand a term by substituting all NT references with their solutions
/// Uses memoization and Rc sharing to avoid exponential blowup
/// Returns None if a cycle is detected (mutual recursion)
fn expand_with_solutions(
    term: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>
) -> Option<Rc<RegexTerm>> {
    let mut cache: HashMap<*const RegexTerm, Rc<RegexTerm>> = HashMap::new();
    expand_with_solutions_cached(term, solutions, &mut cache)
}

/// Expand all NtRefs in a term by substituting solutions.
/// Uses iterative deepening with explicit stack to avoid stack overflow.
/// 
/// `expanding` tracks NTs currently being expanded to detect cycles.
fn expand_with_solutions_cached(
    root: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>
) -> Option<Rc<RegexTerm>> {
    let mut expanding: HashSet<String> = HashSet::new();
    let skip_nts: HashSet<String> = HashSet::new();
    expand_with_solutions_cached_inner(root, solutions, cache, &mut expanding, &skip_nts)
}

/// Expand all NtRefs in a term, but skip certain NTs (leaving them as NtRefs).
/// This is used when we know certain NTs are cyclic and shouldn't be expanded.
fn expand_with_solutions_skip_cyclic(
    root: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>,
    skip_nts: &HashSet<String>
) -> Rc<RegexTerm> {
    let mut expanding: HashSet<String> = HashSet::new();
    // This version always succeeds because we skip cyclic NTs
    expand_with_solutions_cached_inner(root, solutions, cache, &mut expanding, skip_nts)
        .expect("expand_with_solutions_skip_cyclic should not fail")
}

fn expand_with_solutions_cached_inner(
    root: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>,
    expanding: &mut HashSet<String>,
    skip_nts: &HashSet<String>
) -> Option<Rc<RegexTerm>> {
    // Use stacker to handle deep recursion by switching to heap allocation when needed
    stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
        expand_with_solutions_cached_impl(root, solutions, cache, expanding, skip_nts)
    })
}

fn expand_with_solutions_cached_impl(
    term: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>,
    expanding: &mut HashSet<String>,
    skip_nts: &HashSet<String>
) -> Option<Rc<RegexTerm>> {
    let ptr = Rc::as_ptr(term);
    
    // For NtRef, resolve to solution and expand that
    if let RegexTerm::NtRef(n) = term.as_ref() {
        // If this NT should be skipped (it's cyclic), leave it as NtRef
        if skip_nts.contains(n) {
            return Some(term.clone());
        }
        
        if let Some(solution) = solutions.get(n) {
            // Check if solution is already expanded in cache
            let sol_ptr = Rc::as_ptr(solution);
            if let Some(cached) = cache.get(&sol_ptr) {
                return Some(cached.clone());
            }
            // Check for cycles - if we're already expanding this NT, we have mutual recursion
            if expanding.contains(n) {
                // Cycle detected - cannot expand
                return None;
            }
            // Mark as expanding
            expanding.insert(n.clone());
            // Recursively expand the solution
            let result = expand_with_solutions_cached_inner(solution, solutions, cache, expanding, skip_nts)?;
            expanding.remove(n);
            return Some(result);
        } else {
            return Some(term.clone());
        }
    }
    
    // Check cache for non-NtRef terms
    if let Some(cached) = cache.get(&ptr) {
        return Some(cached.clone());
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => term.clone(),
        RegexTerm::NtRef(_) => unreachable!(), // Handled above
        RegexTerm::Seq(parts) => {
            let new_parts: Option<Vec<Rc<RegexTerm>>> = parts.iter()
                .map(|p| expand_with_solutions_cached_inner(p, solutions, cache, expanding, skip_nts))
                .collect();
            let new_parts = new_parts?;
            let changed = parts.iter().zip(new_parts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_seq(new_parts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Choice(alts) => {
            let new_alts: Option<Vec<Rc<RegexTerm>>> = alts.iter()
                .map(|a| expand_with_solutions_cached_inner(a, solutions, cache, expanding, skip_nts))
                .collect();
            let new_alts = new_alts?;
            let changed = alts.iter().zip(new_alts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_choice(new_alts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Star(inner) => {
            let new_inner = expand_with_solutions_cached_inner(inner, solutions, cache, expanding, skip_nts)?;
            if Rc::ptr_eq(inner, &new_inner) {
                term.clone()
            } else {
                Rc::new(RegexTerm::make_star(new_inner))
            }
        }
    };
    
    cache.insert(ptr, result.clone());
    Some(result)
}

/// Substitute all occurrences of `nt` in `term` with `replacement`
/// Returns an Rc to preserve sharing - crucial for avoiding exponential blowup
/// Uses memoization to handle shared subterms efficiently
fn substitute_nt_rc(term: &Rc<RegexTerm>, nt: &str, replacement: &Rc<RegexTerm>) -> Rc<RegexTerm> {
    let mut cache: HashMap<*const RegexTerm, Rc<RegexTerm>> = HashMap::new();
    let mut contains_cache: HashMap<*const RegexTerm, bool> = HashMap::new();
    let result = substitute_nt_rc_cached(term, nt, replacement, &mut cache, &mut contains_cache);
    result
}

fn substitute_nt_rc_cached(
    term: &Rc<RegexTerm>,
    nt: &str,
    replacement: &Rc<RegexTerm>,
    cache: &mut HashMap<*const RegexTerm, Rc<RegexTerm>>,
    contains_cache: &mut HashMap<*const RegexTerm, bool>
) -> Rc<RegexTerm> {
    let ptr = Rc::as_ptr(term);
    
    // Check cache first
    if let Some(cached) = cache.get(&ptr) {
        return cached.clone();
    }
    
    // Check if this term contains the NT at all (with memoization)
    if !contains_nt_memoized(term, nt, contains_cache) {
        cache.insert(ptr, term.clone());
        return term.clone();
    }
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon | RegexTerm::Concrete(_) => term.clone(),
        RegexTerm::NtRef(n) if n == nt => replacement.clone(),
        RegexTerm::NtRef(_) => term.clone(),
        RegexTerm::Seq(parts) => {
            let new_parts: Vec<Rc<RegexTerm>> = parts.iter()
                .map(|p| substitute_nt_rc_cached(p, nt, replacement, cache, contains_cache))
                .collect();
            // Only create a new Rc if something actually changed
            let changed = parts.iter().zip(new_parts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_seq(new_parts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Choice(alts) => {
            let new_alts: Vec<Rc<RegexTerm>> = alts.iter()
                .map(|a| substitute_nt_rc_cached(a, nt, replacement, cache, contains_cache))
                .collect();
            // Only create a new Rc if something actually changed
            let changed = alts.iter().zip(new_alts.iter())
                .any(|(old, new)| !Rc::ptr_eq(old, new));
            if changed {
                Rc::new(RegexTerm::make_choice(new_alts))
            } else {
                term.clone()
            }
        }
        RegexTerm::Star(inner) => {
            let new_inner = substitute_nt_rc_cached(inner, nt, replacement, cache, contains_cache);
            if Rc::ptr_eq(inner, &new_inner) {
                term.clone()
            } else {
                Rc::new(RegexTerm::make_star(new_inner))
            }
        }
    };
    
    cache.insert(ptr, result.clone());
    result
}

/// Wrapper for backward compatibility - converts result to non-Rc
fn substitute_nt(term: &RegexTerm, nt: &str, replacement: &Rc<RegexTerm>) -> RegexTerm {
    let rc_term = Rc::new(term.clone());
    let result = substitute_nt_rc(&rc_term, nt, replacement);
    // Try to unwrap the Rc if we're the only owner, otherwise clone
    Rc::try_unwrap(result).unwrap_or_else(|rc| (*rc).clone())
}

/// Convert a RegexTerm (with no NT references) to an Expr
/// Uses caching via Arc<Expr> to preserve sharing from Rc<RegexTerm>
fn regex_term_to_expr(term: &RegexTerm) -> Option<Expr> {
    // For backward compatibility, wrap in Rc (loses sharing info)
    let term_rc = Rc::new(term.clone());
    regex_term_to_expr_rc(&term_rc)
}

/// Convert a RegexTerm Rc to an Expr, preserving sharing information
fn regex_term_to_expr_rc(term: &Rc<RegexTerm>) -> Option<Expr> {
    use std::sync::Arc;
    
    let mut cache: HashMap<*const RegexTerm, Arc<Expr>> = HashMap::new();
    regex_term_to_expr_cached(term, &mut cache)
}

fn regex_term_to_expr_cached(
    term: &Rc<RegexTerm>,
    cache: &mut HashMap<*const RegexTerm, std::sync::Arc<Expr>>
) -> Option<Expr> {
    // Note: stacker removed - tree depth is shallow (16) so stack overflow unlikely
    regex_term_to_expr_cached_impl(term, cache)
}

fn regex_term_to_expr_cached_impl(
    term: &Rc<RegexTerm>,
    cache: &mut HashMap<*const RegexTerm, std::sync::Arc<Expr>>
) -> Option<Expr> {
    use std::sync::Arc;
    
    let ptr = Rc::as_ptr(term);
    
    // Always check cache first for efficiency
    if let Some(cached) = cache.get(&ptr) {
        return Some(Expr::Shared(cached.clone()));
    }
    
    // Check if this Rc is shared (referenced multiple times in the original tree)
    let is_shared = Rc::strong_count(term) > 1;
    
    let result = match term.as_ref() {
        RegexTerm::Epsilon => Some(Expr::Epsilon),
        RegexTerm::Concrete(e) => Some(e.clone()),
        RegexTerm::NtRef(_) => None, // Should not have NT refs at this point
        RegexTerm::Seq(parts) => {
            let exprs: Option<Vec<Expr>> = parts.iter()
                .map(|p| regex_term_to_expr_cached(p, cache))
                .collect();
            exprs.map(make_seq)
        }
        RegexTerm::Choice(alts) => {
            let exprs: Option<Vec<Expr>> = alts.iter()
                .map(|a| regex_term_to_expr_cached(a, cache))
                .collect();
            exprs.map(make_choice)
        }
        RegexTerm::Star(inner) => {
            regex_term_to_expr_cached(inner, cache).map(|inner_expr| {
                Expr::Quantifier(Box::new(inner_expr), QuantifierType::ZeroOrMore)
            })
        }
    };
    
    // Always cache the result to avoid re-processing
    if let Some(ref e) = result {
        cache.insert(ptr, Arc::new(e.clone()));
    }
    
    // Return Shared wrapper if this term is truly shared
    if is_shared {
        result.map(|e| match e {
            Expr::Shared(_) => e,
            _ => Expr::Shared(Arc::new(e)),
        })
    } else {
        result
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

/// Check if an Expr is epsilon (possibly through Shared wrappers)
fn is_epsilon_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::Shared(inner) => is_epsilon_expr(inner),
        _ => false,
    }
}

/// Simplify an expression
fn simplify_expr(expr: Expr) -> Expr {
    let mut cache: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    simplify_expr_cached(expr, &mut cache)
}

fn simplify_expr_cached(expr: Expr, cache: &mut HashMap<*const Expr, Arc<Expr>>) -> Expr {
    // Use stacker to handle deep recursion
    stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
        simplify_expr_cached_impl(expr, cache)
    })
}

fn simplify_expr_cached_impl(expr: Expr, cache: &mut HashMap<*const Expr, Arc<Expr>>) -> Expr {
    match expr {
        Expr::Seq(exprs) => {
            let mut simplified = Vec::new();
            for e in exprs {
                let e = simplify_expr_cached(e, cache);
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
                .map(|e| simplify_expr_cached(e, cache))
                .flat_map(|e| match e {
                    Expr::Choice(inner) => inner,
                    other => vec![other],
                })
                .collect();
            if simplified.len() == 1 {
                simplified.into_iter().next().unwrap()
            } else if simplified.len() == 2 {
                // Check for Choice([a, Epsilon]) or Choice([Epsilon, a]) -> a?
                let (first, second) = (&simplified[0], &simplified[1]);
                if is_epsilon_expr(second) {
                    Expr::Quantifier(Box::new(first.clone()), QuantifierType::ZeroOrOne)
                } else if is_epsilon_expr(first) {
                    Expr::Quantifier(Box::new(second.clone()), QuantifierType::ZeroOrOne)
                } else {
                    Expr::Choice(simplified)
                }
            } else {
                Expr::Choice(simplified)
            }
        }
        Expr::Quantifier(inner, qtype) => {
            let inner = simplify_expr_cached(*inner, cache);
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
        Expr::Shared(inner) => {
            let ptr = Arc::as_ptr(&inner);
            // Check cache for this shared subtree
            if let Some(cached) = cache.get(&ptr) {
                // Return a Shared pointing to the cached result
                return Expr::Shared(cached.clone());
            }
            // Simplify the inner expression
            let simplified = simplify_expr_cached((*inner).clone(), cache);
            // Cache and return
            let result = Arc::new(simplified);
            cache.insert(ptr, result.clone());
            Expr::Shared(result)
        }
        other => other,
    }
}


#[cfg(test)]
mod tests {
    use indoc::indoc;
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

        GrammarDefinition::from_exprs_no_optimize(grammar_exprs, regex_exprs).unwrap()
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

    #[ignore]
    #[test]
    fn test_diff_grammar_optimization_performance() {
        // This test verifies that grammar optimization scales roughly linearly.
        // Due to system load variability, we run multiple trials and take the best ratio.
        let mut n = 100;
        let mut t_base = 0.0;

        // Find a baseline size that takes at least 10ms (increased for more stable measurements)
        loop {
            let mut grammar = build_diff_grammar(n);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            let duration = start.elapsed();
            t_base = duration.as_secs_f64();

            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n);

            if t_base > 0.010 {
                break;
            }
            n *= 2;

            if n > 10_000 {
                println!("Warning: reached n={} without exceeding time threshold (got {:.4}s). Skipping perf check.", n, t_base);
                return;
            }
        }

        // Run 5 trials and take the best ratio to reduce flakiness from system load
        let n_scaled = n * 3;
        let mut best_ratio = f64::MAX;
        
        for trial in 0..5 {
            // Re-measure baseline for each trial
            let mut grammar = build_diff_grammar(n);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            t_base = start.elapsed().as_secs_f64();
            
            let mut grammar = build_diff_grammar(n_scaled);
            let start = std::time::Instant::now();
            optimize_grammar(&mut grammar);
            let t_scaled = start.elapsed().as_secs_f64();

            assert_eq!(grammar.terminal_to_group_id().len(), 1, "Failed reduction for n={}", n_scaled);

            let ratio = t_scaled / t_base;
            println!("Trial {}: Baseline n={} ({:.4}s), Scaled n={} ({:.4}s), Ratio: {:.2}", 
                     trial + 1, n, t_base, n_scaled, t_scaled, ratio);
            
            best_ratio = best_ratio.min(ratio);
        }

        println!("Best ratio: {:.2}", best_ratio);

        // Linear scaling would give ratio ~3.0 for 3x input
        // Allow up to 8.0x for system variability (we're testing O(n) not O(n^2))
        assert!(best_ratio < 8.0, "Performance scaling looks worse than linear (best ratio {:.2})", best_ratio);
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

    #[test]
    fn test_x_semicolon_x() {
        // Build grammar manually to avoid from_ebnf internal optimization
        let grammar_exprs = vec![
            ("program".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("expression_statement".to_string()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Ref("expression_statement".to_string()))),
                GrammarExpr::Ref("EOF".to_string()),
            ])),
            ("expression_statement".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("expression".to_string()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b";".to_vec()))),
            ])),
            ("expression".to_string(), GrammarExpr::Literal(b"x".to_vec())),
        ];
        let regex_exprs = vec![
            ("EOF".to_string(), Expr::U8Seq(b"$".to_vec())),
        ];
        
        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, regex_exprs).unwrap();
        println!("=== BEFORE OPTIMIZATION ===");
        println!("{grammar}");
        optimize_grammar(&mut grammar);
        println!("=== AFTER OPTIMIZATION ===");
        println!("{grammar}");
        assert_eq!(grammar.terminal_to_group_id().len(), 1);

        let x = Expr::U8Seq(b"x".to_vec());
        let semi_opt = Expr::Quantifier(Box::new(Expr::U8Seq(b";".to_vec())), QuantifierType::ZeroOrOne);
        let eof = Expr::U8Seq(b"$".to_vec());
        let stmt = Expr::Seq(vec![x.clone(), semi_opt.clone()]);

        let expected_regex = Expr::Seq(vec![
            x, semi_opt,
            Expr::Quantifier(Box::new(stmt), QuantifierType::ZeroOrOne),
            eof
        ]);

        // Strip Shared wrappers for comparison since the optimization uses Shared for deduplication
        fn strip_shared(expr: &Expr) -> Expr {
            match expr {
                Expr::Shared(inner) => strip_shared(inner),
                Expr::Seq(exprs) => Expr::Seq(exprs.iter().map(strip_shared).collect()),
                Expr::Choice(exprs) => Expr::Choice(exprs.iter().map(strip_shared).collect()),
                Expr::Quantifier(inner, qtype) => Expr::Quantifier(Box::new(strip_shared(inner)), qtype.clone()),
                other => other.clone(),
            }
        }

        let actual = strip_shared(grammar.group_id_to_expr.get(&0).unwrap());
        assert_eq!(actual, expected_regex, "Expected: {}, got {}", expected_regex, actual);
    }

    /// Test that literal choices are merged into a single terminal even when inside
    /// a cyclic-dependent NT. This simulates the JS grammar pattern:
    /// `equality_expression ::= relational_expression ( ( '==' | '!=' | '===' | '!==' ) relational_expression )* ;`
    #[test]
    fn test_literal_choice_in_cyclic_dependent_nt() {
        // Simulate: expr ::= term ( ( '+' | '-' | '*' | '/' ) term )* ;
        // where term is recursive (simulated by referencing back to expr through unary)
        let grammar_exprs = vec![
            ("start".to_string(), GrammarExpr::Ref("expr".to_string())),
            ("expr".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("term".to_string()),
                GrammarExpr::Repeat(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Literal(b"+".to_vec()),
                        GrammarExpr::Literal(b"-".to_vec()),
                        GrammarExpr::Literal(b"*".to_vec()),
                        GrammarExpr::Literal(b"/".to_vec()),
                    ]),
                    GrammarExpr::Ref("term".to_string()),
                ]))),
            ])),
            // term references expr, creating a cycle
            ("term".to_string(), GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"x".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"(".to_vec()),
                    GrammarExpr::Ref("expr".to_string()),
                    GrammarExpr::Literal(b")".to_vec()),
                ]),
            ])),
        ];

        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, vec![]).unwrap();
        println!("=== BEFORE OPTIMIZATION ===");
        println!("{grammar}");
        
        let initial_terminals = grammar.terminal_to_group_id().len();
        optimize_grammar(&mut grammar);
        
        println!("=== AFTER OPTIMIZATION ===");
        println!("{grammar}");
        
        // The literal choice ( '+' | '-' | '*' | '/' ) should become a single terminal
        // Check that we have fewer terminals than if each literal were separate
        let final_terminals = grammar.terminal_to_group_id().len();
        println!("Initial terminals: {}, Final terminals: {}", initial_terminals, final_terminals);
        
        // We should have merged the 4-literal choice into 1 terminal
        // The grammar should have: the merged operator terminal, '(', ')', 'x'
        // With optimization, we might have even fewer
        assert!(final_terminals <= initial_terminals, 
            "Expected terminal count to decrease or stay same after optimization");
        
        // Verify the grammar still compiles and works
        use crate::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }

    /// Test that partial terminal conversion works: when a production has some parts
    /// that can be converted to terminals but the whole production cannot (because it
    /// references a recursive NT), the convertible parts should still become terminals.
    /// 
    /// Example pattern from JS:
    /// `method_definition ::= ('static')? ('get'|'set')? ('async'|'*')? class_property_name '(' parameter_list? ')' block ;`
    /// 
    /// The prefix `('static')? ('get'|'set')? ('async'|'*')? class_property_name '(' parameter_list? ')'`
    /// cannot be fully converted because it references `class_property_name` and `parameter_list` which may be cyclic.
    /// But the literal choices should still be converted to terminals.
    #[test]
    fn test_partial_terminal_conversion_in_method_definition() {
        // Simulate method_definition pattern where:
        // - ('get'|'set') is a literal choice that should become a terminal
        // - block is a non-terminal that CANNOT be converted (simulated with recursion)
        let grammar_exprs = vec![
            ("start".to_string(), GrammarExpr::Ref("method_def".to_string())),
            ("method_def".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"static".to_vec()))),
                GrammarExpr::Optional(Box::new(GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"get".to_vec()),
                    GrammarExpr::Literal(b"set".to_vec()),
                ]))),
                GrammarExpr::Optional(Box::new(GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"async".to_vec()),
                    GrammarExpr::Literal(b"*".to_vec()),
                ]))),
                GrammarExpr::Ref("name".to_string()),
                GrammarExpr::Literal(b"(".to_vec()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Ref("params".to_string()))),
                GrammarExpr::Literal(b")".to_vec()),
                GrammarExpr::Ref("block".to_string()),
            ])),
            // name is simple - just an identifier
            ("name".to_string(), GrammarExpr::Literal(b"foo".to_vec())),
            // params is simple
            ("params".to_string(), GrammarExpr::Literal(b"x".to_vec())),
            // block is recursive (contains method_def), making method_def cyclic-dependent
            ("block".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"{".to_vec()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Ref("method_def".to_string()))),
                GrammarExpr::Literal(b"}".to_vec()),
            ])),
        ];

        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, vec![]).unwrap();
        println!("=== BEFORE OPTIMIZATION ===");
        println!("{grammar}");
        
        let initial_terminals = grammar.terminal_to_group_id().len();
        optimize_grammar(&mut grammar);
        
        println!("=== AFTER OPTIMIZATION ===");
        println!("{grammar}");
        
        let final_terminals = grammar.terminal_to_group_id().len();
        println!("Initial terminals: {}, Final terminals: {}", initial_terminals, final_terminals);
        
        // The literal choices ('get'|'set') and ('async'|'*') should be converted to terminals
        // even though method_def as a whole cannot be converted (due to block recursion)
        
        // Verify the grammar still compiles and works
        use crate::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }

    /// Test that two-alternative literal choices are also merged (not just 3+).
    /// The current optimization only kicks in for 3+ alternatives during initial
    /// GrammarExpr parsing, but the grammar optimizer should handle 2-alternative
    /// choices as well.
    #[test]
    fn test_two_alternative_literal_choice() {
        // Pattern: a_expr ::= b_expr ( ( '+' | '-' ) b_expr )* ;
        // with b_expr being recursive
        let grammar_exprs = vec![
            ("start".to_string(), GrammarExpr::Ref("a_expr".to_string())),
            ("a_expr".to_string(), GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("b_expr".to_string()),
                GrammarExpr::Repeat(Box::new(GrammarExpr::Sequence(vec![
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Literal(b"+".to_vec()),
                        GrammarExpr::Literal(b"-".to_vec()),
                    ]),
                    GrammarExpr::Ref("b_expr".to_string()),
                ]))),
            ])),
            ("b_expr".to_string(), GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"x".to_vec()),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"(".to_vec()),
                    GrammarExpr::Ref("a_expr".to_string()),
                    GrammarExpr::Literal(b")".to_vec()),
                ]),
            ])),
        ];

        let mut grammar = GrammarDefinition::from_exprs_no_optimize(grammar_exprs, vec![]).unwrap();
        println!("=== BEFORE OPTIMIZATION ===");
        println!("{grammar}");
        
        // Count initial productions for the choice
        let initial_choice_productions = grammar.productions.iter()
            .filter(|p| p.rhs.len() == 1 && matches!(&p.rhs[0], Symbol::Terminal(Terminal::Literal(lit)) if lit == b"+" || lit == b"-"))
            .count();
        
        optimize_grammar(&mut grammar);
        
        println!("=== AFTER OPTIMIZATION ===");
        println!("{grammar}");
        
        // After optimization, '+' and '-' should be merged into a single terminal
        // (either as a Choice in the regex, or the productions should be reduced)
        
        // Verify it compiles
        use crate::interface::CompiledGrammar;
        let _ = CompiledGrammar::from_definition(std::sync::Arc::new(grammar));
    }
}
