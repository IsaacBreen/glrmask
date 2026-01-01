use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::rc::Rc;
use bimap::BiBTreeMap;
use rustc_hash::FxHashMap;
use crate::finite_automata::{Expr, QuantifierType};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::{GrammarDefinition, GrammarExpr, ExprNullability, get_expr_nullability};
use crate::types::TerminalID;
use crate::debug;

pub fn optimize_grammar(grammar: &mut GrammarDefinition) {
    if std::env::var("DISABLE_GRAMMAR_OPTIMIZATION").is_ok() {
        let mut optimizer = GrammarOptimizer::new(grammar);
        optimizer.remove_unused_terminals_and_compact();
        return;
    }
    
    // NOTE: We do NOT call handle_nullable_terminals() here anymore.
    // Nullable terminals are already handled in from_exprs_impl() during
    // GrammarDefinition construction. Calling it again here would cause
    // double-processing, creating additional wrapper non-terminals and
    // epsilon productions that lead to severe grammar bloat (e.g., 459
    // productions instead of 228, causing 3x slower builds).
    //
    // The optimizer (partial_optimize) will call handle_nullable_terminals_except()
    // at line 610 to handle any NEW nullable terminals created during optimization.
    
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

    fn optimize(&mut self) {
        self.stats.initial_productions = self.grammar.productions.len();
        self.stats.initial_terminals = self.count_terminals();

        // Extract complex alternatives before optimization (reduces terminal count)
        if std::env::var("EXTRACT_COMPLEX_ALTERNATIVES").is_ok() {
            crate::interface::extract_alternatives::extract_complex_alternatives(self.grammar);
        }

        // Try to optimize the grammar by converting regular sub-grammars to regexes
        self.optimize_regular_subgrammars();

        // Optimize the exprs
        for expr in self.grammar.group_id_to_expr.values_mut() {
            *expr = expr.clone().optimize();
        }

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
        // UPDATE: With terminal DWA simplification, grammar optimization can help, but
        // the tokenizer explosion still dominates. Keep threshold at 500 for now.
        const MAX_PRODUCTIONS_FOR_OPTIMIZATION: usize = 50000;
        if self.grammar.productions.len() > MAX_PRODUCTIONS_FOR_OPTIMIZATION {
            debug!(4, "Skipping grammar optimization: {} productions exceeds threshold of {}",
                self.grammar.productions.len(), MAX_PRODUCTIONS_FOR_OPTIMIZATION);
            debug!(4, "Large grammars benefit more from a simple tokenizer than from optimization");
            return;
        }
        
        // NOTE: Repetitive pattern detection code exists below but is NOT enabled by default.
        // Reason: The NFA→DFA conversion may create many intermediate states (e.g., 21K for
        // diff grammars), but minimization reduces them to a reasonable number (e.g., 151).
        // Disabling optimization would harm runtime performance by using GLR parser instead
        // of an optimized DFA. The compile-time cost is acceptable.
        //
        // To enable detection (e.g., for debugging):
        // if std::env::var("DETECT_REPETITIVE_PATTERNS").is_ok() {
        //     if let Some(reason) = detect_repetitive_pattern_grammar(&self.grammar.productions) {
        //         debug!(4, "Skipping grammar optimization: repetitive pattern detected");
        //         debug!(4, "  {}", reason);
        //         return;
        //     }
        // }
        
        let total_start = std::time::Instant::now();
        debug!(4, "Starting grammar optimization with {} productions", self.grammar.productions.len());
        
        // Debug: show first few productions to understand structure
        if std::env::var("DEBUG_PRODUCTIONS").is_ok() {
            debug!(4, "Production details:");
            for (i, prod) in self.grammar.productions.iter().take(10).enumerate() {
                debug!(4, "  {}: {:?}", i, prod);
            }
            if self.grammar.productions.len() > 10 {
                debug!(4, "  ... ({} more)", self.grammar.productions.len() - 10);
            }
        }
        
        // Collect all non-terminal names
        let nt_names: HashSet<String> = self.grammar.productions.iter()
            .map(|p| p.lhs.0.clone())
            .collect();
        
        // Get the start non-terminal
        let start_prod = &self.grammar.productions[self.grammar.start_production_id];
        let start_nt = start_prod.lhs.0.clone();
        
        // Build initial equations: each NT maps to a RegexTerm
        let mut equations: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        
        // Build a combined ignore term from all ignore expressions
        let ignore_exprs: Vec<Expr> = self.grammar.ignore_terminal_ids.iter()
            .filter_map(|id| self.grammar.group_id_to_expr.get(&id.0).cloned())
            .collect();
        
        // Check if the ignore pattern is "explosion-prone".
        // An explosion-prone pattern has overlapping alternatives that can cause
        // exponential DFA state growth when inlined at multiple positions.
        //
        // Examples:
        // - Simple WS like `[ \t\n]+` is NOT explosion-prone (no overlapping alternatives)
        // - Comment patterns like `// [^\n]* | /* [^*]* */` ARE explosion-prone because:
        //   - Both start with `/`
        //   - The DFA must track which alternative we're "inside" at each WS* position
        //   - For n WS* positions, this leads to O(k^n) states where k is number of alternatives
        //
        // If the ignore pattern is explosion-prone, we skip inlining it during optimization.
        // The parser will handle WS between terminals instead.
        let is_explosion_prone = ignore_exprs.iter().any(|expr| {
            has_overlapping_alternatives(expr)
        });
        
        if is_explosion_prone {
            debug!(4, "Detected explosion-prone ignore pattern (overlapping alternatives with unbounded repetition)");
            debug!(4, "Skipping WS* inlining during optimization to prevent tokenizer DFA explosion");
        }
        
        let ignore_term = if ignore_exprs.is_empty() || is_explosion_prone {
            None
        } else if ignore_exprs.len() == 1 {
            Some(Rc::new(RegexTerm::Star(Rc::new(RegexTerm::Concrete(ignore_exprs[0].clone())))))
        } else {
            // Multiple ignore terminals: (ignore1 | ignore2 | ...)*
            let choices: Vec<Rc<RegexTerm>> = ignore_exprs.iter()
                .map(|e| Rc::new(RegexTerm::Concrete(e.clone())))
                .collect();
            Some(Rc::new(RegexTerm::Star(Rc::new(RegexTerm::Choice(choices)))))
        };

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
        debug!(4, "Built {} equations in {:?}", equations.len(), eq_start.elapsed());
        
        // Find NTs that are part of mutual recursion cycles
        let cycle_start = std::time::Instant::now();
        let cyclic_nts = find_cyclic_nts(&equations);
        debug!(4, "Found {} cyclic NTs in {:?}", cyclic_nts.len(), cycle_start.elapsed());
        if !cyclic_nts.is_empty() {
            debug!(5, "Cyclic NTs: {:?}", cyclic_nts);
        }
        
        // Build dependency graph and compute elimination order (reverse topological)
        let order_start = std::time::Instant::now();
        let elimination_order = compute_elimination_order(&equations, &start_nt);
        debug!(4, "Computed elimination order ({} NTs) in {:?}", elimination_order.len(), order_start.elapsed());
        
        // PHASE 1: Build solution map (solve each NT's equation with Arden's lemma)
        // We DON'T substitute into other equations yet - just solve each NT in isolation
        let mut solutions: HashMap<String, Rc<RegexTerm>> = HashMap::new();
        // Persistent cache for expansion - shared across all expand_with_solutions calls
        let mut expansion_cache: FxHashMap<*const RegexTerm, Rc<RegexTerm>> = FxHashMap::default();
        
        let phase1_start = std::time::Instant::now();
        let mut solved_count = 0;
        let mut failed_count = 0;
        let mut solve_time = std::time::Duration::ZERO;
        for (i, nt_to_eliminate) in elimination_order.iter().enumerate() {
            if i > 0 && i % 5000 == 0 {
                debug!(4, "Phase 1 progress: {}/{} NTs, solved={}, failed={}, solve={:?}", 
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
        
        // Log solution sizes
        let total_solution_size: usize = solutions.values().map(|s| s.size()).sum();
        let max_solution_size = solutions.values().map(|s| s.size()).max().unwrap_or(0);
        debug!(5, "Solutions: {} NTs, total size {}, max size {}", solutions.len(), total_solution_size, max_solution_size);
        
        debug!(4, "Phase 1: solved {} NTs, failed {} in {:?} (solve={:?})", 
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
        debug!(4, "Phase 2a: solved start NT symbolically in {:?}", phase2_start.elapsed());
        
        // Now expand all NtRefs in the solved start equation
        let expand_start = std::time::Instant::now();
        let solved_start_rc = Rc::new(solved_start_symbolic);
        debug!(5, "Start term size before expansion: {}", solved_start_rc.size());
        
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
            debug!(5, "Expanded term size: {}", expanded.size());
            // Check if the result still has NtRefs
            let remaining = expanded.referenced_nts();
            debug!(5, "After expansion, {} NtRefs remain: {:?}", remaining.len(), remaining);
            (expanded, !remaining.is_empty())
        };
        debug!(4, "Phase 2b: expanded start NT in {:?}", expand_start.elapsed());
        
        // If there are remaining NtRefs, we need to do partial optimization
        if has_remaining_ntrefs {
            // Compute transitive closure of NTs that depend on cyclic NTs
            let cyclic_dependent_nts = compute_cyclic_dependent_nts(&cyclic_nts, &self.grammar.productions);
            debug!(4, "Found {} NTs that depend on cyclic NTs (out of {} total)", 
                cyclic_dependent_nts.len(), nt_names.len());
            
            self.partial_optimize(&solutions, &mut expansion_cache, &cyclic_dependent_nts, ignore_term.as_ref());
            
            // Factor choice productions to group safe alternatives
            // NOTE: This optimization reduces GLR parser complexity (fewer productions)
            // but can INCREASE tokenizer complexity (more DFA states). For the Apollo
            // Router JSON schema, it reduces productions 22% (1066→835) but increases
            // DFA states 37% (81K→112K), resulting in net SLOWER compilation (26s→60s).
            // Therefore, disabled by default. Enable with ENABLE_CHOICE_FACTORING=1.
            if std::env::var("ENABLE_CHOICE_FACTORING").is_ok() {
                self.factor_choice_productions(&cyclic_dependent_nts, &solutions, &mut expansion_cache, ignore_term.as_ref());
            }
            
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
        
        // If we have ignore terminals and they're not explosion-prone, wrap the final expression
        // with WS* at the start and end. This allows the grammar to accept leading/trailing WS.
        let final_expr = if !ignore_exprs.is_empty() && !is_explosion_prone {
            let ignore_choice = if ignore_exprs.len() == 1 {
                ignore_exprs[0].clone()
            } else {
                Expr::Choice(ignore_exprs.clone())
            };
            let ignore_star = Expr::Quantifier(Box::new(ignore_choice), QuantifierType::ZeroOrMore);
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

        // Clear ignore terminal IDs since we're replacing the entire grammar
        self.grammar.ignore_terminal_ids.clear();

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
        expansion_cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
        cyclic_dependent_nts: &HashSet<String>,
        ignore_term: Option<&Rc<RegexTerm>>,
    ) {
        let start = std::time::Instant::now();
        
        debug!(5, "Pre-existing terminals before partial_optimize: {}", 
            self.grammar.regex_name_to_group_id.len());
        
        // Track which terminals already existed before optimization.
        // These should not be re-processed by handle_nullable_terminals_except.
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
        // Nullable terminals will be handled by transform_nullable_terminals in table.rs
        // Deduplicate terminals with identical expressions (but skip for very large expressions)
        let convert_start = std::time::Instant::now();
        
        // Map from expression to (canonical_nt_name, group_id)
        // This allows us to share terminals for NTs with identical expressions
        // NOTE: We only deduplicate small expressions to avoid expensive hash/eq for large ones
        let mut expr_to_terminal: HashMap<Expr, (String, usize)> = HashMap::new();
        // Map from NT name to the canonical NT name it should use
        let mut nt_to_canonical: HashMap<String, String> = HashMap::new();
        
        // Threshold for deduplication: skip if expression is larger than this
        const DEDUP_SIZE_THRESHOLD: usize = 500;
        
        debug!(5, "Starting NT->Expr conversion loop for {} NTs", nt_expand_order.len());
        let mut convert_count = 0;
        let mut skipped_dedup_count = 0;
        for nt in &nt_expand_order {
            if let Some(expanded) = expanded_solutions.get(nt) {
                if let Some(expr) = regex_term_to_expr_rc(expanded) {
                    // Estimate expression size for deduplication decision
                    let expr_size = expanded.size();
                    
                    if expr_size <= DEDUP_SIZE_THRESHOLD {
                        // Small expression: try to deduplicate
                        if let Some((canonical_nt, _group_id)) = expr_to_terminal.get(&expr) {
                            // Reuse existing terminal
                            nt_to_canonical.insert(nt.clone(), canonical_nt.clone());
                        } else {
                            let group_id = next_group_id;
                            next_group_id += 1;
                            new_terminals.insert(nt.clone(), (group_id, expr.clone()));
                            expr_to_terminal.insert(expr, (nt.clone(), group_id));
                            nt_to_canonical.insert(nt.clone(), nt.clone());
                        }
                    } else {
                        // Large expression: skip deduplication, just add it
                        skipped_dedup_count += 1;
                        let group_id = next_group_id;
                        next_group_id += 1;
                        new_terminals.insert(nt.clone(), (group_id, expr));
                        nt_to_canonical.insert(nt.clone(), nt.clone());
                    }
                }
            }
            convert_count += 1;
            if convert_count % 500 == 0 {
                debug!(5, "  Converted {}/{} NTs in {:?} (skipped_dedup={})", 
                    convert_count, nt_expand_order.len(), convert_start.elapsed(), skipped_dedup_count);
            }
        }
        
        let deduped_count = nt_expand_order.len() - new_terminals.len();
        if deduped_count > 0 {
            debug!(4, "Deduplicated {} terminals with identical expressions", deduped_count);
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
        
        // Handle nullable terminals that were created during optimization.
        // By doing this early (in optimization.rs), we enable better downstream optimizations.
        // We only handle the newly created terminals, not pre-existing ones.
        self.grammar.handle_nullable_terminals_except(&pre_existing_terminals);
        
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
    
    /// Deep check if a production references cyclic NTs transitively
    fn production_references_cyclic_deep(
        &self,
        prod: &Production,
        prods_by_lhs: &HashMap<String, Vec<Production>>,
        cyclic_dependent_nts: &HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> bool {
        for sym in &prod.rhs {
            if let Symbol::NonTerminal(nt) = sym {
                // Direct cyclic reference
                if cyclic_dependent_nts.contains(&nt.0) {
                    return true;
                }
                
                // Avoid infinite recursion
                if visited.contains(&nt.0) {
                    continue;
                }
                visited.insert(nt.0.clone());
                
                // Check if this NT's productions reference cyclic NTs
                if let Some(nt_prods) = prods_by_lhs.get(&nt.0) {
                    for nt_prod in nt_prods {
                        if self.production_references_cyclic_deep(nt_prod, prods_by_lhs, cyclic_dependent_nts, visited) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    
    /// Factor choice-based productions to group safe alternatives and common tails.
    /// 
    /// This optimization applies to non-terminals with multiple alternative productions.
    /// Unlike automata-level optimization (converting to regex terminals), this does
    /// GRAMMAR-LEVEL factoring by creating helper non-terminals.
    /// 
    /// Example transformation:
    ///   A ::= safe1 | safe2 | key1 : cyclic | key2 : cyclic
    /// becomes:
    ///   A ::= A_SAFE | A_KEYS : cyclic
    ///   A_SAFE ::= safe1 | safe2  -- new NON-TERMINAL (not regex terminal!)
    ///   A_KEYS ::= key1 | key2     -- new NON-TERMINAL
    /// 
    /// This reduces the number of alternatives the GLR parser must handle without
    /// forcing the tokenizer to expand everything into a DFA.
    fn factor_choice_productions(
        &mut self,
        cyclic_dependent_nts: &HashSet<String>,
        _solutions: &HashMap<String, Rc<RegexTerm>>,
        _expansion_cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
        ignore_term: Option<&Rc<RegexTerm>>,
    ) {
        let start = std::time::Instant::now();
        
        // Group productions by their LHS
        let mut prods_by_lhs: HashMap<String, Vec<Production>> = HashMap::new();
        for prod in &self.grammar.productions {
            prods_by_lhs.entry(prod.lhs.0.clone())
                .or_insert_with(Vec::new)
                .push(prod.clone());
        }
        
        // Identify NTs with multiple alternatives that can benefit from factoring
        // Match Python script logic: factor rules starting with '_' (except '_json')
        // that have multiple alternatives. SKIP auto-generated NTs with [..] suffixes
        // since they're intermediate and will be optimized differently.
        let mut factoring_candidates: Vec<String> = Vec::new();
        let mut filtered_out_count = 0;
        for (nt, prods) in &prods_by_lhs {
            if prods.len() < 2 {
                continue; // Need at least 2 alternatives
            }
            
            // Skip auto-generated NTs with [..] suffixes
            if nt.contains('[') || nt.contains(']') {
                continue;
            }
            
            // Factor internal NTs (starting with '_'), skip JSON primitives
            if !nt.starts_with('_') || nt.starts_with("_json") {
                filtered_out_count += 1;
                continue;
            }
            
            factoring_candidates.push(nt.clone());
        }
        
        if factoring_candidates.is_empty() {
            debug!(5, "Factor choice: no candidates found");
            return;
        }
        
        debug!(4, "Factor choice: {} NTs can be factored ({} filtered out)", factoring_candidates.len(), filtered_out_count);
        
        if factoring_candidates.len() < 20 {
            debug!(5, "Factoring candidates: {:?}", factoring_candidates);
        } else {
            debug!(5, "First 20 candidates: {:?}", &factoring_candidates[..20]);
        }
        
        // Build factored replacements
        let mut factored_replacements: HashMap<String, Vec<Production>> = HashMap::new();
        let mut new_helper_productions: Vec<Production> = Vec::new();
        let mut helper_nt_counter = 0;
        
        for nt_name in &factoring_candidates {
            let prods = &prods_by_lhs[nt_name];
            
            // Separate into safe and unsafe using deep checking
            let mut safe_prods: Vec<&Production> = Vec::new();
            let mut unsafe_prods: Vec<&Production> = Vec::new();
            
            for prod in prods {
                let mut visited = HashSet::new();
                let refs_cyclic = self.production_references_cyclic_deep(
                    prod,
                    &prods_by_lhs,
                    cyclic_dependent_nts,
                    &mut visited
                );
                
                if refs_cyclic {
                    unsafe_prods.push(prod);
                } else {
                    safe_prods.push(prod);
                }
            }
            
            let mut nt_replacements: Vec<Production> = Vec::new();
            
            // Create helper NT for safe productions if > 1
            if safe_prods.len() > 1 {
                let helper_nt_name = format!("__FACTOR_{}_SAFE_{}", nt_name, helper_nt_counter);
                helper_nt_counter += 1;
                
                // Create helper productions (one per safe alternative)
                for safe_prod in &safe_prods {
                    new_helper_productions.push(Production {
                        lhs: NonTerminal(helper_nt_name.clone()),
                        rhs: safe_prod.rhs.clone(),
                    });
                }
                
                // Add single production referencing helper
                nt_replacements.push(Production {
                    lhs: NonTerminal(nt_name.clone()),
                    rhs: vec![Symbol::NonTerminal(NonTerminal(helper_nt_name.clone()))],
                });
                
                debug!(5, "Created helper NT '{}' with {} alternatives", helper_nt_name, safe_prods.len());
            } else {
                // Keep safe productions as-is
                nt_replacements.extend(safe_prods.iter().map(|&p| p.clone()));
            }
            
            // Group unsafe productions by their tail pattern
            let mut tail_groups: HashMap<String, Vec<&Production>> = HashMap::new();
            let mut other_unsafe: Vec<&Production> = Vec::new();
            
            for prod in &unsafe_prods {
                if let Some((head_symbols, tail_nt)) = self.extract_key_tail_pattern(prod) {
                    // Check if head is "simple" (no cyclic refs) using deep checking
                    let head_prod = Production {
                        lhs: NonTerminal("__temp".to_string()),
                        rhs: head_symbols.clone(),
                    };
                    let mut visited = HashSet::new();
                    let head_is_simple = !self.production_references_cyclic_deep(
                        &head_prod,
                        &prods_by_lhs,
                        cyclic_dependent_nts,
                        &mut visited
                    );
                    
                    if head_is_simple {
                        tail_groups.entry(tail_nt).or_insert_with(Vec::new).push(prod);
                        continue;
                    }
                }
                
                other_unsafe.push(prod);
            }
            
            // Create helper NTs for each tail group
            // Even single key+tail patterns benefit from factoring! It moves literals to tokenizer.
            for (tail_nt, group_prods) in tail_groups {
                let clean_tail = tail_nt.replace(|c: char| !c.is_alphanumeric(), "");
                let helper_nt_name = if group_prods.len() == 1 {
                    // Single head: name after the head
                    let head_str = group_prods[0].rhs.iter()
                        .take(group_prods[0].rhs.len() - 2) // exclude ':' and tail
                        .filter_map(|sym| match sym {
                            Symbol::Terminal(Terminal::Literal(bytes)) => {
                                String::from_utf8(bytes.clone()).ok()
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    let clean_head = head_str.replace(|c: char| !c.is_alphanumeric(), "");
                    format!("__FACTOR_KEY_{}_{}", clean_head, helper_nt_counter)
                } else {
                    // Multiple heads: name after the tail
                    format!("__FACTOR_{}_KEYS_{}_{}", nt_name, clean_tail, helper_nt_counter)
                };
                helper_nt_counter += 1;
                
                // Create helper productions for keys (without the tail)
                for group_prod in &group_prods {
                    if let Some((head_symbols, _)) = self.extract_key_tail_pattern(group_prod) {
                        new_helper_productions.push(Production {
                            lhs: NonTerminal(helper_nt_name.clone()),
                            rhs: head_symbols,
                        });
                    }
                }
                
                // Add single production: NT ::= HELPER_KEYS ':' TAIL
                nt_replacements.push(Production {
                    lhs: NonTerminal(nt_name.clone()),
                    rhs: vec![
                        Symbol::NonTerminal(NonTerminal(helper_nt_name.clone())),
                        Symbol::Terminal(Terminal::Literal(vec![b':'])),
                        Symbol::NonTerminal(NonTerminal(tail_nt.clone())),
                    ],
                });
                
                debug!(5, "Created helper NT '{}' with {} key alternative(s)", helper_nt_name, group_prods.len());
            }
            
            // Keep other unsafe productions
            nt_replacements.extend(other_unsafe.iter().map(|&p| p.clone()));
            
            factored_replacements.insert(nt_name.clone(), nt_replacements);
        }
        
        // Build new productions list maintaining original order
        let mut new_productions: Vec<Production> = Vec::new();
        let mut seen_factored_nts: HashSet<String> = HashSet::new();
        
        for prod in &self.grammar.productions {
            let nt_name = &prod.lhs.0;
            
            if factored_replacements.contains_key(nt_name) {
                if !seen_factored_nts.contains(nt_name) {
                    seen_factored_nts.insert(nt_name.clone());
                    new_productions.extend(factored_replacements[nt_name].iter().cloned());
                }
            } else {
                new_productions.push(prod.clone());
            }
        }
        
        // Append helper productions at the end
        new_productions.extend(new_helper_productions.clone());
        
        if helper_nt_counter > 0 {
            let old_count = self.grammar.productions.len();
            
            // Find the start production
            let start_prod_lhs = self.grammar.productions[self.grammar.start_production_id].lhs.clone();
            self.grammar.productions = new_productions;
            self.grammar.start_production_id = self.grammar.productions.iter()
                .position(|p| p.lhs == start_prod_lhs)
                .unwrap_or(0);
            
            debug!(4, "Factor choice: {} productions -> {}, {} helper NTs created (in {:?})",
                old_count, self.grammar.productions.len(), helper_nt_counter, start.elapsed());
            
            // CRITICAL: Convert the helper NTs to terminals immediately!
            // This matches Python's approach where helpers become terminals via uppercase naming.
            // Group helper productions by LHS
            let mut helper_prods_by_lhs: HashMap<String, Vec<&Production>> = HashMap::new();
            for helper_prod in &new_helper_productions {
                helper_prods_by_lhs.entry(helper_prod.lhs.0.clone())
                    .or_insert_with(Vec::new)
                    .push(helper_prod);
            }
            
            // Convert each helper NT's productions to a terminal regex
            let current_max_group_id = self.grammar.group_id_to_expr.keys().max().copied().unwrap_or(0);
            let mut next_group_id = current_max_group_id + 1;
            let mut terminals_created = 0;
            
            for (helper_nt_name, helper_prods) in helper_prods_by_lhs {
                // Convert each alternative to a RegexTerm
                let mut alt_terms: Vec<Rc<RegexTerm>> = Vec::new();
                for prod in helper_prods {
                    if let Some(term) = self.symbols_to_regex_term(&prod.rhs, _solutions, _expansion_cache, ignore_term) {
                        alt_terms.push(term);
                    }
                }
                
                if alt_terms.is_empty() {
                    continue;
                }
                
                // Create choice or single term
                let final_term = if alt_terms.len() == 1 {
                    alt_terms.into_iter().next().unwrap()
                } else {
                    Rc::new(RegexTerm::make_choice(alt_terms))
                };
                
                // Convert to Expr using the module-level function
                let Some(expr) = regex_term_to_expr_rc(&final_term) else {
                    debug!(5, "Failed to convert helper NT '{}' to Expr", helper_nt_name);
                    continue;
                };
                
                // Add as terminal
                let terminal_name = helper_nt_name.clone();
                self.grammar.regex_name_to_group_id.insert(terminal_name.clone(), next_group_id);
                self.grammar.group_id_to_expr.insert(next_group_id, expr);
                next_group_id += 1;
                terminals_created += 1;
                
                debug!(5, "Converted helper NT '{}' to terminal", helper_nt_name);
            }
            
            // Remove helper productions from grammar since they're now terminals
            self.grammar.productions.retain(|p| {
                !p.lhs.0.starts_with("__FACTOR_")
            });
            
            // Update references to helper NTs to use Terminal instead of NonTerminal
            for prod in &mut self.grammar.productions {
                for sym in &mut prod.rhs {
                    if let Symbol::NonTerminal(nt) = sym {
                        if nt.0.starts_with("__FACTOR_") {
                            *sym = Symbol::Terminal(Terminal::RegexName(nt.0.clone()));
                        }
                    }
                }
            }
            
            debug!(4, "Converted {} helper NTs to terminals, {} productions remain",
                terminals_created, self.grammar.productions.len());
        } else {
            debug!(5, "Factor choice: no helpers created");
        }
    }
    fn extract_key_tail_pattern(&self, prod: &Production) -> Option<(Vec<Symbol>, String)> {
        // Pattern: ... ':' NT (at least 3 symbols, last is NT, second-to-last is ':')
        if prod.rhs.len() < 3 {
            return None;
        }
        
        // Check if last symbol is a non-terminal
        let last_sym = prod.rhs.last()?;
        let tail_nt = match last_sym {
            Symbol::NonTerminal(nt) => nt.0.clone(),
            _ => return None,
        };
        
        // Check if second-to-last is ':'
        let colon_sym = &prod.rhs[prod.rhs.len() - 2];
        let is_colon = match colon_sym {
            Symbol::Terminal(Terminal::Literal(bytes)) => bytes == &[b':'],
            _ => false,
        };
        
        if !is_colon {
            return None;
        }
        
        // Head is everything before the colon
        let head_symbols = prod.rhs[..prod.rhs.len() - 2].to_vec();
        Some((head_symbols, tail_nt))
    }
    
    /// Convert a production's RHS to a RegexTerm (unused - kept for potential future use)
    #[allow(dead_code)]
    fn production_to_regex_term(
        &self,
        prod: &Production,
        solutions: &HashMap<String, Rc<RegexTerm>>,
        expansion_cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
    ) -> Option<Rc<RegexTerm>> {
        self.symbols_to_regex_term(&prod.rhs, solutions, expansion_cache, None)
    }
    
    /// Convert a sequence of symbols to a RegexTerm.
    /// If ignore_term is provided, it will be interleaved between symbols (like WS*).
    fn symbols_to_regex_term(
        &self,
        symbols: &[Symbol],
        solutions: &HashMap<String, Rc<RegexTerm>>,
        expansion_cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
        ignore_term: Option<&Rc<RegexTerm>>,
    ) -> Option<Rc<RegexTerm>> {
        let terms: Option<Vec<Rc<RegexTerm>>> = symbols.iter().map(|sym| {
            match sym {
                Symbol::Terminal(term) => {
                    // Convert terminal to Expr, then to RegexTerm
                    let expr = match term {
                        Terminal::RegexName(name) => {
                            self.grammar.regex_name_to_group_id.get_by_left(name)
                                .and_then(|gid| self.grammar.group_id_to_expr.get(gid).cloned())
                        }
                        Terminal::Literal(bytes) => {
                            Some(Expr::U8Seq(bytes.clone()))
                        }
                    }?;
                    Some(Rc::new(RegexTerm::Concrete(expr)))
                }
                Symbol::NonTerminal(nt) => {
                    // Look up the solution and expand it
                    let solution = solutions.get(&nt.0)?;
                    Some(Self::expand_nt_refs_static(solution, solutions, expansion_cache))
                }
            }
        }).collect();
        
        terms.map(|ts| {
            // Interleave ignore_term (e.g., WS*) between symbols, just like production_rhs_to_regex_term does
            if let Some(ign) = ignore_term {
                let mut with_ignore = Vec::with_capacity(ts.len() * 2);
                for (i, t) in ts.into_iter().enumerate() {
                    if i > 0 {
                        with_ignore.push(ign.clone());
                    }
                    with_ignore.push(t);
                }
                Rc::new(RegexTerm::make_seq(with_ignore))
            } else {
                Rc::new(RegexTerm::make_seq(ts))
            }
        })
    }
    
    /// Static version of expand_nt_refs for use in symbols_to_regex_term
    fn expand_nt_refs_static(
        term: &Rc<RegexTerm>,
        solutions: &HashMap<String, Rc<RegexTerm>>,
        cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
    ) -> Rc<RegexTerm> {
        let ptr = Rc::as_ptr(term);
        if let Some(cached) = cache.get(&ptr) {
            return cached.clone();
        }
        
        let result = match term.as_ref() {
            RegexTerm::NtRef(nt_name) => {
                if let Some(solution) = solutions.get(nt_name) {
                    Self::expand_nt_refs_static(solution, solutions, cache)
                } else {
                    term.clone()
                }
            }
            RegexTerm::Seq(parts) => {
                let expanded: Vec<Rc<RegexTerm>> = parts.iter()
                    .map(|p| Self::expand_nt_refs_static(p, solutions, cache))
                    .collect();
                Rc::new(RegexTerm::make_seq(expanded))
            }
            RegexTerm::Choice(alts) => {
                let expanded: Vec<Rc<RegexTerm>> = alts.iter()
                    .map(|a| Self::expand_nt_refs_static(a, solutions, cache))
                    .collect();
                Rc::new(RegexTerm::make_choice(expanded))
            }
            RegexTerm::Star(inner) => {
                let expanded = Self::expand_nt_refs_static(inner, solutions, cache);
                Rc::new(RegexTerm::make_star(expanded))
            }
            _ => term.clone(),
        };
        
        cache.insert(ptr, result.clone());
        result
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
        
        // Also preserve ignore terminals - they're not in productions but are used by the parser
        for ignore_id in &self.grammar.ignore_terminal_ids {
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
            
            // Update ignore_terminal_ids if any were renumbered
            let old_ignore_ids: HashSet<TerminalID> = self.grammar.ignore_terminal_ids.drain().collect();
            for old_id in old_ignore_ids {
                if let Some(&new_id) = old_to_new.get(&old_id.0) {
                    self.grammar.ignore_terminal_ids.insert(TerminalID(new_id));
                    debug!(5, "Updated ignore_terminal_id: {} -> {}", old_id.0, new_id);
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
    
    // Build reverse dependency graph for propagation
    let mut rev_deps: HashMap<String, HashSet<String>> = HashMap::new();
    for (nt, refs) in &deps {
        for r in refs {
            rev_deps.entry(r.clone()).or_insert_with(HashSet::new).insert(nt.clone());
        }
    }
    
    // Use worklist algorithm: start with cyclic NTs and propagate to dependents
    let mut result: HashSet<String> = cyclic_nts.clone();
    let mut worklist: Vec<String> = cyclic_nts.iter().cloned().collect();
    
    while let Some(nt) = worklist.pop() {
        // Any NT that references this one becomes cyclic-dependent
        if let Some(dependents) = rev_deps.get(&nt) {
            for dep in dependents {
                if result.insert(dep.clone()) {
                    worklist.push(dep.clone());
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
    
    fn make_star(inner: Rc<RegexTerm>) -> RegexTerm {
        match inner.as_ref() {
            RegexTerm::Epsilon => RegexTerm::Epsilon,
            RegexTerm::Star(_) => (*inner).clone(), // (a*)* = a*
            _ => RegexTerm::Star(inner),
        }
    }
    
    // Check if this term matches the empty string
    fn is_nullable(&self) -> bool {
        match self {
            RegexTerm::Epsilon => true,
            RegexTerm::Star(_) => true,
            RegexTerm::Concrete(e) => {
                matches!(get_expr_nullability(e), ExprNullability::AlwaysNull | ExprNullability::CanBeNull)
            }
            // Non-terminals nullability is unknown here (requires solving), assume false to be safe
            // for local simplification.
            RegexTerm::NtRef(_) => false,
            RegexTerm::Seq(terms) => terms.iter().all(|t| t.is_nullable()),
            RegexTerm::Choice(terms) => terms.iter().any(|t| t.is_nullable()),
        }
    }

    fn make_choice(terms: Vec<Rc<RegexTerm>>) -> RegexTerm {
        let mut flat = Vec::new();
        let mut has_epsilon = false;
        
        for t in terms {
            // Only flatten if this Rc is NOT shared (strong_count == 1)
            // This avoids O(n²) blowup when flattening shared Choice subtrees
            let should_flatten = Rc::strong_count(&t) == 1;
            match t.as_ref() {
                RegexTerm::Choice(inner) if should_flatten => {
                    flat.extend(inner.iter().cloned())
                }
                RegexTerm::Epsilon => {
                    has_epsilon = true;
                }
                _ => flat.push(t),
            }
        }
        
        // Epsilon Absorption: DISABLED for performance
        // The is_nullable() traversal is too expensive for large regex trees.
        // We'll keep explicit epsilons - they'll be simplified later or are harmless.
        if has_epsilon {
            flat.push(Rc::new(RegexTerm::Epsilon));
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
        let mut visited: HashSet<*const RegexTerm> = HashSet::new();
        self.collect_nts_memoized(&mut result, &mut visited);
        result
    }
    
    fn collect_nts_memoized(&self, result: &mut HashSet<String>, visited: &mut HashSet<*const RegexTerm>) {
        match self {
            RegexTerm::Epsilon | RegexTerm::Concrete(_) => {}
            RegexTerm::NtRef(n) => { result.insert(n.clone()); }
            RegexTerm::Seq(terms) | RegexTerm::Choice(terms) => {
                for t in terms {
                    let ptr = Rc::as_ptr(t);
                    if visited.contains(&ptr) {
                        continue;
                    }
                    visited.insert(ptr);
                    t.collect_nts_memoized(result, visited);
                }
            }
            RegexTerm::Star(inner) => {
                let ptr = Rc::as_ptr(inner);
                if !visited.contains(&ptr) {
                    visited.insert(ptr);
                    inner.collect_nts_memoized(result, visited);
                }
            }
        }
    }
    
    fn collect_nts(&self, result: &mut HashSet<String>) {
        // Legacy non-memoized version - kept for compatibility
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
    contains_cache: &mut FxHashMap<*const RegexTerm, bool>
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
    contains_cache: &mut FxHashMap<*const RegexTerm, bool>
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
    visited: &mut FxHashMap<*const RegexTerm, bool>
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
    let mut cache: FxHashMap<*const RegexTerm, Rc<RegexTerm>> = FxHashMap::default();
    expand_with_solutions_cached(term, solutions, &mut cache)
}

/// Expand all NtRefs in a term by substituting solutions.
/// Uses iterative deepening with explicit stack to avoid stack overflow.
/// 
/// `expanding` tracks NTs currently being expanded to detect cycles.
fn expand_with_solutions_cached(
    root: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>
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
    cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
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
    cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
    expanding: &mut HashSet<String>,
    skip_nts: &HashSet<String>
) -> Option<Rc<RegexTerm>> {
    // Skip stacker for now to test performance impact
    expand_with_solutions_cached_impl(root, solutions, cache, expanding, skip_nts)
}

fn expand_with_solutions_cached_impl(
    term: &Rc<RegexTerm>,
    solutions: &HashMap<String, Rc<RegexTerm>>,
    cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
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
            // CACHE the expanded solution - this is crucial for avoiding re-expansion!
            cache.insert(sol_ptr, result.clone());
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
    let mut cache: FxHashMap<*const RegexTerm, Rc<RegexTerm>> = FxHashMap::default();
    let mut contains_cache: FxHashMap<*const RegexTerm, bool> = FxHashMap::default();
    let result = substitute_nt_rc_cached(term, nt, replacement, &mut cache, &mut contains_cache);
    result
}

fn substitute_nt_rc_cached(
    term: &Rc<RegexTerm>,
    nt: &str,
    replacement: &Rc<RegexTerm>,
    cache: &mut FxHashMap<*const RegexTerm, Rc<RegexTerm>>,
    contains_cache: &mut FxHashMap<*const RegexTerm, bool>
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
/// Check if an expression has overlapping alternatives that could cause DFA explosion.
/// 
/// An expression is "explosion-prone" if it has a Choice where:
/// 1. Multiple alternatives have the same first byte (overlapping prefixes)
/// 2. The alternatives have different termination conditions (e.g., one ends at '\n', another at '*/')
/// 3. The expression contains unbounded repetition (star/plus)
///
/// Example of explosion-prone pattern:
/// ```text
/// // [^\n]*       # line comment: starts with '/', ends at '\n'
/// /* [^*]* */    # block comment: starts with '/', ends at '*/'
/// ```
/// Both start with '/', so the DFA must track which alternative we're "inside".
/// When this pattern appears at multiple positions (e.g., WS* A WS* B WS* C),
/// the number of states grows exponentially: O(k^n) where k is alternatives and n is positions.
fn has_overlapping_alternatives(expr: &Expr) -> bool {
    use crate::datastructures::u8set::U8Set;
    
    // First, check if the expression contains any unbounded repetition
    if !has_unbounded_repetition(expr) {
        return false;
    }
    
    // Then check if there are overlapping alternatives
    has_overlapping_choice(expr)
}

/// Check if an expression contains any star (*) or plus (+) quantifier.
fn has_unbounded_repetition(expr: &Expr) -> bool {
    match expr {
        Expr::Quantifier(inner, q_type) => {
            matches!(q_type, QuantifierType::ZeroOrMore | QuantifierType::OneOrMore)
                || has_unbounded_repetition(inner)
        }
        Expr::Seq(children) | Expr::Choice(children) => {
            children.iter().any(|c| has_unbounded_repetition(c))
        }
        Expr::Shared(inner) => has_unbounded_repetition(inner),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
    }
}

/// Check if an expression has a Choice with overlapping first bytes.
fn has_overlapping_choice(expr: &Expr) -> bool {
    use crate::datastructures::u8set::U8Set;
    
    match expr {
        Expr::Choice(alternatives) => {
            // Get first bytes of each alternative
            let first_bytes: Vec<U8Set> = alternatives.iter()
                .map(|alt| get_first_bytes(alt))
                .collect();
            
            // Check for overlap between any pair of alternatives
            for i in 0..first_bytes.len() {
                for j in (i + 1)..first_bytes.len() {
                    let intersection = first_bytes[i].intersection(&first_bytes[j]);
                    if !intersection.is_empty() {
                        // Found overlapping alternatives
                        // Now check if they have different "end conditions"
                        // (This is a conservative check - we assume any overlap is problematic)
                        return true;
                    }
                }
            }
            
            // Also recursively check children
            alternatives.iter().any(|alt| has_overlapping_choice(alt))
        }
        Expr::Seq(children) => children.iter().any(|c| has_overlapping_choice(c)),
        Expr::Quantifier(inner, _) => has_overlapping_choice(inner),
        Expr::Shared(inner) => has_overlapping_choice(inner),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
    }
}

/// Get the set of possible first bytes for an expression.
fn get_first_bytes(expr: &Expr) -> crate::datastructures::u8set::U8Set {
    use crate::datastructures::u8set::U8Set;
    
    match expr {
        Expr::U8Seq(bytes) => {
            if bytes.is_empty() {
                U8Set::none()
            } else {
                U8Set::from_u8(bytes[0])
            }
        }
        Expr::U8Class(set) => set.clone(),
        Expr::Epsilon => U8Set::none(),
        Expr::Seq(children) => {
            if children.is_empty() {
                U8Set::none()
            } else {
                // For a sequence, the first bytes are the first bytes of the first non-epsilon child
                for child in children {
                    let first = get_first_bytes(child);
                    if !first.is_empty() || !can_be_empty(child) {
                        return first;
                    }
                }
                U8Set::none()
            }
        }
        Expr::Choice(alternatives) => {
            // Union of first bytes of all alternatives
            let mut result = U8Set::none();
            for alt in alternatives {
                result = result.union(&get_first_bytes(alt));
            }
            result
        }
        Expr::Quantifier(inner, q_type) => {
            match q_type {
                QuantifierType::ZeroOrMore | QuantifierType::ZeroOrOne => {
                    // Can be empty, so first bytes might be from what follows
                    // For our purposes, just return the inner's first bytes
                    get_first_bytes(inner)
                }
                QuantifierType::OneOrMore => get_first_bytes(inner),
            }
        }
        Expr::Shared(inner) => get_first_bytes(inner),
    }
}

/// Check if an expression can match the empty string (epsilon).
fn can_be_empty(expr: &Expr) -> bool {
    match expr {
        Expr::Epsilon => true,
        Expr::U8Seq(bytes) => bytes.is_empty(),
        Expr::U8Class(_) => false, // Must match exactly one byte
        Expr::Seq(children) => children.iter().all(|c| can_be_empty(c)),
        Expr::Choice(alternatives) => alternatives.iter().any(|a| can_be_empty(a)),
        Expr::Quantifier(_, q_type) => {
            matches!(q_type, QuantifierType::ZeroOrMore | QuantifierType::ZeroOrOne)
        }
        Expr::Shared(inner) => can_be_empty(inner),
    }
}

/// Detect grammars with repetitive patterns that cause exponential DFA blowup.
///
/// Returns Some(reason) if the grammar is repetitive, None otherwise.
///
/// Example: diff grammars for files with N identical lines:
/// ```text
/// S0 ::= LINE0 | S1;
/// S1 ::= LINE1 | S2;
/// ...
/// LINE0 ::= PLUS_LINE* CONTENT0 ( LINE1 | ... )?;
/// LINE1 ::= PLUS_LINE* CONTENT1 ( LINE2 | ... )?;
/// ...
/// ```
///
/// All S rules follow pattern: S{i} ::= LINE{i} | S{i+1}
/// All LINE rules follow pattern: LINE{i} ::= PLUS_LINE* CONTENT{i} ( LINE{i+1} | ... )?
///
/// When optimized to a single regex and converted to DFA, this creates 2^N states
/// because after reading identical content, the NFA is in multiple positions.
///
/// Detection heuristic:
/// - Count productions with identical structural patterns
/// - If >60% follow the same pattern, it's likely a repetitive grammar
fn detect_repetitive_pattern_grammar(productions: &[Production]) -> Option<String> {
    use std::collections::HashMap;
    
    let num_productions = productions.len();
    
    // Need at least 10 productions to be problematic
    if num_productions < 10 {
        return None;
    }
    
    // Group productions by their RHS pattern signature
    // Pattern signature: sequence of (symbol_type, is_optional) pairs
    let mut pattern_groups: HashMap<Vec<PatternElement>, Vec<String>> = HashMap::new();
    
    for prod in productions {
        let pattern = extract_pattern_signature(&prod.rhs);
        pattern_groups.entry(pattern).or_default().push(prod.lhs.0.clone());
    }
    
    // Find largest pattern group
    let Some((largest_pattern, largest_group)) = pattern_groups.iter()
        .max_by_key(|(_, group)| group.len())
    else {
        return None;
    };
    
    let largest_group_size = largest_group.len();
    let largest_group_pct = largest_group_size as f64 / num_productions as f64;
    
    // Threshold: if >60% of productions follow the same pattern
    const THRESHOLD: f64 = 0.6;
    
    if largest_group_pct > THRESHOLD {
        // Also check if the pattern involves alternatives or quantifiers
        // (which contribute to non-determinism)
        let has_nondeterminism = pattern_contains_nondeterminism(largest_pattern);
        
        if has_nondeterminism {
            return Some(format!(
                "{}/{} productions ({:.0}%) follow identical pattern with non-determinism",
                largest_group_size, num_productions, largest_group_pct * 100.0
            ));
        }
    }
    
    None
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PatternElement {
    /// Non-terminal reference
    NonTerminal,
    /// Terminal with symbol type (T=Terminal, NT=NonTerminal in its RHS)
    Terminal(TerminalPatternType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TerminalPatternType {
    /// Regex name terminal
    RegexName,
    /// Literal sequence of specific length
    Literal(usize),
    /// Character class (not used in simplified version)
    Class,
    /// Sequence of patterns (not used in simplified version)
    Seq(usize),
    /// Choice between alternatives (not used in simplified version)
    Choice(usize),
    /// Quantifier (*, +, ?) (not used in simplified version)
    Quantifier(QuantifierType),
    /// Epsilon (not used in simplified version)
    Epsilon,
}

/// Extract structural pattern from production RHS.
fn extract_pattern_signature(rhs: &[Symbol]) -> Vec<PatternElement> {
    rhs.iter().map(|symbol| {
        match symbol {
            Symbol::NonTerminal(_) => PatternElement::NonTerminal,
            Symbol::Terminal(term) => {
                // We can't access the expr here, so just use the terminal type
                match term {
                    Terminal::RegexName(_) => PatternElement::Terminal(TerminalPatternType::RegexName),
                    Terminal::Literal(bytes) => PatternElement::Terminal(TerminalPatternType::Literal(bytes.len())),
                }
            }
        }
    }).collect()
}

/// Extract pattern type from expression (not used currently).
fn extract_expr_pattern(expr: &Expr) -> TerminalPatternType {
    match expr {
        Expr::U8Seq(bytes) => TerminalPatternType::Literal(bytes.len()),
        Expr::U8Class(_) => TerminalPatternType::Class,
        Expr::Epsilon => TerminalPatternType::Epsilon,
        Expr::Seq(children) => TerminalPatternType::Seq(children.len()),
        Expr::Choice(alternatives) => TerminalPatternType::Choice(alternatives.len()),
        Expr::Quantifier(_, q_type) => TerminalPatternType::Quantifier(*q_type),
        Expr::Shared(inner) => extract_expr_pattern(inner),
    }
}

/// Check if a pattern contains elements that contribute to non-determinism.
fn pattern_contains_nondeterminism(pattern: &[PatternElement]) -> bool {
    pattern.iter().any(|elem| {
        match elem {
            PatternElement::NonTerminal => {
                // Non-terminals can introduce non-determinism if they have alternatives
                // We conservatively assume they do
                true
            }
            PatternElement::Terminal(term_type) => {
                matches!(term_type, 
                    TerminalPatternType::Choice(_) |
                    TerminalPatternType::Quantifier(_)
                )
            }
        }
    })
}