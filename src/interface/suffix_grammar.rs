//! Suffix Grammar Construction
//!
//! Transforms a grammar for language L into a grammar for Suf(L) = { y | ∃x: xy ∈ L }
//!
//! Algorithm:
//! 1. For each nonterminal A, create suffix nonterminal A• 
//! 2. For each terminal a, create helper T_a → a | ε
//! 3. For each production A → X₁...Xₖ, add: A• → sufSym(Xᵢ) X_{i+1}...Xₖ for i=1..k
//! 4. Start symbol is S• (suffix of original start symbol)

use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::interface::GrammarDefinition;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use bimap::BiBTreeMap;
use crate::finite_automata::Expr;

/// Marker for suffix nonterminals
const SUFFIX_MARKER: &str = "_suffix";

/// Marker for terminal helper nonterminals
const TERMINAL_HELPER_PREFIX: &str = "_T_";

/// Create the suffix nonterminal name from an original nonterminal
fn suffix_nonterminal_name(name: &str) -> String {
    format!("{}{}", name, SUFFIX_MARKER)
}

/// Create the terminal helper nonterminal name
fn terminal_helper_name(terminal: &Terminal) -> String {
    match terminal {
        Terminal::Literal(bytes) => {
            // Use hex encoding for the literal bytes
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            format!("{}{}", TERMINAL_HELPER_PREFIX, hex)
        }
        Terminal::RegexName(name) => {
            format!("{}{}", TERMINAL_HELPER_PREFIX, name)
        }
    }
}

/// Get the suffix symbol for a grammar symbol
/// - For nonterminal B, returns B•
/// - For terminal a, returns T_a (helper nonterminal)
fn suffix_symbol(symbol: &Symbol) -> Symbol {
    match symbol {
        Symbol::NonTerminal(nt) => {
            Symbol::NonTerminal(NonTerminal(suffix_nonterminal_name(&nt.0)))
        }
        Symbol::Terminal(t) => {
            Symbol::NonTerminal(NonTerminal(terminal_helper_name(t)))
        }
    }
}

/// Transform a grammar for language L into a grammar for Suf(L)
///
/// Given a grammar G = (N, Σ, P, S), constructs G_suf = (N', Σ, P', S•) where:
/// - N' = N ∪ { A• | A ∈ N } ∪ { T_a | a ∈ Σ }
/// - P' includes:
///   - All original productions P (to generate the "tail" after the suffix start)
///   - Terminal helpers: T_a → a | ε for each terminal a
///   - Suffix rules: A• → sufSym(X_i) X_{i+1}...X_k for each A → X_1...X_k and 1 ≤ i ≤ k
///
/// # Example
/// For the grammar S → aSb | ε (generating a^n b^n):
/// - Adds T_a → a | ε, T_b → b | ε
/// - Adds S• → T_a S b | S• b | T_b (for S → aSb)
/// - Adds S• → ε (for S → ε)
///
/// The resulting grammar generates all suffixes of a^n b^n.
pub fn grammar_to_suffix_grammar(grammar: &GrammarDefinition) -> GrammarDefinition {
    let mut new_productions = grammar.productions.clone();
    let mut new_literal_to_group_id = grammar.literal_to_group_id.clone();
    let mut new_regex_name_to_group_id = grammar.regex_name_to_group_id.clone();
    let mut new_group_id_to_expr = grammar.group_id_to_expr.clone();
    
    // Track which terminals we've seen (to create helpers)
    let mut terminals_seen: BTreeSet<Terminal> = BTreeSet::new();
    
    // Collect all terminals from productions
    for prod in &grammar.productions {
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                terminals_seen.insert(t.clone());
            }
        }
    }
    
    // Find the next available group_id
    let mut next_group_id = grammar.group_id_to_expr.keys()
        .chain(grammar.literal_to_group_id.right_values())
        .chain(grammar.regex_name_to_group_id.right_values())
        .max()
        .copied()
        .unwrap_or(0) + 1;
    
    // Create terminal helper productions: T_a → a | ε
    // We model this as a nonterminal with productions for 'a' and 'ε'
    for terminal in &terminals_seen {
        let helper_name = terminal_helper_name(terminal);
        let helper_nt = NonTerminal(helper_name.clone());
        
        // T_a → a (just the terminal)
        new_productions.push(Production {
            lhs: helper_nt.clone(),
            rhs: vec![Symbol::Terminal(terminal.clone())],
        });
        
        // T_a → ε (empty string)
        new_productions.push(Production {
            lhs: helper_nt.clone(),
            rhs: vec![], // Empty RHS = epsilon
        });
        
        // Register the helper as a nonterminal (no group_id needed since it's a nonterminal)
    }
    
    // Create suffix productions
    for prod in &grammar.productions {
        let suffix_lhs = NonTerminal(suffix_nonterminal_name(&prod.lhs.0));
        
        if prod.rhs.is_empty() {
            // A → ε produces A• → ε
            new_productions.push(Production {
                lhs: suffix_lhs.clone(),
                rhs: vec![],
            });
        } else {
            // For A → X_1 X_2 ... X_k, add:
            // A• → sufSym(X_i) X_{i+1} ... X_k for i = 1..k
            for i in 0..prod.rhs.len() {
                let mut suffix_rhs = Vec::new();
                
                // The suffix symbol for position i (where suffix starts)
                suffix_rhs.push(suffix_symbol(&prod.rhs[i]));
                
                // Keep the rest of the symbols as-is (X_{i+1} ... X_k)
                for j in (i + 1)..prod.rhs.len() {
                    suffix_rhs.push(prod.rhs[j].clone());
                }
                
                new_productions.push(Production {
                    lhs: suffix_lhs.clone(),
                    rhs: suffix_rhs,
                });
            }
        }
    }
    
    // Find the new start production (S•) and move it to index 0
    // This is required because generate_glr_parser_with_maps hardcodes start_production_id = 0
    let original_start_name = &grammar.productions[grammar.start_production_id].lhs.0;
    let suffix_start_name = suffix_nonterminal_name(original_start_name);
    
    // Find the index of the first suffix start production
    let suffix_start_idx = new_productions
        .iter()
        .position(|p| p.lhs.0 == suffix_start_name)
        .expect("Suffix start production must exist");
    
    // Reorder productions: put suffix start first, then other suffix, then helpers, then original
    // This is required because generate_glr_parser_with_maps hardcodes start_production_id = 0
    let mut reordered_productions = Vec::new();
    
    // First, collect all suffix start productions (the ones for the start nonterminal's suffix)
    for prod in &new_productions {
        if prod.lhs.0 == suffix_start_name {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Then, collect all OTHER suffix productions (non-start suffix nonterminals)
    for prod in &new_productions {
        if prod.lhs.0.ends_with(SUFFIX_MARKER) && prod.lhs.0 != suffix_start_name {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Then, collect all terminal helper productions (needed by suffix productions)
    for prod in &new_productions {
        if prod.lhs.0.starts_with(TERMINAL_HELPER_PREFIX) {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Finally, collect original productions (needed for the "tail" after suffix)
    for prod in &new_productions {
        if !prod.lhs.0.ends_with(SUFFIX_MARKER) && !prod.lhs.0.starts_with(TERMINAL_HELPER_PREFIX) {
            reordered_productions.push(prod.clone());
        }
    }
    
    // Now reordered_productions[0] should be a suffix start production
    let new_start_production_id = 0;
    
    GrammarDefinition {
        productions: reordered_productions,
        start_production_id: new_start_production_id,
        literal_to_group_id: new_literal_to_group_id,
        regex_name_to_group_id: new_regex_name_to_group_id,
        group_id_to_expr: new_group_id_to_expr,
        ignore_terminal_ids: grammar.ignore_terminal_ids.clone(),
        external_name_to_group_id: grammar.external_name_to_group_id.clone(),
    }
}

/// Validate terminal DWA paths against the suffix grammar.
/// 
/// Samples paths from the terminal DWA and checks what proportion are accepted
/// by the suffix parser. This validates that the terminal DWA isn't generating
/// spurious paths that don't correspond to valid grammar derivations.
///
/// # Arguments
/// * `dwa` - The terminal DWA to sample paths from
/// * `grammar` - The original grammar definition
/// * `terminals_count` - Number of terminals in the grammar (labels < this are terminal IDs)
/// * `num_samples` - Number of paths to sample
///
/// # Returns
/// The proportion of sampled paths that are accepted (0.0 to 1.0)
pub fn validate_terminal_dwa_paths(
    dwa: &crate::dwa_i32::DWA,
    grammar: &GrammarDefinition,
    terminals_count: usize,
    num_samples: usize,
) -> f64 {
    validate_terminal_dwa_paths_verbose(dwa, grammar, terminals_count, num_samples, false)
}

/// Verbose version of validate_terminal_dwa_paths that prints debug info
pub fn validate_terminal_dwa_paths_verbose(
    dwa: &crate::dwa_i32::DWA,
    grammar: &GrammarDefinition,
    terminals_count: usize,
    num_samples: usize,
    verbose: bool,
) -> f64 {
    use crate::interface::CompiledGrammar;
    use crate::glr::table::{NonTerminalID, StateID, TerminalID};
    use rand::Rng;
    
    if verbose {
        println!("\n=== Original Grammar ===");
        println!("Productions:");
        for (i, prod) in grammar.productions.iter().enumerate() {
            let marker = if i == grammar.start_production_id { " <-- START" } else { "" };
            println!("  {}: {}{}", i, prod, marker);
        }
        println!("\nLiteral to group_id:");
        for (val, id) in &grammar.literal_to_group_id {
            println!("  {:?} -> {}", val, id);
        }
        println!("\nRegex name to group_id:");
        for (name, id) in &grammar.regex_name_to_group_id {
            println!("  {} -> {}", name, id);
        }
    }
    
    // Build suffix grammar and compile it
    let suffix_grammar = grammar_to_suffix_grammar(grammar);
    
    if verbose {
        println!("\n=== Suffix Grammar Productions ===");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            let marker = if i == suffix_grammar.start_production_id { " <-- START" } else { "" };
            println!("  {}: {}{}", i, prod, marker);
        }
    }
    
    let suffix_parser = CompiledGrammar::glr_parser_from_definition(&suffix_grammar);
    
    if verbose {
        println!("\n=== Suffix Parser Terminal Map ===");
        for (term, tid) in suffix_parser.terminal_map.iter() {
            println!("  {:?} -> TerminalID({})", term, tid.0);
        }
        
        // Print the parser table to understand what's happening
        println!("\n=== Suffix Parser Table (first few rows) ===");
        println!("{}", suffix_parser);
    }
    
    // Sample paths from the terminal DWA
    let mut rng = rand::thread_rng();
    let paths = dwa.sample_paths(num_samples, &mut rng);
    
    if verbose {
        println!("\n=== Terminal DWA Info ===");
        println!("  DWA: {}", dwa.stats());
        println!("  Terminals count: {}", terminals_count);
        println!("  Sampled {} paths", paths.len());
    }
    
    if paths.is_empty() {
        return 1.0; // No paths = vacuously valid
    }
    
    let mut valid_count = 0;
    
    for (i, path) in paths.iter().enumerate() {
        // Extract terminal labels (filter out TSID labels which are >= terminals_count)
        let terminal_ids: Vec<TerminalID> = path
            .iter()
            .map(|(label, _state)| *label as usize)
            .filter(|&label| label < terminals_count)
            .map(|label| TerminalID(label))
            .collect();
        
        // Parse the terminal sequence with the suffix parser
        let state = suffix_parser.parse(&terminal_ids, None);
        let is_valid = state.is_ok();
        
        if verbose && i < 10 {
            let all_labels: Vec<_> = path.iter().map(|(l, _)| *l).collect();
            let term_labels: Vec<_> = terminal_ids.iter().map(|t| t.0).collect();
            println!("\nPath {}: all_labels={:?}, terminal_ids={:?}, valid={}", 
                     i, all_labels, term_labels, is_valid);
            
            // Debug: step through parsing terminal by terminal
            if !is_valid && !terminal_ids.is_empty() {
                let mut debug_state = suffix_parser.init_glr_parser(None);
                println!("  Initial: is_ok={}", debug_state.is_ok());
                for (j, tid) in terminal_ids.iter().enumerate() {
                    debug_state.step(*tid);
                    println!("  After step {}: terminal={}, is_ok={}", 
                             j, tid.0, debug_state.is_ok());
                }
            }
        }
        
        if is_valid {
            valid_count += 1;
        }
    }
    
    if verbose {
        println!("\nValid: {}/{} ({:.2}%)", valid_count, paths.len(), 
                 100.0 * valid_count as f64 / paths.len() as f64);
    }
    
    valid_count as f64 / paths.len() as f64
}

/// Get the original nonterminal name from a suffix nonterminal name
pub fn original_nonterminal_name(suffix_name: &str) -> Option<&str> {
    suffix_name.strip_suffix(SUFFIX_MARKER)
}

/// Check if a nonterminal name is a suffix nonterminal
pub fn is_suffix_nonterminal(name: &str) -> bool {
    name.ends_with(SUFFIX_MARKER)
}

/// Check if a nonterminal name is a terminal helper
pub fn is_terminal_helper(name: &str) -> bool {
    name.starts_with(TERMINAL_HELPER_PREFIX)
}

/// Prune a terminal DWA using the suffix grammar.
///
/// This walks the DWA and removes transitions that would lead to invalid
/// suffix parser states. A transition on terminal T is pruned if, after
/// feeding T to the suffix parser, the parser has no valid states.
///
/// The DWA labels are:
/// - 0..(terminals_count-1): terminal IDs  
/// - terminals_count..: TSID labels (tokenizer state IDs)
///
/// TSID transitions are not pruned (they represent tokenizer state changes,
/// not grammar terminals).
///
/// Ignored terminals (like IGNORE/whitespace) are also not pruned because
/// they can appear anywhere between other terminals.
pub fn prune_dwa_with_suffix_grammar(
    dwa: &mut crate::dwa_i32::DWA,
    grammar: &GrammarDefinition,
    terminal_map: &bimap::BiBTreeMap<Terminal, crate::glr::table::TerminalID>,
    terminals_count: usize,
) -> (usize, usize) {
    use crate::interface::CompiledGrammar;
    use crate::glr::table::{NonTerminalID, StateID, TerminalID};
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    
    crate::debug!(4, "Starting suffix grammar DWA pruning");
    
    // Collect ignored terminal IDs - these should not be pruned
    let ignored_tids: BTreeSet<usize> = grammar.ignore_terminal_ids.iter()
        .map(|tid| tid.0)
        .collect();
    crate::debug!(4, "  Ignored terminal IDs: {:?}", ignored_tids);
    
    // Build suffix grammar
    crate::debug!(4, "  Building suffix grammar...");
    let suffix_grammar = grammar_to_suffix_grammar(grammar);
    crate::debug!(4, "  Suffix grammar: {} productions", suffix_grammar.productions.len());
    
    // Print suffix grammar productions at debug level 6 (level 5 is already very noisy).
    for (i, prod) in suffix_grammar.productions.iter().enumerate() {
        crate::debug!(6, "    Suffix prod {}: {} -> {:?}", i, prod.lhs.0, prod.rhs);
    }
    
    // NOTE: Previously had a limit on suffix grammar size (MAX_SUFFIX_PRODUCTIONS = 1250)
    // that would skip suffix pruning for complex grammars. This limit has been removed
    // because we need to handle all grammars without skipping. If compilation is slow,
    // we need to optimize the suffix grammar construction itself rather than skip.
    
    crate::debug!(4, "  Compiling suffix grammar...");
    let suffix_parser = CompiledGrammar::glr_parser_from_definition(&suffix_grammar);
    crate::debug!(4, "  Suffix grammar compiled");
    
    // Build mapping from original terminal IDs to suffix parser terminal IDs
    // The suffix parser may have different terminal IDs
    let mut orig_to_suffix_tid: BTreeMap<usize, TerminalID> = BTreeMap::new();
    for (term, orig_tid) in terminal_map.iter() {
        // Look up the same terminal in the suffix parser
        if let Some(suffix_tid) = suffix_parser.terminal_map.get_by_left(term) {
            orig_to_suffix_tid.insert(orig_tid.0, *suffix_tid);
        }
    }
    crate::debug!(4, "    Terminal mapping: {} of {} terminals mapped", orig_to_suffix_tid.len(), terminal_map.len());

    // Precompute a conservative mapping of NonTerminalID -> possible GOTO states.
    // This is used to approximate successor states when we see reduce actions with len > 0.
    let mut reduce_goto_states_by_nt: BTreeMap<NonTerminalID, BTreeSet<StateID>> = BTreeMap::new();
    for (_state_id, row) in crate::glr::table::iter_rows(&suffix_parser.table) {
        for (nt_id, goto) in row.get_gotos() {
            if let Some(goto_state) = goto.state_id {
                reduce_goto_states_by_nt
                    .entry(*nt_id)
                    .or_default()
                    .insert(goto_state);
            }
        }
    }

    // Precompute which nonterminals can be reduced (len > 0) from each parser state,
    // ignoring lookahead. This helps build a conservative closure for parser states.
    let mut reduce_nts_by_state: BTreeMap<StateID, BTreeSet<NonTerminalID>> = BTreeMap::new();
    for (state_id, row) in crate::glr::table::iter_rows(&suffix_parser.table) {
        let mut nts: BTreeSet<NonTerminalID> = BTreeSet::new();
        for (_term, action) in row.get_shifts_and_reduces_map() {
            use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                    if len > 0 {
                        nts.insert(nonterminal_id);
                    }
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                    for (&reduce_len, nonterminals) in reduces.iter() {
                        if reduce_len > 0 {
                            for nt_id in nonterminals.keys() {
                                nts.insert(*nt_id);
                            }
                        }
                    }
                }
                Stage7ShiftsAndReducesLookaheadValue::Shift(_) => {}
            }
        }
        if let Some(default_reduce) = &row.default_reduce {
            use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
            if let Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } = default_reduce {
                if *len > 0 {
                    nts.insert(*nonterminal_id);
                }
            }
        }
        if !nts.is_empty() {
            reduce_nts_by_state.insert(*state_id, nts);
        }
    }

    let expand_parser_states = |states: &BTreeSet<StateID>| -> BTreeSet<StateID> {
        let mut expanded = states.clone();
        let mut queue: VecDeque<StateID> = states.iter().copied().collect();
        while let Some(state_id) = queue.pop_front() {
            if let Some(nts) = reduce_nts_by_state.get(&state_id) {
                for nt_id in nts {
                    if let Some(gotos) = reduce_goto_states_by_nt.get(nt_id) {
                        for &goto_state in gotos {
                            if expanded.insert(goto_state) {
                                queue.push_back(goto_state);
                            }
                        }
                    }
                }
            }
        }
        expanded
    };

    let debug_suffix_prune = std::env::var("DEBUG_SUFFIX_PRUNE_JSON").is_ok();
    let mut debug_label_ids: BTreeSet<usize> = BTreeSet::new();
    if debug_suffix_prune {
        let debug_terms = [
            Terminal::RegexName("MINUS".to_string()),
            Terminal::RegexName("QUOTE".to_string()),
            Terminal::Literal(vec![b']']),
            Terminal::Literal(vec![b',']),
            Terminal::Literal(vec![b':']),
        ];
        for term in debug_terms.iter() {
            if let Some(tid) = terminal_map.get_by_left(term) {
                debug_label_ids.insert(tid.0);
                eprintln!("DEBUG_SUFFIX_PRUNE_JSON label {} => {:?}", tid.0, term);
            }
        }
        eprintln!("DEBUG_SUFFIX_PRUNE_JSON labels: {:?}", debug_label_ids);
    }
    
    // Track which suffix parser states are reachable at each DWA state
    // Key: DWA state ID, Value: Set of GLR parser state IDs
    type ParserStateSet = BTreeSet<crate::glr::table::StateID>;
    let mut dwa_to_parser_states: BTreeMap<usize, ParserStateSet> = BTreeMap::new();
    
    // Initialize: start DWA state maps to initial suffix parser state
    let initial_parser_state_id = suffix_parser.start_state_id;
    dwa_to_parser_states.insert(dwa.body.start_state, {
        let mut set = BTreeSet::new();
        set.insert(initial_parser_state_id);
        set
    });
    
    // BFS to propagate parser states through the DWA
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut in_queue: BTreeSet<usize> = BTreeSet::new();
    queue.push_back(dwa.body.start_state);
    in_queue.insert(dwa.body.start_state);
    
    while let Some(dwa_state) = queue.pop_front() {
        in_queue.remove(&dwa_state);
        let mut parser_states = dwa_to_parser_states.get(&dwa_state).cloned().unwrap_or_default();
        let expanded = expand_parser_states(&parser_states);
        if expanded.len() > parser_states.len() {
            dwa_to_parser_states.insert(dwa_state, expanded.clone());
            parser_states = expanded;
        }
        crate::debug!(6, "  BFS: DWA state {} with parser_states {:?}", dwa_state, parser_states);
        
        let mut fallback_reduce_goto_states: BTreeSet<StateID> = BTreeSet::new();
        for state_id in parser_states.iter() {
            if let Some(nts) = reduce_nts_by_state.get(state_id) {
                for nt_id in nts {
                    if let Some(gotos) = reduce_goto_states_by_nt.get(nt_id) {
                        fallback_reduce_goto_states.extend(gotos.iter().copied());
                    }
                }
            }
        }

        // For each transition from this DWA state
        let transitions: Vec<(i32, usize)> = dwa.states[dwa_state].transitions.iter()
            .map(|(&label, &dest)| (label, dest))
            .collect();
        
        for (label, dest_dwa_state) in transitions {
            let label_usize = label as usize;
            crate::debug!(6, "    Transition: {} --{}--> {}", dwa_state, label, dest_dwa_state);
            
            // Skip TSID transitions (not grammar terminals)
            if label_usize >= terminals_count {
                // TSID transition - always keep, parser state propagates unchanged
                let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                let before_len = dest_states.len();
                dest_states.extend(parser_states.iter().copied());
                if dest_states.len() > before_len && in_queue.insert(dest_dwa_state) {
                    queue.push_back(dest_dwa_state);
                }
                continue;
            }
            
            // Skip ignored terminals (like IGNORE/whitespace) - they can appear anywhere
            if ignored_tids.contains(&label_usize) {
                let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                let before_len = dest_states.len();
                dest_states.extend(parser_states.iter().copied());
                if dest_states.len() > before_len && in_queue.insert(dest_dwa_state) {
                    queue.push_back(dest_dwa_state);
                }
                continue;
            }
            
            // Terminal transition - check if suffix parser can accept it
            if let Some(&suffix_tid) = orig_to_suffix_tid.get(&label_usize) {
                // Check if ANY parser state can accept this terminal and compute successor states
                let mut successor_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();
                
                // Track if we found any valid action (including reduces we can't fully simulate)
                // If there's a reduce with len>0, we can't compute successors but the terminal IS valid
                let mut has_reduce_with_len_gt_0 = false;
                let mut reduce_goto_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();
                
                // Process parser states iteratively to handle ε-reduction chains
                // ε-reductions (len=0) don't consume input, so after the reduce+GOTO,
                // we need to check if the new state can handle the terminal
                let mut states_to_process: VecDeque<crate::glr::table::StateID> = parser_states.iter().copied().collect();
                let mut processed_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();
                
                while let Some(parser_state_id) = states_to_process.pop_front() {
                    if !processed_states.insert(parser_state_id) {
                        continue; // Already processed
                    }
                    
                    if let Some(row) = crate::glr::table::get_row(&suffix_parser.table, parser_state_id) {
                        crate::debug!(6, "        Checking parser state {} for terminal {}", parser_state_id.0, suffix_tid.0);
                        if let Some(action) = row.get_shifts_and_reduces_for_terminal(&suffix_tid) {
                            crate::debug!(6, "          Action: {:?}", action);
                            use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(target_state) => {
                                    // Shift actually consumes the terminal
                                    successor_states.insert(target_state);
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                                    if len == 0 {
                                        // ε-reduction: compute GOTO and then re-check terminal from new state
                                        if let Some(goto) = row.get_gotos().get(&nonterminal_id) {
                                            if let Some(goto_state) = goto.state_id {
                                                // Don't add to successor_states yet - queue for processing
                                                states_to_process.push_back(goto_state);
                                            }
                                        }
                                    } else {
                                        // Reduce with len > 0: we can't simulate the stack,
                                        // but this terminal IS valid as a lookahead.
                                        // Mark that we should keep this transition.
                                        has_reduce_with_len_gt_0 = true;
                                        if let Some(states) = reduce_goto_states_by_nt.get(&nonterminal_id) {
                                            reduce_goto_states.extend(states.iter().copied());
                                        }
                                        crate::debug!(6, "          -> Reduce len>0, marking as valid");
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    // Handle shifts (actually consume terminal)
                                    if let Some(target_state) = shift {
                                        successor_states.insert(target_state);
                                    }
                                    // Handle reduces
                                    for (&reduce_len, nonterminals) in reduces.iter() {
                                        if reduce_len == 0 {
                                            // ε-reduction
                                            for nt_id in nonterminals.keys() {
                                                if let Some(goto) = row.get_gotos().get(nt_id) {
                                                    if let Some(goto_state) = goto.state_id {
                                                        states_to_process.push_back(goto_state);
                                                    }
                                                }
                                            }
                                        } else {
                                            // Reduce with len > 0: valid lookahead
                                            has_reduce_with_len_gt_0 = true;
                                            for nt_id in nonterminals.keys() {
                                                if let Some(states) = reduce_goto_states_by_nt.get(nt_id) {
                                                    reduce_goto_states.extend(states.iter().copied());
                                                }
                                            }
                                            crate::debug!(6, "          -> Split reduce len>0, marking as valid");
                                        }
                                    }
                                }
                            }
                        } else {
                            crate::debug!(6, "          No action for terminal {} from state {}", suffix_tid.0, parser_state_id.0);
                        }
                    }
                }

                if !reduce_goto_states.is_empty() {
                    successor_states.extend(reduce_goto_states.iter().copied());
                }
                
                if !successor_states.is_empty() {
                    // Keep the transition, propagate successor states
                    crate::debug!(6, "      KEEP: successor_states = {:?}", successor_states);
                    let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                    let before_len = dest_states.len();
                    dest_states.extend(successor_states);
                    if dest_states.len() > before_len && in_queue.insert(dest_dwa_state) {
                        queue.push_back(dest_dwa_state);
                    }
                } else if has_reduce_with_len_gt_0 {
                    // Keep the transition because a reduce action exists for this terminal
                    // We can't compute exact successor states without stack simulation,
                    // but we know the terminal is valid as a lookahead for the reduce.
                    // Don't propagate parser states (we don't know what they'll be after reduce).
                    crate::debug!(6, "      KEEP (reduce lookahead): terminal {} valid from parser_states {:?}", label_usize, parser_states);
                    let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                    let before_len = dest_states.len();
                    // Initialize the dest state's parser states with the suffix parser start state
                    // This is conservative - after reduces, we could end up anywhere
                    dest_states.insert(initial_parser_state_id);
                    if dest_states.len() > before_len && in_queue.insert(dest_dwa_state) {
                        queue.push_back(dest_dwa_state);
                    }
                } else if !fallback_reduce_goto_states.is_empty() {
                    // Fallback: treat any len>0 reduce in the current parser state set as a valid
                    // lookahead for this terminal, and propagate the corresponding goto states.
                    let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                    let before_len = dest_states.len();
                    dest_states.extend(fallback_reduce_goto_states.iter().copied());
                    if dest_states.len() > before_len && in_queue.insert(dest_dwa_state) {
                        queue.push_back(dest_dwa_state);
                    }
                } else {
                    // No valid actions - do not propagate; pruning is handled after fixpoint.
                    crate::debug!(6, "      PRUNE (deferred): no successor states for terminal {} from parser_states {:?}", label_usize, parser_states);
                }
            } else {
                // Terminal not found in suffix parser - this shouldn't happen
                // Keep the transition to be safe
                let dest_states = dwa_to_parser_states.entry(dest_dwa_state).or_default();
                let before_len = dest_states.len();
                dest_states.extend(parser_states.iter().copied());
                if dest_states.len() > before_len && in_queue.insert(dest_dwa_state) {
                    queue.push_back(dest_dwa_state);
                }
            }
        }
    }

    // After fixpoint, decide which transitions to prune based on final parser state sets.
    let mut transitions_to_remove: Vec<(usize, i32)> = Vec::new();
    let mut pruned_count = 0usize;
    let mut kept_count = 0usize;

    for dwa_state in 0..dwa.states.len() {
        let parser_states = expand_parser_states(
            &dwa_to_parser_states.get(&dwa_state).cloned().unwrap_or_default(),
        );
        let mut fallback_reduce_goto_states: BTreeSet<StateID> = BTreeSet::new();
        for state_id in parser_states.iter() {
            if let Some(nts) = reduce_nts_by_state.get(state_id) {
                for nt_id in nts {
                    if let Some(gotos) = reduce_goto_states_by_nt.get(nt_id) {
                        fallback_reduce_goto_states.extend(gotos.iter().copied());
                    }
                }
            }
        }
        let transitions: Vec<(i32, usize)> = dwa.states[dwa_state].transitions.iter()
            .map(|(&label, &dest)| (label, dest))
            .collect();

        for (label, _dest_dwa_state) in transitions {
            let label_usize = label as usize;

            // Skip TSID transitions (not grammar terminals)
            if label_usize >= terminals_count {
                kept_count += 1;
                continue;
            }

            // Skip ignored terminals (like IGNORE/whitespace)
            if ignored_tids.contains(&label_usize) {
                kept_count += 1;
                continue;
            }

            if let Some(&suffix_tid) = orig_to_suffix_tid.get(&label_usize) {
                let mut successor_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();
                let mut has_reduce_with_len_gt_0 = false;
                let mut reduce_goto_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();

                let mut states_to_process: VecDeque<crate::glr::table::StateID> = parser_states.iter().copied().collect();
                let mut processed_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();

                while let Some(parser_state_id) = states_to_process.pop_front() {
                    if !processed_states.insert(parser_state_id) {
                        continue;
                    }

                    if let Some(row) = crate::glr::table::get_row(&suffix_parser.table, parser_state_id) {
                        if let Some(action) = row.get_shifts_and_reduces_for_terminal(&suffix_tid) {
                            use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(target_state) => {
                                    successor_states.insert(target_state);
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                                    if len == 0 {
                                        if let Some(goto) = row.get_gotos().get(&nonterminal_id) {
                                            if let Some(goto_state) = goto.state_id {
                                                states_to_process.push_back(goto_state);
                                            }
                                        }
                                    } else {
                                        has_reduce_with_len_gt_0 = true;
                                        if let Some(states) = reduce_goto_states_by_nt.get(&nonterminal_id) {
                                            reduce_goto_states.extend(states.iter().copied());
                                        }
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    if let Some(target_state) = shift {
                                        successor_states.insert(target_state);
                                    }
                                    for (&reduce_len, nonterminals) in reduces.iter() {
                                        if reduce_len == 0 {
                                            for nt_id in nonterminals.keys() {
                                                if let Some(goto) = row.get_gotos().get(nt_id) {
                                                    if let Some(goto_state) = goto.state_id {
                                                        states_to_process.push_back(goto_state);
                                                    }
                                                }
                                            }
                                        } else {
                                            has_reduce_with_len_gt_0 = true;
                                            for nt_id in nonterminals.keys() {
                                                if let Some(states) = reduce_goto_states_by_nt.get(nt_id) {
                                                    reduce_goto_states.extend(states.iter().copied());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !reduce_goto_states.is_empty() {
                    successor_states.extend(reduce_goto_states.iter().copied());
                }

                if !successor_states.is_empty() || has_reduce_with_len_gt_0 {
                    if debug_suffix_prune && debug_label_ids.contains(&label_usize) {
                        eprintln!(
                            "DEBUG_SUFFIX_PRUNE_JSON KEEP state={} label={} parser_states={:?} succ_count={} reduce_len_gt_0={}",
                            dwa_state,
                            label_usize,
                            parser_states,
                            successor_states.len(),
                            has_reduce_with_len_gt_0
                        );
                    }
                    kept_count += 1;
                } else if !fallback_reduce_goto_states.is_empty() {
                    if debug_suffix_prune && debug_label_ids.contains(&label_usize) {
                        eprintln!(
                            "DEBUG_SUFFIX_PRUNE_JSON KEEP_FALLBACK state={} label={} parser_states={:?}",
                            dwa_state,
                            label_usize,
                            parser_states
                        );
                    }
                    kept_count += 1;
                } else {
                    if debug_suffix_prune && debug_label_ids.contains(&label_usize) {
                        eprintln!(
                            "DEBUG_SUFFIX_PRUNE_JSON PRUNE state={} label={} parser_states={:?}",
                            dwa_state,
                            label_usize,
                            parser_states
                        );
                    }
                    transitions_to_remove.push((dwa_state, label));
                    pruned_count += 1;
                }
            } else {
                kept_count += 1;
            }
        }
    }
    
    // Remove pruned transitions
    for (from_state, label) in &transitions_to_remove {
        dwa.states[*from_state].transitions.remove(label);
        dwa.states[*from_state].trans_weights.remove(label);
    }
    
    crate::debug!(4, "Suffix grammar DWA pruning: kept={}, pruned={}", kept_count, pruned_count);
    
    (kept_count, pruned_count)
}

/// Prune a terminal NWA using the suffix grammar.
///
/// This walks the NWA and removes transitions that would lead to invalid
/// suffix parser states. A transition on terminal T is pruned if, after
/// feeding T to the suffix parser, the parser has no valid states.
///
/// The NWA labels are:
/// - 0..(terminals_count-1): terminal IDs
/// - terminals_count..: TSID labels (tokenizer state IDs)
///
/// TSID transitions are not pruned (they represent tokenizer state changes,
/// not grammar terminals). Ignored terminals are also not pruned.
pub fn prune_nwa_with_suffix_grammar(
    nwa: &mut crate::dwa_i32::NWA,
    grammar: &GrammarDefinition,
    terminal_map: &bimap::BiBTreeMap<Terminal, crate::glr::table::TerminalID>,
    terminals_count: usize,
) -> (usize, usize) {
    use crate::interface::CompiledGrammar;
    use crate::glr::table::TerminalID;
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    crate::debug!(4, "Starting suffix grammar NWA pruning");

    // Collect ignored terminal IDs - these should not be pruned
    let ignored_tids: BTreeSet<usize> = grammar
        .ignore_terminal_ids
        .iter()
        .map(|tid| tid.0)
        .collect();
    crate::debug!(4, "  Ignored terminal IDs: {:?}", ignored_tids);

    // Build suffix grammar
    crate::debug!(4, "  Building suffix grammar...");
    let suffix_grammar = grammar_to_suffix_grammar(grammar);
    crate::debug!(4, "  Suffix grammar: {} productions", suffix_grammar.productions.len());

    for (i, prod) in suffix_grammar.productions.iter().enumerate() {
        crate::debug!(6, "    Suffix prod {}: {} -> {:?}", i, prod.lhs.0, prod.rhs);
    }

    crate::debug!(4, "  Compiling suffix grammar...");
    let suffix_parser = CompiledGrammar::glr_parser_from_definition(&suffix_grammar);
    crate::debug!(4, "  Suffix grammar compiled");

    // Build mapping from original terminal IDs to suffix parser terminal IDs
    let mut orig_to_suffix_tid: BTreeMap<usize, TerminalID> = BTreeMap::new();
    for (term, orig_tid) in terminal_map.iter() {
        if let Some(suffix_tid) = suffix_parser.terminal_map.get_by_left(term) {
            orig_to_suffix_tid.insert(orig_tid.0, *suffix_tid);
        }
    }
    crate::debug!(4, "    Terminal mapping: {} of {} terminals mapped", orig_to_suffix_tid.len(), terminal_map.len());

    type ParserStateSet = BTreeSet<crate::glr::table::StateID>;
    let mut nwa_to_parser_states: BTreeMap<usize, ParserStateSet> = BTreeMap::new();

    let initial_parser_state_id = suffix_parser.start_state_id;

    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut visited: BTreeSet<usize> = BTreeSet::new();
    for &start_state in &nwa.body.start_states {
        nwa_to_parser_states
            .entry(start_state)
            .or_default()
            .insert(initial_parser_state_id);
        if visited.insert(start_state) {
            queue.push_back(start_state);
        }
    }

    let mut transitions_to_remove: Vec<(usize, crate::dwa_i32::Label)> = Vec::new();
    let mut pruned_count = 0usize;
    let mut kept_count = 0usize;

    while let Some(nwa_state) = queue.pop_front() {
        let parser_states = nwa_to_parser_states
            .get(&nwa_state)
            .cloned()
            .unwrap_or_default();
        crate::debug!(6, "  BFS: NWA state {} with parser_states {:?}", nwa_state, parser_states);

        let epsilon_dests: Vec<usize> = nwa.states[nwa_state]
            .epsilons
            .iter()
            .map(|(dst, _)| *dst)
            .collect();
        for dest_state in epsilon_dests {
            let dest_states = nwa_to_parser_states.entry(dest_state).or_default();
            for &ps in &parser_states {
                dest_states.insert(ps);
            }
            if visited.insert(dest_state) {
                queue.push_back(dest_state);
            }
        }

        let transitions: Vec<(crate::dwa_i32::Label, Vec<usize>)> = nwa.states[nwa_state]
            .transitions
            .iter()
            .map(|(&label, targets)| (label, targets.iter().map(|(dst, _)| *dst).collect()))
            .collect();

        for (label, dest_states_list) in transitions {
            let label_usize = label as usize;
            crate::debug!(6, "    Transition: {} --{}--> {:?}", nwa_state, label, dest_states_list);

            if label_usize >= terminals_count {
                for dest_state in dest_states_list {
                    let dest_states = nwa_to_parser_states.entry(dest_state).or_default();
                    for &ps in &parser_states {
                        dest_states.insert(ps);
                    }
                    if visited.insert(dest_state) {
                        queue.push_back(dest_state);
                    }
                    kept_count += 1;
                }
                continue;
            }

            if ignored_tids.contains(&label_usize) {
                for dest_state in dest_states_list {
                    let dest_states = nwa_to_parser_states.entry(dest_state).or_default();
                    for &ps in &parser_states {
                        dest_states.insert(ps);
                    }
                    if visited.insert(dest_state) {
                        queue.push_back(dest_state);
                    }
                    kept_count += 1;
                }
                continue;
            }

            if let Some(&suffix_tid) = orig_to_suffix_tid.get(&label_usize) {
                let mut successor_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();
                let mut has_reduce_with_len_gt_0 = false;

                let mut states_to_process: VecDeque<crate::glr::table::StateID> = parser_states.iter().copied().collect();
                let mut processed_states: BTreeSet<crate::glr::table::StateID> = BTreeSet::new();

                while let Some(parser_state_id) = states_to_process.pop_front() {
                    if !processed_states.insert(parser_state_id) {
                        continue;
                    }

                    if let Some(row) = crate::glr::table::get_row(&suffix_parser.table, parser_state_id) {
                        crate::debug!(6, "        Checking parser state {} for terminal {}", parser_state_id.0, suffix_tid.0);
                        if let Some(action) = row.get_shifts_and_reduces_for_terminal(&suffix_tid) {
                            crate::debug!(6, "          Action: {:?}", action);
                            use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
                            match action {
                                Stage7ShiftsAndReducesLookaheadValue::Shift(target_state) => {
                                    successor_states.insert(target_state);
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                                    if len == 0 {
                                        if let Some(goto) = row.get_gotos().get(&nonterminal_id) {
                                            if let Some(goto_state) = goto.state_id {
                                                states_to_process.push_back(goto_state);
                                            }
                                        }
                                    } else {
                                        has_reduce_with_len_gt_0 = true;
                                        crate::debug!(6, "          -> Reduce len>0, marking as valid");
                                    }
                                }
                                Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                                    if let Some(target_state) = shift {
                                        successor_states.insert(target_state);
                                    }
                                    for (&reduce_len, nonterminals) in reduces.iter() {
                                        if reduce_len == 0 {
                                            for nt_id in nonterminals.keys() {
                                                if let Some(goto) = row.get_gotos().get(nt_id) {
                                                    if let Some(goto_state) = goto.state_id {
                                                        states_to_process.push_back(goto_state);
                                                    }
                                                }
                                            }
                                        } else {
                                            has_reduce_with_len_gt_0 = true;
                                            crate::debug!(6, "          -> Split reduce len>0, marking as valid");
                                        }
                                    }
                                }
                            }
                        } else {
                            crate::debug!(6, "          No action for terminal {} from state {}", suffix_tid.0, parser_state_id.0);
                        }
                    }
                }

                if !successor_states.is_empty() {
                    crate::debug!(6, "      KEEP: successor_states = {:?}", successor_states);
                    for dest_state in dest_states_list {
                        let dest_states = nwa_to_parser_states.entry(dest_state).or_default();
                        dest_states.extend(successor_states.iter().copied());
                        if visited.insert(dest_state) {
                            queue.push_back(dest_state);
                        }
                        kept_count += 1;
                    }
                } else if has_reduce_with_len_gt_0 {
                    crate::debug!(6, "      KEEP (reduce lookahead): terminal {} valid from parser_states {:?}", label_usize, parser_states);
                    for dest_state in dest_states_list {
                        if visited.insert(dest_state) {
                            queue.push_back(dest_state);
                        }
                        let dest_states = nwa_to_parser_states.entry(dest_state).or_default();
                        dest_states.insert(initial_parser_state_id);
                        kept_count += 1;
                    }
                } else {
                    crate::debug!(6, "      PRUNE: no successor states for terminal {} from parser_states {:?}", label_usize, parser_states);
                    transitions_to_remove.push((nwa_state, label));
                    pruned_count += dest_states_list.len();
                }
            } else {
                for dest_state in dest_states_list {
                    let dest_states = nwa_to_parser_states.entry(dest_state).or_default();
                    for &ps in &parser_states {
                        dest_states.insert(ps);
                    }
                    if visited.insert(dest_state) {
                        queue.push_back(dest_state);
                    }
                    kept_count += 1;
                }
            }
        }
    }

    for (from_state, label) in &transitions_to_remove {
        nwa.states[*from_state].transitions.remove(label);
    }

    crate::debug!(4, "Suffix grammar NWA pruning: kept={}, pruned={}", kept_count, pruned_count);

    (kept_count, pruned_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::GrammarDefinition;
    
    /// Test the suffix grammar construction on a recursive grammar
    /// Note: Currently skipped due to EBNF parser limitations with single-rule recursive grammars
    #[test]
    #[ignore]
    fn test_suffix_grammar_recursive() {
        // Parse a grammar with recursion: A → aAb | c
        let ebnf = r#"
            A ::= 'a' A 'b' | 'c';
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        
        // Convert to suffix grammar
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        // Print for debugging
        println!("Original productions:");
        for (i, prod) in grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        println!("\nSuffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Verify we have the expected structure
        let prod_strs: Vec<String> = suffix_grammar.productions.iter()
            .map(|p| format!("{}", p))
            .collect();
        
        // Should have terminal helpers
        assert!(prod_strs.iter().any(|s| s.contains("_T_") && s.contains("61")), // 'a' = 0x61
            "Should have terminal helper for 'a'");
        assert!(prod_strs.iter().any(|s| s.contains("_T_") && s.contains("62")), // 'b' = 0x62
            "Should have terminal helper for 'b'");
        
        // Should have suffix rules
        assert!(prod_strs.iter().any(|s| s.contains("A_suffix")),
            "Should have suffix nonterminal A_suffix");
    }
    
    /// Test suffix grammar on a simple grammar
    #[test]
    fn test_suffix_grammar_simple() {
        let ebnf = r#"
            start ::= "abc";
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Simple grammar suffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // The suffix grammar for "abc" should accept: "abc", "bc", "c", ""
        // This is modeled by creating helpers for the terminal "abc" and allowing
        // starting at any position within the expansion.
    }
    
    /// Test suffix grammar with alternation
    #[test]
    fn test_suffix_grammar_alternation() {
        let ebnf = r#"
            start ::= "a" | "b";
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Alternation grammar suffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Should have suffix rules for both alternatives
        let has_a_suffix = suffix_grammar.productions.iter()
            .any(|p| p.lhs.0.ends_with("_suffix"));
        assert!(has_a_suffix, "Should have suffix productions");
    }
    
    /// Test suffix grammar with sequence
    #[test]
    fn test_suffix_grammar_sequence() {
        let ebnf = r#"
            start ::= 'a' 'b' 'c';
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Sequence grammar suffix productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Should have suffix rules for each position in the sequence:
        // start_suffix -> T_a 'b' 'c'  (suffix starts in first element)
        // start_suffix -> T_b 'c'      (suffix starts in second element)
        // start_suffix -> T_c          (suffix starts in third element)
        let prod_strs: Vec<String> = suffix_grammar.productions.iter()
            .map(|p| format!("{}", p))
            .collect();
        
        // Count how many start_suffix productions we have
        let suffix_count = prod_strs.iter()
            .filter(|s| s.starts_with("start_suffix ->"))
            .count();
        
        // We should have 3 suffix productions for the 3-element sequence
        assert_eq!(suffix_count, 3, "Should have 3 suffix productions for 3-element sequence");
    }
    
    /// Test that suffix grammar preserves terminal helpers correctly
    #[test]
    fn test_suffix_grammar_terminal_helpers() {
        let ebnf = r#"
            start ::= 'hello';
        "#;
        let grammar = GrammarDefinition::from_ebnf(ebnf).unwrap();
        let suffix_grammar = grammar_to_suffix_grammar(&grammar);
        
        println!("Terminal helper test productions:");
        for (i, prod) in suffix_grammar.productions.iter().enumerate() {
            println!("  {}: {}", i, prod);
        }
        
        // Check that we have exactly 2 terminal helper productions:
        // T_hello -> 'hello'  (the terminal itself)
        // T_hello ->          (epsilon)
        let helper_prods: Vec<_> = suffix_grammar.productions.iter()
            .filter(|p| p.lhs.0.starts_with("_T_"))
            .collect();
        
        assert_eq!(helper_prods.len(), 2, "Should have exactly 2 terminal helper productions");
        
        // One should be non-empty (terminal) and one empty (epsilon)
        let non_empty = helper_prods.iter().filter(|p| !p.rhs.is_empty()).count();
        let empty = helper_prods.iter().filter(|p| p.rhs.is_empty()).count();
        assert_eq!(non_empty, 1, "Should have 1 non-empty terminal helper production");
        assert_eq!(empty, 1, "Should have 1 empty (epsilon) terminal helper production");
    }
}
