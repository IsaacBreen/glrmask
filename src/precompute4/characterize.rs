use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::{self, Display, Formatter};

use crate::glr::parser::GLRParser;
use crate::glr::table::{get_row, iter_rows, NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, StateID, TerminalID};

/// (initial_state, shift_state)
type InitialShift = (StateID, StateID);
/// (initial_state, reduction_len_minus_one, reduced_nonterminal)
type InitialReduce = (StateID, usize, NonTerminalID);
/// (revealed_state, remaining_len_minus_one, reduced_nonterminal)
type RevealAndRereduce = (StateID, usize, NonTerminalID);
/// (revealed_state, goto_state, shift_state)
type RevealGotoShiftEscape = (StateID, StateID, StateID);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReduceCharacterization {
    pub terminal: TerminalID,
    pub nonterminal: NonTerminalID,
    pub reveal_and_rereduces: BTreeSet<RevealAndRereduce>,
    pub reveal_goto_shift_escapes: BTreeSet<RevealGotoShiftEscape>,
}

impl ReduceCharacterization {
    fn new(terminal: TerminalID, nonterminal: NonTerminalID) -> Self {
        Self {
            terminal,
            nonterminal,
            reveal_and_rereduces: BTreeSet::new(),
            reveal_goto_shift_escapes: BTreeSet::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.reveal_and_rereduces.is_empty() && self.reveal_goto_shift_escapes.is_empty()
    }
}

impl Display for ReduceCharacterization {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "    Reduce Char for NT {}:", self.nonterminal.0)?;
        if !self.reveal_and_rereduces.is_empty() {
            writeln!(f, "      Reveal-and-rereduces:")?;
            for (revealed, len, nt) in &self.reveal_and_rereduces {
                writeln!(f, "        - revealed: {}, len: {}, reduce_nt: {}", revealed.0, len, nt.0)?;
            }
        }
        if !self.reveal_goto_shift_escapes.is_empty() {
            writeln!(f, "      Reveal-goto-shift escapes:")?;
            for (revealed, goto, shift) in &self.reveal_goto_shift_escapes {
                writeln!(f, "        - revealed: {}, goto: {}, shift: {}", revealed.0, goto.0, shift.0)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BelowBottomCharacterization {
    pub terminal: TerminalID,
    pub initial_shifts: BTreeSet<InitialShift>,
    pub initial_reduces: BTreeSet<InitialReduce>,
    pub reduce_characterizations: BTreeMap<NonTerminalID, ReduceCharacterization>,
    pub all_nts: BTreeSet<NonTerminalID>,
}

impl BelowBottomCharacterization {
    /// Check if there is any cycle in the reduce characterization graph.
    /// A cycle exists if following `reveal_and_rereduces` edges between non-terminals
    /// can lead back to a previously visited non-terminal.
    /// Returns Some(cycle_path) if a cycle exists, None otherwise.
    pub fn find_cycle(&self) -> Option<Vec<NonTerminalID>> {
        // Build adjacency list: NT -> set of NTs reachable via reveal_and_rereduces
        let mut adj: BTreeMap<NonTerminalID, BTreeSet<NonTerminalID>> = BTreeMap::new();
        for (nt, rc) in &self.reduce_characterizations {
            for &(_revealed_state, _len, target_nt) in &rc.reveal_and_rereduces {
                adj.entry(*nt).or_default().insert(target_nt);
            }
        }

        // DFS-based cycle detection using coloring:
        // White (0) = unvisited, Gray (1) = in current path, Black (2) = fully processed
        let mut color: BTreeMap<NonTerminalID, u8> = BTreeMap::new();
        let mut path: Vec<NonTerminalID> = Vec::new();
        
        fn dfs(
            node: NonTerminalID,
            adj: &BTreeMap<NonTerminalID, BTreeSet<NonTerminalID>>,
            color: &mut BTreeMap<NonTerminalID, u8>,
            path: &mut Vec<NonTerminalID>,
        ) -> Option<Vec<NonTerminalID>> {
            color.insert(node, 1); // Gray - in current path
            path.push(node);
            
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    match color.get(&neighbor).copied().unwrap_or(0) {
                        1 => {
                            // Back edge found - cycle!
                            // Find where the cycle starts in the path
                            let cycle_start = path.iter().position(|&n| n == neighbor).unwrap();
                            let mut cycle = path[cycle_start..].to_vec();
                            cycle.push(neighbor); // Close the cycle
                            return Some(cycle);
                        }
                        0 => {
                            if let Some(cycle) = dfs(neighbor, adj, color, path) {
                                return Some(cycle);
                            }
                        }
                        _ => {} // Black - already fully processed, skip
                    }
                }
            }
            
            path.pop();
            color.insert(node, 2); // Black - fully processed
            None
        }

        // Check all NTs that have outgoing edges
        for &nt in adj.keys() {
            if color.get(&nt).copied().unwrap_or(0) == 0 {
                if let Some(cycle) = dfs(nt, &adj, &mut color, &mut path) {
                    return Some(cycle);
                }
            }
        }

        None
    }
    
    /// Check if there is any cycle in the reduce characterization graph.
    pub fn has_cycle(&self) -> bool {
        self.find_cycle().is_some()
    }
}

impl Display for BelowBottomCharacterization {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Characterization for Terminal {}:", self.terminal.0)?;
        if !self.initial_shifts.is_empty() {
            writeln!(f, "  Initial Shifts:")?;
            for (initial, shift) in &self.initial_shifts {
                writeln!(f, "    - initial: {}, shift: {}", initial.0, shift.0)?;
            }
        }
        if !self.initial_reduces.is_empty() {
            writeln!(f, "  Initial Reduces:")?;
            for (initial, len, nt) in &self.initial_reduces {
                writeln!(f, "    - initial: {}, len: {}, nt: {}", initial.0, len, nt.0)?;
            }
        }
        if !self.reduce_characterizations.is_empty() {
            writeln!(f, "  Reduce Characterizations:")?;
            for rc in self.reduce_characterizations.values() {
                write!(f, "{rc}")?;
            }
        }
        Ok(())
    }
}

pub fn compute_all_characterizations(parser: &GLRParser) -> BTreeMap<TerminalID, BelowBottomCharacterization> {
    parser
        .terminal_map
        .right_values()
        .cloned()
        .map(|terminal_id| (terminal_id, compute_below_bottom_characterization(parser, terminal_id)))
        .collect()
}

pub fn compute_below_bottom_characterization(parser: &GLRParser, terminal_id: TerminalID) -> BelowBottomCharacterization {
    let all_nts: BTreeSet<_> = parser.non_terminal_map.right_values().cloned().collect();
    let (initial_shifts, initial_reduces) = collect_initial_actions(parser, terminal_id);
    let reduce_characterizations = collect_reduce_characterizations(parser, terminal_id);

    let result = BelowBottomCharacterization {
        terminal: terminal_id,
        initial_shifts,
        initial_reduces,
        reduce_characterizations,
        all_nts,
    };

    if let Some(cycle) = result.find_cycle() {
        // Get NT names for better debugging
        let cycle_names: Vec<_> = cycle.iter()
            .filter_map(|nt_id| parser.non_terminal_map.get_by_right(nt_id).map(|nt| nt.0.clone()))
            .collect();
        
        // Build detailed diagnostic information
        let mut diagnostic = String::new();
        diagnostic.push_str(&format!(
            "\n\n=== CYCLE DETECTED IN REDUCTION GRAPH ===\n\
             Terminal ID: {}\n\
             Cycle: {:?}\n\n",
            terminal_id.0, cycle_names
        ));
        
        // Show the reveal_and_rereduces edges that form the cycle
        diagnostic.push_str("=== EDGES IN THE REDUCTION GRAPH ===\n");
        for nt_id in &cycle {
            if let Some(rc) = result.reduce_characterizations.get(nt_id) {
                let nt_name = parser.non_terminal_map.get_by_right(nt_id)
                    .map(|nt| nt.0.as_str())
                    .unwrap_or("???");
                diagnostic.push_str(&format!("From NT '{}' (id={}):\n", nt_name, nt_id.0));
                for &(revealed_state, len_minus_2, target_nt) in &rc.reveal_and_rereduces {
                    let target_name = parser.non_terminal_map.get_by_right(&target_nt)
                        .map(|nt| nt.0.as_str())
                        .unwrap_or("???");
                    diagnostic.push_str(&format!(
                        "  -> '{}' (id={}) via revealed_state={}, reduction_len={}\n",
                        target_name, target_nt.0, revealed_state.0, len_minus_2 + 2
                    ));
                }
            }
        }
        
        // Dump all productions involving the cycle NTs
        diagnostic.push_str("\n=== PRODUCTIONS INVOLVING CYCLE NTs ===\n");
        let cycle_nt_ids: BTreeSet<_> = cycle.iter().cloned().collect();
        for (prod_idx, prod) in parser.productions.iter().enumerate() {
            let lhs_id = parser.non_terminal_map.get_by_left(&prod.lhs);
            let rhs_contains_cycle_nt = prod.rhs.iter().any(|sym| {
                match sym {
                    crate::glr::grammar::Symbol::NonTerminal(nt) => {
                        parser.non_terminal_map.get_by_left(nt)
                            .map(|id| cycle_nt_ids.contains(id))
                            .unwrap_or(false)
                    }
                    _ => false
                }
            });
            
            let lhs_in_cycle = lhs_id.map(|id| cycle_nt_ids.contains(id)).unwrap_or(false);
            
            if lhs_in_cycle || rhs_contains_cycle_nt {
                let lhs_marker = if lhs_in_cycle { " [CYCLE]" } else { "" };
                diagnostic.push_str(&format!("  [{}]{} {} ::=", prod_idx, lhs_marker, prod.lhs.0));
                for sym in &prod.rhs {
                    match sym {
                        crate::glr::grammar::Symbol::NonTerminal(nt) => {
                            let nt_in_cycle = parser.non_terminal_map.get_by_left(nt)
                                .map(|id| cycle_nt_ids.contains(id))
                                .unwrap_or(false);
                            let marker = if nt_in_cycle { "[*]" } else { "" };
                            diagnostic.push_str(&format!(" {}{}", nt.0, marker));
                        }
                        crate::glr::grammar::Symbol::Terminal(t) => {
                            diagnostic.push_str(&format!(" {:?}", t));
                        }
                    }
                }
                diagnostic.push_str(" ;\n");
            }
        }
        
        // Show nullable NTs in the cycle
        diagnostic.push_str("\n=== NULLABLE STATUS OF CYCLE NTs ===\n");
        let nullable_nts = crate::glr::automaton::compute_nullable_nonterminals(&parser.productions);
        for nt_id in &cycle {
            if let Some(nt) = parser.non_terminal_map.get_by_right(nt_id) {
                let is_nullable = nullable_nts.contains(nt);
                diagnostic.push_str(&format!("  '{}': {}\n", nt.0, if is_nullable { "NULLABLE" } else { "not nullable" }));
            }
        }
        
        // Check for right-recursive patterns in the cycle
        diagnostic.push_str("\n=== RIGHT-RECURSIVE PATTERN ANALYSIS ===\n");
        for nt_id in &cycle {
            if let Some(nt) = parser.non_terminal_map.get_by_right(nt_id) {
                // Find productions for this NT that end with a cycle NT
                for prod in &parser.productions {
                    if &prod.lhs != nt {
                        continue;
                    }
                    // Check if production ends with a cycle NT (considering nullable suffix)
                    for i in (0..prod.rhs.len()).rev() {
                        match &prod.rhs[i] {
                            crate::glr::grammar::Symbol::NonTerminal(rhs_nt) => {
                                let rhs_in_cycle = parser.non_terminal_map.get_by_left(rhs_nt)
                                    .map(|id| cycle_nt_ids.contains(id))
                                    .unwrap_or(false);
                                if rhs_in_cycle {
                                    // Check if suffix is nullable
                                    let suffix_nullable = prod.rhs[i+1..].iter().all(|s| {
                                        match s {
                                            crate::glr::grammar::Symbol::NonTerminal(n) => nullable_nts.contains(n),
                                            crate::glr::grammar::Symbol::Terminal(_) => false,
                                        }
                                    });
                                    if suffix_nullable {
                                        diagnostic.push_str(&format!(
                                            "  RIGHT-RECURSIVE: '{}' -> ... '{}' (suffix nullable: {})\n",
                                            nt.0, rhs_nt.0, 
                                            prod.rhs[i+1..].iter().map(|s| match s {
                                                crate::glr::grammar::Symbol::NonTerminal(n) => n.0.clone(),
                                                crate::glr::grammar::Symbol::Terminal(t) => format!("{:?}", t),
                                            }).collect::<Vec<_>>().join(" ")
                                        ));
                                    }
                                }
                                if !nullable_nts.contains(rhs_nt) {
                                    break;
                                }
                            }
                            crate::glr::grammar::Symbol::Terminal(_) => break,
                        }
                    }
                }
            }
        }
        
        diagnostic.push_str("\n=== END DIAGNOSTIC ===\n");
        
        // Per the theorem in "Even Faster Generalized LR Parsing", grammars without right 
        // and hidden left recursion have bounded consecutive reductions. Cycles here indicate
        // that right recursion elimination failed or was incomplete.
        panic!(
            "BelowBottomCharacterization for terminal {} has a cycle: {:?}. \
             This indicates unbounded reduction chains which violate the bounded-reductions guarantee. \
             Right recursion elimination should have removed these cycles.\n{}",
            terminal_id.0, cycle_names, diagnostic
        );
    }

    crate::debug!(6, "Computed Below-Bottom Characterization for terminal {}:\n{}", terminal_id.0, result);
    result
}

fn collect_initial_actions(
    parser: &GLRParser,
    terminal_id: TerminalID,
) -> (BTreeSet<InitialShift>, BTreeSet<InitialReduce>) {
    use Stage7ShiftsAndReducesLookaheadValue::*;

    let mut initial_shifts = BTreeSet::new();
    let mut initial_reduces = BTreeSet::new();

    for (&initial_state, row) in iter_rows(&parser.table) {
        if let Some(action) = row.get_shifts_and_reduces_for_terminal(&terminal_id) {
            match action {
                Shift(shift_state) => {
                    initial_shifts.insert((initial_state, shift_state));
                }
                Reduce { nonterminal_id, len, .. } => {
                    if len > 0 {
                        initial_reduces.insert((initial_state, len - 1, nonterminal_id));
                    }
                }
                Split { shift, reduces } => {
                    if let Some(shift_state) = shift {
                        initial_shifts.insert((initial_state, shift_state));
                    }
                    for (len, nts) in reduces {
                        if len > 0 {
                            for (nt_id, _) in nts {
                                initial_reduces.insert((initial_state, len - 1, nt_id));
                            }
                        }
                    }
                }
            }
        }
    }

    (initial_shifts, initial_reduces)
}

fn collect_reduce_characterizations(
    parser: &GLRParser,
    terminal_id: TerminalID,
) -> BTreeMap<NonTerminalID, ReduceCharacterization> {
    let mut result: BTreeMap<NonTerminalID, ReduceCharacterization> = BTreeMap::new();

    for (revealed_state, row) in iter_rows(&parser.table) {
        for (&nt_id, goto) in &row.gotos {
            if let Some(goto_state) = goto.state_id {
                let reduce_char = result.entry(nt_id).or_insert_with(|| ReduceCharacterization::new(terminal_id, nt_id));
                explore_from_goto(parser, terminal_id, *revealed_state, goto_state, reduce_char);
            }
        }
    }

    result.retain(|_, rc| !rc.is_empty());
    result
}

fn explore_from_goto(
    parser: &GLRParser,
    terminal_id: TerminalID,
    revealed_state: StateID,
    start_state: StateID,
    reduce_char: &mut ReduceCharacterization,
) {
    use Stage7ShiftsAndReducesLookaheadValue::*;

    let mut worklist = VecDeque::new();
    let mut visited = BTreeSet::new();

    visited.insert(start_state);
    worklist.push_back(start_state);

    while let Some(current_state) = worklist.pop_front() {
        let Some(row) = get_row(&parser.table, current_state) else { continue };
        let Some(action) = row.get_shifts_and_reduces_for_terminal(&terminal_id) else { continue };

        match action {
            Shift(shift_state) => {
                reduce_char.reveal_goto_shift_escapes.insert((revealed_state, current_state, shift_state));
            }
            Reduce { nonterminal_id: reduce_nt, len, .. } => {
                handle_reduce(parser, revealed_state, len, reduce_nt, &mut visited, &mut worklist, reduce_char);
            }
            Split { shift, reduces } => {
                if let Some(shift_state) = shift {
                    reduce_char.reveal_goto_shift_escapes.insert((revealed_state, current_state, shift_state));
                }
                for (len, nts) in reduces {
                    for (reduce_nt, _) in nts {
                        handle_reduce(parser, revealed_state, len, reduce_nt, &mut visited, &mut worklist, reduce_char);
                    }
                }
            }
        }
    }
}

fn handle_reduce(
    parser: &GLRParser,
    revealed_state: StateID,
    len: usize,
    reduce_nt: NonTerminalID,
    visited: &mut BTreeSet<StateID>,
    worklist: &mut VecDeque<StateID>,
    reduce_char: &mut ReduceCharacterization,
) {
    if len == 1 {
        if let Some(next_goto_state) = get_row(&parser.table, revealed_state)
            .and_then(|row| row.gotos.get(&reduce_nt))
            .and_then(|goto| goto.state_id)
        {
            if visited.insert(next_goto_state) {
                worklist.push_back(next_goto_state);
            }
        }
    } else if len > 1 {
        reduce_char.reveal_and_rereduces.insert((revealed_state, len - 2, reduce_nt));
    }
}
