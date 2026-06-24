use super::*;

use crate::ds::bitset::BitSet;

const DISABLE_UNIT_REDUCTION_INLINING_ENV: &str = "GLRMASK_DISABLE_UNIT_REDUCTION_INLINING";
const GLR_TABLE_CONSTRUCTION_ENV: &str = "GLRMASK_GLR_TABLE_CONSTRUCTION";
const UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_MAX_PRE_MERGE_STATES";
// Unit reduction inlining is worthwhile only while the table is modest.
// Above this point it expands table optimization work and slows all non-terminal
// phases; terminal-DWA effects are intentionally handled by its separate pass.
const DEFAULT_UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES: u32 = 10_000;

fn glr_table_construction(default: GlrTableConstruction) -> GlrTableConstruction {
    match std::env::var(GLR_TABLE_CONSTRUCTION_ENV) {
        Ok(value) if value.trim().eq_ignore_ascii_case("legacy")
            || value.trim().eq_ignore_ascii_case("legacy-row-bisim")
            || value.trim().eq_ignore_ascii_case("row-bisim") =>
        {
            GlrTableConstruction::LegacyRowBisim
        }
        Ok(value) if value.trim().eq_ignore_ascii_case("lalr") => {
            GlrTableConstruction::Lalr
        }
        Ok(value) if value.trim().eq_ignore_ascii_case("core-lac") => {
            GlrTableConstruction::ExperimentalCoreMerged
        }
        _ => default,
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn unit_reduction_inlining_enabled() -> bool {
    !env_flag_enabled(DISABLE_UNIT_REDUCTION_INLINING_ENV)
}

fn unit_reduction_inlining_max_pre_merge_states() -> Option<u32> {
    match std::env::var(UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES_ENV) {
        Ok(value) => match value.trim().parse::<u32>() {
            Ok(0) => None,
            Ok(parsed) => Some(parsed),
            Err(_) => Some(DEFAULT_UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES),
        },
        Err(_) => Some(DEFAULT_UNIT_REDUCTION_INLINING_MAX_PRE_MERGE_STATES),
    }
}

pub(super) fn build_table(grammar: &AnalyzedGrammar) -> GLRTable {
    build_table_with_default_construction(grammar, GlrTableConstruction::ExperimentalCoreMerged)
}

pub(super) fn build_table_with_default_construction(
    grammar: &AnalyzedGrammar,
    default_construction: GlrTableConstruction,
) -> GLRTable {
    let t1 = std::time::Instant::now();
    let construction = glr_table_construction(default_construction);
    let mut lr1_ms = 0.0;
    let mut table = match construction {
        GlrTableConstruction::LegacyRowBisim => {
            let t0 = std::time::Instant::now();
            let (item_sets, transitions) = build_lr1_item_sets(grammar);
            lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;
            build_legacy_row_bisim_table(grammar, &item_sets, &transitions)
        }
        GlrTableConstruction::Lalr => build_lalr_table(grammar),
        GlrTableConstruction::ExperimentalCoreMerged => {
            let t0 = std::time::Instant::now();
            let (item_sets, transitions) = build_lr1_item_sets(grammar);
            lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;
            build_experimental_core_merged_table(grammar, &item_sets, &transitions)
                .unwrap_or_else(|| build_legacy_row_bisim_table(grammar, &item_sets, &transitions))
        }
    };
    let construction = table.construction;
    let construction_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let pre_merge_states = table.num_states;
    let t2 = std::time::Instant::now();
    let merge1_started_at = std::time::Instant::now();
    table.merge_identical_rows();
    let merge_identical1_ms = merge1_started_at.elapsed().as_secs_f64() * 1000.0;
    // From here on, `action` is allowed to become an optimized execution table
    // containing guarded stack effects. Capture the exact recognizer/admission
    // row support before that lowering so runtime `may_advance` stays a pure
    // row-presence query.
    table.rebuild_advance_rows_from_actions();
    let unit_collapse_skip_reason = if construction != GlrTableConstruction::LegacyRowBisim {
        "construction"
    } else if !unit_reduction_inlining_enabled() {
        "disabled"
    } else if unit_reduction_inlining_max_pre_merge_states()
        .is_some_and(|max_pre_merge_states| pre_merge_states > max_pre_merge_states)
    {
        "pre_merge_states"
    } else {
        "none"
    };
    let unit_collapse_enabled = unit_collapse_skip_reason == "none";
    let collapse_started_at = std::time::Instant::now();
    let unit_collapse_report = if unit_collapse_enabled {
        Some(table.collapse_sr_unit_reductions_with_compatible_gotos())
    } else {
        None
    };
    if unit_collapse_enabled {
        // Unit-collapse may append synthetic merged states. Preserve the
        // captured admission semantics for existing rows while backfilling the
        // new synthetic rows from their current action support.
        table.extend_advance_rows_from_actions();
        if !table.advance.is_empty() {
            debug_assert_eq!(table.advance.len(), table.num_states as usize);
        }
    }
    let unit_collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;
    let prune_started_at = std::time::Instant::now();
    table.prune_unreachable_states();
    let prune_ms = prune_started_at.elapsed().as_secs_f64() * 1000.0;
    let merge2_started_at = std::time::Instant::now();
    table.merge_identical_rows();
    let merge_identical2_ms = merge2_started_at.elapsed().as_secs_f64() * 1000.0;
    let merge_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = std::time::Instant::now();
    // The downstream parser and template builders already merge equivalent
    // artifacts. Running the recognizer-only equivalence pass here costs more
    // on large schemas than it saves in later phases.
    if construction == GlrTableConstruction::LegacyRowBisim {
        table.canonicalize_stack_shift_predecessors();
        table.quotient_recognizer_stack_suffixes();
    }
    let recog_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let _ = (
        lr1_ms,
        construction_ms,
        pre_merge_states,
        merge_ms,
        merge_identical1_ms,
        unit_collapse_ms,
        prune_ms,
        merge_identical2_ms,
        recog_ms,
    );

    if construction == GlrTableConstruction::LegacyRowBisim && default_action_rows_enabled() {
        table.compress_default_action_rows();
    }

    table.rebuild_guarded_shift_index();

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
    {
        eprintln!(
            "[glrmask/profile][glr_table] construction={:?} lr1_item_sets_ms={:.3} construction_ms={:.3} pre_merge_states={} post_merge_states={} unit_collapse={} unit_collapse_aborted={} unit_collapse_reason={} unit_collapse_skip_reason={} merge_ms={:.3} merge_identical1_ms={:.3} unit_collapse_ms={:.3} prune_ms={:.3} merge_identical2_ms={:.3} stack_shift_canon_ms={:.3}",
            construction,
            lr1_ms,
            construction_ms,
            pre_merge_states,
            table.num_states,
            unit_collapse_enabled,
            unit_collapse_report
                .as_ref()
                .is_some_and(|report| report.aborted),
            unit_collapse_report
                .as_ref()
                .and_then(|report| report.reason)
                .unwrap_or("none"),
            unit_collapse_skip_reason,
            merge_ms,
            merge_identical1_ms,
            unit_collapse_ms,
            prune_ms,
            merge_identical2_ms,
            recog_ms,
        );
    }

    table
}

fn replace_shifts_enabled() -> bool {
    true
}

fn replace_gotos_enabled() -> bool {
    true
}

std::thread_local! {
    pub(crate) static LOCAL_FORWARD_REPLACE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn local_forward_replace_enabled() -> bool {
    LOCAL_FORWARD_REPLACE_OVERRIDE.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        false
    })
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Item {
    pub(super) rule: u32,
    pub(super) dot: u32,
    pub(super) stack_depth: u32,
}

impl Item {
    pub(super) fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {
        Self { rule, dot, stack_depth }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct PendingAction {
    pub(super) shift: Option<(u32, bool)>,
    pub(super) reduces: Vec<(NonterminalID, u32)>,
    pub(super) accept: bool,
}

impl PendingAction {
    pub(super) fn push_shift(&mut self, target: u32, is_replace: bool) {
        match self.shift {
            Some((existing, _)) => debug_assert_eq!(existing, target),
            None => self.shift = Some((target, is_replace)),
        }
    }

    pub(super) fn push_reduce(&mut self, nt: NonterminalID, len: u32) {
        self.reduces.push((nt, len));
    }

    pub(super) fn push_accept(&mut self) {
        self.accept = true;
    }

    pub(super) fn maybe_finish(mut self) -> Option<Action> {
        self.reduces.sort_unstable();
        self.reduces.dedup();
        match (self.shift, self.reduces.len(), self.accept) {
            (None, 0, false) => None,
            (Some((target, replace)), 0, false) => Some(Action::Shift(target, replace)),
            (None, 1, false) => Some(Action::Reduce(self.reduces[0].0, self.reduces[0].1)),
            (None, 0, true) => Some(Action::Accept),
            (shift, _, accept) => Some(Action::Split {
                shift,
                reduces: self.reduces,
                accept,
            }),
        }
    }

    pub(super) fn finish(self) -> Action {
        self.maybe_finish()
            .expect("PendingAction::finish called on an empty action")
    }
}

fn initialize_pending_and_goto(
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> (
    Vec<BTreeMap<TerminalID, PendingAction>>,
    Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    FxHashSet<(u32, TerminalID)>,
) {
    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(transitions.len())
        .collect::<Vec<_>>();
    let mut goto: Vec<FxHashMap<NonterminalID, (u32, bool)>> = (0..transitions.len()).map(|_| FxHashMap::default()).collect();
    let mut forwarded_shifts = FxHashSet::default();

    for (state_id, by_symbol) in transitions.iter().enumerate() {
        for (symbol, &(target, is_replace, is_forwarded)) in by_symbol {
            match symbol {
                Symbol::Terminal(terminal) => {
                    pending[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push_shift(target, is_replace);
                    if is_forwarded {
                        forwarded_shifts.insert((state_id as u32, *terminal));
                    }
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[state_id].insert(*nonterminal, (target, is_replace));
                }
            }
        }
    }

    (pending, goto, forwarded_shifts)
}

fn finish_table(
    grammar: &AnalyzedGrammar,
    pending: Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    forwarded_shifts: FxHashSet<(u32, TerminalID)>,
    construction: GlrTableConstruction,
    admission_policy: AdmissionPolicy,
) -> GLRTable {
    let action: Vec<ActionRow> = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .map(|(terminal, pending)| (terminal, pending.finish()))
                .collect()
        })
        .collect();
    let goto: Vec<GotoRow> = goto.into_iter().map(IntoIterator::into_iter).map(Iterator::collect).collect();
    let num_states = action.len() as u32;

    GLRTable {
        action,
        goto,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
        nonterminal_display_names: grammar.nonterminal_display_names.clone(),
        construction,
        admission_policy,
        advance: Vec::new(),
        forwarded_shifts,
        guarded_shift_index: Vec::new(),
    }
}

#[derive(Debug, Clone)]
struct Lr0State {
    kernel: BTreeSet<Item>,
    closure: Vec<Item>,
}

fn item_next_symbol<'a>(item: &Item, rules: &'a [Rule]) -> Option<&'a Symbol> {
    rules[item.rule as usize].rhs.get(item.dot as usize)
}

fn lr0_closure(grammar: &AnalyzedGrammar, kernel: &BTreeSet<Item>) -> Vec<Item> {
    let mut result = kernel.clone();
    let mut queue: VecDeque<Item> = kernel.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        let Some(Symbol::Nonterminal(nonterminal)) = item_next_symbol(&item, &grammar.rules) else {
            continue;
        };
        for &rule_id in &grammar.rules_by_lhs[*nonterminal as usize] {
            let stack_depth = grammar.rules[rule_id as usize].rhs.len() as u32;
            let next = Item::new(rule_id, 0, stack_depth);
            if result.insert(next) {
                queue.push_back(next);
            }
        }
    }

    result.into_iter().collect()
}

fn build_lr0_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<Lr0State>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let mut start_kernel = BTreeSet::new();
    start_kernel.insert(Item::new(0, 0, grammar.rules[0].rhs.len() as u32));
    let start_closure = lr0_closure(grammar, &start_kernel);

    let mut states = vec![Lr0State {
        kernel: start_kernel.clone(),
        closure: start_closure,
    }];
    let mut transitions = vec![BTreeMap::new()];
    let mut state_by_kernel: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    state_by_kernel.insert(start_kernel.iter().copied().collect(), 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(source) = queue.pop_front() {
        let mut kernels: BTreeMap<Symbol, BTreeSet<Item>> = BTreeMap::new();
        for item in &states[source as usize].closure {
            let Some(symbol) = item_next_symbol(item, &grammar.rules) else {
                continue;
            };
            kernels
                .entry(symbol.clone())
                .or_default()
                .insert(Item::new(item.rule, item.dot + 1, item.stack_depth));
        }

        for (symbol, kernel) in kernels {
            let has_dot_1 = kernel.iter().any(|item| item.dot == 1);
            let is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };

            let adjusted_kernel: BTreeSet<Item> = if is_replace {
                kernel
                    .iter()
                    .map(|item| Item::new(item.rule, item.dot, item.stack_depth.saturating_sub(1)))
                    .collect()
            } else {
                kernel
            };
            if adjusted_kernel.is_empty() {
                continue;
            }

            let key = adjusted_kernel.iter().copied().collect::<Vec<_>>();
            let target = if let Some(&target) = state_by_kernel.get(&key) {
                target
            } else {
                let target = states.len() as u32;
                let closure = lr0_closure(grammar, &adjusted_kernel);
                state_by_kernel.insert(key, target);
                states.push(Lr0State {
                    kernel: adjusted_kernel,
                    closure,
                });
                transitions.push(BTreeMap::new());
                queue.push_back(target);
                target
            };

            transitions[source as usize].insert(symbol, (target, is_replace, false));
        }
    }

    (states, transitions)
}

fn sequence_nullable(symbols: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> bool {
    symbols.iter().all(|symbol| match symbol {
        Symbol::Terminal(_) => false,
        Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
    })
}

fn lalr_global_node_id(offsets: &[usize], state: usize, item: usize) -> usize {
    offsets[state] + item
}

fn compute_lalr_item_lookaheads(
    grammar: &AnalyzedGrammar,
    states: &[Lr0State],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> Vec<Vec<BitSet>> {
    let lookahead_len = grammar.num_terminals as usize + 1;
    let first = first_bitsets(grammar);

    let mut offsets = Vec::with_capacity(states.len() + 1);
    offsets.push(0usize);
    for state in states {
        offsets.push(offsets.last().copied().unwrap() + state.closure.len());
    }
    let total_nodes = *offsets.last().unwrap();

    let mut item_index_by_state = Vec::with_capacity(states.len());
    for state in states {
        let mut index = BTreeMap::new();
        for (item_index, item) in state.closure.iter().enumerate() {
            index.insert(*item, item_index);
        }
        item_index_by_state.push(index);
    }

    let mut lookaheads = vec![BitSet::new(lookahead_len); total_nodes];
    let mut edges = vec![Vec::<usize>::new(); total_nodes];
    let mut worklist = VecDeque::<usize>::new();
    let mut queued = vec![false; total_nodes];

    let start = Item::new(0, 0, grammar.rules[0].rhs.len() as u32);
    if let Some(&start_index) = item_index_by_state[0].get(&start) {
        let start_node = lalr_global_node_id(&offsets, 0, start_index);
        lookaheads[start_node].set(lookahead_bit(EOF, grammar.num_terminals));
        worklist.push_back(start_node);
        queued[start_node] = true;
    }

    let empty_lookahead = BitSet::new(lookahead_len);

    for (state_id, state) in states.iter().enumerate() {
        for (item_index, item) in state.closure.iter().enumerate() {
            let source_node = lalr_global_node_id(&offsets, state_id, item_index);

            if let Some(symbol) = item_next_symbol(item, &grammar.rules) {
                let Some(&(target_state, is_replace, _)) = transitions[state_id].get(symbol) else {
                    continue;
                };
                let mut advanced = Item::new(item.rule, item.dot + 1, item.stack_depth);
                if is_replace {
                    advanced.stack_depth = advanced.stack_depth.saturating_sub(1);
                }
                if let Some(&target_item_index) = item_index_by_state[target_state as usize].get(&advanced) {
                    edges[source_node].push(lalr_global_node_id(
                        &offsets,
                        target_state as usize,
                        target_item_index,
                    ));
                }
            }

            let Some(Symbol::Nonterminal(nonterminal)) = item_next_symbol(item, &grammar.rules) else {
                continue;
            };
            let rhs = &grammar.rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];
            let spontaneous = first_of_sequence_bits(
                beta,
                &empty_lookahead,
                &first,
                &grammar.nullable,
                grammar.num_terminals,
            );
            let propagates = sequence_nullable(beta, &grammar.nullable);

            for &rule_id in &grammar.rules_by_lhs[*nonterminal as usize] {
                let closure_item = Item::new(rule_id, 0, grammar.rules[rule_id as usize].rhs.len() as u32);
                let Some(&target_item_index) = item_index_by_state[state_id].get(&closure_item) else {
                    continue;
                };
                let target_node = lalr_global_node_id(&offsets, state_id, target_item_index);
                if !spontaneous.is_empty() {
                    let delta = spontaneous.difference(&lookaheads[target_node]);
                    if !delta.is_empty() {
                        lookaheads[target_node].union_with(&delta);
                        if !queued[target_node] {
                            queued[target_node] = true;
                            worklist.push_back(target_node);
                        }
                    }
                }
                if propagates {
                    edges[source_node].push(target_node);
                }
            }
        }
    }

    while let Some(source_node) = worklist.pop_front() {
        queued[source_node] = false;
        let source_lookahead = lookaheads[source_node].clone();
        for &target_node in &edges[source_node] {
            let delta = source_lookahead.difference(&lookaheads[target_node]);
            if delta.is_empty() {
                continue;
            }
            lookaheads[target_node].union_with(&delta);
            if !queued[target_node] {
                queued[target_node] = true;
                worklist.push_back(target_node);
            }
        }
    }

    states
        .iter()
        .enumerate()
        .map(|(state_id, state)| {
            (0..state.closure.len())
                .map(|item_index| lookaheads[lalr_global_node_id(&offsets, state_id, item_index)].clone())
                .collect()
        })
        .collect()
}

fn build_lalr_table(grammar: &AnalyzedGrammar) -> GLRTable {
    let (states, transitions) = build_lr0_item_sets(grammar);
    let lookaheads = compute_lalr_item_lookaheads(grammar, &states, &transitions);
    let (mut pending, goto, forwarded_shifts) = initialize_pending_and_goto(&transitions);

    for (state_id, state) in states.iter().enumerate() {
        for (item_index, item) in state.closure.iter().enumerate() {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            for bit in lookaheads[state_id][item_index].iter_ones() {
                let lookahead = bit_lookahead(bit, grammar.num_terminals);
                if item.rule == 0 {
                    pending[state_id].entry(lookahead).or_default().push_accept();
                } else {
                    pending[state_id]
                        .entry(lookahead)
                        .or_default()
                        .push_reduce(rule.lhs, item.stack_depth);
                }
            }
        }
    }

    finish_table(
        grammar,
        pending,
        goto,
        forwarded_shifts,
        GlrTableConstruction::Lalr,
        AdmissionPolicy::ExactSimulation,
    )
}

// LR(1) item set construction.

fn lookahead_bit(term: TerminalID, num_terminals: u32) -> usize {
    if term == EOF {
        num_terminals as usize
    } else {
        term as usize
    }
}

fn bit_lookahead(bit: usize, num_terminals: u32) -> TerminalID {
    if bit == num_terminals as usize {
        EOF
    } else {
        bit as TerminalID
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1ItemCore {
    rule: u32,
    dot: u32,
    stack_depth: u32,
    transferred: bool,
}

impl LR1ItemCore {
    fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {
        Self {
            rule,
            dot,
            stack_depth,
            transferred: false,
        }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

type LR1ItemSet = BTreeMap<LR1ItemCore, BitSet>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1Item {
    rule: u32,
    dot: u32,
    lookahead: TerminalID,
    stack_depth: u32,
    /// When true, this item was "transferred" from a parent state to provide
    /// goto information.  Transferred items do NOT participate in closure,
    /// shift actions, or reduce actions — only goto transitions.
    transferred: bool,
}

impl LR1Item {
    fn new(rule: u32, dot: u32, lookahead: TerminalID, stack_depth: u32) -> Self {
        Self { rule, dot, lookahead, stack_depth, transferred: false }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

/// Compute FIRST set for a sequence of symbols followed by a lookahead terminal.
fn first_of_sequence_bits(
    symbols: &[Symbol],
    lookaheads: &BitSet,
    first: &[BitSet],
    nullable: &BTreeSet<NonterminalID>,
    num_terminals: u32,
) -> BitSet {
    let mut result = BitSet::new(num_terminals as usize + 1);
    let mut all_nullable = true;
    for sym in symbols {
        match sym {
            Symbol::Terminal(t) => {
                result.set(*t as usize);
                all_nullable = false;
                break;
            }
            Symbol::Nonterminal(nt) => {
                result.union_with(&first[*nt as usize]);
                if !nullable.contains(nt) {
                    all_nullable = false;
                    break;
                }
            }
        }
    }
    if all_nullable {
        result.union_with(lookaheads);
    }
    result
}

fn first_bitsets(grammar: &AnalyzedGrammar) -> Vec<BitSet> {
    grammar.first.clone()
}

fn union_lookaheads(item_set: &mut LR1ItemSet, core: LR1ItemCore, lookaheads: &BitSet) -> BitSet {
    let entry = item_set
        .entry(core)
        .or_insert_with(|| BitSet::new(lookaheads.len()));
    let delta = lookaheads.difference(entry);
    if !delta.is_empty() {
        entry.union_with(&delta);
    }
    delta
}

fn lr1_closure(
    items: &LR1ItemSet,
    grammar: &AnalyzedGrammar,
    first: &[BitSet],
) -> LR1ItemSet {
    let rules = &grammar.rules;
    let mut result = items.clone();
    let mut queue: VecDeque<(LR1ItemCore, BitSet)> = items
        .iter()
        .map(|(core, lookaheads)| (*core, lookaheads.clone()))
        .collect();

    while let Some((item, lookahead_delta)) = queue.pop_front() {
        // Transferred items do not participate in closure.
        if item.transferred {
            continue;
        }
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let rhs = &rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];

            let lookaheads = first_of_sequence_bits(
                beta,
                &lookahead_delta,
                &first,
                &grammar.nullable,
                grammar.num_terminals,
            );

            for &i in &grammar.rules_by_lhs[*nt as usize] {
                let sd = grammar.rules[i as usize].rhs.len() as u32;
                let new_item = LR1ItemCore::new(i, 0, sd);
                let delta = union_lookaheads(&mut result, new_item, &lookaheads);
                if !delta.is_empty() {
                    queue.push_back((new_item, delta));
                }
            }
        }
    }
    result
}

fn item_set_key(items: &LR1ItemSet) -> Vec<(LR1ItemCore, BitSet)> {
    items.iter().map(|(core, lookaheads)| (*core, lookaheads.clone())).collect()
}

/// Compute transferred items for the local-forward replace optimisation.
///
/// For each dot-1 item `[A → X . rest, la]` in `kernel`, find "foo items" in
/// `source_items` — items whose symbol-after-dot is `Nonterminal(A)`.  These
/// are the items that generate gotos for `A` in the source state.  Transferring
/// them into the target kernel provides the same gotos at the target so the
/// transition can be marked replace.
///
/// Returns `None` if:
/// - any dot-1 item belongs to `rule == 0` (augmented start), or
/// - any dot-1 item is NOT completed (i.e., not a single-symbol production), or
/// - any dot-1 item's LHS nonterminal has NO foo items in the source.
///
/// Recursively follows single-symbol production chains: when a foo item is
/// itself a single-symbol production at dot=0, its LHS nonterminal also needs
/// foo items in the source.
///
/// Returns `Some(transferred)` with the set of transferred items otherwise.

/// Eagerly advance transferred items past completed nonterminals in the
fn compute_transfer_items(
    kernel: &BTreeSet<LR1Item>,
    source_items: &BTreeSet<LR1Item>,
    rules: &[Rule],
) -> Option<Vec<LR1Item>> {
    let mut transferred = Vec::new();

    // Collect the LHS nonterminals of all dot-1 items.
    // Only completed dot-1 items (single-symbol productions) are eligible.
    let mut needed_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    for item in kernel.iter().filter(|it| it.dot == 1) {
        if item.rule == 0 {
            return None;
        }
        let rule = &rules[item.rule as usize];
        if (item.dot as usize) != rule.rhs.len() {
            return None;
        }
        needed_nts.insert(rule.lhs);
    }

    if needed_nts.is_empty() {
        return None;
    }

    // Iteratively find foo items, following the nonterminal chain.
    // The chain extends through ALL foo items' LHS nonterminals so that
    // gotos for every nonterminal in the reduce chain are available.
    let mut all_needed = needed_nts.clone();
    let mut found_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    loop {
        let mut new_needed: BTreeSet<NonterminalID> = BTreeSet::new();
        for item in source_items {
            if item.transferred {
                continue;
            }
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                if all_needed.contains(nt) && !found_nts.contains(nt) {
                    transferred.push(LR1Item {
                        transferred: true,
                        ..*item
                    });
                    found_nts.insert(*nt);
                    // Add this foo item's LHS to the chain so that gotos
                    // for it are also generated in the target state.
                    if item.dot == 0 {
                        let foo_rule = &rules[item.rule as usize];
                        let chain_nt = foo_rule.lhs;
                        if !all_needed.contains(&chain_nt) {
                            new_needed.insert(chain_nt);
                        }
                    }
                }
            }
        }
        if new_needed.is_empty() {
            break;
        }
        all_needed.extend(&new_needed);
    }

    // ALL initially needed nonterminals must have at least one foo item.
    // Chain-extended nonterminals may not have foo items (e.g. the
    // augmented start nonterminal) which is fine.
    if !needed_nts.is_subset(&found_nts) {
        return None;
    }

    if transferred.is_empty() {
        return None;
    }

    Some(transferred)
}

fn build_lr1_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<LR1ItemSet>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let rules = &grammar.rules;
    let lookahead_len = grammar.num_terminals as usize + 1;
    let first = first_bitsets(grammar);

    let initial = {
        let mut s = LR1ItemSet::new();
        let sd = rules[0].rhs.len() as u32;
        let mut lookaheads = BitSet::new(lookahead_len);
        lookaheads.set(lookahead_bit(EOF, grammar.num_terminals));
        s.insert(LR1ItemCore::new(0, 0, sd), lookaheads);
        lr1_closure(&s, grammar, &first)
    };

    let mut item_sets = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<(LR1ItemCore, BitSet)>, u32> = FxHashMap::default();
    set_to_id.insert(item_set_key(&initial), 0);

    // Grouped lookaheads keep ordinary LR(1) correctness, but the older
    // local-forward transfer logic is written for scalar LR1 items.
    // Keep replace flags conservative on this path.
    let transfer_safe = false;

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let source_items = item_sets[state_id as usize].clone();

        // Build all goto kernels in a single pass over items.
        let mut kernels: BTreeMap<Symbol, LR1ItemSet> = BTreeMap::new();
        for (item, lookaheads) in &source_items {
            // Transferred items only advance through nonterminal gotos.
            if item.transferred {
                if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                    let advanced = LR1ItemCore {
                        rule: item.rule,
                        dot: item.dot + 1,
                        stack_depth: item.stack_depth,
                        transferred: false,
                    };
                    union_lookaheads(
                        kernels.entry(Symbol::Nonterminal(*nt)).or_default(),
                        advanced,
                        lookaheads,
                    );
                }
                continue;
            }
            if let Some(sym) = item.next_symbol(rules) {
                let advanced = LR1ItemCore::new(item.rule, item.dot + 1, item.stack_depth);
                union_lookaheads(kernels.entry(sym.clone()).or_default(), advanced, lookaheads);
            }
        }

        for (symbol, mut kernel) in kernels {
            let base_kernel = kernel.clone();
            // Check replace condition: is_replace iff no item in the kernel
            // has dot at position 1 (i.e., all items came from items that
            // already had dot > 0).
            let has_dot_1 = kernel.keys().any(|item| item.dot == 1);
            let mut is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };
            let used_transfer = false;

            // Local-forward transfer: when has_dot_1 but transfer is enabled,
            // try to transfer foo items from the source state into the kernel
            // so the transition can still be marked as replace.
            let _ = transfer_safe;

            let kernel_has_transferred = used_transfer && kernel.keys().any(|item| item.transferred);

            // If replace, decrement stack_depth for non-transferred kernel
            // items — the replace absorbs one stack level for items that
            // went through the shift. Transferred items didn't go through
            // the shift; they were copied from the source state and await
            // nonterminal gotos, so their sd is already correct.
            let adjusted_kernel: LR1ItemSet = if is_replace {
                kernel
                    .iter()
                    .map(|(item, lookaheads)| {
                        let adjusted = if item.transferred {
                            *item
                        } else {
                            LR1ItemCore {
                                rule: item.rule,
                                dot: item.dot,
                                stack_depth: item.stack_depth.saturating_sub(1),
                                ..*item
                            }
                        };
                        (adjusted, lookaheads.clone())
                    })
                    .collect()
            } else {
                kernel
            };

            let mut target_items = lr1_closure(&adjusted_kernel, grammar, &first);
            if used_transfer && matches!(symbol, Symbol::Terminal(_)) {
                let base_target_items = lr1_closure(&base_kernel, grammar, &first);
                let base_has_zero_pop_completed = base_target_items.iter().any(|(item, _)| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                let transferred_has_zero_pop_completed = target_items.iter().any(|(item, _)| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                if transferred_has_zero_pop_completed && !base_has_zero_pop_completed {
                    kernel = base_kernel;
                    is_replace = false;
                    target_items = lr1_closure(&kernel, grammar, &first);
                }
            }

            // Track whether this replace was created by the transfer mechanism.
            let is_forwarded = is_replace && kernel_has_transferred;

            if target_items.is_empty() {
                continue;
            }

            let key = item_set_key(&target_items);
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol, (target_id, is_replace, is_forwarded));
        }
    }

    (item_sets, transitions)
}

fn current_unique_reduce_len(
    pending: &[BTreeMap<TerminalID, PendingAction>],
    state: u32,
    lookahead: TerminalID,
    nonterminal: NonterminalID,
) -> Option<u32> {
    let pending_action = pending.get(state as usize)?.get(&lookahead)?;
    let mut unique_len = None;
    for &(reduce_nt, reduce_len) in &pending_action.reduces {
        if reduce_nt != nonterminal {
            continue;
        }
        match unique_len {
            None => unique_len = Some(reduce_len),
            Some(existing) if existing == reduce_len => {}
            Some(_) => return None,
        }
    }
    unique_len
}

/// Check if any nonterminal can transitively reach itself through the
/// grammar's production rules. Returns true if any recursion exists.
fn grammar_has_recursion(rules: &[Rule]) -> bool {
    let max_nt = rules.iter().map(|r| r.lhs).max().unwrap_or(0) as usize + 1;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); max_nt];
    for rule in rules {
        let lhs = rule.lhs as usize;
        for sym in &rule.rhs {
            if let Symbol::Nonterminal(nt) = sym {
                adj[lhs].push(*nt as usize);
            }
        }
    }

    // 0 = unvisited, 1 = visiting, 2 = done.
    let mut color = vec![0u8; max_nt];
    fn dfs(node: usize, adj: &[Vec<usize>], color: &mut [u8]) -> bool {
        color[node] = 1;
        for &next in &adj[node] {
            match color[next] {
                1 => return true,
                0 => {
                    if dfs(next, adj, color) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        color[node] = 2;
        false
    }

    for nt in 0..max_nt {
        if color[nt] == 0 && dfs(nt, &adj, &mut color) {
            return true;
        }
    }
    false
}

fn apply_local_forward_replace(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
    rules: &[Rule],
) {
    // Build incoming-transition count per state so we can check whether
    // a reduce-state is uniquely reachable.  If multiple transitions
    // lead into the reduce state, rewriting its reduce length would
    // corrupt other paths (e.g. recursive grammars).
    let mut in_count = vec![0u32; item_sets.len()];
    for t in transitions {
        for &(target, _, _) in t.values() {
            in_count[target as usize] += 1;
        }
    }

    loop {
        let mut changed = false;

        for source in 0..item_sets.len() {
            for (symbol, &(target, _, _)) in &transitions[source] {
                let currently_non_replace = match symbol {
                    Symbol::Terminal(terminal) => pending[source]
                        .get(terminal)
                        .and_then(|pending_action| pending_action.shift)
                        .is_some_and(|(_, is_replace)| !is_replace),
                    Symbol::Nonterminal(nonterminal) => goto[source]
                        .get(nonterminal)
                        .is_some_and(|&(_, is_replace)| !is_replace),
                };
                if !currently_non_replace {
                    continue;
                }

                let dot1_items: Vec<_> = item_sets[target as usize]
                    .iter()
                    .copied()
                    .filter(|item| item.dot == 1)
                    .collect();
                if dot1_items.is_empty() {
                    continue;
                }

                let mut forwarded: BTreeMap<(u32, NonterminalID), u32> = BTreeMap::new();
                let mut rewrites = Vec::new();
                let mut valid = true;

                for item in dot1_items {
                    if item.rule == 0 {
                        valid = false;
                        break;
                    }

                    let rule = &rules[item.rule as usize];
                    let reduce_nt = rule.lhs;
                    let Some(&(forward_target, _)) = goto[source].get(&reduce_nt) else {
                        valid = false;
                        break;
                    };

                    let mut reduce_state = target;
                    let mut chain_unique = in_count[target as usize] <= 1;
                    for next_symbol in &rule.rhs[item.dot as usize..] {
                        let Some(&(next_state, _, _)) = transitions[reduce_state as usize].get(next_symbol) else {
                            valid = false;
                            break;
                        };
                        reduce_state = next_state;
                        // If a state in the chain is reachable from
                        // multiple predecessors, we can't safely rewrite
                        // its reduce because other paths share it.
                        if in_count[reduce_state as usize] > 1 {
                            chain_unique = false;
                        }
                    }
                    if !valid {
                        break;
                    }
                    if !chain_unique {
                        valid = false;
                        break;
                    }

                    let Some(current_len) = current_unique_reduce_len(
                        pending,
                        reduce_state,
                        item.lookahead,
                        reduce_nt,
                    ) else {
                        valid = false;
                        break;
                    };
                    if current_len != 1 {
                        valid = false;
                        break;
                    }

                    match forwarded.get(&(reduce_state, reduce_nt)) {
                        Some(&existing_target) if existing_target != forward_target => {
                            valid = false;
                            break;
                        }
                        Some(_) => {}
                        None => {
                            if let Some(&(existing_target, existing_replace)) =
                                goto[reduce_state as usize].get(&reduce_nt)
                            {
                                if existing_target != forward_target || !existing_replace {
                                    valid = false;
                                    break;
                                }
                            }
                            forwarded.insert((reduce_state, reduce_nt), forward_target);
                        }
                    }

                    rewrites.push((reduce_state, item.lookahead, reduce_nt));
                }

                if !valid || rewrites.is_empty() {
                    continue;
                }

                changed = true;

                match symbol {
                    Symbol::Terminal(terminal) => {
                        if let Some(pending_action) = pending[source].get_mut(terminal) {
                            if let Some((_, is_replace)) = pending_action.shift.as_mut() {
                                *is_replace = true;
                            }
                        }
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some((_, is_replace)) = goto[source].get_mut(nonterminal) {
                            *is_replace = true;
                        }
                    }
                }

                for &(reduce_state, lookahead, reduce_nt) in &rewrites {
                    if let Some(pending_action) = pending[reduce_state as usize].get_mut(&lookahead) {
                        for (existing_nt, reduce_len) in pending_action.reduces.iter_mut() {
                            if *existing_nt == reduce_nt && *reduce_len == 1 {
                                *reduce_len = 0;
                            }
                        }
                    }
                }

                for ((reduce_state, reduce_nt), forward_target) in forwarded {
                    goto[reduce_state as usize].insert(reduce_nt, (forward_target, true));
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Inline zero-pop reduces: when a state has Reduce(nt, 0) on some terminal,
/// follow the goto to the target state and copy the target's action for that
/// terminal into the current state. This eliminates the zero-pop reduce
/// entirely, replacing it with a direct shift or accept.
///
/// Iterates until no more inlining is possible (handles chains of zero-pop
/// reduces).
fn inline_zero_pop_reduces(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
) {
    loop {
        let mut changed = false;

        for state in 0..pending.len() {
            // Collect (terminal, nt, target_state) triples for zero-pop reduces.
            let mut to_inline: Vec<(TerminalID, NonterminalID, u32)> = Vec::new();
            if let Some(by_terminal) = pending.get(state) {
                for (&terminal, pa) in by_terminal {
                    for &(reduce_nt, reduce_len) in &pa.reduces {
                        if reduce_len == 0 {
                            if let Some(&(target, _)) = goto[state].get(&reduce_nt) {
                                to_inline.push((terminal, reduce_nt, target));
                            }
                        }
                    }
                }
            }

            for (terminal, reduce_nt, target) in to_inline {
                if target as usize == state {
                    continue; // avoid self-reference
                }

                // Read action at target state for the same terminal.
                let target_pa = pending[target as usize].get(&terminal).cloned();

                if let Some(pa) = pending[state].get_mut(&terminal) {
                    // Remove the zero-pop reduce.
                    pa.reduces.retain(|&(nt, len)| !(nt == reduce_nt && len == 0));

                    // Inline target's actions.
                    if let Some(tpa) = target_pa {
                        if let Some((shift_target, _)) = tpa.shift {
                            pa.push_shift(shift_target, true);
                        }
                        for (nt, len) in &tpa.reduces {
                            pa.push_reduce(*nt, *len);
                        }
                        if tpa.accept {
                            pa.push_accept();
                        }

                        // Propagate gotos needed for any zero-pop reduces
                        // we just inlined from the target state.
                        for &(nt, len) in &tpa.reduces {
                            if len == 0 {
                                if let Some(&goto_entry) = goto[target as usize].get(&nt) {
                                    goto[state].entry(nt).or_insert(goto_entry);
                                }
                            }
                        }
                    }

                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn build_lr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[LR1ItemSet],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let (pending, goto, forwarded_shifts) = initialize_pending_and_goto(transitions);

    // Replace flags are now carried in the transitions map from
    // build_lr1_item_sets, so we don't need to recompute them here.

    let mut pending = pending;
    let goto = goto;
    for (state_id, items) in item_sets.iter().enumerate() {

        for (item, lookaheads) in items {
            // Transferred items do not generate reduces.
            if item.transferred {
                continue;
            }
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            for bit in lookaheads.iter_ones() {
                let lookahead = bit_lookahead(bit, grammar.num_terminals);
                if item.rule == 0 {
                    pending[state_id].entry(lookahead).or_default().push_accept();
                    continue;
                }

                pending[state_id]
                    .entry(lookahead)
                    .or_default()
                    .push_reduce(rule.lhs, item.stack_depth);
            }
        }
    }

    // Grouped LR(1) lookahead sets delay scalar fanout until pending-action
    // emission. The old local-forward path is written for scalar LR1 items,
    // so keep replace handling conservative here rather than approximating.

    finish_table(
        grammar,
        pending,
        goto,
        forwarded_shifts,
        GlrTableConstruction::LegacyRowBisim,
        AdmissionPolicy::RowPresenceExact,
    )
}

// Legacy row-bisimulation merge over canonical LR(1) item sets.

fn lr1_core_key(items: &LR1ItemSet) -> Vec<Item> {
    let mut core = BTreeSet::new();
    for item in items.keys() {
        core.insert(Item::new(item.rule, item.dot, item.stack_depth));
    }
    core.into_iter().collect()
}

fn build_experimental_core_merged_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[LR1ItemSet],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> Option<GLRTable> {
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();
    let partition = refine_experimental_core_partition(&canonical, &core_keys);
    let mut table = union_experimental_core_rows(canonical, &partition)?;
    table.construction = GlrTableConstruction::ExperimentalCoreMerged;
    table.admission_policy = AdmissionPolicy::ExactSimulation;
    table.rebuild_advance_rows_from_actions();
    Some(table)
}

fn refine_experimental_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let mut class_by_core: BTreeMap<Vec<Item>, u32> = BTreeMap::new();
    let mut partition = Vec::with_capacity(core_keys.len());
    for key in core_keys {
        let next = class_by_core.len() as u32;
        partition.push(*class_by_core.entry(key.clone()).or_insert(next));
    }

    loop {
        let mut sig_to_class: BTreeMap<ExperimentalCoreCompatibilitySig, u32> = BTreeMap::new();
        let mut next_partition = Vec::with_capacity(partition.len());
        for state in 0..table.num_states as usize {
            let sig = ExperimentalCoreCompatibilitySig::new(table, state, partition[state], &partition);
            let next = sig_to_class.len() as u32;
            next_partition.push(*sig_to_class.entry(sig).or_insert(next));
        }
        if next_partition == partition {
            return partition;
        }
        partition = next_partition;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExperimentalCoreCompatibilitySig {
    core_class: u32,
    shifts: Vec<(TerminalID, u32, bool, bool)>,
    gotos: Vec<(NonterminalID, u32, bool)>,
}

impl ExperimentalCoreCompatibilitySig {
    fn new(table: &GLRTable, state: usize, core_class: u32, partition: &[u32]) -> Self {
        let mut shifts = Vec::new();
        for (terminal, action) in &table.action[state] {
            if let Some((target, replace)) = action_shift(action) {
                shifts.push((
                    terminal,
                    partition[target as usize],
                    replace,
                    table.forwarded_shifts.contains(&(state as u32, terminal)),
                ));
            }
        }
        shifts.sort_unstable();

        let mut gotos = table.goto[state]
            .iter()
            .map(|(&nt, &(target, replace))| (nt, partition[target as usize], replace))
            .collect::<Vec<_>>();
        gotos.sort_unstable();

        Self {
            core_class,
            shifts,
            gotos,
        }
    }
}

fn action_shift(action: &Action) -> Option<(u32, bool)> {
    match action {
        Action::Shift(target, replace) => Some((*target, *replace)),
        Action::Split {
            shift: Some((target, replace)),
            ..
        } => Some((*target, *replace)),
        _ => None,
    }
}

fn union_experimental_core_rows(table: GLRTable, partition: &[u32]) -> Option<GLRTable> {
    let nstates = table.num_states as usize;
    let ngroups = partition.iter().copied().max().map(|x| x + 1).unwrap_or(0) as usize;

    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(ngroups)
        .collect::<Vec<_>>();
    let mut goto = (0..ngroups).map(|_| FxHashMap::default()).collect::<Vec<_>>();
    let mut forwarded_shifts = FxHashSet::default();

    for state in 0..nstates {
        let group = partition[state] as usize;
        for (terminal, action) in &table.action[state] {
            add_remapped_action_to_pending(
                action,
                &mut pending[group].entry(terminal).or_default(),
                partition,
            )?;
            if action_shift(action).is_some()
                && table.forwarded_shifts.contains(&(state as u32, terminal))
            {
                forwarded_shifts.insert((group as u32, terminal));
            }
        }
        for (&nt, &(target, replace)) in &table.goto[state] {
            let remapped = (partition[target as usize], replace);
            match goto[group].get(&nt).copied() {
                Some(existing) if existing != remapped => return None,
                Some(_) => {}
                None => {
                    goto[group].insert(nt, remapped);
                }
            }
        }
    }

    let action = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .filter_map(|(terminal, pending)| pending.maybe_finish().map(|action| (terminal, action)))
                .collect::<ActionRow>()
        })
        .collect();
    let goto = goto
        .into_iter()
        .map(|row| row.into_iter().collect::<GotoRow>())
        .collect();

    Some(GLRTable {
        action,
        goto,
        num_states: ngroups as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
        nonterminal_display_names: table.nonterminal_display_names,
        construction: table.construction,
        admission_policy: table.admission_policy,
        advance: Vec::new(),
        forwarded_shifts,
        guarded_shift_index: Vec::new(),
    })
}

fn add_remapped_action_to_pending(
    action: &Action,
    pending: &mut PendingAction,
    partition: &[u32],
) -> Option<()> {
    match action {
        Action::Shift(target, replace) => pending.push_shift(partition[*target as usize], *replace),
        Action::Reduce(nt, len) => pending.push_reduce(*nt, *len),
        Action::Accept => pending.push_accept(),
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                pending.push_shift(partition[*target as usize], *replace);
            }
            for &(nt, len) in reduces {
                pending.push_reduce(nt, len);
            }
            if *accept {
                pending.push_accept();
            }
        }
        Action::StackShifts(_) | Action::GuardedStackShifts(_) => return None,
    }
    Some(())
}

fn build_legacy_row_bisim_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[LR1ItemSet],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();
    merge_same_core_lr1_states(canonical, &core_keys)
}

#[cfg(test)]
fn grouped_item_lookahead_counts(grammar: &AnalyzedGrammar) -> Vec<Vec<(u32, u32, u32, usize)>> {
    let (item_sets, _) = build_lr1_item_sets(grammar);
    item_sets
        .into_iter()
        .map(|items| {
            items
                .into_iter()
                .map(|(core, lookaheads)| (core.rule, core.dot, core.stack_depth, lookaheads.count_ones()))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        build_lalr_table, build_table, build_table_with_default_construction,
        grouped_item_lookahead_counts,
    };
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::{Action, AdmissionPolicy, GlrTableConstruction};
    use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

    fn multi_lookahead_grammar() -> AnalyzedGrammar {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(3)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"x".to_vec() },
                Terminal::Literal { id: 1, bytes: b"a".to_vec() },
                Terminal::Literal { id: 2, bytes: b"b".to_vec() },
                Terminal::Literal { id: 3, bytes: b"c".to_vec() },
            ],
            ..GrammarDef::default()
        };
        AnalyzedGrammar::from_grammar_def(&grammar)
    }

    fn mysterious_conflict_grammar() -> AnalyzedGrammar {
        let grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(3),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(3),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(2)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() },
                Terminal::Literal { id: 2, bytes: b"c".to_vec() },
                Terminal::Literal { id: 3, bytes: b"d".to_vec() },
                Terminal::Literal { id: 4, bytes: b"e".to_vec() },
            ],
            ..GrammarDef::default()
        };
        AnalyzedGrammar::from_grammar_def(&grammar)
    }

    #[test]
    fn grouped_lr1_items_merge_multiple_lookaheads_on_one_core() {
        let grammar = multi_lookahead_grammar();
        let counts = grouped_item_lookahead_counts(&grammar);

        assert!(
            counts
                .iter()
                .flatten()
                .any(|&(rule, dot, _stack_depth, lookahead_count)| {
                    rule == 4 && dot == 1 && lookahead_count == 3
                }),
            "{counts:?}"
        );
    }

    #[test]
    fn grouped_lr1_items_still_emit_expected_lowered_shift_actions() {
        let grammar = multi_lookahead_grammar();
        let table = build_table(&grammar);

        assert!(table.action.iter().any(|row| {
            matches!(row.get(&1), Some(Action::Shift(_, true)))
                && matches!(row.get(&2), Some(Action::Shift(_, true)))
                && matches!(row.get(&3), Some(Action::Shift(_, true)))
        }));
    }

    #[test]
    fn default_build_uses_core_merged_exact_admission() {
        let grammar = multi_lookahead_grammar();
        let table = build_table(&grammar);

        assert_eq!(
            table.construction,
            GlrTableConstruction::ExperimentalCoreMerged
        );
        assert_eq!(table.admission_policy, AdmissionPolicy::ExactSimulation);
    }

    #[test]
    fn legacy_row_bisim_can_be_requested_as_default() {
        let grammar = multi_lookahead_grammar();
        let table = build_table_with_default_construction(
            &grammar,
            GlrTableConstruction::LegacyRowBisim,
        );

        assert_eq!(table.construction, GlrTableConstruction::LegacyRowBisim);
        assert_eq!(table.admission_policy, AdmissionPolicy::RowPresenceExact);
    }

    #[test]
    fn lalr_builds_real_lr0_based_table() {
        let grammar = multi_lookahead_grammar();
        let table = build_lalr_table(&grammar);

        assert_eq!(table.construction, GlrTableConstruction::Lalr);
        assert_eq!(table.admission_policy, AdmissionPolicy::ExactSimulation);
        assert!(!table.has_ambiguity(), "{:#?}", table.ambiguous_actions());
    }

    #[test]
    fn lalr_exposes_classic_lr1_not_lalr_conflict() {
        let grammar = mysterious_conflict_grammar();
        let table = build_lalr_table(&grammar);

        assert_eq!(table.construction, GlrTableConstruction::Lalr);
        assert!(table.has_ambiguity(), "expected GLR split from LALR merge");
    }

}
