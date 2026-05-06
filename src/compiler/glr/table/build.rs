use super::*;

pub(super) fn build_table(grammar: &AnalyzedGrammar) -> GLRTable {
    let t0 = std::time::Instant::now();
    let (item_sets, transitions) = build_lr1_item_sets(grammar);
    let lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let mut table = build_ielr_table(grammar, &item_sets, &transitions);
    let ielr_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let pre_merge_states = table.num_states;
    let t2 = std::time::Instant::now();
    table.merge_identical_rows();
    table.collapse_sr_unit_reductions_with_compatible_gotos();
    table.merge_identical_rows();
    let merge_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = std::time::Instant::now();
    table.merge_recognizer_equivalent();
    let recog_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let _ = (lr1_ms, ielr_ms, pre_merge_states, merge_ms, recog_ms, item_sets);

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
        forwarded_shifts,
    }
}

// LR(1) item set construction.

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
fn first_of_sequence(
    symbols: &[Symbol],
    lookahead: TerminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeSet<TerminalID> {
    let mut result = BTreeSet::new();
    let mut all_nullable = true;
    for sym in symbols {
        match sym {
            Symbol::Terminal(t) => {
                result.insert(*t);
                all_nullable = false;
                break;
            }
            Symbol::Nonterminal(nt) => {
                result.extend(&first[*nt as usize]);
                if !nullable.contains(nt) {
                    all_nullable = false;
                    break;
                }
            }
        }
    }
    if all_nullable {
        result.insert(lookahead);
    }
    result
}

fn lr1_closure(
    items: &BTreeSet<LR1Item>,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut result = items.clone();
    let mut queue: VecDeque<LR1Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        // Transferred items do not participate in closure.
        if item.transferred {
            continue;
        }
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let rhs = &rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];

            let lookaheads = first_of_sequence(beta, item.lookahead, &grammar.first, &grammar.nullable);

            for &i in &grammar.rules_by_lhs[*nt as usize] {
                let sd = grammar.rules[i as usize].rhs.len() as u32;
                for &la in &lookaheads {
                    let new_item = LR1Item::new(i, 0, la, sd);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
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
) -> (Vec<BTreeSet<LR1Item>>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        let sd = rules[0].rhs.len() as u32;
        s.insert(LR1Item::new(0, 0, EOF, sd));
        lr1_closure(&s, grammar)
    };

    let mut item_sets = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<LR1Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    // Pre-compute safety checks for the transfer mechanism only when the
    // mechanism is enabled. Recursion detection can be expensive on large
    // grammars, so avoid paying that cost on the default path.
    let transfer_safe = if local_forward_replace_enabled() {
        !grammar.nullable.iter().any(|&n| n != 0) && !grammar_has_recursion(rules)
    } else {
        false
    };

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let source_items = item_sets[state_id as usize].clone();

        // Build all goto kernels in a single pass over items.
        let mut kernels: BTreeMap<Symbol, BTreeSet<LR1Item>> = BTreeMap::new();
        for item in &source_items {
            // Transferred items only advance through nonterminal gotos.
            if item.transferred {
                if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                    kernels
                        .entry(Symbol::Nonterminal(*nt))
                        .or_default()
                        .insert(LR1Item {
                            rule: item.rule,
                            dot: item.dot + 1,
                            lookahead: item.lookahead,
                            stack_depth: item.stack_depth,
                            transferred: false, // un-transfer on goto
                        });
                }
                continue;
            }
            if let Some(sym) = item.next_symbol(rules) {
                kernels
                    .entry(sym.clone())
                    .or_default()
                    .insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead, item.stack_depth));
            }
        }

        for (symbol, mut kernel) in kernels {
            let base_kernel = kernel.clone();
            // Check replace condition: is_replace iff no item in the kernel
            // has dot at position 1 (i.e., all items came from items that
            // already had dot > 0).
            let has_dot_1 = kernel.iter().any(|item| item.dot == 1);
            let mut is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };
            let mut used_transfer = false;

            // Local-forward transfer: when has_dot_1 but transfer is enabled,
            // try to transfer foo items from the source state into the kernel
            // so the transition can still be marked as replace.
            if has_dot_1 && !is_replace && local_forward_replace_enabled() && transfer_safe {
                let enable_for_this = match &symbol {
                    Symbol::Terminal(_) => replace_shifts_enabled(),
                    Symbol::Nonterminal(_) => replace_gotos_enabled(),
                };
                if enable_for_this {
                    if let Some(transferred) =
                        compute_transfer_items(&kernel, &source_items, rules)
                    {
                        kernel.extend(transferred);
                        // De-duplicate: remove transferred items that also
                        // exist as non-transferred.
                        let to_remove: Vec<LR1Item> = kernel
                            .iter()
                            .filter(|it| it.transferred)
                            .filter(|it| {
                                kernel.contains(&LR1Item {
                                    transferred: false,
                                    ..**it
                                })
                            })
                            .copied()
                            .collect();
                        for item in to_remove {
                            kernel.remove(&item);
                        }
                        is_replace = true;
                        used_transfer = true;
                    }
                }
            }

            let mut kernel_has_transferred = used_transfer && kernel.iter().any(|it| it.transferred);

            // If replace, decrement stack_depth for non-transferred kernel
            // items — the replace absorbs one stack level for items that
            // went through the shift. Transferred items didn't go through
            // the shift; they were copied from the source state and await
            // nonterminal gotos, so their sd is already correct.
            let mut adjusted_kernel: BTreeSet<LR1Item> = if is_replace {
                kernel
                    .iter()
                    .map(|item| {
                        if item.transferred {
                            *item
                        } else {
                            LR1Item {
                                stack_depth: item.stack_depth.saturating_sub(1),
                                ..*item
                            }
                        }
                    })
                    .collect()
            } else {
                kernel
            };

            let mut target_items = lr1_closure(&adjusted_kernel, grammar);
            if used_transfer && matches!(symbol, Symbol::Terminal(_)) {
                let base_target_items = lr1_closure(&base_kernel, grammar);
                let base_has_zero_pop_completed = base_target_items.iter().any(|item| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                let transferred_has_zero_pop_completed = target_items.iter().any(|item| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                if transferred_has_zero_pop_completed && !base_has_zero_pop_completed {
                    kernel = base_kernel;
                    is_replace = false;
                    kernel_has_transferred = false;
                    adjusted_kernel = kernel;
                    target_items = lr1_closure(&adjusted_kernel, grammar);
                }
            }

            // Track whether this replace was created by the transfer mechanism.
            let is_forwarded = is_replace && kernel_has_transferred;

            if target_items.is_empty() {
                continue;
            }

            let key: Vec<LR1Item> = target_items.iter().copied().collect();
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
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let (pending, goto, forwarded_shifts) = initialize_pending_and_goto(transitions);

    // Replace flags are now carried in the transitions map from
    // build_lr1_item_sets, so we don't need to recompute them here.

    let mut pending = pending;
    let mut goto = goto;
    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
            // Transferred items do not generate reduces.
            if item.transferred {
                continue;
            }
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(item.lookahead).or_default().push_accept();
                continue;
            }

            pending[state_id]
                .entry(item.lookahead)
                .or_default()
                .push_reduce(rule.lhs, item.stack_depth);
        }
    }

    // Post-process: convert non-replace transitions to replace where
    // possible by following the dot-1 items through to their reduce
    // states and rewriting sd=1 reduces to sd=0 with forwarded gotos.
    // Only safe for grammars that are:
    // - not using the transferred-item mechanism
    // - non-recursive (no shared reduce states)
    // - non-nullable (no epsilon productions that create split actions)
    let local_forward_enabled = local_forward_replace_enabled();
    let has_forwarded = transitions.iter().any(|t| t.values().any(|&(_, _, f)| f));
    let has_nullable = grammar.nullable.iter().any(|&n| n != 0);
    let has_recursion = if local_forward_enabled {
        grammar_has_recursion(&grammar.rules)
    } else {
        false
    };
    if local_forward_enabled && !has_forwarded && !has_recursion && !has_nullable {
        apply_local_forward_replace(
            &mut pending,
            &mut goto,
            item_sets,
            transitions,
            &grammar.rules,
        );
    }
    // For non-forwarded grammars (apply_local_forward_replace path),
    // inline the zero-pop reduces it created.
    if local_forward_enabled && !has_forwarded && !has_recursion && !has_nullable {
        inline_zero_pop_reduces(&mut pending, &mut goto);
    }

    finish_table(grammar, pending, goto, forwarded_shifts)
}

// IELR-style merge.

fn lr1_core_key(items: &BTreeSet<LR1Item>) -> Vec<Item> {
    let mut core = BTreeSet::new();
    for item in items {
        core.insert(Item::new(item.rule, item.dot, item.stack_depth));
    }
    core.into_iter().collect()
}
fn build_ielr_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();
    merge_same_core_lr1_states(canonical, &core_keys)
}

