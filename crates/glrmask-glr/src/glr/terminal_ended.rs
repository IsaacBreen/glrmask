//! Budgeted binarization, nullable elimination, and unit-SCC preprocessing.

use std::collections::{BTreeMap, BTreeSet};

use crate::grammar::flat::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};

/// Dense internal nonterminal index. Never a public `NonterminalID`.
type NtIx = u32;

/// Construction stage a budget charge belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Preprocess,
}

/// Resource category a budget charge belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    CandidateWork,
    RetainedRules,
    Symbols,
    FreshIds,
}

/// A typed, recoverable failure. The construction never panics and never
/// returns a partially published grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileDecline {
    ResourceLimit {
        stage: Stage,
        kind: ResourceKind,
        used: u64,
        limit: u64,
    },
    StrategyUnsupported {
        stage: Stage,
        reason: String,
    },
}

/// Hard limits for one attempt.
#[derive(Debug, Clone, Copy)]
pub struct CompileLimits {
    /// Candidate rules considered, before deduplication.
    pub max_candidate_work: u64,
    /// Distinct rules retained.
    pub max_retained_rules: u64,
    /// Retained RHS symbols.
    pub max_symbols: u64,
}

impl Default for CompileLimits {
    fn default() -> Self {
        Self {
            max_candidate_work: 5_000_000,
            max_retained_rules: 1_000_000,
            max_symbols: 3_000_000,
        }
    }
}

/// A single aggregate attempt budget. Allowances are never reset per stage or
/// per fixed-point round; every charge is cumulative.
#[derive(Debug, Clone, Default)]
pub struct CompileBudget {
    pub candidate_work: u64,
    pub retained_rules: u64,
    pub symbols: u64,
}

impl CompileBudget {
    pub fn new() -> Self {
        Self::default()
    }

    fn charge(
        &mut self,
        stage: Stage,
        kind: ResourceKind,
        limits: &CompileLimits,
        amount: u64,
    ) -> Result<(), CompileDecline> {
        let (used, limit) = match kind {
            ResourceKind::CandidateWork => {
                self.candidate_work = self.candidate_work.saturating_add(amount);
                (self.candidate_work, limits.max_candidate_work)
            }
            ResourceKind::RetainedRules => {
                self.retained_rules = self.retained_rules.saturating_add(amount);
                (self.retained_rules, limits.max_retained_rules)
            }
            ResourceKind::Symbols => {
                self.symbols = self.symbols.saturating_add(amount);
                (self.symbols, limits.max_symbols)
            }
            ResourceKind::FreshIds => {
                self.retained_rules = self.retained_rules.saturating_add(amount);
                (self.retained_rules, limits.max_retained_rules)
            }
        };
        if used > limit {
            return Err(CompileDecline::ResourceLimit {
                stage,
                kind,
                used,
                limit,
            });
        }
        Ok(())
    }

    fn charge_candidate_work(
        &mut self,
        stage: Stage,
        limits: &CompileLimits,
        amount: u64,
    ) -> Result<(), CompileDecline> {
        self.charge(stage, ResourceKind::CandidateWork, limits, amount)
    }

    fn charge_retained_rule(
        &mut self,
        stage: Stage,
        limits: &CompileLimits,
    ) -> Result<(), CompileDecline> {
        self.charge(stage, ResourceKind::RetainedRules, limits, 1)
    }

    fn charge_symbols(
        &mut self,
        stage: Stage,
        limits: &CompileLimits,
        amount: u64,
    ) -> Result<(), CompileDecline> {
        self.charge(stage, ResourceKind::Symbols, limits, amount)
    }
}

/// Binary, epsilon-free input to the core construction. Every nonterminal is
/// non-nullable; every terminal is non-nullable.
struct BinaryGrammar {
    terminal_rules: Vec<(NtIx, TerminalID)>,
    unit_rules: Vec<(NtIx, NtIx)>,
    binary_rules: Vec<(NtIx, NtIx, NtIx)>,
}

struct Preprocessed {
    grammar: BinaryGrammar,
    start: NtIx,
}

// ── Dense id allocation ─────────────────────────────────────────────────────

struct DenseIds {
    index_of: BTreeMap<NonterminalID, NtIx>,
    next: u32,
}

impl DenseIds {
    fn new() -> Self {
        Self {
            index_of: BTreeMap::new(),
            next: 0,
        }
    }

    fn intern(&mut self, public: NonterminalID) -> Result<NtIx, CompileDecline> {
        if let Some(&dense) = self.index_of.get(&public) {
            return Ok(dense);
        }
        let dense = self.fresh()?;
        self.index_of.insert(public, dense);
        Ok(dense)
    }

    fn fresh(&mut self) -> Result<NtIx, CompileDecline> {
        if self.next == u32::MAX {
            return Err(CompileDecline::ResourceLimit {
                stage: Stage::Preprocess,
                kind: ResourceKind::FreshIds,
                used: u32::MAX as u64,
                limit: u32::MAX as u64,
            });
        }
        let dense = self.next;
        self.next += 1;
        Ok(dense)
    }
}

// ── Public entry points ─────────────────────────────────────────────────────

/// Run only the terminal-ended preprocessing step and return it as an ordinary
/// `GrammarDef`.
///
/// The result is the compact, binary, epsilon-free grammar consumed by the
/// terminal-ended construction. It is not the right-spine normal form. Source
/// epsilon membership is preserved as an explicit `start -> epsilon` rule
/// because the ordinary grammar pipeline expects it in-band; the terminal-ended
/// construction instead carries it out of band (`accepts_empty`).
///
/// Nonterminal names are cleared because preprocessing renumbers nonterminals;
/// the start symbol is still protected downstream. A frontend direct-regular
/// automaton is terminal-level and language-preserving, so it is retained.
pub fn preprocess_to_grammar_def(
    source: &GrammarDef,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<GrammarDef, CompileDecline> {
    let start_nullable = source.start_is_nullable();
    let preprocessed = preprocess(source, limits, budget)?;
    let grammar = &preprocessed.grammar;
    let mut rules = Vec::with_capacity(
        grammar.terminal_rules.len() + grammar.unit_rules.len() + grammar.binary_rules.len(),
    );
    for &(lhs, terminal) in &grammar.terminal_rules {
        rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Terminal(terminal)],
        });
    }
    for &(lhs, right) in &grammar.unit_rules {
        rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Nonterminal(right)],
        });
    }
    for &(lhs, left, right) in &grammar.binary_rules {
        rules.push(Rule {
            lhs,
            rhs: vec![Symbol::Nonterminal(left), Symbol::Nonterminal(right)],
        });
    }
    if start_nullable {
        rules.push(Rule {
            lhs: preprocessed.start,
            rhs: Vec::new(),
        });
    }

    let mut out = source.clone();
    out.rules = rules;
    out.start = preprocessed.start;
    out.nonterminal_names.clear();
    Ok(out)
}

fn preprocess(
    source: &GrammarDef,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<Preprocessed, CompileDecline> {
    let mut ids = DenseIds::new();

    // Deterministic dense indexing: allocate public coordinates in ascending
    // order before any fresh helper, so the source-to-dense map is stable.
    let mut public_ids = BTreeSet::new();
    public_ids.insert(source.start);
    for rule in &source.rules {
        public_ids.insert(rule.lhs);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(nonterminal) = symbol {
                public_ids.insert(*nonterminal);
            }
        }
    }
    for &public in &public_ids {
        ids.intern(public)?;
    }
    let start_dense = ids.intern(source.start)?;

    // 1. Split nullable terminals compactly: T_t -> t | epsilon.
    let nullable_terminals: BTreeSet<TerminalID> = source
        .terminals
        .iter()
        .filter(|terminal| terminal.is_nullable())
        .map(|terminal| terminal.id())
        .collect();

    let mut rules: Vec<Rule> = Vec::with_capacity(source.rules.len() + nullable_terminals.len());
    for rule in &source.rules {
        rules.push(Rule {
            lhs: ids.intern(rule.lhs)?,
            rhs: rule
                .rhs
                .iter()
                .map(|symbol| match symbol {
                    Symbol::Terminal(terminal) => Ok(Symbol::Terminal(*terminal)),
                    Symbol::Nonterminal(nonterminal) => {
                        ids.intern(*nonterminal).map(Symbol::Nonterminal)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        });
    }

    if !nullable_terminals.is_empty() {
        let mut nt_for_terminal = BTreeMap::new();
        for &terminal in &nullable_terminals {
            nt_for_terminal.insert(terminal, ids.fresh()?);
        }
        for rule in &mut rules {
            for symbol in &mut rule.rhs {
                if let Symbol::Terminal(terminal) = symbol
                    && let Some(&wrapper) = nt_for_terminal.get(terminal)
                {
                    *symbol = Symbol::Nonterminal(wrapper);
                }
            }
        }
        for (&terminal, &wrapper) in &nt_for_terminal {
            rules.push(Rule {
                lhs: wrapper,
                rhs: Vec::new(),
            });
            rules.push(Rule {
                lhs: wrapper,
                rhs: vec![Symbol::Terminal(terminal)],
            });
        }
    }

    // 2. Lift terminals in multi-symbol productions into terminal nonterminals.
    let mut lift = BTreeMap::<TerminalID, NtIx>::new();
    for rule in &rules {
        if rule.rhs.len() > 1 {
            for symbol in &rule.rhs {
                if let Symbol::Terminal(terminal) = symbol {
                    lift.entry(*terminal).or_insert(0);
                }
            }
        }
    }
    if !lift.is_empty() {
        for value in lift.values_mut() {
            *value = ids.fresh()?;
        }
        for rule in &mut rules {
            if rule.rhs.len() > 1 {
                for symbol in &mut rule.rhs {
                    if let Symbol::Terminal(terminal) = symbol
                        && let Some(&wrapper) = lift.get(terminal)
                    {
                        *symbol = Symbol::Nonterminal(wrapper);
                    }
                }
            }
        }
        for (&terminal, &wrapper) in &lift {
            rules.push(Rule {
                lhs: wrapper,
                rhs: vec![Symbol::Terminal(terminal)],
            });
        }
    }

    // 3. Binarize long productions *before* epsilon elimination.
    let mut binarized: Vec<Rule> = Vec::with_capacity(rules.len());
    for rule in rules {
        if rule.rhs.len() <= 2 {
            binarized.push(rule);
            continue;
        }
        let rhs = rule.rhs;
        let last = rhs.len() - 1;
        let mut current = rule.lhs;
        let mut index = 0usize;
        while index + 2 < rhs.len() {
            let helper = ids.fresh()?;
            binarized.push(Rule {
                lhs: current,
                rhs: vec![rhs[index].clone(), Symbol::Nonterminal(helper)],
            });
            budget.charge_retained_rule(Stage::Preprocess, limits)?;
            budget.charge_symbols(Stage::Preprocess, limits, 2)?;
            current = helper;
            index += 1;
        }
        binarized.push(Rule {
            lhs: current,
            rhs: vec![rhs[last - 1].clone(), rhs[last].clone()],
        });
    }

    // 4. Compute nullability and record entry nullability before elimination.
    let num_nts = ids.next as usize;
    let nullable = compute_nullable(&binarized, num_nts);

    // 5. Eliminate epsilon from the binary productions.
    let mut eliminated: Vec<Rule> = Vec::with_capacity(binarized.len() + binarized.len() / 2);
    for rule in binarized {
        match rule.rhs.len() {
            0 => {
                // The empty variant is represented by saved nullability.
                budget.charge_candidate_work(Stage::Preprocess, limits, 1)?;
            }
            1 => eliminated.push(rule),
            2 => {
                let left_nullable = matches!(
                    rule.rhs[0],
                    Symbol::Nonterminal(nonterminal) if nullable[nonterminal as usize]
                );
                let right_nullable = matches!(
                    rule.rhs[1],
                    Symbol::Nonterminal(nonterminal) if nullable[nonterminal as usize]
                );
                budget.charge_candidate_work(Stage::Preprocess, limits, 1)?;
                eliminated.push(rule.clone());
                if right_nullable {
                    budget.charge_candidate_work(Stage::Preprocess, limits, 1)?;
                    eliminated.push(Rule {
                        lhs: rule.lhs,
                        rhs: vec![rule.rhs[0].clone()],
                    });
                }
                if left_nullable {
                    budget.charge_candidate_work(Stage::Preprocess, limits, 1)?;
                    eliminated.push(Rule {
                        lhs: rule.lhs,
                        rhs: vec![rule.rhs[1].clone()],
                    });
                }
            }
            _ => {
                return Err(CompileDecline::StrategyUnsupported {
                    stage: Stage::Preprocess,
                    reason: "binarization invariant violated".to_owned(),
                });
            }
        }
    }

    // 6. Deduplicate, then remove nonproductive and unreachable material.
    let mut rules = dedup_rules(eliminated);
    rules.retain(|rule| !is_reflexive_unit(rule));
    prune_productive(&mut rules, num_nts);
    prune_unreachable(&mut rules, start_dense, num_nts);
    let (mut rules, mut start_dense) = compact_rules(rules, start_dense);

    // 7. Collapse unit SCCs proactively, retaining unit exits.
    let (collapsed, collapsed_start) = collapse_unit_sccs(rules, start_dense, limits, budget)?;
    rules = collapsed;
    start_dense = collapsed_start;
    rules = dedup_rules(rules);
    rules.retain(|rule| !is_reflexive_unit(rule));

    let mut terminal_rules = Vec::new();
    let mut unit_rules = Vec::new();
    let mut binary_rules = Vec::new();
    for rule in &rules {
        match rule.rhs.as_slice() {
            [Symbol::Terminal(terminal)] => terminal_rules.push((rule.lhs, *terminal)),
            [Symbol::Nonterminal(right)] => unit_rules.push((rule.lhs, *right)),
            [Symbol::Nonterminal(left), Symbol::Nonterminal(right)] => {
                binary_rules.push((rule.lhs, *left, *right));
            }
            _ => {
                return Err(CompileDecline::StrategyUnsupported {
                    stage: Stage::Preprocess,
                    reason: format!("unexpected preprocessed rule {rule:?}"),
                });
            }
        }
    }

    Ok(Preprocessed {
        grammar: BinaryGrammar {
            terminal_rules,
            unit_rules,
            binary_rules,
        },
        start: start_dense,
    })
}

fn compute_nullable(rules: &[Rule], num_nts: usize) -> Vec<bool> {
    let mut nullable = vec![false; num_nts];
    loop {
        let mut changed = false;
        for rule in rules {
            if nullable[rule.lhs as usize] {
                continue;
            }
            let all_nullable = rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => false,
                Symbol::Nonterminal(nonterminal) => nullable[*nonterminal as usize],
            });
            if all_nullable {
                nullable[rule.lhs as usize] = true;
                changed = true;
            }
        }
        if !changed {
            return nullable;
        }
    }
}

fn is_reflexive_unit(rule: &Rule) -> bool {
    matches!(rule.rhs.as_slice(), [Symbol::Nonterminal(right)] if *right == rule.lhs)
}

fn dedup_rules(rules: Vec<Rule>) -> Vec<Rule> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        if seen.insert((rule.lhs, rule.rhs.clone())) {
            out.push(rule);
        }
    }
    out
}

fn prune_productive(rules: &mut Vec<Rule>, num_nts: usize) {
    loop {
        let mut productive = vec![false; num_nts];
        loop {
            let mut changed = false;
            for rule in rules.iter() {
                if productive[rule.lhs as usize] {
                    continue;
                }
                let all = rule.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => true,
                    Symbol::Nonterminal(nonterminal) => productive[*nonterminal as usize],
                });
                if all {
                    productive[rule.lhs as usize] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let before = rules.len();
        rules.retain(|rule| {
            rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::Nonterminal(nonterminal) => productive[*nonterminal as usize],
            })
        });
        if rules.len() == before {
            return;
        }
    }
}

fn prune_unreachable(rules: &mut Vec<Rule>, start: NtIx, num_nts: usize) {
    let mut reachable = vec![false; num_nts];
    reachable[start as usize] = true;
    loop {
        let mut changed = false;
        for rule in rules.iter() {
            if !reachable[rule.lhs as usize] {
                continue;
            }
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol
                    && !reachable[*nonterminal as usize]
                {
                    reachable[*nonterminal as usize] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    rules.retain(|rule| reachable[rule.lhs as usize]);
}

/// Re-densify nonterminal coordinates after pruning so closure matrices stay
/// compact. Returns the remapped start.
fn compact_rules(rules: Vec<Rule>, start: NtIx) -> (Vec<Rule>, NtIx) {
    let mut used = BTreeSet::new();
    for rule in &rules {
        used.insert(rule.lhs);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(nonterminal) = symbol {
                used.insert(*nonterminal);
            }
        }
    }
    used.insert(start);
    let map: BTreeMap<NtIx, NtIx> = used
        .into_iter()
        .enumerate()
        .map(|(index, old)| (old, index as NtIx))
        .collect();
    let remap = |nonterminal: NtIx| map[&nonterminal];
    let rules = rules
        .into_iter()
        .map(|rule| Rule {
            lhs: remap(rule.lhs),
            rhs: rule
                .rhs
                .into_iter()
                .map(|symbol| match symbol {
                    Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(remap(nonterminal)),
                    Symbol::Terminal(terminal) => Symbol::Terminal(terminal),
                })
                .collect(),
        })
        .collect();
    (rules, remap(start))
}

/// Collapse every non-trivial unit SCC onto its smallest member, retaining the
/// component's non-unit and outgoing unit productions. Full transitive unit
/// elimination is unnecessary: the closure construction below handles the
/// remaining unit DAG.
fn collapse_unit_sccs(
    rules: Vec<Rule>,
    start: NtIx,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<(Vec<Rule>, NtIx), CompileDecline> {
    let num_nts = rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .map(|maximum| maximum as usize + 1)
        .unwrap_or(0)
        .max(start as usize + 1);

    // Kosaraju's two passes visit each vertex and edge a constant number of
    // times. Charge before allocating either adjacency list or DFS stacks.
    let unit_edges = rules.iter().filter(|r| matches!(r.rhs.as_slice(), [Symbol::Nonterminal(_)])).count() as u64;
    budget.charge_candidate_work(
        Stage::Preprocess,
        limits,
        (num_nts as u64).saturating_mul(3)
            .saturating_add(unit_edges.saturating_mul(4))
            .saturating_add(rules.len() as u64),
    )?;
    let mut forward = vec![Vec::<usize>::new(); num_nts];
    let mut reverse = vec![Vec::<usize>::new(); num_nts];
    for rule in &rules {
        if let [Symbol::Nonterminal(right)] = rule.rhs.as_slice() {
            let left = rule.lhs as usize;
            let right = *right as usize;
            forward[left].push(right);
            reverse[right].push(left);
        }
    }

    // Iterative postorder: a long unit chain must not consume the call stack.
    let mut seen = vec![false; num_nts];
    let mut finished = Vec::with_capacity(num_nts);
    let mut dfs = Vec::<(usize, usize)>::new();
    for root in 0..num_nts {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        dfs.push((root, 0));
        while let Some((node, next)) = dfs.last_mut() {
            if let Some(&child) = forward[*node].get(*next) {
                *next += 1;
                if !seen[child] {
                    seen[child] = true;
                    dfs.push((child, 0));
                }
            } else {
                finished.push(*node);
                dfs.pop();
            }
        }
    }

    seen.fill(false);
    let mut representative = vec![0; num_nts];
    let mut pending = Vec::new();
    let mut members = Vec::new();
    for root in finished.into_iter().rev() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        pending.push(root);
        let mut minimum = root;
        while let Some(node) = pending.pop() {
            minimum = minimum.min(node);
            members.push(node);
            for &child in &reverse[node] {
                if !seen[child] {
                    seen[child] = true;
                    pending.push(child);
                }
            }
        }
        for member in members.drain(..) {
            representative[member] = minimum as NtIx;
        }
    }

    let redirect = |nonterminal: NtIx| representative[nonterminal as usize];
    let mut collapsed = Vec::with_capacity(rules.len());
    for rule in rules {
        let lhs = redirect(rule.lhs);
        let rhs = rule
            .rhs
            .into_iter()
            .map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Symbol::Nonterminal(redirect(nonterminal)),
                Symbol::Terminal(terminal) => Symbol::Terminal(terminal),
            })
            .collect();
        collapsed.push(Rule { lhs, rhs });
    }
    let collapsed = dedup_rules(collapsed);
    let (collapsed, start) = compact_rules(collapsed, redirect(start));
    Ok((collapsed, start))
}

#[cfg(test)]
#[path = "terminal_ended_tests.rs"]
mod tests;
