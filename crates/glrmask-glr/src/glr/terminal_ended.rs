//! Terminal-ended normal form: the shared right-spine construction.
//!
//! Every ordinary production this module emits has one of the forms
//!
//! ```text
//! X -> a,        X -> Y a,        X -> Y Z a,
//! ```
//!
//! with `a` a terminal, so every retained RHS has length one to three and ends
//! in a non-nullable terminal. The empty word is *not* an ordinary production;
//! it is carried out of band by [`PreparedGrammar::accepts_empty`], exactly
//! mirroring the compiler's existing mechanism (`GrammarDef::start_is_nullable`
//! captured before epsilon elimination and later installed through
//! `GLRTable::set_embedded_start_nullable`).
//!
//! This file is deliberately phase-1 construction only. It is not wired into
//! the compilation pipeline; phase 2 selects it as a rescue route.
//!
//! Construction sketch (see the design for the full proof):
//!
//! 1. preprocess the source into a compact, binary, epsilon-free grammar;
//! 2. build the unit closure `U`, the right-spine closure `R`, and the
//!    positive (`>= 1` binary edge) closure `V`;
//! 3. emit entry productions `A -> f a` and, demand-driven, context
//!    productions `P_{A,C} -> f g a` for every referenced pair.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use smallvec::SmallVec;

use crate::ds::bitset::BitSet;
use crate::grammar::flat::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};

/// Dense internal nonterminal index. Never a public `NonterminalID`.
type NtIx = u32;

/// Every generated RHS has at most three symbols.
type Rhs = SmallVec<[Sym; 3]>;

// ── Budgets and typed decline ───────────────────────────────────────────────

/// Construction stage a budget charge belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Preprocess,
    Closures,
    Construct,
    Certify,
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

/// The construction result. `rules` generates the nonempty part; the empty word
/// is accepted exactly when `accepts_empty` is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedGrammar {
    pub rules: Vec<Rule>,
    pub start: NonterminalID,
    pub accepts_empty: bool,
}

/// Diagnostic counts for tests and release gates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TerminalEndedStats {
    pub preprocessed_nonterminals: usize,
    pub preprocessed_terminal_rules: usize,
    pub preprocessed_unit_rules: usize,
    pub preprocessed_binary_rules: usize,
    pub demanded_pairs: usize,
    pub candidate_work: u64,
    pub retained_rules: u64,
}

// ── Symbolic output coordinates ─────────────────────────────────────────────

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
struct PairKey {
    from: NtIx,
    to: NtIx,
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
enum Name {
    Entry(NonterminalID),
    Pair(PairKey),
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
enum Sym {
    Nt(Name),
    Terminal(TerminalID),
}

/// `F(A, C)` choice element: `Empty` denotes `epsilon` (a unit-only path) and
/// `One` denotes the positive context helper `P_{A,C}`.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum Fragment {
    Empty,
    One(PairKey),
}

// ── Preprocessed binary grammar ─────────────────────────────────────────────

/// Binary, epsilon-free input to the core construction. Every nonterminal is
/// non-nullable; every terminal is non-nullable.
struct BinaryGrammar {
    num_nts: usize,
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

/// Prepare `source` into terminal-ended normal form.
///
/// On failure no caller-visible rule vector is modified; the caller's `budget`
/// records the aggregate work charged up to the decline.
pub fn prepare_terminal_ended(
    source: &GrammarDef,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<PreparedGrammar, CompileDecline> {
    prepare_terminal_ended_with_stats(source, limits, budget).map(|(grammar, _)| grammar)
}

/// As [`prepare_terminal_ended`], additionally returning diagnostic counts.
pub fn prepare_terminal_ended_with_stats(
    source: &GrammarDef,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<(PreparedGrammar, TerminalEndedStats), CompileDecline> {
    // Source epsilon membership must be captured before epsilon elimination.
    // This is the same out-of-band mechanism the ordinary pipeline uses.
    let accepts_empty = source.start_is_nullable();

    let preprocessed = preprocess(source, limits, budget)?;
    let closures = Closures::compute(&preprocessed.grammar, limits, budget)?;
    let (prepared, mut stats) = construct(
        &preprocessed,
        &closures,
        accepts_empty,
        source,
        limits,
        budget,
    )?;
    certify_terminal_ended(&prepared)?;

    stats.preprocessed_nonterminals = preprocessed.grammar.num_nts;
    stats.preprocessed_terminal_rules = preprocessed.grammar.terminal_rules.len();
    stats.preprocessed_unit_rules = preprocessed.grammar.unit_rules.len();
    stats.preprocessed_binary_rules = preprocessed.grammar.binary_rules.len();
    stats.candidate_work = budget.candidate_work;
    stats.retained_rules = budget.retained_rules;
    Ok((prepared, stats))
}

/// §6.4 final certificate.
///
/// Checks the structural terminal-ended invariant plus the pruning obligations
/// that make the construction's table proof apply. It intentionally does *not*
/// require an acyclic left-corner graph.
pub fn certify_terminal_ended(g: &PreparedGrammar) -> Result<(), CompileDecline> {
    for rule in &g.rules {
        if rule.rhs.is_empty() || rule.rhs.len() > 3 {
            return Err(CompileDecline::StrategyUnsupported {
                stage: Stage::Certify,
                reason: format!("rule {rule:?} has RHS length {}", rule.rhs.len()),
            });
        }
        match rule.rhs.last() {
            Some(Symbol::Terminal(_)) => {}
            _ => {
                return Err(CompileDecline::StrategyUnsupported {
                    stage: Stage::Certify,
                    reason: format!("rule {rule:?} does not end in a terminal"),
                });
            }
        }
    }

    // Every retained rule must be productive and reachable from the entry.
    if !output_pruning_complete(&g.rules, g.start) {
        return Err(CompileDecline::StrategyUnsupported {
            stage: Stage::Certify,
            reason: "output grammar is not fully productive/reachable".to_owned(),
        });
    }
    Ok(())
}

// ── §3.1 preprocessing ──────────────────────────────────────────────────────

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
    let (collapsed, collapsed_start) = collapse_unit_sccs(rules, start_dense)?;
    rules = collapsed;
    start_dense = collapsed_start;
    rules = dedup_rules(rules);
    rules.retain(|rule| !is_reflexive_unit(rule));

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
        .max(start_dense as usize + 1);

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
            num_nts,
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
fn collapse_unit_sccs(rules: Vec<Rule>, start: NtIx) -> Result<(Vec<Rule>, NtIx), CompileDecline> {
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

    let mut unit_adj = vec![BitSet::new(num_nts); num_nts];
    for rule in &rules {
        if let [Symbol::Nonterminal(right)] = rule.rhs.as_slice() {
            unit_adj[rule.lhs as usize].set(*right as usize);
        }
    }
    let unit_closure = transitive_closure(&unit_adj, num_nts);

    let mut representative: Vec<NtIx> = (0..num_nts as NtIx).collect();
    for i in 0..num_nts {
        for j in (i + 1)..num_nts {
            if unit_closure[i].contains(j) && unit_closure[j].contains(i) {
                let minimum = i as NtIx;
                representative[i] = representative[i].min(minimum);
                representative[j] = representative[j].min(minimum);
            }
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

/// Reflexive-transitive closure of a bitset adjacency matrix.
fn transitive_closure(adj: &[BitSet], n: usize) -> Vec<BitSet> {
    let mut result = Vec::with_capacity(n);
    for source in 0..n {
        let mut row = BitSet::new(n);
        row.set(source);
        let mut frontier = vec![source];
        while let Some(node) = frontier.pop() {
            for target in adj[node].iter_ones() {
                if !row.contains(target) {
                    row.set(target);
                    frontier.push(target);
                }
            }
        }
        result.push(row);
    }
    result
}

// ── §3.2 closures ───────────────────────────────────────────────────────────

struct Closures {
    num_nts: usize,
    /// `U`: unit-only paths, including length zero.
    unit: Vec<BitSet>,
    /// `R`: unit or binary-right-spine paths, including length zero.
    right: Vec<BitSet>,
    /// `V`: paths containing at least one binary edge.
    positive: Vec<BitSet>,
}

impl Closures {
    fn compute(
        grammar: &BinaryGrammar,
        limits: &CompileLimits,
        budget: &mut CompileBudget,
    ) -> Result<Self, CompileDecline> {
        let n = grammar.num_nts;
        budget.charge_candidate_work(Stage::Closures, limits, (n as u64) * 2)?;

        let mut unit_adj = vec![BitSet::new(n); n];
        for &(lhs, rhs) in &grammar.unit_rules {
            unit_adj[lhs as usize].set(rhs as usize);
        }
        let unit = transitive_closure(&unit_adj, n);

        let mut right_adj = vec![BitSet::new(n); n];
        let mut bin_right = vec![BitSet::new(n); n];
        for &(lhs, right) in &grammar.unit_rules {
            right_adj[lhs as usize].set(right as usize);
        }
        for &(lhs, _left, right) in &grammar.binary_rules {
            right_adj[lhs as usize].set(right as usize);
            bin_right[lhs as usize].set(right as usize);
        }
        let right = transitive_closure(&right_adj, n);

        let mut positive = vec![BitSet::new(n); n];
        for a in 0..n {
            let mut tails = BitSet::new(n);
            for d in right[a].iter_ones() {
                tails.union_with(&bin_right[d]);
            }
            for e in tails.iter_ones() {
                positive[a].union_with(&right[e]);
            }
        }

        Ok(Self {
            num_nts: n,
            unit,
            right,
            positive,
        })
    }

    fn choices(&self, a: NtIx, c: NtIx) -> SmallVec<[Fragment; 2]> {
        let mut out = SmallVec::new();
        if self.unit[a as usize].contains(c as usize) {
            out.push(Fragment::Empty);
        }
        if self.positive[a as usize].contains(c as usize) {
            out.push(Fragment::One(PairKey { from: a, to: c }));
        }
        out
    }
}

// ── §3.3–3.5 construction ───────────────────────────────────────────────────

struct Builder {
    rows: BTreeMap<Name, Vec<Rhs>>,
    seen: BTreeSet<(Name, Rhs)>,
    demanded: BTreeSet<PairKey>,
    expanded: BTreeSet<PairKey>,
    queue: VecDeque<PairKey>,
}

impl Builder {
    fn new() -> Self {
        Self {
            rows: BTreeMap::new(),
            seen: BTreeSet::new(),
            demanded: BTreeSet::new(),
            expanded: BTreeSet::new(),
            queue: VecDeque::new(),
        }
    }

    fn emit(
        &mut self,
        name: Name,
        rhs: Rhs,
        limits: &CompileLimits,
        budget: &mut CompileBudget,
    ) -> Result<(), CompileDecline> {
        // Charge candidate work even when the candidate is later deduplicated.
        budget.charge_candidate_work(Stage::Construct, limits, 1)?;
        if self.seen.insert((name, rhs.clone())) {
            budget.charge_retained_rule(Stage::Construct, limits)?;
            budget.charge_symbols(Stage::Construct, limits, rhs.len() as u64)?;
            for symbol in &rhs {
                if let Sym::Nt(Name::Pair(pair)) = symbol
                    && self.demanded.insert(*pair)
                {
                    self.queue.push_back(*pair);
                }
            }
            self.rows.entry(name).or_default().push(rhs);
        }
        Ok(())
    }

    fn next_unexpanded_pair(&mut self) -> Option<PairKey> {
        while let Some(pair) = self.queue.pop_front() {
            if self.expanded.insert(pair) {
                return Some(pair);
            }
        }
        None
    }
}

fn fragment_symbol(fragment: Fragment) -> Option<Sym> {
    match fragment {
        Fragment::Empty => None,
        Fragment::One(pair) => Some(Sym::Nt(Name::Pair(pair))),
    }
}

fn entry_rhs(fragment: Fragment, terminal: TerminalID) -> Rhs {
    let mut rhs = Rhs::new();
    if let Some(symbol) = fragment_symbol(fragment) {
        rhs.push(symbol);
    }
    rhs.push(Sym::Terminal(terminal));
    rhs
}

fn context_rhs(first: Fragment, second: Fragment, terminal: TerminalID) -> Rhs {
    let mut rhs = Rhs::new();
    if let Some(symbol) = fragment_symbol(first) {
        rhs.push(symbol);
    }
    if let Some(symbol) = fragment_symbol(second) {
        rhs.push(symbol);
    }
    rhs.push(Sym::Terminal(terminal));
    rhs
}

fn construct(
    preprocessed: &Preprocessed,
    closures: &Closures,
    accepts_empty: bool,
    source: &GrammarDef,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<(PreparedGrammar, TerminalEndedStats), CompileDecline> {
    let grammar = &preprocessed.grammar;
    let n = grammar.num_nts;

    // terminal_exits[B] = (T, a) with T -> a and R[B, T].
    let mut terminal_exits: Vec<Vec<(NtIx, TerminalID)>> = vec![Vec::new(); n];
    for &(lhs, terminal) in &grammar.terminal_rules {
        for b in 0..n {
            if closures.right[b].contains(lhs as usize) {
                terminal_exits[b].push((lhs, terminal));
            }
        }
    }
    for exits in &mut terminal_exits {
        exits.sort_unstable();
        exits.dedup();
    }

    // last_binary[C] = (D, B, E) with D -> B E and U[E, C].
    let mut last_binary: Vec<Vec<(NtIx, NtIx, NtIx)>> = vec![Vec::new(); n];
    for &(d, b, e) in &grammar.binary_rules {
        for c in 0..n {
            if closures.unit[e as usize].contains(c as usize) {
                last_binary[c].push((d, b, e));
            }
        }
    }
    for binaries in &mut last_binary {
        binaries.sort_unstable();
        binaries.dedup();
    }

    let mut builder = Builder::new();

    // Entry rows are processed in ascending public id order. For a single-start
    // grammar this is exactly the original start.
    let start = preprocessed.start;
    for &(t, terminal) in &terminal_exits[start as usize] {
        for fragment in closures.choices(start, t) {
            builder.emit(
                Name::Entry(source.start),
                entry_rhs(fragment, terminal),
                limits,
                budget,
            )?;
        }
    }

    // Demand-driven context expansion. Only referenced pairs get rows.
    while let Some(pair) = builder.next_unexpanded_pair() {
        let a = pair.from;
        let c = pair.to;
        debug_assert!(closures.positive[a as usize].contains(c as usize));
        for &(d, b, _e) in &last_binary[c as usize] {
            if !closures.right[a as usize].contains(d as usize) {
                continue;
            }
            for &(t, terminal) in &terminal_exits[b as usize] {
                for first in closures.choices(a, d) {
                    for second in closures.choices(b, t) {
                        let rhs = context_rhs(first, second, terminal);
                        debug_assert!((1..=3).contains(&rhs.len()));
                        builder.emit(Name::Pair(pair), rhs, limits, budget)?;
                    }
                }
            }
        }
    }

    let stats = TerminalEndedStats {
        demanded_pairs: builder.demanded.len(),
        ..TerminalEndedStats::default()
    };
    let prepared = assemble(source, accepts_empty, builder, limits, budget)?;
    Ok((prepared, stats))
}

/// Assign deterministic ids and materialize the final rule vector.
///
/// Discovery used symbolic `PairKey`s; after discovery, demanded pairs are
/// sorted by stable dense indices and assigned a contiguous fresh-id range.
/// Entry rows come first in ascending public-id order, then helper rows in pair
/// order.
fn assemble(
    source: &GrammarDef,
    accepts_empty: bool,
    builder: Builder,
    limits: &CompileLimits,
    budget: &mut CompileBudget,
) -> Result<PreparedGrammar, CompileDecline> {
    let max_source_nt = source
        .rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .chain(std::iter::once(source.start))
        .max()
        .unwrap_or(source.start);

    let mut pairs: Vec<PairKey> = builder.demanded.iter().copied().collect();
    pairs.sort_unstable();

    let mut cursor: u64 = max_source_nt as u64 + 1;
    let mut id_of = BTreeMap::new();
    for pair in &pairs {
        if cursor > u32::MAX as u64 {
            return Err(CompileDecline::ResourceLimit {
                stage: Stage::Construct,
                kind: ResourceKind::FreshIds,
                used: cursor,
                limit: u32::MAX as u64,
            });
        }
        id_of.insert(Name::Pair(*pair), cursor as NonterminalID);
        cursor += 1;
    }

    let convert = |symbol: &Sym| -> Symbol {
        match symbol {
            Sym::Nt(name) => {
                let id = match name {
                    Name::Entry(public) => *public,
                    Name::Pair(_) => id_of[name],
                };
                Symbol::Nonterminal(id)
            }
            Sym::Terminal(terminal) => Symbol::Terminal(*terminal),
        }
    };

    let mut rules: Vec<Rule> = Vec::new();
    // Entry rows, ascending public id.
    for (&name, rows) in &builder.rows {
        let Name::Entry(public) = name else {
            continue;
        };
        for rhs in rows {
            budget.charge_retained_rule(Stage::Construct, limits)?;
            rules.push(Rule {
                lhs: public,
                rhs: rhs.iter().map(convert).collect(),
            });
        }
    }
    // Helper rows, pair order.
    for pair in &pairs {
        let name = Name::Pair(*pair);
        let Some(rows) = builder.rows.get(&name) else {
            continue;
        };
        let lhs = id_of[&name];
        for rhs in rows {
            budget.charge_retained_rule(Stage::Construct, limits)?;
            rules.push(Rule {
                lhs,
                rhs: rhs.iter().map(convert).collect(),
            });
        }
    }

    prune_output(&mut rules, source.start);

    Ok(PreparedGrammar {
        rules,
        start: source.start,
        accepts_empty,
    })
}

/// Remove nonproductive and unreachable rules from the materialized output.
/// Nonterminal ids may be sparse (entry public ids plus fresh helper ids), so
/// the analysis is index-mapped rather than array-indexed.
fn prune_output(rules: &mut Vec<Rule>, start: NonterminalID) {
    let mut ids = BTreeSet::new();
    ids.insert(start);
    for rule in rules.iter() {
        ids.insert(rule.lhs);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(nonterminal) = symbol {
                ids.insert(*nonterminal);
            }
        }
    }
    let index_of: BTreeMap<NonterminalID, usize> =
        ids.iter().enumerate().map(|(index, id)| (*id, index)).collect();
    let count = ids.len();

    let mut productive = vec![false; count];
    loop {
        let mut changed = false;
        for rule in rules.iter() {
            let lhs = index_of[&rule.lhs];
            if productive[lhs] {
                continue;
            }
            let all = rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::Nonterminal(nonterminal) => productive[index_of[nonterminal]],
            });
            if all {
                productive[lhs] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    rules.retain(|rule| {
        rule.rhs.iter().all(|symbol| match symbol {
            Symbol::Terminal(_) => true,
            Symbol::Nonterminal(nonterminal) => productive[index_of[nonterminal]],
        })
    });

    let mut reachable = vec![false; count];
    reachable[index_of[&start]] = true;
    loop {
        let mut changed = false;
        for rule in rules.iter() {
            if !reachable[index_of[&rule.lhs]] {
                continue;
            }
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    let index = index_of[nonterminal];
                    if !reachable[index] {
                        reachable[index] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    rules.retain(|rule| reachable[index_of[&rule.lhs]]);
}

/// Whether the output grammar is already fully productive and reachable.
fn output_pruning_complete(rules: &[Rule], start: NonterminalID) -> bool {
    let mut ids = BTreeSet::new();
    ids.insert(start);
    for rule in rules {
        ids.insert(rule.lhs);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(nonterminal) = symbol {
                ids.insert(*nonterminal);
            }
        }
    }
    let index_of: BTreeMap<NonterminalID, usize> =
        ids.iter().enumerate().map(|(index, id)| (*id, index)).collect();
    let count = ids.len();

    let mut productive = vec![false; count];
    loop {
        let mut changed = false;
        for rule in rules {
            let lhs = index_of[&rule.lhs];
            if productive[lhs] {
                continue;
            }
            if rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::Nonterminal(nonterminal) => productive[index_of[nonterminal]],
            }) {
                productive[lhs] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut reachable = vec![false; count];
    reachable[index_of[&start]] = true;
    loop {
        let mut changed = false;
        for rule in rules {
            if !reachable[index_of[&rule.lhs]] {
                continue;
            }
            for symbol in &rule.rhs {
                if let Symbol::Nonterminal(nonterminal) = symbol {
                    let index = index_of[nonterminal];
                    if !reachable[index] {
                        reachable[index] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    rules.iter().all(|rule| {
        productive[index_of[&rule.lhs]]
            && reachable[index_of[&rule.lhs]]
            && rule.rhs.iter().all(|symbol| match symbol {
                Symbol::Terminal(_) => true,
                Symbol::Nonterminal(nonterminal) => productive[index_of[nonterminal]],
            })
    })
}

#[cfg(test)]
#[path = "terminal_ended_tests.rs"]
mod tests;
