use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::build_regex;
use crate::automata::lexer::lightweight::nfa::Nfa as LightweightNfa;
use crate::automata::unweighted_u32::dfa::{DFA, Label};
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic;
use crate::automata::unweighted_u32::minimize_cyclic::minimize_cyclic;
use crate::automata::unweighted_u32::nfa::NFA;

use super::ast::GrammarExpr;

/// An NFA whose transition labels are indices into `symbols`.
///
/// This keeps the transition graph compact while allowing each transition
/// symbol to be an arbitrary [`GrammarExpr`]. A transition label is valid when
/// it is non-negative and less than `symbols.len()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExprNFA {
    pub nfa: NFA,
    pub symbols: Vec<GrammarExpr>,
}

impl ExprNFA {
    pub fn new(nfa: NFA, symbols: Vec<GrammarExpr>) -> Self {
        Self { nfa, symbols }
    }

    pub fn into_determinized_and_minimized(self) -> Self {
        let partitioned = self.partition_overlapping_terminal_languages();
        let dfa = partitioned.determinize_and_minimize();
        let symbols = partitioned.symbols;
        let mut nfa = NFA::new_empty();
        for _ in &dfa.states {
            nfa.add_state();
        }
        if !dfa.states.is_empty() {
            nfa.start_states.push(dfa.start_state);
        }
        for (state_id, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                nfa.set_accepting(state_id as u32);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(state_id as u32, label, target);
            }
        }
        Self::new(nfa, symbols)
    }

    pub fn determinize(&self) -> DFA {
        determinize_nfa(&self.nfa)
    }

    pub fn determinize_and_minimize(&self) -> DFA {
        minimize_dfa(&self.determinize())
    }

    pub fn symbol_for_label(&self, label: Label) -> Option<&GrammarExpr> {
        usize::try_from(label).ok().and_then(|index| self.symbols.get(index))
    }

    pub fn partition_overlapping_terminal_languages(&self) -> Self {
        let terminal_partitions = self.global_terminal_language_partitions();
        let mut builder = ExprNfaBuilder::new();
        for _ in 1..self.nfa.states.len() {
            builder.add_state();
        }
        for &start in &self.nfa.start_states {
            builder.add_start_state(start);
        }

        for (state_id, state) in self.nfa.states.iter().enumerate() {
            if state.is_accepting {
                builder.set_accepting(state_id as u32);
            }
            for &target in &state.epsilons {
                builder.add_epsilon(state_id as u32, target);
            }

            for (&label, targets) in &state.transitions {
                if let Some(partitions) = terminal_partitions.get(&label) {
                    for expr in partitions {
                        let symbol = GrammarExpr::TerminalLanguage(Box::new(expr.clone()));
                        for &target in targets {
                            builder.add_transition(state_id as u32, symbol.clone(), target);
                        }
                    }
                    continue;
                }
                let Some(symbol) = self.symbol_for_label(label).cloned() else {
                    continue;
                };
                for &target in targets {
                    builder.add_transition(state_id as u32, symbol.clone(), target);
                }
            }
        }

        builder.build()
    }

    fn global_terminal_language_partitions(&self) -> HashMap<Label, Vec<Expr>> {
        let mut terminal_symbols = Vec::<(Label, Vec<u32>, LightweightNfa, Expr)>::new();
        for (label, symbol) in self.symbols.iter().enumerate() {
            let label = label
                .try_into()
                .expect("ExprNFA symbol table exceeded i32 labels");
            let GrammarExpr::TerminalLanguage(expr) = symbol else {
                continue;
            };
            let Some(language) = terminal_language_nfa(expr) else {
                continue;
            };
            terminal_symbols.push((label, vec![label as u32], language, (**expr).clone()));
        }

        let Some(partitioned) = partition_terminal_edges(&terminal_symbols) else {
            return HashMap::new();
        };

        let mut out = HashMap::<Label, Vec<Expr>>::new();
        for (expr, labels) in partitioned {
            for label in labels {
                out.entry(label as Label).or_default().push(expr.clone());
            }
        }
        out
    }
}

#[derive(Clone)]
struct PartitionRegion {
    expr: Expr,
    language: LightweightNfa,
    targets: Vec<u32>,
}

fn partition_terminal_edges(
    terminal_edges: &[(Label, Vec<u32>, LightweightNfa, Expr)],
) -> Option<Vec<(Expr, Vec<u32>)>> {
    if terminal_edges.len() < 2 {
        return Some(
            terminal_edges
                .iter()
                .map(|(_, targets, _, expr)| (expr.clone(), targets.clone()))
                .collect(),
        );
    }

    let mut regions: Vec<PartitionRegion> = Vec::new();
    let mut changed = false;

    for (_, edge_targets, edge_language, edge_expr) in terminal_edges {
        let mut next_regions = Vec::new();
        let mut uncovered = Some(PartitionRegion {
            expr: edge_expr.clone(),
            language: edge_language.clone(),
            targets: edge_targets.clone(),
        });

        for region in regions.into_iter() {
            let overlap = region.language.intersect(edge_language);
            if overlap.is_empty() {
                next_regions.push(region);
                continue;
            }
            changed = true;

            let region_only = region.language.subtract(edge_language);
            if !region_only.is_empty() {
                let expr = compose_terminal_difference(
                    &region.expr,
                    &region.language,
                    edge_expr,
                    edge_language,
                )?;
                next_regions.push(PartitionRegion {
                    expr,
                    language: region_only,
                    targets: region.targets.clone(),
                });
            }

            let expr = compose_terminal_intersection(
                &region.expr,
                &region.language,
                edge_expr,
                edge_language,
            )?;
            next_regions.push(PartitionRegion {
                expr,
                language: overlap,
                targets: union_targets(&region.targets, edge_targets),
            });

            let Some(previous_uncovered) = uncovered.take() else {
                continue;
            };
            let uncovered_only = previous_uncovered.language.subtract(&region.language);
            if !uncovered_only.is_empty() {
                let expr = compose_terminal_difference(
                    &previous_uncovered.expr,
                    &previous_uncovered.language,
                    &region.expr,
                    &region.language,
                )?;
                uncovered = Some(PartitionRegion {
                    expr,
                    language: uncovered_only,
                    targets: previous_uncovered.targets,
                });
            }
        }

        if let Some(region) = uncovered {
            next_regions.push(region);
        }
        regions = next_regions;
    }

    if !changed {
        return Some(
            terminal_edges
                .iter()
                .map(|(_, targets, _, expr)| (expr.clone(), targets.clone()))
                .collect(),
        );
    }

    Some(
        regions
            .into_iter()
            .map(|region| (region.expr, region.targets))
            .collect(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SimpleTerminalForm {
    LiteralSet(BTreeSet<Vec<u8>>),
    BaseMinusLiterals {
        base: Expr,
        excluded: BTreeSet<Vec<u8>>,
    },
}

fn literal_set(expr: &Expr) -> Option<BTreeSet<Vec<u8>>> {
    match expr {
        Expr::U8Seq(bytes) => Some(std::iter::once(bytes.clone()).collect()),
        Expr::Choice(options) => {
            let mut out = BTreeSet::new();
            for option in options {
                out.extend(literal_set(option)?);
            }
            Some(out)
        }
        Expr::Shared(inner) => literal_set(inner.as_ref()),
        _ => None,
    }
}

fn classify_simple_terminal_form(expr: &Expr) -> Option<SimpleTerminalForm> {
    if let Some(literals) = literal_set(expr) {
        return Some(SimpleTerminalForm::LiteralSet(literals));
    }

    match expr {
        Expr::Exclude { expr: base, exclude } => Some(SimpleTerminalForm::BaseMinusLiterals {
            base: (**base).clone(),
            excluded: literal_set(exclude)?,
        }),
        Expr::Shared(inner) => classify_simple_terminal_form(inner.as_ref()),
        _ => None,
    }
}

fn expr_from_literal_set(literals: &BTreeSet<Vec<u8>>) -> Option<Expr> {
    (!literals.is_empty()).then(|| {
        Expr::make_choice(
            literals
                .iter()
                .cloned()
                .map(Expr::U8Seq)
                .collect::<Vec<_>>(),
        )
        .optimize()
    })
}

fn expr_from_base_minus_literals(base: &Expr, excluded: &BTreeSet<Vec<u8>>) -> Expr {
    if excluded.is_empty() {
        return base.clone();
    }

    Expr::Exclude {
        expr: Box::new(base.clone()),
        exclude: Box::new(
            expr_from_literal_set(excluded).expect("excluded literal set should be non-empty"),
        ),
    }
    .optimize()
}

fn filter_literals_against_language(
    literals: &BTreeSet<Vec<u8>>,
    language: &LightweightNfa,
    keep_matches: bool,
) -> BTreeSet<Vec<u8>> {
    literals
        .iter()
        .filter(|literal| language.accepts_bytes(literal) == keep_matches)
        .cloned()
        .collect()
}

fn union_literal_sets(left: &BTreeSet<Vec<u8>>, right: &BTreeSet<Vec<u8>>) -> BTreeSet<Vec<u8>> {
    left.union(right).cloned().collect()
}

fn difference_literal_sets(
    left: &BTreeSet<Vec<u8>>,
    right: &BTreeSet<Vec<u8>>,
) -> BTreeSet<Vec<u8>> {
    left.difference(right).cloned().collect()
}

fn compose_terminal_difference(
    expr: &Expr,
    _expr_language: &LightweightNfa,
    exclude: &Expr,
    exclude_language: &LightweightNfa,
) -> Option<Expr> {
    match (
        classify_simple_terminal_form(expr),
        classify_simple_terminal_form(exclude),
    ) {
        (Some(SimpleTerminalForm::LiteralSet(literals)), _) => {
            let remaining = filter_literals_against_language(&literals, exclude_language, false);
            if let Some(composed) = expr_from_literal_set(&remaining) {
                return Some(composed);
            }
        }
        (
            Some(SimpleTerminalForm::BaseMinusLiterals { base, excluded }),
            Some(SimpleTerminalForm::LiteralSet(other_literals)),
        ) => {
            return Some(expr_from_base_minus_literals(
                &base,
                &union_literal_sets(&excluded, &other_literals),
            ));
        }
        (
            Some(SimpleTerminalForm::BaseMinusLiterals {
                base: left_base,
                excluded: left_excluded,
            }),
            Some(SimpleTerminalForm::BaseMinusLiterals {
                base: right_base,
                excluded: right_excluded,
            }),
        ) if left_base == right_base => {
            if let Some(composed) = expr_from_literal_set(&difference_literal_sets(
                &right_excluded,
                &left_excluded,
            )) {
                return Some(composed);
            }
        }
        _ => {}
    }

    let composed = Expr::Exclude {
        expr: Box::new(expr.clone()),
        exclude: Box::new(exclude.clone()),
    }
    .optimize();
    terminal_language_expr_is_compile_safe(&composed).then_some(composed)
}

fn compose_terminal_intersection(
    expr: &Expr,
    expr_language: &LightweightNfa,
    intersect: &Expr,
    intersect_language: &LightweightNfa,
) -> Option<Expr> {
    match (
        classify_simple_terminal_form(expr),
        classify_simple_terminal_form(intersect),
    ) {
        (Some(SimpleTerminalForm::LiteralSet(literals)), _) => {
            if let Some(composed) =
                expr_from_literal_set(&filter_literals_against_language(&literals, intersect_language, true))
            {
                return Some(composed);
            }
        }
        (_, Some(SimpleTerminalForm::LiteralSet(literals))) => {
            if let Some(composed) =
                expr_from_literal_set(&filter_literals_against_language(&literals, expr_language, true))
            {
                return Some(composed);
            }
        }
        (
            Some(SimpleTerminalForm::BaseMinusLiterals {
                base: left_base,
                excluded: left_excluded,
            }),
            Some(SimpleTerminalForm::BaseMinusLiterals {
                base: right_base,
                excluded: right_excluded,
            }),
        ) if left_base == right_base => {
            return Some(expr_from_base_minus_literals(
                &left_base,
                &union_literal_sets(&left_excluded, &right_excluded),
            ));
        }
        _ => {}
    }

    let composed = Expr::Intersect {
        expr: Box::new(expr.clone()),
        intersect: Box::new(intersect.clone()),
    }
    .optimize();
    terminal_language_expr_is_compile_safe(&composed).then_some(composed)
}

fn union_targets(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut out = left.to_vec();
    for &target in right {
        if !out.contains(&target) {
            out.push(target);
        }
    }
    out
}

fn terminal_language_nfa(expr: &crate::automata::lexer::ast::Expr) -> Option<LightweightNfa> {
    if terminal_language_contains_nested_group_ops(expr) {
        return None;
    }
    let regex = build_regex(std::slice::from_ref(expr));
    Some(LightweightNfa::from_minimal_lexer_dfa(&regex.dfa))
}

fn terminal_language_expr_is_compile_safe(expr: &Expr) -> bool {
    let (base, excluded, intersections) = split_top_level_group_ops(expr);
    !terminal_language_contains_any_group_ops(&base)
        && excluded
            .iter()
            .all(|branch| !terminal_language_contains_any_group_ops(branch))
        && intersections
            .iter()
            .all(|branch| !terminal_language_contains_any_group_ops(branch))
}

fn split_top_level_group_ops(expr: &Expr) -> (Expr, Vec<Expr>, Vec<Expr>) {
    match expr {
        Expr::Exclude { expr, exclude } => {
            let (base, mut excluded, intersections) = split_top_level_group_ops(expr);
            excluded.push((**exclude).clone());
            (base, excluded, intersections)
        }
        Expr::Intersect { expr, intersect } => {
            let (base, excluded, mut intersections) = split_top_level_group_ops(expr);
            intersections.push((**intersect).clone());
            (base, excluded, intersections)
        }
        Expr::Shared(inner)
            if matches!(inner.as_ref(), Expr::Exclude { .. } | Expr::Intersect { .. }) =>
        {
            split_top_level_group_ops(inner.as_ref())
        }
        _ => (expr.clone(), Vec::new(), Vec::new()),
    }
}

fn terminal_language_contains_nested_group_ops(expr: &crate::automata::lexer::ast::Expr) -> bool {
    use crate::automata::lexer::ast::Expr;

    fn contains_group_op(expr: &Expr) -> bool {
        match expr {
            Expr::Exclude { .. } | Expr::Intersect { .. } => true,
            Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(contains_group_op),
            Expr::Repeat { expr, .. } => contains_group_op(expr),
            Expr::Shared(expr) => contains_group_op(expr),
            Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
        }
    }

    match expr {
        Expr::Exclude { expr, exclude } => contains_group_op(expr) || contains_group_op(exclude),
        Expr::Intersect { expr, intersect } => {
            contains_group_op(expr) || contains_group_op(intersect)
        }
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(contains_group_op),
        Expr::Repeat { expr, .. } => terminal_language_contains_nested_group_ops(expr),
        Expr::Shared(expr) => terminal_language_contains_nested_group_ops(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
    }
}

fn terminal_language_contains_any_group_ops(expr: &crate::automata::lexer::ast::Expr) -> bool {
    use crate::automata::lexer::ast::Expr;

    match expr {
        Expr::Exclude { .. } | Expr::Intersect { .. } => true,
        Expr::Seq(parts) | Expr::Choice(parts) => parts.iter().any(terminal_language_contains_any_group_ops),
        Expr::Repeat { expr, .. } => terminal_language_contains_any_group_ops(expr),
        Expr::Shared(expr) => terminal_language_contains_any_group_ops(expr),
        Expr::U8Seq(_) | Expr::U8Class(_) | Expr::Epsilon => false,
    }
}

fn terminal_language_is_empty(language: &LightweightNfa) -> bool {
    language.is_empty()
}

/// Builder for an [`ExprNFA`] through an intermediate NFA.
///
/// Transitions are labeled by arbitrary [`GrammarExpr`] symbols. Equal symbols
/// are automatically interned to the same label, so callers can construct paths
/// directly without managing the side table by hand.
#[derive(Debug, Clone)]
pub struct ExprNfaBuilder {
    nfa: NFA,
    symbols: Vec<GrammarExpr>,
    symbol_labels: HashMap<GrammarExpr, Label>,
}

impl Default for ExprNfaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprNfaBuilder {
    pub fn new() -> Self {
        Self {
            nfa: NFA::new(),
            symbols: Vec::new(),
            symbol_labels: HashMap::new(),
        }
    }

    pub fn add_state(&mut self) -> u32 {
        self.nfa.add_state()
    }

    pub fn start_state(&self) -> u32 {
        self.nfa.start_states.first().copied().unwrap_or(0)
    }

    pub fn add_start_state(&mut self, state: u32) {
        if !self.nfa.start_states.contains(&state) {
            self.nfa.start_states.push(state);
        }
    }

    pub fn set_accepting(&mut self, state: u32) {
        self.nfa.set_accepting(state);
    }

    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        self.nfa.add_epsilon(from, to);
    }

    pub fn add_symbol(&mut self, symbol: GrammarExpr) -> Label {
        if let Some(&label) = self.symbol_labels.get(&symbol) {
            return label;
        }
        let label = i32::try_from(self.symbols.len())
            .expect("ExprNFA symbol table exceeded i32 labels");
        self.symbols.push(symbol.clone());
        self.symbol_labels.insert(symbol, label);
        label
    }

    pub fn add_transition(&mut self, from: u32, symbol: GrammarExpr, to: u32) -> Label {
        let label = self.add_symbol(symbol);
        self.add_labeled_transition(from, label, to);
        label
    }

    pub fn add_labeled_transition(&mut self, from: u32, label: Label, to: u32) {
        self.nfa.add_transition(from, label, to);
    }

    pub fn into_nfa_and_symbols(self) -> (NFA, Vec<GrammarExpr>) {
        (self.nfa, self.symbols)
    }

    pub fn build(self) -> ExprNFA {
        let (nfa, symbols) = self.into_nfa_and_symbols();
        ExprNFA::new(nfa, symbols)
    }
}

pub fn minimize_dfa(dfa: &DFA) -> DFA {
    if dfa.is_acyclic() {
        minimize_acyclic(dfa)
    } else {
        minimize_cyclic(dfa)
    }
}

fn subset_is_accepting(nfa: &NFA, subset: &[u32]) -> bool {
    subset.iter().any(|&state| nfa.states[state as usize].is_accepting)
}

fn epsilon_closure(nfa: &NFA, seeds: &[u32]) -> BTreeSet<u32> {
    let mut closed = BTreeSet::new();
    let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
    while let Some(state) = queue.pop_front() {
        if !closed.insert(state) {
            continue;
        }
        let Some(nfa_state) = nfa.states.get(state as usize) else {
            continue;
        };
        for &target in &nfa_state.epsilons {
            if !closed.contains(&target) {
                queue.push_back(target);
            }
        }
    }
    closed
}

fn gather_label_targets(nfa: &NFA, subset: &[u32]) -> BTreeMap<Label, BTreeSet<u32>> {
    let mut label_targets = BTreeMap::<Label, BTreeSet<u32>>::new();
    for &state in subset {
        let Some(nfa_state) = nfa.states.get(state as usize) else {
            continue;
        };
        for (&label, targets) in &nfa_state.transitions {
            label_targets
                .entry(label)
                .or_default()
                .extend(targets.iter().copied());
        }
    }
    label_targets
}

fn get_or_create_subset_state(
    dfa: &mut DFA,
    subset_map: &mut HashMap<Vec<u32>, u32>,
    worklist: &mut VecDeque<Vec<u32>>,
    subset: Vec<u32>,
) -> u32 {
    if let Some(&state) = subset_map.get(&subset) {
        return state;
    }
    let state = dfa.add_state();
    subset_map.insert(subset.clone(), state);
    worklist.push_back(subset);
    state
}

pub fn determinize_nfa(nfa: &NFA) -> DFA {
    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }

    let mut dfa = DFA {
        states: Vec::new(),
        start_state: 0,
    };
    let mut subset_map = HashMap::<Vec<u32>, u32>::new();
    let mut worklist = VecDeque::<Vec<u32>>::new();

    let start_closure = epsilon_closure(nfa, &nfa.start_states);
    let start_key = start_closure.iter().copied().collect::<Vec<_>>();
    let start_id = dfa.add_state();
    dfa.start_state = start_id;
    subset_map.insert(start_key.clone(), start_id);
    worklist.push_back(start_key);

    while let Some(subset_key) = worklist.pop_front() {
        let dfa_state = subset_map[&subset_key];
        if subset_is_accepting(nfa, &subset_key) {
            dfa.set_accepting(dfa_state, true);
        }

        for (label, raw_targets) in gather_label_targets(nfa, &subset_key) {
            let seeds = raw_targets.iter().copied().collect::<Vec<_>>();
            let next_key = epsilon_closure(nfa, &seeds).into_iter().collect::<Vec<_>>();
            if next_key.is_empty() {
                continue;
            }
            let next_state =
                get_or_create_subset_state(&mut dfa, &mut subset_map, &mut worklist, next_key);
            dfa.add_transition(dfa_state, label, next_state);
        }
    }

    dfa
}

#[cfg(test)]
mod tests {
    use crate::automata::lexer::ast::Expr;
    use crate::grammar::ast::{lower, NamedGrammar, NamedRule};
    use crate::grammar::flat::Symbol;

    use super::*;

    #[test]
    fn lowers_expr_nfa_transition_symbols() {
        let mut nfa = NFA::new();
        let accept = nfa.add_state();
        nfa.add_transition(0, 0, accept);
        nfa.set_accepting(accept);

        let grammar = NamedGrammar {
            rules: vec![NamedRule {
                name: "start".into(),
                expr: GrammarExpr::ExprNFA(Box::new(ExprNFA::new(nfa, vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                ]))),
                is_terminal: false,
                is_internal: false,
            }],
            start: "start".into(),
            ignore: None,
        };

        let lowered = lower(&grammar).expect("ExprNFA should lower");
        assert_eq!(lowered.terminals.len(), 1);
        assert!(lowered
            .rules
            .iter()
            .any(|rule| matches!(rule.rhs.as_slice(), [Symbol::Nonterminal(_), Symbol::Terminal(_)])));
    }

    #[test]
    fn builder_preserves_nfa_and_exposes_determinize_minimize() {
        let mut builder = ExprNfaBuilder::new();
        let start = builder.start_state();
        let loop_state = builder.add_state();
        let accept = builder.add_state();

        builder.add_epsilon(start, loop_state);
        builder.add_transition(loop_state, GrammarExpr::Literal(b"a".to_vec()), loop_state);
        builder.add_transition(loop_state, GrammarExpr::Literal(b"b".to_vec()), accept);
        builder.set_accepting(accept);

        let expr_nfa = builder.build();
        assert_eq!(expr_nfa.symbols.len(), 2);
        assert_eq!(expr_nfa.nfa.states[start as usize].epsilons, vec![loop_state]);

        let dfa = expr_nfa.determinize_and_minimize();
        assert!(dfa.states[dfa.start_state as usize]
            .transitions
            .values()
            .any(|&target| target == dfa.start_state));
        assert!(dfa.states.iter().any(|state| state.is_accepting));

        let minimized_expr_nfa = expr_nfa.into_determinized_and_minimized();
        assert_eq!(minimized_expr_nfa.symbols.len(), 2);
        assert!(minimized_expr_nfa
            .nfa
            .states
            .iter()
            .all(|state| state.epsilons.is_empty()));
    }

    #[test]
    fn lowers_terminal_language_symbols() {
        let grammar = NamedGrammar {
            rules: vec![NamedRule {
                name: "start".into(),
                expr: GrammarExpr::TerminalLanguage(Box::new(Expr::U8Seq(b"ab".to_vec()))),
                is_terminal: false,
                is_internal: false,
            }],
            start: "start".into(),
            ignore: None,
        };

        let lowered = lower(&grammar).expect("TerminalLanguage should lower");
        assert_eq!(lowered.terminals.len(), 1);
        assert!(lowered
            .rules
            .iter()
            .any(|rule| matches!(rule.rhs.as_slice(), [Symbol::Terminal(_)])));
    }

    #[test]
    fn partitions_two_overlapping_plain_terminal_languages() {
        let mut builder = ExprNfaBuilder::new();
        let accept_left = builder.add_state();
        let accept_right = builder.add_state();
        builder.set_accepting(accept_left);
        builder.set_accepting(accept_right);
        builder.add_transition(
            builder.start_state(),
            GrammarExpr::TerminalLanguage(Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::U8Seq(b"b".to_vec()),
            ]))),
            accept_left,
        );
        builder.add_transition(
            builder.start_state(),
            GrammarExpr::TerminalLanguage(Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"b".to_vec()),
                Expr::U8Seq(b"c".to_vec()),
            ]))),
            accept_right,
        );

        let partitioned = builder.build().partition_overlapping_terminal_languages();
        assert_eq!(partitioned.symbols.len(), 3);

        let start = &partitioned.nfa.states[partitioned.nfa.start_states[0] as usize];
        assert_eq!(start.transitions.len(), 3);
        let overlap_targets = start
            .transitions
            .iter()
            .find_map(|(_, targets)| (targets.len() == 2).then_some(targets.clone()))
            .expect("expected overlap transition");
        assert!(overlap_targets.contains(&accept_left));
        assert!(overlap_targets.contains(&accept_right));

        let terminal_symbols = start
            .transitions
            .keys()
            .filter_map(|label| partitioned.symbol_for_label(*label))
            .filter_map(|symbol| match symbol {
                GrammarExpr::TerminalLanguage(expr) => Some(expr.as_ref().clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let compiled = terminal_symbols
            .iter()
            .map(|expr| terminal_language_nfa(expr).expect("partitioned expr should compile"))
            .collect::<Vec<_>>();

        for left in 0..compiled.len() {
            for right in (left + 1)..compiled.len() {
                assert!(
                    compiled[left].intersect(&compiled[right]).is_empty(),
                    "partitioned regions should be disjoint"
                );
            }
        }
    }

    #[test]
    fn partitions_ap_like_terminal_languages_with_literal_exclusions() {
        let mut builder = ExprNfaBuilder::new();
        let accept_generic = builder.add_state();
        let accept_subset = builder.add_state();
        let accept_superset = builder.add_state();
        builder.set_accepting(accept_generic);
        builder.set_accepting(accept_subset);
        builder.set_accepting(accept_superset);

        builder.add_transition(
            builder.start_state(),
            GrammarExpr::TerminalLanguage(Box::new(Expr::Exclude {
                expr: Box::new(Expr::Choice(vec![
                    Expr::U8Seq(b"a\"".to_vec()),
                    Expr::U8Seq(b"b\"".to_vec()),
                    Expr::U8Seq(b"c\"".to_vec()),
                    Expr::U8Seq(b"d\"".to_vec()),
                    Expr::U8Seq(b"e\"".to_vec()),
                ])),
                exclude: Box::new(Expr::Choice(vec![
                    Expr::U8Seq(b"b\"".to_vec()),
                    Expr::U8Seq(b"c\"".to_vec()),
                    Expr::U8Seq(b"d\"".to_vec()),
                ])),
            })),
            accept_generic,
        );
        builder.add_transition(
            builder.start_state(),
            GrammarExpr::TerminalLanguage(Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
            ]))),
            accept_subset,
        );
        builder.add_transition(
            builder.start_state(),
            GrammarExpr::TerminalLanguage(Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"b\"".to_vec()),
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
            ]))),
            accept_superset,
        );

        let partitioned = builder.build().partition_overlapping_terminal_languages();
        let start = &partitioned.nfa.states[partitioned.nfa.start_states[0] as usize];
        let terminal_symbols = start
            .transitions
            .keys()
            .filter_map(|label| partitioned.symbol_for_label(*label))
            .filter_map(|symbol| match symbol {
                GrammarExpr::TerminalLanguage(expr) => Some(expr.as_ref().clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(terminal_symbols.iter().any(|expr| matches!(expr, Expr::Exclude { .. })));
        assert!(terminal_symbols.iter().any(|expr| matches!(expr, Expr::Choice(_) | Expr::U8Seq(_))));

        let compiled = terminal_symbols
            .iter()
            .map(|expr| terminal_language_nfa(expr).expect("partitioned expr should compile"))
            .collect::<Vec<_>>();

        for left in 0..compiled.len() {
            for right in (left + 1)..compiled.len() {
                assert!(
                    compiled[left].intersect(&compiled[right]).is_empty(),
                    "partitioned regions should be disjoint"
                );
            }
        }
    }

    #[test]
    fn same_base_exclusion_difference_can_collapse_to_literal_set() {
        let left = Expr::Exclude {
            expr: Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"a\"".to_vec()),
                Expr::U8Seq(b"b\"".to_vec()),
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
                Expr::U8Seq(b"e\"".to_vec()),
            ])),
            exclude: Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
            ])),
        }
        .optimize();
        let right = Expr::Exclude {
            expr: Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"a\"".to_vec()),
                Expr::U8Seq(b"b\"".to_vec()),
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
                Expr::U8Seq(b"e\"".to_vec()),
            ])),
            exclude: Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"b\"".to_vec()),
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
            ])),
        }
        .optimize();

        let left_language = terminal_language_nfa(&left).expect("left should compile");
        let right_language = terminal_language_nfa(&right).expect("right should compile");
        let diff = compose_terminal_difference(&left, &left_language, &right, &right_language)
            .expect("same-base difference should compose");

        assert!(matches!(diff, Expr::U8Seq(_) | Expr::Choice(_)));
        let diff_language = terminal_language_nfa(&diff).expect("diff should compile");
        assert!(diff_language.accepts_bytes(b"b\""));
        assert!(!diff_language.accepts_bytes(b"a\""));
        assert!(!diff_language.accepts_bytes(b"c\""));
        assert!(!diff_language.accepts_bytes(b"d\""));
        assert!(!diff_language.accepts_bytes(b"e\""));
    }

    #[test]
    fn literal_set_difference_against_exclusion_uses_membership_filter() {
        let literal_set = Expr::Choice(vec![
            Expr::U8Seq(b"b\"".to_vec()),
            Expr::U8Seq(b"c\"".to_vec()),
            Expr::U8Seq(b"d\"".to_vec()),
        ])
        .optimize();
        let broad_minus_literals = Expr::Exclude {
            expr: Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"a\"".to_vec()),
                Expr::U8Seq(b"b\"".to_vec()),
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
                Expr::U8Seq(b"e\"".to_vec()),
            ])),
            exclude: Box::new(Expr::Choice(vec![
                Expr::U8Seq(b"b\"".to_vec()),
                Expr::U8Seq(b"c\"".to_vec()),
                Expr::U8Seq(b"d\"".to_vec()),
            ])),
        }
        .optimize();

        let literal_language = terminal_language_nfa(&literal_set).expect("literal set should compile");
        let broad_language = terminal_language_nfa(&broad_minus_literals)
            .expect("broad exclusion should compile");
        let diff = compose_terminal_difference(
            &literal_set,
            &literal_language,
            &broad_minus_literals,
            &broad_language,
        )
        .expect("literal-set difference should compose");

        assert!(matches!(diff, Expr::U8Seq(_) | Expr::Choice(_)));
        let diff_language = terminal_language_nfa(&diff).expect("diff should compile");
        assert!(diff_language.accepts_bytes(b"b\""));
        assert!(diff_language.accepts_bytes(b"c\""));
        assert!(diff_language.accepts_bytes(b"d\""));
        assert!(!diff_language.accepts_bytes(b"a\""));
        assert!(!diff_language.accepts_bytes(b"e\""));
    }
}
