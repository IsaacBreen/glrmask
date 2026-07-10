use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::grammar::ast::{GrammarExpr, Quantifier, NamedGrammar, NamedRule};

/// Promote literal alternatives from nonterminal choices into generated terminal rules
/// when doing so provably reduces the number of terminal definitions after lowering.
///
/// By default, only direct `GrammarExpr::Literal` alternatives are eligible.  When
/// `include_non_literal_terminals` is true, direct inline terminal expressions such
/// as character classes, raw regexes, and any-byte alternatives are eligible too.
/// Named terminal refs are intentionally not eligible here: the referenced terminal
/// rule already exists, so wrapping it in a generated terminal does not remove that
/// original terminal definition.
pub fn promote_choice_terminals_exact(
    grammar: &mut NamedGrammar,
    include_non_literal_terminals: bool,
) -> PromotionStats {
    let mut collector = CandidateCollector::new(include_non_literal_terminals);
    collector.collect(grammar);

    if collector.candidates.is_empty() {
        return PromotionStats::default();
    }

    let selected = solve_exact(&collector.atom_total_counts, &collector.candidates);
    if selected.is_empty() {
        return PromotionStats::default();
    }

    let existing_names = grammar
        .rules
        .iter()
        .map(|rule| rule.name.clone())
        .collect::<BTreeSet<_>>();
    let mut name_generator = TerminalNameGenerator::new(existing_names);
    let mut new_rules = Vec::new();

    for candidate_idx in selected.iter().copied() {
        let name = name_generator.next();
        let candidate = &collector.candidates[candidate_idx];
        new_rules.push(NamedRule {
            name: name.clone(),
            expr: GrammarExpr::Choice(candidate.options.clone()),
            is_terminal: true,
            is_internal: false,
        });
        replace_candidate(grammar, candidate, &name);
    }

    let stats = PromotionStats {
        promoted_choices: selected.len(),
        generated_terminals: new_rules.len(),
        baseline_terminal_atoms: collector.atom_total_counts.len(),
        optimized_terminal_atoms: promoted_cost(&collector.atom_total_counts, &collector.candidates, &selected),
    };
    grammar.rules.extend(new_rules);
    stats
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromotionStats {
    pub promoted_choices: usize,
    pub generated_terminals: usize,
    pub baseline_terminal_atoms: usize,
    pub optimized_terminal_atoms: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PathStep {
    Sequence(usize),
    Choice(usize),
    Optional,
    Repeat,
    RepeatOne,
    RepeatRange,
    ExcludeExpr,
    ExcludeExclude,
    IntersectExpr,
    IntersectIntersect,
    SeparatedItem(usize),
    SeparatedSeparator,
    ExprNFASymbol(usize),
}

#[derive(Debug, Clone)]
struct Candidate {
    rule_idx: usize,
    path: Vec<PathStep>,
    option_indices: Vec<usize>,
    options: Vec<GrammarExpr>,
    atom_counts: Vec<(usize, usize)>,
    potential_saving: usize,
}

struct CandidateCollector {
    include_non_literal_terminals: bool,
    atom_ids: HashMap<GrammarExpr, usize>,
    atom_total_counts: Vec<usize>,
    candidates: Vec<Candidate>,
}

impl CandidateCollector {
    fn new(include_non_literal_terminals: bool) -> Self {
        Self {
            include_non_literal_terminals,
            atom_ids: HashMap::new(),
            atom_total_counts: Vec::new(),
            candidates: Vec::new(),
        }
    }

    fn collect(&mut self, grammar: &NamedGrammar) {
        for (rule_idx, rule) in grammar.rules.iter().enumerate() {
            if rule.is_terminal {
                continue;
            }
            self.collect_expr(rule_idx, &rule.expr, &mut Vec::new());
        }

        self.candidates.retain(|candidate| candidate.atom_counts.len() >= 2);
        self.candidates.sort_by(|a, b| {
            b.potential_saving
                .cmp(&a.potential_saving)
                .then_with(|| a.rule_idx.cmp(&b.rule_idx))
                .then_with(|| a.path.cmp(&b.path))
        });
    }

    fn collect_expr(&mut self, rule_idx: usize, expr: &GrammarExpr, path: &mut Vec<PathStep>) {
        match expr {
            GrammarExpr::Grouped(inner) => {
                self.collect_expr(rule_idx, inner, path);
            }
            GrammarExpr::Choice(options) => {
                let mut option_indices = Vec::new();
                let mut promoted_options = Vec::new();
                let mut atom_counts = BTreeMap::<usize, usize>::new();

                for (idx, option) in options.iter().enumerate() {
                    if let Some(atom) = self.eligible_atom(option) {
                        let atom_id = self.atom_id(atom);
                        self.atom_total_counts[atom_id] += 1;
                        option_indices.push(idx);
                        promoted_options.push(option.clone());
                        *atom_counts.entry(atom_id).or_insert(0) += 1;
                    } else {
                        path.push(PathStep::Choice(idx));
                        self.collect_expr(rule_idx, option, path);
                        path.pop();
                    }
                }

                if option_indices.len() >= 2 {
                    let atom_counts = atom_counts.into_iter().collect::<Vec<_>>();
                    let potential_saving = atom_counts.len().saturating_sub(1);
                    self.candidates.push(Candidate {
                        rule_idx,
                        path: path.clone(),
                        option_indices,
                        options: promoted_options,
                        atom_counts,
                        potential_saving,
                    });
                }
            }
            GrammarExpr::Sequence(parts) => {
                for (idx, part) in parts.iter().enumerate() {
                    path.push(PathStep::Sequence(idx));
                    self.collect_expr(rule_idx, part, path);
                    path.pop();
                }
            }
            GrammarExpr::Exclude { expr, exclude } => {
                path.push(PathStep::ExcludeExpr);
                self.collect_expr(rule_idx, expr, path);
                path.pop();
                path.push(PathStep::ExcludeExclude);
                self.collect_expr(rule_idx, exclude, path);
                path.pop();
            }
            GrammarExpr::Intersect { expr, intersect } => {
                path.push(PathStep::IntersectExpr);
                self.collect_expr(rule_idx, expr, path);
                path.pop();
                path.push(PathStep::IntersectIntersect);
                self.collect_expr(rule_idx, intersect, path);
                path.pop();
            }
            GrammarExpr::Quantified(inner, Quantifier::Optional) => {
                path.push(PathStep::Optional);
                self.collect_expr(rule_idx, inner, path);
                path.pop();
            }
            GrammarExpr::Quantified(inner, Quantifier::ZeroPlus) => {
                path.push(PathStep::Repeat);
                self.collect_expr(rule_idx, inner, path);
                path.pop();
            }
            GrammarExpr::Quantified(inner, Quantifier::OnePlus) => {
                path.push(PathStep::RepeatOne);
                self.collect_expr(rule_idx, inner, path);
                path.pop();
            }
            GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => {
                path.push(PathStep::RepeatRange);
                self.collect_expr(rule_idx, expr, path);
                path.pop();
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (idx, (item, _)) in items.iter().enumerate() {
                    path.push(PathStep::SeparatedItem(idx));
                    self.collect_expr(rule_idx, item, path);
                    path.pop();
                }
                path.push(PathStep::SeparatedSeparator);
                self.collect_expr(rule_idx, separator, path);
                path.pop();
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                for (idx, symbol) in expr_nfa.symbols.iter().enumerate() {
                    path.push(PathStep::ExprNFASymbol(idx));
                    self.collect_expr(rule_idx, symbol, path);
                    path.pop();
                }
            }
            GrammarExpr::Ref(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => {
                if let Some(atom) = self.eligible_atom(expr) {
                    let atom_id = self.atom_id(atom);
                    self.atom_total_counts[atom_id] += 1;
                }
            }
        }
    }

    fn eligible_atom(&self, expr: &GrammarExpr) -> Option<GrammarExpr> {
        match expr {
            GrammarExpr::Grouped(inner) => self.eligible_atom(inner),
            GrammarExpr::Literal(_) => Some(expr.clone()),
            GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte
                if self.include_non_literal_terminals =>
            {
                Some(expr.clone())
            }
            _ => None,
        }
    }

    fn atom_id(&mut self, atom: GrammarExpr) -> usize {
        if let Some(id) = self.atom_ids.get(&atom) {
            return *id;
        }
        let id = self.atom_total_counts.len();
        self.atom_ids.insert(atom, id);
        self.atom_total_counts.push(0);
        id
    }
}

fn solve_exact(atom_total_counts: &[usize], candidates: &[Candidate]) -> Vec<usize> {
    let atom_count = atom_total_counts.len();
    let candidate_count = candidates.len();
    let mut suffix_cover = vec![vec![0usize; atom_count]; candidate_count + 1];
    for idx in (0..candidate_count).rev() {
        suffix_cover[idx] = suffix_cover[idx + 1].clone();
        for (atom_id, count) in &candidates[idx].atom_counts {
            suffix_cover[idx][*atom_id] += *count;
        }
    }

    let mut search = ExactSearch {
        atom_total_counts,
        candidates,
        suffix_cover,
        covered_counts: vec![0usize; atom_count],
        selected: Vec::new(),
        best_cost: atom_total_counts.len(),
        best_selected: Vec::new(),
    };
    search.visit(0);
    search.best_selected
}

struct ExactSearch<'a> {
    atom_total_counts: &'a [usize],
    candidates: &'a [Candidate],
    suffix_cover: Vec<Vec<usize>>,
    covered_counts: Vec<usize>,
    selected: Vec<usize>,
    best_cost: usize,
    best_selected: Vec<usize>,
}

impl ExactSearch<'_> {
    fn visit(&mut self, idx: usize) {
        let lower_bound = self.selected.len() + self.unavoidable_standalone_atoms(idx);
        if lower_bound >= self.best_cost {
            return;
        }

        if idx == self.candidates.len() {
            let cost = self.selected.len() + self.standalone_atoms();
            if cost < self.best_cost {
                self.best_cost = cost;
                self.best_selected = self.selected.clone();
            }
            return;
        }

        self.include(idx);
        self.visit(idx + 1);
        self.exclude(idx);

        self.visit(idx + 1);
    }

    fn include(&mut self, idx: usize) {
        self.selected.push(idx);
        for (atom_id, count) in &self.candidates[idx].atom_counts {
            self.covered_counts[*atom_id] += *count;
        }
    }

    fn exclude(&mut self, idx: usize) {
        for (atom_id, count) in &self.candidates[idx].atom_counts {
            self.covered_counts[*atom_id] -= *count;
        }
        self.selected.pop();
    }

    fn standalone_atoms(&self) -> usize {
        self.atom_total_counts
            .iter()
            .enumerate()
            .filter(|(idx, total)| self.covered_counts[*idx] < **total)
            .count()
    }

    fn unavoidable_standalone_atoms(&self, idx: usize) -> usize {
        self.atom_total_counts
            .iter()
            .enumerate()
            .filter(|(atom_id, total)| {
                self.covered_counts[*atom_id] + self.suffix_cover[idx][*atom_id] < **total
            })
            .count()
    }
}

fn promoted_cost(
    atom_total_counts: &[usize],
    candidates: &[Candidate],
    selected: &[usize],
) -> usize {
    let mut covered_counts = vec![0usize; atom_total_counts.len()];
    for idx in selected {
        for (atom_id, count) in &candidates[*idx].atom_counts {
            covered_counts[*atom_id] += *count;
        }
    }
    selected.len()
        + atom_total_counts
            .iter()
            .enumerate()
            .filter(|(idx, total)| covered_counts[*idx] < **total)
            .count()
}

struct TerminalNameGenerator {
    used: BTreeSet<String>,
    counter: usize,
}

impl TerminalNameGenerator {
    fn new(used: BTreeSet<String>) -> Self {
        Self { used, counter: 0 }
    }

    fn next(&mut self) -> String {
        loop {
            let name = format!("__GLRMASK_LITERAL_CHOICE_{}", self.counter);
            self.counter += 1;
            if self.used.insert(name.clone()) {
                return name;
            }
        }
    }
}

fn replace_candidate(grammar: &mut NamedGrammar, candidate: &Candidate, terminal_name: &str) {
    let expr = &mut grammar.rules[candidate.rule_idx].expr;
    let expr = expr_at_path_mut(expr, &candidate.path);
    let GrammarExpr::Choice(options) = expr else {
        panic!("candidate path no longer points to a choice");
    };

    let selected_indices = candidate
        .option_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut rewritten = Vec::with_capacity(options.len() - selected_indices.len() + 1);
    let mut inserted_terminal = false;
    for (idx, option) in std::mem::take(options).into_iter().enumerate() {
        if selected_indices.contains(&idx) {
            if !inserted_terminal {
                rewritten.push(GrammarExpr::Ref(terminal_name.to_string()));
                inserted_terminal = true;
            }
        } else {
            rewritten.push(option);
        }
    }
    *options = rewritten;
}

fn expr_at_path_mut<'a>(mut expr: &'a mut GrammarExpr, path: &[PathStep]) -> &'a mut GrammarExpr {
    for step in path {
        expr = match (step, expr) {
            (PathStep::Sequence(idx), GrammarExpr::Sequence(parts)) => &mut parts[*idx],
            (PathStep::Choice(idx), GrammarExpr::Choice(options)) => &mut options[*idx],
            (PathStep::Optional, GrammarExpr::Quantified(inner, Quantifier::Optional))
            | (PathStep::Repeat, GrammarExpr::Quantified(inner, Quantifier::ZeroPlus))
            | (PathStep::RepeatOne, GrammarExpr::Quantified(inner, Quantifier::OnePlus))
            | (PathStep::RepeatRange, GrammarExpr::Quantified(inner, Quantifier::Range(_, _))) => inner,
            (PathStep::ExcludeExpr, GrammarExpr::Exclude { expr: inner, .. }) => inner,
            (PathStep::ExcludeExclude, GrammarExpr::Exclude { exclude: inner, .. }) => inner,
            (PathStep::IntersectExpr, GrammarExpr::Intersect { expr: inner, .. }) => inner,
            (PathStep::IntersectIntersect, GrammarExpr::Intersect { intersect: inner, .. }) => {
                inner
            }
            (
                PathStep::SeparatedItem(idx),
                GrammarExpr::SeparatedSequence { items, .. },
            ) => &mut items[*idx].0,
            (
                PathStep::SeparatedSeparator,
                GrammarExpr::SeparatedSequence { separator, .. },
            ) => separator,
            (PathStep::ExprNFASymbol(idx), GrammarExpr::ExprNFA(expr_nfa)) => {
                &mut expr_nfa.symbols[*idx]
            }
            _ => panic!("candidate path no longer matches expression shape"),
        };
    }
    expr
}

#[cfg(test)]
mod tests {
    use super::promote_choice_terminals_exact;
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};

    fn lit(s: &str) -> GrammarExpr {
        GrammarExpr::Literal(s.as_bytes().to_vec())
    }

    fn nt(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        }
    }

    fn terminal_rule_count(grammar: &NamedGrammar) -> usize {
        grammar.rules.iter().filter(|rule| rule.is_terminal).count()
    }

    #[test]
    fn promotes_mixed_choice_literal_subset_when_it_reduces_terminals() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nt(
                    "start",
                    GrammarExpr::Choice(vec![lit("a"), GrammarExpr::Ref("other".into()), lit("b")]),
                ),
                nt("other", lit("x")),
            ],
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
        };

        let stats = promote_choice_terminals_exact(&mut grammar, false);

        assert_eq!(stats.promoted_choices, 1);
        assert_eq!(terminal_rule_count(&grammar), 1);
        assert!(matches!(
            &grammar.rules[0].expr,
            GrammarExpr::Choice(options)
                if matches!(&options[0], GrammarExpr::Ref(name) if name.starts_with("__GLRMASK_LITERAL_CHOICE_"))
                    && matches!(&options[1], GrammarExpr::Ref(name) if name == "other")
        ));
    }

    #[test]
    fn does_not_promote_dense_pair_cover_when_standalone_literals_are_cheaper() {
        let pairs = [
            ("A", "t1", "t2"),
            ("B", "t1", "t3"),
            ("C", "t1", "t4"),
            ("D", "t2", "t3"),
            ("E", "t2", "t4"),
            ("F", "t3", "t4"),
        ];
        let mut rules = pairs
            .iter()
            .map(|(name, left, right)| nt(name, GrammarExpr::Choice(vec![lit(left), lit(right)])))
            .collect::<Vec<_>>();
        rules.push(nt("start", GrammarExpr::Ref("A".into())));
        let mut grammar = NamedGrammar {
            rules,
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
        };

        let stats = promote_choice_terminals_exact(&mut grammar, false);

        assert_eq!(stats.promoted_choices, 0);
        assert_eq!(terminal_rule_count(&grammar), 0);
    }

    #[test]
    fn does_not_rewrite_cost_neutral_ties() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nt("start", GrammarExpr::Choice(vec![lit("a"), lit("b")])),
                nt("other", lit("a")),
            ],
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
        };

        let before = grammar.rules[0].expr.clone();
        let stats = promote_choice_terminals_exact(&mut grammar, false);

        assert_eq!(stats.promoted_choices, 0);
        assert_eq!(grammar.rules[0].expr, before);
    }

    #[test]
    fn non_literal_terminal_atoms_are_opt_in() {
        let expr = GrammarExpr::Choice(vec![
            GrammarExpr::RawRegex("[a-z]+".into()),
            GrammarExpr::CharClass { def: "0-9".into(), negate: false, utf8: true },
        ]);
        let mut grammar = NamedGrammar {
            rules: vec![nt("start", expr)],
            start: "start".into(),
            ignore: None,
            lexer_partitions: Default::default(),
        };

        let stats = promote_choice_terminals_exact(&mut grammar, false);
        assert_eq!(stats.promoted_choices, 0);

        let stats = promote_choice_terminals_exact(&mut grammar, true);
        assert_eq!(stats.promoted_choices, 1);
    }
}
