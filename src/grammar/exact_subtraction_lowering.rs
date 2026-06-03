use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{GlrMaskError, Result};

use super::ast::{GrammarExpr, NamedGrammar, NamedRule};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ExactSubtractionLoweringStats {
    pub rewritten_sites: usize,
    pub lhs_rule_groups: usize,
    pub partition_rules: usize,
    pub tree_rules: usize,
    pub result_rules: usize,
}

pub fn lower_exact_subtractions(
    grammar: &mut NamedGrammar,
) -> Result<ExactSubtractionLoweringStats> {
    let rule_exprs: HashMap<String, GrammarExpr> = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.expr.clone()))
        .collect();
    let rule_is_terminal: HashMap<String, bool> = grammar
        .rules
        .iter()
        .map(|rule| (rule.name.clone(), rule.is_terminal))
        .collect();

    let resolver = ExactSubtractionResolver {
        rule_exprs: &rule_exprs,
        rule_is_terminal: &rule_is_terminal,
    };

    let mut collector = SiteCollector::default();
    for rule in &grammar.rules {
        if !rule.is_terminal {
            collector.collect_expr(&rule.expr, &resolver)?;
        }
    }

    if collector.per_lhs.is_empty() {
        return Ok(ExactSubtractionLoweringStats::default());
    }

    let mut allocator = NameAllocator::new(grammar.rules.iter().map(|rule| rule.name.clone()));
    let mut generated_rules = Vec::new();
    let mut rewrite_targets = HashMap::new();
    let mut stats = ExactSubtractionLoweringStats {
        rewritten_sites: collector.rewritten_sites,
        lhs_rule_groups: collector.per_lhs.len(),
        ..ExactSubtractionLoweringStats::default()
    };

    for (lhs_name, collection) in collector.per_lhs {
        let lhs_alts = top_level_alternatives(
            rule_exprs
                .get(&lhs_name)
                .expect("lhs rule must exist while lowering exact subtractions"),
        );
        let generated = build_helpers_for_lhs(&lhs_name, lhs_alts, collection, &mut allocator);
        stats.partition_rules += generated.partition_rules;
        stats.tree_rules += generated.tree_rules;
        stats.result_rules += generated.result_rules;
        rewrite_targets.insert(lhs_name, generated.result_names);
        generated_rules.extend(generated.rules);
    }

    for rule in &mut grammar.rules {
        rewrite_expr(
            &mut rule.expr,
            !rule.is_terminal,
            &resolver,
            &rewrite_targets,
        )?;
    }

    grammar.rules.extend(generated_rules);
    Ok(stats)
}

#[derive(Debug, Clone)]
struct ResolvedSubtraction {
    lhs_name: String,
    removed_indices: Vec<usize>,
}

struct ExactSubtractionResolver<'a> {
    rule_exprs: &'a HashMap<String, GrammarExpr>,
    rule_is_terminal: &'a HashMap<String, bool>,
}

impl<'a> ExactSubtractionResolver<'a> {
    fn resolve_site(&self, expr: &GrammarExpr) -> Result<Option<ResolvedSubtraction>> {
        let GrammarExpr::Exclude { expr: lhs_expr, exclude } = expr else {
            return Ok(None);
        };
        let GrammarExpr::Ref(lhs_name) = strip_grouping(lhs_expr) else {
            return Ok(None);
        };
        let Some(false) = self.rule_is_terminal.get(lhs_name).copied() else {
            return Ok(None);
        };

        let lhs_rule_expr = self.rule_exprs.get(lhs_name).ok_or_else(|| {
            GlrMaskError::GrammarParse(format!(
                "unknown nonterminal referenced in exact alternative subtraction: {lhs_name}"
            ))
        })?;
        let lhs_alts = top_level_alternatives(lhs_rule_expr);
        let lhs_alt_keys = lhs_alts
            .iter()
            .map(|alt| self.canonical_exact_expr(alt))
            .collect::<Vec<_>>();
        let mut remaining_indices = (0..lhs_alts.len()).collect::<Vec<_>>();
        let mut removed_indices = Vec::new();

        for remove_alt in self.exact_subtraction_alternatives(lhs_name, exclude)? {
            let remove_alt_key = self.canonical_exact_expr(&remove_alt);
            let Some(position) = remaining_indices
                .iter()
                .position(|&index| lhs_alt_keys[index] == remove_alt_key)
            else {
                return Err(GlrMaskError::GrammarParse(format!(
                    "no exact alternative {:?} in {}",
                    remove_alt, lhs_name
                )));
            };
            removed_indices.push(remaining_indices.remove(position));
        }

        removed_indices.sort_unstable();
        Ok(Some(ResolvedSubtraction {
            lhs_name: lhs_name.clone(),
            removed_indices,
        }))
    }

    fn exact_subtraction_alternatives(
        &self,
        lhs_name: &str,
        exclude: &GrammarExpr,
    ) -> Result<Vec<GrammarExpr>> {
        match exclude {
            GrammarExpr::Choice(options) => {
                let mut out = Vec::new();
                for option in options {
                    out.extend(self.exact_subtraction_alternatives(lhs_name, option)?);
                }
                Ok(out)
            }
            GrammarExpr::Grouped(inner) => Ok(top_level_alternatives(inner)),
            GrammarExpr::Ref(name) => {
                let Some(false) = self.rule_is_terminal.get(name).copied() else {
                    return Err(GlrMaskError::GrammarParse(format!(
                        "{lhs_name} - {name} requires {name} to name a nonterminal rule"
                    )));
                };
                let referenced_expr = self.rule_exprs.get(name).ok_or_else(|| {
                    GlrMaskError::GrammarParse(format!(
                        "unknown rule referenced in exact alternative subtraction: {name}"
                    ))
                })?;
                Ok(top_level_alternatives(referenced_expr))
            }
            other => Ok(top_level_alternatives(other)),
        }
    }

    fn canonical_exact_expr(&self, expr: &GrammarExpr) -> GrammarExpr {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        self.canonical_exact_expr_inner(expr, &mut visiting, &mut memo)
    }

    fn canonical_exact_expr_inner(
        &self,
        expr: &GrammarExpr,
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, GrammarExpr>,
    ) -> GrammarExpr {
        match strip_grouping(expr) {
            GrammarExpr::Ref(name) => {
                if self.rule_is_terminal.get(name).copied().unwrap_or(false) {
                    return GrammarExpr::Ref(name.clone());
                }
                let Some(referenced) = self.rule_exprs.get(name) else {
                    return GrammarExpr::Ref(name.clone());
                };
                if let Some(canonical) = memo.get(name) {
                    return canonical.clone();
                }
                if !visiting.insert(name.clone()) {
                    return GrammarExpr::Ref(name.clone());
                }
                let canonical = self.canonical_exact_expr_inner(referenced, visiting, memo);
                visiting.remove(name);
                memo.insert(name.clone(), canonical.clone());
                canonical
            }
            GrammarExpr::Grouped(inner) => self.canonical_exact_expr_inner(inner, visiting, memo),
            GrammarExpr::Optional(inner) => GrammarExpr::Optional(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::Repeat(inner) => GrammarExpr::Repeat(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::RepeatOne(inner) => GrammarExpr::RepeatOne(Box::new(
                self.canonical_exact_expr_inner(inner, visiting, memo),
            )),
            GrammarExpr::RepeatRange { expr, min, max } => GrammarExpr::RepeatRange {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                min: *min,
                max: *max,
            },
            GrammarExpr::Sequence(items) => GrammarExpr::Sequence(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Choice(items) => GrammarExpr::Choice(
                items
                    .iter()
                    .map(|item| self.canonical_exact_expr_inner(item, visiting, memo))
                    .collect(),
            ),
            GrammarExpr::Exclude { expr, exclude } => GrammarExpr::Exclude {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                exclude: Box::new(self.canonical_exact_expr_inner(exclude, visiting, memo)),
            },
            GrammarExpr::Intersect { expr, intersect } => GrammarExpr::Intersect {
                expr: Box::new(self.canonical_exact_expr_inner(expr, visiting, memo)),
                intersect: Box::new(self.canonical_exact_expr_inner(intersect, visiting, memo)),
            },
            GrammarExpr::SeparatedSequence {
                items,
                separator,
                allow_empty,
            } => GrammarExpr::SeparatedSequence {
                items: items
                    .iter()
                    .map(|(item, required)| {
                        (
                            self.canonical_exact_expr_inner(item, visiting, memo),
                            *required,
                        )
                    })
                    .collect(),
                separator: Box::new(self.canonical_exact_expr_inner(separator, visiting, memo)),
                allow_empty: *allow_empty,
            },
            GrammarExpr::ExprNFA(expr_nfa) => GrammarExpr::ExprNFA(Box::new(
                crate::grammar::expr_nfa::ExprNFA {
                    nfa: expr_nfa.nfa.clone(),
                    symbols: expr_nfa
                        .symbols
                        .iter()
                        .map(|symbol| self.canonical_exact_expr_inner(symbol, visiting, memo))
                        .collect(),
                },
            )),
            GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => strip_grouping(expr).clone(),
        }
    }
}

#[derive(Default)]
struct SiteCollector {
    rewritten_sites: usize,
    per_lhs: BTreeMap<String, LhsCollection>,
}

impl SiteCollector {
    fn collect_expr(&mut self, expr: &GrammarExpr, resolver: &ExactSubtractionResolver<'_>) -> Result<()> {
        if let Some(resolved) = resolver.resolve_site(expr)? {
            self.rewritten_sites += 1;
            self.per_lhs
                .entry(resolved.lhs_name)
                .or_default()
                .add_removal_set(resolved.removed_indices);
            return Ok(());
        }

        match expr {
            GrammarExpr::Grouped(inner)
            | GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner) => self.collect_expr(inner, resolver),
            GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
                for item in items {
                    self.collect_expr(item, resolver)?;
                }
                Ok(())
            }
            GrammarExpr::Exclude { expr, exclude }
            | GrammarExpr::Intersect {
                expr,
                intersect: exclude,
            } => {
                self.collect_expr(expr, resolver)?;
                self.collect_expr(exclude, resolver)
            }
            GrammarExpr::RepeatRange { expr, .. } => self.collect_expr(expr, resolver),
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                for (item, _) in items {
                    self.collect_expr(item, resolver)?;
                }
                self.collect_expr(separator, resolver)
            }
            GrammarExpr::ExprNFA(expr_nfa) => {
                for symbol in &expr_nfa.symbols {
                    self.collect_expr(symbol, resolver)?;
                }
                Ok(())
            }
            GrammarExpr::Ref(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => Ok(()),
        }
    }
}

#[derive(Default)]
struct LhsCollection {
    removal_sets: Vec<Vec<usize>>,
    seen_removal_sets: HashSet<Vec<usize>>,
}

impl LhsCollection {
    fn add_removal_set(&mut self, removed_indices: Vec<usize>) {
        if self.seen_removal_sets.insert(removed_indices.clone()) {
            self.removal_sets.push(removed_indices);
        }
    }
}

struct GeneratedHelpers {
    rules: Vec<NamedRule>,
    result_names: HashMap<Vec<usize>, String>,
    partition_rules: usize,
    tree_rules: usize,
    result_rules: usize,
}

fn build_helpers_for_lhs(
    lhs_name: &str,
    lhs_alts: Vec<GrammarExpr>,
    collection: LhsCollection,
    allocator: &mut NameAllocator,
) -> GeneratedHelpers {
    let sanitized = sanitize_name_component(lhs_name);
    let mut rules = Vec::new();
    let mut result_names = HashMap::new();

    let mut signatures = vec![Vec::new(); lhs_alts.len()];
    for (site_index, removal_set) in collection.removal_sets.iter().enumerate() {
        for &alt_index in removal_set {
            signatures[alt_index].push(site_index);
        }
    }

    let mut partition_lookup: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut partitions = Vec::<Partition>::new();
    for (alt_index, signature) in signatures.into_iter().enumerate() {
        if let Some(&partition_index) = partition_lookup.get(&signature) {
            partitions[partition_index].alt_indices.push(alt_index);
        } else {
            let partition_index = partitions.len();
            partition_lookup.insert(signature.clone(), partition_index);
            partitions.push(Partition {
                alt_indices: vec![alt_index],
            });
        }
    }

    let mut part_names = Vec::with_capacity(partitions.len());
    for partition in &partitions {
        let name = allocator.alloc(&format!("__exact_sub_{sanitized}_part"));
        let expr = choice_or_single(
            partition
                .alt_indices
                .iter()
                .map(|&index| lhs_alts[index].clone())
                .collect(),
        );
        rules.push(named_helper_rule(name.clone(), expr));
        part_names.push(name);
    }

    let total_partitions = partitions.len();
    let partition_rule_count = part_names.len();
    let mut tree = SegmentTreeBuilder {
        part_names,
        sanitized: sanitized.clone(),
        cache: HashMap::new(),
        tree_rules: 0,
    };

    for removal_set in &collection.removal_sets {
        let name = allocator.alloc(&format!("__exact_sub_{sanitized}_result"));
        let expr = if total_partitions == 0 {
            GrammarExpr::Choice(Vec::new())
        } else {
            let included = partitions
                .iter()
                .map(|partition| {
                    !removal_set
                        .binary_search(&partition.alt_indices[0])
                        .is_ok()
                })
                .collect::<Vec<_>>();
            let refs = cover_included_partitions(&included, &mut tree, allocator, &mut rules);
            if refs.is_empty() {
                GrammarExpr::Choice(Vec::new())
            } else {
                choice_or_single(refs.into_iter().map(GrammarExpr::Ref).collect())
            }
        };
        rules.push(named_helper_rule(name.clone(), expr));
        result_names.insert(removal_set.clone(), name);
    }

    GeneratedHelpers {
        rules,
        result_names,
        partition_rules: partition_rule_count,
        tree_rules: tree.tree_rules,
        result_rules: collection.removal_sets.len(),
    }
}

#[derive(Debug)]
struct Partition {
    alt_indices: Vec<usize>,
}

struct SegmentTreeBuilder {
    part_names: Vec<String>,
    sanitized: String,
    cache: HashMap<(usize, usize), String>,
    tree_rules: usize,
}

impl SegmentTreeBuilder {
    fn node_ref(
        &mut self,
        start: usize,
        end: usize,
        allocator: &mut NameAllocator,
        rules: &mut Vec<NamedRule>,
    ) -> String {
        if end - start == 1 {
            return self.part_names[start].clone();
        }
        if let Some(existing) = self.cache.get(&(start, end)) {
            return existing.clone();
        }

        let mid = start + (end - start) / 2;
        let left = self.node_ref(start, mid, allocator, rules);
        let right = self.node_ref(mid, end, allocator, rules);
        let name = allocator.alloc(&format!("__exact_sub_{}_tree", self.sanitized));
        let expr = GrammarExpr::Choice(vec![GrammarExpr::Ref(left), GrammarExpr::Ref(right)]);
        rules.push(named_helper_rule(name.clone(), expr));
        self.cache.insert((start, end), name.clone());
        self.tree_rules += 1;
        name
    }
}

fn cover_included_partitions(
    included: &[bool],
    tree: &mut SegmentTreeBuilder,
    allocator: &mut NameAllocator,
    rules: &mut Vec<NamedRule>,
) -> Vec<String> {
    let mut refs = Vec::new();
    let mut index = 0;
    while index < included.len() {
        if !included[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < included.len() && included[index] {
            index += 1;
        }
        collect_cover_refs(
            start,
            index,
            0,
            included.len(),
            tree,
            allocator,
            rules,
            &mut refs,
        );
    }
    refs
}

fn collect_cover_refs(
    target_start: usize,
    target_end: usize,
    node_start: usize,
    node_end: usize,
    tree: &mut SegmentTreeBuilder,
    allocator: &mut NameAllocator,
    rules: &mut Vec<NamedRule>,
    refs: &mut Vec<String>,
) {
    if target_end <= node_start || node_end <= target_start {
        return;
    }
    if target_start <= node_start && node_end <= target_end {
        refs.push(tree.node_ref(node_start, node_end, allocator, rules));
        return;
    }
    if node_end - node_start == 1 {
        refs.push(tree.node_ref(node_start, node_end, allocator, rules));
        return;
    }
    let mid = node_start + (node_end - node_start) / 2;
    collect_cover_refs(target_start, target_end, node_start, mid, tree, allocator, rules, refs);
    collect_cover_refs(target_start, target_end, mid, node_end, tree, allocator, rules, refs);
}

fn rewrite_expr(
    expr: &mut GrammarExpr,
    allow_exact_subtractions: bool,
    resolver: &ExactSubtractionResolver<'_>,
    rewrite_targets: &HashMap<String, HashMap<Vec<usize>, String>>,
) -> Result<()> {
    if allow_exact_subtractions {
        if let Some(resolved) = resolver.resolve_site(expr)? {
            if let Some(name) = rewrite_targets
                .get(&resolved.lhs_name)
                .and_then(|sites| sites.get(&resolved.removed_indices))
            {
                *expr = GrammarExpr::Ref(name.clone());
                return Ok(());
            }
        }
    }

    match expr {
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => {
            rewrite_expr(inner, allow_exact_subtractions, resolver, rewrite_targets)
        }
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            for item in items {
                rewrite_expr(item, allow_exact_subtractions, resolver, rewrite_targets)?;
            }
            Ok(())
        }
        GrammarExpr::Exclude { expr, exclude }
        | GrammarExpr::Intersect {
            expr,
            intersect: exclude,
        } => {
            rewrite_expr(expr, allow_exact_subtractions, resolver, rewrite_targets)?;
            rewrite_expr(exclude, allow_exact_subtractions, resolver, rewrite_targets)
        }
        GrammarExpr::RepeatRange { expr, .. } => {
            rewrite_expr(expr, allow_exact_subtractions, resolver, rewrite_targets)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            for (item, _) in items {
                rewrite_expr(item, allow_exact_subtractions, resolver, rewrite_targets)?;
            }
            rewrite_expr(separator, allow_exact_subtractions, resolver, rewrite_targets)
        }
        GrammarExpr::ExprNFA(expr_nfa) => {
            for symbol in &mut expr_nfa.symbols {
                rewrite_expr(symbol, allow_exact_subtractions, resolver, rewrite_targets)?;
            }
            Ok(())
        }
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => Ok(()),
    }
}

fn strip_grouping(expr: &GrammarExpr) -> &GrammarExpr {
    match expr {
        GrammarExpr::Grouped(inner) => strip_grouping(inner),
        _ => expr,
    }
}

fn top_level_alternatives(expr: &GrammarExpr) -> Vec<GrammarExpr> {
    match strip_grouping(expr) {
        GrammarExpr::Choice(options) => options
            .iter()
            .map(|option| strip_grouping(option).clone())
            .collect(),
        other => vec![other.clone()],
    }
}

fn named_helper_rule(name: String, expr: GrammarExpr) -> NamedRule {
    NamedRule {
        name,
        expr,
        is_terminal: false,
        is_internal: true,
    }
}

fn choice_or_single(mut options: Vec<GrammarExpr>) -> GrammarExpr {
    if options.len() == 1 {
        options.pop().unwrap()
    } else {
        GrammarExpr::Choice(options)
    }
}

fn sanitize_name_component(name: &str) -> String {
    let sanitized = name
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    if sanitized.is_empty() {
        "rule".to_string()
    } else {
        sanitized
    }
}

struct NameAllocator {
    used: HashSet<String>,
    counters: HashMap<String, usize>,
}

impl NameAllocator {
    fn new<I>(existing_names: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let used = existing_names.into_iter().collect::<HashSet<_>>();
        Self {
            used,
            counters: HashMap::new(),
        }
    }

    fn alloc(&mut self, prefix: &str) -> String {
        let counter = self.counters.entry(prefix.to_string()).or_insert(0);
        loop {
            let candidate = if *counter == 0 {
                prefix.to_string()
            } else {
                format!("{prefix}_{}", *counter)
            };
            *counter += 1;
            if self.used.insert(candidate.clone()) {
                return candidate;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::lower_exact_subtractions;
    use crate::grammar::ast::{lower, GrammarExpr, NamedGrammar, NamedRule};
    use crate::dump_json_schema_grammar_glrm;
    use std::{env, ffi::OsString, sync::Mutex};
    use serde_json::json;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::set_var(key, value);
            }
            Self { key, original }
        }

        fn unset(key: &'static str) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::remove_var(key);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    env::set_var(self.key, value);
                },
                None => unsafe {
                    env::remove_var(self.key);
                },
            }
        }
    }

    fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        }
    }

    fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: false,
        }
    }

    fn literal(text: &str) -> GrammarExpr {
        GrammarExpr::Literal(text.as_bytes().to_vec())
    }

    fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {
        GrammarExpr::Exclude {
            expr: Box::new(GrammarExpr::Ref(lhs.to_string())),
            exclude: Box::new(exclude),
        }
    }

    fn find_rule<'a>(grammar: &'a NamedGrammar, name: &str) -> &'a NamedRule {
        grammar
            .rules
            .iter()
            .find(|rule| rule.name == name)
            .unwrap()
    }

    fn contains_exclude(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Exclude { .. } => true,
            GrammarExpr::Grouped(inner)
            | GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner) => contains_exclude(inner),
            GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
                items.iter().any(contains_exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                contains_exclude(expr) || contains_exclude(intersect)
            }
            GrammarExpr::RepeatRange { expr, .. } => contains_exclude(expr),
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items.iter().any(|(item, _)| contains_exclude(item)) || contains_exclude(separator)
            }
            GrammarExpr::ExprNFA(expr_nfa) => expr_nfa.symbols.iter().any(contains_exclude),
            GrammarExpr::Ref(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    #[test]
    fn exact_subtraction_rewrites_sites_into_shared_helpers() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        literal("a"),
                        literal("b"),
                        literal("c"),
                        literal("d"),
                    ]),
                ),
                nonterminal(
                    "start",
                    GrammarExpr::Choice(vec![
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("a"),
                                literal("d"),
                            ]))),
                        ),
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("c"),
                                literal("d"),
                            ]))),
                        ),
                        subtract("A", GrammarExpr::Grouped(Box::new(literal("d")))),
                    ]),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let stats = lower_exact_subtractions(&mut grammar).unwrap();

        assert_eq!(stats.rewritten_sites, 3);
        let start_rule = find_rule(&grammar, "start");
        assert!(!contains_exclude(&start_rule.expr));
        let GrammarExpr::Choice(options) = &start_rule.expr else {
            panic!("expected rewritten start choice: {:?}", start_rule.expr);
        };
        assert!(options.iter().all(|expr| matches!(expr, GrammarExpr::Ref(name) if name.starts_with("__exact_sub_A_result"))));
        assert!(grammar.rules.iter().any(|rule| rule.name.starts_with("__exact_sub_A_part")));
        assert!(grammar.rules.iter().any(|rule| rule.name.starts_with("__exact_sub_A_tree")));
        assert!(grammar.rules.iter().any(|rule| rule.name.starts_with("__exact_sub_A_result")));
        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_partitions_alternatives_by_shared_signature() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        literal("a"),
                        literal("b"),
                        literal("c"),
                        literal("d"),
                    ]),
                ),
                nonterminal(
                    "start",
                    GrammarExpr::Choice(vec![
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("b"),
                                literal("c"),
                            ]))),
                        ),
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("b"),
                                literal("c"),
                                literal("d"),
                            ]))),
                        ),
                    ]),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        lower_exact_subtractions(&mut grammar).unwrap();

        assert!(grammar.rules.iter().any(|rule| {
            rule.name.starts_with("__exact_sub_A_part")
                && rule.expr
                    == GrammarExpr::Choice(vec![literal("b"), literal("c")])
        }));
    }

    #[test]
    fn exact_subtraction_errors_on_missing_exact_alternative() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal("A", GrammarExpr::Choice(vec![literal("a"), literal("b")])),
                nonterminal(
                    "start",
                    subtract("A", GrammarExpr::Grouped(Box::new(literal("c")))),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let err = lower_exact_subtractions(&mut grammar).unwrap_err();
        assert!(format!("{err}").contains("no exact alternative"), "{err}");
    }

    #[test]
    fn exact_subtraction_matches_nonterminal_alias_body() {
        let mut grammar = NamedGrammar {
            rules: vec![
                terminal("JSON_STRING_BODY", literal("body\"")),
                nonterminal(
                    "json_string",
                    GrammarExpr::Sequence(vec![
                        literal("\""),
                        GrammarExpr::Ref("JSON_STRING_BODY".to_string()),
                    ]),
                ),
                nonterminal(
                    "json_value",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("json_string".to_string()),
                        literal("0"),
                    ]),
                ),
                nonterminal(
                    "start",
                    subtract("json_value", GrammarExpr::Ref("json_string".to_string())),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let stats = lower_exact_subtractions(&mut grammar).unwrap();

        assert_eq!(stats.rewritten_sites, 1);
        assert!(!contains_exclude(&find_rule(&grammar, "start").expr));
        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_canonicalization_is_cycle_safe() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "loop",
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("y"),
                    ]),
                ),
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("x"),
                    ]),
                ),
                nonterminal("start", subtract("A", literal("z"))),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let err = lower_exact_subtractions(&mut grammar).unwrap_err();
        assert!(format!("{err}").contains("no exact alternative"), "{err}");
    }

    #[test]
    fn exact_subtraction_json_schema_dump_uses_helpers_when_enabled() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let _lower = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS");

        let schema = json!({
            "type": "object",
            "properties": {
                "first": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"}
                    },
                    "additionalProperties": {"type": "string"}
                },
                "second": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "string"}
                    },
                    "patternProperties": {
                        "^x_": {"type": "number"}
                    },
                    "additionalProperties": {"type": "string"}
                }
            },
            "additionalProperties": false
        });

        let glrm = dump_json_schema_grammar_glrm(&schema.to_string()).unwrap();
        assert!(glrm.contains("JSON_STRING JSON_KEY_SEPARATOR - \"\\\"a\\\"\" JSON_KEY_SEPARATOR - \"\\\"b\\\"\" JSON_KEY_SEPARATOR"), "{glrm}");
        assert!(!glrm.contains("__exact_sub_AP_SHARED_LITERAL_KEY_SET_result"), "{glrm}");
    }

    #[test]
    fn exact_subtraction_json_schema_dump_keeps_direct_subtraction_when_disabled() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let _lower = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS", "0");
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "type": "object",
            "properties": {
                "first": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"}
                    },
                    "additionalProperties": {"type": "string"}
                },
                "second": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "string"}
                    },
                    "patternProperties": {
                        "^x_": {"type": "number"}
                    },
                    "additionalProperties": {"type": "string"}
                }
            },
            "additionalProperties": false
        });

        let glrm = dump_json_schema_grammar_glrm(&schema.to_string()).unwrap();
        assert!(glrm.contains("JSON_STRING JSON_KEY_SEPARATOR - \"\\\"a\\\"\" JSON_KEY_SEPARATOR - \"\\\"b\\\"\" JSON_KEY_SEPARATOR"), "{glrm}");
        assert!(!glrm.contains("__exact_sub_AP_SHARED_LITERAL_KEY_SET_result"), "{glrm}");
    }
}
