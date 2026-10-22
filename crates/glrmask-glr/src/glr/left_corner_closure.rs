//! Exact repair for a left-corner SCC containing an interior cycle.
//!
//! Enumerating only simple head paths is exact for a simple directed ring,
//! but loses loops that do not pass through the selected destination. Use
//! ordered substitution and represent each eliminated head's positive loop
//! closure as an ordinary left-recursive nonterminal. No derivation bound,
//! runtime fallback, or nullable closure helper is introduced.
//!
//! Both storage implementations use this semantic correction. Their ordered
//! comparison still checks the surrounding pipeline; independent finite-word
//! and Earley tests, rather than the old buggy rewrite, qualify this operation.
use std::collections::{BTreeMap, BTreeSet};
use crate::grammar::flat::{NonterminalID, Rule, Symbol};
use glrmask_invariant::__private::charge_template_build_budget;

// One unit per generated symbol, so a combinatorial left-corner cross product
// exhausts the deterministic template budget instead of memory.
fn charge(units: usize) {
    charge_template_build_budget("grammar_normalization", 1 + units as u64);
}

fn needs_closure(rules: &[Rule], component: &BTreeSet<NonterminalID>,
    nullable: &BTreeSet<NonterminalID>) -> bool {
    let mut edges = BTreeMap::<u32, BTreeSet<u32>>::new();
    for rule in rules.iter().filter(|rule| component.contains(&rule.lhs)) {
        for symbol in &rule.rhs {
            let Symbol::Nonterminal(head) = symbol else { break; };
            if component.contains(head) { edges.entry(rule.lhs).or_default().insert(*head); }
            if !nullable.contains(head) { break; }
        }
    }
    edges.iter().any(|(lhs, targets)| targets.contains(lhs) || targets.len() > 1)
}

/// Return a corrected inventory only when a nontrivial component has a proper
/// subcycle. Keep the existing allocation/order fast path for simple rings.
pub(super) fn repair(rules: &[Rule], components: &[BTreeSet<NonterminalID>],
    nullable: &BTreeSet<NonterminalID>) -> Option<Vec<Rule>> {
    let mut selected = components.iter().filter(|component|
        needs_closure(rules, component, nullable)).collect::<Vec<_>>();
    if selected.is_empty() { return None; }
    let mut next = rules.iter().flat_map(|rule| std::iter::once(rule.lhs)
        .chain(rule.rhs.iter().filter_map(|s| match s {
            Symbol::Nonterminal(nt) => Some(*nt), _ => None,
        }))).max().unwrap_or(0).checked_add(1).expect("nonterminal coordinate exhausted");
    if !nullable.is_empty() {
        // Ordered head substitution assumes nonempty exits. The normalizer
        // already removes epsilon while retaining source nullability elsewhere;
        // newly introduced right-recursion helpers need the same operation.
        let rewritten = super::normalization_reference::inline_null_productions(rules, next);
        assert_ne!(rewritten, rules, "nullable left-corner repair must make progress");
        return Some(rewritten);
    }
    selected.sort_unstable_by_key(|component| *component.first().expect("nonempty SCC"));
    // Large SCCs can blow up before the result is available for the size
    // check below, so start with the small-rule-set order directly.
    let force_by_size = selected.iter().any(|component| component.len() > 8);
    let result = repair_pass(rules, &selected, &mut next, force_by_size);
    // Ordered substitution multiplies the alternatives of every head in the
    // SCC. A head order that starts with the largest rule sets can therefore
    // create a cross product even though another order stays small. Keep the
    // historic order for ordinary grammars and retry with the smallest rule
    // sets first only when it would expand the inventory substantially.
    if !force_by_size && result.len() > rules.len().saturating_mul(4).saturating_add(64) {
        let mut retry_next = next;
        return Some(repair_pass(rules, &selected, &mut retry_next, true));
    }
    Some(result)
}

fn repair_pass(rules: &[Rule], selected: &[&BTreeSet<NonterminalID>],
    next: &mut u32, by_size: bool) -> Vec<Rule> {
    charge(rules.len());
    let mut result = rules.to_vec();
    for component in selected {
        charge(result.len());
        let mut rows = BTreeMap::<u32, Vec<Vec<Symbol>>>::new();
        for rule in result.iter().filter(|rule| component.contains(&rule.lhs)) {
            rows.entry(rule.lhs).or_default().push(rule.rhs.clone());
        }
        for rhs in rows.values_mut() { rhs.sort_unstable(); rhs.dedup(); }
        let mut order = component.iter().copied().collect::<Vec<_>>();
        if by_size {
            charge(result.len().saturating_mul(order.len()));
            order.sort_unstable_by_key(|&nt| {
                (result.iter().filter(|rule| rule.lhs == nt).count(), nt)
            });
        }
        let mut closures = BTreeMap::<u32, u32>::new();
        let mut helpers = BTreeMap::<u32, Vec<Vec<Symbol>>>::new();
        for (position, &lhs) in order.iter().enumerate() {
            let mut current = rows.remove(&lhs).unwrap_or_default();
            for &head in &order[..position] {
                if !current.iter().any(|rhs| rhs.first() == Some(&Symbol::Nonterminal(head))) {
                    continue;
                }
                let alternatives = rows.get(&head).expect("earlier SCC member processed");
                let (loops, exits): (Vec<_>, Vec<_>) = alternatives.iter()
                    .partition(|rhs| rhs.first() == Some(&Symbol::Nonterminal(head)));
                let loops = loops.into_iter().map(|rhs| &rhs[1..])
                    .filter(|tail| !tail.is_empty()).collect::<Vec<_>>();
                let closure = if loops.is_empty() || exits.is_empty() { None } else {
                    let id = *closures.entry(head).or_insert_with(|| {
                        let id = *next;
                        *next = next.checked_add(1).expect("nonterminal coordinate exhausted");
                        let mut generated = Vec::with_capacity(loops.len() * 2);
                        for tail in &loops {
                            charge(2 * tail.len());
                            generated.push(tail.to_vec());
                            let mut repeated = vec![Symbol::Nonterminal(id)];
                            repeated.extend_from_slice(tail);
                            generated.push(repeated);
                        }
                        generated.sort_unstable(); generated.dedup();
                        helpers.insert(id, generated);
                        id
                    });
                    Some(id)
                };
                let mut expanded = Vec::new();
                for rhs in current {
                    if rhs.first() != Some(&Symbol::Nonterminal(head)) {
                        expanded.push(rhs); continue;
                    }
                    for exit in &exits {
                        assert!(!exit.is_empty(), "left-corner exits must be nonnullable");
                        charge(2 * (exit.len() + rhs.len()));
                        let mut direct = (*exit).clone();
                        direct.extend_from_slice(&rhs[1..]); expanded.push(direct);
                        if let Some(id) = closure {
                            let mut repeated = (*exit).clone();
                            repeated.push(Symbol::Nonterminal(id));
                            repeated.extend_from_slice(&rhs[1..]); expanded.push(repeated);
                        }
                    }
                }
                expanded.sort_unstable(); expanded.dedup(); current = expanded;
            }
            rows.insert(lhs, current);
        }
        result.retain(|rule| !component.contains(&rule.lhs));
        for (lhs, alternatives) in rows.into_iter().chain(helpers) {
            result.extend(alternatives.into_iter().map(|rhs| Rule { lhs, rhs }));
        }
    }
    result
}
