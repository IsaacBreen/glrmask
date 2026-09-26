//! Exact comparison of raw parser-DWA mask semantics.
//!
//! A mask is the union of accepting weights at EVERY visible stack prefix.
//! An explicit parser-state edge has priority over DEFAULT, even when its
//! weight rejects a particular coordinate. This differs from ordinary
//! weighted-word acceptance and from nondeterministic wildcard semantics.

use std::collections::{BTreeSet, VecDeque};

use rustc_hash::FxHashMap;
#[cfg(test)]
use rustc_hash::FxHashSet;
use range_set_blaze::RangeSetBlaze;

use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

#[derive(Debug)]
pub struct ParserMaskDifference {
    pub stack_top_first: Vec<u32>,
    pub left_mask: Weight,
    pub right_mask: Weight,
}

#[derive(Debug)]
pub struct ParserMaskComparison {
    /// None is returned only after exhausting the exact reachable product.
    pub difference: Option<ParserMaskDifference>,
    pub product_states: usize,
    pub compared_branches: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Product {
    left: Option<u32>,
    right: Option<u32>,
    left_weight: Weight,
    right_weight: Weight,
    accepted: Weight,
}

fn contribution(dwa: &DWA, state: Option<u32>, weight: &Weight, ops: &mut ScopedWeightOpCache) -> Weight {
    state.and_then(|state| dwa.states().get(state as usize))
        .and_then(|state| state.final_weight.as_ref())
        .map_or_else(Weight::empty, |final_weight| ops.intersection(weight, final_weight))
}

fn advance(
    dwa: &DWA, state: Option<u32>, weight: &Weight, parser: u32,
    ops: &mut ScopedWeightOpCache,
) -> (Option<u32>, Weight) {
    if weight.is_empty() { return (None, Weight::empty()); }
    let Some(row) = state.and_then(|state| dwa.states().get(state as usize)) else {
        return (None, Weight::empty());
    };
    let Some((target, edge)) = row.transitions.get(&(parser as i32))
        .or_else(|| row.transitions.get(&DEFAULT_LABEL)) else {
        return (None, Weight::empty());
    };
    let weight = ops.intersection(weight, edge);
    ((!weight.is_empty()).then_some(*target), weight)
}

/// Compare masks for EVERY finite top-first stack over `0..num_parser_states`.
///
/// The product carries both remaining path weights and the common mask already
/// accepted on earlier prefixes. Different current masks are a witness. Equal
/// accepted coordinates can be removed from both remaining path weights: later
/// acceptance cannot change their membership in the union. A visited product
/// is therefore a complete residual observation, not just a pair of DFA IDs.
///
/// At each product, only explicit outgoing labels plus one representative of
/// all remaining labels are needed. Every unnamed label follows the same two
/// DEFAULT/missing-edge choices, proving that this finite alphabet partition
/// covers the full parser-state domain. Raw state labels are required; expand
/// any additional runtime domain aliases before calling this function.
///
/// A resource limit returns Err (UNKNOWN), never an equivalence claim. Hash
/// collisions are resolved by full structural equality of the product/weights.
#[cfg(test)]
fn compare_parser_mask_prefix_languages_history_reference(
    left: &DWA, right: &DWA, num_parser_states: u32, max_product_states: usize,
) -> Result<ParserMaskComparison, String> {
    if max_product_states == 0 || num_parser_states > DEFAULT_LABEL as u32 {
        return Err("invalid parser-mask comparison domain/budget".into());
    }
    for dwa in [left, right] {
        for row in dwa.states() {
            for (&label, (target, _)) in &row.transitions {
                if label != DEFAULT_LABEL && (label < 0 || label as u32 >= num_parser_states) {
                    return Err(format!("parser comparison requires raw parser-state labels; found {label}"));
                }
                if *target as usize >= dwa.states().len() {
                    return Err("parser comparison found an invalid transition target".into());
                }
            }
        }
    }
    let start = Product {
        left: Some(left.start_state()), right: Some(right.start_state()),
        left_weight: Weight::all(), right_weight: Weight::all(), accepted: Weight::empty(),
    };
    let mut seen = FxHashSet::default();
    seen.insert(start.clone());
    let mut pending = VecDeque::from([(start, Vec::<u32>::new())]);
    let mut ops = ScopedWeightOpCache::default();
    let mut compared_branches = 0usize;
    while let Some((current, word)) = pending.pop_front() {
        let extra_left = contribution(left, current.left, &current.left_weight, &mut ops);
        let extra_right = contribution(right, current.right, &current.right_weight, &mut ops);
        let left_mask = ops.union(&current.accepted, &extra_left);
        let right_mask = ops.union(&current.accepted, &extra_right);
        if left_mask != right_mask {
            return Ok(ParserMaskComparison {
                difference: Some(ParserMaskDifference { stack_top_first: word, left_mask, right_mask }),
                product_states: seen.len(), compared_branches,
            });
        }
        let left_remaining = ops.difference(&current.left_weight, &left_mask);
        let right_remaining = ops.difference(&current.right_weight, &left_mask);
        if left_remaining.is_empty() && right_remaining.is_empty() { continue; }
        let mut labels = BTreeSet::<u32>::new();
        for (dwa, state, remaining) in [
            (left, current.left, &left_remaining), (right, current.right, &right_remaining),
        ] {
            if !remaining.is_empty() {
                if let Some(row) = state.and_then(|state| dwa.states().get(state as usize)) {
                    labels.extend(row.transitions.keys().filter_map(|&label| {
                        (label != DEFAULT_LABEL).then_some(label as u32)
                    }));
                }
            }
        }
        let mut unnamed = 0u32;
        for &label in &labels {
            if label == unnamed { unnamed += 1; } else if label > unnamed { break; }
        }
        if unnamed < num_parser_states { labels.insert(unnamed); }
        for parser in labels {
            compared_branches += 1;
            let (left_state, left_weight) = advance(left, current.left, &left_remaining, parser, &mut ops);
            let (right_state, right_weight) = advance(right, current.right, &right_remaining, parser, &mut ops);
            if left_weight.is_empty() && right_weight.is_empty() { continue; }
            let next = Product {
                left: left_state, right: right_state, left_weight, right_weight,
                accepted: left_mask.clone(),
            };
            if seen.contains(&next) { continue; }
            if seen.len() >= max_product_states {
                return Err(format!("parser-mask comparison budget exhausted at {} products", seen.len()));
            }
            seen.insert(next.clone());
            let mut next_word = word.clone(); next_word.push(parser);
            pending.push_back((next, next_word));
        }
    }
    Ok(ParserMaskComparison { difference: None, product_states: seen.len(), compared_branches })
}

fn evaluate_prefix_mask(dwa: &DWA, word: &[u32]) -> Weight {
    let mut state = Some(dwa.start_state());
    let mut path = Weight::all();
    let mut accepted = Weight::empty();
    let mut ops = ScopedWeightOpCache::default();
    for depth in 0..=word.len() {
        let extra = contribution(dwa, state, &path, &mut ops);
        accepted = ops.union(&accepted, &extra);
        if depth == word.len() { break; }
        (state, path) = advance(dwa, state, &path, word[depth], &mut ops);
        if path.is_empty() { break; }
    }
    accepted
}

/// Exact DEFAULT-fallback, prefix-mask equivalence over every concrete stack.
///
/// A product pair carries a region D of original weight coordinates for which
/// neither side has accepted at an earlier prefix. Both paths are alive on D;
/// a dead side is represented explicitly by None. At a transition, split D
/// into three regions: both edges admit, only the left admits, only the right
/// admits. The neither-admits region can never yield a future difference.
/// Compare final membership and permanently remove coordinates accepted on
/// both sides. Consequently future behavior depends only on the state pair
/// and coordinate, not on the path history. Unioning visited regions per pair
/// is exact and avoids enumerating combinations of accumulated path weights.
///
/// For each state pair, the union of explicit labels plus one unnamed label
/// covers all distinct successor choices, with explicit edges overriding
/// DEFAULT. Full structural weight operations retain coordinate correlations.
/// Limits return Err/unknown. Extra runtime state-domain aliases must be
/// expanded before comparison; this function accepts raw positive labels.
/// The region universe is represented explicitly: `Weight::all()` is an
/// identity sentinel whose subtraction is intentionally only conservative.
/// Raw TSID u32::MAX is reserved, so the compact rectangle below covers all
/// valid coordinates without colliding with that sentinel. No IDs are
/// enumerated; it consists of one outer and one inner interval.
pub fn compare_parser_mask_prefix_languages(
    left: &DWA, right: &DWA, num_parser_states: u32, max_product_states: usize,
) -> Result<ParserMaskComparison, String> {
    if max_product_states == 0 || num_parser_states > DEFAULT_LABEL as u32 {
        return Err("invalid parser-mask comparison domain/budget".into());
    }
    for dwa in [left, right] {
        for row in dwa.states() {
            for weight in row.final_weight.iter().chain(row.transitions.values().map(|(_, weight)| weight)) {
                if !weight.is_full() && weight.range_entries().any(|(_, hi, _)| hi == u32::MAX) {
                    return Err("parser comparison found a reserved raw TSID in a finite weight".into());
                }
            }
            for (&label, (target, _)) in &row.transitions {
                if label != DEFAULT_LABEL && (label < 0 || label as u32 >= num_parser_states) {
                    return Err(format!("parser comparison requires raw parser-state labels; found {label}"));
                }
                if *target as usize >= dwa.states().len() {
                    return Err("parser comparison found an invalid transition target".into());
                }
            }
        }
    }
    type Pair = (Option<u32>, Option<u32>);
    let initial = (Some(left.start_state()), Some(right.start_state()));
    let universe = Weight::from_uniform(
        0..=u32::MAX - 1, RangeSetBlaze::from_iter([0..=u32::MAX]),
    );
    assert!(!universe.is_full(), "the explicit comparison universe must not become the identity sentinel");
    let mut visited = FxHashMap::<Pair, Weight>::default();
    visited.insert(initial, universe.clone());
    let mut queue = VecDeque::from([(initial, universe, Vec::<u32>::new())]);
    let mut ops = ScopedWeightOpCache::default();
    let mut branches = 0;
    let mut region_visits = 0usize;
    while let Some(((ls, rs), domain, word)) = queue.pop_front() {
        region_visits += 1;
        if region_visits > max_product_states {
            return Err(format!("parser-mask comparison budget exhausted after {region_visits} region visits at {} pairs", visited.len()));
        }
        let lf = contribution(left, ls, &domain, &mut ops);
        let rf = contribution(right, rs, &domain, &mut ops);
        if lf != rf {
            return Ok(ParserMaskComparison {
                difference: Some(ParserMaskDifference {
                    left_mask: evaluate_prefix_mask(left, &word),
                    right_mask: evaluate_prefix_mask(right, &word),
                    stack_top_first: word,
                }),
                product_states: visited.len(), compared_branches: branches,
            });
        }
        let active = ops.difference(&domain, &lf);
        if active.is_empty() { continue; }
        let mut labels = BTreeSet::<u32>::new();
        for (dwa, state) in [(left, ls), (right, rs)] {
            if let Some(row) = state.and_then(|state| dwa.states().get(state as usize)) {
                labels.extend(row.transitions.keys().filter_map(|&label| {
                    (label != DEFAULT_LABEL).then_some(label as u32)
                }));
            }
        }
        let mut unnamed = 0u32;
        for &label in &labels {
            if label == unnamed { unnamed += 1; } else if label > unnamed { break; }
        }
        if unnamed < num_parser_states { labels.insert(unnamed); }
        for label in labels {
            branches += 1;
            let (lt, lw) = advance(left, ls, &active, label, &mut ops);
            let (rt, rw) = advance(right, rs, &active, label, &mut ops);
            let both = ops.intersection(&lw, &rw);
            let only_left = ops.difference(&lw, &rw);
            let only_right = ops.difference(&rw, &lw);
            for (pair, region) in [((lt, rt), both), ((lt, None), only_left), ((None, rt), only_right)] {
                if region.is_empty() { continue; }
                let novel = visited.get(&pair).map_or_else(|| region.clone(), |old| ops.difference(&region, old));
                if novel.is_empty() { continue; }
                if visited.len() >= max_product_states && !visited.contains_key(&pair) {
                    return Err(format!("parser-mask comparison budget exhausted at {} pairs", visited.len()));
                }
                visited.entry(pair).and_modify(|old| *old = ops.union(old, &novel)).or_insert_with(|| novel.clone());
                let mut child_word = word.clone(); child_word.push(label);
                queue.push_back((pair, novel, child_word));
            }
        }
    }
    Ok(ParserMaskComparison { difference: None, product_states: visited.len(), compared_branches: branches })
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    fn token(id: u32) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([id..=id]))
    }

    fn evaluate(dwa: &DWA, word: &[u32]) -> Weight {
        let mut state = dwa.start_state();
        let mut path = Weight::all();
        let mut accepted = Weight::empty();
        for depth in 0..=word.len() {
            let Some(row) = dwa.states().get(state as usize) else { break; };
            if let Some(final_weight) = &row.final_weight {
                accepted = accepted.union(&path.intersection(final_weight));
            }
            if depth == word.len() { break; }
            let Some((next, weight)) = row.transitions.get(&(word[depth] as i32))
                .or_else(|| row.transitions.get(&DEFAULT_LABEL)) else { break; };
            path = path.intersection(weight); state = *next;
        }
        accepted
    }

    #[test]
    fn parser_mask_equivalence_handles_default_and_prefix_acceptance() {
        let mut left = DWA::new(1, 2); let leaf = left.add_state();
        left.add_transition(0, DEFAULT_LABEL, leaf, Weight::all()); left.set_final_weight(leaf, token(0));
        let mut right = DWA::new(1, 2); let leaf = right.add_state();
        right.add_transition(0, 0, leaf, Weight::all()); right.add_transition(0, 1, leaf, Weight::all());
        right.set_final_weight(leaf, token(0));
        assert!(crate::automata::weighted::equivalence::find_difference(&left, &right).unwrap().is_some());
        assert!(compare_parser_mask_prefix_languages(&left, &right, 2, 100).unwrap().difference.is_none());
        let extra = right.add_state(); right.add_transition(leaf, 0, extra, Weight::all());
        right.set_final_weight(extra, token(0));
        assert!(compare_parser_mask_prefix_languages(&left, &right, 2, 100).unwrap().difference.is_none());
        right.set_final_weight(extra, token(1));
        let difference = compare_parser_mask_prefix_languages(&left, &right, 2, 100).unwrap().difference.unwrap();
        assert_eq!(difference.stack_top_first.len(), 2);
    }

    #[test]
    fn parser_mask_equivalence_respects_explicit_override_and_budget() {
        let mut left = DWA::new(1, 1); let leaf = left.add_state(); let dead = left.add_state();
        left.add_transition(0, DEFAULT_LABEL, leaf, Weight::all()); left.set_final_weight(leaf, token(0));
        let right = left.clone();
        left.add_transition(0, 0, dead, Weight::all());
        assert_eq!(compare_parser_mask_prefix_languages(&left, &right, 2, 100).unwrap()
            .difference.unwrap().stack_top_first, vec![0]);
        assert!(compare_parser_mask_prefix_languages(&right, &right, 2, 1).is_err());
        let mut cycle = DWA::new(1, 1); cycle.add_transition(0, DEFAULT_LABEL, 0, Weight::all());
        assert!(compare_parser_mask_prefix_languages(&cycle, &DWA::new(1, 1), 1000, 10).unwrap().difference.is_none());
    }

    #[test]
    fn parser_mask_equivalence_matches_exhaustive_generated_stacks() {
        let mut rng = 43u64;
        let mut next = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 32) as usize };
        let weights = [Weight::empty(), token(0), token(1), token(0).union(&token(1)), Weight::all()];
        for case in 0..128 {
            let mut pair = Vec::new();
            for _ in 0..2 {
                let n = 3 + next() % 3; let mut dwa = DWA::new(1, 2);
                for _ in 1..n { dwa.add_state(); }
                for source in 0..n {
                    if next() % 3 == 0 { dwa.set_final_weight(source as u32, weights[next() % weights.len()].clone()); }
                    if source + 1 == n { continue; }
                    for label in [0, 1, 2, DEFAULT_LABEL] {
                        if next() % 3 == 0 {
                            let target = source + 1 + next() % (n - source - 1);
                            dwa.add_transition(source as u32, label, target as u32, weights[next() % weights.len()].clone());
                        }
                    }
                }
                pair.push(dwa);
            }
            let mut words = VecDeque::from([Vec::<u32>::new()]);
            let max_depth = pair.iter().map(|d| d.num_states() as usize).max().unwrap();
            let mut expected = None;
            while let Some(word) = words.pop_front() {
                if evaluate(&pair[0], &word) != evaluate(&pair[1], &word) { expected = Some(word); break; }
                if word.len() < max_depth {
                    for label in 0..3 { let mut child = word.clone(); child.push(label); words.push_back(child); }
                }
            }
            let actual = compare_parser_mask_prefix_languages(&pair[0], &pair[1], 3, 10_000).unwrap();
            let history = compare_parser_mask_prefix_languages_history_reference(&pair[0], &pair[1], 3, 10_000).unwrap();
            assert_eq!(actual.difference.is_some(), expected.is_some(), "case {case}");
            assert_eq!(actual.difference.is_some(), history.difference.is_some(), "history reference case {case}");
            let identity = compare_parser_mask_prefix_languages(&pair[0], &pair[0], 3, 10_000).unwrap();
            assert!(identity.difference.is_none(), "nontrivial self equivalence case {case}");
            if let Some(witness) = actual.difference {
                assert_ne!(evaluate(&pair[0], &witness.stack_top_first), evaluate(&pair[1], &witness.stack_top_first));
                assert_eq!(witness.stack_top_first.len(), expected.unwrap().len(), "shortest witness case {case}");
            }
        }
    }
}
