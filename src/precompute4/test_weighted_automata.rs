use crate::precompute4::weighted_automata::{DWA, DWABuildError, NWA, NWABuildError, Weight, format_word, SimpleBitset, DWAState};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::precompute4::resolve_negatives::resolve_negative_codes_in_dwa;
use crate::precompute4::weighted_automata::common::Label;
use crate::json_serialization::JSONConvertible;

// --- Stochastic validation controls and RNG ---
const VALIDATION_SAMPLES: usize = 32;
const VALIDATION_MAX_STEPS: usize = 32;
const SAMPLING_TRIES: usize = 100;

#[derive(Clone, Debug)]
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { SimpleRng(seed) }
    fn from_time() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let mixed = (now.as_nanos() as u128 ^ ((now.as_secs() as u128) << 64)) as u64;
        SimpleRng::new(mixed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn gen_usize(&mut self, upper: usize) -> usize { if upper <= 1 { 0 } else { (self.next_u64() as usize) % upper } }
    fn gen_bool_ratio(&mut self, num: u32, den: u32) -> bool { if den == 0 { true } else { (self.next_u64() % (den as u64)) < (num as u64) } }
}

fn weight_subset(sub: &Weight, sup: &Weight) -> bool { (sub & sup) == sub.clone() }

// Helper extension to support sampling on DWA
trait DwaTestExt {
    fn sample_accepted_path(&self, max_steps: usize) -> Option<(Vec<Label>, Weight)>;
    fn sample_accepted_path_with_rng(&self, rng: &mut SimpleRng, max_steps: usize) -> Option<(Vec<Label>, Weight)>;
}

impl DwaTestExt for DWA {
    fn sample_accepted_path(&self, max_steps: usize) -> Option<(Vec<Label>, Weight)> {
        let mut rng = SimpleRng::from_time();
        self.sample_accepted_path_with_rng(&mut rng, max_steps)
    }

    fn sample_accepted_path_with_rng(&self, rng: &mut SimpleRng, max_steps: usize) -> Option<(Vec<Label>, Weight)> {
        if self.states.0.is_empty() { return None; }
        for _attempt in 0..SAMPLING_TRIES {
            let mut word: Vec<Label> = Vec::new();
            let mut s = self.body.start_state;
            let mut acc = Weight::all();
            if let Some(sw) = &self.states[s].state_weight { acc &= sw; if acc.is_empty() { return None; } }

            for step in 0..max_steps {
                if let Some(fw) = &self.states[s].final_weight {
                    if rng.gen_bool_ratio(1, 3) || step == max_steps - 1 {
                        let w = &acc & fw;
                        if !w.is_empty() { return Some((word, w)); }
                    }
                }
                let st = &self.states[s];
                let choices: Vec<Label> = st.transitions.keys().copied().collect();
                let total = choices.len();
                if total == 0 {
                    if let Some(fw) = &st.final_weight {
                        let w = &acc & fw;
                        if !w.is_empty() { return Some((word, w)); }
                    }
                    break;
                }
                let pick = rng.gen_usize(total);
                let ch = choices[pick];
                let next = st.transitions.get(&ch).copied();
                if next.is_none() { break; }
                let edge_w = st.trans_weights.get(&ch).cloned().unwrap_or_else(Weight::zeros);
                if edge_w.is_empty() { break; }
                let new_acc = &acc & &edge_w;
                if new_acc.is_empty() { break; }
                acc = new_acc;
                s = next.unwrap();
                word.push(ch);
                if let Some(sw) = &self.states[s].state_weight {
                    acc &= sw; if acc.is_empty() { break; }
                }
            }
            if s < self.states.len() {
                if let Some(fw) = &self.states[s].final_weight {
                    let w = &acc & fw;
                    if !w.is_empty() { return Some((word, w)); }
                }
            }
        }
        None
    }
}

fn expected_union_weight(a: &DWA, b: &DWA, word: &[Label]) -> Weight {
    let wa = a.eval_word_weight(word);
    let wb = b.eval_word_weight(word);
    &wa | &wb
}

fn expected_concat_weight(a: &DWA, b: &DWA, word: &[Label], eps_weight: &Weight) -> Weight {
    let mut acc = Weight::zeros();
    for i in 0..=word.len() {
        let wa = a.eval_word_weight(&word[..i]);
        if wa.is_empty() { continue; }
        let wb = b.eval_word_weight(&word[i..]);
        if wb.is_empty() { continue; }
        let both = &(&wa & &wb) & eps_weight;
        if !both.is_empty() { acc |= &both; }
    }
    acc
}

fn stochastic_validate_union(a: &DWA, b: &DWA, u: &DWA) {
    let mut rng = SimpleRng::from_time();
    for _ in 0..VALIDATION_SAMPLES {
        if let Some((w, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let wu = u.eval_word_weight(&w);
            assert!(!wu.is_empty(), "Union rejected word from A: {}\nA(w): {}\nU(w): {}", format_word(&w), wa, wu);
            assert!(weight_subset(&wa, &wu), "Union weight too small for A.\nword: {}\nA(w): {}\nU(w): {}", format_word(&w), wa, wu);
        }
        if let Some((w, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let wu = u.eval_word_weight(&w);
            assert!(!wu.is_empty(), "Union rejected word from B: {}", format_word(&w));
            assert!(weight_subset(&wb, &wu), "Union weight too small for B.\nword: {}\nB(w): {}\nU(w): {}", format_word(&w), wb, wu);
        }
        if let Some((w, wu)) = u.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let expected = expected_union_weight(a, b, &w);
            assert_eq!(wu, expected, "Union mismatch.\nword: {}\nU(w): {}\nExp: {}", format_word(&w), wu, expected);
        }
    }
}

fn stochastic_validate_concatenate(a: &DWA, b: &DWA, c: &DWA, eps_weight: &Weight) {
    let mut rng = SimpleRng::from_time();
    for _ in 0..VALIDATION_SAMPLES {
        if let Some((wa_word, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            if let Some((wb_word, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let mut w = wa_word.clone();
                w.extend_from_slice(&wb_word);
                let wc = c.eval_word_weight(&w);
                let expected_simple = &(&wa & &wb) & eps_weight;
                if !expected_simple.is_empty() {
                    assert!(weight_subset(&expected_simple, &wc), "Concat missing subset.\nWord: {}", format_word(&w));
                }
                let expected_all = expected_concat_weight(a, b, &w, eps_weight);
                assert_eq!(wc, expected_all, "Concat mismatch.\nWord: {}\nGot: {}\nExp: {}", format_word(&w), wc, expected_all);
            }
        }
        if let Some((w, wc)) = c.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let expected = expected_concat_weight(a, b, &w, eps_weight);
            assert_eq!(wc, expected, "Concat spurious accept.\nWord: {}\nGot: {}\nExp: {}", format_word(&w), wc, expected);
        }
    }
}

pub fn stochastic_equivalence_test(a: DWA, b: DWA) {
    let mut rng = SimpleRng::from_time();
    for _ in 0..VALIDATION_SAMPLES {
        if let Some((w, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let wb = b.eval_word_weight(&w);
            assert_eq!(wa, wb, "A(w) != B(w). Word: {}", format_word(&w));
        }
        if let Some((w, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let wa = a.eval_word_weight(&w);
            assert_eq!(wb, wa, "B(w) != A(w). Word: {}", format_word(&w));
        }
    }
}

#[test]
fn test_simple_bitset_ops() {
    let set1 = SimpleBitset::from_iter(vec![1, 2, 5]);
    let set2 = SimpleBitset::from_iter(vec![2, 3, 5, 6]);
    let all = SimpleBitset::all();
    let zeros = SimpleBitset::zeros();
    assert_eq!((&set1 & &set2).iter_up_to(10).collect::<Vec<_>>(), vec![2, 5]);
    assert_eq!((&set1 | &set2).iter_up_to(10).collect::<Vec<_>>(), vec![1, 2, 3, 5, 6]);
    assert!((&set1 & &all).contains(1));
    assert!((&set1 | &zeros).contains(1));
}

#[test]
fn test_dwa_builder() {
    let mut dwa = DWA::new();
    let s1 = dwa.add_state();
    dwa.set_final_weight(1, SimpleBitset::from_item(20)).unwrap();
    dwa.add_transition(0, b'a' as Label, 1, SimpleBitset::from_item(30)).unwrap();
    assert_eq!(*dwa.states[0].transitions.get(&(b'a' as Label)).unwrap(), 1);
    let res = dwa.add_transition(0, b'a' as Label, 1, SimpleBitset::zeros());
    assert!(matches!(res, Err(DWABuildError::TransitionAlreadyExists { .. })));
    let res = dwa.set_final_weight(10, SimpleBitset::zeros());
    assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { .. })));
}

fn dwa_accepts_char(ch: char, final_weight: Weight) -> DWA {
    let mut dwa = DWA::new();
    let final_state = dwa.add_state();
    dwa.add_transition(dwa.body.start_state, ch as Label, final_state, Weight::all()).unwrap();
    dwa.set_final_weight(final_state, final_weight).unwrap();
    dwa
}

fn dwa_from_str(s: &str, final_weight: Weight) -> DWA {
    let mut dwa = DWA::new();
    let mut current_state = dwa.body.start_state;
    for ch in s.chars() {
        let next_state = dwa.add_state();
        dwa.add_transition(current_state, ch as Label, next_state, Weight::all()).unwrap();
        current_state = next_state;
    }
    dwa.set_final_weight(current_state, final_weight).unwrap();
    dwa
}

#[test]
fn test_simplify_redundant_states() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();
    let _s5 = d.add_state(); // unreachable
    d.add_transition(0, 'a' as Label, s1, Weight::all()).unwrap();
    d.add_transition(0, 'b' as Label, s2, Weight::all()).unwrap();
    d.add_transition(0, 'c' as Label, s3, Weight::all()).unwrap();
    d.add_transition(s1, 'x' as Label, s4, Weight::all()).unwrap();
    d.add_transition(s2, 'y' as Label, s4, Weight::all()).unwrap();
    d.add_transition(s3, 'y' as Label, s4, Weight::all()).unwrap();
    d.set_final_weight(s4, Weight::from_item(1)).unwrap();
    d.simplify();
    // s2 and s3 should merge. s5 removed. 0, 1, 2(merged), 4. Total 4.
    assert_eq!(d.states.len(), 4);
}

#[test]
fn test_union_simple() {
    let d1 = dwa_accepts_char('a', Weight::from_item(1));
    let d2 = dwa_accepts_char('b', Weight::from_item(2));
    let u = NWA::union(&NWA::from_dwa(&d1), &NWA::from_dwa(&d2)).determinize_to_dwa();
    
    let mut expected = DWA::new();
    let s_a = expected.add_state();
    let s_b = expected.add_state();
    expected.add_transition(0, 'a' as Label, s_a, Weight::all()).unwrap();
    expected.add_transition(0, 'b' as Label, s_b, Weight::all()).unwrap();
    expected.set_final_weight(s_a, Weight::from_item(1)).unwrap();
    expected.set_final_weight(s_b, Weight::from_item(2)).unwrap();
    stochastic_equivalence_test(u, expected);
}

#[test]
fn test_union_overlapping() {
    let d1 = dwa_accepts_char('a', Weight::from_item(1));
    let mut d2 = dwa_accepts_char('b', Weight::from_item(3));
    let s_a2 = d2.add_state();
    d2.add_transition(d2.body.start_state, 'a' as Label, s_a2, Weight::all()).unwrap();
    d2.set_final_weight(s_a2, Weight::from_item(2)).unwrap();

    let u = NWA::union(&NWA::from_dwa(&d1), &NWA::from_dwa(&d2)).determinize_to_dwa();
    
    let mut expected = DWA::new();
    let s_a = expected.add_state();
    let s_b = expected.add_state();
    expected.add_transition(0, 'a' as Label, s_a, Weight::all()).unwrap();
    expected.add_transition(0, 'b' as Label, s_b, Weight::all()).unwrap();
    expected.set_final_weight(s_a, Weight::from_iter(vec![1, 2])).unwrap();
    expected.set_final_weight(s_b, Weight::from_item(3)).unwrap();
    stochastic_equivalence_test(u, expected);
}

#[test]
fn test_concatenate_simple() {
    let d1 = dwa_accepts_char('a', Weight::from_iter([1, 2]));
    let d2 = dwa_accepts_char('b', Weight::from_iter([2, 3]));
    let c = NWA::concatenate(&NWA::from_dwa(&d1), &NWA::from_dwa(&d2)).determinize_to_dwa();
    let expected = dwa_from_str("ab", Weight::from_item(2));
    stochastic_equivalence_test(c, expected);
}

#[test]
fn test_apply_weight_inplace() {
    let mut d = DWA::new();
    d.set_final_weight(0, Weight::from_iter(vec![5, 6])).unwrap();
    let gate = Weight::from_iter(vec![6, 11]);
    d.apply_weight_inplace(&gate);
    assert_eq!(d.states[0].final_weight, Some(Weight::from_item(6)));
}

#[test]
fn test_union_transition_weight_union() {
    fn build(ch: char, ew: usize, fw: usize) -> DWA {
        let mut d = DWA::new();
        let s = d.add_state();
        d.add_transition(0, ch as Label, s, Weight::from_item(ew)).unwrap();
        d.set_final_weight(s, Weight::from_item(fw)).unwrap();
        d
    }
    let d1 = build('x', 10, 1);
    let d2 = build('x', 20, 2);
    let u = NWA::union(&NWA::from_dwa(&d1), &NWA::from_dwa(&d2)).determinize_to_dwa();

    let mut expected = DWA::new();
    let s = expected.add_state();
    expected.add_transition(0, 'x' as Label, s, Weight::from_iter(vec![10, 20])).unwrap();
    expected.set_final_weight(s, Weight::from_iter(vec![1, 2])).unwrap();
    stochastic_equivalence_test(u, expected);
}

#[test]
fn test_json_roundtrip_complex() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    d.add_transition(d.body.start_state, 'y' as Label, s1, Weight::from_iter(vec![1, 2, 3])).unwrap();
    d.add_transition(d.body.start_state, 'x' as Label, s2, Weight::from_item(99)).unwrap();
    d.set_final_weight(s2, Weight::from_iter(vec![5, 7])).unwrap();

    let node = d.to_json();
    let d2 = DWA::from_json(node.clone()).expect("from_json should succeed");
    assert_eq!(node, d2.to_json(), "Roundtrip JSON stable");
}

#[test]
fn test_add_transition_out_of_bounds() {
    let mut d = DWA::new();
    let res = d.add_transition(5, 'a' as Label, 0, Weight::zeros());
    assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { state: 5 })));
}

#[test]
fn test_prune_unreachable_with_default_chain() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let _s2 = d.add_state(); 
    d.add_transition(d.body.start_state, 'y' as Label, s1, Weight::all()).unwrap();
    d.set_final_weight(s1, Weight::from_item(1)).unwrap();
    d.add_transition(s1, 'x' as Label, s1, Weight::all()).unwrap();
    let s_unreach = d.add_state();
    d.add_transition(s_unreach, 'z' as Label, s_unreach, Weight::all()).unwrap();

    let before = d.states.len();
    d.simplify();
    let after = d.states.len();
    assert!(after < before);
    assert_eq!(after, 2);
}

#[test]
fn test_equivalence_via_simplification() {
    let mut a = DWA::new();
    let s1a = a.add_state();
    let s2a = a.add_state();
    a.add_transition(0, 0, s1a, Weight::from_item(1)).unwrap();
    a.add_transition(0, 1, s2a, Weight::from_iter(0..=1)).unwrap();
    a.add_transition(0, 2, s1a, Weight::from_item(0)).unwrap();
    a.add_transition(0, 3, s1a, Weight::from_iter(0..=1)).unwrap();

    let mut b = DWA::new();
    let s1b = b.add_state();
    b.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    b.add_transition(0, 2, s1b, Weight::from_item(0)).unwrap();

    stochastic_equivalence_test(a, b);
}

#[test]
fn test_concatenate_left_start_is_final() {
    let mut left = DWA::new();
    left.set_final_weight(left.body.start_state, Weight::from_iter([0, 1])).unwrap();
    let mut right = DWA::new();
    right.set_final_weight(right.body.start_state, Weight::from_iter([1, 2])).unwrap();
    let c = NWA::concatenate(&NWA::from_dwa(&left), &NWA::from_dwa(&right)).determinize_to_dwa();
    let mut expected = DWA::new();
    expected.set_final_weight(expected.body.start_state, Weight::from_item(1)).unwrap();
    stochastic_equivalence_test(c, expected);
}

#[test]
fn test_simplify_propagates_future_weights() {
    let mut a = DWA::new();
    let s1 = a.add_state();
    let s2 = a.add_state();
    a.add_transition(0, 'a' as Label, s1, Weight::all()).unwrap();
    a.add_transition(s1, 'b' as Label, s2, Weight::from_iter([1..=2])).unwrap();
    a.set_final_weight(s2, Weight::from_item(2)).unwrap();

    let mut b = DWA::new();
    let s1_b = b.add_state();
    let s2_b = b.add_state();
    b.add_transition(0, 'a' as Label, s1_b, Weight::all()).unwrap();
    b.add_transition(s1_b, 'b' as Label, s2_b, Weight::all()).unwrap();
    b.set_final_weight(s2_b, Weight::from_item(2)).unwrap();

    a.simplify();
    stochastic_equivalence_test(a, b);
}

#[test]
fn test_union_identical_cyclic() {
    let mut d1 = DWA::new();
    d1.add_transition(0, 'a' as Label, 0, Weight::all()).unwrap();
    d1.set_final_weight(0, Weight::from_item(1)).unwrap();
    let d2 = d1.clone();
    let u = NWA::union(&NWA::from_dwa(&d1), &NWA::from_dwa(&d2)).determinize_to_dwa();
    stochastic_equivalence_test(u, d1);
}

#[test]
fn test_concatenate_disjoint_weights() {
    fn neg(x: Label) -> Label { Label::MIN + x }
    let word_a = vec![10, 5, 3, neg(3), neg(0), neg(9)];
    let mut dwa_a = DWA::new();
    let mut current = dwa_a.body.start_state;
    for &ch in &word_a {
        let next = dwa_a.add_state();
        dwa_a.add_transition(current, ch, next, Weight::all()).unwrap();
        current = next;
    }
    dwa_a.set_final_weight(current, Weight::from_item(1)).unwrap();

    let word_b = vec![9, 3, neg(3), neg(5), neg(10), 9, 7, neg(7), neg(5), neg(10)];
    let mut dwa_b = DWA::new();
    current = dwa_b.body.start_state;
    for &ch in &word_b {
        let next = dwa_b.add_state();
        dwa_b.add_transition(current, ch, next, Weight::all()).unwrap();
        current = next;
    }
    dwa_b.set_final_weight(current, Weight::from_item(0)).unwrap();

    let c = NWA::concatenate(&NWA::from_dwa(&dwa_a), &NWA::from_dwa(&dwa_b)).determinize_to_dwa();
    let mut combined = word_a.clone();
    combined.extend_from_slice(&word_b);
    let wc = c.eval_word_weight(&combined);
    assert_eq!(wc, Weight::zeros());
}

#[test]
fn test_concatenate_default_path_to_final() {
    let mut a = DWA::new();
    let s1a = a.add_state();
    a.add_transition(0, 'a' as Label, s1a, Weight::all()).unwrap();
    a.set_final_weight(s1a, Weight::from_item(1)).unwrap();

    let mut b = DWA::new();
    let s1b = b.add_state();
    b.add_transition(0, 'x' as Label, s1b, Weight::all()).unwrap();
    b.set_final_weight(s1b, Weight::from_item(1)).unwrap();

    let c = NWA::concatenate(&NWA::from_dwa(&a), &NWA::from_dwa(&b)).determinize_to_dwa();
    let weight = c.eval_word_weight(&['a' as Label, 'x' as Label]);
    assert_eq!(weight, Weight::from_item(1));
    let weight_x = c.eval_word_weight(&['x' as Label]);
    assert_eq!(weight_x, Weight::zeros());
}

#[test]
fn test_dwa_to_nwa_to_dwa_roundtrip() {
    let mut a = DWA::new();
    let s1 = a.add_state();
    a.add_transition(0, 1, s1, Weight::all()).unwrap();
    a.set_final_weight(s1, Weight::all()).unwrap();
    
    let nwa = NWA::from_dwa(&a);
    let roundtrip = nwa.determinize_to_dwa();
    stochastic_equivalence_test(a, roundtrip);
}

#[test]
fn test_union_complex_from_attachment() {
    fn neg(x: Label) -> Label { Label::MIN + x }
    let mut left = DWA::new();
    for _ in 0..47 { left.add_state(); }
    left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    left.add_transition(0, 2, 2, Weight::from_item(1)).unwrap();
    left.add_transition(1, neg(0), 11, Weight::all()).unwrap();
    left.set_final_weight(45, Weight::all()).unwrap();

    let mut right = DWA::new();
    for _ in 0..42 { right.add_state(); }
    right.add_transition(0, 2, 1, Weight::from_item(0)).unwrap();
    right.set_final_weight(40, Weight::all()).unwrap();

    let u = NWA::union(&NWA::from_dwa(&left), &NWA::from_dwa(&right)).determinize_to_dwa();
    stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified1() {
    let mut left = DWA::new();
    left.set_final_weight(0, Weight::from_item(0)).unwrap();
    let mut right = DWA::new();
    let s1b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.set_final_weight(s1b, Weight::all()).unwrap();
    let u = NWA::union(&NWA::from_dwa(&left), &NWA::from_dwa(&right)).determinize_to_dwa();
    stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified2() {
    let mut left = DWA::new();
    let s1a = left.add_state();
    left.add_transition(0, 0, s1a, Weight::from_item(0)).unwrap();
    left.set_final_weight(s1a, Weight::all()).unwrap();
    let mut right = DWA::new();
    let s1b = right.add_state();
    let s2b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.add_transition(s1b, 1, s2b, Weight::all()).unwrap();
    right.set_final_weight(s2b, Weight::all()).unwrap();
    let u = NWA::union(&NWA::from_dwa(&left), &NWA::from_dwa(&right)).determinize_to_dwa();
    stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_simplify_complex_dwa_from_attachment() {
    let mut left = DWA::new();
    for _ in 0..25 { left.add_state(); }
    left.body.start_state = 25;
    left.add_transition(25, 2, 9, Weight::all()).unwrap();
    left.set_final_weight(8, Weight::all()).unwrap();
    let mut simplified = left.clone();
    simplified.simplify();
    stochastic_equivalence_test(left, simplified);
}

#[test]
fn test_union_from_debug_log() {
    fn neg(x: Label) -> Label { Label::MIN + x }
    let mut left = DWA::new();
    for _ in 0..9 { left.add_state(); }
    left.set_final_weight(0, Weight::from_item(2)).unwrap();
    left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    left.add_transition(0, 1, 2, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(0, 2, 3, Weight::from_item(0)).unwrap();
    left.add_transition(0, 3, 4, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(1, neg(0), 5, Weight::all()).unwrap();
    left.add_transition(2, 100, 6, Weight::all()).unwrap();
    left.add_transition(3, neg(2), 7, Weight::all()).unwrap();
    left.add_transition(5, neg(1), 8, Weight::all()).unwrap();
    left.add_transition(7, neg(0), 9, Weight::all()).unwrap();
    left.set_final_weight(8, Weight::all()).unwrap();
    left.set_final_weight(9, Weight::all()).unwrap();

    let mut right = DWA::new();
    for _ in 0..12 { right.add_state(); }
    right.add_transition(0, 1, 1, Weight::from_item(3)).unwrap();
    right.add_transition(0, 2, 2, Weight::from_item(3)).unwrap();
    right.add_transition(0, 3, 3, Weight::from_item(3)).unwrap();
    right.add_transition(1, 100, 4, Weight::all()).unwrap();
    right.add_transition(2, neg(2), 5, Weight::all()).unwrap();
    right.add_transition(5, neg(0), 6, Weight::all()).unwrap();
    right.add_transition(6, 0, 7, Weight::from_item(3)).unwrap();
    right.add_transition(6, 1, 8, Weight::from_item(3)).unwrap();
    right.add_transition(6, 3, 9, Weight::from_item(3)).unwrap();
    right.add_transition(7, neg(0), 10, Weight::all()).unwrap();
    right.add_transition(8, 100, 11, Weight::all()).unwrap();
    right.add_transition(10, neg(1), 12, Weight::all()).unwrap();
    right.set_final_weight(12, Weight::all()).unwrap();

    let u = NWA::union(&NWA::from_dwa(&left), &NWA::from_dwa(&right)).determinize_to_dwa();
    stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified3() {
    let mut left = DWA::new();
    let s1a = left.add_state();
    let s2a = left.add_state();
    left.add_transition(0, 0, s1a, Weight::from_item(0)).unwrap();
    left.add_transition(s1a, 1, s2a, Weight::all()).unwrap();
    left.set_final_weight(s2a, Weight::all()).unwrap();

    let mut right = DWA::new();
    let s1b = right.add_state();
    let s2b = right.add_state();
    let s3b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.add_transition(s1b, 1, s2b, Weight::all()).unwrap();
    right.add_transition(s2b, 2, s3b, Weight::all()).unwrap();
    right.set_final_weight(s3b, Weight::all()).unwrap();

    let u = NWA::union(&NWA::from_dwa(&left), &NWA::from_dwa(&right)).determinize_to_dwa();
    stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_concatenate_from_debug_log() {
    fn neg(x: Label) -> Label { Label::MIN + x }
    let mut base_dwa = DWA::new();
    for _ in 0..12 { base_dwa.add_state(); }
    base_dwa.add_transition(0, 6, 1, Weight::all()).unwrap();
    base_dwa.add_transition(0, 7, 4, Weight::all()).unwrap();
    base_dwa.add_transition(0, 10, 5, Weight::all()).unwrap();
    base_dwa.add_transition(0, 11, 6, Weight::all()).unwrap();
    base_dwa.add_transition(0, 12, 3, Weight::all()).unwrap();
    base_dwa.add_transition(1, 9, 6, Weight::all()).unwrap();
    base_dwa.add_transition(2, 0, 7, Weight::all()).unwrap();
    base_dwa.add_transition(2, 4, 11, Weight::all()).unwrap();
    base_dwa.add_transition(2, 9, 12, Weight::all()).unwrap();
    base_dwa.add_transition(3, 6, 1, Weight::all()).unwrap();
    base_dwa.add_transition(4, 100, 1, Weight::all()).unwrap();
    base_dwa.add_transition(5, 100, 6, Weight::all()).unwrap();
    base_dwa.add_transition(6, 100, 2, Weight::all()).unwrap();
    base_dwa.add_transition(7, neg(0), 8, Weight::all()).unwrap();
    base_dwa.add_transition(8, neg(6), 9, Weight::all()).unwrap();
    base_dwa.add_transition(9, neg(12), 10, Weight::all()).unwrap();
    base_dwa.set_final_weight(10, Weight::all()).unwrap();
    base_dwa.add_transition(11, neg(4), 8, Weight::all()).unwrap();
    base_dwa.add_transition(12, neg(9), 8, Weight::all()).unwrap();

    let mut dwa1 = base_dwa.clone();
    dwa1.apply_weight_inplace(&Weight::from_item(0));
    let mut dwa2 = base_dwa.clone();
    dwa2.apply_weight_inplace(&Weight::from_item(0));

    let c = NWA::concatenate(&NWA::from_dwa(&dwa1), &NWA::from_dwa(&dwa2)).determinize_to_dwa();
    stochastic_validate_concatenate(&dwa1, &dwa2, &c, &Weight::all());
}

#[test]
fn test_union_from_panicked_log() {
    fn neg(x: Label) -> Label { Label::MIN + x }
    let mut a = DWA::new();
    for _ in 0..23 { a.add_state(); }
    a.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    a.add_transition(0, 1, 2, Weight::from_item(0)).unwrap();
    a.add_transition(0, 2, 3, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 3, 4, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 4, 5, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 5, 6, Weight::from_item(0)).unwrap();
    a.add_transition(0, 7, 7, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 8, 8, Weight::from_item(0)).unwrap();
    a.add_transition(0, 9, 9, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(1, neg(0), 10, Weight::from_item(1)).unwrap();
    a.add_transition(2, neg(1), 11, Weight::from_item(0)).unwrap();
    a.add_transition(3, 100, 12, Weight::from_item(1)).unwrap();
    a.add_transition(3, neg(2), 13, Weight::from_item(2)).unwrap();
    a.add_transition(4, neg(3), 13, Weight::from_item(2)).unwrap();
    a.add_transition(4, 5, 14, Weight::from_item(1)).unwrap();
    a.add_transition(5, 1, 15, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(5, 5, 16, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(5, 8, 17, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(6, neg(5), 11, Weight::from_item(0)).unwrap();
    a.add_transition(7, 1, 15, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(7, 5, 16, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(8, neg(8), 11, Weight::from_item(0)).unwrap();
    a.add_transition(9, 100, 17, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(10, neg(1), 18, Weight::from_item(1)).unwrap();
    a.add_transition(11, neg(4), 19, Weight::from_item(0)).unwrap();
    a.add_transition(12, 100, 20, Weight::from_item(1)).unwrap();
    a.add_transition(13, neg(8), 21, Weight::from_item(2)).unwrap();
    a.add_transition(14, neg(5), 1, Weight::from_item(1)).unwrap();
    a.add_transition(15, 100, 20, Weight::from_item(1)).unwrap();
    a.add_transition(15, neg(1), 22, Weight::from_item(2)).unwrap();
    a.add_transition(16, neg(5), 23, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(17, 100, 7, Weight::from_iter(1..=2)).unwrap();
    a.set_final_weight(18, Weight::from_item(1)).unwrap();
    a.set_final_weight(19, Weight::from_item(0)).unwrap();
    a.add_transition(20, 5, 14, Weight::from_item(1)).unwrap();
    a.set_final_weight(21, Weight::from_item(2)).unwrap();
    a.add_transition(22, neg(2), 13, Weight::from_item(2)).unwrap();
    a.add_transition(23, neg(0), 10, Weight::from_item(1)).unwrap();
    a.add_transition(23, neg(3), 13, Weight::from_item(2)).unwrap();

    let mut b = DWA::new();
    for _ in 0..16 { b.add_state(); }
    b.add_transition(0, 0, 1, Weight::from_item(3)).unwrap();
    b.add_transition(0, 2, 2, Weight::from_item(3)).unwrap();
    b.add_transition(0, 3, 3, Weight::from_item(3)).unwrap();
    b.add_transition(0, 4, 4, Weight::from_item(3)).unwrap();
    b.add_transition(0, 7, 5, Weight::from_item(3)).unwrap();
    b.add_transition(0, 9, 6, Weight::from_item(3)).unwrap();
    b.add_transition(1, neg(0), 7, Weight::from_item(3)).unwrap();
    b.add_transition(2, 100, 8, Weight::from_item(3)).unwrap();
    b.add_transition(3, 5, 9, Weight::from_item(3)).unwrap();
    b.add_transition(4, 1, 8, Weight::from_item(3)).unwrap();
    b.add_transition(4, 5, 9, Weight::from_item(3)).unwrap();
    b.add_transition(4, 8, 10, Weight::from_item(3)).unwrap();
    b.add_transition(5, 1, 8, Weight::from_item(3)).unwrap();
    b.add_transition(5, 5, 9, Weight::from_item(3)).unwrap();
    b.add_transition(6, 100, 10, Weight::from_item(3)).unwrap();
    b.add_transition(7, neg(1), 11, Weight::from_item(3)).unwrap();
    b.add_transition(8, 100, 3, Weight::from_item(3)).unwrap();
    b.add_transition(9, neg(5), 1, Weight::from_item(3)).unwrap();
    b.add_transition(10, 100, 5, Weight::from_item(3)).unwrap();
    b.add_transition(11, 1, 12, Weight::from_item(3)).unwrap();
    b.add_transition(11, 5, 13, Weight::from_item(3)).unwrap();
    b.add_transition(11, 8, 14, Weight::from_item(3)).unwrap();
    b.add_transition(12, neg(1), 15, Weight::from_item(3)).unwrap();
    b.add_transition(13, neg(5), 15, Weight::from_item(3)).unwrap();
    b.add_transition(14, neg(8), 15, Weight::from_item(3)).unwrap();
    b.add_transition(15, neg(4), 16, Weight::from_item(3)).unwrap();
    b.set_final_weight(16, Weight::from_item(3)).unwrap();

    let u = NWA::union(&NWA::from_dwa(&a), &NWA::from_dwa(&b)).determinize_to_dwa();
    stochastic_validate_union(&a, &b, &u);
}

#[cfg(test)]
mod determinization_tests {
    use super::*;
    use crate::precompute4::weighted_automata::{NWA, Weight};

    fn nwa_accepts_char(ch: char, weight: Weight) -> NWA {
        let mut nwa = NWA::new();
        let final_state = nwa.states.add_state();
        nwa.add_transition(nwa.body.start_states[0], ch as Label, final_state, Weight::all()).unwrap();
        nwa.states[final_state].final_weight = Some(weight);
        nwa
    }

    #[test]
    fn test_det_simple_char() {
        let nwa = nwa_accepts_char('a', Weight::from_item(1));
        let dwa = nwa.determinize_to_dwa();
        let expected = dwa_accepts_char('a', Weight::from_item(1));
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_union_of_chars() {
        let mut nwa = NWA::new();
        let start = nwa.body.start_states[0];
        let s_a = nwa.states.add_state();
        let s_b = nwa.states.add_state();
        let final_a = nwa.states.add_state();
        let final_b = nwa.states.add_state();
        nwa.add_epsilon(start, s_a, Weight::all());
        nwa.add_epsilon(start, s_b, Weight::all());
        nwa.add_transition(s_a, 'a' as Label, final_a, Weight::all()).unwrap();
        nwa.add_transition(s_b, 'b' as Label, final_b, Weight::all()).unwrap();
        nwa.states[final_a].final_weight = Some(Weight::from_item(1));
        nwa.states[final_b].final_weight = Some(Weight::from_item(2));

        let dwa = nwa.determinize_to_dwa();
        let mut expected = DWA::new();
        let final_a_dwa = expected.add_state();
        let final_b_dwa = expected.add_state();
        expected.add_transition(0, 'a' as Label, final_a_dwa, Weight::all()).unwrap();
        expected.add_transition(0, 'b' as Label, final_b_dwa, Weight::all()).unwrap();
        expected.set_final_weight(final_a_dwa, Weight::from_item(1)).unwrap();
        expected.set_final_weight(final_b_dwa, Weight::from_item(2)).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_nondeterminism_on_char() {
        let mut nwa = NWA::new();
        let start = nwa.body.start_states[0];
        let f1 = nwa.states.add_state();
        let f2 = nwa.states.add_state();
        nwa.add_transition(start, 'a' as Label, f1, Weight::from_item(1)).unwrap();
        nwa.add_transition(start, 'a' as Label, f2, Weight::from_item(2)).unwrap();
        nwa.states[f1].final_weight = Some(Weight::all());
        nwa.states[f2].final_weight = Some(Weight::all());

        let dwa = nwa.determinize_to_dwa();
        let mut expected = DWA::new();
        let final_state = expected.add_state();
        expected.add_transition(0, 'a' as Label, final_state, Weight::from_iter([1, 2])).unwrap();
        expected.set_final_weight(final_state, Weight::all()).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_empty_nwa() {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        let dwa = nwa.determinize_to_dwa();
        assert_eq!(dwa.states.len(), 1);
    }

    #[test]
    fn test_det_accepts_nothing() {
        let nwa = NWA::new();
        let dwa = nwa.determinize_to_dwa();
        let expected = DWA::new();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_accepts_empty_word() {
        let mut nwa = NWA::new();
        nwa.states[nwa.body.start_states[0]].final_weight = Some(Weight::from_item(42));
        let dwa = nwa.determinize_to_dwa();
        let mut expected = DWA::new();
        expected.set_final_weight(0, Weight::from_item(42)).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_determinize_complex_nwa_from_template() {
        fn neg(x: Label) -> Label { Label::MIN + x }
        let mut nwa = NWA::new();
        for _ in 0..38 { nwa.states.add_state(); }
        nwa.add_epsilon(0, 6, Weight::all());
        nwa.add_epsilon(0, 10, Weight::all());
        nwa.add_epsilon(0, 13, Weight::all());
        nwa.add_epsilon(0, 14, Weight::all());
        nwa.add_epsilon(0, 15, Weight::all());
        nwa.add_epsilon(0, 17, Weight::all());
        nwa.add_epsilon(0, 19, Weight::all());
        nwa.add_epsilon(0, 20, Weight::all());
        nwa.add_epsilon(3, 21, Weight::all());
        nwa.add_epsilon(4, 22, Weight::all());
        nwa.add_epsilon(4, 23, Weight::all());
        nwa.add_epsilon(4, 28, Weight::all());
        nwa.add_epsilon(4, 33, Weight::all());
        nwa.add_epsilon(5, 38, Weight::all());
        nwa.add_transition(6, 5, 7, Weight::all()).unwrap();
        nwa.add_transition(7, neg(5), 8, Weight::all()).unwrap();
        nwa.add_transition(8, neg(10), 9, Weight::all()).unwrap();
        nwa.states[9].final_weight = Some(Weight::all());
        nwa.add_transition(10, 2, 11, Weight::all()).unwrap();
        nwa.add_transition(13, 4, 1, Weight::all()).unwrap();
        nwa.add_transition(14, 5, 3, Weight::all()).unwrap();
        nwa.add_transition(15, 6, 16, Weight::all()).unwrap();
        nwa.add_transition(17, 8, 18, Weight::all()).unwrap();
        nwa.add_transition(19, 9, 4, Weight::all()).unwrap();
        nwa.add_transition(20, 10, 5, Weight::all()).unwrap();
        nwa.add_transition(21, 7, 4, Weight::all()).unwrap();
        nwa.add_transition(22, 7, 4, Weight::all()).unwrap();
        nwa.add_transition(23, 0, 24, Weight::all()).unwrap();
        nwa.add_transition(24, neg(0), 25, Weight::all()).unwrap();
        nwa.add_transition(25, neg(5), 26, Weight::all()).unwrap();
        nwa.add_transition(26, neg(10), 27, Weight::all()).unwrap();
        nwa.states[27].final_weight = Some(Weight::all());
        nwa.add_transition(28, 3, 29, Weight::all()).unwrap();
        nwa.add_transition(29, neg(3), 30, Weight::all()).unwrap();
        nwa.add_transition(30, neg(5), 31, Weight::all()).unwrap();
        nwa.add_transition(31, neg(10), 32, Weight::all()).unwrap();
        nwa.states[32].final_weight = Some(Weight::all());
        nwa.add_transition(33, 7, 34, Weight::all()).unwrap();
        nwa.add_transition(34, neg(7), 35, Weight::all()).unwrap();
        nwa.add_transition(35, neg(5), 36, Weight::all()).unwrap();
        nwa.add_transition(36, neg(10), 37, Weight::all()).unwrap();
        nwa.states[37].final_weight = Some(Weight::all());
        nwa.add_transition(38, 5, 3, Weight::all()).unwrap();

        let dwa = nwa.determinize_to_dwa();
        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        assert!(!weight.is_empty());
    }

    #[test]
    fn test_determinize_minimal_failing_nwa_repro() {
        fn neg(x: Label) -> Label { Label::MIN + x }
        let mut nwa = NWA::new();
        for _ in 0..37 { nwa.states.add_state(); }
        let w_all = Weight::all();
        nwa.add_epsilon(0, 6, w_all.clone());
        nwa.add_epsilon(0, 10, w_all.clone());
        nwa.add_epsilon(0, 14, w_all.clone());
        nwa.add_epsilon(0, 19, w_all.clone());
        nwa.add_epsilon(3, 21, w_all.clone());
        nwa.add_epsilon(4, 28, w_all.clone());
        nwa.add_epsilon(4, 33, w_all.clone());
        nwa.add_transition(6, 5, 7, w_all.clone()).unwrap();
        nwa.add_transition(7, neg(5), 8, w_all.clone()).unwrap();
        nwa.add_transition(8, neg(10), 9, w_all.clone()).unwrap();
        nwa.states[9].final_weight = Some(w_all.clone());
        nwa.add_transition(10, 2, 11, w_all.clone()).unwrap();
        nwa.add_transition(14, 5, 3, w_all.clone()).unwrap();
        nwa.add_transition(21, 7, 4, w_all.clone()).unwrap();
        nwa.add_transition(19, 9, 4, w_all.clone()).unwrap();
        nwa.add_transition(28, 3, 29, w_all.clone()).unwrap();
        nwa.add_transition(29, neg(3), 30, w_all.clone()).unwrap();
        nwa.add_transition(30, neg(5), 31, w_all.clone()).unwrap();
        nwa.add_transition(31, neg(10), 32, w_all.clone()).unwrap();
        nwa.states[32].final_weight = Some(w_all.clone());
        nwa.add_transition(33, 7, 34, w_all.clone()).unwrap();
        nwa.add_transition(34, neg(7), 35, w_all.clone()).unwrap();
        nwa.add_transition(35, neg(5), 36, w_all.clone()).unwrap();
        nwa.add_transition(36, neg(10), 37, w_all.clone()).unwrap();
        nwa.states[37].final_weight = Some(w_all.clone());

        let dwa = nwa.determinize_to_dwa();
        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        assert!(!weight.is_empty());
    }

    #[test]
    fn test_determinize_minimal_failing_nwa() {
        fn neg(x: Label) -> Label { Label::MIN + x }
        let mut nwa = NWA::new();
        for _ in 0..33 { nwa.states.add_state(); }
        nwa.add_epsilon(0, 19, Weight::all());
        nwa.add_transition(19, 9, 4, Weight::all()).unwrap();
        nwa.add_epsilon(4, 28, Weight::all());
        nwa.add_transition(28, 3, 29, Weight::all()).unwrap();
        nwa.add_transition(29, neg(3), 30, Weight::all()).unwrap();
        nwa.add_transition(30, neg(5), 31, Weight::all()).unwrap();
        nwa.add_transition(31, neg(10), 32, Weight::all()).unwrap();
        nwa.states[32].final_weight = Some(Weight::all());
        let dwa = nwa.determinize_to_dwa();
        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        assert!(!weight.is_empty());
    }

    #[test]
    fn test_det_rustfst_simple_char() {
        let nwa = nwa_accepts_char('a', Weight::from_item(1));
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let expected = dwa_accepts_char('a', Weight::from_item(1));
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_union_of_chars_rustfst() {
        let mut nwa = NWA::new();
        let start = nwa.body.start_states[0];
        let s_a = nwa.states.add_state();
        let s_b = nwa.states.add_state();
        let final_a = nwa.states.add_state();
        let final_b = nwa.states.add_state();
        nwa.add_epsilon(start, s_a, Weight::all());
        nwa.add_epsilon(start, s_b, Weight::all());
        nwa.add_transition(s_a, 'a' as Label, final_a, Weight::all()).unwrap();
        nwa.add_transition(s_b, 'b' as Label, final_b, Weight::all()).unwrap();
        nwa.states[final_a].final_weight = Some(Weight::from_item(1));
        nwa.states[final_b].final_weight = Some(Weight::from_item(2));
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let mut expected = DWA::new();
        let final_a_dwa = expected.add_state();
        let final_b_dwa = expected.add_state();
        expected.add_transition(0, 'a' as Label, final_a_dwa, Weight::all()).unwrap();
        expected.add_transition(0, 'b' as Label, final_b_dwa, Weight::all()).unwrap();
        expected.set_final_weight(final_a_dwa, Weight::from_item(1)).unwrap();
        expected.set_final_weight(final_b_dwa, Weight::from_item(2)).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_nondeterminism_on_char_rustfst() {
        let mut nwa = NWA::new();
        let start = nwa.body.start_states[0];
        let f1 = nwa.states.add_state();
        let f2 = nwa.states.add_state();
        nwa.add_transition(start, 'a' as Label, f1, Weight::from_item(1)).unwrap();
        nwa.add_transition(start, 'a' as Label, f2, Weight::from_item(2)).unwrap();
        nwa.states[f1].final_weight = Some(Weight::all());
        nwa.states[f2].final_weight = Some(Weight::all());
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let mut expected = DWA::new();
        let final_state = expected.add_state();
        expected.add_transition(0, 'a' as Label, final_state, Weight::from_iter([1, 2])).unwrap();
        expected.set_final_weight(final_state, Weight::all()).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_empty_nwa_rustfst() {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        assert_eq!(dwa.states.len(), 1);
    }

    #[test]
    fn test_det_accepts_nothing_rustfst() {
        let nwa = NWA::new();
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let expected = DWA::new();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_accepts_empty_word_rustfst() {
        let mut nwa = NWA::new();
        nwa.states[nwa.body.start_states[0]].final_weight = Some(Weight::from_item(42));
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let mut expected = DWA::new();
        expected.set_final_weight(0, Weight::from_item(42)).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_determinize_complex_nwa_from_template_rustfst() {
        fn neg(x: Label) -> Label { Label::MIN + x }
        let mut nwa = NWA::new();
        for _ in 0..38 { nwa.states.add_state(); }
        nwa.add_epsilon(0, 6, Weight::all());
        nwa.add_epsilon(0, 10, Weight::all());
        nwa.add_epsilon(0, 13, Weight::all());
        nwa.add_epsilon(0, 14, Weight::all());
        nwa.add_epsilon(0, 15, Weight::all());
        nwa.add_epsilon(0, 17, Weight::all());
        nwa.add_epsilon(0, 19, Weight::all());
        nwa.add_epsilon(0, 20, Weight::all());
        nwa.add_epsilon(3, 21, Weight::all());
        nwa.add_epsilon(4, 22, Weight::all());
        nwa.add_epsilon(4, 23, Weight::all());
        nwa.add_epsilon(4, 28, Weight::all());
        nwa.add_epsilon(4, 33, Weight::all());
        nwa.add_epsilon(5, 38, Weight::all());
        nwa.add_transition(6, 5, 7, Weight::all()).unwrap();
        nwa.add_transition(7, neg(5), 8, Weight::all()).unwrap();
        nwa.add_transition(8, neg(10), 9, Weight::all()).unwrap();
        nwa.states[9].final_weight = Some(Weight::all());
        nwa.add_transition(10, 2, 11, Weight::all()).unwrap();
        nwa.add_transition(13, 4, 1, Weight::all()).unwrap();
        nwa.add_transition(14, 5, 3, Weight::all()).unwrap();
        nwa.add_transition(15, 6, 16, Weight::all()).unwrap();
        nwa.add_transition(17, 8, 18, Weight::all()).unwrap();
        nwa.add_transition(19, 9, 4, Weight::all()).unwrap();
        nwa.add_transition(20, 10, 5, Weight::all()).unwrap();
        nwa.add_transition(21, 7, 4, Weight::all()).unwrap();
        nwa.add_transition(22, 7, 4, Weight::all()).unwrap();
        nwa.add_transition(23, 0, 24, Weight::all()).unwrap();
        nwa.add_transition(24, neg(0), 25, Weight::all()).unwrap();
        nwa.add_transition(25, neg(5), 26, Weight::all()).unwrap();
        nwa.add_transition(26, neg(10), 27, Weight::all()).unwrap();
        nwa.states[27].final_weight = Some(Weight::all());
        nwa.add_transition(28, 3, 29, Weight::all()).unwrap();
        nwa.add_transition(29, neg(3), 30, Weight::all()).unwrap();
        nwa.add_transition(30, neg(5), 31, Weight::all()).unwrap();
        nwa.add_transition(31, neg(10), 32, Weight::all()).unwrap();
        nwa.states[32].final_weight = Some(Weight::all());
        nwa.add_transition(33, 7, 34, Weight::all()).unwrap();
        nwa.add_transition(34, neg(7), 35, Weight::all()).unwrap();
        nwa.add_transition(35, neg(5), 36, Weight::all()).unwrap();
        nwa.add_transition(36, neg(10), 37, Weight::all()).unwrap();
        nwa.states[37].final_weight = Some(Weight::all());
        nwa.add_transition(38, 5, 3, Weight::all()).unwrap();

        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        assert!(!weight.is_empty());
    }

    #[test]
    fn test_determinize_minimal_failing_nwa_repro_rustfst() {
        fn neg(x: Label) -> Label { Label::MIN + x }
        let mut nwa = NWA::new();
        for _ in 0..37 { nwa.states.add_state(); }
        let w_all = Weight::all();
        nwa.add_epsilon(0, 6, w_all.clone());
        nwa.add_epsilon(0, 10, w_all.clone());
        nwa.add_epsilon(0, 14, w_all.clone());
        nwa.add_epsilon(0, 19, w_all.clone());
        nwa.add_epsilon(3, 21, w_all.clone());
        nwa.add_epsilon(4, 28, w_all.clone());
        nwa.add_epsilon(4, 33, w_all.clone());
        nwa.add_transition(6, 5, 7, w_all.clone()).unwrap();
        nwa.add_transition(7, neg(5), 8, w_all.clone()).unwrap();
        nwa.add_transition(8, neg(10), 9, w_all.clone()).unwrap();
        nwa.states[9].final_weight = Some(w_all.clone());
        nwa.add_transition(10, 2, 11, w_all.clone()).unwrap();
        nwa.add_transition(14, 5, 3, w_all.clone()).unwrap();
        nwa.add_transition(21, 7, 4, w_all.clone()).unwrap();
        nwa.add_transition(19, 9, 4, w_all.clone()).unwrap();
        nwa.add_transition(28, 3, 29, w_all.clone()).unwrap();
        nwa.add_transition(29, neg(3), 30, w_all.clone()).unwrap();
        nwa.add_transition(30, neg(5), 31, w_all.clone()).unwrap();
        nwa.add_transition(31, neg(10), 32, w_all.clone()).unwrap();
        nwa.states[32].final_weight = Some(w_all.clone());
        nwa.add_transition(33, 7, 34, w_all.clone()).unwrap();
        nwa.add_transition(34, neg(7), 35, w_all.clone()).unwrap();
        nwa.add_transition(35, neg(5), 36, w_all.clone()).unwrap();
        nwa.add_transition(36, neg(10), 37, w_all.clone()).unwrap();
        nwa.states[37].final_weight = Some(w_all.clone());

        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        assert!(!weight.is_empty());
    }

    #[test]
    fn test_determinize_minimal_failing_nwa_rustfst() {
        fn neg(x: Label) -> Label { Label::MIN + x }
        let mut nwa = NWA::new();
        for _ in 0..33 { nwa.states.add_state(); }
        nwa.add_epsilon(0, 19, Weight::all());
        nwa.add_transition(19, 9, 4, Weight::all()).unwrap();
        nwa.add_epsilon(4, 28, Weight::all());
        nwa.add_transition(28, 3, 29, Weight::all()).unwrap();
        nwa.add_transition(29, neg(3), 30, Weight::all()).unwrap();
        nwa.add_transition(30, neg(5), 31, Weight::all()).unwrap();
        nwa.add_transition(31, neg(10), 32, Weight::all()).unwrap();
        nwa.states[32].final_weight = Some(Weight::all());
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        assert!(!weight.is_empty());
    }
}
