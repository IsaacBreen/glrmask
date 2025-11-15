use crate::precompute4::weighted_automata::{DWAState, SimpleBitset, DWA, DWABuildError, NWA, NWABuildError, Weight, format_word};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::precompute4::resolve_negatives::resolve_negative_codes_in_dwa;

// --- Stochastic validation controls and RNG ---
const VALIDATION_SAMPLES: usize = 32;
const VALIDATION_MAX_STEPS: usize = 32;
const SAMPLING_TRIES: usize = 100;

#[derive(Clone, Debug)]
struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng(seed)
    }
    fn from_time() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let mixed = (now.as_nanos() as u128 ^ ((now.as_secs() as u128) << 64)) as u64;
        SimpleRng::new(mixed)
    }
    fn next_u64(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
    fn gen_bool_ratio(&mut self, num: u32, den: u32) -> bool {
        if den == 0 {
            true
        } else {
            (self.next_u64() % (den as u64)) < (num as u64)
        }
    }
}

// Small fixed alphabet used for default-edge sampling and variety.
// Includes ASCII letters/digits, some small integers, and negative-coded inputs used in tests.
const BASE_ALPHABET: &[i16] = &[
    b'a' as i16, b'b' as i16, b'c' as i16, b'd' as i16, b'e' as i16, b'f' as i16, b'g' as i16,
    b'h' as i16, b'i' as i16, b'j' as i16, b'k' as i16, b'l' as i16, b'm' as i16, b'n' as i16,
    b'o' as i16, b'p' as i16, b'q' as i16, b'r' as i16, b's' as i16, b't' as i16, b'u' as i16,
    b'v' as i16, b'w' as i16, b'x' as i16, b'y' as i16, b'z' as i16, b' ' as i16,
    b'0' as i16, b'1' as i16, b'2' as i16, b'3' as i16, b'4' as i16, b'5' as i16,
    b'6' as i16, b'7' as i16, b'8' as i16, b'9' as i16,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    i16::MIN + 0, i16::MIN + 1, i16::MIN + 2, i16::MIN + 3, i16::MIN + 4,
    i16::MIN + 5, i16::MIN + 6, i16::MIN + 7, i16::MIN + 8, i16::MIN + 9, i16::MIN + 10,
];

fn weight_subset(sub: &Weight, sup: &Weight) -> bool {
    (sub & sup) == sub.clone()
}

impl DWA {
    /// Sample an accepted path (word and weight) using a time-based seed.
    /// Returns None if no accepted path was found within the attempt budget.
    pub fn sample_accepted_path(&self, max_steps: usize) -> Option<(Vec<i16>, Weight)> {
        let mut rng = SimpleRng::from_time();
        self.sample_accepted_path_with_rng(&mut rng, max_steps)
    }

    /// Sample an accepted path (word and weight) with a fixed seed (deterministic).
    pub fn sample_accepted_path_with_seed(&self, seed: u64, max_steps: usize) -> Option<(Vec<i16>, Weight)> {
        let mut rng = SimpleRng::new(seed);
        self.sample_accepted_path_with_rng(&mut rng, max_steps)
    }

    /// Core sampler with a provided RNG. Tries multiple attempts to find an accepted word.
    pub fn sample_accepted_path_with_rng(&self, rng: &mut SimpleRng, max_steps: usize) -> Option<(Vec<i16>, Weight)> {
        if self.states.0.is_empty() {
            return None;
        }
        for _attempt in 0..SAMPLING_TRIES {
            let mut word: Vec<i16> = Vec::new();
            let mut s = self.body.start_state;
            let mut acc = Weight::all();

            for step in 0..max_steps {
                // Early stop with some probability if we can accept here.
                if let Some(fw) = &self.states[s].final_weight {
                    if rng.gen_bool_ratio(1, 3) || step == max_steps - 1 {
                        let w = &acc & fw;
                        if !w.is_empty() {
                            return Some((word, w));
                        }
                    }
                }

                // Choose next character: one of the exception keys or a default-character if default exists.
                let st = &self.states[s];
                let choices: Vec<i16> = st.transitions.keys().copied().collect();
                let total = choices.len();
                if total == 0 {
                    // Dead-end; try to finalize or abort attempt.
                    if let Some(fw) = &st.final_weight {
                        let w = &acc & fw;
                        if !w.is_empty() {
                            return Some((word, w));
                        }
                    }
                    break; // new attempt
                }
                let pick = rng.gen_usize(total);
                let ch = choices[pick];

                let next = st.transitions.get(&ch).copied();
                if next.is_none() {
                    break;
                }
                let edge_w = st.get_weight(ch).cloned().unwrap_or_else(Weight::zeros);
                if edge_w.is_empty() {
                    break;
                }
                let new_acc = &acc & &edge_w;
                if new_acc.is_empty() {
                    break;
                }
                acc = new_acc;
                s = next.unwrap();
                word.push(ch);
            }

            // Finalize at end of attempt if possible:
            if s < self.states.len() {
                if let Some(fw) = &self.states[s].final_weight {
                    let w = &acc & fw;
                    if !w.is_empty() {
                        return Some((word, w));
                    }
                }
            }
        }
        None
    }

    fn expected_union_weight(a: &DWA, b: &DWA, word: &[i16]) -> Weight {
        let wa = a.eval_word_weight(word);
        let wb = b.eval_word_weight(word);
        &wa | &wb
    }

    /// Expected concatenation weight:
    /// union over all split points i of (A(word[..i]) ∧ B(word[i..])).
    fn expected_concat_weight(a: &DWA, b: &DWA, word: &[i16], eps_weight: &Weight) -> Weight {
        let mut acc = Weight::zeros();
        for i in 0..=word.len() {
            let wa = a.eval_word_weight(&word[..i]);
            if wa.is_empty() {
                continue;
            }
            let wb = b.eval_word_weight(&word[i..]);
            if wb.is_empty() {
                continue;
            }
            let both = &(&wa & &wb) & eps_weight;
            if !both.is_empty() {
                acc |= &both;
            }
        }
        acc
    }

    pub fn stochastic_validate_union(a: &DWA, b: &DWA, u: &DWA) {
        let mut rng = SimpleRng::from_time();
        for _ in 0..VALIDATION_SAMPLES {
            // Sample a path from A -> must be in U, and U == A ∪ B for that word.
            if let Some((w, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let wu = u.eval_word_weight(&w);
                assert!(!wu.is_empty(), "Union rejected a word accepted by A.\nword: {}\nA(w): {}\nU(w): {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), wa, wu, a, b, u);
                assert!(weight_subset(&wa, &wu), "Union weight missing subset from A.\nword: {}\nA(w): {}\nU(w): {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), wa, wu, a, b, u);
                let expected = DWA::expected_union_weight(a, b, &w);
                assert_eq!(wu, expected, "Union weight mismatch vs expected A∪B.\nword: {}\nA(w): {}\nB(w): {}\nU(w): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), wa, b.eval_word_weight(&w), wu, expected, a, b, u);
            }

            // Sample a path from B -> must be in U, and U == A ∪ B for that word.
            if let Some((w, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let wu = u.eval_word_weight(&w);
                assert!(!wu.is_empty(), "Union rejected a word accepted by B.\nword: {}\nB(w): {}\nU(w): {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), wb, wu, a, b, u);
                assert!(weight_subset(&wb, &wu), "Union weight missing subset from B.\nword: {}\nB(w): {}\nU(w): {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), wb, wu, a, b, u);
                let expected = DWA::expected_union_weight(a, b, &w);
                assert_eq!(wu, expected, "Union weight mismatch vs expected A∪B.\nword: {}\nA(w): {}\nB(w): {}\nU(w): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), a.eval_word_weight(&w), wb, wu, expected, a, b, u);
            }

            // Sample a path from U -> ensure it's in A ∪ B (equality check).
            if let Some((w, wu)) = u.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let expected = DWA::expected_union_weight(a, b, &w);
                assert_eq!(wu, expected, "U accepted a word with weight not equal to A∪B.\nword: {}\nA(w): {}\nB(w): {}\nU(w): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA U:\n{}", format_word(&w), a.eval_word_weight(&w), b.eval_word_weight(&w), wu, expected, a, b, u);
            }
        }
    }

    pub fn stochastic_validate_concatenate(a: &DWA, b: &DWA, c: &DWA, eps_weight: &Weight) {
        let mut rng = SimpleRng::from_time();
        for _ in 0..VALIDATION_SAMPLES {
            // Sample accepted paths in A and B; the concatenation of the words should be in C and contain WA ∧ WB.
            if let Some((wa_word, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                if let Some((wb_word, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                    let mut w = wa_word.clone();
                    w.extend_from_slice(&wb_word);
                    let wc = c.eval_word_weight(&w);
                    let expected_simple = &(&wa & &wb) & eps_weight;
                    if !expected_simple.is_empty() {
                        assert!(weight_subset(&expected_simple, &wc), "Concatenation missing expected subset.\nword_a: {}\nword_b: {}\nword: {}\nA(wA): {}\nB(wB): {}\nC(wA∘wB): {}\nExpected subset: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA C:\n{}", format_word(&wa_word), format_word(&wb_word), format_word(&w), wa, wb, wc, expected_simple, a, b, c);
                    }
                    // Also verify full expected across all splits equals C's result
                    let expected_all = DWA::expected_concat_weight(a, b, &w, eps_weight);
                    assert_eq!(wc, expected_all, "C(word) != expected union-over-splits(A(prefix) ∧ B(suffix)).\nword_a: {}\nword_b: {}\nword: {}\nC(word): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA C:\n{}", format_word(&wa_word), format_word(&wb_word), format_word(&w), wc, expected_all, a, b, c);
                }
            }

            // Sample accepted paths from C -> must equal union-over-splits(A(prefix) ∧ B(suffix)).
            if let Some((w, wc)) = c.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let expected = DWA::expected_concat_weight(a, b, &w, eps_weight);
                assert_eq!(wc, expected, "C(word) != expected union-over-splits(A(prefix) ∧ B(suffix)).\nword: {}\nC(word): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA C:\n{}", format_word(&w), wc, expected, a, b, c);
            }
        }
    }
}

pub fn stochastic_equivalence_test(mut a: DWA, mut b: DWA) {
    crate::debug!(5, "Starting stochastic equivalence test");
    let mut rng = SimpleRng::from_time();
    for _ in 0..VALIDATION_SAMPLES {
        // Sample from A, check against B
        if let Some((w, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let wb = b.eval_word_weight(&w);
            assert_eq!(
                wa, wb,
                "Equivalence fail: A(w) != B(w) for word from A.\nword: {}\nA(w): {}\nB(w): {}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                format_word(&w), wa, wb, a, b
            );
        }

        // Sample from B, check against A
        if let Some((w, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
            let wa = a.eval_word_weight(&w);
            assert_eq!(
                wb, wa,
                "Equivalence fail: B(w) != A(w) for word from B.\nword: {}\nB(w): {}\nA(w): {}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                format_word(&w), wb, wa, a, b
            );
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
    assert_eq!((&set1 | &zeros).len(), 3);
    assert_eq!((&set1 & &zeros), Weight::zeros());
}

#[test]
fn test_dwa_builder() {
    let mut dwa = DWA::new();
    assert_eq!(dwa.states.len(), 1);
    assert_eq!(dwa.body.start_state, 0);

    let s1 = dwa.add_state();
    assert_eq!(s1, 1);
    assert_eq!(dwa.states.len(), 2);

    dwa.set_final_weight(1, SimpleBitset::from_item(20)).unwrap();

    assert_eq!(dwa.states[1].final_weight, Some(SimpleBitset::from_item(20)));

    dwa.add_transition(0, b'a' as i16, 1, SimpleBitset::from_item(30)).unwrap();
    assert_eq!(*dwa.states[0].transitions.get(&(b'a' as i16)).unwrap(), 1);
    assert_eq!(*dwa.states[0].trans_weights.get(&(b'a' as i16)).unwrap(), SimpleBitset::from_item(30));

    // Test error cases
    let res = dwa.add_transition(0, b'a' as i16, 1, SimpleBitset::zeros());
    assert!(matches!(res, Err(DWABuildError::TransitionAlreadyExists { from: 0, on: 97 })));

    let res = dwa.set_final_weight(10, SimpleBitset::zeros());
    assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { state: 10 })));
}

// --- Advanced Tests ---

/// Helper to create a DWA that accepts a single character and produces a final weight.
fn dwa_accepts_char(ch: char, final_weight: Weight) -> DWA {
    let mut dwa = DWA::new();
    let final_state = dwa.add_state();
    dwa.add_transition(dwa.body.start_state, ch as i16, final_state, Weight::all()).unwrap();
    dwa.set_final_weight(final_state, final_weight).unwrap();
    dwa
}

/// Helper to create a DWA that accepts a string and produces a final weight.
fn dwa_from_str(s: &str, final_weight: Weight) -> DWA {
    let mut dwa = DWA::new();
    let mut current_state = dwa.body.start_state;
    for ch in s.chars() {
        let next_state = dwa.add_state();
        dwa.add_transition(current_state, ch as i16, next_state, Weight::all()).unwrap();
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
    let s3 = d.add_state(); // Should be merged with s2
    let s4 = d.add_state(); // Final state
    let s5 = d.add_state(); // Unreachable

    d.add_transition(0, 'a' as i16, s1, Weight::all()).unwrap();
    d.add_transition(0, 'b' as i16, s2, Weight::all()).unwrap();
    d.add_transition(0, 'c' as i16, s3, Weight::all()).unwrap();
    d.add_transition(s1, 'x' as i16, s4, Weight::all()).unwrap();
    d.add_transition(s2, 'y' as i16, s4, Weight::all()).unwrap();
    d.add_transition(s3, 'y' as i16, s4, Weight::all()).unwrap(); // Same behavior as s2
    d.set_final_weight(s4, Weight::from_item(1)).unwrap();

    assert_eq!(d.states.len(), 6);
    d.simplify();
    // s5 pruned (unreachable). s2 and s3 merged.
    // Expected states: start, 'a'-state, 'b'/'c'-state, final-state. Total 4.
    assert_eq!(d.states.len(), 4);
}

#[test]
fn test_union_simple() {
    let d1 = dwa_accepts_char('a', Weight::from_item(1));
    let d2 = dwa_accepts_char('b', Weight::from_item(2));

    let mut expected = DWA::new();
    let s_a = expected.add_state();
    let s_b = expected.add_state();
    expected.add_transition(0, 'a' as i16, s_a, Weight::all()).unwrap();
    expected.add_transition(0, 'b' as i16, s_b, Weight::all()).unwrap();
    expected.set_final_weight(s_a, Weight::from_item(1)).unwrap();
    expected.set_final_weight(s_b, Weight::from_item(2)).unwrap();

    let u = d1.union(&d2);
    stochastic_equivalence_test(u, expected);
}

#[test]
fn test_union_overlapping() {
    let d1 = dwa_accepts_char('a', Weight::from_item(1));
    let mut d2 = dwa_accepts_char('b', Weight::from_item(3));
    let s_a2 = d2.add_state();
    d2.add_transition(d2.body.start_state, 'a' as i16, s_a2, Weight::all()).unwrap();
    d2.set_final_weight(s_a2, Weight::from_item(2)).unwrap();

    let mut expected = DWA::new();
    let s_a = expected.add_state();
    let s_b = expected.add_state();
    expected.add_transition(0, 'a' as i16, s_a, Weight::all()).unwrap();
    expected.add_transition(0, 'b' as i16, s_b, Weight::all()).unwrap();
    expected.set_final_weight(s_a, Weight::from_iter(vec![1, 2])).unwrap();
    expected.set_final_weight(s_b, Weight::from_item(3)).unwrap();

    let u = d1.union(&d2);
    stochastic_equivalence_test(u, expected);
}

#[test]
fn test_concatenate_simple() {
    let d1 = dwa_accepts_char('a', Weight::from_iter([1, 2]));
    let d2 = dwa_accepts_char('b', Weight::from_iter([2, 3]));
    let c = d1.concatenate(&d2);
    let expected = dwa_from_str("ab", Weight::from_item(2));
    stochastic_equivalence_test(c, expected);
}

#[test]
fn test_apply_weight() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    d.set_final_weight(0, Weight::from_iter(vec![5, 6])).unwrap();
    d.add_transition(0, 'a' as i16, s1, Weight::from_iter(vec![100, 101])).unwrap();
    d.add_transition(0, 'b' as i16, 0, Weight::from_iter(vec![200, 201])).unwrap();

    let gate = Weight::from_iter(vec![6, 11, 101, 201]);
    let new_start = d.apply_weight(&gate);

    assert_eq!(d.body.start_state, new_start);
    let new_start_state = &d.states[new_start];
    assert_eq!(new_start_state.final_weight, Some(Weight::from_item(6)));
    assert_eq!(new_start_state.trans_weights.get(&('a' as i16)), Some(&Weight::from_item(101)));
    assert_eq!(new_start_state.trans_weights.get(&('b' as i16)), Some(&Weight::from_item(201)));
    assert_eq!(new_start_state.transitions.get(&('a' as i16)), Some(&s1));
    assert_eq!(new_start_state.transitions.get(&('b' as i16)), Some(&0));
}

/// Helper that creates a DWA with a single transition on `ch` with a given
/// per-edge weight, landing in a final state with the provided final weight.
fn dwa_with_char_and_weights(ch: char, edge_weight: Weight, final_weight: Weight) -> DWA {
    let mut d = DWA::new();
    let s = d.add_state();
    d.add_transition(d.body.start_state, ch as i16, s, edge_weight).unwrap();
    d.set_final_weight(s, final_weight).unwrap();
    d
}

#[test]
fn test_simple_bitset_iter_up_to_all() {
    let all = Weight::all();
    let vals: Vec<_> = all.iter_up_to(5).collect();
    assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn test_union_transition_weight_union() {
    fn build(ch: char, ew: usize, fw: usize) -> DWA {
        dwa_with_char_and_weights(ch, Weight::from_item(ew), Weight::from_item(fw))
    }
    let d1 = build('x', 10, 1);
    let d2 = build('x', 20, 2);
    let u = d1.union(&d2);

    // Expected automaton with unioned transition and final weights on 'x'.
    let mut expected = DWA::new();
    let s = expected.add_state();
    expected
        .add_transition(0, 'x' as i16, s, Weight::from_iter(vec![10, 20]))
        .unwrap();
    expected
        .set_final_weight(s, Weight::from_iter(vec![1, 2]))
        .unwrap();

    stochastic_equivalence_test(u, expected);
}

#[test]
fn test_json_roundtrip_complex() {
    use crate::json_serialization::JSONConvertible;

    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    d.add_transition(d.body.start_state, 'y' as i16, s1, Weight::from_iter(vec![1, 2, 3])).unwrap();
    d.add_transition(d.body.start_state, 'x' as i16, s2, Weight::from_item(99))
        .unwrap();
    d.set_final_weight(s2, Weight::from_iter(vec![5, 7])).unwrap();

    let node = d.to_json();
    let d2 = DWA::from_json(node.clone()).expect("from_json should succeed");
    assert_eq!(node, d2.to_json(), "Roundtrip JSON should be stable");
}

#[test]
fn test_add_transition_out_of_bounds() {
    let mut d = DWA::new();
    let res = d.add_transition(5, 'a' as i16, 0, Weight::zeros());
    assert!(matches!(res, Err(DWABuildError::StateOutOfBounds { state: 5 })));

    let res2 = d.add_transition(0, 'a' as i16, 99, Weight::zeros());
    assert!(matches!(res2, Err(DWABuildError::StateOutOfBounds { state: 99 })));
}

#[test]
fn test_prune_unreachable_with_default_chain() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let _s2 = d.add_state(); // Unused, unreachable
    d.add_transition(d.body.start_state, 'y' as i16, s1, Weight::all()).unwrap();
    d.set_final_weight(s1, Weight::from_item(1)).unwrap();
    d.add_transition(s1, 'x' as i16, s1, Weight::all()).unwrap();

    // Completely unreachable component
    let s_unreach = d.add_state();
    d.add_transition(s_unreach, 'z' as i16, s_unreach, Weight::all())
        .unwrap();

    let before = d.states.len();
    d.simplify();
    let after = d.states.len();
    assert!(after < before, "Unreachable states should be pruned");
    assert_eq!(after, 2, "Only start and s1 should remain reachable");
}

#[test]
fn test_equivalence_via_simplification() {
    // DWA 'a' has explicit transitions for inputs '1' and '3' which lead
    // to non-final, sink-like states. State 1 is a true sink, and state 2
    // only transitions to state 1.
    let mut a = DWA::new();
    let s1a = a.add_state();
    let s2a = a.add_state();
    a.add_transition(0, 0, s1a, Weight::from_item(1)).unwrap();
    a.add_transition(0, 1, s2a, Weight::from_iter(0..=1)).unwrap();
    a.add_transition(0, 2, s1a, Weight::from_item(0)).unwrap();
    a.add_transition(0, 3, s1a, Weight::from_iter(0..=1)).unwrap();

    // DWA 'b' lacks these transitions. For inputs '1' and '3', it transitions
    // to an implicit sink. The simplification process should make 'a' equivalent
    // to 'b'.
    let mut b = DWA::new();
    let s1b = b.add_state();
    b.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    b.add_transition(0, 2, s1b, Weight::from_item(0)).unwrap();

    stochastic_equivalence_test(a, b);
}

#[test]
fn test_concatenate_left_start_is_final() {
    // LEFT: DWA (start: 0)
    //   State 0:
    //     weight: []
    //     final_weight: [0, 1]
    let mut left = DWA::new();
    left.set_final_weight(left.body.start_state, Weight::from_iter([0, 1])).unwrap();

    // RIGHT: DWA (start: 0)
    //   State 0:
    //     weight: []
    //     final_weight: [1, 2]
    let mut right = DWA::new();
    right.set_final_weight(right.body.start_state, Weight::from_iter([1, 2])).unwrap();

    let c = left.concatenate(&right);

    let mut expected = DWA::new();
    expected.set_final_weight(expected.body.start_state, Weight::from_item(1)).unwrap();

    stochastic_equivalence_test(c, expected);
}

#[test]
fn test_simplify_propagates_future_weights() {
    // This test checks that weight constraints from final states are propagated
    // backward to relax unnecessarily restrictive edge weights.
    // DWA A has a transition 1 -> 2 with weight [1..=2], but the final
    // state 2 only has weight [2]. The path weight for "ab" is thus
    // ALL & [1..=2] & [2] = [2].
    let mut a = DWA::new();
    let s1 = a.add_state();
    let s2 = a.add_state();
    a.add_transition(0, 'a' as i16, s1, Weight::all()).unwrap();
    a.add_transition(s1, 'b' as i16, s2, Weight::from_iter([1..=2])).unwrap();
    a.set_final_weight(s2, Weight::from_item(2)).unwrap();

    // DWA B is the expected simplified form. The transition 1 -> 2 has its
    // weight relaxed to ALL, because any components of the weight other than
    // [2] would be filtered by the final state anyway. The path weight for "ab"
    // is ALL & ALL & [2] = [2], which is equivalent.
    let mut b = DWA::new();
    let s1_b = b.add_state();
    let s2_b = b.add_state();
    b.add_transition(0, 'a' as i16, s1_b, Weight::all()).unwrap();
    b.add_transition(s1_b, 'b' as i16, s2_b, Weight::all()).unwrap();
    b.set_final_weight(s2_b, Weight::from_item(2)).unwrap();

    println!("Before simplification A:\n{}", a);
    a.simplify();
    println!("After simplification A:\n{}", a);

    stochastic_equivalence_test(a, b);
}

#[test]
fn test_union_complex_from_attachment() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    // --- Build LEFT DWA ---
    let mut left = DWA::new();
    for _ in 0..47 {
        left.add_state();
    }

    left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    left.add_transition(0, 2, 2, Weight::from_item(1)).unwrap();
    left.add_transition(0, 3, 3, Weight::from_item(1)).unwrap();
    left.add_transition(0, 4, 4, Weight::from_item(1)).unwrap();
    left.add_transition(0, 5, 5, Weight::from_item(1)).unwrap();
    left.add_transition(0, 6, 6, Weight::from_item(1)).unwrap();
    left.add_transition(0, 7, 7, Weight::from_item(1)).unwrap();
    left.add_transition(0, 8, 8, Weight::from_item(1)).unwrap();
    left.add_transition(0, 9, 9, Weight::from_item(1)).unwrap();
    left.add_transition(0, 10, 10, Weight::from_item(1)).unwrap();
    left.add_transition(1, neg(0), 11, Weight::all()).unwrap();
    left.add_transition(2, 100, 12, Weight::all()).unwrap();
    left.add_transition(3, neg(3), 13, Weight::all()).unwrap();
    left.add_transition(5, 3, 14, Weight::all()).unwrap();
    left.add_transition(5, 7, 9, Weight::all()).unwrap();
    left.add_transition(6, 100, 5, Weight::all()).unwrap();
    left.add_transition(7, neg(7), 15, Weight::all()).unwrap();
    left.add_transition(8, 100, 9, Weight::all()).unwrap();
    left.add_transition(9, 3, 16, Weight::all()).unwrap();
    left.add_transition(9, 7, 9, Weight::all()).unwrap();
    left.add_transition(10, 5, 5, Weight::all()).unwrap();
    left.add_transition(11, neg(9), 17, Weight::all()).unwrap();
    left.add_transition(12, 100, 18, Weight::all()).unwrap();
    left.add_transition(13, neg(9), 19, Weight::all()).unwrap();
    left.add_transition(14, neg(3), 20, Weight::all()).unwrap();
    left.add_transition(15, neg(9), 21, Weight::all()).unwrap();
    left.add_transition(16, neg(3), 22, Weight::all()).unwrap();
    let w01 = Weight::from_iter(0..=1);
    left.add_transition(17, 2, 23, w01.clone()).unwrap();
    left.add_transition(17, 4, 24, w01.clone()).unwrap();
    left.add_transition(17, 5, 25, w01.clone()).unwrap();
    left.add_transition(17, 6, 26, w01.clone()).unwrap();
    left.add_transition(17, 8, 27, w01.clone()).unwrap();
    left.add_transition(17, 9, 28, w01.clone()).unwrap();
    left.add_transition(17, 10, 29, w01.clone()).unwrap();
    left.add_transition(19, 2, 23, w01.clone()).unwrap();
    left.add_transition(19, 4, 24, w01.clone()).unwrap();
    left.add_transition(19, 5, 25, w01.clone()).unwrap();
    left.add_transition(19, 6, 26, w01.clone()).unwrap();
    left.add_transition(19, 8, 27, w01.clone()).unwrap();
    left.add_transition(19, 9, 28, w01.clone()).unwrap();
    left.add_transition(19, 10, 29, w01.clone()).unwrap();
    left.add_transition(20, neg(0), 30, Weight::all()).unwrap();
    left.add_transition(21, 2, 23, w01.clone()).unwrap();
    left.add_transition(21, 4, 24, w01.clone()).unwrap();
    left.add_transition(21, 5, 25, w01.clone()).unwrap();
    left.add_transition(21, 6, 26, w01.clone()).unwrap();
    left.add_transition(21, 8, 27, w01.clone()).unwrap();
    left.add_transition(21, 9, 28, w01.clone()).unwrap();
    left.add_transition(21, 10, 29, w01.clone()).unwrap();
    left.add_transition(22, neg(0), 31, Weight::all()).unwrap();
    left.add_transition(23, 100, 32, Weight::all()).unwrap();
    left.add_transition(25, 7, 28, Weight::all()).unwrap();
    left.add_transition(26, 100, 25, Weight::all()).unwrap();
    left.add_transition(27, 100, 28, Weight::all()).unwrap();
    left.add_transition(28, 0, 33, Weight::all()).unwrap();
    left.add_transition(28, 3, 34, Weight::all()).unwrap();
    left.add_transition(28, 7, 35, Weight::all()).unwrap();
    left.add_transition(29, 5, 25, Weight::all()).unwrap();
    left.add_transition(30, neg(9), 36, Weight::all()).unwrap();
    left.add_transition(31, neg(9), 37, Weight::all()).unwrap();
    left.add_transition(32, 100, 38, Weight::all()).unwrap();
    left.add_transition(33, neg(0), 39, Weight::all()).unwrap();
    left.add_transition(34, neg(3), 40, Weight::all()).unwrap();
    left.add_transition(35, neg(7), 41, Weight::all()).unwrap();
    left.add_transition(36, 2, 23, w01.clone()).unwrap();
    left.add_transition(36, 4, 24, w01.clone()).unwrap();
    left.add_transition(36, 5, 25, w01.clone()).unwrap();
    left.add_transition(36, 6, 26, w01.clone()).unwrap();
    left.add_transition(36, 8, 27, w01.clone()).unwrap();
    left.add_transition(36, 9, 28, w01.clone()).unwrap();
    left.add_transition(36, 10, 29, w01.clone()).unwrap();
    left.add_transition(37, 2, 23, w01.clone()).unwrap();
    left.add_transition(37, 4, 24, w01.clone()).unwrap();
    left.add_transition(37, 5, 25, w01.clone()).unwrap();
    left.add_transition(37, 6, 26, w01.clone()).unwrap();
    left.add_transition(37, 8, 27, w01.clone()).unwrap();
    left.add_transition(37, 9, 28, w01.clone()).unwrap();
    left.add_transition(37, 10, 29, w01.clone()).unwrap();
    left.add_transition(39, neg(5), 42, Weight::all()).unwrap();
    left.add_transition(40, neg(5), 43, Weight::all()).unwrap();
    left.add_transition(41, neg(5), 44, Weight::all()).unwrap();
    left.add_transition(42, neg(10), 45, Weight::all()).unwrap();
    left.add_transition(43, neg(10), 46, Weight::all()).unwrap();
    left.add_transition(44, neg(10), 47, Weight::all()).unwrap();
    left.set_final_weight(45, Weight::all()).unwrap();
    left.set_final_weight(46, Weight::all()).unwrap();
    left.set_final_weight(47, Weight::all()).unwrap();

    // --- Build RIGHT DWA ---
    let mut right = DWA::new();
    for _ in 0..42 {
        right.add_state();
    }

    right.add_transition(0, 2, 1, Weight::from_item(0)).unwrap();
    right.add_transition(0, 4, 2, Weight::from_item(0)).unwrap();
    right.add_transition(0, 5, 3, Weight::from_item(0)).unwrap();
    right.add_transition(0, 6, 4, Weight::from_item(0)).unwrap();
    right.add_transition(0, 8, 5, Weight::from_item(0)).unwrap();
    right.add_transition(0, 9, 6, Weight::from_item(0)).unwrap();
    right.add_transition(0, 10, 7, Weight::from_item(0)).unwrap();
    right.add_transition(1, 100, 8, Weight::all()).unwrap();
    right.add_transition(3, 7, 6, Weight::all()).unwrap();
    right.add_transition(4, 100, 3, Weight::all()).unwrap();
    right.add_transition(5, 100, 6, Weight::all()).unwrap();
    right.add_transition(6, 0, 9, Weight::all()).unwrap();
    right.add_transition(6, 3, 10, Weight::all()).unwrap();
    right.add_transition(6, 7, 11, Weight::all()).unwrap();
    right.add_transition(7, 5, 3, Weight::all()).unwrap();
    right.add_transition(8, 100, 12, Weight::all()).unwrap();
    right.add_transition(9, neg(0), 13, Weight::all()).unwrap();
    right.add_transition(10, neg(3), 14, Weight::all()).unwrap();
    right.add_transition(11, neg(7), 15, Weight::all()).unwrap();
    right.add_transition(13, neg(5), 16, Weight::all()).unwrap();
    right.add_transition(14, neg(5), 17, Weight::all()).unwrap();
    right.add_transition(15, neg(5), 18, Weight::all()).unwrap();
    right.add_transition(16, neg(10), 19, Weight::all()).unwrap();
    right.add_transition(17, neg(10), 20, Weight::all()).unwrap();
    right.add_transition(18, neg(10), 21, Weight::all()).unwrap();
    right.add_transition(19, 2, 22, w01.clone()).unwrap();
    right.add_transition(19, 4, 23, w01.clone()).unwrap();
    right.add_transition(19, 5, 24, w01.clone()).unwrap();
    right.add_transition(19, 6, 25, w01.clone()).unwrap();
    right.add_transition(19, 8, 26, w01.clone()).unwrap();
    right.add_transition(19, 9, 27, w01.clone()).unwrap();
    right.add_transition(19, 10, 28, w01.clone()).unwrap();
    right.add_transition(20, 2, 22, w01.clone()).unwrap();
    right.add_transition(20, 4, 23, w01.clone()).unwrap();
    right.add_transition(20, 5, 24, w01.clone()).unwrap();
    right.add_transition(20, 6, 25, w01.clone()).unwrap();
    right.add_transition(20, 8, 26, w01.clone()).unwrap();
    right.add_transition(20, 9, 27, w01.clone()).unwrap();
    right.add_transition(20, 10, 28, w01.clone()).unwrap();
    right.add_transition(21, 2, 22, w01.clone()).unwrap();
    right.add_transition(21, 4, 23, w01.clone()).unwrap();
    right.add_transition(21, 5, 24, w01.clone()).unwrap();
    right.add_transition(21, 6, 25, w01.clone()).unwrap();
    right.add_transition(21, 8, 26, w01.clone()).unwrap();
    right.add_transition(21, 9, 27, w01.clone()).unwrap();
    right.add_transition(21, 10, 28, w01.clone()).unwrap();
    right.add_transition(22, 100, 29, Weight::all()).unwrap();
    right.add_transition(24, 7, 27, Weight::all()).unwrap();
    right.add_transition(25, 100, 24, Weight::all()).unwrap();
    right.add_transition(26, 100, 27, Weight::all()).unwrap();
    right.add_transition(27, 0, 30, Weight::all()).unwrap();
    right.add_transition(27, 3, 31, Weight::all()).unwrap();
    right.add_transition(27, 7, 32, Weight::all()).unwrap();
    right.add_transition(28, 5, 24, Weight::all()).unwrap();
    right.add_transition(29, 100, 33, Weight::all()).unwrap();
    right.add_transition(30, neg(0), 34, Weight::all()).unwrap();
    right.add_transition(31, neg(3), 35, Weight::all()).unwrap();
    right.add_transition(32, neg(7), 36, Weight::all()).unwrap();
    right.add_transition(34, neg(5), 37, Weight::all()).unwrap();
    right.add_transition(35, neg(5), 38, Weight::all()).unwrap();
    right.add_transition(36, neg(5), 39, Weight::all()).unwrap();
    right.add_transition(37, neg(10), 40, Weight::all()).unwrap();
    right.add_transition(38, neg(10), 41, Weight::all()).unwrap();
    right.add_transition(39, neg(10), 42, Weight::all()).unwrap();
    right.set_final_weight(40, Weight::all()).unwrap();
    right.set_final_weight(41, Weight::all()).unwrap();
    right.set_final_weight(42, Weight::all()).unwrap();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_complex_from_attachment_simpified() {
    fn neg(val: i16) -> i16 {
        val.wrapping_add(i16::MIN)
    }

    // Build left DWA
    let mut left = DWA::new();
    for _ in 0..20 {
        left.add_state();
    }
    assert_eq!(left.states.len(), 21);

    // State 0
    left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    left.add_transition(0, 3, 2, Weight::from_item(1)).unwrap();
    left.add_transition(0, 5, 3, Weight::from_item(1)).unwrap();
    left.add_transition(0, 6, 4, Weight::from_item(1)).unwrap();
    left.add_transition(0, 7, 5, Weight::from_item(1)).unwrap();
    left.add_transition(0, 8, 4, Weight::from_item(1)).unwrap();
    left.add_transition(0, 9, 3, Weight::from_item(1)).unwrap();
    left.add_transition(0, 10, 6, Weight::from_item(1)).unwrap();
    // State 1
    left.add_transition(1, neg(0), 7, Weight::all()).unwrap();
    // State 2
    left.add_transition(2, neg(3), 7, Weight::all()).unwrap();
    // State 3
    left.add_transition(3, 3, 8, Weight::all()).unwrap();
    left.add_transition(3, 7, 3, Weight::all()).unwrap();
    // State 4
    left.add_transition(4, 100, 3, Weight::all()).unwrap();
    // State 5
    left.add_transition(5, neg(7), 7, Weight::all()).unwrap();
    // State 6
    left.add_transition(6, 5, 3, Weight::all()).unwrap();
    // State 7
    left.add_transition(7, neg(9), 9, Weight::all()).unwrap();
    // State 8
    left.add_transition(8, neg(3), 1, Weight::all()).unwrap();
    // State 9
    left.add_transition(9, 5, 10, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(9, 6, 11, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(9, 8, 12, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(9, 9, 13, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(9, 10, 14, Weight::from_iter(0..=1)).unwrap();
    // State 10
    left.add_transition(10, 7, 13, Weight::all()).unwrap();
    // State 11
    left.add_transition(11, 100, 10, Weight::all()).unwrap();
    // State 12
    left.add_transition(12, 100, 13, Weight::all()).unwrap();
    // State 13
    left.add_transition(13, 0, 15, Weight::all()).unwrap();
    left.add_transition(13, 3, 16, Weight::all()).unwrap();
    left.add_transition(13, 7, 17, Weight::all()).unwrap();
    // State 14
    left.add_transition(14, 5, 10, Weight::all()).unwrap();
    // State 15
    left.add_transition(15, neg(0), 18, Weight::all()).unwrap();
    // State 16
    left.add_transition(16, neg(3), 18, Weight::all()).unwrap();
    // State 17
    left.add_transition(17, neg(7), 18, Weight::all()).unwrap();
    // State 18
    left.add_transition(18, neg(5), 19, Weight::all()).unwrap();
    // State 19
    left.add_transition(19, neg(10), 20, Weight::all()).unwrap();
    // State 20
    left.set_final_weight(20, Weight::from_item(1)).unwrap();

    // Build right DWA
    let mut right = DWA::new();
    for _ in 0..22 {
        right.add_state();
    }
    assert_eq!(right.states.len(), 23);

    // State 0
    right.add_transition(0, 5, 1, Weight::from_item(0)).unwrap();
    right.add_transition(0, 6, 2, Weight::from_item(0)).unwrap();
    right.add_transition(0, 8, 3, Weight::from_item(0)).unwrap();
    right.add_transition(0, 9, 4, Weight::from_item(0)).unwrap();
    right.add_transition(0, 10, 5, Weight::from_item(0)).unwrap();
    // State 1
    right.add_transition(1, 7, 4, Weight::all()).unwrap();
    // State 2
    right.add_transition(2, 100, 1, Weight::all()).unwrap();
    // State 3
    right.add_transition(3, 100, 4, Weight::all()).unwrap();
    // State 4
    right.add_transition(4, 0, 6, Weight::all()).unwrap();
    right.add_transition(4, 3, 7, Weight::all()).unwrap();
    right.add_transition(4, 7, 8, Weight::all()).unwrap();
    // State 5
    right.add_transition(5, 5, 1, Weight::all()).unwrap();
    // State 6
    right.add_transition(6, neg(0), 9, Weight::all()).unwrap();
    // State 7
    right.add_transition(7, neg(3), 9, Weight::all()).unwrap();
    // State 8
    right.add_transition(8, neg(7), 9, Weight::all()).unwrap();
    // State 9
    right.add_transition(9, neg(5), 10, Weight::all()).unwrap();
    // State 10
    right.add_transition(10, neg(10), 11, Weight::all()).unwrap();
    // State 11
    right.add_transition(11, 5, 12, Weight::from_iter(0..=1)).unwrap();
    right.add_transition(11, 6, 13, Weight::from_iter(0..=1)).unwrap();
    right.add_transition(11, 8, 14, Weight::from_iter(0..=1)).unwrap();
    right.add_transition(11, 9, 15, Weight::from_iter(0..=1)).unwrap();
    right.add_transition(11, 10, 16, Weight::from_iter(0..=1)).unwrap();
    // State 12
    right.add_transition(12, 7, 15, Weight::all()).unwrap();
    // State 13
    right.add_transition(13, 100, 12, Weight::all()).unwrap();
    // State 14
    right.add_transition(14, 100, 15, Weight::all()).unwrap();
    // State 15
    right.add_transition(15, 0, 17, Weight::all()).unwrap();
    right.add_transition(15, 3, 18, Weight::all()).unwrap();
    right.add_transition(15, 7, 19, Weight::all()).unwrap();
    // State 16
    right.add_transition(16, 5, 12, Weight::all()).unwrap();
    // State 17
    right.add_transition(17, neg(0), 20, Weight::all()).unwrap();
    // State 18
    right.add_transition(18, neg(3), 20, Weight::all()).unwrap();
    // State 19
    right.add_transition(19, neg(7), 20, Weight::all()).unwrap();
    // State 20
    right.add_transition(20, neg(5), 21, Weight::all()).unwrap();
    // State 21
    right.add_transition(21, neg(10), 22, Weight::all()).unwrap();
    // State 22
    right.set_final_weight(22, Weight::from_item(0)).unwrap();

    left.simplify();
    right.simplify();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_concatenate_complex_from_attachment() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    // --- Build LEFT DWA ---
    let mut left = DWA::new();
    for _ in 0..25 {
        left.add_state();
    }
    left.body.start_state = 25;
    assert_eq!(left.states.len(), 26);

    let w_all = Weight::all();
    let w_01 = Weight::from_iter(0..=1);

    // State 0
    left.add_transition(0, 2, 9, w_all.clone()).unwrap();
    left.add_transition(0, 4, 1, w_all.clone()).unwrap();
    left.add_transition(0, 5, 3, w_all.clone()).unwrap();
    left.add_transition(0, 6, 11, w_all.clone()).unwrap();
    left.add_transition(0, 8, 12, w_all.clone()).unwrap();
    left.add_transition(0, 9, 4, w_all.clone()).unwrap();
    left.add_transition(0, 10, 5, w_all.clone()).unwrap();
    // State 3
    left.add_transition(3, 7, 4, w_all.clone()).unwrap();
    // State 4
    left.add_transition(4, 0, 13, w_all.clone()).unwrap();
    left.add_transition(4, 3, 17, w_all.clone()).unwrap();
    left.add_transition(4, 7, 21, w_all.clone()).unwrap();
    // State 5
    left.add_transition(5, 5, 3, w_all.clone()).unwrap();
    // State 6
    left.add_transition(6, neg(5), 7, w_all.clone()).unwrap();
    // State 7
    left.add_transition(7, neg(10), 8, w_all.clone()).unwrap();
    // State 8
    left.set_final_weight(8, w_all.clone()).unwrap();
    // State 9
    left.add_transition(9, 100, 10, w_all.clone()).unwrap();
    // State 10
    left.add_transition(10, 101, 2, w_all.clone()).unwrap();
    // State 11
    left.add_transition(11, 100, 3, w_all.clone()).unwrap();
    // State 12
    left.add_transition(12, 100, 4, w_all.clone()).unwrap();
    // State 13
    left.add_transition(13, neg(0), 14, w_all.clone()).unwrap();
    // State 14
    left.add_transition(14, neg(5), 15, w_all.clone()).unwrap();
    // State 15
    left.add_transition(15, neg(10), 16, w_all.clone()).unwrap();
    // State 16
    left.set_final_weight(16, w_all.clone()).unwrap();
    // State 17
    left.add_transition(17, neg(3), 18, w_all.clone()).unwrap();
    // State 18
    left.add_transition(18, neg(5), 19, w_all.clone()).unwrap();
    // State 19
    left.add_transition(19, neg(10), 20, w_all.clone()).unwrap();
    // State 20
    left.set_final_weight(20, w_all.clone()).unwrap();
    // State 21
    left.add_transition(21, neg(7), 22, w_all.clone()).unwrap();
    // State 22
    left.add_transition(22, neg(5), 23, w_all.clone()).unwrap();
    // State 23
    left.add_transition(23, neg(10), 24, w_all.clone()).unwrap();
    // State 24
    left.set_final_weight(24, w_all.clone()).unwrap();
    // State 25
    left.add_transition(25, 2, 9, w_01.clone()).unwrap();
    left.add_transition(25, 4, 1, w_01.clone()).unwrap();
    left.add_transition(25, 5, 3, w_01.clone()).unwrap();
    left.add_transition(25, 6, 11, w_01.clone()).unwrap();
    left.add_transition(25, 8, 12, w_01.clone()).unwrap();
    left.add_transition(25, 9, 4, w_01.clone()).unwrap();
    left.add_transition(25, 10, 5, w_01.clone()).unwrap();

    // --- Build RIGHT DWA ---
    let mut right = DWA::new();
    right.set_final_weight(0, Weight::all()).unwrap();

    left.simplify();
    right.simplify();

    let c = left.concatenate(&right);
    DWA::stochastic_validate_concatenate(&left, &right, &c, &Weight::all());
}

#[test]
fn test_union_from_debug_log() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    // --- Build LEFT DWA ---
    let mut left = DWA::new();
    for _ in 0..9 {
        left.add_state();
    }
    assert_eq!(left.states.len(), 10);

    left.set_final_weight(0, Weight::from_item(2)).unwrap();
    left.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    left.add_transition(0, 1, 2, Weight::from_iter(0..=1)).unwrap();
    left.add_transition(0, 2, 3, Weight::from_item(0)).unwrap();
    left.add_transition(0, 3, 4, Weight::from_iter(0..=1)).unwrap();

    left.add_transition(1, neg(0), 5, Weight::all()).unwrap();
    left.add_transition(2, 100, 6, Weight::all()).unwrap();
    left.add_transition(3, neg(2), 7, Weight::all()).unwrap();
    // state 4 is sink
    left.add_transition(5, neg(1), 8, Weight::all()).unwrap();
    left.add_transition(7, neg(0), 9, Weight::all()).unwrap();

    left.set_final_weight(8, Weight::all()).unwrap();
    left.set_final_weight(9, Weight::all()).unwrap();

    // --- Build RIGHT DWA ---
    let mut right = DWA::new();
    for _ in 0..12 {
        right.add_state();
    }
    assert_eq!(right.states.len(), 13);

    right.add_transition(0, 1, 1, Weight::from_item(3)).unwrap();
    right.add_transition(0, 2, 2, Weight::from_item(3)).unwrap();
    right.add_transition(0, 3, 3, Weight::from_item(3)).unwrap();

    right.add_transition(1, 100, 4, Weight::all()).unwrap();
    right.add_transition(2, neg(2), 5, Weight::all()).unwrap();
    // state 3 is sink
    // state 4 is sink
    right.add_transition(5, neg(0), 6, Weight::all()).unwrap();

    right.add_transition(6, 0, 7, Weight::from_item(3)).unwrap();
    right.add_transition(6, 1, 8, Weight::from_item(3)).unwrap();
    right.add_transition(6, 3, 9, Weight::from_item(3)).unwrap();

    right.add_transition(7, neg(0), 10, Weight::all()).unwrap();
    right.add_transition(8, 100, 11, Weight::all()).unwrap();
    // state 9 is sink
    right.add_transition(10, neg(1), 12, Weight::all()).unwrap();
    // state 11 is sink
    right.set_final_weight(12, Weight::all()).unwrap();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified1() {
    // This test isolates two simple paths with different initial edge weights.
    // A: path [0, 1] with weight [0]
    // B: path [0, 1, 2] with weight [1]
    // The union should correctly handle both.

    // --- Build LEFT DWA (A) ---
    let mut left = DWA::new();
    left.set_final_weight(0, Weight::from_item(0)).unwrap();

    // --- Build RIGHT DWA (B) ---
    let mut right = DWA::new();
    let s1b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.set_final_weight(s1b, Weight::all()).unwrap();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified2() {
    // This test isolates two simple paths with different initial edge weights.
    // A: path [0, 1] with weight [0]
    // B: path [0, 1, 2] with weight [1]
    // The union should correctly handle both.

    // --- Build LEFT DWA (A) ---
    let mut left = DWA::new();
    let s1a = left.add_state();
    left.add_transition(0, 0, s1a, Weight::from_item(0)).unwrap();
    left.set_final_weight(s1a, Weight::all()).unwrap();

    // --- Build RIGHT DWA (B) ---
    let mut right = DWA::new();
    let s1b = right.add_state();
    let s2b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.add_transition(s1b, 1, s2b, Weight::all()).unwrap();
    right.set_final_weight(s2b, Weight::all()).unwrap();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified2_with_simplification_trick() {
    // This test isolates two simple paths with different initial edge weights.
    // A: path [0, 1] with weight [0]
    // B: path [0, 1, 2] with weight [1]
    // The union should correctly handle both.

    // --- Build LEFT DWA (A) ---
    let mut left = DWA::new();
    let s1a = left.add_state();
    left.add_transition(0, 0, s1a, Weight::from_item(0)).unwrap();
    left.add_transition(0, 1, s1a, Weight::from_item(1)).unwrap();
    left.set_final_weight(s1a, Weight::all()).unwrap();

    // --- Build RIGHT DWA (B) ---
    let mut right = DWA::new();
    let s1b = right.add_state();
    let s2b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.add_transition(s1b, 1, s2b, Weight::all()).unwrap();
    right.set_final_weight(s2b, Weight::all()).unwrap();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_from_debug_log_simplified3() {
    // This test isolates two simple paths with different initial edge weights.
    // A: path [0, 1] with weight [0]
    // B: path [0, 1, 2] with weight [1]
    // The union should correctly handle both.

    // --- Build LEFT DWA (A) ---
    let mut left = DWA::new();
    let s1a = left.add_state();
    let s2a = left.add_state();
    left.add_transition(0, 0, s1a, Weight::from_item(0)).unwrap();
    left.add_transition(s1a, 1, s2a, Weight::all()).unwrap();
    left.set_final_weight(s2a, Weight::all()).unwrap();

    // --- Build RIGHT DWA (B) ---
    let mut right = DWA::new();
    let s1b = right.add_state();
    let s2b = right.add_state();
    let s3b = right.add_state();
    right.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    right.add_transition(s1b, 1, s2b, Weight::all()).unwrap();
    right.add_transition(s2b, 2, s3b, Weight::all()).unwrap();
    right.set_final_weight(s3b, Weight::all()).unwrap();

    let u = left.union(&right);
    DWA::stochastic_validate_union(&left, &right, &u);
}

#[test]
fn test_union_identical_cyclic() {
    // DWA that accepts a* with final weight [1].
    let mut d1 = DWA::new();
    d1.add_transition(d1.body.start_state, 'a' as i16, d1.body.start_state, Weight::all()).unwrap();
    d1.set_final_weight(d1.body.start_state, Weight::from_item(1)).unwrap();

    let d2 = d1.clone();

    let u = d1.union(&d2);

    // The union of two identical automata should be equivalent to the original.
    stochastic_equivalence_test(u, d1);
}

#[test]
fn test_concatenate_disjoint_weights() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

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

    let c = dwa_a.concatenate(&dwa_b);

    let mut combined_word = word_a.clone();
    combined_word.extend_from_slice(&word_b);

    // The weight for this specific split should be empty.
    let wa = dwa_a.eval_word_weight(&word_a);
    let wb = dwa_b.eval_word_weight(&word_b);
    assert_eq!(wa, Weight::from_item(1));
    assert_eq!(wb, Weight::from_item(0));
    assert_eq!((&wa & &wb), Weight::zeros());

    // The concatenated DWA should not accept the combined word, because there are no other
    // accepting paths/splits.
    let wc = c.eval_word_weight(&combined_word);
    assert_eq!(wc, Weight::zeros());

    // The expected weight over all splits should also be empty.
    let expected = DWA::expected_concat_weight(&dwa_a, &dwa_b, &combined_word, &Weight::all());
    assert_eq!(expected, Weight::zeros());
    assert_eq!(wc, expected);
}

#[test]
fn test_simplify_complex_dwa_from_attachment() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    // --- Build LEFT DWA from test_concatenate_complex_from_attachment ---
    let mut left = DWA::new();
    for _ in 0..25 {
        left.add_state();
    }
    left.body.start_state = 25;
    assert_eq!(left.states.len(), 26);

    let w_all = Weight::all();
    let w_01 = Weight::from_iter(0..=1);

    // State 0
    left.add_transition(0, 2, 9, w_all.clone()).unwrap();
    left.add_transition(0, 4, 1, w_all.clone()).unwrap();
    left.add_transition(0, 5, 3, w_all.clone()).unwrap();
    left.add_transition(0, 6, 11, w_all.clone()).unwrap();
    left.add_transition(0, 8, 12, w_all.clone()).unwrap();
    left.add_transition(0, 9, 4, w_all.clone()).unwrap();
    left.add_transition(0, 10, 5, w_all.clone()).unwrap();
    // State 3
    left.add_transition(3, 7, 4, w_all.clone()).unwrap();
    // State 4
    left.add_transition(4, 0, 13, w_all.clone()).unwrap();
    left.add_transition(4, 3, 17, w_all.clone()).unwrap();
    left.add_transition(4, 7, 21, w_all.clone()).unwrap();
    // State 5
    left.add_transition(5, 5, 3, w_all.clone()).unwrap();
    // State 6
    left.add_transition(6, neg(5), 7, w_all.clone()).unwrap();
    // State 7
    left.add_transition(7, neg(10), 8, w_all.clone()).unwrap();
    // State 8
    left.set_final_weight(8, w_all.clone()).unwrap();
    // State 9
    left.add_transition(9, 100, 10, w_all.clone()).unwrap();
    // State 10
    left.add_transition(10, 101, 2, w_all.clone()).unwrap();
    // State 11
    left.add_transition(11, 100, 3, w_all.clone()).unwrap();
    // State 12
    left.add_transition(12, 100, 4, w_all.clone()).unwrap();
    // State 13
    left.add_transition(13, neg(0), 14, w_all.clone()).unwrap();
    // State 14
    left.add_transition(14, neg(5), 15, w_all.clone()).unwrap();
    // State 15
    left.add_transition(15, neg(10), 16, w_all.clone()).unwrap();
    // State 16
    left.set_final_weight(16, w_all.clone()).unwrap();
    // State 17
    left.add_transition(17, neg(3), 18, w_all.clone()).unwrap();
    // State 18
    left.add_transition(18, neg(5), 19, w_all.clone()).unwrap();
    // State 19
    left.add_transition(19, neg(10), 20, w_all.clone()).unwrap();
    // State 20
    left.set_final_weight(20, w_all.clone()).unwrap();
    // State 21
    left.add_transition(21, neg(7), 22, w_all.clone()).unwrap();
    // State 22
    left.add_transition(22, neg(5), 23, w_all.clone()).unwrap();
    // State 23
    left.add_transition(23, neg(10), 24, w_all.clone()).unwrap();
    // State 24
    left.set_final_weight(24, w_all.clone()).unwrap();
    // State 25
    left.add_transition(25, 2, 9, w_01.clone()).unwrap();
    left.add_transition(25, 4, 1, w_01.clone()).unwrap();
    left.add_transition(25, 5, 3, w_01.clone()).unwrap();
    left.add_transition(25, 6, 11, w_01.clone()).unwrap();
    left.add_transition(25, 8, 12, w_01.clone()).unwrap();
    left.add_transition(25, 9, 4, w_01.clone()).unwrap();
    left.add_transition(25, 10, 5, w_01.clone()).unwrap();

    let mut simplified = left.clone();
    simplified.simplify();

    stochastic_equivalence_test(left, simplified);
}

#[test]
fn test_concatenate_from_debug_log() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    let mut base_dwa = DWA::new();
    for _ in 0..12 {
        base_dwa.add_state();
    }
    assert_eq!(base_dwa.states.len(), 13);

    // State 0
    base_dwa.add_transition(0, 6, 1, Weight::all()).unwrap();
    base_dwa.add_transition(0, 7, 4, Weight::all()).unwrap();
    base_dwa.add_transition(0, 10, 5, Weight::all()).unwrap();
    base_dwa.add_transition(0, 11, 6, Weight::all()).unwrap();
    base_dwa.add_transition(0, 12, 3, Weight::all()).unwrap();
    // State 1
    base_dwa.add_transition(1, 9, 6, Weight::all()).unwrap();
    // State 2
    base_dwa.add_transition(2, 0, 7, Weight::all()).unwrap();
    base_dwa.add_transition(2, 4, 11, Weight::all()).unwrap();
    base_dwa.add_transition(2, 9, 12, Weight::all()).unwrap();
    // State 3
    base_dwa.add_transition(3, 6, 1, Weight::all()).unwrap();
    // State 4
    base_dwa.add_transition(4, 100, 1, Weight::all()).unwrap();
    // State 5
    base_dwa.add_transition(5, 100, 6, Weight::all()).unwrap();
    // State 6
    base_dwa.add_transition(6, 100, 2, Weight::all()).unwrap();
    // State 7
    base_dwa.add_transition(7, neg(0), 8, Weight::all()).unwrap();
    // State 8
    base_dwa.add_transition(8, neg(6), 9, Weight::all()).unwrap();
    // State 9
    base_dwa.add_transition(9, neg(12), 10, Weight::all()).unwrap();
    // State 10
    base_dwa.set_final_weight(10, Weight::all()).unwrap();
    // State 11
    base_dwa.add_transition(11, neg(4), 8, Weight::all()).unwrap();
    // State 12
    base_dwa.add_transition(12, neg(9), 8, Weight::all()).unwrap();

    let mut dwa1 = base_dwa.clone();
    dwa1.apply_weight(&Weight::from_item(0));

    let mut dwa2 = base_dwa.clone();
    dwa2.apply_weight(&Weight::from_item(0));

    let c = dwa1.concatenate(&dwa2);

    DWA::stochastic_validate_concatenate(&dwa1, &dwa2, &c, &Weight::all());
}

#[test]
fn test_union_from_panicked_log() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    // --- Build LEFT DWA (A) ---
    let mut a = DWA::new();
    for _ in 0..23 { a.add_state(); }
    assert_eq!(a.states.len(), 24);

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

    // --- Build RIGHT DWA (B) ---
    let mut b = DWA::new();
    for _ in 0..16 { b.add_state(); }
    assert_eq!(b.states.len(), 17);

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

    let u = a.union(&b);
    DWA::stochastic_validate_union(&a, &b, &u);
}

#[test]
fn test_concatenate_default_path_to_final() {
    let mut a = DWA::new();
    let s1a = a.add_state();
    a.add_transition(a.body.start_state, 'a' as i16, s1a, Weight::all()).unwrap();
    a.set_final_weight(s1a, Weight::from_item(1)).unwrap();

    let mut b = DWA::new();
    let s1b = b.add_state();
    b.add_transition(b.body.start_state, 'x' as i16, s1b, Weight::all()).unwrap();
    b.set_final_weight(s1b, Weight::from_item(1)).unwrap();

    let c = a.concatenate(&b);

    // Word "ax" should be accepted. 'a' uses the default transition in A.
    let weight = c.eval_word_weight(&['a' as i16, 'x' as i16]);
    assert_eq!(weight, Weight::from_item(1));

    // Word "x" should not be accepted by C.
    let weight_x = c.eval_word_weight(&['x' as i16]);
    assert_eq!(weight_x, Weight::zeros());
}

#[test]
fn test_simplify() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();
    let s5 = d.add_state();
    let s6 = d.add_state();
    let s7 = d.add_state();
    let s8 = d.add_state();
    let s9 = d.add_state();
    let s10 = d.add_state();
    let s11 = d.add_state();
    let s12 = d.add_state();
    let s13 = d.add_state();

    let w_all = Weight::all(); // Corresponds to [0..=2] in the dump
    let w_1_2 = Weight::from_iter(1..=2);

    // State 0 (start)
    d.add_transition(d.body.start_state, 0, s1, w_all.clone()).unwrap();
    d.add_transition(d.body.start_state, 1, s2, w_all.clone()).unwrap();

    // State 1
    d.add_transition(s1, 0, s3, w_1_2.clone()).unwrap();
    d.add_transition(s1, 3, s4, w_1_2.clone()).unwrap();
    d.add_transition(s1, 7, s5, w_all.clone()).unwrap();
    d.add_transition(s1, 10, s6, w_all.clone()).unwrap();
    d.add_transition(s1, 12, s7, w_all.clone()).unwrap();
    d.add_transition(s1, 13, s5, w_all.clone()).unwrap();

    // State 2
    d.add_transition(s2, 0, s8, w_all.clone()).unwrap();
    d.add_transition(s2, 3, s9, w_all.clone()).unwrap();
    d.add_transition(s2, 7, s8, w_all.clone()).unwrap();
    d.add_transition(s2, 10, s10, w_all.clone()).unwrap();
    d.add_transition(s2, 12, s11, w_all.clone()).unwrap();
    d.add_transition(s2, 13, s8, w_all.clone()).unwrap();

    // State 3
    d.set_final_weight(s3, w_1_2.clone()).unwrap();

    // State 4
    d.add_transition(s4, 7, s3, w_1_2.clone()).unwrap();
    d.add_transition(s4, 13, s3, w_1_2.clone()).unwrap();

    // State 5
    d.set_final_weight(s5, w_all.clone()).unwrap();

    // State 6
    d.add_transition(s6, 100, s12, w_all.clone()).unwrap();

    // State 7
    d.add_transition(s7, 100, s6, w_all.clone()).unwrap();

    // State 8
    d.set_final_weight(s8, w_all.clone()).unwrap();

    // State 9
    d.add_transition(s9, 7, s8, w_all.clone()).unwrap();
    d.add_transition(s9, 13, s8, w_all.clone()).unwrap();

    // State 10
    d.add_transition(s10, 100, s13, w_all.clone()).unwrap();

    // State 11
    d.add_transition(s11, 100, s10, w_all.clone()).unwrap();

    // State 12
    d.add_transition(s12, 13, s5, w_all.clone()).unwrap();

    // State 13
    d.add_transition(s13, 13, s8, w_all.clone()).unwrap();

    // Since there are no negative codes, the DWA should not be changed.
    let expected = d.clone();
    println!("Before simplification:\n{}", d);
    d.simplify();
    println!("After simplification:\n{}", d);

    stochastic_equivalence_test(d, expected);
}

#[test]
fn test_dwa_to_nwa_to_dwa_roundtrip() {
    fn neg(x: i16) -> i16 {
        i16::MIN + x
    }

    // --- Build DWA A ---
    let mut a = DWA::new();
    for _ in 0..23 { a.add_state(); }
    assert_eq!(a.states.len(), 24);

    // State 0:
    a.add_transition(0, 0, 1, Weight::from_item(1)).unwrap();
    a.add_transition(0, 1, 2, Weight::from_item(0)).unwrap();
    a.add_transition(0, 2, 3, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 3, 4, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 4, 5, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 5, 6, Weight::from_item(0)).unwrap();
    a.add_transition(0, 7, 7, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(0, 8, 8, Weight::from_item(0)).unwrap();
    a.add_transition(0, 9, 9, Weight::from_iter(1..=2)).unwrap();

    // State 1:
    a.add_transition(1, neg(0), 10, Weight::from_item(1)).unwrap();

    // State 2:
    a.add_transition(2, neg(1), 11, Weight::from_item(0)).unwrap();

    // State 3:
    a.add_transition(3, 100, 12, Weight::from_item(1)).unwrap();
    a.add_transition(3, neg(2), 13, Weight::from_item(2)).unwrap();

    // State 4:
    a.add_transition(4, neg(3), 13, Weight::from_item(2)).unwrap();
    a.add_transition(4, 5, 14, Weight::from_item(1)).unwrap();

    // State 5:
    a.add_transition(5, 1, 15, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(5, 5, 16, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(5, 8, 17, Weight::from_iter(1..=2)).unwrap();

    // State 6:
    a.add_transition(6, neg(5), 11, Weight::from_item(0)).unwrap();

    // State 7:
    a.add_transition(7, 1, 15, Weight::from_iter(1..=2)).unwrap();
    a.add_transition(7, 5, 16, Weight::from_iter(1..=2)).unwrap();

    // State 8:
    a.add_transition(8, neg(8), 11, Weight::from_item(0)).unwrap();

    // State 9:
    a.add_transition(9, 100, 17, Weight::from_iter(1..=2)).unwrap();

    // State 10:
    a.add_transition(10, neg(1), 18, Weight::from_item(1)).unwrap();

    // State 11:
    a.add_transition(11, neg(4), 19, Weight::from_item(0)).unwrap();

    // State 12:
    a.add_transition(12, 100, 20, Weight::from_item(1)).unwrap();

    // State 13:
    a.add_transition(13, neg(8), 21, Weight::from_item(2)).unwrap();

    // State 14:
    a.add_transition(14, neg(5), 1, Weight::from_item(1)).unwrap();

    // State 15:
    a.add_transition(15, 100, 20, Weight::from_item(1)).unwrap();
    a.add_transition(15, neg(1), 22, Weight::from_item(2)).unwrap();

    // State 16:
    a.add_transition(16, neg(5), 23, Weight::from_iter(1..=2)).unwrap();

    // State 17:
    a.add_transition(17, 100, 7, Weight::from_iter(1..=2)).unwrap();

    // State 18:
    a.set_final_weight(18, Weight::from_item(1)).unwrap();

    // State 19:
    a.set_final_weight(19, Weight::from_item(0)).unwrap();

    // State 20:
    a.add_transition(20, 5, 14, Weight::from_item(1)).unwrap();

    // State 21:
    a.set_final_weight(21, Weight::from_item(2)).unwrap();

    // State 22:
    a.add_transition(22, neg(2), 13, Weight::from_item(2)).unwrap();

    // State 23:
    a.add_transition(23, neg(0), 10, Weight::from_item(1)).unwrap();
    a.add_transition(23, neg(3), 13, Weight::from_item(2)).unwrap();

    println!("Original DWA:\n{}", a);

    let nwa = NWA::from_dwa(&a);
    println!("Converted NWA:\n{}", nwa);

    let mut roundtrip_dwa = nwa.determinize_to_dwa();
    roundtrip_dwa.simplify();

    println!("Roundtrip DWA:\n{}", roundtrip_dwa);

    stochastic_equivalence_test(a, roundtrip_dwa);
}

#[cfg(test)]
mod determinization_tests {
    use super::*;
    use crate::precompute4::weighted_automata::{NWA, NWABuildError, Weight};

    // Helper to build a simple NWA for testing.
    fn nwa_accepts_char(ch: char, weight: Weight) -> NWA {
        let mut nwa = NWA::new();
        let final_state = nwa.states.add_state();
        nwa.add_transition(nwa.body.start_state, ch as i16, final_state, Weight::all()).unwrap();
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
        // NWA for a|b
        let mut nwa = NWA::new();
        let s_a = nwa.states.add_state();
        let s_b = nwa.states.add_state();
        let final_a = nwa.states.add_state();
        let final_b = nwa.states.add_state();
        nwa.add_epsilon(nwa.body.start_state, s_a, Weight::all());
        nwa.add_epsilon(nwa.body.start_state, s_b, Weight::all());
        nwa.add_transition(s_a, 'a' as i16, final_a, Weight::all()).unwrap();
        nwa.add_transition(s_b, 'b' as i16, final_b, Weight::all()).unwrap();
        nwa.states[final_a].final_weight = Some(Weight::from_item(1));
        nwa.states[final_b].final_weight = Some(Weight::from_item(2));

        let dwa = nwa.determinize_to_dwa();

        let mut expected = DWA::new();
        let final_a_dwa = expected.add_state();
        let final_b_dwa = expected.add_state();
        expected.add_transition(expected.body.start_state, 'a' as i16, final_a_dwa, Weight::all()).unwrap();
        expected.add_transition(expected.body.start_state, 'b' as i16, final_b_dwa, Weight::all()).unwrap();
        expected.set_final_weight(final_a_dwa, Weight::from_item(1)).unwrap();
        expected.set_final_weight(final_b_dwa, Weight::from_item(2)).unwrap();

        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_nondeterminism_on_char() {
        // NWA with two transitions on 'a'
        let mut nwa = NWA::new();
        let f1 = nwa.states.add_state();
        let f2 = nwa.states.add_state();
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f1, Weight::from_item(1)).unwrap();
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f2, Weight::from_item(2)).unwrap();
        nwa.states[f1].final_weight = Some(Weight::all());
        nwa.states[f2].final_weight = Some(Weight::all());

        let dwa = nwa.determinize_to_dwa();

        // Expected DWA accepts 'a' with weight [1, 2]
        let mut expected = DWA::new();
        let final_state = expected.add_state();
        expected.add_transition(expected.body.start_state, 'a' as i16, final_state, Weight::from_iter([1, 2])).unwrap();
        expected.set_final_weight(final_state, Weight::all()).unwrap();

        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_weight_partitioning() {
        // NWA with overlapping weights on 'a'
        let mut nwa = NWA::new();
        let f1 = nwa.states.add_state();
        let f2 = nwa.states.add_state();
        // 'a' can lead to f1 with weight [0,1] or f2 with weight [1,2]
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f1, Weight::from_iter(0..=1)).unwrap();
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f2, Weight::from_iter(1..=2)).unwrap();
        // f1 is final for its path, f2 is final for its path
        nwa.states[f1].final_weight = Some(Weight::all());
        nwa.states[f2].final_weight = Some(Weight::all());

        let dwa = nwa.determinize_to_dwa();

        // Expected DWA:
        // On 'a', we get a state. The final weight of this state should be:
        // Atom [0]: from f1 -> [0]
        // Atom [1]: from f1 and f2 -> [1]
        // Atom [2]: from f2 -> [2]
        // Total final weight: [0,1,2]
        // The edge weight should also be [0,1,2]
        let mut expected = DWA::new();
        let final_state = expected.add_state();
        expected.add_transition(expected.body.start_state, 'a' as i16, final_state, Weight::from_iter(0..=2)).unwrap();
        expected.set_final_weight(final_state, Weight::from_iter(0..=2)).unwrap();

        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_empty_nwa() {
        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Truly empty
        let dwa = nwa.determinize_to_dwa();
        assert_eq!(dwa.states.len(), 1);
        assert!(dwa.states[dwa.body.start_state].final_weight.is_none());
        assert!(dwa.states[dwa.body.start_state].transitions.is_empty());
    }

    #[test]
    fn test_det_accepts_nothing() {
        let nwa = NWA::new(); // start state, but no transitions and not final
        let dwa = nwa.determinize_to_dwa();
        let expected = DWA::new();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_accepts_empty_word() {
        let mut nwa = NWA::new();
        nwa.states[nwa.body.start_state].final_weight = Some(Weight::from_item(42));
        let dwa = nwa.determinize_to_dwa();
        let mut expected = DWA::new();
        expected.set_final_weight(expected.body.start_state, Weight::from_item(42)).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_determinize_complex_nwa_from_template() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

        let mut nwa = NWA::new();
        for _ in 0..38 {
            nwa.states.add_state();
        }

        // State 0
        nwa.add_epsilon(0, 6, Weight::all());
        nwa.add_epsilon(0, 10, Weight::all());
        nwa.add_epsilon(0, 13, Weight::all());
        nwa.add_epsilon(0, 14, Weight::all());
        nwa.add_epsilon(0, 15, Weight::all());
        nwa.add_epsilon(0, 17, Weight::all());
        nwa.add_epsilon(0, 19, Weight::all());
        nwa.add_epsilon(0, 20, Weight::all());
        // State 3
        nwa.add_epsilon(3, 21, Weight::all());
        // State 4
        nwa.add_epsilon(4, 22, Weight::all());
        nwa.add_epsilon(4, 23, Weight::all());
        nwa.add_epsilon(4, 28, Weight::all());
        nwa.add_epsilon(4, 33, Weight::all());
        // State 5
        nwa.add_epsilon(5, 38, Weight::all());
        // State 6
        nwa.add_transition(6, 5, 7, Weight::all()).unwrap();
        // State 7
        nwa.add_transition(7, neg(5), 8, Weight::all()).unwrap();
        // State 8
        nwa.add_transition(8, neg(10), 9, Weight::all()).unwrap();
        // State 9
        nwa.states[9].final_weight = Some(Weight::all());
        // State 10
        nwa.add_transition(10, 2, 11, Weight::all()).unwrap();
        // State 11
        // State 12
        // State 13
        nwa.add_transition(13, 4, 1, Weight::all()).unwrap();
        // State 14
        nwa.add_transition(14, 5, 3, Weight::all()).unwrap();
        // State 15
        nwa.add_transition(15, 6, 16, Weight::all()).unwrap();
        // State 16
        // State 17
        nwa.add_transition(17, 8, 18, Weight::all()).unwrap();
        // State 18
        // State 19
        nwa.add_transition(19, 9, 4, Weight::all()).unwrap();
        // State 20
        nwa.add_transition(20, 10, 5, Weight::all()).unwrap();
        // State 21
        nwa.add_transition(21, 7, 4, Weight::all()).unwrap();
        // State 22
        nwa.add_transition(22, 7, 4, Weight::all()).unwrap();
        // State 23
        nwa.add_transition(23, 0, 24, Weight::all()).unwrap();
        // State 24
        nwa.add_transition(24, neg(0), 25, Weight::all()).unwrap();
        // State 25
        nwa.add_transition(25, neg(5), 26, Weight::all()).unwrap();
        // State 26
        nwa.add_transition(26, neg(10), 27, Weight::all()).unwrap();
        // State 27
        nwa.states[27].final_weight = Some(Weight::all());
        // State 28
        nwa.add_transition(28, 3, 29, Weight::all()).unwrap();
        // State 29
        nwa.add_transition(29, neg(3), 30, Weight::all()).unwrap();
        // State 30
        nwa.add_transition(30, neg(5), 31, Weight::all()).unwrap();
        // State 31
        nwa.add_transition(31, neg(10), 32, Weight::all()).unwrap();
        // State 32
        nwa.states[32].final_weight = Some(Weight::all());
        // State 33
        nwa.add_transition(33, 7, 34, Weight::all()).unwrap();
        // State 34
        nwa.add_transition(34, neg(7), 35, Weight::all()).unwrap();
        // State 35
        nwa.add_transition(35, neg(5), 36, Weight::all()).unwrap();
        // State 36
        nwa.add_transition(36, neg(10), 37, Weight::all()).unwrap();
        // State 37
        nwa.states[37].final_weight = Some(Weight::all());
        // State 38
        nwa.add_transition(38, 5, 3, Weight::all()).unwrap();

        let dwa = nwa.determinize_to_dwa();

        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);

        assert!(!weight.is_empty(), "Path should be valid after determinization. Word: {}", format_word(&word));
    }

    #[test]
    fn test_determinize_minimal_failing_nwa_repro() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

        let mut nwa = NWA::new();
        // Need states 0 to 37, so 38 states total.
        for _ in 0..37 { nwa.states.add_state(); }
        assert_eq!(nwa.states.len(), 38);

        let w_all = Weight::all();

        // Epsilon transitions from state 0
        nwa.add_epsilon(0, 6, w_all.clone());
        nwa.add_epsilon(0, 10, w_all.clone());
        nwa.add_epsilon(0, 14, w_all.clone());
        nwa.add_epsilon(0, 19, w_all.clone());

        // Epsilon transitions from state 3
        nwa.add_epsilon(3, 21, w_all.clone());

        // Epsilon transitions from state 4
        nwa.add_epsilon(4, 28, w_all.clone());
        nwa.add_epsilon(4, 33, w_all.clone());

        // Path 1: 0 --eps--> 6 --5--> 7 --neg(5)--> 8 --neg(10)--> 9 (Final)
        nwa.add_transition(6, 5, 7, w_all.clone()).unwrap();
        nwa.add_transition(7, neg(5), 8, w_all.clone()).unwrap();
        nwa.add_transition(8, neg(10), 9, w_all.clone()).unwrap();
        nwa.states[9].final_weight = Some(w_all.clone());

        // Path 2: 0 --eps--> 10 --2--> 11 (sink)
        nwa.add_transition(10, 2, 11, w_all.clone()).unwrap();

        // Path 3: 0 --eps--> 14 --5--> 3 --eps--> 21 --7--> 4
        nwa.add_transition(14, 5, 3, w_all.clone()).unwrap();
        nwa.add_transition(21, 7, 4, w_all.clone()).unwrap();

        // Path 4: 0 --eps--> 19 --9--> 4 --eps--> 28 --3--> 29 --neg(3)--> 30 --neg(5)--> 31 --neg(10)--> 32 (Final)
        nwa.add_transition(19, 9, 4, w_all.clone()).unwrap();
        nwa.add_transition(28, 3, 29, w_all.clone()).unwrap();
        nwa.add_transition(29, neg(3), 30, w_all.clone()).unwrap();
        nwa.add_transition(30, neg(5), 31, w_all.clone()).unwrap();
        nwa.add_transition(31, neg(10), 32, w_all.clone()).unwrap();
        nwa.states[32].final_weight = Some(w_all.clone());

        // Path 5: 0 --eps--> 19 --9--> 4 --eps--> 33 --7--> 34 --neg(7)--> 35 --neg(5)--> 36 --neg(10)--> 37 (Final)
        nwa.add_transition(33, 7, 34, w_all.clone()).unwrap();
        nwa.add_transition(34, neg(7), 35, w_all.clone()).unwrap();
        nwa.add_transition(35, neg(5), 36, w_all.clone()).unwrap();
        nwa.add_transition(36, neg(10), 37, w_all.clone()).unwrap();
        nwa.states[37].final_weight = Some(w_all.clone());

        println!("Constructed NWA:\n{}", nwa);

        let dwa = nwa.determinize_to_dwa();
        println!("Determinized DWA:\n{}", dwa);

        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        // The word [9, 3, neg(3), neg(5), neg(10)] should be accepted by Path 4.
        assert!(!weight.is_empty(), "Path should be valid after determinization. Word: {}", format_word(&word));
    }

    #[test]
    #[ignore] // This test is for finding the minimal repro, it's slow and prints a lot.
    fn test_minimize_failing_nwa() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        enum NwaComponent {
            Epsilon { from: usize, to: usize },
            Transition { from: usize, on: i16, to: usize },
            FinalWeight { state: usize },
        }

        fn build_nwa_from_components(num_states: usize, components: &[NwaComponent]) -> NWA {
            let mut nwa = NWA::new();
            while nwa.states.len() < num_states {
                nwa.states.add_state();
            }
            for component in components {
                match component {
                    NwaComponent::Epsilon { from, to } => nwa.add_epsilon(*from, *to, Weight::all()),
                    NwaComponent::Transition { from, on, to } => nwa.add_transition(*from, *on, *to, Weight::all()).unwrap(),
                    NwaComponent::FinalWeight { state } => nwa.states[*state].final_weight = Some(Weight::all()),
                }
            }
            nwa
        }

        fn check(nwa: &NWA) -> bool {
            let dwa = nwa.determinize_to_dwa();
            let word = vec![9, 3, neg(3), neg(5), neg(10)];
            let weight = dwa.eval_word_weight(&word);
            weight.is_empty() // returns true if it fails (weight is empty)
        }

        let all_components = vec![
            NwaComponent::Epsilon { from: 0, to: 6 }, NwaComponent::Epsilon { from: 0, to: 10 },
            NwaComponent::Epsilon { from: 0, to: 13 }, NwaComponent::Epsilon { from: 0, to: 14 },
            NwaComponent::Epsilon { from: 0, to: 15 }, NwaComponent::Epsilon { from: 0, to: 17 },
            NwaComponent::Epsilon { from: 0, to: 19 }, NwaComponent::Epsilon { from: 0, to: 20 },
            NwaComponent::Epsilon { from: 3, to: 21 }, NwaComponent::Epsilon { from: 4, to: 22 },
            NwaComponent::Epsilon { from: 4, to: 23 }, NwaComponent::Epsilon { from: 4, to: 28 },
            NwaComponent::Epsilon { from: 4, to: 33 }, NwaComponent::Epsilon { from: 5, to: 38 },
            NwaComponent::Transition { from: 6, on: 5, to: 7 }, NwaComponent::Transition { from: 7, on: neg(5), to: 8 },
            NwaComponent::Transition { from: 8, on: neg(10), to: 9 }, NwaComponent::FinalWeight { state: 9 },
            NwaComponent::Transition { from: 10, on: 2, to: 11 }, NwaComponent::Transition { from: 13, on: 4, to: 1 },
            NwaComponent::Transition { from: 14, on: 5, to: 3 }, NwaComponent::Transition { from: 15, on: 6, to: 16 },
            NwaComponent::Transition { from: 17, on: 8, to: 18 }, NwaComponent::Transition { from: 19, on: 9, to: 4 },
            NwaComponent::Transition { from: 20, on: 10, to: 5 }, NwaComponent::Transition { from: 21, on: 7, to: 4 },
            NwaComponent::Transition { from: 22, on: 7, to: 4 }, NwaComponent::Transition { from: 23, on: 0, to: 24 },
            NwaComponent::Transition { from: 24, on: neg(0), to: 25 }, NwaComponent::Transition { from: 25, on: neg(5), to: 26 },
            NwaComponent::Transition { from: 26, on: neg(10), to: 27 }, NwaComponent::FinalWeight { state: 27 },
            NwaComponent::Transition { from: 28, on: 3, to: 29 }, NwaComponent::Transition { from: 29, on: neg(3), to: 30 },
            NwaComponent::Transition { from: 30, on: neg(5), to: 31 }, NwaComponent::Transition { from: 31, on: neg(10), to: 32 },
            NwaComponent::FinalWeight { state: 32 }, NwaComponent::Transition { from: 33, on: 7, to: 34 },
            NwaComponent::Transition { from: 34, on: neg(7), to: 35 }, NwaComponent::Transition { from: 35, on: neg(5), to: 36 },
            NwaComponent::Transition { from: 36, on: neg(10), to: 37 }, NwaComponent::FinalWeight { state: 37 },
            NwaComponent::Transition { from: 38, on: 5, to: 3 },
        ];

        let essential_components = vec![
            NwaComponent::Epsilon { from: 0, to: 19 },
            NwaComponent::Transition { from: 19, on: 9, to: 4 },
            NwaComponent::Epsilon { from: 4, to: 28 },
            NwaComponent::Transition { from: 28, on: 3, to: 29 },
            NwaComponent::Transition { from: 29, on: neg(3), to: 30 },
            NwaComponent::Transition { from: 30, on: neg(5), to: 31 },
            NwaComponent::Transition { from: 31, on: neg(10), to: 32 },
            NwaComponent::FinalWeight { state: 32 },
        ];

        let essential_set: BTreeSet<_> = essential_components.iter().cloned().collect();
        let removable_components: Vec<_> = all_components.into_iter().filter(|c| !essential_set.contains(c)).collect();

        let mut minimal_removable = removable_components.clone();
        let mut i = minimal_removable.len();
        while i > 0 {
            i -= 1;
            let mut next_try_removable = minimal_removable.clone();
            next_try_removable.remove(i);

            let mut current_test_components = essential_components.clone();
            current_test_components.extend_from_slice(&next_try_removable);

            let nwa_to_check = build_nwa_from_components(39, &current_test_components);
            if check(&nwa_to_check) {
                minimal_removable = next_try_removable;
            }
        }

        let mut final_minimal_components = essential_components;
        final_minimal_components.extend(minimal_removable);
        final_minimal_components.sort();

        println!("Minimal set of components to reproduce the bug:");
        for c in &final_minimal_components {
            println!("    {:?},", c);
        }

        let minimal_nwa = build_nwa_from_components(39, &final_minimal_components);
        assert!(check(&minimal_nwa), "Minimal NWA should still fail");
    }

    #[test]
    fn test_determinize_minimal_failing_nwa() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

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
        assert!(!weight.is_empty(), "Path should be valid after determinization. Word: {}", format_word(&word));
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
        // NWA for a|b
        let mut nwa = NWA::new();
        let s_a = nwa.states.add_state();
        let s_b = nwa.states.add_state();
        let final_a = nwa.states.add_state();
        let final_b = nwa.states.add_state();
        nwa.add_epsilon(nwa.body.start_state, s_a, Weight::all());
        nwa.add_epsilon(nwa.body.start_state, s_b, Weight::all());
        nwa.add_transition(s_a, 'a' as i16, final_a, Weight::all()).unwrap();
        nwa.add_transition(s_b, 'b' as i16, final_b, Weight::all()).unwrap();
        nwa.states[final_a].final_weight = Some(Weight::from_item(1));
        nwa.states[final_b].final_weight = Some(Weight::from_item(2));

        let dwa = nwa.determinize_to_dwa_with_rustfst();

        let mut expected = DWA::new();
        let final_a_dwa = expected.add_state();
        let final_b_dwa = expected.add_state();
        expected.add_transition(expected.body.start_state, 'a' as i16, final_a_dwa, Weight::all()).unwrap();
        expected.add_transition(expected.body.start_state, 'b' as i16, final_b_dwa, Weight::all()).unwrap();
        expected.set_final_weight(final_a_dwa, Weight::from_item(1)).unwrap();
        expected.set_final_weight(final_b_dwa, Weight::from_item(2)).unwrap();

        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_nondeterminism_on_char_rustfst() {
        // NWA with two transitions on 'a'
        let mut nwa = NWA::new();
        let f1 = nwa.states.add_state();
        let f2 = nwa.states.add_state();
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f1, Weight::from_item(1)).unwrap();
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f2, Weight::from_item(2)).unwrap();
        nwa.states[f1].final_weight = Some(Weight::all());
        nwa.states[f2].final_weight = Some(Weight::all());

        let dwa = nwa.determinize_to_dwa_with_rustfst();

        // Expected DWA accepts 'a' with weight [1, 2]
        let mut expected = DWA::new();
        let final_state = expected.add_state();
        expected.add_transition(expected.body.start_state, 'a' as i16, final_state, Weight::from_iter([1, 2])).unwrap();
        expected.set_final_weight(final_state, Weight::all()).unwrap();

        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_weight_partitioning_rustfst() {
        // NWA with overlapping weights on 'a'
        let mut nwa = NWA::new();
        let f1 = nwa.states.add_state();
        let f2 = nwa.states.add_state();
        // 'a' can lead to f1 with weight [0,1] or f2 with weight [1,2]
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f1, Weight::from_iter(0..=1)).unwrap();
        nwa.add_transition(nwa.body.start_state, 'a' as i16, f2, Weight::from_iter(1..=2)).unwrap();
        // f1 is final for its path, f2 is final for its path
        nwa.states[f1].final_weight = Some(Weight::all());
        nwa.states[f2].final_weight = Some(Weight::all());

        let dwa = nwa.determinize_to_dwa_with_rustfst();

        // Expected DWA:
        // On 'a', we get a state. The final weight of this state should be:
        // Atom [0]: from f1 -> [0]
        // Atom [1]: from f1 and f2 -> [1]
        // Atom [2]: from f2 -> [2]
        // Total final weight: [0,1,2]
        // The edge weight should also be [0,1,2]
        let mut expected = DWA::new();
        let final_state = expected.add_state();
        expected.add_transition(expected.body.start_state, 'a' as i16, final_state, Weight::from_iter(0..=2)).unwrap();
        expected.set_final_weight(final_state, Weight::from_iter(0..=2)).unwrap();

        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_empty_nwa_rustfst() {
        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Truly empty
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        assert_eq!(dwa.states.len(), 1);
        assert!(dwa.states[dwa.body.start_state].final_weight.is_none());
        assert!(dwa.states[dwa.body.start_state].transitions.is_empty());
    }

    #[test]
    fn test_det_accepts_nothing_rustfst() {
        let nwa = NWA::new(); // start state, but no transitions and not final
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let expected = DWA::new();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_det_accepts_empty_word_rustfst() {
        let mut nwa = NWA::new();
        nwa.states[nwa.body.start_state].final_weight = Some(Weight::from_item(42));
        let dwa = nwa.determinize_to_dwa_with_rustfst();
        let mut expected = DWA::new();
        expected.set_final_weight(expected.body.start_state, Weight::from_item(42)).unwrap();
        stochastic_equivalence_test(dwa, expected);
    }

    #[test]
    fn test_determinize_complex_nwa_from_template_rustfst() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

        let mut nwa = NWA::new();
        for _ in 0..38 {
            nwa.states.add_state();
        }

        // State 0
        nwa.add_epsilon(0, 6, Weight::all());
        nwa.add_epsilon(0, 10, Weight::all());
        nwa.add_epsilon(0, 13, Weight::all());
        nwa.add_epsilon(0, 14, Weight::all());
        nwa.add_epsilon(0, 15, Weight::all());
        nwa.add_epsilon(0, 17, Weight::all());
        nwa.add_epsilon(0, 19, Weight::all());
        nwa.add_epsilon(0, 20, Weight::all());
        // State 3
        nwa.add_epsilon(3, 21, Weight::all());
        // State 4
        nwa.add_epsilon(4, 22, Weight::all());
        nwa.add_epsilon(4, 23, Weight::all());
        nwa.add_epsilon(4, 28, Weight::all());
        nwa.add_epsilon(4, 33, Weight::all());
        // State 5
        nwa.add_epsilon(5, 38, Weight::all());
        // State 6
        nwa.add_transition(6, 5, 7, Weight::all()).unwrap();
        // State 7
        nwa.add_transition(7, neg(5), 8, Weight::all()).unwrap();
        // State 8
        nwa.add_transition(8, neg(10), 9, Weight::all()).unwrap();
        // State 9
        nwa.states[9].final_weight = Some(Weight::all());
        // State 10
        nwa.add_transition(10, 2, 11, Weight::all()).unwrap();
        // State 11
        // State 12
        // State 13
        nwa.add_transition(13, 4, 1, Weight::all()).unwrap();
        // State 14
        nwa.add_transition(14, 5, 3, Weight::all()).unwrap();
        // State 15
        nwa.add_transition(15, 6, 16, Weight::all()).unwrap();
        // State 16
        // State 17
        nwa.add_transition(17, 8, 18, Weight::all()).unwrap();
        // State 18
        // State 19
        nwa.add_transition(19, 9, 4, Weight::all()).unwrap();
        // State 20
        nwa.add_transition(20, 10, 5, Weight::all()).unwrap();
        // State 21
        nwa.add_transition(21, 7, 4, Weight::all()).unwrap();
        // State 22
        nwa.add_transition(22, 7, 4, Weight::all()).unwrap();
        // State 23
        nwa.add_transition(23, 0, 24, Weight::all()).unwrap();
        // State 24
        nwa.add_transition(24, neg(0), 25, Weight::all()).unwrap();
        // State 25
        nwa.add_transition(25, neg(5), 26, Weight::all()).unwrap();
        // State 26
        nwa.add_transition(26, neg(10), 27, Weight::all()).unwrap();
        // State 27
        nwa.states[27].final_weight = Some(Weight::all());
        // State 28
        nwa.add_transition(28, 3, 29, Weight::all()).unwrap();
        // State 29
        nwa.add_transition(29, neg(3), 30, Weight::all()).unwrap();
        // State 30
        nwa.add_transition(30, neg(5), 31, Weight::all()).unwrap();
        // State 31
        nwa.add_transition(31, neg(10), 32, Weight::all()).unwrap();
        // State 32
        nwa.states[32].final_weight = Some(Weight::all());
        // State 33
        nwa.add_transition(33, 7, 34, Weight::all()).unwrap();
        // State 34
        nwa.add_transition(34, neg(7), 35, Weight::all()).unwrap();
        // State 35
        nwa.add_transition(35, neg(5), 36, Weight::all()).unwrap();
        // State 36
        nwa.add_transition(36, neg(10), 37, Weight::all()).unwrap();
        // State 37
        nwa.states[37].final_weight = Some(Weight::all());
        // State 38
        nwa.add_transition(38, 5, 3, Weight::all()).unwrap();

        let dwa = nwa.determinize_to_dwa_with_rustfst();

        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);

        assert!(!weight.is_empty(), "Path should be valid after determinization. Word: {}", format_word(&word));
    }

    #[test]
    fn test_determinize_minimal_failing_nwa_repro_rustfst() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

        let mut nwa = NWA::new();
        // Need states 0 to 37, so 38 states total.
        for _ in 0..37 { nwa.states.add_state(); }
        assert_eq!(nwa.states.len(), 38);

        let w_all = Weight::all();

        // Epsilon transitions from state 0
        nwa.add_epsilon(0, 6, w_all.clone());
        nwa.add_epsilon(0, 10, w_all.clone());
        nwa.add_epsilon(0, 14, w_all.clone());
        nwa.add_epsilon(0, 19, w_all.clone());

        // Epsilon transitions from state 3
        nwa.add_epsilon(3, 21, w_all.clone());

        // Epsilon transitions from state 4
        nwa.add_epsilon(4, 28, w_all.clone());
        nwa.add_epsilon(4, 33, w_all.clone());

        // Path 1: 0 --eps--> 6 --5--> 7 --neg(5)--> 8 --neg(10)--> 9 (Final)
        nwa.add_transition(6, 5, 7, w_all.clone()).unwrap();
        nwa.add_transition(7, neg(5), 8, w_all.clone()).unwrap();
        nwa.add_transition(8, neg(10), 9, w_all.clone()).unwrap();
        nwa.states[9].final_weight = Some(w_all.clone());

        // Path 2: 0 --eps--> 10 --2--> 11 (sink)
        nwa.add_transition(10, 2, 11, w_all.clone()).unwrap();

        // Path 3: 0 --eps--> 14 --5--> 3 --eps--> 21 --7--> 4
        nwa.add_transition(14, 5, 3, w_all.clone()).unwrap();
        nwa.add_transition(21, 7, 4, w_all.clone()).unwrap();

        // Path 4: 0 --eps--> 19 --9--> 4 --eps--> 28 --3--> 29 --neg(3)--> 30 --neg(5)--> 31 --neg(10)--> 32 (Final)
        nwa.add_transition(19, 9, 4, w_all.clone()).unwrap();
        nwa.add_transition(28, 3, 29, w_all.clone()).unwrap();
        nwa.add_transition(29, neg(3), 30, w_all.clone()).unwrap();
        nwa.add_transition(30, neg(5), 31, w_all.clone()).unwrap();
        nwa.add_transition(31, neg(10), 32, w_all.clone()).unwrap();
        nwa.states[32].final_weight = Some(w_all.clone());

        // Path 5: 0 --eps--> 19 --9--> 4 --eps--> 33 --7--> 34 --neg(7)--> 35 --neg(5)--> 36 --neg(10)--> 37 (Final)
        nwa.add_transition(33, 7, 34, w_all.clone()).unwrap();
        nwa.add_transition(34, neg(7), 35, w_all.clone()).unwrap();
        nwa.add_transition(35, neg(5), 36, w_all.clone()).unwrap();
        nwa.add_transition(36, neg(10), 37, w_all.clone()).unwrap();
        nwa.states[37].final_weight = Some(w_all.clone());

        println!("Constructed NWA:\n{}", nwa);

        let dwa = nwa.determinize_to_dwa_with_rustfst();
        println!("Determinized DWA:\n{}", dwa);

        let word = vec![9, 3, neg(3), neg(5), neg(10)];
        let weight = dwa.eval_word_weight(&word);
        // The word [9, 3, neg(3), neg(5), neg(10)] should be accepted by Path 4.
        assert!(!weight.is_empty(), "Path should be valid after determinization. Word: {}", format_word(&word));
    }

    #[test]
    fn test_determinize_minimal_failing_nwa_rustfst() {
        fn neg(x: i16) -> i16 {
            i16::MIN + x
        }

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

        assert!(!weight.is_empty(), "Path should be valid after determinization. Word: {}", format_word(&word));
    }
}