use crate::precompute4::weighted_automata::{DWAState, SimpleBitset, DWA, DWABuildError, I16Map, Weight, format_word};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// --- Stochastic validation controls and RNG ---
const VALIDATION_SAMPLES: usize = 32;
const VALIDATION_MAX_STEPS: usize = 12;
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

fn pick_default_char_for_state(st: &DWAState, rng: &mut SimpleRng) -> i16 {
    let ex = &st.transitions.exceptions;
    // Try random from base alphabet
    if !BASE_ALPHABET.is_empty() {
        let mut idx = rng.gen_usize(BASE_ALPHABET.len());
        for _ in 0..BASE_ALPHABET.len() {
            let ch = BASE_ALPHABET[idx % BASE_ALPHABET.len()];
            if !ex.contains_key(&ch) {
                return ch;
            }
            idx = idx.wrapping_add(1);
        }
    }
    // Fallback: scan integers to find a non-exception char (always exists since exceptions are finite).
    let mut probe: i16 = 0;
    loop {
        if !ex.contains_key(&probe) {
            return probe;
        }
        probe = probe.wrapping_add(1);
    }
}

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
                let mut choices: Vec<i16> = st.transitions.exceptions.keys().copied().collect();
                let has_default = st.transitions.default.is_some();
                let total = choices.len() + if has_default { 1 } else { 0 };
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
                let ch = if has_default && pick == choices.len() {
                    pick_default_char_for_state(st, rng)
                } else {
                    choices[pick]
                };

                let next = st.transitions.get(ch).copied();
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
    fn expected_concat_weight(a: &DWA, b: &DWA, word: &[i16]) -> Weight {
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
            let both = &wa & &wb;
            if !both.is_empty() {
                acc |= &both;
            }
        }
        acc
    }

    fn stochastic_validate_union(a: &DWA, b: &DWA, u: &DWA) {
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

    fn stochastic_validate_concatenate(a: &DWA, b: &DWA, c: &DWA) {
        let mut rng = SimpleRng::from_time();
        for _ in 0..VALIDATION_SAMPLES {
            // Sample accepted paths in A and B; the concatenation of the words should be in C and contain WA ∧ WB.
            if let Some((wa_word, wa)) = a.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                if let Some((wb_word, wb)) = b.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                    let mut w = wa_word.clone();
                    w.extend_from_slice(&wb_word);
                    let wc = c.eval_word_weight(&w);
                    let expected_simple = &wa & &wb;
                    if !expected_simple.is_empty() {
                        assert!(weight_subset(&expected_simple, &wc), "Concatenation missing expected subset.\nword_a: {}\nword_b: {}\nword: {}\nA(wA): {}\nB(wB): {}\nC(wA∘wB): {}\nExpected subset: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA C:\n{}", format_word(&wa_word), format_word(&wb_word), format_word(&w), wa, wb, wc, expected_simple, a, b, c);
                    }
                    // Also verify full expected across all splits equals C's result
                    let expected_all = DWA::expected_concat_weight(a, b, &w);
                    assert_eq!(wc, expected_all, "C(word) != expected union-over-splits(A(prefix) ∧ B(suffix)).\nword_a: {}\nword_b: {}\nword: {}\nC(word): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA C:\n{}", format_word(&wa_word), format_word(&wb_word), format_word(&w), wc, expected_all, a, b, c);
                }
            }

            // Sample accepted paths from C -> must equal union-over-splits(A(prefix) ∧ B(suffix)).
            if let Some((w, wc)) = c.sample_accepted_path_with_rng(&mut rng, VALIDATION_MAX_STEPS) {
                let expected = DWA::expected_concat_weight(a, b, &w);
                assert_eq!(wc, expected, "C(word) != expected union-over-splits(A(prefix) ∧ B(suffix)).\nword: {}\nC(word): {}\nExpected: {}\n\nDWA A:\n{}\n\nDWA B:\n{}\n\nDWA C:\n{}", format_word(&w), wc, expected, a, b, c);
            }
        }
    }
}

pub fn assert_dwa_equivalent(mut a: DWA, mut b: DWA) {
    // Strategy:
    // 1) Simplify both automata to obtain canonical, minimized, and normalized forms
    //    (unreachable pruned, sink-like states collapsed, redundant exceptions removed),
    //    while aggregating edge/default weights as unions across merged states.
    // 2) Perform a BFS isomorphism test from the start states. For each matched pair:
    //    - Compare state weights and final weights (None treated as zeros).
    //    - Compare default transition weights (None treated as zeros).
    //    - For each character in the union of exception keys, compare per-edge weights
    //      (falling back to default if exception weight absent).
    //    - Ensure default and per-exception targets correspond under the evolving bijection.
    // 3) Verify that all states reachable in `b` are matched by some state of `a`.

    a.simplify();
    b.simplify();

    // Helper: convert Option<Weight> to Weight (None => zeros).
    fn opt_w_to_w(ow: &Option<Weight>) -> Weight {
        ow.clone().unwrap_or_else(Weight::zeros)
    }

    // Map a-state -> b-state and its inverse to ensure a bijection.
    let mut map_ab: HashMap<usize, usize> = HashMap::new();
    let mut map_ba: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();

    assert!(
        !a.states.is_empty() && !b.states.is_empty(),
        "Automata should have at least one state after simplification.\n\nDWA A:\n{}\n\nDWA B:\n{}",
        a, b
    );

    let start_a = a.body.start_state;
    let start_b = b.body.start_state;
    map_ab.insert(start_a, start_b);
    map_ba.entry(start_b).or_default().insert(start_a);
    q.push_back((start_a, start_b));

    // Lookup per-edge weight for a specific character in a given state:
    // - If an exception weight is present, use it.
    // - Else, use the default weight (or zeros if absent).
    let edge_weight = |st: &DWAState, ch: i16| -> Weight {
        if let Some(w) = st.trans_weights_exceptions.get(&ch) {
            w.clone()
        } else {
            opt_w_to_w(&st.trans_weight_default)
        }
    };

    while let Some((ia, ib)) = q.pop_front() {
        let sa = &a.states[ia];
        let sb = &b.states[ib];

        // Compare default transition weights (None considered zeros).
        let dwa = opt_w_to_w(&sa.trans_weight_default);
        let dwb = opt_w_to_w(&sb.trans_weight_default);
        assert_eq!(
            dwa, dwb,
            "Default transition weight mismatch at (a:{}, b:{}): a.def_weight={} vs b.def_weight={}\n\nDWA A:\n{}\n\nDWA B:\n{}",
            ia, ib, dwa, dwb, a, b
        );

        // Union of exception keys; after simplify(), both representations should be normalized,
        // but we compute the union to be robust.
        let keys_a: BTreeSet<i16> = sa.transitions.exceptions.keys().cloned().collect();
        let keys_b: BTreeSet<i16> = sb.transitions.exceptions.keys().cloned().collect();
        let all_keys: BTreeSet<i16> = keys_a.union(&keys_b).cloned().collect();

        // Compare and enqueue per-exception transitions.
        let def_a = sa.transitions.default;
        let def_b = sb.transitions.default;

        for ch in all_keys {
            let ta = sa.transitions.exceptions.get(&ch).copied().or(def_a);
            let tb = sb.transitions.exceptions.get(&ch).copied().or(def_b);

            let wa = edge_weight(sa, ch);
            let wb = edge_weight(sb, ch);
            assert_eq!(
                wa, wb,
                "Per-edge weight mismatch on char {} at (a:{}, b:{}): a.edge_weight={} vs b.edge_weight={}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                ch, ia, ib, wa, wb, a, b
            );

            match (ta, tb) {
                (Some(ta_id), Some(tb_id)) => {
                    if let Some(&mapped) = map_ab.get(&ta_id) {
                        assert_eq!(
                            mapped, tb_id,
                            "Transition mismatch on char {} from (a:{}, b:{}): a-target {} is already mapped to b-target {}, but encountered b-target {}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                            ch, ia, ib, ta_id, mapped, tb_id, a, b
                        );
                    } else {
                        map_ab.insert(ta_id, tb_id);
                        map_ba.entry(tb_id).or_default().insert(ta_id);
                        q.push_back((ta_id, tb_id));
                    }
                }
                (None, None) => { /* Both lack transition for this char; fine. */ }
                _ => {
                    panic!(
                        "Presence mismatch for transition on char {} at (a:{}, b:{}): a-target={:?}, b-target={:?}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                        ch, ia, ib, ta, tb, a, b
                    );
                }
            }
        }

        // Compare and enqueue default transitions.
        match (def_a, def_b) {
            (Some(ta_id), Some(tb_id)) => {
                if let Some(&mapped) = map_ab.get(&ta_id) {
                    assert_eq!(
                        mapped, tb_id,
                        "Default transition mismatch from (a:{}, b:{}): a-target {} already mapped to b-target {}, but encountered b-target {}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                        ia, ib, ta_id, mapped, tb_id, a, b
                    );
                } else {
                    map_ab.insert(ta_id, tb_id);
                    map_ba.entry(tb_id).or_default().insert(ta_id);
                    q.push_back((ta_id, tb_id));
                }
            }
            (None, None) => { /* No default on either side; fine. */ }
            _ => {
                panic!(
                    "Default transition presence mismatch at (a:{}, b:{}): a.default={:?}, b.default={:?}\n\nDWA A:\n{}\n\nDWA B:\n{}",
                    ia, ib, def_a, def_b, a, b
                );
            }
        }
    }

    // After establishing the state mapping, verify that the union of final weights
    // for all `a` states mapping to a given `b` state is equal to the final weight
    // of that `b` state. This handles cases where `a` is an unminimized version of `b`.
    for (ib, ias) in &map_ba {
        let mut union_fa = Weight::zeros();
        for &ia in ias {
            union_fa |= &opt_w_to_w(&a.states[ia].final_weight);
        }
        let fb = opt_w_to_w(&b.states[*ib].final_weight);
        assert_eq!(
            union_fa, fb,
            "Aggregated final weight mismatch for b-state {} (a-states: {:?}): union(a)={}, b={}\n\nDWA A:\n{}\n\nDWA B:\n{}",
            ib, ias, union_fa, fb, a, b
        );
    }

    // Ensure we've covered all states reachable from b.start.
    let mut reachable_b: HashSet<usize> = HashSet::new();
    let mut qb: VecDeque<usize> = VecDeque::new();
    reachable_b.insert(b.body.start_state);
    qb.push_back(b.body.start_state);
    while let Some(u) = qb.pop_front() {
        if let Some(d) = b.states[u].transitions.default {
            if reachable_b.insert(d) {
                qb.push_back(d);
            }
        }
        for &v in b.states[u].transitions.exceptions.values() {
            if reachable_b.insert(v) {
                qb.push_back(v);
            }
        }
    }
    let mapped_b: HashSet<usize> = map_ab.values().cloned().collect();
    assert_eq!(
        mapped_b, reachable_b,
        "Reachable-state mismatch in `b`: matched set = {:?}, reachable set = {:?}\n\nDWA A:\n{}\n\nDWA B:\n{}",
        mapped_b, reachable_b, a, b
    );
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
    assert!((&set1 & &zeros).is_empty());
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
    assert_eq!(*dwa.states[0].transitions.get(b'a' as i16).unwrap(), 1);
    assert_eq!(*dwa.states[0].trans_weights_exceptions.get(&(b'a' as i16)).unwrap(), SimpleBitset::from_item(30));

    // Test error cases
    let res = dwa.add_transition(0, b'a' as i16, 1, SimpleBitset::zeros());
    assert!(matches!(res, Err(DWABuildError::TransitionAlreadyExists { from: 0, on: 97 })));

    dwa.set_default_transition(0, 0, SimpleBitset::from_item(40)).unwrap();
    assert_eq!(dwa.states[0].transitions.default, Some(0));
    assert_eq!(dwa.states[0].trans_weight_default, Some(SimpleBitset::from_item(40)));

    let res = dwa.set_default_transition(0, 0, SimpleBitset::zeros());
    assert!(matches!(res, Err(DWABuildError::DefaultTransitionAlreadyExists { from: 0 })));

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
    assert_dwa_equivalent(u, expected);
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
    assert_dwa_equivalent(u, expected);
}

#[test]
fn test_concatenate_simple() {
    let d1 = dwa_accepts_char('a', Weight::from_iter([1, 2]));
    let d2 = dwa_accepts_char('b', Weight::from_iter([2, 3]));
    let c = d1.concatenate(&d2);
    let expected = dwa_from_str("ab", Weight::from_item(2));
    assert_dwa_equivalent(c, expected);
}

#[test]
fn test_apply_weight() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    d.set_final_weight(0, Weight::from_iter(vec![5, 6])).unwrap();
    d.add_transition(0, 'a' as i16, s1, Weight::from_iter(vec![100, 101])).unwrap();
    d.set_default_transition(0, 0, Weight::from_iter(vec![200, 201])).unwrap();

    let gate = Weight::from_iter(vec![6, 11, 101, 201]);
    let new_start = d.apply_weight(&gate);

    assert_eq!(d.body.start_state, new_start);
    let new_start_state = &d.states[new_start];
    assert_eq!(new_start_state.final_weight, Some(Weight::from_item(6)));
    assert_eq!(new_start_state.trans_weights_exceptions.get(&('a' as i16)), Some(&Weight::from_item(101)));
    assert_eq!(new_start_state.trans_weight_default, Some(Weight::from_item(201)));
    assert_eq!(new_start_state.transitions.exceptions.get(&('a' as i16)), Some(&s1));
    assert_eq!(new_start_state.transitions.default, Some(0));
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
fn test_i16map_get_prefers_exception() {
    let mut m = I16Map::with_default(7);
    m.exceptions.insert(3, 42);
    assert_eq!(m.get(3), Some(&42));
    assert_eq!(m.get(4), Some(&7));
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

    assert_dwa_equivalent(u, expected);
}

#[test]
fn test_json_roundtrip_complex() {
    use crate::json_serialization::JSONConvertible;

    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    d.set_default_transition(d.body.start_state, s1, Weight::from_iter(vec![1, 2, 3]))
        .unwrap();
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
    d.set_default_transition(d.body.start_state, s1, Weight::all())
        .unwrap();
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
    a.set_default_transition(s2a, s1a, Weight::all()).unwrap();

    // DWA 'b' lacks these transitions. For inputs '1' and '3', it transitions
    // to an implicit sink. The simplification process should make 'a' equivalent
    // to 'b'.
    let mut b = DWA::new();
    let s1b = b.add_state();
    b.add_transition(0, 0, s1b, Weight::from_item(1)).unwrap();
    b.add_transition(0, 2, s1b, Weight::from_item(0)).unwrap();

    assert_dwa_equivalent(a, b);
}

#[test]
fn test_concatenate_left_start_is_final() {
    // LEFT: DWA (start: 0)
    //   State 0:
    //     weight: []
    //     final_weight: [0]
    let mut left = DWA::new();
    left.set_final_weight(left.body.start_state, Weight::from_item(0)).unwrap();

    // RIGHT: DWA (start: 0)
    //   State 0:
    //     weight: []
    //     final_weight: ALL
    let mut right = DWA::new();
    right.set_final_weight(right.body.start_state, Weight::all()).unwrap();

    let c = left.concatenate(&right);

    let mut expected = DWA::new();
    expected.set_final_weight(expected.body.start_state, Weight::from_item(0)).unwrap();

    assert_dwa_equivalent(c, expected);
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

    a.simplify();

    assert_dwa_equivalent(a, b);
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
    left.set_default_transition(2, 12, Weight::all()).unwrap();
    left.add_transition(3, neg(3), 13, Weight::all()).unwrap();
    left.add_transition(5, 3, 14, Weight::all()).unwrap();
    left.add_transition(5, 7, 9, Weight::all()).unwrap();
    left.set_default_transition(6, 5, Weight::all()).unwrap();
    left.add_transition(7, neg(7), 15, Weight::all()).unwrap();
    left.set_default_transition(8, 9, Weight::all()).unwrap();
    left.add_transition(9, 3, 16, Weight::all()).unwrap();
    left.add_transition(9, 7, 9, Weight::all()).unwrap();
    left.add_transition(10, 5, 5, Weight::all()).unwrap();
    left.add_transition(11, neg(9), 17, Weight::all()).unwrap();
    left.set_default_transition(12, 18, Weight::all()).unwrap();
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
    left.set_default_transition(23, 32, Weight::all()).unwrap();
    left.add_transition(25, 7, 28, Weight::all()).unwrap();
    left.set_default_transition(26, 25, Weight::all()).unwrap();
    left.set_default_transition(27, 28, Weight::all()).unwrap();
    left.add_transition(28, 0, 33, Weight::all()).unwrap();
    left.add_transition(28, 3, 34, Weight::all()).unwrap();
    left.add_transition(28, 7, 35, Weight::all()).unwrap();
    left.add_transition(29, 5, 25, Weight::all()).unwrap();
    left.add_transition(30, neg(9), 36, Weight::all()).unwrap();
    left.add_transition(31, neg(9), 37, Weight::all()).unwrap();
    left.set_default_transition(32, 38, Weight::all()).unwrap();
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
    right.set_default_transition(1, 8, Weight::all()).unwrap();
    right.add_transition(3, 7, 6, Weight::all()).unwrap();
    right.set_default_transition(4, 3, Weight::all()).unwrap();
    right.set_default_transition(5, 6, Weight::all()).unwrap();
    right.add_transition(6, 0, 9, Weight::all()).unwrap();
    right.add_transition(6, 3, 10, Weight::all()).unwrap();
    right.add_transition(6, 7, 11, Weight::all()).unwrap();
    right.add_transition(7, 5, 3, Weight::all()).unwrap();
    right.set_default_transition(8, 12, Weight::all()).unwrap();
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
    right.set_default_transition(22, 29, Weight::all()).unwrap();
    right.add_transition(24, 7, 27, Weight::all()).unwrap();
    right.set_default_transition(25, 24, Weight::all()).unwrap();
    right.set_default_transition(26, 27, Weight::all()).unwrap();
    right.add_transition(27, 0, 30, Weight::all()).unwrap();
    right.add_transition(27, 3, 31, Weight::all()).unwrap();
    right.add_transition(27, 7, 32, Weight::all()).unwrap();
    right.add_transition(28, 5, 24, Weight::all()).unwrap();
    right.set_default_transition(29, 33, Weight::all()).unwrap();
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
    left.set_default_transition(4, 3, Weight::all()).unwrap();
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
    left.set_default_transition(11, 10, Weight::all()).unwrap();
    // State 12
    left.set_default_transition(12, 13, Weight::all()).unwrap();
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
    right.set_default_transition(2, 1, Weight::all()).unwrap();
    // State 3
    right.set_default_transition(3, 4, Weight::all()).unwrap();
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
    right.set_default_transition(13, 12, Weight::all()).unwrap();
    // State 14
    right.set_default_transition(14, 15, Weight::all()).unwrap();
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
    left.set_default_transition(9, 10, w_all.clone()).unwrap();
    // State 10
    left.set_default_transition(10, 2, w_all.clone()).unwrap();
    // State 11
    left.set_default_transition(11, 3, w_all.clone()).unwrap();
    // State 12
    left.set_default_transition(12, 4, w_all.clone()).unwrap();
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

    let c = left.concatenate(&right);
    DWA::stochastic_validate_concatenate(&left, &right, &c);
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
    left.set_default_transition(2, 6, Weight::all()).unwrap();
    left.add_transition(3, neg(2), 7, Weight::all()).unwrap();
    // state 4 is sink
    left.add_transition(5, neg(1), 8, Weight::all()).unwrap();
    // state 6 is sink
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

    right.set_default_transition(1, 4, Weight::all()).unwrap();
    right.add_transition(2, neg(2), 5, Weight::all()).unwrap();
    // state 3 is sink
    // state 4 is sink
    right.add_transition(5, neg(0), 6, Weight::all()).unwrap();

    right.add_transition(6, 0, 7, Weight::from_item(3)).unwrap();
    right.add_transition(6, 1, 8, Weight::from_item(3)).unwrap();
    right.add_transition(6, 3, 9, Weight::from_item(3)).unwrap();

    right.add_transition(7, neg(0), 10, Weight::all()).unwrap();
    right.set_default_transition(8, 11, Weight::all()).unwrap();
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
    assert_dwa_equivalent(u, d1);
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
    assert!((&wa & &wb).is_empty());

    // The concatenated DWA should not accept the combined word, because there are no other
    // accepting paths/splits.
    let wc = c.eval_word_weight(&combined_word);
    assert!(wc.is_empty());

    // The expected weight over all splits should also be empty.
    let expected = DWA::expected_concat_weight(&dwa_a, &dwa_b, &combined_word);
    assert!(expected.is_empty());
    assert_eq!(wc, expected);
}

#[test]
fn test_concatenate_complex_and_simplify() {
    // This test checks if concatenation followed by simplification produces a minimal, correct DWA.
    // The weights are chosen to be overlapping, forcing the concatenated DWA to handle shared paths correctly.

    // dwa1 accepts "a" with weight {1, 10} or "b" with weight {2, 10}.
    // The two paths merge into a single final state.
    let mut dwa1 = DWA::new();
    let s1_final = dwa1.add_state();
    dwa1.add_transition(dwa1.body.start_state, 'a' as i16, s1_final, Weight::from_iter([1, 10])).unwrap();
    dwa1.add_transition(dwa1.body.start_state, 'b' as i16, s1_final, Weight::from_iter([2, 10])).unwrap();
    dwa1.set_final_weight(s1_final, Weight::all()).unwrap();

    // dwa2 accepts "c" with weight {10, 20}.
    let mut dwa2 = DWA::new();
    let s2_final = dwa2.add_state();
    dwa2.add_transition(dwa2.body.start_state, 'c' as i16, s2_final, Weight::from_iter([10, 20])).unwrap();
    dwa2.set_final_weight(s2_final, Weight::all()).unwrap();

    // The concatenate operation will convert to NWA, connect final states of dwa1 to start of dwa2,
    // determinize, and simplify.
    let concatenated = dwa1.concatenate(&dwa2);

    // Expected DWA accepts "ac" and "bc".
    // The weight for "ac" is intersect({1, 10}, {10, 20}) = {10}.
    // The weight for "bc" is intersect({2, 10}, {10, 20}) = {10}.
    // Since both paths lead to the same outcome after 'c', a minimal DWA should merge the paths for 'a' and 'b'.
    let mut expected = DWA::new();
    let s_after_ab = expected.add_state();
    let s_final = expected.add_state();
    expected.add_transition(expected.body.start_state, 'a' as i16, s_after_ab, Weight::from_iter([1, 10])).unwrap();
    expected.add_transition(expected.body.start_state, 'b' as i16, s_after_ab, Weight::from_iter([2, 10])).unwrap();
    expected.add_transition(s_after_ab, 'c' as i16, s_final, Weight::from_iter([10, 20])).unwrap();
    expected.set_final_weight(s_final, Weight::all()).unwrap();

    assert_dwa_equivalent(concatenated, expected);
}