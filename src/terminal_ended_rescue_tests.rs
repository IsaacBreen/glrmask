//! Phase-2 terminal-ended rescue coverage.
//!
//! Every mask expectation is computed by the exact prefix-viability oracle: a
//! CYK-style fixpoint over `(nonterminal, from, to)` intersected with the
//! prefix DFA. Bounded word enumeration is never used to decide masks.
//!
//! The same oracle and token vocabulary are used to compare a rescued template
//! constraint against the exact TokenTrie compilation of the original source.
//! A rescue that declines is surfaced as the original typed strategy decline,
//! never as a silent TokenTrie recompile.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

use crate::compiler::build_budget::TestBudgetLimit;
use crate::compiler::terminal_ended_rescue::{
    reset_test_rescue_entries, test_rescue_entries,
};
use crate::{BuildOptions, Constraint, Grammar, MaskStrategy, ParserBackend, Vocab};

// One single-byte token per terminal used by the witnesses. Keeping the
// vocabulary small keeps the exact oracle and the TokenTrie comparison focused
// on the grammar's mask language rather than tokenizer cross-products.
/// The rescue cases all drive the shared Rayon pools. Running them
/// concurrently oversubscribes the machine and turns a ~20 s sequential suite
/// into minutes of contention, so the module serializes its own tests.
static RESCUE_TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    RESCUE_TEST_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const TOKEN_BYTES: [&[u8]; 4] = [b"a", b"b", b"c", b"d"];

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"d".to_vec()),
    ])
}

#[derive(Clone, Debug)]
struct Case {
    start: usize,
    bound: usize,
    rules: Vec<(usize, Vec<isize>)>,
}

fn case(start: usize, bound: usize, rules: &[(usize, &[isize])]) -> Case {
    Case {
        start,
        bound,
        rules: rules
            .iter()
            .map(|(lhs, rhs)| (*lhs, rhs.to_vec()))
            .collect(),
    }
}

fn terminal_byte(symbol: isize) -> u8 {
    b'a' + ((-symbol - 1) as u8)
}

fn lark(case: &Case) -> String {
    let mut by_lhs: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (lhs, rhs) in &case.rules {
        let mut text = String::new();
        for symbol in rhs {
            if !text.is_empty() {
                text.push(' ');
            }
            if *symbol >= 0 {
                text.push_str(&format!("n{symbol}"));
            } else {
                text.push('"');
                text.push(terminal_byte(*symbol) as char);
                text.push('"');
            }
        }
        if text.is_empty() {
            text.push_str("\"\"");
        }
        by_lhs.entry(*lhs).or_default().push(text);
    }
    let mut source = format!("start: n{}\n", case.start);
    for (lhs, alternatives) in &by_lhs {
        source.push_str(&format!("n{lhs}: {}\n", alternatives.join(" | ")));
    }
    source
}

/// Exact CFG/DFA intersection fixpoint. See the oracle in
/// `budget_fallback_tests.rs`.
fn derive_relation(case: &Case, target: &[u8], absorbing: bool) -> Vec<Vec<Vec<bool>>> {
    let n = target.len();
    let states = n + 1;
    let nonterminals = case
        .rules
        .iter()
        .flat_map(|(lhs, rhs)| {
            std::iter::once(*lhs).chain(rhs.iter().filter(|s| **s >= 0).map(|s| *s as usize))
        })
        .max()
        .map_or(1, |max| max + 1);
    let mut relation = vec![vec![vec![false; states]; states]; nonterminals];
    for (lhs, rhs) in &case.rules {
        if rhs.is_empty() {
            for state in 0..states {
                relation[*lhs][state][state] = true;
            }
        }
    }
    loop {
        let mut changed = false;
        for (lhs, rhs) in &case.rules {
            let mut composed: Vec<Vec<bool>> = (0..states)
                .map(|from| (0..states).map(|to| from == to).collect())
                .collect();
            for symbol in rhs {
                let step: Vec<Vec<bool>> = if *symbol >= 0 {
                    relation[*symbol as usize].clone()
                } else {
                    let byte = terminal_byte(*symbol);
                    let mut step = vec![vec![false; states]; states];
                    for from in 0..states {
                        let to = if from < n {
                            (target[from] == byte).then_some(from + 1)
                        } else if absorbing {
                            Some(n)
                        } else {
                            None
                        };
                        if let Some(to) = to {
                            step[from][to] = true;
                        }
                    }
                    step
                };
                let mut next = vec![vec![false; states]; states];
                for from in 0..states {
                    for mid in 0..states {
                        if composed[from][mid] {
                            for to in 0..states {
                                if step[mid][to] {
                                    next[from][to] = true;
                                }
                            }
                        }
                    }
                }
                composed = next;
            }
            for from in 0..states {
                for to in 0..states {
                    if composed[from][to] && !relation[*lhs][from][to] {
                        relation[*lhs][from][to] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    relation
}

fn viable(case: &Case, prefix: &[u8]) -> bool {
    derive_relation(case, prefix, true)[case.start][0][prefix.len()]
}

fn accepts(case: &Case, word: &[u8]) -> bool {
    derive_relation(case, word, false)[case.start][0][word.len()]
}

fn allowed(mask: &[u32], token: usize) -> bool {
    mask.get(token / 32)
        .is_some_and(|word| word & (1 << (token % 32)) != 0)
}

fn fill_mask(constraint: &Constraint, prefix: &[u32]) -> (Vec<u32>, bool) {
    let mut state = constraint.start();
    for &token in prefix {
        state
            .commit_token(token)
            .unwrap_or_else(|error| panic!("commit failed at {prefix:?}: {error}"));
    }
    let mut mask = vec![0u32; constraint.mask_len()];
    state.fill_mask(&mut mask);
    (mask, state.is_accepting())
}

fn compile(case: &Case, strategy: MaskStrategy) -> Constraint {
    try_compile(case, strategy)
        .unwrap_or_else(|error| panic!("{strategy:?} compile failed: {error}"))
}

fn try_compile(case: &Case, strategy: MaskStrategy) -> crate::Result<Constraint> {
    let source = lark(case);
    Grammar::from_lark(&source).compile_with_build_options(
        &vocab(),
        BuildOptions::default().optimization(strategy),
    )
}

/// A declined rescue must surface the requested strategy's typed decline; a
/// successful compile would mean it silently recompiled with TokenTrie.
fn expect_strategy_decline(result: crate::Result<Constraint>, context: &str) -> crate::Error {
    match result {
        Ok(constraint) => panic!(
            "{context}: compiled with {:?} instead of returning a typed decline",
            constraint.parser_backend(),
        ),
        Err(error) => {
            assert!(
                error.is_strategy_decline(),
                "{context}: returned a non-decline error: {error}",
            );
            error
        }
    }
}

/// Exact oracle: templates must agree with prefix viability and membership.
fn check_exact_oracle(constraint: &Constraint, case: &Case) {
    let mut queue = VecDeque::from([(constraint.start(), Vec::<u8>::new())]);
    let mut seen = BTreeSet::<Vec<u8>>::new();
    let mut viability = BTreeMap::<Vec<u8>, bool>::new();
    let mut membership = BTreeMap::<Vec<u8>, bool>::new();
    let mut checked = 0usize;
    while let Some((state, prefix)) = queue.pop_front() {
        if !seen.insert(prefix.clone()) {
            continue;
        }
        let mut mask = vec![0u32; constraint.mask_len()];
        state.fill_mask(&mut mask);
        for (token, token_bytes) in TOKEN_BYTES.iter().enumerate() {
            let mut candidate = prefix.clone();
            candidate.extend_from_slice(token_bytes);
            let expected = *viability
                .entry(candidate.clone())
                .or_insert_with(|| viable(case, &candidate));
            assert_eq!(
                allowed(&mask, token),
                expected,
                "oracle mismatch at prefix {:?} token {token} (candidate {:?})",
                String::from_utf8_lossy(&prefix),
                String::from_utf8_lossy(&candidate),
            );
        }
        for id in TOKEN_BYTES.len()..(mask.len() * 32) {
            assert!(
                !allowed(&mask, id),
                "unexpected token {id} at prefix {:?}",
                String::from_utf8_lossy(&prefix),
            );
        }
        let expected_accept = *membership
            .entry(prefix.clone())
            .or_insert_with(|| accepts(case, &prefix));
        assert_eq!(
            state.is_accepting(),
            expected_accept,
            "accept mismatch at {:?}",
            String::from_utf8_lossy(&prefix),
        );
        for (token, token_bytes) in TOKEN_BYTES.iter().enumerate() {
            if prefix.len() + token_bytes.len() > case.bound || !allowed(&mask, token) {
                continue;
            }
            let mut next = prefix.clone();
            next.extend_from_slice(token_bytes);
            let mut committed = state.clone();
            committed.commit_token(token as u32).unwrap_or_else(|error| {
                panic!(
                    "commit failed at {:?}: {error}",
                    String::from_utf8_lossy(&prefix)
                )
            });
            queue.push_back((committed, next));
        }
        checked += 1;
    }
    assert!(checked > 0);
}

/// The rescued template must match the exact TokenTrie compilation at every
/// token prefix up to the case bound.
fn check_matches_token_trie(rescued: &Constraint, reference: &Constraint, case: &Case) {
    let mut queue = VecDeque::from([(Vec::<u32>::new(), Vec::<u8>::new())]);
    let mut seen = BTreeSet::<Vec<u32>>::new();
    let mut checked = 0usize;
    while let Some((path, prefix)) = queue.pop_front() {
        if !seen.insert(path.clone()) {
            continue;
        }
        let (mask, accepting) = fill_mask(rescued, &path);
        let (reference_mask, reference_accepting) = fill_mask(reference, &path);
        assert_eq!(
            mask,
            reference_mask,
            "rescue mask differs from TokenTrie at prefix {:?}",
            String::from_utf8_lossy(&prefix),
        );
        assert_eq!(
            accepting,
            reference_accepting,
            "rescue acceptance differs from TokenTrie at {:?}",
            String::from_utf8_lossy(&prefix),
        );
        for (token, token_bytes) in TOKEN_BYTES.iter().enumerate() {
            if prefix.len() + token_bytes.len() > case.bound || !allowed(&mask, token) {
                continue;
            }
            let mut next_path = path.clone();
            next_path.push(token as u32);
            let mut next_prefix = prefix.clone();
            next_prefix.extend_from_slice(token_bytes);
            queue.push_back((next_path, next_prefix));
        }
        checked += 1;
    }
    assert!(checked > 0);
}

fn check_rescued(case: &Case, label: &str) {
    let reference = compile(case, MaskStrategy::TokenTrie);
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let started = Instant::now();
        let rescued = compile(case, strategy);
        eprintln!(
            "[terminal-ended-rescue] {label} strategy={strategy:?} compile_ms={:.1} backend={:?}",
            started.elapsed().as_secs_f64() * 1000.0,
            rescued.parser_backend(),
        );
        assert_eq!(
            rescued.parser_backend(),
            ParserBackend::TemplateDfa,
            "{label} {strategy:?} did not take the terminal-ended rescue",
        );
        check_matches_token_trie(&rescued, &reference, case);
        check_exact_oracle(&rescued, case);
    }
}

const V2_494: &[(usize, &[isize])] = &[
    (0, &[-3]),
    (0, &[2, 1, -2]),
    (0, &[1]),
    (1, &[-3]),
    (1, &[2]),
    (1, &[1, 2]),
    (2, &[-3]),
    (2, &[-1, 0]),
    (2, &[-2, 1]),
];

const V2_267: &[(usize, &[isize])] = &[
    (2, &[-1]),
    (2, &[-3, 1]),
    (2, &[0, 0, 2]),
    (2, &[]),
    (0, &[-1]),
    (0, &[-2]),
    (0, &[-3, 1]),
    (1, &[-1]),
    (1, &[-3, 2]),
    (1, &[-3]),
    (1, &[-1, 1, 2]),
    (3, &[-2]),
    (3, &[2, 0, -1]),
    (3, &[0, 1, 1]),
    (3, &[]),
];

const V2_1654: &[(usize, &[isize])] = &[
    (0, &[-2]),
    (0, &[1, 0]),
    (0, &[-3, 0]),
    (0, &[2]),
    (1, &[-3]),
    (1, &[-2, 2]),
    (1, &[0, 0]),
    (2, &[-2]),
    (2, &[]),
    (2, &[2, -3, 0]),
    (2, &[-3, 1, -3]),
];

const SEED_2530_MINIMISED: &str = "start: n3\nn0: \"c\" | n3 |\nn1: \"a\" |\nn2: | n0 n1 n2\nn3: \"b\" n2 n0 | n2 \"b\"\n";
const SEED_2530_FULL: &str = "start: n3\nn0: \"c\" | n3 |  |\nn1: \"a\" |  | \"b\" \"b\" \"c\" | \"c\" \"c\"\nn2: \"a\" |  | n0 n1 n2\nn3: \"c\" | \"b\" n2 n0 | n0 \"a\" n0 | n2 \"b\" n0\n";
const SEED_360_MINIMISED: &str = "start: n2\nn0: \"b\" |  | n1 \"c\" n2\nn1: n1 n0 n0 | n1 n1 |\nn2: n2 n0 \"a\" | n0 n1\n";
const SEED_360_FULL: &str = "start: n2\nn0: \"b\" |  | n1 \"c\" n2 |\nn1: \"a\" | n1 n0 n0 | n1 n1 |\nn2: \"b\" |  | n2 n0 \"a\" | n0 n1\n";

fn seed_cfg(rules: &[(usize, &[isize])], start: usize, bound: usize) -> Case {
    case(start, bound, rules)
}

/// The grammar source for the 2530/360 seeds already includes empty
/// alternatives in a shape the generic Lark emitter would reorder. Parse the
/// canonical text directly instead.
fn compile_source(source: &str, strategy: MaskStrategy) -> Constraint {
    Grammar::from_lark(source)
        .compile_with_build_options(&vocab(), BuildOptions::default().optimization(strategy))
        .unwrap_or_else(|error| panic!("{strategy:?} compile failed for {source:?}: {error}"))
}

fn check_source_case(case: &Case, source: &str, label: &str) {
    let reference = compile_source(source, MaskStrategy::TokenTrie);
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let started = Instant::now();
        let rescued = compile_source(source, strategy);
        eprintln!(
            "[terminal-ended-rescue] {label} strategy={strategy:?} compile_ms={:.1} backend={:?}",
            started.elapsed().as_secs_f64() * 1000.0,
            rescued.parser_backend(),
        );
        assert_eq!(
            rescued.parser_backend(),
            ParserBackend::TemplateDfa,
            "{label} {strategy:?} did not take the terminal-ended rescue",
        );
        check_matches_token_trie(&rescued, &reference, case);
        check_exact_oracle(&rescued, case);
    }
}

/// Embedded phase-1 fuzz witnesses. Terminal values are `-(id + 1)`.
fn witness_rules(seed: u32) -> (usize, Vec<(usize, Vec<isize>)>) {
    let (start, raw): (usize, Vec<(usize, Vec<isize>)>) = match seed {
        385 => (
            0,
            vec![
                (0, vec![-2]),
                (1, vec![-1]),
                (0, vec![1, -1, -1]),
                (0, vec![1]),
                (0, vec![0, -1, 1]),
                (1, vec![0, 0, 1]),
                (1, vec![]),
                (1, vec![-1, 0]),
            ],
        ),
        315 => (
            0,
            vec![
                (0, vec![-3]),
                (1, vec![-1]),
                (2, vec![-1]),
                (3, vec![-2]),
                (0, vec![0, 3]),
                (0, vec![1]),
                (1, vec![0, 2, 2]),
                (2, vec![]),
                (2, vec![-1, -1]),
                (2, vec![-1]),
                (3, vec![]),
                (3, vec![3, 1, 1]),
                (3, vec![-3, 3]),
            ],
        ),
        218 => (
            2,
            vec![
                (0, vec![-2]),
                (1, vec![-2]),
                (2, vec![-3]),
                (0, vec![-3, 1]),
                (0, vec![-2, 2, 1]),
                (1, vec![-3, 0, 1]),
                (1, vec![]),
                (2, vec![]),
                (2, vec![-1, -3, 1]),
                (2, vec![0, 1]),
            ],
        ),
        2633 => (
            2,
            vec![
                (0, vec![-2]),
                (1, vec![-3]),
                (2, vec![-1]),
                (3, vec![-1]),
                (0, vec![3, 3]),
                (1, vec![2, 2]),
                (1, vec![1, 2, 1]),
                (1, vec![]),
                (2, vec![-2]),
                (2, vec![3]),
                (2, vec![0, -2, 3]),
                (3, vec![0, 1]),
                (3, vec![3]),
                (3, vec![1, -3, 1]),
            ],
        ),
        other => panic!("no embedded witness for seed {other}"),
    };
    (start, raw)
}

#[test]
fn reduction_cycle_seeds_take_the_rescue_and_match_the_exact_oracle() {
    let _serial = serial();
    check_rescued(
        &case(0, 3, V2_494),
        "V2_494",
    );
    check_rescued(
        &case(0, 3, V2_1654),
        "V2_1654",
    );

    let seed_2530_minimised = seed_cfg(
        &[
            (0, &[4]),
            (1, &[-3]),
            (1, &[4]),
            (1, &[]),
            (2, &[-1]),
            (2, &[]),
            (3, &[]),
            (3, &[1, 2, 3]),
            (4, &[-2, 3, 1]),
            (4, &[3, -2]),
        ],
        0,
        4,
    );
    check_source_case(&seed_2530_minimised, SEED_2530_MINIMISED, "seed_2530_minimised");

    let seed_360_minimised = seed_cfg(
        &[
            (0, &[3]),
            (1, &[-2]),
            (1, &[]),
            (1, &[2, -3, 3]),
            (2, &[2, 1, 1]),
            (2, &[2, 2]),
            (2, &[]),
            (3, &[3, 1, -1]),
            (3, &[1, 2]),
        ],
        0,
        4,
    );
    check_source_case(&seed_360_minimised, SEED_360_MINIMISED, "seed_360_minimised");
}

#[test]
fn full_reduction_cycle_seeds_take_the_rescue_and_match_the_exact_oracle() {
    let _serial = serial();
    let seed_2530_full = seed_cfg(
        &[
            (0, &[4]),
            (1, &[-3]),
            (1, &[4]),
            (1, &[]),
            (2, &[-1]),
            (2, &[]),
            (2, &[-2, -2, -3]),
            (2, &[-3, -3]),
            (3, &[-1]),
            (3, &[]),
            (3, &[1, 2, 3]),
            (4, &[-3]),
            (4, &[-2, 3, 1]),
            (4, &[1, -1, 1]),
            (4, &[3, -2, 1]),
        ],
        0,
        4,
    );
    check_source_case(&seed_2530_full, SEED_2530_FULL, "seed_2530_full");

    let seed_360_full = seed_cfg(
        &[
            (0, &[3]),
            (1, &[-2]),
            (1, &[]),
            (1, &[2, -3, 3]),
            (2, &[-1]),
            (2, &[2, 1, 1]),
            (2, &[2, 2]),
            (2, &[]),
            (3, &[-2]),
            (3, &[]),
            (3, &[3, 1, -1]),
            (3, &[1, 2]),
        ],
        0,
        4,
    );
    check_source_case(&seed_360_full, SEED_360_FULL, "seed_360_full");
}

fn check_witness(seed: u32) {
    let (start, rules) = witness_rules(seed);
    let case = Case {
        start,
        bound: 3,
        rules,
    };
    check_rescued(&case, &format!("witness_{seed}"));
}

// Split per witness so the (much larger) 2633 rescue build overlaps the other
// witnesses under libtest's normal parallel schedule.
#[test]
fn phase1_witness_315_takes_the_rescue_and_matches_the_exact_oracle() {
    let _serial = serial();
    check_witness(315);
}

#[test]
fn phase1_witness_218_takes_the_rescue_and_matches_the_exact_oracle() {
    let _serial = serial();
    check_witness(218);
}

/// The rescued 2633 template costs more than the default template budget
/// (template_specialization alone charges ~8.4M of the 10M units, about 15x the
/// largest real grammar), so the rescue declines within ~2 s and the public
/// boundary returns the original typed decline. This pins the decline and
/// bounds the time spent before it is returned instead of recompiling with
/// TokenTrie.
#[test]
fn phase1_witness_2633_rescue_declines_quickly_with_typed_decline() {
    let _serial = serial();
    let (start, rules) = witness_rules(2633);
    let case = Case {
        start,
        bound: 3,
        rules,
    };
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let started = Instant::now();
        let error = expect_strategy_decline(
            try_compile(&case, strategy),
            &format!("witness_2633 {strategy:?}"),
        );
        eprintln!(
            "[terminal-ended-rescue] witness_2633 strategy={strategy:?} decline_ms={:.1} error={error}",
            started.elapsed().as_secs_f64() * 1000.0,
        );
        assert!(
            started.elapsed().as_secs_f64() < 10.0,
            "witness_2633 {strategy:?} took {:?} before returning its decline",
            started.elapsed(),
        );
    }
}

#[test]
fn phase1_witness_385_takes_the_rescue_and_matches_the_exact_oracle() {
    let _serial = serial();
    check_witness(385);
}

#[test]
fn v2_267_legacy_attempt_declines_quickly_with_exact_masks() {
    let _serial = serial();
    let case = case(2, 4, V2_267);
    let reference = compile(&case, MaskStrategy::TokenTrie);
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let started = Instant::now();
        let compiled = compile(&case, strategy);
        eprintln!(
            "[terminal-ended-rescue] V2_267 strategy={strategy:?} compile_ms={:.1} backend={:?}",
            started.elapsed().as_secs_f64() * 1000.0,
            compiled.parser_backend(),
        );
        assert_eq!(compiled.parser_backend(), ParserBackend::TemplateDfa);
        check_matches_token_trie(&compiled, &reference, &case);
        check_exact_oracle(&compiled, &case);
    }
}

#[test]
fn legacy_template_grammar_does_not_enter_the_rescue() {
    let _serial = serial();
    let case = case(0, 3, &[(0, &[-1]), (0, &[-2, -1])]);
    reset_test_rescue_entries();
    let reference = compile(&case, MaskStrategy::TokenTrie);
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let compiled = compile(&case, strategy);
        assert_eq!(
            compiled.parser_backend(),
            ParserBackend::TemplateDfa,
            "{strategy:?} did not use the legacy template path",
        );
        check_matches_token_trie(&compiled, &reference, &case);
        check_exact_oracle(&compiled, &case);
    }
    assert_eq!(
        test_rescue_entries(),
        0,
        "a grammar the legacy template path already compiles entered the rescue route",
    );
}

#[test]
fn nullable_right_recursive_start_is_preserved_by_the_rescue() {
    let _serial = serial();
    // start -> a | epsilon ; a -> "a" a | "a"
    let case = case(0, 4, &[(0, &[1]), (0, &[]), (1, &[-1, 1]), (1, &[-1])]);
    let reference = compile(&case, MaskStrategy::TokenTrie);
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let rescued = compile(&case, strategy);
        assert_eq!(
            rescued.parser_backend(),
            ParserBackend::TemplateDfa,
            "{strategy:?} did not take the terminal-ended rescue",
        );
        assert!(
            rescued.start().is_accepting(),
            "{strategy:?} lost source start nullability",
        );
        check_matches_token_trie(&rescued, &reference, &case);
        check_exact_oracle(&rescued, &case);
    }
}

/// Under a budget of 1 unit even grammar normalization declines, so both
/// template strategies must attempt the rescue and then return their original
/// typed decline. Neither may recompile with TokenTrie.
#[test]
fn tiny_budget_returns_typed_decline() {
    let _serial = serial();
    let case = case(
        0,
        4,
        &[
            (0, &[4]),
            (1, &[-1]),
            (1, &[4]),
            (1, &[]),
            (2, &[0]),
            (2, &[]),
            (3, &[]),
            (3, &[1, 2, 3]),
            (4, &[1, 3, 1]),
            (4, &[3, 1]),
        ],
    );
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        reset_test_rescue_entries();
        eprintln!("[tiny-budget] compiling {strategy:?} under limit=1");
        let error = {
            let _limit = TestBudgetLimit::set(1);
            expect_strategy_decline(
                try_compile(&case, strategy),
                &format!("tiny_budget {strategy:?}"),
            )
        };
        eprintln!(
            "[tiny-budget] {strategy:?} declined rescue_entries={} error={error}",
            test_rescue_entries(),
        );
        assert!(
            test_rescue_entries() > 0,
            "{strategy:?} did not attempt the rescue before returning its decline",
        );
    }
}

/// Fuzz pass-A cases on 605da7551 that exhausted memory in grammar
/// normalization or ParserDwa construction before the budget could decline.
/// Each must publish exact masks within the time bound under both template
/// strategies through the rescue; a rescue decline is returned as the original
/// typed strategy error.
fn check_fuzz_case(label: &str, start: usize, bound: usize, rules: &[(usize, &[isize])]) {
    let case = case(start, bound, rules);
    let reference = compile(&case, MaskStrategy::TokenTrie);
    for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let started = Instant::now();
        let compiled = compile(&case, strategy);
        eprintln!(
            "[terminal-ended-rescue] {label} strategy={strategy:?} compile_ms={:.1} backend={:?}",
            started.elapsed().as_secs_f64() * 1000.0,
            compiled.parser_backend(),
        );
        assert!(
            started.elapsed().as_secs_f64() < 10.0,
            "{label} {strategy:?} took {:?} before publishing",
            started.elapsed(),
        );
        check_matches_token_trie(&compiled, &reference, &case);
        check_exact_oracle(&compiled, &case);
    }
}

#[test]
fn fuzz_605_seed_581_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_605_581", 0, 3, &[(0, &[-3]), (1, &[-1]), (0, &[1]), (1, &[0]), (1, &[1, -3, 1]), (1, &[1, -2, 0])]);
}

#[test]
fn fuzz_605_seed_780_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_605_780", 1, 4, &[(0, &[-2]), (1, &[-3]), (0, &[]), (0, &[0, -2, 1]), (0, &[1, -3, 0]), (1, &[1, 1]), (1, &[0, -1]), (1, &[-3, -2, 0])]);
}

#[test]
fn fuzz_605_seed_3293_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_605_3293", 1, 3, &[(0, &[-2]), (1, &[-2]), (2, &[-1]), (0, &[0, 2]), (1, &[]), (1, &[]), (2, &[0, 2]), (2, &[]), (2, &[2])]);
}

#[test]
fn fuzz_605_seed_336_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_605_336", 1, 3, &[(0, &[-3]), (1, &[-1]), (2, &[-1]), (0, &[-3]), (0, &[2, 2, 1]), (0, &[-2, 0, 0]), (1, &[2, -3, -1]), (1, &[-1]), (1, &[0, 2, 0]), (2, &[-2, 1, 1]), (2, &[])]);
}

#[test]
fn fuzz_605_seed_594_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_605_594", 0, 3, &[(0, &[-1]), (1, &[-3]), (2, &[-3]), (0, &[-1, -3, 1]), (0, &[1]), (1, &[2, 1]), (1, &[-1]), (1, &[1, 1, 2]), (2, &[2, 1, -2]), (2, &[0, 2]), (2, &[-1, -3])]);
}

#[test]
fn fuzz_605_seed_830_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_605_830", 2, 4, &[(0, &[-2]), (1, &[-3]), (2, &[-1]), (3, &[-1]), (0, &[0, 0, -2]), (0, &[-1]), (0, &[]), (1, &[2]), (2, &[]), (2, &[3, 1, 2]), (2, &[]), (3, &[-1, 3, 3]), (3, &[-2]), (3, &[-3, -3, 1])]);
}

/// Template determinization produced ~11M transitions uncharged, and the
/// ParserDwa build after it exhausted memory.
#[test]
fn fuzz_e86d_seed_4754_publishes_exact_masks_within_budget() {
    let _serial = serial();
    check_fuzz_case("fuzz_e86d_4754", 0, 3, &[(0, &[-2]), (1, &[-3]), (2, &[-1]), (3, &[-2]), (0, &[-2, 2]), (0, &[-2]), (0, &[]), (1, &[3, 1, 1]), (1, &[-2, 2, 3]), (1, &[]), (2, &[0]), (2, &[]), (2, &[3, -3]), (3, &[2, 0, 3]), (3, &[2])]);
}
