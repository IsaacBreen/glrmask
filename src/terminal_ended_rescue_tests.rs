//! Exact mask and typed-decline coverage for the ordinary template pipeline and epsilon-free prepass.
use crate::ErrorExt as _;
#[cfg(test)]
use crate::test_modes::MaskMode;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::build_budget::TestBudgetLimit;
use crate::compiler::grammar::transforms::set_test_epsfree_prepass;
use crate::{runtime::Constraint, Grammar, ParserBackend, Vocab};

// One single-byte token per terminal used by the witnesses. Keeping the
// vocabulary small keeps the exact oracle and the TokenTrie comparison focused
// on the grammar's mask language rather than tokenizer cross-products.
/// These cases all drive the shared Rayon pools. Running them
/// concurrently oversubscribes the machine and turns a ~20 s sequential suite
/// into minutes of contention, so the module serializes its own tests.
static TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    TEST_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const TOKEN_BYTES: [&[u8]; 4] = [b"a", b"b", b"c", b"d"];

fn vocab() -> Vocab {
    Vocab::from_entries_unchecked(vec![
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
    derive_relation_with_tokens(case, target, absorbing, &TOKEN_BYTES)
}

/// As [`derive_relation`], with negative CFG symbols mapping into `tokens`
/// (`-1` is the first token, `-2` the second, ...). This lets the same exact
/// oracle cover frontends whose vocabularies are not the four single-byte
/// witness tokens.
fn derive_relation_with_tokens(
    case: &Case,
    target: &[u8],
    absorbing: bool,
    tokens: &[&[u8]],
) -> Vec<Vec<Vec<bool>>> {
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
                    let token = tokens[(-symbol - 1) as usize];
                    let mut step = vec![vec![false; states]; states];
                    for from in 0..states {
                        let to = if from < n {
                            target[from..].starts_with(token).then_some(from + token.len())
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
    viable_with_tokens(case, prefix, &TOKEN_BYTES)
}

fn viable_with_tokens(case: &Case, prefix: &[u8], tokens: &[&[u8]]) -> bool {
    derive_relation_with_tokens(case, prefix, true, tokens)[case.start][0][prefix.len()]
}

fn accepts(case: &Case, word: &[u8]) -> bool {
    accepts_with_tokens(case, word, &TOKEN_BYTES)
}

fn accepts_with_tokens(case: &Case, word: &[u8], tokens: &[&[u8]]) -> bool {
    derive_relation_with_tokens(case, word, false, tokens)[case.start][0][word.len()]
}

fn allowed(mask: &[u32], token: usize) -> bool {
    mask.get(token / 32)
        .is_some_and(|word| word & (1 << (token % 32)) != 0)
}

fn fill_mask(constraint: &Constraint, prefix: &[u32]) -> (Vec<u32>, bool) {
    let mut state = constraint.start();
    for &token in prefix {
        assert!(state
            .commit_token(token));
    }
    let mut mask = vec![0u32; constraint.mask_bit_len().div_ceil(32)];
    state.fill_mask(&mut mask);
    (mask, state.is_accepting())
}

fn compile(case: &Case, strategy: MaskMode) -> Constraint {
    try_compile(case, strategy)
        .unwrap_or_else(|error| panic!("{strategy:?} compile failed: {error}"))
}

fn try_compile(case: &Case, strategy: MaskMode) -> crate::Result<Constraint> {
    let source = lark(case);
    Grammar::from_lark(&source)?.compile_with_build_options(
        &vocab(),
        strategy.build_options(),
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
    check_exact_oracle_with_tokens(constraint, case, &TOKEN_BYTES);
}

fn check_exact_oracle_with_tokens(constraint: &Constraint, case: &Case, tokens: &[&[u8]]) {
    let mut queue = VecDeque::from([(constraint.start(), Vec::<u8>::new())]);
    let mut seen = BTreeSet::<Vec<u8>>::new();
    let mut viability = BTreeMap::<Vec<u8>, bool>::new();
    let mut membership = BTreeMap::<Vec<u8>, bool>::new();
    let mut checked = 0usize;
    while let Some((state, prefix)) = queue.pop_front() {
        if !seen.insert(prefix.clone()) {
            continue;
        }
        let mut mask = vec![0u32; constraint.mask_bit_len().div_ceil(32)];
        state.fill_mask(&mut mask);
        for (token, token_bytes) in tokens.iter().enumerate() {
            let mut candidate = prefix.clone();
            candidate.extend_from_slice(token_bytes);
            let expected = *viability
                .entry(candidate.clone())
                .or_insert_with(|| viable_with_tokens(case, &candidate, tokens));
            assert_eq!(
                allowed(&mask, token),
                expected,
                "oracle mismatch at prefix {:?} token {token} (candidate {:?})",
                String::from_utf8_lossy(&prefix),
                String::from_utf8_lossy(&candidate),
            );
        }
        for id in tokens.len()..(mask.len() * 32) {
            assert!(
                !allowed(&mask, id),
                "unexpected token {id} at prefix {:?}",
                String::from_utf8_lossy(&prefix),
            );
        }
        let expected_accept = *membership
            .entry(prefix.clone())
            .or_insert_with(|| accepts_with_tokens(case, &prefix, tokens));
        assert_eq!(
            state.is_accepting(),
            expected_accept,
            "accept mismatch at {:?}",
            String::from_utf8_lossy(&prefix),
        );
        for (token, token_bytes) in tokens.iter().enumerate() {
            if prefix.len() + token_bytes.len() > case.bound || !allowed(&mask, token) {
                continue;
            }
            let mut next = prefix.clone();
            next.extend_from_slice(token_bytes);
            let mut committed = state.clone();
            assert!(committed.commit_token(token as u32));
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

fn seed_cfg(rules: &[(usize, &[isize])], start: usize, bound: usize) -> Case {
    case(start, bound, rules)
}

#[test]
fn ordinary_template_grammar_matches_exact_oracle() {
    let _serial = serial();
    let case = case(0, 3, &[(0, &[-1]), (0, &[-2, -1])]);
    let reference = compile(&case, MaskMode::A);
    for strategy in [MaskMode::C, MaskMode::ParserDwa] {
        let compiled = compile(&case, strategy);
        assert_eq!(
            compiled.parser_backend(),
            ParserBackend::TemplateDfa,
            "{strategy:?} did not use the legacy template path",
        );
        check_matches_token_trie(&compiled, &reference, &case);
        check_exact_oracle(&compiled, &case);
    }
}

/// Under a budget of one unit, explicit template strategies return a typed decline.
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
    for strategy in [MaskMode::C, MaskMode::ParserDwa] {
        {
            let _limit = TestBudgetLimit::set(1);
            expect_strategy_decline(
                try_compile(&case, strategy),
                &format!("tiny_budget {strategy:?}"),
            );
        }
    }
}

#[test]
fn difficult_grammar_returns_normal_typed_decline() {
    let _serial = serial();
    let case = case(0, 3, &[(0, &[-1]), (1, &[-3]), (2, &[-3]), (0, &[-1, -3, 1]), (0, &[1]), (1, &[2, 1]), (1, &[-1]), (1, &[1, 1, 2]), (2, &[2, 1, -2]), (2, &[0, 2]), (2, &[-1, -3])]);
    let _limit = TestBudgetLimit::set(10_000_000);
    expect_strategy_decline(try_compile(&case, MaskMode::ParserDwa), "normal decline");
}
/// Compile one already-declining witness with the prepass forced on. Returns
/// true when the normal route now compiles, false when it still returns its
/// typed decline.
fn check_prepass_case(case: &Case, label: &str, budget: u64) -> bool {
    let _limit = TestBudgetLimit::set(budget);
    set_test_epsfree_prepass(Some(true));
    let result = try_compile(case, MaskMode::ParserDwa);
    let compiled = match result {
        Ok(constraint) => {
            assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
            let reference = compile(case, MaskMode::A);
            check_matches_token_trie(&constraint, &reference, case);
            check_exact_oracle(&constraint, case);
            true
        }
        Err(error) => {
            assert!(
                error.is_strategy_decline(),
                "{label}: prepass returned a non-decline error: {error}",
            );
            eprintln!("[epsfree-prepass] {label} still declines: {error}");
            false
        }
    };
    set_test_epsfree_prepass(None);
    compiled
}

#[test]
fn epsfree_prepass_keeps_exact_masks_on_compiling_witnesses() {
    let _serial = serial();
    let witnesses = [
        (
            case(0, 3, &[(0, &[-1]), (0, &[-2, -1])]),
            "simple",
        ),
        (
            case(0, 4, &[(0, &[-1, -2]), (0, &[1]), (1, &[-3]), (1, &[])]),
            "nullable_unit",
        ),
        (
            case(0, 4, &[(0, &[-1, 0]), (0, &[-2]), (0, &[])]),
            "left_recursive_nullable",
        ),
    ];
    let mut compiled = 0usize;
    for (case, label) in witnesses {
        if check_prepass_case(&case, label, 10_000_000) {
            compiled += 1;
        }
    }
    assert!(
        compiled > 0,
        "the prepass did not produce an exactly checked compile",
    );
}

/// The grammars from the inflation study, now with the real prepass applied.
/// 4754, 594 and 2530 compile and match the exact oracle; fuzz 780 still
/// declines at a later stage.
#[test]
fn epsfree_prepass_named_witnesses_match_oracles_or_decline_later() {
    let _serial = serial();
    let fuzz_4754 = case(0, 3, &[(0, &[-2]), (1, &[-3]), (2, &[-1]), (3, &[-2]), (0, &[-2, 2]), (0, &[-2]), (0, &[]), (1, &[3, 1, 1]), (1, &[-2, 2, 3]), (1, &[]), (2, &[0]), (2, &[]), (2, &[3, -3]), (3, &[2, 0, 3]), (3, &[2])]);
    let fuzz_594 = case(0, 3, &[(0, &[-1]), (1, &[-3]), (2, &[-3]), (0, &[-1, -3, 1]), (0, &[1]), (1, &[2, 1]), (1, &[-1]), (1, &[1, 1, 2]), (2, &[2, 1, -2]), (2, &[0, 2]), (2, &[-1, -3])]);
    let seed_2530 = seed_cfg(
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
    assert!(
        check_prepass_case(&fuzz_4754, "fuzz_4754", 10_000_000),
        "fuzz_4754 did not compile with the prepass",
    );
    assert!(
        check_prepass_case(&fuzz_594, "fuzz_594", 1_000_000_000),
        "fuzz_594 did not compile with the prepass at the default budget",
    );
    assert!(
        check_prepass_case(&seed_2530, "seed_2530_min", 10_000_000),
        "seed_2530_min did not compile with the prepass",
    );
    assert!(
        !check_prepass_case(
            &case(1, 3, &[(0, &[-2]), (1, &[-3]), (0, &[]), (0, &[0, -2, 1]), (0, &[1, -3, 0]), (1, &[1, 1]), (1, &[0, -1]), (1, &[-3, -2, 0])]),
            "fuzz_780",
            10_000_000,
        ),
        "fuzz_780 unexpectedly compiled with the prepass",
    );
}
