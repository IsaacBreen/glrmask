//! Public regression coverage for the no-silent-fallback contract: a template
//! strategy that declines under a tiny budget must return its typed decline
//! instead of recompiling with TokenTrie. Also guards that real grammars stay
//! on their template strategy at the default budget and that concurrent
//! compiles keep separate budgets.
//!
//! The oracle is an exact prefix-viability check: it intersects the CFG with
//! the prefix DFA (a CYK-style fixpoint over `(nonterminal, from, to)`), so a
//! prefix is viable exactly when some accepted word has that prefix. A bounded
//! word enumeration is not sufficient because the shortest completion may be
//! longer than the enumeration bound. It is applied to the exact TokenTrie
//! reference of every fuzz seed so the seeds themselves stay exact even though
//! the template strategies now decline them.
#[cfg(test)]
use crate::test_modes::MaskMode;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::build_budget::{TestBudgetLimit, DEFAULT_LIMIT};
use crate::{BuildOptions, Constraint, Grammar, MaskStrategy, ParserBackend, Vocab};

#[derive(Clone, Copy)]
struct Case {
    start: usize,
    bound: usize,
    rules: &'static [(usize, &'static [isize])],
}

fn terminal_byte(symbol: isize) -> u8 {
    b'a' + ((-symbol - 1) as u8)
}

fn lark(case: Case) -> String {
    let mut by_lhs: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for &(lhs, rhs) in case.rules {
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
        by_lhs.entry(lhs).or_default().push(text);
    }
    let mut source = format!("start: n{}\n", case.start);
    for (lhs, alternatives) in &by_lhs {
        source.push_str(&format!("n{lhs}: {}\n", alternatives.join(" | ")));
    }
    source
}

/// Exact CFG/DFA intersection fixpoint.
///
/// `target` is the input word for the DFA; `absorbing` selects prefix
/// viability (state `n` accepts any continuation) or exact membership (state
/// `n` has no outgoing edges). Returns `relation[nonterminal][from][to]`.
fn derive_relation(case: Case, target: &[u8], absorbing: bool) -> Vec<Vec<Vec<bool>>> {
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
    for &(lhs, rhs) in case.rules {
        if rhs.is_empty() {
            for state in 0..states {
                relation[lhs][state][state] = true;
            }
        }
    }
    loop {
        let mut changed = false;
        for &(lhs, rhs) in case.rules {
            // Compose the right-hand side left to right.
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
                    if composed[from][to] && !relation[lhs][from][to] {
                        relation[lhs][from][to] = true;
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

/// `L(G) ∩ prefix·Σ* ≠ ∅`.
fn viable(case: Case, prefix: &[u8]) -> bool {
    derive_relation(case, prefix, true)[case.start][0][prefix.len()]
}

/// Exact membership of `word` in `L(G)`.
fn accepts(case: Case, word: &[u8]) -> bool {
    derive_relation(case, word, false)[case.start][0][word.len()]
}

fn fuzz_vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"ab".to_vec()),
        (4, b"bc".to_vec()),
        (5, b"ca".to_vec()),
    ])
}

fn token_bytes() -> [&'static [u8]; 6] {
    [b"a", b"b", b"c", b"ab", b"bc", b"ca"]
}

fn allowed(mask: &[u32], token: usize) -> bool {
    mask.get(token / 32)
        .is_some_and(|word| word & (1 << (token % 32)) != 0)
}

/// Exact oracle: masks must be sound and complete against prefix viability,
/// and acceptance must be exact membership.
fn check_exact_oracle(constraint: &Constraint, case: Case) {
    let bytes = token_bytes();
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
        for (token, token_bytes) in bytes.iter().enumerate() {
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
        for id in 6..(mask.len() * 32) {
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
        for (token, token_bytes) in bytes.iter().enumerate() {
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

const V2_233: Case = Case {
    start: 0,
    bound: 4,
    rules: &[
        (0, &[-3]),
        (0, &[0, 3]),
        (0, &[]),
        (1, &[-1]),
        (1, &[]),
        (1, &[-3, -2, 2]),
        (1, &[2, 2, -1]),
        (2, &[-1]),
        (2, &[1, -1]),
        (2, &[2, 3, 2]),
        (2, &[-2, 2, 0]),
        (3, &[-1]),
        (3, &[-3, 0, 1]),
        (3, &[]),
    ],
};

const V2_267: Case = Case {
    start: 2,
    bound: 4,
    rules: &[
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
    ],
};

const V2_296: Case = Case {
    start: 1,
    bound: 4,
    rules: &[
        (1, &[-3]),
        (1, &[1, 0, 1]),
        (1, &[]),
        (0, &[-2]),
        (0, &[-2, -1]),
        (0, &[]),
        (0, &[1, -2, -3]),
    ],
};

const V2_494: Case = Case {
    start: 0,
    bound: 3,
    rules: &[
        (0, &[-3]),
        (0, &[2, 1, -2]),
        (0, &[1]),
        (1, &[-3]),
        (1, &[2]),
        (1, &[1, 2]),
        (2, &[-3]),
        (2, &[-1, 0]),
        (2, &[-2, 1]),
    ],
};

const V2_1654: Case = Case {
    start: 0,
    bound: 3,
    rules: &[
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
    ],
};

fn try_compile_case(case: Case, strategy: MaskMode) -> crate::Result<Constraint> {
    let source = lark(case);
    Grammar::from_lark(&source)?.compile_with_build_options(
        &fuzz_vocab(),
        strategy.build_options(),
    )
}

fn compile_case(case: Case, strategy: MaskMode) -> Constraint {
    try_compile_case(case, strategy)
        .unwrap_or_else(|error| panic!("{strategy:?} compile failed: {error}"))
}

/// The requested template strategy must surface its own typed decline; a
/// successful compile here would mean it silently recompiled with TokenTrie.
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

#[test]
fn fuzz_seeds_return_typed_decline_under_tiny_budget() {
    for case in [V2_233, V2_267, V2_296, V2_494, V2_1654] {
        // The exact TokenTrie reference stays the mask oracle for each seed, so
        // the grammars themselves keep exact coverage even though every
        // template strategy now declines them under the tiny budget.
        let reference = compile_case(case, MaskMode::A);
        check_exact_oracle(&reference, case);
        // Auto alone may fall back to the exact TokenTrie strategy; its masks
        // must still match the oracle.
        {
            let _limit = TestBudgetLimit::set(1_000);
            let auto = try_compile_case(case, MaskMode::Auto)
                .unwrap_or_else(|error| panic!("start={} Auto must fall back: {error}", case.start));
            check_exact_oracle(&auto, case);
        }
        for strategy in [MaskMode::C, MaskMode::ParserDwa] {
            let _ = {
                let _limit = TestBudgetLimit::set(1_000);
                expect_strategy_decline(
                    try_compile_case(case, strategy),
                    &format!("start={} {strategy:?}", case.start),
                )
            };
        }
    }
}

#[test]
fn forced_tiny_budget_returns_typed_decline() {
    const SMALL: Case = Case {
        start: 0,
        bound: 3,
        rules: &[(0, &[-1]), (0, &[-2, -3])],
    };
    let reference = compile_case(SMALL, MaskMode::A);
    check_exact_oracle(&reference, SMALL);
    for strategy in [MaskMode::C, MaskMode::ParserDwa] {
        let _ = {
            let _limit = TestBudgetLimit::set(1);
            expect_strategy_decline(
                try_compile_case(SMALL, strategy),
                &format!("forced_tiny_budget {strategy:?}"),
            )
        };
    }
}

#[test]
fn concurrent_compiles_keep_separate_budgets() {
    // A real grammar that succeeds on its template strategy at the default
    // budget but declines under a tiny one. The tiny compile must surface its
    // own typed decline without disturbing the concurrent default compile.
    const SCHEMA: &str = include_str!("../benches/data/o9838_problem_schema.json");
    for _ in 0..3 {
        let tiny = std::thread::spawn(|| {
            let _limit = TestBudgetLimit::set(1_000);
            let result = Grammar::from_json_schema(SCHEMA).unwrap().compile_with_build_options(
                &real_vocab(),
                BuildOptions::default().optimization(MaskStrategy::ParserDwa),
            );
            let _ = expect_strategy_decline(result, "concurrent tiny-budget compile");
        });
        let default = std::thread::spawn(|| {
            Grammar::from_json_schema(SCHEMA).unwrap()
                .compile_with_build_options(
                    &real_vocab(),
                    BuildOptions::default().optimization(MaskStrategy::ParserDwa),
                )
                .expect("default-budget compile")
                .parser_backend()
        });
        tiny.join().unwrap();
        assert_eq!(default.join().unwrap(), ParserBackend::TemplateDfa);
    }
}

const REAL_GRAMMARS: &[(&str, &str)] = &[
    (
        "reduced_sql.lark",
        include_str!("../tests/data/parser_family_fallback/reduced_sql.lark"),
    ),
    (
        "github_hard_o21074.schema.json",
        include_str!("../benches/data/github_hard_o21074.schema.json"),
    ),
    (
        "github_ultra_o62058.schema.json",
        include_str!("../benches/data/github_ultra_o62058.schema.json"),
    ),
    (
        "kubernetes_kb_543_normalized.schema.json",
        include_str!("../benches/data/kubernetes_kb_543_normalized.schema.json"),
    ),
    (
        "o9838_problem_schema.json",
        include_str!("../benches/data/o9838_problem_schema.json"),
    ),
];

fn real_vocab() -> Vocab {
    Vocab::new((0..=255u32).map(|id| (id, vec![id as u8])).collect())
}

#[test]
fn real_grammars_compile_without_fallback_at_the_default_budget() {
    // The default must stay at least 10x the observed real-grammar maximum
    // (671,896 units for o9838 with ParserDwa) measured with the calibrated
    // inner-loop charges.
    const REAL_GRAMMAR_MAX_UNITS: u64 = 671_896;
    assert!(
        DEFAULT_LIMIT >= 10 * REAL_GRAMMAR_MAX_UNITS,
        "default budget {DEFAULT_LIMIT} is below the 10x real-grammar floor {}",
        10 * REAL_GRAMMAR_MAX_UNITS,
    );
    for (name, source) in REAL_GRAMMARS {
        for strategy in [MaskMode::C, MaskMode::ParserDwa] {
            let grammar = if name.ends_with(".lark") {
                Grammar::from_lark(source).unwrap()
            } else {
                Grammar::from_json_schema(source).unwrap()
            };
            let constraint = grammar
                .compile_with_build_options(
                    &real_vocab(),
                    strategy.build_options(),
                )
                .unwrap_or_else(|error| panic!("{name} {strategy:?} failed: {error}"));
            assert_eq!(
                constraint.parser_backend(),
                ParserBackend::TemplateDfa,
                "{name} {strategy:?} fell back at the default budget",
            );
        }
    }
}

#[test]
fn auto_decline_fallback_honours_template_dfas() {
    const SMALL: Case = Case {
        start: 0,
        bound: 3,
        rules: &[(0, &[-1]), (0, &[-2, -3])],
    };
    let reference = compile_case(SMALL, MaskMode::ParserDwa);
    // Auto only reaches TokenTrie through the decline fallback. That fallback
    // must compile exactly as an explicit TokenTrie request with the same two
    // switches would, so with `template_dfas` it is table-free.
    let auto = {
        let _limit = TestBudgetLimit::set(1);
        Grammar::from_lark(&lark(SMALL))
            .unwrap()
            .compile_with_build_options(
                &fuzz_vocab(),
                MaskMode::Auto.build_options().template_dfas(true),
            )
            .expect("Auto must fall back to TokenTrie")
    };
    assert_eq!(
        auto.parser_backend(),
        ParserBackend::TemplateDfa,
        "Auto's fallback must honour template_dfas",
    );
    assert!(!auto.table.is_present(), "the fallback must not retain an LR table");
    check_exact_oracle(&auto, SMALL);
    for prefix in [b"".as_slice(), b"a", b"b", b"c", b"ab", b"ba", b"abc"] {
        let mut fallback = auto.start();
        let mut dwa = reference.start();
        assert_eq!(
            fallback.commit_bytes(prefix),
            dwa.commit_bytes(prefix),
            "commit {prefix:?}",
        );
        assert_eq!(fallback.mask(), dwa.mask(), "mask after {prefix:?}");
        assert_eq!(
            fallback.is_accepting(),
            dwa.is_accepting(),
            "acceptance after {prefix:?}",
        );
    }
}
