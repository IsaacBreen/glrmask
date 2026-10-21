//! Public regression coverage for exact TokenTrie fallback after a template
//! strategy declines, plus guards that real grammars stay on their template
//! strategy at the default budget and that concurrent compiles keep separate
//! budgets.
//!
//! The oracle is an exact prefix-viability check: it intersects the CFG with
//! the prefix DFA (a CYK-style fixpoint over `(nonterminal, from, to)`), so a
//! prefix is viable exactly when some accepted word has that prefix. A bounded
//! word enumeration is not sufficient because the shortest completion may be
//! longer than the enumeration bound.

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

/// The design guarantee: a template decline must recompile to exactly the
/// masks and acceptance of the TokenTrie strategy.
fn check_fallback_matches_token_trie(fallback: &Constraint, reference: &Constraint, case: Case) {
    let bytes = token_bytes();
    let mut queue = VecDeque::from([(Vec::<u32>::new(), Vec::<u8>::new())]);
    let mut seen = BTreeSet::<Vec<u32>>::new();
    let mut checked = 0usize;
    while let Some((path, prefix)) = queue.pop_front() {
        if !seen.insert(path.clone()) {
            continue;
        }
        let (mask, accepting) = fill_mask(fallback, &path);
        let (reference_mask, reference_accepting) = fill_mask(reference, &path);
        assert_eq!(
            mask,
            reference_mask,
            "fallback mask differs from TokenTrie at prefix {:?}",
            String::from_utf8_lossy(&prefix),
        );
        assert_eq!(
            accepting,
            reference_accepting,
            "fallback acceptance differs from TokenTrie at {:?}",
            String::from_utf8_lossy(&prefix),
        );
        for (token, token_bytes) in bytes.iter().enumerate() {
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

fn compile_case(case: Case, strategy: MaskStrategy) -> Constraint {
    let source = lark(case);
    Grammar::from_lark(&source)
        .compile_with_build_options(
            &fuzz_vocab(),
            BuildOptions::default().optimization(strategy),
        )
        .unwrap_or_else(|error| panic!("{strategy:?} compile failed: {error}"))
}

#[test]
fn fuzz_seeds_fall_back_to_exact_token_trie_masks() {
    for case in [V2_233, V2_267, V2_296, V2_494, V2_1654] {
        for strategy in [
            MaskStrategy::TokenClassTrie,
            MaskStrategy::ParserDwa,
            MaskStrategy::Auto,
        ] {
            let fallback = {
                let _limit = TestBudgetLimit::set(1_000);
                compile_case(case, strategy)
            };
            assert_eq!(
                fallback.parser_backend(),
                ParserBackend::LrTable,
                "start={} {strategy:?} did not take the TokenTrie fallback",
                case.start,
            );
            let reference = compile_case(case, MaskStrategy::TokenTrie);
            check_fallback_matches_token_trie(&fallback, &reference, case);
            if strategy == MaskStrategy::TokenClassTrie {
                check_exact_oracle(&fallback, case);
            }
        }
    }
}

#[test]
fn forced_tiny_budget_still_yields_exact_masks() {
    const SMALL: Case = Case {
        start: 0,
        bound: 3,
        rules: &[(0, &[-1]), (0, &[-2, -3])],
    };
    let fallback = {
        let _limit = TestBudgetLimit::set(1);
        compile_case(SMALL, MaskStrategy::TokenClassTrie)
    };
    assert_eq!(fallback.parser_backend(), ParserBackend::LrTable);
    let reference = compile_case(SMALL, MaskStrategy::TokenTrie);
    check_fallback_matches_token_trie(&fallback, &reference, SMALL);
    check_exact_oracle(&fallback, SMALL);
}

#[test]
fn concurrent_compiles_keep_separate_budgets() {
    // A real grammar that succeeds on its template strategy at the default
    // budget but declines under a tiny one.
    const SCHEMA: &str = include_str!("../benches/data/o9838_problem_schema.json");
    for _ in 0..3 {
        let tiny = std::thread::spawn(|| {
            let _limit = TestBudgetLimit::set(1_000);
            Grammar::from_json_schema(SCHEMA)
                .compile_with_build_options(
                    &real_vocab(),
                    BuildOptions::default().optimization(MaskStrategy::ParserDwa),
                )
                .expect("tiny-budget compile")
                .parser_backend()
        });
        let default = std::thread::spawn(|| {
            Grammar::from_json_schema(SCHEMA)
                .compile_with_build_options(
                    &real_vocab(),
                    BuildOptions::default().optimization(MaskStrategy::ParserDwa),
                )
                .expect("default-budget compile")
                .parser_backend()
        });
        assert_eq!(tiny.join().unwrap(), ParserBackend::LrTable);
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
    // (643,126 units for o9838 with ParserDwa).
    assert!(DEFAULT_LIMIT >= 6_500_000);
    for (name, source) in REAL_GRAMMARS {
        for strategy in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
            let grammar = if name.ends_with(".lark") {
                Grammar::from_lark(source)
            } else {
                Grammar::from_json_schema(source)
            };
            let constraint = grammar
                .compile_with_build_options(
                    &real_vocab(),
                    BuildOptions::default().optimization(strategy),
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
