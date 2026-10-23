//! Differential harness for the mixed static/dynamic compile.
//!
//! A `TokenTrie` request over an assembly that has at least one retained
//! `ParserDwa` preparation is compiled mixed: retained parts stay static and
//! the remaining modules are compiled as dynamic component shells. If that
//! build fails it silently retries the all-static build. This ignored suite
//! compares every such route against an all-`ParserDwa` compile and records
//! whether the mixed route was taken or the fallback was used.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::__private::VocabExt as _;

const PARSER_DWA: MaskStrategy = MaskStrategy::ParserDwa;
const TOKEN_TRIE: MaskStrategy = MaskStrategy::TokenTrie;

fn vocab() -> Vocab {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"x".to_vec()),
        (3, b"y".to_vec()),
        (4, b"c".to_vec()),
        (5, b"p".to_vec()),
        (6, b"q".to_vec()),
        (7, b"ab".to_vec()),
        (8, b"xy".to_vec()),
        (9, b"xay".to_vec()),
        (10, b"xby".to_vec()),
        (11, b"\"".to_vec()),
        (12, b"\"a\"".to_vec()),
        (13, b"\"ab\"".to_vec()),
    ]);
    vocab.prepare_for_compile();
    vocab
}

#[derive(Clone, Copy)]
enum Kind {
    Glrm(&'static str),
    Ebnf(&'static str),
    Lark(&'static str),
    JsonSchema(&'static str),
}

impl Kind {
    fn module(self) -> Module {
        match self {
            Kind::Glrm(source) => Module::new(&Grammar::from_glrm(source)),
            Kind::Ebnf(source) => Module::new(&Grammar::from_ebnf(source)),
            Kind::Lark(source) => Module::new(&Grammar::from_lark(source)),
            Kind::JsonSchema(source) => Module::new(&Grammar::from_json_schema(source)),
        }
    }
}

/// One module node. Slot and token children must have a higher index than
/// their parent, so the assembly can be built from the leaves up.
struct NodeSpec {
    kind: Kind,
    slots: &'static [(&'static str, usize)],
    tokens: &'static [(&'static str, u32)],
}

struct AssemblySpec {
    name: &'static str,
    nodes: Vec<NodeSpec>,
    root: usize,
    /// `(label, node indices precompiled with ParserDwa)`.
    subsets: Vec<(&'static str, Vec<usize>)>,
}

const fn node(kind: Kind) -> NodeSpec {
    NodeSpec { kind, slots: &[], tokens: &[] }
}

const HOST: &str = r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#;
const MIDDLE: &str = r#"glrm 1; start root; extern grammar leaf; nt root = "p" leaf "q";"#;
const TWO_EXTERNS: &str = r#"glrm 1; start root; extern grammar LEFT; extern grammar RIGHT; nt root = "x" LEFT "c" RIGHT "y";"#;
const TOKEN_HOST: &str = r#"glrm 1; start root; extern token MARK; nt root = "x" MARK "y";"#;

fn assemblies() -> Vec<AssemblySpec> {
    vec![
        AssemblySpec {
            name: "glrm_parent_ebnf_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::Ebnf(r#"start ::= "a" "b""#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
        AssemblySpec {
            name: "depth2_glrm_glrm_ebnf",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                NodeSpec {
                    kind: Kind::Glrm(MIDDLE),
                    slots: &[("leaf", 2)],
                    tokens: &[],
                },
                node(Kind::Ebnf(r#"start ::= "a" | "b""#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1, 2]),
                ("parent", vec![0]),
                ("all", vec![0, 1, 2]),
            ],
        },
        AssemblySpec {
            name: "two_externs_reused_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(TWO_EXTERNS),
                    slots: &[("LEFT", 1), ("RIGHT", 1)],
                    tokens: &[],
                },
                node(Kind::Ebnf(r#"start ::= "a" | "b""#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
        AssemblySpec {
            name: "token_binding",
            nodes: vec![NodeSpec {
                kind: Kind::Glrm(TOKEN_HOST),
                slots: &[],
                tokens: &[("MARK", 0)],
            }],
            root: 0,
            subsets: vec![("none", vec![]), ("parent", vec![0])],
        },
        AssemblySpec {
            name: "json_schema_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::JsonSchema(r#"{"type":"string","maxLength":2}"#)),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
        AssemblySpec {
            name: "glrm_parent_lark_child",
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::Lark("start: \"a\" \"b\"")),
            ],
            root: 0,
            subsets: vec![
                ("none", vec![]),
                ("child", vec![1]),
                ("parent", vec![0]),
                ("all", vec![0, 1]),
            ],
        },
    ]
}

fn compile_assembly(spec: &AssemblySpec, precompiled: &[usize], vocab: &Vocab) -> Module {
    let mut built: Vec<Option<Module>> = (0..spec.nodes.len()).map(|_| None).collect();
    for index in (0..spec.nodes.len()).rev() {
        let node = &spec.nodes[index];
        let mut module = node.kind.module();
        if precompiled.contains(&index) {
            module
                .precompile_with(vocab, &CompileOptions::default().mask_strategy(PARSER_DWA))
                .unwrap_or_else(|error| {
                    panic!("{}: precompile node {index}: {error}", spec.name)
                });
        }
        for &(slot, child) in node.slots {
            let child = built[child]
                .clone()
                .unwrap_or_else(|| panic!("{}: child {child} is not built", spec.name));
            module
                .bind(slot, child)
                .unwrap_or_else(|error| panic!("{}: bind {slot:?}: {error}", spec.name));
        }
        for &(slot, id) in node.tokens {
            module
                .bind(slot, Token::new(id))
                .unwrap_or_else(|error| panic!("{}: bind token {slot:?}: {error}", spec.name));
        }
        built[index] = Some(module);
    }
    built[spec.root].take().unwrap()
}

fn compile_reference(spec: &AssemblySpec, strategy: MaskStrategy, vocab: &Vocab) -> RuntimeConstraint {
    compile_assembly(spec, &[], vocab)
        .compile_with(vocab, CompileOptions::default().mask_strategy(strategy))
        .unwrap_or_else(|error| panic!("{}: {strategy:?} reference compile: {error}", spec.name))
}

fn allowed_tokens(mask: &[u32]) -> Vec<u32> {
    let mut tokens = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            tokens.push((word_index * 32 + bit as usize) as u32);
            bits &= bits - 1;
        }
    }
    tokens
}

/// Exhaustive walk to `depth`, comparing the mask and acceptance of every
/// reachable state. Returns the number of visited states.
fn compare_exhaustive(
    reference: &crate::ConstraintState<'_>,
    candidate: &crate::ConstraintState<'_>,
    depth: usize,
    label: &str,
) -> usize {
    let reference_mask = reference.mask();
    let candidate_mask = candidate.mask();
    assert_eq!(
        reference_mask, candidate_mask,
        "{label}: mask mismatch at depth {depth}",
    );
    assert_eq!(
        reference.is_accepting(),
        candidate.is_accepting(),
        "{label}: acceptance mismatch at depth {depth}",
    );
    if depth == 0 {
        return 1;
    }
    let mut visited = 1;
    for token in allowed_tokens(&reference_mask) {
        let mut next_reference = reference.clone();
        next_reference
            .commit_token(token)
            .unwrap_or_else(|error| panic!("{label}: reference rejected admitted token {token}: {error}"));
        let mut next_candidate = candidate.clone();
        next_candidate
            .commit_token(token)
            .unwrap_or_else(|error| panic!("{label}: candidate rejected admitted token {token}: {error}"));
        visited += compare_exhaustive(&next_reference, &next_candidate, depth - 1, label);
    }
    visited
}

fn compare_against_parser_dwa_at(
    reference: &RuntimeConstraint,
    candidate: &RuntimeConstraint,
    label: &str,
    depth: usize,
) -> usize {
    compare_exhaustive(&reference.start(), &candidate.start(), depth, label)
}

fn compare_against_parser_dwa(
    reference: &RuntimeConstraint,
    candidate: &RuntimeConstraint,
    label: &str,
) -> usize {
    compare_against_parser_dwa_at(reference, candidate, label, 3)
}

#[test]
#[ignore = "differential mixed static/dynamic compile validation; run explicitly"]
fn mixed_compile_matches_parser_dwa_on_composed_assemblies() {
    let vocab = vocab();
    let mut fallbacks = Vec::<(String, String)>::new();
    let mut comparisons = 0usize;

    for spec in assemblies() {
        let reference = compile_reference(&spec, PARSER_DWA, &vocab);
        for (subset_label, precompiled) in &spec.subsets {
            // Clear anything left on this thread by an earlier compile.
            let _ = crate::public_api::module::mixed_compile_probe::take();
            let module = compile_assembly(&spec, precompiled, &vocab);
            let candidate = module
                .compile_with(&vocab, CompileOptions::default().mask_strategy(TOKEN_TRIE))
                .unwrap_or_else(|error| {
                    panic!(
                        "{} subset {subset_label}: TokenTrie compile failed: {error}",
                        spec.name,
                    )
                });
            let fallback_errors =
                crate::public_api::module::mixed_compile_probe::take();
            let route = if precompiled.is_empty() {
                "pure-token-trie"
            } else if fallback_errors.is_empty() {
                "mixed"
            } else {
                "fallback"
            };
            for error in &fallback_errors {
                fallbacks.push((format!("{} subset {subset_label}", spec.name), error.clone()));
            }
            let label = format!(
                "{} subset {subset_label} route {route}",
                spec.name,
            );
            let visited = compare_against_parser_dwa(&reference, &candidate, &label);
            comparisons += 1;
            eprintln!(
                "[mixed-validation] assembly={} subset={subset_label} precompiled={precompiled:?} \
                 route={route} states={visited} fallback={fallback_errors:?}",
                spec.name,
            );
        }
    }

    eprintln!(
        "[mixed-validation] summary: {} assemblies, {} compiles compared, {} fallbacks",
        assemblies().len(),
        comparisons,
        fallbacks.len(),
    );
    for (label, error) in &fallbacks {
        eprintln!("[mixed-validation] fallback {label}: {error}");
    }
    assert!(
        fallbacks.is_empty(),
        "mixed compile fell back for {} subset(s): {fallbacks:?}",
        fallbacks.len(),
    );
}

/// A retained GLRM parent plus a non-precompiled JSON Schema child must take
/// the mixed route: the child gets a dynamic component shell without retrying
/// the whole assembly as all-static.
#[test]
fn json_schema_child_mixed_compile_has_no_fallback() {
    let vocab = vocab();
    let spec = AssemblySpec {
        name: "json_schema_child_unit",
        nodes: vec![
            NodeSpec {
                kind: Kind::Glrm(HOST),
                slots: &[("child", 1)],
                tokens: &[],
            },
            node(Kind::JsonSchema(r#"{"type":"string","maxLength":2}"#)),
        ],
        root: 0,
        subsets: vec![("parent", vec![0])],
    };
    let reference = compile_reference(&spec, PARSER_DWA, &vocab);

    let _ = crate::public_api::module::mixed_compile_probe::take();
    let candidate = compile_assembly(&spec, &[0], &vocab)
        .compile_with(&vocab, CompileOptions::default().mask_strategy(TOKEN_TRIE))
        .unwrap_or_else(|error| panic!("{}: mixed TokenTrie compile failed: {error}", spec.name));
    let fallbacks = crate::public_api::module::mixed_compile_probe::take();
    assert!(
        fallbacks.is_empty(),
        "{}: mixed compile fell back: {fallbacks:?}",
        spec.name,
    );

    compare_against_parser_dwa(&reference, &candidate, &spec.name);
}

fn json_vocab() -> Vocab {
    let entries = [
        "{", "}", "[", "]", ",", ":", "\"", "\"a\"", "\"b\"", "\"ab\"", "\"abc\"",
        "\"key\"", "a", "b", "0", "1", "12", "-1", "true", "false", "null", " ", "x", "y",
    ]
    .iter()
    .enumerate()
    .map(|(id, bytes)| (id as u32, bytes.as_bytes().to_vec()))
    .collect::<Vec<_>>();
    let vocab = Vocab::new(entries);
    vocab.prepare_for_compile();
    vocab
}

/// The JSON Schema component shell must recognize the same token language as
/// the static schema compiler it replaces; a retained parent plus this child
/// must stay on the mixed route for every shape below.
#[test]
#[ignore = "differential JSON Schema component language validation; run explicitly"]
fn json_schema_component_matches_static_schema_compiler() {
    let vocab = json_vocab();
    let schemas: [(&str, &str); 8] = [
        ("string_max_length", r#"{"type":"string","maxLength":2}"#),
        ("string_pattern", r#"{"type":"string","pattern":"^[ab]+$"}"#),
        ("integer", r#"{"type":"integer"}"#),
        ("number", r#"{"type":"number"}"#),
        ("enum", r#"{"enum":["a","b",1]}"#),
        (
            "object_required",
            r#"{"type":"object","properties":{"key":{"type":"integer"}},"required":["key"]}"#,
        ),
        (
            "array_items",
            r#"{"type":"array","items":{"type":"boolean"},"maxItems":2}"#,
        ),
        (
            "nested_object",
            r#"{"type":"object","properties":{"a":{"type":"array","items":{"type":"string","maxLength":2}},"b":{"type":"integer"}},"required":["a","b"]}"#,
        ),
    ];

    for (name, schema) in schemas {
        let spec = AssemblySpec {
            name,
            nodes: vec![
                NodeSpec {
                    kind: Kind::Glrm(HOST),
                    slots: &[("child", 1)],
                    tokens: &[],
                },
                node(Kind::JsonSchema(schema)),
            ],
            root: 0,
            subsets: vec![("parent", vec![0])],
        };
        let reference = compile_reference(&spec, PARSER_DWA, &vocab);
        let _ = crate::public_api::module::mixed_compile_probe::take();
        let candidate = compile_assembly(&spec, &[0], &vocab)
            .compile_with(&vocab, CompileOptions::default().mask_strategy(TOKEN_TRIE))
            .unwrap_or_else(|error| panic!("{name}: mixed TokenTrie compile failed: {error}"));
        let fallbacks = crate::public_api::module::mixed_compile_probe::take();
        assert!(
            fallbacks.is_empty(),
            "{name}: mixed compile fell back: {fallbacks:?}",
        );
        let visited = compare_against_parser_dwa_at(&reference, &candidate, name, 5);
        eprintln!("[json-language] schema={name} states={visited} fallback=[]");
    }
}
